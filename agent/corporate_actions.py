"""Gross A-share entitlements, replayed separately from actual paper trades."""
from datetime import date
from decimal import Decimal, ROUND_HALF_UP
import json
import math

from evals.experiments import canonical_hash

VERSION = 'cn-gross-entitlements-v1'
MONEY = Decimal('.01')


def number(value):
    if isinstance(value, bool):
        raise ValueError('权益数值不能为布尔值')
    value = Decimal(str(value))
    if not value.is_finite():
        raise ValueError('权益数值必须有限')
    return value


def cents(value):
    return number(value).quantize(MONEY, rounding=ROUND_HALF_UP)


def validate_action(row):
    required = {'event_id','symbol','record_date','ex_date','pay_date','listing_date',
                'cash_per_share','bonus_per_share','kind','source','known_at'}
    if not required <= row.keys() or row['kind'] not in {'distribution','forward_split'}:
        raise ValueError('权益事件契约不完整或类型不支持')
    if not row['record_date'] or not row['ex_date']:
        raise ValueError('权益事件缺少登记日或除权日')
    import re
    if not re.fullmatch(r'\d{6}',row['symbol']) or not re.fullmatch(r'[A-Za-z0-9:_-]{1,120}',row['event_id']):
        raise ValueError('权益事件身份无效')
    for key in ('record_date','ex_date','pay_date','listing_date'):
        if row[key] is not None and date.fromisoformat(row[key]).isoformat() != row[key]:
            raise ValueError('权益日期无效')
    cash, bonus = number(row['cash_per_share']), number(row['bonus_per_share'])
    if not (0 <= cash <= 1000 and 0 <= bonus <= 100) or cash+bonus <= 0:
        raise ValueError('权益比例无效或超出支持范围')
    if row['ex_date'] <= row['record_date']:
        raise ValueError('除权日必须晚于股权登记日')
    if cash and (not row['pay_date'] or row['pay_date'] < row['ex_date']):
        raise ValueError('现金分红缺少明确到账日')
    if bonus and (not row['listing_date'] or row['listing_date'] < row['ex_date']):
        raise ValueError('送转/拆股缺少明确上市日')
    if row['kind']=='forward_split' and cash:
        raise ValueError('拆股事件不能夹带现金分红')
    from servers.data_quality import event_time
    event_time(row['known_at'],'cn')
    return row


def replay(ledger, initial, through=None):
    """Reconstruct cash, tradable stock and receivables at a dated close.

    Record-date trades settle before entitlement capture. Ex-date recognition and
    payment/listing happen before that day's trades. No price adjustment factor
    is used as a dividend, share allocation or payment date.
    """
    from agent.trading_clock import now
    through = through or ledger.get('as_of_date') or now().date().isoformat()
    actions = [validate_action(r) for r in ledger.get('corporate_actions',[])]
    if len({r['event_id'] for r in actions}) != len(actions):
        raise ValueError('权益事件身份重复')
    if len({(r['symbol'],r['ex_date']) for r in actions}) != len(actions):
        raise ValueError('同股同除权日存在冲突事件，暂停核算')
    timeline = []
    for i,r in enumerate(ledger['trades']):
        timeline.append((r['trade_date'],2,i,'trade',r))
    for i,r in enumerate(actions):
        for key,phase,kind in [('record_date',3,'record'),('ex_date',0,'recognize'),
                               ('pay_date',1,'pay'),('listing_date',1,'list')]:
            if r[key]:timeline.append((r[key],phase,i,kind,r))
    cash, realized, income = number(initial), Decimal(0), Decimal(0)
    positions, entitlements, rights, recognized = {}, {}, {}, []
    def position(symbol, name=None):
        p = positions.setdefault(symbol,{'symbol':symbol,'name':name or symbol,'shares':0,
                                         'basis':Decimal(0),'last_buy_date':None})
        if name:p['name']=name
        return p
    for day,_,_,kind,r in sorted(timeline,key=lambda t:t[:4]):
        if day > through:continue
        symbol = r['symbol']
        p = position(symbol,r.get('name'))
        if kind=='trade':
            qty, price, fee = r['shares'], number(r['price']), number(r.get('fee') or 0)
            if isinstance(qty,bool) or not isinstance(qty,int) or qty<=0 or price<=0 or fee<0:
                raise ValueError('无效成交记录')
            if r['action']=='BUY':
                p['basis'] += qty*price+fee;p['shares'] += qty;p['last_buy_date'] = day
                cash = cents(cash-qty*price-fee)
            elif r['action']=='SELL' and p['shares']>=qty:
                cost = p['basis']/p['shares']*qty
                proceeds=cents(qty*price)
                realized += proceeds-fee-cost
                p['basis'] -= cost;p['shares'] -= qty
                cash = cents(cash+proceeds-fee)
            else:raise ValueError('成交账本方向或持股数量不一致')
            if cash < 0:raise ValueError('成交账本出现负现金')
        elif kind=='record':
            if any(v['symbol']==symbol and v['bonus_shares'] for v in rights.values()):
                raise ValueError('未上市权益跨越下一次登记日，暂不支持自动推断登记数量')
            entitlements[r['event_id']] = p['shares']
        elif kind=='recognize':
            qty = entitlements.get(r['event_id'],0)
            if not qty:continue
            bonus = qty*number(r['bonus_per_share'])
            if bonus != bonus.to_integral_value():
                raise ValueError('送转涉及零碎股，不能猜测中国结算的尾数分配结果')
            allocated = p['basis']*bonus/(p['shares']+bonus) if bonus else Decimal(0)
            p['basis'] -= allocated
            dividend = cents(qty*number(r['cash_per_share']))
            income += dividend
            recognized.append({'event_id':r['event_id'],'record_shares':qty,'cash':float(dividend),
                               'bonus_shares':int(bonus),'ex_date':r['ex_date'],
                               'source_hash':r['source'].get('pdf_sha256')})
            rights[r['event_id']] = {'event_id':r['event_id'],'symbol':symbol,'name':p['name'],
                'record_shares':qty,'cash_receivable':dividend,'bonus_shares':int(bonus),
                'basis':allocated,'last_buy_date':p['last_buy_date'],
                'ex_date':r['ex_date'],'pay_date':r['pay_date'],'listing_date':r['listing_date']}
        elif r['event_id'] in rights:
            right = rights[r['event_id']]
            if kind=='pay':cash += right['cash_receivable'];right['cash_receivable']=Decimal(0)
            elif kind=='list':
                p['basis'] += right['basis'];p['shares'] += right['bonus_shares']
                p['last_buy_date'] = max(filter(None,[p['last_buy_date'],right['last_buy_date']]),default=None)
                right['basis']=Decimal(0);right['bonus_shares']=0
    output_positions = [{**{k:v for k,v in p.items() if k!='basis'},'avg_cost':float(p['basis']/p['shares'])}
                        for p in positions.values() if p['shares']]
    output_rights = [{**r,'cash_receivable':float(r['cash_receivable']),'basis':float(r['basis'])}
                    for r in rights.values() if r['cash_receivable'] or r['bonus_shares']]
    return {'cash':float(cents(cash)),'positions':output_positions,'rights':output_rights,
            'realized_pnl':float(realized),'dividend_income':float(income), 'as_of_date':through,
            'recognized_events':recognized,
            'cash_receivable':float(sum((number(r['cash_receivable']) for r in output_rights),Decimal(0)))}


def schema(conn):
    conn.executescript('''
        CREATE TABLE IF NOT EXISTS corporate_actions(event_id TEXT PRIMARY KEY,symbol TEXT NOT NULL,payload TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS corporate_settlements(id INTEGER PRIMARY KEY,recorded_at TEXT NOT NULL,
            through_date TEXT NOT NULL,state_hash TEXT NOT NULL,coverage TEXT NOT NULL);
    ''')


def read_actions(conn):
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    return [json.loads(r[0]) for r in conn.execute('SELECT payload FROM corporate_actions ORDER BY event_id')] if 'corporate_actions' in tables else []


def settle(conn, actions, initial, through, recorded_at, coverage):
    """Atomic, idempotent journal migration; a mismatch never overwrites the book."""
    schema(conn)
    conn.execute('BEGIN IMMEDIATE')
    try:
        trades = [dict(r) for r in conn.execute('SELECT * FROM trades ORDER BY trade_date,id')]
        old_actions = read_actions(conn)
        prior = conn.execute('SELECT through_date FROM corporate_settlements ORDER BY id DESC LIMIT 1').fetchone()
        prior_day = max([r['trade_date'] for r in trades]+([prior[0]] if prior else []) or [through])
        if through < prior_day:raise ValueError('权益结算不倒退到已记账日期之前')
        before = replay({'trades':trades,'corporate_actions':old_actions},initial,prior_day)
        actual_cash = conn.execute("SELECT value FROM meta WHERE key='cash'").fetchone()[0]
        actual = {r['symbol']:dict(r) for r in conn.execute('SELECT * FROM positions WHERE shares>0')}
        expected = {r['symbol']:r for r in before['positions']}
        if abs(actual_cash-before['cash'])>.02 or set(actual)!=set(expected) or any(
            actual[s]['shares']!=p['shares'] or abs(actual[s]['avg_cost']-p['avg_cost'])>.000002 or
            actual[s]['last_buy_date']!=p['last_buy_date'] for s,p in expected.items()):
            raise ValueError('原账户不能与成交及权益记录对账，停止自动修订')
        by_id = {r['event_id']:r for r in old_actions}
        for r in actions:
            validate_action(r)
            from servers.data_quality import event_time
            if event_time(r['known_at'],'cn')>event_time(recorded_at,'cn'):
                raise ValueError('权益事件不能在知晓之前登记')
            if r['event_id'] in by_id and canonical_hash(by_id[r['event_id']])!=canonical_hash(r):
                raise ValueError('已登记的实施公告发生修订，冻结权益处理')
            by_id[r['event_id']] = r
        after = replay({'trades':trades,'corporate_actions':list(by_id.values())},initial,through)
        for r in actions:
            conn.execute('INSERT OR IGNORE INTO corporate_actions VALUES(?,?,?)',
                         (r['event_id'],r['symbol'],json.dumps(r,ensure_ascii=False,allow_nan=False)))
        conn.execute("UPDATE meta SET value=? WHERE key='cash'",(after['cash'],))
        conn.execute('DELETE FROM positions')
        for p in after['positions']:
            conn.execute('INSERT INTO positions(symbol,name,shares,avg_cost,last_buy_date) VALUES(?,?,?,?,?)',
                         tuple(p[k] for k in ('symbol','name','shares','avg_cost','last_buy_date')))
        state_hash = canonical_hash(after)
        latest = conn.execute('SELECT state_hash,coverage FROM corporate_settlements ORDER BY id DESC LIMIT 1').fetchone()
        encoded = json.dumps(coverage,ensure_ascii=False,sort_keys=True)
        if not latest or (latest[0],latest[1]) != (state_hash,encoded):
            conn.execute('INSERT INTO corporate_settlements(recorded_at,through_date,state_hash,coverage) VALUES(?,?,?,?)',
                         (recorded_at,through,state_hash,encoded))
        conn.commit()
        return after
    except Exception:
        conn.rollback()
        raise


def receivable_value(state, prices):
    value = state.get('cash_receivable',0.)
    for right in state.get('rights',[]):
        if right['bonus_shares']:
            price = prices.get(right['symbol'])
            if price is None or not math.isfinite(price) or price<=0:
                raise ValueError('未上市送转权益缺少对应真实估值价')
            value += right['bonus_shares']*price
    return round(value,2)
