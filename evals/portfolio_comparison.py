"""Prospectively registered benchmark and immutable observed paper valuations."""
from copy import deepcopy
from datetime import datetime
import json
import math
from pathlib import Path

from config import DATA_DIR, MARKETS
from evals.experiments import canonical_hash
from servers.data_quality import event_time, market_close

HOME = DATA_DIR / 'portfolio_comparison'
SPEC = {'version': 'prospective-portfolio-comparison-v1', 'index_code': 'H00300',
        'index_name': '沪深300全收益指数', 'index_basis': 'gross_cash_dividends_reinvested',
        'cash_basis': 'principal_without_interest', 'benchmark_fees': 'not_included',
        'source_url': 'https://www.csindex.com.cn/zh-CN/indices/index-detail/H00300'}


def _seal(value):
    return {**value, 'fingerprint': canonical_hash(value)}


def _read(path):
    value = json.loads(Path(path).read_text(encoding='utf-8'))
    if value.get('fingerprint') != canonical_hash({k:v for k,v in value.items() if k!='fingerprint'}):
        raise ValueError('基准记录指纹不符，停止比较')
    return value


def _write(path, value):
    from agent.analysis import write_log
    write_log(Path(path), _seal(value))


class ComparisonBook:
    def __init__(self, root=HOME):
        self.root = Path(root)

    def register(self, sessions, registered_at):
        path = self.root / 'policy.json'
        if path.exists():
            policy = _read(path)
            if policy['spec'] != SPEC:
                raise ValueError('基准规则已变化，不能覆盖原登记或重置成绩')
            return policy
        stamp = event_time(registered_at, 'cn')
        future = [d for d in sessions if market_close(d, 'cn') > stamp]
        if not future:
            raise ValueError('真实交易日历未覆盖登记后的收盘')
        body = {'spec': deepcopy(SPEC), 'registered_at': stamp.isoformat(), 'start_session': min(future),
                'calendar_fingerprint': canonical_hash(sessions)}
        _write(path, body)
        return _read(path)

    def context(self, cutoff):
        """Read only; historical reports cannot see later registrations/observations."""
        path = self.root / 'policy.json'
        if not path.exists():
            return None
        policy = _read(path)
        end = event_time(cutoff, 'cn')
        if event_time(policy['registered_at'], 'cn') > end:
            return None
        if policy['spec'] != SPEC:
            raise ValueError('已登记的基准规则与当前代码不符')
        observations = _read(self.root / 'observations.json') if (self.root/'observations.json').exists() else None
        if observations and observations['policy_id'] != policy['fingerprint']:
            raise ValueError('基准估值与登记身份不一致')
        points = [r for r in (observations or {}).get('points', [])
                  if market_close(r['date'], 'cn') <= end and event_time(r['recorded_at'], 'cn') <= end]
        history = _read(self.root/'status.json') if (self.root/'status.json').exists() else {}
        statuses = [r for r in history.get('entries', []) if event_time(r['recorded_at'], 'cn') <= end]
        status = max(enumerate(statuses), key=lambda pair:(event_time(pair[1]['recorded_at'],'cn'),pair[0]))[1] if statuses else {}
        return {'policy': policy, 'points': points, 'refresh': status}

    def status(self, recorded_at, state, source_date=None):
        path = self.root/'status.json'
        history = _read(path).get('entries', []) if path.exists() else []
        entry = {'recorded_at': recorded_at, 'state': state, 'source_date': source_date}
        if not history or history[-1] != entry:
            _write(path, {'entries':history+[entry]})

    def observe(self, policy, index_rows, curve, ledger, sessions, recorded_at, *, initial=None):
        """Append matched closes; never replace a past observation with revised data."""
        end = event_time(recorded_at, 'cn')
        path = self.root/'observations.json'
        stored = _read(path) if path.exists() else {'policy_id': policy['fingerprint'], 'points': []}
        if stored['policy_id'] != policy['fingerprint']:
            raise ValueError('估值记录的基准身份不一致')
        points = {p['date']:p for p in stored['points']}
        prices = {r['date']:r['close'] for r in index_rows}
        eligible = [p for p in curve if p['date'] >= policy['start_session'] and market_close(p['date'],'cn') <= end]
        if not points and not any(p['date']==policy['start_session'] for p in eligible):
            return 'waiting_for_start' if market_close(policy['start_session'],'cn') > end else 'waiting_for_portfolio_prices'
        source_end = max(prices) if prices else None
        if not source_end:
            raise ValueError('官方基准没有已收盘数据')
        for p in eligible:
            day = p['date']
            if day not in prices:
                raise ValueError('基准缺少对应交易日，不跳过后计算超额收益')
            day_trades = [r for r in ledger['trades'] if r['trade_date'] <= day]
            from agent.corporate_actions import replay
            daily_ledger={**ledger,'trades':day_trades,'as_of_date':day}
            state=replay(daily_ledger,initial or MARKETS['cn']['initial_cash'],day)
            cash, held = state['cash'],bool(state['positions'] or state['rights'])
            if not math.isfinite(float(p['total'])) or p['total'] <= 0 or not 0 <= cash <= p['total'] + .02:
                raise ValueError('账户估值或现金比例无效')
            journal=[{k:r.get(k) for k in ('symbol','action','shares','price','fee','trade_date')} for r in day_trades]
            if state['recognized_events']:journal += state['recognized_events']
            body = {'date':day, 'portfolio_total':float(p['total']), 'index_close':prices[day],
                    'cash_ratio_pct':round(cash/float(p['total'])*100, 6), 'all_cash':not held,
                    'journal_hash':canonical_hash(journal),
                    'equity_accounting_complete':bool(p.get('equity_accounting_complete')) if held else False}
            if day in points:
                old = {k:points[day].get(k,False) for k in body}
                if old != body:
                    raise ValueError('历史价格或账本估值发生修订，已冻结对照，需核对原记录')
            else:
                points[day] = {**body, 'recorded_at':recorded_at}
        ordered = [points[d] for d in sorted(points)]
        if ordered and ordered[0]['date'] != policy['start_session']:
            raise ValueError('基准起点缺失，不滚动到有利日期重新开始')
        expected = [d for d in sessions if policy['start_session'] <= d <= ordered[-1]['date']] if ordered else []
        if set(expected) != set(points):
            raise ValueError('基准区间存在估值缺口，停止完整绩效比较')
        _write(path, {'policy_id':policy['fingerprint'], 'points':ordered})
        return 'ready'


def normalize_index(rows, cutoff):
    end = event_time(cutoff, 'cn')
    values = {}
    for row in rows:
        if row.get('indexCode') != SPEC['index_code'] or '全收益' not in str(row.get('indexNameCnAll','')):
            raise ValueError('指数身份不是已登记的沪深300全收益')
        day = datetime.strptime(str(row['tradeDate']), '%Y%m%d').date().isoformat()
        if market_close(day, 'cn') > end:
            continue
        close = float(row['close'])
        if not math.isfinite(close) or close <= 0:
            raise ValueError('基准收盘值无效')
        if day in values and values[day] != close:
            raise ValueError('同日基准收盘值冲突')
        values[day] = close
    if not values:
        raise ValueError('基准缺少已收盘数据')
    return [{'date':d, 'close':values[d]} for d in sorted(values)]


def build_comparison(context, price_date, journal_ok=True):
    if not context:
        return {'status':'not_registered', 'message':'该报告时点尚未登记组合基准。', 'series':[]}
    policy, points, refresh = context['policy'], context['points'], context['refresh']
    result = {'status':'waiting_for_start', 'policy_id':policy['fingerprint'], 'registered_at':policy['registered_at'],
              'start_session':policy['start_session'], 'index_code':SPEC['index_code'], 'index_name':SPEC['index_name'],
              'source_url':SPEC['source_url'], 'source_as_of_date':refresh.get('source_date'),
              'source_checked_at':refresh.get('recorded_at'), 'observations':len(points), 'series':[],
              'methodology':'沪深300全收益含税前现金分红再投资，不计交易或管理费用；现金本金不计利息。账户按实际模拟账本估值。',
              'message':'正式比较从登记后的首个收盘开始；此前行情不回填为事前基准。'}
    points = [p for p in points if p['date'] <= price_date]
    result['observations'] = len(points)
    if not journal_ok:
        result.update(status='account_unreconciled', message='账本尚未完整对账，组合对照暂停。')
        return result
    if not points:
        if refresh.get('state') == 'waiting_for_portfolio_prices':
            result.update(status='waiting_for_portfolio_prices', message='登记起点或后续组合估值缺失，暂不计算相对表现；不移动登记起点。')
        elif refresh.get('state') not in {None,'ready','waiting_for_start'}:
            result.update(status='source_unavailable', message='基准来源暂不可用；保留原登记和真实日期。')
        return result
    first = points[0]
    all_cash = all(p['all_cash'] for p in points)
    complete=all(p['all_cash'] or p.get('equity_accounting_complete',False) for p in points)
    for p in points:
        result['series'].append({'date':p['date'], 'portfolio_return_pct':round((p['portfolio_total']/first['portfolio_total']-1)*100,6),
            'benchmark_return_pct':round((p['index_close']/first['index_close']-1)*100,6), 'cash_return_pct':0})
    last = result['series'][-1]
    result.update(observations=len(points), comparison_start=first['date'], comparison_end=points[-1]['date'],
                  reference_assets=first['portfolio_total'], portfolio_return_pct=last['portfolio_return_pct'],
                  benchmark_return_pct=last['benchmark_return_pct'], cash_ratio_pct=round(sum(p['cash_ratio_pct'] for p in points)/len(points),4),
                  all_cash=all_cash, status='ready' if complete else 'limited_accounting',
                  message='同一起点的实际模拟账本、市场全收益与现金本金对照；账户股息税前，已扣成交费用，指数不计费用。' if complete else
                  '区间存在股票持仓；账本尚未计入完整分红/拆股权益，曲线供参考，暂不作严格全收益绩效评分。')
    if points[-1]['date'] != price_date or refresh.get('state') != 'ready':
        result.update(status='stale', message='对照尚未覆盖当前行情日，暂不计算当前相对表现。')
    if result['status']=='ready':
        gap = last['portfolio_return_pct']-last['benchmark_return_pct']
        result.update(return_gap_pp=round(gap,6), relative_pnl=round(first['portfolio_total']*gap/100,2))
        if all_cash:result['cash_opportunity_amount']=round(first['portfolio_total']*last['benchmark_return_pct']/100,2)
    return result
