"""Host-only synchronization. No LLM tool can create corporate actions."""
from contextlib import closing
import json
from pathlib import Path
import sqlite3
from time import monotonic

from agent.corporate_actions import schema, settle
from agent.corporate_sources import CorporateBook
from agent.trading_clock import calendar, now
from config import MARKETS
from servers.data_quality import market_close


def synchronize(extra_symbols=(), *, at=None, db=None, book=None, budget_seconds=60):
    deadline=monotonic()+budget_seconds
    at=at or now();path=Path(db or MARKETS['cn']['db']);book=book or CorporateBook()
    sessions=calendar()['sessions']
    through=max(d for d in sessions if market_close(d,'cn')<=at)
    from evals.trading_view import read_ledger
    ledger=read_ledger(path)
    starts={s:min(r['trade_date'] for r in ledger['trades'] if r['symbol']==s)
            for s in {r['symbol'] for r in ledger['trades']}}
    for symbol in extra_symbols:starts.setdefault(symbol,through)
    # Empty accounts need no invented "no dividend" source responses.
    if not path.exists():
        if starts:raise ValueError('模拟账户尚未初始化')
        return {'status':'no_account','sources':0,'actions':0,'model_requests':0}
    with closing(sqlite3.connect(path)) as conn:
        conn.row_factory=sqlite3.Row
        tables={r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if 'corporate_settlements' not in tables:
            backup=path.parent/'backups'/(path.stem+'-before-corporate-v1.db')
            backup.parent.mkdir(parents=True,exist_ok=True)
            if not backup.exists():
                with closing(sqlite3.connect(backup)) as target:conn.backup(target)
        schema(conn)
        conn.execute('CREATE TABLE IF NOT EXISTS corporate_refreshes(id INTEGER PRIMARY KEY,recorded_at TEXT,state TEXT,through_date TEXT)')
        conn.commit()
        try:
            remaining=deadline-monotonic()
            if remaining<=0:raise TimeoutError('权益刷新预算到期')
            catalog=book.collect(starts,at.isoformat(),remaining) if starts else book.read()
            if monotonic()>=deadline:raise TimeoutError('权益刷新预算到期，未修改账户')
            recorded_at=now().isoformat()
            result=settle(conn,catalog['actions'],MARKETS['cn']['initial_cash'],through,recorded_at,catalog['coverage'])
            conn.execute('INSERT INTO corporate_refreshes(recorded_at,state,through_date) VALUES(?,?,?)',(recorded_at,'ready',through))
            conn.commit()
            return {'status':'ready','through_date':through,'sources':len(starts),'actions':len(catalog['actions']),
                    'cash_receivable':result['cash_receivable'],'model_requests':0,'orders_submitted':0}
        except Exception:
            conn.execute('INSERT INTO corporate_refreshes(recorded_at,state,through_date) VALUES(?,?,?)',(now().isoformat(),'failed',through))
            conn.commit()
            raise


def coverage_ok(ledger,through):
    """Coverage is explicit and dated, not inferred from a missing event list."""
    settlements=ledger.get('corporate_settlements',[])
    refreshes=ledger.get('corporate_refreshes',[])
    if not settlements or (refreshes and refreshes[-1]['state']!='ready'):return False
    current=settlements[-1]
    if current['through_date']<through:return False
    coverage=json.loads(current['coverage'])
    for symbol in {r['symbol'] for r in ledger['trades'] if r['trade_date']<=through}:
        start=min(r['trade_date'] for r in ledger['trades'] if r['symbol']==symbol)
        row=coverage.get(symbol,{})
        if row.get('from_date','9999')>start or row.get('through_date','')<through:return False
    return True
