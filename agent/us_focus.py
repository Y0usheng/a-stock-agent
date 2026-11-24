"""After-close US research. The host uses the shadow path only."""
import asyncio
from datetime import datetime

from config import MARKETS, NEWS_DB
from evals.news_store import ingest_events, query_events
from servers.data_quality import market_zone
from servers.us_research_sources import ISSUER_FEEDS, ISSUER_SOURCES, PRICE_SOURCE, issuer_releases, nasdaq_bars

SYMBOLS = ('NVDA', 'AMD', 'MSFT')
PROMPT_VERSION = 'manager-us-focus-v1'


def session_gate(now=None):
    import pandas_market_calendars as calendars
    zone = market_zone('us')
    local = (now or datetime.now(zone)).astimezone(zone)
    day = local.date().isoformat()
    schedule = calendars.get_calendar('NASDAQ').schedule(start_date=day, end_date=day)
    if schedule.empty:
        return day, None, 'us_exchange_holiday'
    close = schedule.iloc[0]['market_close'].to_pydatetime().astimezone(zone)
    return day, close, 'us_regular_session_still_open' if local < close else None


def _event_priority(row):
    title = row['title'].casefold()
    if any(phrase in title for phrase in ('ceo transition', 'reinstates', 'resigns')):
        return 0
    return 1 if 'financial results' in title or 'quarter' in title else 2


async def market_inputs(day, close, *, db_path=NEWS_DB):
    import pandas as pd
    from servers.us_market_server import indicators
    names = {row['symbol']: row['name'] for row in MARKETS['us']['candidate_universe']}
    observed = datetime.now(market_zone('us'))
    quotes, releases = await asyncio.gather(
        asyncio.gather(*(asyncio.to_thread(nasdaq_bars, symbol, day) for symbol in SYMBOLS)),
        asyncio.gather(*(asyncio.to_thread(issuer_releases, symbol, observed) for symbol in ISSUER_FEEDS),
                       return_exceptions=True),
    )
    snapshot = [{**indicators(symbol, pd.DataFrame(rows), PRICE_SOURCE), 'name': names[symbol],
                 'price_basis': 'as_returned_by_exchange_historical_series'}
                for symbol, rows in zip(SYMBOLS, quotes)]
    unavailable, available = [], []
    for symbol, rows in zip(ISSUER_FEEDS, releases):
        if isinstance(rows, Exception):
            unavailable.append(symbol)
        else:
            ingest_events(rows, 'us', 'live', db_path)
            available.append(ISSUER_FEEDS[symbol][0])
    cutoff = datetime.now(market_zone('us')).isoformat()
    candidates = []
    for symbol in SYMBOLS:
        events = query_events('us', symbol, day, days=30, limit=24, db_path=db_path, information_cutoff_at=cutoff)
        events = [row for row in events if row['source'] in ISSUER_SOURCES]
        events = sorted(events, key=_event_priority)[:3]
        candidates.append({'symbol': symbol, 'name': names[symbol], 'article_count': len(events), 'top_events': events})
    if unavailable or not any(row['article_count'] for row in candidates):
        raise RuntimeError('美股发行人公告源读取失败或无可验证事件')
    return snapshot, {'ok': True, 'market': 'us', 'as_of_date': day, 'offline': False,
                      'information_cutoff_at': cutoff, 'regular_close_at': close.isoformat(),
                      'data_sources': available, 'candidates': candidates}


async def run_focus(host, day, close):
    from agent.run_daily import _assert_live_data, run_shadow
    from servers.data_quality import snapshot_date
    snapshot, news = await asyncio.wait_for(market_inputs(day, close), timeout=45)
    if snapshot_date(snapshot) != day:
        raise RuntimeError('美股行情未覆盖当前已收盘交易日')
    _assert_live_data(snapshot, news)
    host.active_symbols, host.news_screen = list(SYMBOLS), news
    # Financial results can be cited from timestamped issuer releases. Do not
    # turn an unversioned provider info snapshot into a filed financial record.
    fundamentals = [{'symbol': s, 'ok': False, 'data_source': 'unavailable(尚未接入带披露时点的美股财务报表)'}
                    for s in SYMBOLS]
    return await run_shadow(host, snapshot, day, False, None, None, [PROMPT_VERSION],
                            frameworks=['langgraph'], model_access='subscription', fundamentals=fundamentals)
