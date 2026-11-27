"""US forecasts and delayed settlement. The host uses the shadow path only."""
import asyncio
from datetime import datetime

from agent.us_forecast import VERSION, direction, feedback, forecast_path, reference_session, settle_forecasts
from config import MARKETS, NEWS_DB
from evals.news_store import ingest_events, query_events
from servers.data_quality import market_zone
from servers.us_research_sources import ISSUER_FEEDS, ISSUER_SOURCES, PRICE_SOURCE, issuer_releases, nasdaq_bars

SYMBOLS = ('NVDA', 'AMD', 'MSFT')
PROMPT_VERSION = 'manager-us-focus-v3'


def _event_priority(row):
    title = row['title'].casefold()
    if any(phrase in title for phrase in ('ceo transition', 'reinstates', 'resigns')):
        return 0
    return 1 if 'financial results' in title or 'quarter' in title else 2


async def issuer_news(day, close, *, db_path=NEWS_DB):
    names = {row['symbol']: row['name'] for row in MARKETS['us']['candidate_universe']}
    observed = datetime.now(market_zone('us'))
    releases = await asyncio.gather(*(asyncio.to_thread(issuer_releases, symbol, observed) for symbol in ISSUER_FEEDS),
                                    return_exceptions=True)
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
    return {'ok': True, 'market': 'us', 'as_of_date': day, 'offline': False,
            'information_cutoff_at': cutoff, 'regular_close_at': close.isoformat(),
            'data_sources': available, 'candidates': candidates}


async def gather_forecast(window):
    import pandas as pd
    from servers.us_market_server import indicators
    completed = list(window['completed'])
    latest = completed[-1]
    quotes = await asyncio.gather(*(asyncio.to_thread(nasdaq_bars, s, latest, require_session=False) for s in SYMBOLS))
    histories = dict(zip(SYMBOLS, quotes))
    settle_forecasts(histories, datetime.now(market_zone('us')).isoformat())
    target = window['target_session']
    if forecast_path(target).exists():
        return None
    day = reference_session(histories, completed)
    if day is None:
        raise ValueError('Nasdaq 缺少共同的近期真实收盘价，无法建立预测基线')
    names = {row['symbol']: row['name'] for row in MARKETS['us']['candidate_universe']}
    snapshot = [{**indicators(symbol, pd.DataFrame([row for row in histories[symbol] if row['date'] <= day]), PRICE_SOURCE),
                 'name': names[symbol], 'price_basis': 'as_returned_by_exchange_historical_series'}
                for symbol in SYMBOLS]
    news = await issuer_news(day, window['completed'][day])
    horizon = len(completed) - completed.index(day)
    news['forecast'] = {k: window[k] for k in ('target_session', 'target_open_at', 'target_close_at')}
    news['forecast'].update(protocol=VERSION, reference_session=day, last_completed_session=latest,
                            information_cutoff_at=news['information_cutoff_at'], horizon_days=horizon,
                            references={row['symbol']: {'close': row['close'], 'baseline_direction': direction(row['change_pct'])}
                                        for row in snapshot})
    news['forecast_feedback'] = feedback(news['information_cutoff_at'], horizon)
    return day, snapshot, news


async def run_focus(host, window):
    from agent.run_daily import _assert_live_data, run_shadow
    data = await asyncio.wait_for(gather_forecast(window), timeout=45)
    if data is None:
        return {'status': 'skipped', 'reason': 'forecast_exists', 'log_path': str(forecast_path(window['target_session']))}
    day, snapshot, news = data
    _assert_live_data(snapshot, news)
    host.active_symbols, host.news_screen = list(SYMBOLS), news
    # Financial results can be cited from timestamped issuer releases. Do not
    # turn an unversioned provider info snapshot into a filed financial record.
    fundamentals = [{'symbol': s, 'ok': False, 'data_source': 'unavailable(尚未接入带披露时点的美股财务报表)'}
                    for s in SYMBOLS]
    return await run_shadow(host, snapshot, day, False, None, None, [PROMPT_VERSION],
                            frameworks=['langgraph'], model_access='subscription', fundamentals=fundamentals)
