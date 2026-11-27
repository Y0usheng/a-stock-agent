"""Public exchange prices and timestamped issuer releases for US research."""
from datetime import date, datetime, timedelta
from email.utils import parsedate_to_datetime
from html import unescape
import json
import math
import re
from urllib.parse import urlsplit
from xml.etree import ElementTree

import requests

PRICE_SOURCE = 'Nasdaq historical daily OHLCV'
ISSUER_FEEDS = {
    'NVDA': ('NVIDIA Newsroom', 'https://nvidianews.nvidia.com/releases.xml'),
    'AMD': ('AMD Investor Relations', 'https://ir.amd.com/news-events/press-releases/rss'),
}
ISSUER_HOSTS = frozenset(urlsplit(url).hostname for _, url in ISSUER_FEEDS.values())
NVIDIA_BLOG = 'blogs.nvidia.com'
ISSUER_SOURCES = frozenset(name for name, _ in ISSUER_FEEDS.values()) | {'NVIDIA Blog'}
_HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/131.0.0.0 Safari/537.36',
            'Referer': 'https://www.nasdaq.com/', 'Accept': '*/*'}


def _public_body(url, *, params=None, size=1_000_000):
    with requests.get(url, params=params, headers=_HEADERS, timeout=(5, 12),
                      stream=True, allow_redirects=False) as response:
        response.raise_for_status()
        if response.status_code != 200:
            raise ValueError('Unexpected public source response')
        data = bytearray()
        for chunk in response.iter_content(32768):
            data.extend(chunk)
            if len(data) > size:
                raise ValueError('Public source response exceeded size limit')
        return bytes(data)


def parse_nasdaq_bars(value, symbol, session, days=120, *, require_session=True):
    data = value.get('data') or {}
    if data.get('symbol') != symbol:
        raise ValueError('Nasdaq symbol mismatch')
    result = {}
    for row in (data.get('tradesTable') or {}).get('rows') or []:
        day = datetime.strptime(row['date'], '%m/%d/%Y').date().isoformat()
        if day > session or day in result:
            raise ValueError('Unexpected or duplicate Nasdaq session')
        prices = {k: float(row[k].replace('$', '').replace(',', '')) for k in ('open', 'high', 'low', 'close')}
        volume = int(row['volume'].replace(',', ''))
        if (volume <= 0 or any(not math.isfinite(v) or v <= 0 for v in prices.values())
                or prices['low'] > min(prices['open'], prices['close'])
                or prices['high'] < max(prices['open'], prices['close'])):
            raise ValueError('Invalid Nasdaq daily bar')
        result[day] = {'date': day, **prices, 'volume': volume}
    ordered = [result[day] for day in sorted(result)]
    if len(ordered) < 60 or (require_session and ordered[-1]['date'] != session):
        raise ValueError('Nasdaq prices do not cover the required closed trading session')
    return ordered[-days:]


def nasdaq_bars(symbol, session, *, require_session=True):
    from config import MARKETS
    if symbol not in {row['symbol'] for row in MARKETS['us']['candidate_universe']}:
        raise ValueError('Symbol outside the US research universe')
    start = date.fromisoformat(session) - timedelta(days=365)
    params = {'assetclass': 'stocks', 'fromdate': start.isoformat(), 'todate': session, 'limit': 365}
    data = json.loads(_public_body(f'https://api.nasdaq.com/api/quote/{symbol}/historical', params=params))
    return parse_nasdaq_bars(data, symbol, session, require_session=require_session)


def _plain(text, limit):
    return ' '.join(unescape(re.sub(r'<[^>]+>', ' ', text or '')).split())[:limit]


def parse_issuer_feed(raw, symbol, observed_at):
    if b'<!DOCTYPE' in raw.upper() or b'<!ENTITY' in raw.upper():
        raise ValueError('Unexpected RSS document declarations')
    name, feed = ISSUER_FEEDS[symbol]
    from evals.experiments import canonical_hash
    items = ElementTree.fromstring(raw).findall('./channel/item')
    result = []
    for item in items[:50]:
        title = _plain(item.findtext('title'), 200)
        link = (item.findtext('link') or '').strip()
        parsed = urlsplit(link)
        from_blog = symbol == 'NVDA' and parsed.netloc == NVIDIA_BLOG
        if (not title or parsed.scheme != 'https' or (parsed.netloc != urlsplit(feed).netloc and not from_blog)
                or not item.findtext('pubDate')):
            continue
        try:
            stamp = parsedate_to_datetime(item.findtext('pubDate'))
        except (ValueError, TypeError):
            continue
        if stamp.tzinfo is None or stamp > observed_at:
            continue
        summary = _plain(item.findtext('description'), 500)
        published = stamp.isoformat()
        affected = [symbol]
        if symbol == 'NVDA' and re.search(r'\bNVIDIA and Microsoft\b', title, re.I):
            affected.append('MSFT')  # NVIDIA remains the publisher of this joint event.
        for ticker in affected:
            event_id = 'us-release-' + canonical_hash([ticker, link, title, published, summary])[:24]
            result.append({'event_id': event_id, 'symbol': ticker, 'title': title,
                           'published_at': published, 'source': 'NVIDIA Blog' if from_blog else name,
                           'url': link, 'summary': summary})
    if not result:
        raise ValueError('No timestamped issuer releases')
    return result


def issuer_releases(symbol, observed_at):
    return parse_issuer_feed(_public_body(ISSUER_FEEDS[symbol][1]), symbol, observed_at)
