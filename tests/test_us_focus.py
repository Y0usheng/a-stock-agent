import unittest
from datetime import date, datetime, timedelta
from unittest.mock import patch
from zoneinfo import ZoneInfo

from agent.evidence import build_catalog
from agent.us_focus import session_gate
from agent.run_daily import run_market
from servers.us_research_sources import parse_issuer_feed, parse_nasdaq_bars


class USFocusTests(unittest.IsolatedAsyncioTestCase):
    def test_calendar_skips_thanksgiving_and_recognizes_early_close(self):
        zone = ZoneInfo('America/New_York')
        holiday = datetime(2026, 11, 26, 17, 30, tzinfo=zone)
        self.assertEqual(session_gate(holiday), ('2026-11-26', None, 'us_exchange_holiday'))
        early = datetime(2026, 11, 27, 13, 5, tzinfo=zone)
        self.assertEqual(session_gate(early), ('2026-11-27', early.replace(minute=0), None))
        self.assertEqual(session_gate(early.replace(hour=12))[2], 'us_regular_session_still_open')

    async def test_us_focus_cannot_invoke_paper_execution_or_access_data_on_holiday(self):
        with patch('agent.run_daily.McpHost') as host:
            with self.assertRaises(ValueError):
                await run_market('us', False, framework='langgraph', model_access='subscription', us_focus=True)
            with patch('agent.us_focus.session_gate', return_value=('2026-11-26', None, 'us_exchange_holiday')):
                result = await run_market('us', False, shadow=True, framework='langgraph',
                                          model_access='subscription', us_focus=True)
            self.assertEqual(result['status'], 'skipped')
            host.assert_not_called()

    def test_nasdaq_requires_same_symbol_and_exact_session(self):
        right = date(2026, 11, 23)
        bars = [{'date': (right-timedelta(days=i)).strftime('%m/%d/%Y'), 'open':'$150.12',
                 'close':'$165.05', 'high':'$169.13', 'low':'$148.95', 'volume':'100,500'} for i in range(65)]
        payload = {'data': {'symbol':'NVDA', 'tradesTable': {'rows': bars}}}
        self.assertEqual(parse_nasdaq_bars(payload, 'NVDA', right.isoformat())[-1]['close'], 165.05)
        with self.assertRaises(ValueError):
            parse_nasdaq_bars(payload, 'AMD', right.isoformat())
        with self.assertRaises(ValueError):
            parse_nasdaq_bars(payload, 'NVDA', '2026-11-24')

    def test_issuer_provenance_and_after_close_information_boundary(self):
        posted = 'Mon, 23 Nov 2026 21:20:00 GMT'
        raw = (f'<rss><channel><item><title>NVIDIA and Microsoft Expand Computing</title>'
               f'<link>https://blogs.nvidia.com/blog/event/</link><pubDate>{posted}</pubDate></item>'
               f'<item><title>Third-party rumor</title><link>https://example.com/rumor</link>'
               f'<pubDate>{posted}</pubDate></item></channel></rss>').encode()
        cutoff = '2026-11-23T17:30:00-05:00'
        rows = parse_issuer_feed(raw, 'NVDA', datetime.fromisoformat(cutoff))
        self.assertEqual([row['symbol'] for row in rows], ['NVDA', 'MSFT'])
        event = {**rows[-1], 'data_quality':'live', 'first_seen_at':'2026-11-23T17:25:00-05:00'}
        ctx = {'date':'2026-11-23','symbols':['MSFT'],'snapshot':[],'fundamentals':[],
               'news':[{'top_events':[event]}], 'research_cutoff_at':cutoff,
               'regular_close_at':'2026-11-23T16:00:00-05:00'}
        record = build_catalog(ctx, 'us')['records'][0]
        self.assertTrue(record['eligible'])
        self.assertEqual(record['source_tier'], 'issuer_communication')
        self.assertEqual(record['payload']['source'], 'NVIDIA Blog')
        event['first_seen_at'] = '2026-11-23T17:31:00-05:00'
        self.assertFalse(build_catalog(ctx, 'us')['records'][0]['eligible'])
