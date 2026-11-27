import json
import tempfile
import unittest
from datetime import date, datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, patch
from zoneinfo import ZoneInfo

from agent.evidence import build_catalog
from agent.run_daily import run_market
from agent.subscription_roles import SubscriptionRoles
from agent.us_forecast import VERSION, feedback, forecast_path, forecast_window, reference_session, settle_forecasts, validate_lock
from agent.analysis import write_log
from evals.experiments import canonical_hash
from servers.us_research_sources import PRICE_SOURCE, parse_issuer_feed, parse_nasdaq_bars


class USFocusTests(unittest.IsolatedAsyncioTestCase):
    async def test_us_technical_role_uses_real_snapshot_without_account_or_model(self):
        context = {'date': '2026-11-25', 'symbols': ['NVDA'], 'account': None,
                   'research_cutoff_at': '2026-11-27T08:00:00-05:00',
                   'regular_close_at': '2026-11-25T16:00:00-05:00',
                   'snapshot': [{'symbol': 'NVDA', 'date': '2026-11-25', 'close': 149.30,
                                 'volume': 100_000, 'data_source': PRICE_SOURCE}],
                   'fundamentals': [], 'forecast': {'horizon_days': 1}, 'forecast_feedback': {}}
        with patch('agent.subscription_roles.SubscriptionClient') as client:
            client.return_value.complete = AsyncMock()
            roles = SubscriptionRoles(context, 'us', '', 'test')
            result = json.loads(await roles.call('technical-analyst', ''))
            client.return_value.complete.assert_not_awaited()
        evidence = result['claims'][0]['evidence']
        self.assertEqual(len(evidence['supporting_ids']), 1)
        self.assertEqual(evidence['contradicting_ids'], [])
        self.assertIsNone(roles.context['paper_account'])

    def test_calendar_skips_thanksgiving_and_recognizes_early_close(self):
        zone = ZoneInfo('America/New_York')
        holiday = datetime(2026, 11, 26, 17, 30, tzinfo=zone)
        self.assertEqual(forecast_window(holiday), (None, 'us_exchange_holiday'))
        early = datetime(2026, 11, 27, 13, 5, tzinfo=zone)
        window, reason = forecast_window(early)
        self.assertIsNone(reason)
        self.assertEqual(window['target_session'], '2026-11-30')
        self.assertEqual(window['completed']['2026-11-27'], early.replace(minute=0))
        self.assertEqual(forecast_window(early.replace(hour=12))[1], 'us_regular_session_still_open')
        preopen, reason = forecast_window(early.replace(hour=8))
        self.assertIsNone(reason)
        self.assertEqual(preopen['target_session'], '2026-11-27')
        self.assertEqual(preopen['target_close_at'], '2026-11-27T13:00:00-05:00')

    def test_forecast_uses_latest_common_close_with_one_session_lag(self):
        histories = {'NVDA': [{'date': '2026-11-25'}, {'date': '2026-11-27'}],
                     'AMD': [{'date': '2026-11-25'}]}
        completed = ['2026-11-24', '2026-11-25', '2026-11-27']
        self.assertEqual(reference_session(histories, completed), '2026-11-25')
        histories['AMD'] = [{'date': '2026-11-24'}]
        self.assertIsNone(reference_session(histories, completed))

    def test_only_a_frozen_forecast_with_target_close_enters_feedback(self):
        target = '2026-11-27'
        frozen = {'protocol': VERSION, 'target_session': target, 'reference_session': '2026-11-25',
                  'information_cutoff_at': target + 'T08:00:00-05:00',
                  'target_open_at': target + 'T09:30:00-05:00', 'target_close_at': target + 'T13:00:00-05:00',
                  'horizon_days': 1, 'references': {'NVDA': {'close': 150, 'baseline_direction': 'down'}}}
        report_at = target + 'T08:05:00-05:00'
        with self.assertRaises(ValueError):
            validate_lock(frozen, frozen['target_open_at'])
        dataset = {'news_screen': {'forecast': frozen}, 'snapshot': [{'symbol': 'NVDA', 'close': 150}]}
        report = {'us_forecast': {**frozen, 'locked_at': report_at}, 'market': 'us',
                  'execution_mode': 'shadow', 'execution': {'orders_submitted': 0},
                  'input_dataset': dataset, 'dataset_fingerprint': canonical_hash(dataset),
                  'data_sources': [PRICE_SOURCE],
                  'predictions': [{'symbol': 'NVDA', 'direction': 'up', 'confidence': .6}]}
        with tempfile.TemporaryDirectory() as folder:
            logs = Path(folder)
            receipts = logs / 'receipts'
            write_log(forecast_path(target, logs), report)
            outcome = receipts / f'{target}.json'
            actual = {'NVDA': [{'date': target, 'close': 153}]}
            observed = target + 'T15:00:00-05:00'
            settle_forecasts(actual, target + 'T12:59:00-05:00', log_dir=logs, outcomes=receipts)
            settle_forecasts({'NVDA': []}, observed, log_dir=logs, outcomes=receipts)
            self.assertFalse(outcome.exists())
            settle_forecasts(actual, observed, log_dir=logs, outcomes=receipts)
            stored = outcome.read_bytes()
            self.assertEqual(json.loads(stored)['rows'][0]['return_from_reference_pct'], 2)
            self.assertEqual(feedback(report_at, 1, outcomes=receipts)['sample_count'], 0)
            self.assertEqual(feedback(observed, 2, outcomes=receipts)['sample_count'], 0)
            learned = feedback(observed, 1, outcomes=receipts)
            self.assertEqual((learned['sample_count'], learned['accuracy'], learned['baseline_accuracy']), (1, 1, 0))
            self.assertEqual(learned['calibration_review']['status'], 'insufficient_data')
            actual['NVDA'][0]['close'] = 151
            settle_forecasts(actual, observed, log_dir=logs, outcomes=receipts)
            self.assertEqual(outcome.read_bytes(), stored)

    async def test_us_focus_cannot_invoke_paper_execution_or_access_data_on_holiday(self):
        with patch('agent.run_daily.McpHost') as host:
            with self.assertRaises(ValueError):
                await run_market('us', False, framework='langgraph', model_access='subscription', us_focus=True)
            with patch('agent.us_forecast.forecast_window', return_value=(None, 'us_exchange_holiday')):
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
        self.assertEqual(parse_nasdaq_bars(payload, 'NVDA', '2026-11-24', require_session=False)[-1]['date'],
                         right.isoformat())

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
