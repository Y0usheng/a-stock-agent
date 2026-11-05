from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from evals.portfolio_comparison import ComparisonBook, SPEC, build_comparison, normalize_index
from evals.public_reports import sanitize_view

DAYS = ['2026-10-26','2026-10-27','2026-10-28','2026-10-29','2026-10-30','2026-11-02']
STAMP = '2026-10-27T17:00:00+08:00'


def rows(second=11):
    return [{'date':DAYS[0],'close':10.}, {'date':DAYS[1],'close':float(second)}]


def cash_ledger():
    return {'cash':1000.,'positions':[],'trades':[]}


class PortfolioComparisonTests(unittest.TestCase):
    def book(self, root):
        book = ComparisonBook(root)
        policy = book.register(DAYS, '2026-10-26T10:00:00+08:00')
        return book, policy

    def test_registration_is_prospective_idempotent_and_calendar_driven(self):
        with tempfile.TemporaryDirectory() as tmp:
            book, policy = self.book(Path(tmp))
            self.assertEqual(policy['start_session'], DAYS[0])
            self.assertEqual(book.register(DAYS, STAMP), policy)
            self.assertIsNone(book.context('2026-10-25T23:59:59+08:00'))
        with tempfile.TemporaryDirectory() as tmp:
            policy = ComparisonBook(Path(tmp)).register(DAYS, '2026-10-30T16:00:00+08:00')
            self.assertEqual(policy['start_session'], '2026-11-02')

    def test_waits_for_registered_close_and_never_uses_earlier_prices(self):
        with tempfile.TemporaryDirectory() as tmp:
            book, policy = self.book(Path(tmp))
            state = book.observe(policy, rows(), [{'date':DAYS[0],'total':1000.}], cash_ledger(), DAYS,
                                 '2026-10-26T11:00:00+08:00', initial=1000)
            self.assertEqual(state, 'waiting_for_start')
            self.assertFalse((Path(tmp)/'observations.json').exists())
            result = build_comparison(book.context(STAMP), '2026-10-23')
            self.assertEqual(result['series'], [])
            self.assertNotIn('return_gap_pp', result)

    def test_cash_vs_rising_index_has_opportunity_gap_not_fictitious_profit(self):
        with tempfile.TemporaryDirectory() as tmp:
            book, policy = self.book(Path(tmp))
            curve = [{'date':d,'total':1000.} for d in DAYS[:2]]
            book.observe(policy, rows(), curve, cash_ledger(), DAYS, STAMP, initial=1000)
            book.status(STAMP, 'ready', DAYS[1])
            result = build_comparison(book.context(STAMP), DAYS[1])
            self.assertEqual(result['portfolio_return_pct'], 0)
            self.assertEqual(result['benchmark_return_pct'], 10)
            self.assertEqual(result['return_gap_pp'], -10)
            self.assertEqual(result['relative_pnl'], -100)
            self.assertEqual(result['cash_opportunity_amount'], 100)
            self.assertEqual(result['cash_ratio_pct'], 100)

    def test_cash_avoids_index_fall_without_booking_it_as_actual_profit(self):
        with tempfile.TemporaryDirectory() as tmp:
            book, policy = self.book(Path(tmp))
            book.observe(policy, rows(9), [{'date':d,'total':1000.} for d in DAYS[:2]],cash_ledger(),DAYS,STAMP,initial=1000)
            book.status(STAMP,'ready',DAYS[1])
            result = build_comparison(book.context(STAMP),DAYS[1])
            self.assertEqual(result['portfolio_return_pct'],0)
            self.assertEqual(result['return_gap_pp'],10)
            self.assertEqual(result['cash_opportunity_amount'],-100)

    def test_replay_is_idempotent_and_source_or_ledger_revision_freezes_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            book,policy=self.book(Path(tmp));curve=[{'date':d,'total':1000.} for d in DAYS[:2]]
            book.observe(policy,rows(),curve,cash_ledger(),DAYS,STAMP,initial=1000)
            before=(Path(tmp)/'observations.json').read_bytes()
            book.observe(policy,rows(),curve,cash_ledger(),DAYS,'2026-10-28T17:00:00+08:00',initial=1000)
            self.assertEqual((Path(tmp)/'observations.json').read_bytes(),before)
            with self.assertRaises(ValueError):
                book.observe(policy,rows(12),curve,cash_ledger(),DAYS,STAMP,initial=1000)
            altered=deepcopy(curve);altered[0]['total']=999.
            with self.assertRaises(ValueError):
                book.observe(policy,rows(),altered,cash_ledger(),DAYS,STAMP,initial=1000)
            self.assertEqual((Path(tmp)/'observations.json').read_bytes(),before)

    def test_missing_index_or_portfolio_date_does_not_silently_change_interval(self):
        with tempfile.TemporaryDirectory() as tmp:
            book,policy=self.book(Path(tmp));curve=[{'date':d,'total':1000.} for d in DAYS[:2]]
            with self.assertRaises(ValueError):
                book.observe(policy,rows()[:1],curve,cash_ledger(),DAYS,STAMP,initial=1000)
            self.assertFalse((Path(tmp)/'observations.json').exists())
            index=rows()+[{'date':DAYS[2],'close':12}]
            gap=[curve[0],{'date':DAYS[2],'total':1000.}]
            with self.assertRaises(ValueError):
                book.observe(policy,index,gap,cash_ledger(),DAYS,'2026-10-28T17:00:00+08:00',initial=1000)

    def test_stock_holdings_do_not_receive_strict_total_return_score_before_equity_accounting(self):
        with tempfile.TemporaryDirectory() as tmp:
            book,policy=self.book(Path(tmp));ledger={'cash':900.,'positions':[{'symbol':'300750','shares':10}],
                'trades':[{'symbol':'300750','action':'BUY','shares':10,'price':10.,'fee':0.,'trade_date':DAYS[0]}]}
            book.observe(policy,rows(),[{'date':DAYS[0],'total':1000.},{'date':DAYS[1],'total':1100.}],ledger,DAYS,STAMP,initial=1000)
            book.status(STAMP,'ready',DAYS[1]);result=build_comparison(book.context(STAMP),DAYS[1])
            self.assertEqual(result['status'],'limited_accounting')
            self.assertNotIn('return_gap_pp',result)
            self.assertFalse(result['all_cash'])

    def test_historical_cutoff_hides_later_observations_and_refresh_failures(self):
        with tempfile.TemporaryDirectory() as tmp:
            book,policy=self.book(Path(tmp))
            book.observe(policy,rows(),[{'date':d,'total':1000.} for d in DAYS[:2]],cash_ledger(),DAYS,STAMP,initial=1000)
            book.status(STAMP,'refresh_failed')
            then=book.context('2026-10-26T23:59:59+08:00')
            self.assertEqual(then['points'],[])
            self.assertEqual(then['refresh'],{})
            self.assertEqual(build_comparison(book.context(STAMP),DAYS[1])['status'],'stale')

    def test_index_identity_finite_values_and_closed_dates_are_enforced(self):
        source={'indexCode':'H00300','indexNameCnAll':'沪深300全收益指数','tradeDate':'20261026','close':100}
        self.assertEqual(normalize_index([source],STAMP),[{'date':DAYS[0],'close':100.}])
        for change in ({'indexCode':'000300'},{'indexNameCnAll':'沪深300价格指数'},{'close':float('nan')},{'close':0}):
            with self.assertRaises(ValueError):normalize_index([{**source,**change}],STAMP)
        with self.assertRaises(ValueError):normalize_index([source,{**source,'close':101}],STAMP)
        with self.assertRaises(ValueError):normalize_index([source],'2026-10-26T14:00:00+08:00')

    def test_later_source_failure_does_not_change_earlier_healthy_report_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            book,policy=self.book(Path(tmp))
            book.observe(policy,rows(),[{'date':d,'total':1000.} for d in DAYS[:2]],cash_ledger(),DAYS,STAMP,initial=1000)
            book.status(STAMP,'ready',DAYS[1])
            book.status('2026-10-28T17:00:00+08:00','refresh_failed')
            historical=build_comparison(book.context(STAMP),DAYS[1])
            self.assertEqual(historical['status'],'ready')
            self.assertEqual(historical['return_gap_pp'],-10)

    def test_public_projection_is_explicit_and_contains_no_ledger_or_policy_internals(self):
        view={'comparison':{'status':'ready','registered_at':'2026-10-26T10:00:00+08:00',
            'source_url':SPEC['source_url'],'raw_ledger':'PRIVATE','journal_hash':'PRIVATE',
            'series':[{'date':DAYS[0],'portfolio_return_pct':0,'benchmark_return_pct':0,'cash_return_pct':0,'journal_hash':'PRIVATE'}]}}
        public=sanitize_view(view)
        self.assertNotIn('PRIVATE',json.dumps(public))
        self.assertEqual(public,sanitize_view(public))
        self.assertEqual(public['comparison']['source_url'],SPEC['source_url'])

    def test_refresh_failure_preserves_existing_policy_and_never_invokes_research(self):
        from scripts.refresh_portfolio_comparison import main
        with tempfile.TemporaryDirectory() as tmp:
            book,policy=self.book(Path(tmp));before=(Path(tmp)/'policy.json').read_bytes()
            with patch('scripts.refresh_portfolio_comparison.ComparisonBook',return_value=book),\
                 patch('scripts.refresh_portfolio_comparison.calendar',return_value={'sessions':DAYS}),\
                 patch('scripts.refresh_portfolio_comparison.now',return_value=datetime.fromisoformat(STAMP)),\
                 patch('requests.get',side_effect=OSError('PRIVATE')),patch('builtins.print'):
                self.assertEqual(main(),2)
            self.assertEqual((Path(tmp)/'policy.json').read_bytes(),before)
            self.assertEqual(book.context(STAMP)['refresh']['state'],'refresh_failed')

    def test_refresh_real_response_contract_and_missing_prices_fail_after_start(self):
        from scripts.refresh_portfolio_comparison import main
        with tempfile.TemporaryDirectory() as tmp:
            book,policy=self.book(Path(tmp))
            response=MagicMock(status_code=200,content=b'{}')
            response.json.return_value={'data':[{'indexCode':'H00300','indexNameCnAll':'沪深300全收益指数',
                'tradeDate':'20261027','close':100}]}
            with patch('scripts.refresh_portfolio_comparison.ComparisonBook',return_value=book),\
                 patch('scripts.refresh_portfolio_comparison.calendar',return_value={'sessions':DAYS}),\
                 patch('scripts.refresh_portfolio_comparison.now',return_value=datetime.fromisoformat(STAMP)),\
                 patch('scripts.refresh_portfolio_comparison.PRICE_CACHE',Path(tmp)/'no-prices'),\
                 patch('requests.get',return_value=response) as request,patch('builtins.print'):
                self.assertEqual(main(),2)
            request.assert_called_once()
            self.assertFalse(request.call_args.kwargs['allow_redirects'])
            self.assertEqual(request.call_args.kwargs['params']['indexCode'],'H00300')
            self.assertEqual(book.context(STAMP)['refresh']['state'],'waiting_for_portfolio_prices')
            self.assertEqual(build_comparison(book.context(STAMP),DAYS[1])['status'],'waiting_for_portfolio_prices')
            self.assertFalse((Path(tmp)/'observations.json').exists())

    def test_refresh_before_start_is_successful_waiting_without_returns(self):
        from scripts.refresh_portfolio_comparison import main
        with tempfile.TemporaryDirectory() as tmp:
            book=ComparisonBook(Path(tmp));stamp='2026-10-27T10:00:00+08:00'
            response=MagicMock(status_code=200,content=b'{}')
            response.json.return_value={'data':[{'indexCode':'H00300','indexNameCnAll':'沪深300全收益指数',
                'tradeDate':'20261026','close':100}]}
            with patch('scripts.refresh_portfolio_comparison.ComparisonBook',return_value=book),\
                 patch('scripts.refresh_portfolio_comparison.calendar',return_value={'sessions':DAYS}),\
                 patch('scripts.refresh_portfolio_comparison.now',return_value=datetime.fromisoformat(stamp)),\
                 patch('scripts.refresh_portfolio_comparison.PRICE_CACHE',Path(tmp)/'no-prices'),\
                 patch('requests.get',return_value=response),patch('builtins.print'):
                self.assertEqual(main(),0)
            result=build_comparison(book.context(stamp),DAYS[0])
            self.assertEqual(result['status'],'waiting_for_start')
            self.assertEqual(result['series'],[])
            self.assertNotIn('portfolio_return_pct',result)

    def test_calendar_fetch_crossing_close_cannot_backdate_registration(self):
        from scripts.refresh_portfolio_comparison import main
        with tempfile.TemporaryDirectory() as tmp:
            book=ComparisonBook(Path(tmp))
            response=MagicMock(status_code=200,content=b'{}')
            response.json.return_value={'data':[{'indexCode':'H00300','indexNameCnAll':'沪深300全收益指数',
                'tradeDate':'20261026','close':100}]}
            after='2026-10-26T15:00:01+08:00'
            with patch('scripts.refresh_portfolio_comparison.ComparisonBook',return_value=book),\
                 patch('scripts.refresh_portfolio_comparison.calendar',return_value={'sessions':DAYS}),\
                 patch('scripts.refresh_portfolio_comparison.now',side_effect=[datetime.fromisoformat('2026-10-26T14:59:59+08:00'),datetime.fromisoformat(after)]),\
                 patch('scripts.refresh_portfolio_comparison.PRICE_CACHE',Path(tmp)/'no-prices'),\
                 patch('requests.get',return_value=response),patch('builtins.print'):
                self.assertEqual(main(),0)
            self.assertEqual(book.context(after)['policy']['start_session'],DAYS[1])
            self.assertEqual(book.context(after)['policy']['registered_at'],after)


if __name__=='__main__':unittest.main()
