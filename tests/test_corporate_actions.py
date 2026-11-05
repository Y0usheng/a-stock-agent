from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from agent.corporate_actions import replay,receivable_value,settle
from agent.corporate_sources import parse_notice
from evals.experiments import canonical_hash
from evals.trading_view import build_view

def event(**changes):
    return {'event_id':'cninfo:test1','symbol':'300750','kind':'distribution',
        'record_date':'2026-10-21','ex_date':'2026-10-22','pay_date':'2026-10-26','listing_date':None,
        'cash_per_share':'1','bonus_per_share':'0','known_at':'2026-10-20T17:00:00+08:00',
        'source':{'url':'https://static.cninfo.com.cn/finalpage/2026-10-20/123.PDF'},**changes}

def trade(day='2026-10-20',action='BUY',qty=10,price=10,fee=0):
    return {'symbol':'300750','name':'宁德时代','action':action,'shares':qty,'price':price,'fee':fee,'trade_date':day}

def ledger(trades=None,action=None):
    return {'trades':trades if trades is not None else [trade()], 'corporate_actions':[action or event()]}

class CorporateReplayTests(unittest.TestCase):
    def test_record_date_ownership_and_delayed_payment_conserve_assets(self):
        l=ledger()
        before=replay(l,1000,'2026-10-21');ex=replay(l,1000,'2026-10-22');paid=replay(l,1000,'2026-10-26')
        self.assertEqual(before['cash'],900)
        self.assertEqual(before['cash_receivable'],0)
        self.assertEqual(ex['cash'],900);self.assertEqual(ex['cash_receivable'],10)
        self.assertEqual(ex['dividend_income'],10)
        self.assertEqual(ex['cash']+90+receivable_value(ex,{'300750':9}),1000)
        self.assertEqual(paid['cash'],910);self.assertEqual(paid['rights'],[])
        self.assertEqual(paid['dividend_income'],10)

    def test_record_close_buys_qualify_but_ex_date_buys_do_not(self):
        self.assertEqual(replay(ledger([trade('2026-10-21')]),1000,'2026-10-26')['cash'],910)
        self.assertEqual(replay(ledger([trade('2026-10-22')]),1000,'2026-10-26')['cash'],900)
        self.assertEqual(replay(ledger([trade(),trade('2026-10-21','SELL')]),1000,'2026-10-26')['dividend_income'],0)

    def test_selling_after_record_keeps_the_dividend_right(self):
        state=replay(ledger([trade(),trade('2026-10-23','SELL',price=9)]),1000,'2026-10-26')
        self.assertEqual(state['cash'],1000)
        self.assertEqual(state['positions'],[])
        self.assertEqual(state['realized_pnl'],-10)
        self.assertEqual(state['dividend_income'],10)

    def test_unlisted_bonus_is_an_asset_and_cost_basis_is_allocated(self):
        action=event(cash_per_share='0',bonus_per_share='1',pay_date=None,listing_date='2026-10-26')
        l=ledger(action=action)
        ex=replay(l,1000,'2026-10-22')
        self.assertEqual(ex['positions'][0]['shares'],10)
        self.assertEqual(ex['positions'][0]['avg_cost'],5)
        self.assertEqual(ex['rights'][0]['bonus_shares'],10)
        self.assertEqual(ex['cash']+50+receivable_value(ex,{'300750':5}),1000)
        listed=replay(l,1000,'2026-10-26')
        self.assertEqual(listed['positions'][0]['shares'],20)
        self.assertEqual(listed['positions'][0]['avg_cost'],5)
        self.assertEqual(listed['positions'][0]['last_buy_date'],'2026-10-20')
        self.assertEqual(listed['rights'],[])

    def test_selling_original_stock_before_bonus_listing_keeps_bonus_and_its_basis(self):
        action=event(cash_per_share='0',bonus_per_share='1',pay_date=None,listing_date='2026-10-26')
        l=ledger([trade(),trade('2026-10-23','SELL',price=5)],action)
        ex=replay(l,1000,'2026-10-23')
        self.assertEqual(ex['positions'],[])
        self.assertEqual(ex['cash'],950)
        self.assertEqual(ex['realized_pnl'],0)
        listed=replay(l,1000,'2026-10-26')
        self.assertEqual(listed['positions'][0]['avg_cost'],5)
        self.assertEqual(listed['positions'][0]['shares'],10)

    def test_bonus_is_not_sellable_before_listing(self):
        action=event(cash_per_share='0',bonus_per_share='1',pay_date=None,listing_date='2026-10-26')
        with self.assertRaises(ValueError):replay(ledger([trade(),trade('2026-10-23','SELL',qty=20,price=5)],action),1000,'2026-10-23')

    def test_forward_split_does_not_create_profit_or_change_trade_count(self):
        action=event(kind='forward_split',cash_per_share='0',bonus_per_share='1',pay_date=None,listing_date='2026-10-22')
        l=ledger(action=action);state=replay(l,1000,'2026-10-22')
        self.assertEqual(state['positions'][0]['shares'],20)
        self.assertEqual(state['cash']+state['positions'][0]['shares']*5,1000)
        self.assertEqual(len(l['trades']),1)

    def test_unknown_fractional_allocations_and_conflicting_events_fail(self):
        with self.assertRaises(ValueError):replay(ledger(action=event(bonus_per_share='.15',listing_date='2026-10-22')),1000,'2026-10-22')
        l=ledger();l['corporate_actions'].append(event(event_id='cninfo:test2'))
        with self.assertRaises(ValueError):replay(l,1000,'2026-10-22')

    def test_missing_payment_and_listing_dates_never_assume_ex_date(self):
        for change in ({'pay_date':None},{'bonus_per_share':'1','listing_date':None}):
            with self.assertRaises(ValueError):replay(ledger(action=event(**change)),1000,'2026-10-26')

    def test_atomic_settlement_is_idempotent_and_revision_cannot_rewrite_cash(self):
        with tempfile.TemporaryDirectory() as tmp:
            conn=sqlite3.connect(Path(tmp)/'test.db');conn.row_factory=sqlite3.Row
            conn.executescript('''CREATE TABLE meta(key TEXT PRIMARY KEY,value REAL);
                INSERT INTO meta VALUES('cash',900);
                CREATE TABLE positions(symbol TEXT PRIMARY KEY,name TEXT,shares INTEGER,avg_cost REAL,last_buy_date TEXT);
                INSERT INTO positions VALUES('300750','宁德时代',10,10,'2026-10-20');
                CREATE TABLE trades(id INTEGER PRIMARY KEY,symbol TEXT,name TEXT,action TEXT,shares INTEGER,price REAL,fee REAL,trade_date TEXT);
                INSERT INTO trades VALUES(1,'300750','宁德时代','BUY',10,10,0,'2026-10-20');''')
            kwargs=dict(initial=1000,through='2026-10-26',recorded_at='2026-10-26T17:00:00+08:00',coverage={})
            a=settle(conn,[event()],**kwargs);b=settle(conn,[event()],**kwargs)
            self.assertEqual(a,b);self.assertEqual(a['cash'],910)
            self.assertEqual(conn.execute('SELECT COUNT(*) FROM corporate_settlements').fetchone()[0],1)
            with self.assertRaises(ValueError):settle(conn,[event(cash_per_share='2')],**kwargs)
            self.assertEqual(conn.execute("SELECT value FROM meta WHERE key='cash'").fetchone()[0],910)
            self.assertEqual(conn.execute('SELECT COUNT(*) FROM trades').fetchone()[0],1)
            conn.close()

    def test_view_uses_raw_prices_plus_rights_not_adjusted_prices_plus_dividend(self):
        from tests.test_trading_view import fixture
        payload,_,logs=fixture()
        payload['price_basis']='qfq'
        raw=deepcopy(payload['histories'])
        for r in raw['300750']:r.update(open=9,close=9,high=10,low=8)
        payload.update(valuation_histories=raw,valuation_price_basis='unadjusted',valuation_fingerprint=canonical_hash(raw))
        l=ledger();state=replay(l,1000,'2026-10-26');l.update(cash=state['cash'],positions=state['positions'],as_of_date='2026-10-26')
        view=build_view(payload,l,logs,1000)
        self.assertEqual(view['portfolio']['total_assets'],1000)
        self.assertEqual(view['portfolio']['net_pnl'],0)
        self.assertEqual(view['stocks'][0]['last_close'],9)
        self.assertEqual(view['portfolio']['dividend_income'],10)
        self.assertEqual(view['portfolio']['trade_count'],1)

    def test_legacy_adjusted_only_prices_cannot_value_a_stock_holding(self):
        from tests.test_trading_view import fixture
        payload,_,logs=fixture();payload['price_basis']='qfq'
        l=ledger(action=event());s=replay(l,1000,'2026-10-26');l.update(cash=s['cash'],positions=s['positions'])
        view=build_view(payload,l,logs,1000)
        self.assertIsNone(view['portfolio']['total_assets'])
        self.assertIsNone(view['positions'][0]['last_price'])

class NoticeContractTests(unittest.TestCase):
    def parse(self,text,symbol='300750',tables=None,title='2026年权益分派实施公告'):
        document={'symbol':symbol,'title':title,'announcement_id':'123',
            'published_at':'2026-10-20','url':'https://static.cninfo.com.cn/finalpage/2026-10-20/123.PDF'}
        text='证券代码：'+symbol+'\n'+text
        pages={'pdf_sha256':'a'*64,'pages':[{'pdf_page':1,'text':text,'text_hash':canonical_hash(text)}], 'tables':tables or []}
        return parse_notice(document,pages,'2026-10-21T17:00:00+08:00')

    def test_implemented_amount_beats_old_proposal_and_ex_reference(self):
        text='''每10股派发现金红利1.1元。按总股本折算每10股现金红利1.096元。
            二、权益分派方案 向A股股东每10股派发现金红利1.099231元（含税），不送红股，不以公积金转增股本。
            三、股权登记日与除权除息日 股权登记日为2026年10月21日，除权除息日为2026年10月22日。
            本公司此次委托中国结算代派的A股股东现金红利将于2026年10月26日通过证券公司直接划入其资金账户。'''
        r=self.parse(text)
        self.assertEqual(r['cash_per_share'],'0.1099231')
        self.assertEqual(r['pay_date'],'2026-10-26')
        self.assertEqual(r['bonus_per_share'],'0')

    def test_shanghai_table_preserves_payment_date_and_per_share_unit(self):
        text='''重要内容提示 每股分配比例 A股每股现金红利1.003元 相关日期
        股份类别 股权登记日 最后交易日 除权（息）日 现金红利发放日
        A股 2026/10/21 － 2026/10/22 2026/10/26
        一、通过方案 二、分配方案 每股全年股息2.016元。'''
        r=self.parse(text,'600036');self.assertEqual(r['cash_per_share'],'1.003')
        self.assertEqual(r['pay_date'],'2026-10-26')

    def test_missing_dates_ambiguous_amounts_and_nonimplemented_scheme_are_rejected(self):
        for text in ['每10股派发现金红利1元','二、权益分派方案 每10股派发现金红利1元；每10股派发现金红利2元。三、股权登记日']:
            with self.assertRaises(ValueError):self.parse(text)

    def test_multiline_table_headers_use_cells_and_do_not_swap_payment_with_listing(self):
        rows=[['股权登记日','除权（息）日','新增无限售条件流\n通股份上市日','现金红利发放日'],
              ['2026/6/9','2026/6/10','2026/6/12','2026/6/15']]
        tables=[{'pdf_page':1,'bbox':[1,2,3,4],'rows':rows,'rows_hash':canonical_hash(rows)}]
        text='''重要内容提示 每股分配比例 每股现金红利0.50元 每股转增0.3股 相关日期
            新增无限售条件流 股权登记日 除权（息）日 现金红利发放日 通股份上市日
            2026/6/9 2026/6/10 2026/6/12 2026/6/15'''
        r=self.parse(text,'600036',tables)
        self.assertEqual(r['record_date'],'2026-06-09');self.assertEqual(r['bonus_per_share'],'0.3')
        self.assertEqual(r['listing_date'],'2026-06-12');self.assertEqual(r['pay_date'],'2026-06-15')

    def test_gross_suffix_and_A_share_heading_exclude_tax_and_reference_amounts(self):
        text='''每股分配比例会改变除息参考价。二、A股权益分派方案
            向全体股东每10股派1.390814元人民币现金（含税；扣税后每10股派1.251733元）。
            三、日期 股权登记日为2026年10月21日，除权除息日及现金红利发放日为2026年10月22日。'''
        self.assertEqual(self.parse(text)['cash_per_share'],'0.1390814')

    def test_explicit_forward_split_and_listing_dates(self):
        text='''本次每1股普通股拆分为2股。股权登记日为2026年10月21日，股份分拆生效日为2026年10月22日，拆分股份上市日为2026年10月26日。'''
        r=self.parse(text,title='股份分拆实施公告')
        self.assertEqual(r['kind'],'forward_split');self.assertEqual(r['cash_per_share'],'0')
        self.assertEqual(r['bonus_per_share'],'1');self.assertEqual(r['listing_date'],'2026-10-26')

class CorporateSourcePaginationTests(unittest.TestCase):
    def test_has_more_on_last_page_is_reconciled_with_total_not_treated_as_truncation(self):
        from agent.corporate_sources import query_notices
        from unittest.mock import MagicMock
        from time import monotonic
        session=MagicMock();first=MagicMock();first.json.return_value=[{'code':'300750','orgId':'test-org'}]
        def rows(start,count):return [{'secCode':'300750','announcementTitle':'董事会决议公告',
            'announcementId':str(i),'announcementTime':1792411200000,'adjunctUrl':'unused'} for i in range(start,start+count)]
        responses=[]
        for value in [{'announcements':rows(0,30),'totalAnnouncement':31,'hasMore':True},
                      {'announcements':rows(30,1),'totalAnnouncement':31,'hasMore':True},
                      {'announcements':[],'totalAnnouncement':0,'hasMore':False}]:
            r=MagicMock();r.json.return_value=value;responses.append(r)
        session.post.side_effect=[first,*responses]
        self.assertEqual(query_notices(session,'300750','2026-10-01','2026-10-26T17:00:00+08:00',monotonic()+20),[])
        self.assertEqual(session.post.call_count,4)

    def test_duplicate_page_does_not_claim_complete_coverage(self):
        from agent.corporate_sources import query_notices
        from unittest.mock import MagicMock
        from time import monotonic
        session=MagicMock();first=MagicMock();first.json.return_value=[{'code':'300750','orgId':'test-org'}]
        response=MagicMock();response.json.return_value={'announcements':[{'secCode':'300750',
            'announcementId':'duplicate','announcementTitle':'董事会公告','announcementTime':1792411200000}],
            'totalAnnouncement':2,'hasMore':True}
        session.post.side_effect=[first,response,response]
        with self.assertRaises(ValueError):query_notices(session,'300750','2026-10-01','2026-10-26T17:00:00+08:00',monotonic()+20)

class CorporateAccountIntegrationTests(unittest.TestCase):
    def setUp(self):
        from tests.test_account_server import account
        self.account=account;self.temp=tempfile.TemporaryDirectory()
        self.path=Path(self.temp.name)/'account.db'
        self.patch=patch.object(account,'DB_PATH',self.path);self.patch.start()
        self.addCleanup(self.patch.stop);self.addCleanup(self.temp.cleanup)
        self.assertTrue(account.authorize_run('before','2026-10-20',['300750'])['ok'])
        self.assertTrue(account.buy('300750','宁德时代',100,10,'fixture','before','first','2026-10-20',1000000)['ok'])

    def settle(self,action,through='2026-10-26'):
        conn=self.account._db()
        try:return settle(conn,[action],self.account.INITIAL_CASH,through,through+'T17:00:00+08:00',
            {'300750':{'from_date':'2026-10-20','through_date':through}})
        finally:conn.close()

    def test_dividend_settlement_changes_cash_but_not_trade_count(self):
        before=self.account.get_account()['cash']
        self.settle(event())
        after=self.account.get_account({'300750':9})
        self.assertEqual(after['cash'],before+100)
        self.assertEqual(len(self.account.get_trades()),1)
        self.assertEqual(after['positions'][0]['shares'],100)
        self.assertEqual(after['positions'][0]['name'],'宁德时代')

    def test_unlisted_bonus_cannot_be_sold_or_ignored_in_nav(self):
        self.settle(event(cash_per_share='0',pay_date=None,bonus_per_share='1',listing_date='2026-10-26'),'2026-10-22')
        a=self.account.get_account({'300750':5})
        self.assertEqual(a['positions'][0]['shares'],100)
        self.assertEqual(a['corporate_rights'][0]['bonus_shares'],100)
        self.assertEqual(a['total_assets'],a['cash']+1000)
        self.account.authorize_run('ex','2026-10-22',['300750'])
        self.assertFalse(self.account.sell('300750',200,5,'fixture','ex','too-early','2026-10-22',1000000)['ok'])

    def test_bonus_odd_lot_can_only_be_sold_in_full_after_listing(self):
        self.settle(event(cash_per_share='0',pay_date=None,bonus_per_share='.2',listing_date='2026-10-26'))
        self.account.authorize_run('listed','2026-10-26',['300750'])
        self.assertFalse(self.account.sell('300750',20,9,'fixture','listed','partial-odd','2026-10-26',1000000)['ok'])
        self.assertTrue(self.account.sell('300750',120,9,'fixture','listed','full-odd','2026-10-26',1000000)['ok'])
        self.assertEqual(self.account.get_account()['positions'],[])

    def test_stale_coverage_stops_new_fills(self):
        self.settle(event(),'2026-10-22')
        self.account.authorize_run('stale','2026-10-23',['300750'])
        result=self.account.sell('300750',100,9,'fixture','stale','stale-sell','2026-10-23',1000000)
        self.assertFalse(result['ok']);self.assertIn('权益公告',result['error'])
        self.assertEqual(len(self.account.get_trades()),1)

    def test_late_observed_action_does_not_leak_into_historical_report(self):
        from evals.public_reports import ledger_as_of
        from evals.trading_view import read_ledger
        self.settle(event(known_at='2026-10-23T17:00:00+08:00'))
        snapshot=ledger_as_of(read_ledger(self.path),'2026-10-22')
        self.assertEqual(snapshot['corporate_actions'],[])
        self.assertEqual(snapshot['corporate_settlements'],[])
        self.assertEqual(snapshot['cash'],self.account.INITIAL_CASH-1005.5)

    def test_source_failure_does_not_change_cash_or_clear_prior_records(self):
        from agent.corporate_settlement import synchronize,coverage_ok
        from evals.trading_view import read_ledger
        self.settle(event())
        before=read_ledger(self.path);book=unittest.mock.MagicMock();book.collect.side_effect=OSError('not public')
        with patch('agent.corporate_settlement.calendar',return_value={'sessions':['2026-10-26']}):
            with self.assertRaises(OSError):synchronize(db=self.path,book=book,at=datetime.fromisoformat('2026-10-26T17:00:00+08:00'))
        after=read_ledger(self.path)
        self.assertEqual(before['cash'],after['cash'])
        self.assertEqual(before['corporate_actions'],after['corporate_actions'])
        self.assertFalse(coverage_ok(after,'2026-10-26'))

    def test_valuation_snapshot_explains_unlisted_rights(self):
        self.settle(event(cash_per_share='0',pay_date=None,bonus_per_share='1',listing_date='2026-10-26'),'2026-10-22')
        self.account.authorize_run('valuation','2026-10-22',['300750'])
        self.assertTrue(self.account.record_valuation_snapshot('valuation','2026-10-22',{'300750':5},['unit fixture'])['ok'])
        snapshot=self.account.get_valuation_snapshots()[0]
        self.assertEqual(snapshot['rights'][0]['bonus_shares'],100)
        self.assertEqual(snapshot['total_assets'],snapshot['cash']+1000)

    def test_odd_lot_cash_rounding_reconciles_with_journal_in_cents(self):
        from evals.trading_view import read_ledger
        self.settle(event(cash_per_share='0',pay_date=None,bonus_per_share='.2',listing_date='2026-10-26'))
        self.account.authorize_run('rounding','2026-10-26',['300750'])
        self.assertTrue(self.account.sell('300750',120,9.1234,'fixture','rounding','odd-rounding','2026-10-26',1000000)['ok'])
        l=read_ledger(self.path);rebuilt=replay(l,self.account.INITIAL_CASH,'2026-10-26')
        self.assertAlmostEqual(l['cash'],rebuilt['cash'],places=8)
        self.assertAlmostEqual(l['cash'],round(l['cash'],2),places=8)

if __name__=='__main__':unittest.main()
