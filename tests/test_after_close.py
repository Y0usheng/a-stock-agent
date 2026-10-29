import asyncio
from copy import deepcopy
from datetime import date
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from agent.disclosures import select_passages, select_reports, download_pdf
from agent.evidence import build_catalog
from agent.order_queue import OrderQueue, execute_due
from agent.subscription import subscription_workspace
from agent.trading_clock import make_timing
from evals.experiments import canonical_hash
from servers.data_quality import event_time
from tests.test_investment import example

DAY, NEXT = "2026-10-28", "2026-10-29"


def clock(value):
    return event_time(value, "cn")


def timing():
    return make_timing(DAY, [DAY, NEXT, "2026-10-30"], DAY + "T17:30:00+08:00")


def quote(symbol="300750", price=100.):
    row = {"symbol": symbol, "date": NEXT, "open": 100., "close": price, "high": price + 1, "low": price - 1,
           "volume": 1000000, "price_basis": "unadjusted", "data_quality": "live_prices", "source": "unit-fixture"}
    row["fingerprint"] = canonical_hash(row)
    row["observed_at"] = NEXT + "T17:30:00+08:00"
    return row


class FixedDate(date):
    @classmethod
    def today(cls):
        return cls(2026, 10, 30)


class AccountHost:
    def __init__(self, quote_row=None, interrupt=False):
        self.quote = quote_row or quote()
        self.interrupt = interrupt
        self.buy_calls = 0

    async def call(self, name, args):
        from servers import account_server as account
        if name == "execution_session_quote":
            return deepcopy(self.quote)
        if name == "buy":
            self.buy_calls += 1
            result = account.buy(**args)
            if self.interrupt:
                self.interrupt = False
                raise RuntimeError("simulated interruption after durable fill")
            return result
        return getattr(account, name)(**args)


class AfterCloseTests(unittest.TestCase):
    def setup_queue(self, root):
        context, result = example(True)
        queue = OrderQueue(root / "queue.db")
        with patch("agent.order_queue.now", return_value=clock(DAY + "T17:40:00+08:00")):
            identifier = queue.stage(result, timing(), context["account"])
        return queue, identifier

    def run_due(self, queue, host):
        with patch("agent.order_queue.now", return_value=clock(NEXT + "T17:30:00+08:00")):
            return asyncio.run(execute_due(host, queue, NEXT + "T17:30:00+08:00"))

    def test_holiday_and_latest_quote_use_real_session_list(self):
        result = make_timing("2026-09-30", ["2026-09-30", "2026-10-09"], "2026-09-30T17:30:00+08:00")
        self.assertEqual(result["execution_session"], "2026-10-09")
        with self.assertRaises(ValueError):
            make_timing(DAY, [DAY, NEXT, "2026-10-30"], NEXT + "T17:30:00+08:00")
        with self.assertRaises(ValueError):
            make_timing(DAY, [DAY], DAY + "T17:30:00+08:00")

    def test_no_order_is_sent_when_confirmed_before_target_session_close(self):
        with subscription_workspace() as root:
            queue, identifier = self.setup_queue(root)
            queue.confirm(identifier, at=DAY + "T18:00:00+08:00")
            host = AccountHost()
            self.assertEqual(asyncio.run(execute_due(host, queue, NEXT + "T14:30:00+08:00")), [])
            self.assertEqual(host.buy_calls, 0)

    def test_late_confirmation_expires_instead_of_backdating_trade(self):
        with subscription_workspace() as root:
            queue, identifier = self.setup_queue(root)
            with self.assertRaises(ValueError):
                queue.confirm(identifier, at=NEXT + "T15:01:00+08:00")
            self.assertEqual(queue.rows()[0]["status"], "expired")

    def test_late_analysis_or_mismatched_quote_day_cannot_create_plan(self):
        context, result = example(True)
        with subscription_workspace() as root:
            queue = OrderQueue(root / "queue.db")
            with patch("agent.order_queue.now", return_value=clock(NEXT + "T15:01:00+08:00")), self.assertRaises(ValueError):
                queue.stage(result, timing(), context["account"])
            bad = timing();bad["price_as_of_date"] = "2026-10-27"
            with self.assertRaises(ValueError):
                queue.stage(result, bad, context["account"])

    def test_unconfirmed_order_expires_without_fill(self):
        with subscription_workspace() as root:
            queue, _ = self.setup_queue(root)
            receipt = self.run_due(queue, AccountHost())
            self.assertEqual(receipt[0]["status"], "expired")
            self.assertEqual(receipt[0]["fills"], 0)

    def test_confirmed_plan_can_be_cancelled_without_new_model(self):
        with subscription_workspace() as root:
            queue, identifier = self.setup_queue(root)
            queue.confirm(identifier, at=DAY + "T18:00:00+08:00")
            queue.confirm(identifier, approve=False, at=DAY + "T18:05:00+08:00")
            self.assertEqual(self.run_due(queue, AccountHost()), [])

    def test_post_close_cancellation_cannot_choose_whether_confirmed_plan_fills(self):
        with subscription_workspace() as root:
            queue, identifier = self.setup_queue(root)
            queue.confirm(identifier, at=DAY + "T18:00:00+08:00")
            with self.assertRaises(ValueError):
                queue.confirm(identifier, approve=False, at=NEXT + "T15:01:00+08:00")
            self.assertEqual(queue.rows()[0]["status"], "confirmed")

    def test_future_fill_uses_observed_next_session_price_and_native_fees(self):
        from servers import account_server as account
        with subscription_workspace() as root, patch.object(account, "DB_PATH", None), patch.object(account, "date", FixedDate):
            account.DB_PATH = root / "account.db"
            queue, identifier = self.setup_queue(root)
            queue.confirm(identifier, at=DAY + "T18:00:00+08:00")
            receipt = self.run_due(queue, AccountHost())
            self.assertEqual(receipt[0]["status"], "filled")
            fills = account.get_trades()
            self.assertEqual(len(fills), 1)
            self.assertEqual(fills[0]["trade_date"], NEXT)
            self.assertEqual(fills[0]["price"], 100.05)
            self.assertGreater(fills[0]["fee"], 0)
            self.assertEqual(queue.public_state(DAY)[0]["status"], "confirmed")
            self.assertEqual(queue.public_state(NEXT)[0]["status"], "filled")
            self.assertNotIn(identifier, json.dumps(queue.public_state(NEXT)))

    def test_interruption_after_fill_recovers_without_duplicate_purchase(self):
        from servers import account_server as account
        with subscription_workspace() as root, patch.object(account, "DB_PATH", None), patch.object(account, "date", FixedDate):
            account.DB_PATH = root / "account.db"
            queue, identifier = self.setup_queue(root)
            queue.confirm(identifier, at=DAY + "T18:00:00+08:00")
            host = AccountHost(interrupt=True)
            self.assertEqual(self.run_due(queue, host)[0]["status"], "execution_pending")
            self.assertEqual(len(account.get_trades()), 1)
            self.assertEqual(self.run_due(queue, host)[0]["status"], "filled")
            self.assertEqual(len(account.get_trades()), 1)
            self.assertEqual(host.buy_calls, 1)

    def test_price_gap_cancels_instead_of_filling_at_old_close(self):
        from servers import account_server as account
        with subscription_workspace() as root, patch.object(account, "DB_PATH", None), patch.object(account, "date", FixedDate):
            account.DB_PATH = root / "account.db"
            queue, identifier = self.setup_queue(root)
            queue.confirm(identifier, at=DAY + "T18:00:00+08:00")
            self.assertEqual(self.run_due(queue, AccountHost(quote(price=110.)))[0]["status"], "cancelled")
            self.assertEqual(account.get_trades(), [])

    def test_external_account_change_blocks_old_sizing(self):
        from servers import account_server as account
        with subscription_workspace() as root, patch.object(account, "DB_PATH", None), patch.object(account, "date", FixedDate):
            account.DB_PATH = root / "account.db"
            queue, identifier = self.setup_queue(root)
            queue.confirm(identifier, at=DAY + "T18:00:00+08:00")
            conn = account._db()
            conn.execute("UPDATE meta SET value=999000 WHERE key='cash'");conn.commit();conn.close()
            self.assertEqual(self.run_due(queue, AccountHost())[0]["status"], "blocked")
            self.assertEqual(account.get_trades(), [])

    def test_modified_queue_payload_or_adjusted_quote_is_rejected(self):
        from servers import account_server as account
        with subscription_workspace() as root, patch.object(account, "DB_PATH", None), patch.object(account, "date", FixedDate):
            account.DB_PATH = root / "account.db"
            queue, identifier = self.setup_queue(root)
            queue.confirm(identifier, at=DAY + "T18:00:00+08:00")
            bad = quote();bad["price_basis"] = "qfq"
            self.assertEqual(self.run_due(queue, AccountHost(bad))[0]["status"], "execution_pending")
            self.assertEqual(account.get_trades(), [])
            with queue.db() as conn:
                conn.execute("UPDATE plans SET payload='{}'")
            with self.assertRaises(ValueError):
                queue.rows()

    def test_after_close_news_is_eligible_only_if_seen_by_actual_cutoff(self):
        context, _ = example(False)
        from agent.subscription_roles import export_paper_context
        context["news"] = {"candidates": [{"symbol": "300750", "top_events": [{"symbol": "300750", "event_id": "fixture", "title": "fixture", "summary": "fixture", "source": "media", "url": "https://finance.eastmoney.com/a/fixture.html", "published_at": DAY + "T16:10:00+08:00", "first_seen_at": DAY + "T17:00:00+08:00"}]}]}
        legacy = build_catalog(export_paper_context(context, "cn", True), "cn")
        self.assertFalse(next(row for row in legacy["records"] if row["kind"] == "news")["eligible"])
        context["timing"] = timing()
        current = build_catalog(export_paper_context(context, "cn", True), "cn")
        self.assertTrue(next(row for row in current["records"] if row["kind"] == "news")["eligible"])
        context["news"]["candidates"][0]["top_events"][0]["first_seen_at"] = DAY + "T18:00:00+08:00"
        current = build_catalog(export_paper_context(context, "cn", True), "cn")
        self.assertFalse(next(row for row in current["records"] if row["kind"] == "news")["eligible"])

    def test_reference_state_is_derived_without_rewriting_claims_or_ids(self):
        import jsonschema
        from agent.evidence import derived_link_schema, restore_link_status
        from agent.investment import result_schema
        from agent.subscription_roles import RESULT_EVIDENCE_SCHEMA
        _, result = example(True)
        schema = derived_link_schema(result_schema(RESULT_EVIDENCE_SCHEMA))
        case_schema = schema["properties"]["investment_cases"]["items"]
        case = deepcopy(result["investment_cases"][0])
        with self.assertRaises(jsonschema.ValidationError):
            jsonschema.validate(case, case_schema)
        expected = deepcopy(case)
        case["evidence"].pop("status")
        jsonschema.validate(case, case_schema)
        self.assertEqual(restore_link_status({"cases": [case]})["cases"][0], expected)
        case["evidence"]["supporting_ids"] = []
        jsonschema.validate(case, case_schema)
        self.assertEqual(restore_link_status({"cases": [case]})["cases"][0]["evidence"]["status"], "insufficient")

    def test_report_selection_excludes_summary_foreign_company_and_path_injection(self):
        rows = [{"secCode": "300308", "announcementTitle": "2025年年度报告" + suffix,
                 "announcementTime": 1774886400000, "adjunctUrl": path, "announcementId": "../evil"} for suffix, path in
                [("", "finalpage/2026-03-31/1225056459.PDF"), ("摘要", "finalpage/2026-03-31/1225056458.PDF"), ("", "../private.pdf")]]
        selected = select_reports(rows, "300308", DAY + "T17:30:00+08:00")
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["announcement_id"], "1225056459")

    def test_missing_first_observation_and_financial_units_block_new_evidence(self):
        from agent.subscription_roles import export_paper_context
        context, _ = example(True)
        context["timing"] = timing()
        materials = context["investment_context"]["materials"]
        for row in materials:
            row.pop("first_seen_at", None)
        catalog = build_catalog(export_paper_context(context, "cn", True), "cn")
        docs = [row for row in catalog["records"] if row["kind"] == "company_document"]
        self.assertTrue(docs)
        self.assertTrue(all(not row["eligible"] for row in docs))
        for row in materials:
            row["first_seen_at"] = DAY + "T16:00:00+08:00"
            row["unit_context_present"] = False
        catalog = build_catalog(export_paper_context(context, "cn", True), "cn")
        financial = [row for row in catalog["records"] if "financials" in row["payload"].get("categories", [])]
        self.assertTrue(financial)
        self.assertTrue(all("financial_units_not_visible" in row["issues"] for row in financial))

    def test_financial_continuation_keeps_exact_unit_page_and_source_spans(self):
        document = {"symbol": "300308", "document_type": "interim"}
        texts = ["公司代码300308" + "基础信息" * 100, "主要会计数据\n单位：千元 币种：人民币",
                 "主要会计数据 本报告期 上年同期\n营业收入 123 100\n经营活动产生的现金流量净额 20 10\n" + "表格说明文字\n" * 30 + "九、另一张表\n单位：万元\n"]
        extracted = {"pages": [{"pdf_page": i+1, "text": text, "text_hash": canonical_hash(text)} for i, text in enumerate(texts)]}
        material = select_passages(document, extracted)[0]
        self.assertTrue(material["unit_context_present"])
        self.assertIn("千元", material["excerpt"])
        self.assertEqual([span["pdf_page"] for span in material["source_spans"]], [2, 3])
        self.assertFalse(material["source_truth_verified"])

    def test_unrelated_previous_page_units_are_not_borrowed(self):
        document = {"symbol": "300308", "document_type": "interim"}
        texts = ["公司代码300308\n其他表格\n单位：万元", "主要会计数据 本报告期 上年同期\n营业收入 123 100\n" + "表格说明\n" * 30]
        extracted = {"pages": [{"pdf_page": i+1, "text": text, "text_hash": canonical_hash(text)} for i, text in enumerate(texts)]}
        self.assertFalse(select_passages(document, extracted)[0]["unit_context_present"])

    def test_business_section_precedes_keyword_rich_subsidiary_table(self):
        document = {"symbol": "300308", "document_type": "annual"}
        texts = ["公司代码300308" + "基本信息" * 30,
                 "二、报告期内公司从事的业务情况\n公司主要业务及产品说明\n" + "这是集团业务说明\n" * 30,
                 "子公司表\n主要业务 主营业务 经营模式 核心竞争力 主要风险\n" + "子公司表格\n" * 40]
        extracted = {"pages": [{"pdf_page": i+1, "text": text, "text_hash": canonical_hash(text)} for i, text in enumerate(texts)]}
        business = next(row for row in select_passages(document, extracted) if row["categories"] == ["business"])
        self.assertEqual(business["pdf_page"], 2)

    def test_nearly_exhausted_budget_stops_before_launching_risk_call(self):
        from agent.subscription import SubscriptionClient
        client = SubscriptionClient("fixture")
        client.requests_started = 3
        client.calls = [{"role": "fixture", "input_tokens": 13000, "output_tokens": 1, "cached_input_tokens": 0, "reasoning_output_tokens": 0} for _ in range(3)]
        with patch("agent.subscription.check_subscription_login") as login:
            with self.assertRaises(RuntimeError):
                asyncio.run(client.complete("fixture", {}, role="risk-manager"))
            login.assert_not_called()
        self.assertEqual(client.requests_started, 3)

    def test_technical_facts_use_real_snapshot_without_model_or_trade(self):
        from agent.graph_runner import manager_instructions
        from agent.subscription_roles import SubscriptionRoles
        context, _ = example(True)
        context["timing"] = timing()
        roles = SubscriptionRoles(context, "cn", manager_instructions("cn", DAY, ["300750"], "manager-long-v2"), "fixture", long_term=True)
        with patch.object(roles.client, "complete") as complete:
            result = json.loads(asyncio.run(roles.call("technical-analyst", "fixture")))
            complete.assert_not_called()
        self.assertIn(str(context["snapshot"][0]["close"]), result["claims"][0]["claim"])
        self.assertEqual(roles.client.requests_started, 0)
        self.assertEqual(roles.audit()["role_sources"]["technical-analyst"], "host_quote_indicator_facts_v1")


if __name__ == "__main__":
    unittest.main()
