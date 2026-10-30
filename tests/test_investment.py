import asyncio
from copy import deepcopy
from datetime import date, timedelta
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import pandas as pd

from agent.analysis import AnalysisHost
from agent.contracts import capture_risk_review, validate_decision_result
from agent.evidence import build_catalog, validate_evidence_audit
from agent.investment import (VERSION, PROMPT_VERSION, POLICY, ResearchBook, case_variant,
                              prepare_context, validate_cases)
from agent.subscription import subscription_workspace
from agent.subscription_roles import export_paper_context
from evals.experiments import canonical_hash
from evals.investment import evaluate_investments

DAY = "2026-10-28"


def material():
    return {"symbol": "300750", "title": "单元测试公司披露片段", "url": "https://static.cninfo.com.cn/finalpage/fixture.pdf",
        "published_at": "2026-08-31T10:00:00+08:00", "categories": ["business", "financials"],
        "report_period": "2026-06-30", "excerpt": "测试替身：业务、现金流、负债及历史经营数据。并非真实财报。", "locator": "第1页第2段",
        "data_source": "unit fixture disclosure", "data_quality": "imported"}


def example(ready=True):
    account = {"market": "cn", "currency": "CNY", "initial_cash": 1000000., "cash": 1000000., "total_assets": 1000000., "positions": []}
    context = {"date": DAY, "symbols": ["300750"], "snapshot": [{"symbol": "300750", "name": "test", "date": DAY,
        "close": 100., "volume": 1000000, "data_source": "unit test fixture"}], "fundamentals": [{"symbol": "300750", "data_source": "unavailable(unit fixture)"}], "news": {"candidates": []},
        "account": account, "memory": {}, "investment_context": {"policy": POLICY, "materials": [material()] if ready else [], "price_history": [], "previous_cases": [], "tasks": []}}
    projected = export_paper_context(context, "cn", True)
    catalog = build_catalog(projected, "cn")
    price_ids = [row["evidence_id"] for row in catalog["records"] if row["kind"] == "market_snapshot"]
    doc_ids = [row["evidence_id"] for row in catalog["records"] if row["kind"] == "company_document"]
    def links(refs):
        return {"status": "supported" if refs else "insufficient", "supporting_ids": refs, "contradicting_ids": [], "invalidation_condition": "核实新资料后复核"}
    annotation = links(price_ids + doc_ids) if ready else links([])
    decisions = [{"symbol": "300750", "name": "test", "action": "BUY" if ready else "HOLD", "shares": 1000 if ready else 0,
                  "price": 100., "reason": "unit fixture", "evidence": annotation}]
    case = {"symbol": "300750", "research_status": "ready" if ready else "pending", "position_intent": "build" if ready else "research_pending",
        "thesis": "unit thesis", "thesis_horizon_years": 3, "thesis_change": "new", "target_weight": .2 if ready else 0.,
        "valuation": {"method": "fixture cash flow model" if ready else "未完成估值", "bear_price": 80. if ready else None,
            "base_price": 120. if ready else None, "bull_price": 160. if ready else None, "assumptions": "明确的测试假设，不代表真实估值"},
        "entry_condition": "核对资料并达到估值条件", "exit_condition": "经营论证被否定或风险超限",
        "next_review_date": (date.fromisoformat(DAY) + timedelta(days=90 if ready else 7)).isoformat(),
        "core_gaps": [] if ready else [{"category": "financials", "reason": "缺原文", "what_to_fetch": "取得有披露时间的现金流与负债原文片段"}],
        "non_core_gaps": ["无当日新闻"], "evidence": annotation}
    review = {"summary": "fixture", "assessments": [{"symbol": "300750", "action": decisions[0]["action"], "verdict": "APPROVE",
        "max_shares": decisions[0]["shares"], "reason": "fixture", "evidence": annotation}]}
    log = {"market": "cn", "date": DAY, "as_of_date": DAY, "investment_policy": VERSION, "summary": "fixture", "execution_mode": "shadow",
        "execution": {"orders_submitted": 0}, "analyst_views": {"technical": "t", "fundamental": "f", "risk": "r"}, "memory_assessment": "none",
        "decisions": decisions, "investment_cases": [case], "predictions": [{"symbol": "300750", "direction": "up", "horizon_days": 252,
            "confidence": .55, "rationale": "一年阶段观察，不是每日买卖开关", "evidence": links(price_ids)}],
        "risk_review": capture_risk_review(decisions, review, "cn", {"300750"}),
        "evidence_audit": {"catalog": catalog, "analyst_claims": {
            "technical-analyst": [{"symbol": "300750", "claim": "价格观察", "evidence": links(price_ids)}],
            "fundamental-analyst": [{"symbol": "300750", "claim": "业务与财务来源" if ready else "待补资料", "evidence": links(doc_ids)}]}}}
    log["input_dataset"] = {"market": "cn", "as_of_date": DAY, "active_symbols": ["300750"],
        "snapshot": context["snapshot"], "fundamentals": context["fundamentals"], "account": account,
        "news_screen": context["news"], "investment_context": context["investment_context"]}
    log["dataset_fingerprint"] = canonical_hash(log["input_dataset"])
    return context, log


class InvestmentTests(unittest.TestCase):
    def test_autonomous_context_keeps_latest_financial_units_and_bounded_business_original(self):
        context, _ = example()
        context['execution_policy'] = 'autonomous-paper-execution-v1'
        annual = {**material(), 'categories':['financials'], 'published_at':'2026-03-01', 'document_type':'annual'}
        interim = {**material(), 'categories':['financials'], 'published_at':'2026-08-31', 'document_type':'interim', 'excerpt':'单位：万元\n'+ '现金流 财务数据\n'*80, 'unit_context_present':True}
        business = {**material(), 'categories':['business'], 'excerpt':'原始业务内容'*200}
        context['investment_context']['materials'] = [annual, interim, business]
        projected = export_paper_context(context, 'cn', True)
        self.assertEqual(len(projected['research_materials']), 2)
        financial = next(r for r in projected['research_materials'] if r['categories']==['financials'])
        self.assertEqual(financial['excerpt'], interim['excerpt'])
        self.assertIn('万元', financial['excerpt'])
        shown_business = next(r for r in projected['research_materials'] if r['categories']==['business'])
        self.assertEqual(shown_business['excerpt'], business['excerpt'][:400])
        self.assertTrue(shown_business['excerpt_truncated'])
        self.assertEqual(len(context['investment_context']['materials']), 3)

    def test_compact_projection_keeps_business_material_and_marks_truncation(self):
        context, _ = example()
        financial = {**material(), "categories": ["financials"], "excerpt": "财务" * 1000}
        business = {**material(), "title": "业务原文", "categories": ["business"]}
        context["investment_context"]["materials"] = [{**financial, "title": str(index)} for index in range(8)] + [business]
        projected = export_paper_context(context, "cn", True)
        self.assertTrue(any("business" in row["categories"] for row in projected["research_materials"]))
        self.assertTrue(any(row["excerpt_truncated"] for row in projected["research_materials"]))

    def test_complete_research_can_buy_without_news(self):
        context, log = example()
        validate_decision_result(log, "cn", expected_symbols={"300750"})
        validate_evidence_audit(log)
        validate_cases(log, {"300750"}, catalog=log["evidence_audit"]["catalog"], as_of=DAY,
                       account=context["account"], prices={"300750": 100.})
        self.assertEqual(log["decisions"][0]["action"], "BUY")
        self.assertEqual(context["news"]["candidates"], [])

    def test_pending_needs_concrete_task_and_short_review_deadline(self):
        _, log = example(False)
        validate_evidence_audit(log)
        log["investment_cases"][0]["core_gaps"][0]["what_to_fetch"] = ""
        with self.assertRaises(ValueError): validate_evidence_audit(log)
        _, log = example(False)
        log["investment_cases"][0]["next_review_date"] = "2027-01-01"
        with self.assertRaisesRegex(ValueError, "7天"):
            validate_evidence_audit(log)

    def test_buy_cannot_use_only_price_instead_of_company_research(self):
        _, log = example()
        row = log["investment_cases"][0]
        row["evidence"]["supporting_ids"] = [item["evidence_id"] for item in log["evidence_audit"]["catalog"]["records"] if item["kind"] == "market_snapshot"]
        with self.assertRaisesRegex(ValueError, "原始披露"):
            validate_evidence_audit(log)

    def test_valuation_range_target_weight_and_price_are_validated(self):
        context, log = example()
        changed = deepcopy(log)
        changed["investment_cases"][0]["valuation"]["base_price"] = 200.
        with self.assertRaises(ValueError): validate_evidence_audit(changed)
        changed = deepcopy(log)
        changed["investment_cases"][0]["target_weight"] = .31
        with self.assertRaises(ValueError): validate_evidence_audit(changed)
        changed = deepcopy(log)
        changed["investment_cases"][0]["target_weight"] = .01
        with self.assertRaisesRegex(ValueError, "目标仓位"):
            validate_cases(changed, {"300750"}, account=context["account"])
        with self.assertRaisesRegex(ValueError, "冻结行情"):
            validate_cases(log, {"300750"}, prices={"300750": 200.})

    def test_long_and_short_prediction_contracts_are_separate(self):
        _, log = example()
        log["predictions"][0]["horizon_days"] = 5
        with self.assertRaisesRegex(ValueError, "期限"):
            validate_decision_result(log, "cn", expected_symbols={"300750"})

    def test_risk_rejection_keeps_proposal_distinct_from_executed_action(self):
        _, log = example()
        log["risk_review"]["assessments"][0].update(verdict="REJECT", max_shares=0)
        log["decisions"][0] = {**log["decisions"][0], "action": "HOLD", "shares": 0}
        validate_evidence_audit(log)

    def test_materials_are_reused_without_new_news_and_have_expiry(self):
        with subscription_workspace() as root:
            book = ResearchBook(root / "research.db")
            first = book.add_material(material())
            self.assertEqual(first["material_id"], book.add_material(material())["material_id"])
            self.assertEqual(len(book.snapshot(["300750"], DAY)["materials"]), 1)
            self.assertEqual(len(book.snapshot(["300750"], "2026-11-28")["materials"]), 1)
            self.assertEqual(book.snapshot(["300750"], "2028-01-01")["materials"], [])
            with self.assertRaises(ValueError): book.add_material({**material(), "url": "https://example.org/fake.pdf"})
            with self.assertRaises(ValueError): book.add_material({**material(), "published_at": "2099-01-01T00:00:00+08:00"})

    def test_overdue_source_task_does_not_reset_every_day(self):
        with subscription_workspace() as root:
            book = ResearchBook(root / "research.db")
            first = book.snapshot(["300750"], DAY)["tasks"]
            later = book.snapshot(["300750"], "2026-11-28")["tasks"]
            self.assertEqual(first[0]["review_by"], later[0]["review_by"])
            self.assertEqual(later[0]["status"], "overdue")

    def test_first_case_is_immutable_and_previous_case_is_only_opinion(self):
        _, log = example(False)
        with subscription_workspace() as root:
            book = ResearchBook(root / "research.db")
            book.save(log, "variant")
            book.save(log, "variant")
            changed = deepcopy(log)
            changed["summary"] = "rerun"
            with self.assertRaisesRegex(ValueError, "首次"):
                book.save(changed, "variant")
            next_day = book.snapshot(["300750"], "2026-10-29", "variant")
            self.assertEqual(next_day["previous_cases"][0]["kind"], "past_model_opinion_not_external_fact")
            self.assertEqual(book.snapshot(["300750"], DAY, "variant")["previous_cases"], [])
            self.assertTrue(any(row["task_id"].startswith("gap_") for row in next_day["tasks"]))

    def test_news_is_optional_but_real_quote_is_still_required(self):
        from agent.run_daily import _assert_live_data
        snapshot = [{"symbol": "300750", "date": date.today().isoformat(), "volume": 10000, "close": 100., "data_source": "real fixture"}]
        with patch("agent.run_daily._offline_enabled", return_value=False):
            _assert_live_data(snapshot, {"ok": False}, require_news=False)
            with self.assertRaises(RuntimeError): _assert_live_data(snapshot, {"ok": False})
            with self.assertRaises(RuntimeError): _assert_live_data([], {"ok": False}, require_news=False)

    def test_scoped_news_scan_does_not_select_other_stocks(self):
        from servers import news_server
        with patch.object(news_server, "get_symbol_news", return_value=[]) as fetch:
            result = news_server.screen_news_candidates(DAY, symbols=["300750"])
        self.assertEqual([row["symbol"] for row in result["candidates"]], ["300750"])
        self.assertEqual(fetch.call_count, 1)

    def test_long_stage_evaluation_does_not_treat_cash_zero_as_success(self):
        _, log = example(False)
        prices = {"300750": [{"date": DAY, "close": 100.}, {"date": "2026-10-29", "close": 110.}]}
        result = evaluate_investments([log], prices, {"portfolio": {"cash": 1000., "total_assets": 1000., "net_pnl": 0}})
        self.assertEqual(result["status"], "awaiting_outcomes")
        self.assertEqual(result["rows"][0]["cash_vs_stock_opportunity_gap_pct"], -10.)
        self.assertFalse(result["strategy_effect_verified"])

    def test_cash_opportunity_gap_is_not_assigned_to_existing_holdings(self):
        _, log = example(False)
        log["input_dataset"]["account"]["positions"] = [{"symbol": "300750", "shares": 100}]
        result = evaluate_investments([log], {"300750": [{"date": DAY, "close": 100.}, {"date": "2026-10-29", "close": 110.}]})
        self.assertIsNone(result["rows"][0]["cash_vs_stock_opportunity_gap_pct"])

    def test_252_day_prediction_waits_for_complete_horizon(self):
        _, log = example(False)
        days = pd.bdate_range("2025-01-01", periods=253).strftime("%Y-%m-%d").tolist()
        log["as_of_date"] = days[0]
        rows = [{"date": day, "close": 100 + index / 10} for index, day in enumerate(days)]
        self.assertEqual(evaluate_investments([log], {"300750": rows[:-1]})["matured_count"], 0)
        self.assertEqual(evaluate_investments([log], {"300750": rows})["matured_count"], 1)

    def test_long_thesis_and_valuation_enter_semantic_review(self):
        from evals.semantic_review import prepare_review
        _, log = example()
        report = prepare_review(log)
        self.assertTrue(any(row["path"].startswith("investment_cases") for row in report["items"]))
        self.assertTrue(any(row["path"].startswith("investment_valuations") for row in report["items"]))


class LongGraphTests(unittest.IsolatedAsyncioTestCase):
    async def test_four_roles_use_long_schema_materials_and_history_without_live_model(self):
        from agent.graph_runner import run_agent_graph
        context, fixture_log = example()
        calls = []
        class Host:
            market = "cn"
            active_symbols = ["300750"]
            news_screen = {"offline": False, "as_of_date": DAY, "candidates": []}
        reader = AnalysisHost(Host(), context["snapshot"], context["fundamentals"], context["account"], [])
        class Client:
            def __init__(self, model): self.calls = []
            async def complete(self, prompt, schema, role):
                calls.append((role, prompt, schema))
                self.calls.append({"role": role, "input_tokens": 1, "output_tokens": 1})
                if role in {"technical-analyst", "fundamental-analyst"}:
                    return {"summary": "test role", "claims": deepcopy(fixture_log["evidence_audit"]["analyst_claims"][role])}
                if role == "investment-manager":
                    return {key: deepcopy(value) for key, value in fixture_log.items() if key in schema["properties"]}
                return {"summary": "test review", "assessments": deepcopy(fixture_log["risk_review"]["assessments"])}
            def report(self):
                return {"requests_started": len(self.calls), "input_tokens": len(self.calls), "output_tokens": len(self.calls),
                    "total_tokens": len(self.calls) * 2, "reasoning_effort": "low", "max_output_tokens": None}
        with subscription_workspace() as root:
            with patch("agent.investment.prepare_context", return_value=context["investment_context"]), patch("agent.subscription_roles.SubscriptionClient", Client):
                result = await run_agent_graph(reader, "cn", DAY, {}, PROMPT_VERSION, model_access="subscription", checkpoint_path=root / "graph.db")
        self.assertEqual(len(calls), 4)
        self.assertEqual(result["predictions"][0]["horizon_days"], 252)
        self.assertEqual(result["decisions"][0]["action"], "BUY")
        self.assertEqual(result["graph"]["status"], "analysis_complete")
        self.assertIn("investment_context", result["input_dataset"])
        self.assertEqual(result["agent_sdk"]["prompt"]["version"], PROMPT_VERSION)

    async def test_frozen_history_is_visible_to_long_roles_and_future_rows_are_removed(self):
        class Host:
            async def call(self, name, args):
                return [{"date": day, "open": price, "close": price, "volume": 1000}
                        for day, price in [(DAY, 100.), ("2026-10-29", 999.)]]
        with subscription_workspace() as root, patch("agent.investment.DATA_DIR", Path("nonexistent-unit-cache")):
            prepared = await prepare_context(Host(), DAY, ["300750"], book=ResearchBook(root / "research.db"))
        self.assertEqual(prepared["price_history"][0]["recent_daily"][-1]["close"], 100.)
        self.assertTrue(prepared["auxiliary_gaps"])
