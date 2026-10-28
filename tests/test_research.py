from copy import deepcopy
import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
import unittest
from unittest.mock import AsyncMock, patch

from agent.contracts import capture_risk_review
from agent.evidence import build_catalog
from agent.subscription import subscription_workspace
from evals.experiments import canonical_hash, comparable_experiments, decision_log_variant, list_experiments
from evals.research import ResearchBook, PHASES, render_review, review_results, validate_plan
from servers.synthetic_data import random_walk_history


def fixture():
    symbols = ["600036", "600519"]
    histories = {symbol: random_walk_history(symbol, 110, {}, .0003, .015, 1000000, 3000000,
        .008, "2026-09-30").to_dict("records") for symbol in symbols}
    days = [row["date"] for row in histories[symbols[0]]]
    events = [{"event_id": day + "-" + symbol, "symbol": symbol, "title": "工程样本事件",
        "source": "synthetic fixture", "summary": "不代表真实业绩", "published_at": day + "T10:00:00+08:00",
        "first_seen_at": day + "T10:01:00+08:00", "url": "https://example.org/fixture", "data_quality": "synthetic",
        "sentiment_score": -2 if index % 2 else 1} for index, day in enumerate(days[28::3]) for symbol in symbols]
    bundle = {"market": "cn", "histories": histories, "news_events": events,
              "price_quality": "synthetic_prices", "price_basis": "synthetic"}
    spec = {"name": "测试研究", "hypothesis": "新闻门控能否改善技术基线？", "symbols": symbols, "mode": "engineering",
        "splits": {"train": [days[30], days[49]], "validation": [days[50], days[69]], "test": [days[70], days[-1]]},
        "baseline": "technical", "candidate": "news", "variants": [{"id": "buy", "kind": "buy_hold"},
            {"id": "technical", "kind": "technical"}, {"id": "news", "kind": "news_gate"}],
        "criteria": {"min_matured_predictions": 10, "min_nonoverlapping_dates": 2}}
    return spec, bundle, days


def model_log(bundle, day):
    symbols = sorted(bundle["histories"])
    snapshot = [{**next(row for row in bundle["histories"][symbol] if row["date"] == day),
                 "symbol": symbol, "name": symbol, "data_source": "synthetic fixture"} for symbol in symbols]
    context = {"date": day, "symbols": symbols, "snapshot": snapshot, "fundamentals": [], "news": []}
    catalog = build_catalog(context, "cn", allow_synthetic=True)
    refs = {row["symbol"]: row["evidence_id"] for row in catalog["records"]}
    def links(symbol):
        return {"status": "supported", "supporting_ids": [refs[symbol]], "contradicting_ids": [], "invalidation_condition": "新行情反转"}
    decisions = [{"symbol": row["symbol"], "name": row["name"], "action": "HOLD", "shares": 0,
                  "price": row["close"], "reason": "研究替身", "evidence": links(row["symbol"])} for row in snapshot]
    review = {"summary": "test", "assessments": [{"symbol": row["symbol"], "action": "HOLD", "verdict": "APPROVE",
        "max_shares": 0, "reason": "test", "evidence": links(row["symbol"])} for row in snapshot]}
    gap = {"status": "insufficient", "supporting_ids": [], "contradicting_ids": [], "invalidation_condition": "取得财务数据"}
    inputs = {"market": "cn", "as_of_date": day, "active_symbols": symbols, "snapshot": snapshot,
              "news_screen": {"candidates": []}, "tool_data": [], "account": {"cash": 1000000}}
    return {"market": "cn", "date": day, "as_of_date": day, "summary": "synthetic test model output",
        "generated_at": datetime.now(timezone.utc).isoformat(), "mode": "langgraph", "strategy_version": "synthetic-test-model",
        "execution_mode": "shadow", "execution": {"orders_submitted": 0}, "offline": True,
        "analyst_views": {"technical": "test", "fundamental": "test", "risk": "test"}, "memory_assessment": "none",
        "decisions": decisions, "predictions": [{"symbol": symbol, "direction": "up", "confidence": .6, "horizon_days": 5,
            "rationale": "合成模型输出，仅工程验收", "evidence": links(symbol)} for symbol in symbols],
        "risk_review": capture_risk_review(decisions, review, "cn", set(symbols)),
        "input_dataset": inputs, "dataset_fingerprint": canonical_hash(inputs),
        "evidence_audit": {"catalog": catalog, "analyst_claims": {
            "technical-analyst": [{"symbol": symbol, "claim": "test", "evidence": links(symbol)} for symbol in symbols],
            "fundamental-analyst": [{"symbol": symbol, "claim": "缺少财务", "evidence": deepcopy(gap)} for symbol in symbols]}},
        "agent_sdk": {"framework": "langgraph", "model_access": "subscription", "prompt": {"version": "test", "template_sha256": "fixture"},
            "budget": {"reasoning_effort": "low", "max_output_tokens": None}, "role_models": {"manager": {"model": "synthetic-test-only"}},
            "usage": {"requests": 4, "total_tokens": 40}, "duration_ms": 10}}


class ResearchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec, cls.bundle, cls.days = fixture()

    def setup_book(self, root, spec=None, bundle=None):
        book = ResearchBook(root / "research.db")
        study = book.create(deepcopy(spec or self.spec))
        book.freeze(study["id"], deepcopy(bundle or self.bundle))
        return book, study

    def test_plan_rejects_us_overlap_bad_symbols_and_unsafe_variant_ids(self):
        changes = [{"market": "us"}, {"symbols": ["AAPL"]}, {"baseline": "missing"}]
        for change in changes:
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_plan({**deepcopy(self.spec), **change})
        value = deepcopy(self.spec)
        value["splits"]["validation"][0] = value["splits"]["train"][1]
        with self.assertRaises(ValueError): validate_plan(value)
        value = deepcopy(self.spec)
        value["variants"][0]["id"] = "../escape"
        with self.assertRaises(ValueError): validate_plan(value)

    def test_prospective_cannot_be_registered_after_test_starts(self):
        with self.assertRaisesRegex(ValueError, "倒填"):
            validate_plan({**deepcopy(self.spec), "mode": "prospective"})
        with patch("evals.research.now", return_value=datetime(2020, 1, 1, tzinfo=timezone.utc)):
            self.assertEqual(validate_plan({**deepcopy(self.spec), "mode": "prospective"})["visibility_policy"], "first_seen")

    def test_plan_criteria_cannot_be_changed_in_place(self):
        with subscription_workspace() as root:
            book, study = self.setup_book(root)
            with book.db() as conn:
                plan = deepcopy(study["plan"])
                plan["criteria"]["min_gain"] = -100
                conn.execute("UPDATE research_plans SET plan_json=? WHERE id=?", (json.dumps(plan), study["id"]))
            with self.assertRaisesRegex(ValueError, "改写"):
                book.get(study["id"])

    def test_frozen_prices_and_news_cannot_be_rewritten_or_backfilled(self):
        with subscription_workspace() as root:
            book, study = self.setup_book(root)
            changed = deepcopy(self.bundle)
            changed["histories"]["600036"][0]["close"] += 1
            with self.assertRaisesRegex(ValueError, "改写"):
                book.freeze(study["id"], changed)
            changed = deepcopy(self.bundle)
            changed["news_events"].append({**changed["news_events"][0], "event_id": "backfill", "title": "new historical article"})
            with self.assertRaisesRegex(ValueError, "回填"):
                book.freeze(study["id"], changed)
            self.assertEqual(book.latest_data(study["id"])["data"]["histories"]["600036"][0]["close"], self.bundle["histories"]["600036"][0]["close"])

    def test_missing_data_and_early_test_leave_visible_attempts(self):
        with subscription_workspace() as root:
            book = ResearchBook(root / "research.db")
            study = book.create(self.spec)
            self.assertTrue(all(row["status"] == "no_data" for row in book.run_phase(study["id"], "train")))
            book.freeze(study["id"], self.bundle)
            self.assertTrue(all(row["status"] == "pending" for row in book.run_phase(study["id"], "test")))
            report = book.review(study["id"])
            self.assertEqual(report["verdict"], "awaiting_data")
            self.assertEqual(report["attempt_count"], 6)
            self.assertFalse(report["auto_promoted"])

    def test_engineering_loop_same_inputs_costs_closed_review_and_replay(self):
        with subscription_workspace() as root:
            book, study = self.setup_book(root)
            for phase in PHASES:
                rows = book.run_phase(study["id"], phase)
                self.assertTrue(all(row["status"] == "complete" for row in rows), rows)
                self.assertEqual(len({row["dataset_fingerprint"] for row in rows}), 1)
                self.assertTrue(all(row["package"]["cost_model"] == study["plan"]["cost_model"] for row in rows))
                for row in rows:
                    self.assertTrue(all(trade["signal_date"] < trade["trade_date"] for trade in row["portfolio"]["trades"]))
                    self.assertTrue(all(trade["fee"] >= 0 and trade["slippage_cost"] >= 0 for trade in row["portfolio"]["trades"]))
                    self.assertTrue(all(score["target_date"] <= study["plan"]["splits"][phase][1] for score in row["scored_rows"]))
                    self.assertTrue(book.replay(study["id"], row["attempt_id"])["reproduced"])
            report = book.review(study["id"])
            self.assertEqual(report["verdict"], "unverified_engineering")
            self.assertEqual(report["family_attempt_count"], 9)
            self.assertEqual(len(comparable_experiments(book.path)), 3)
            self.assertIn("下一轮", render_review(report))
            self.assertEqual(book.summaries()[0]["verdict"], "unverified_engineering")

    def test_future_extension_does_not_change_earlier_training_results(self):
        short = deepcopy(self.bundle)
        cutoff = self.days[75]
        short["histories"] = {symbol: [row for row in rows if row["date"] <= cutoff] for symbol, rows in short["histories"].items()}
        short["news_events"] = [row for row in short["news_events"] if row["published_at"][:10] <= cutoff]
        extended = deepcopy(self.bundle)
        for rows in extended["histories"].values():
            for row in rows:
                if row["date"] > cutoff:
                    row["close"] *= 2
        with subscription_workspace() as root:
            book, study = self.setup_book(root, bundle=short)
            first = book.run_phase(study["id"], "train")
            book.freeze(study["id"], extended)
            second = book.run_phase(study["id"], "train")
            self.assertEqual([row["metrics"] for row in first], [row["metrics"] for row in second])
            self.assertEqual([row["dataset_fingerprint"] for row in first], [row["dataset_fingerprint"] for row in second])

    def test_no_news_and_missing_model_logs_are_not_replaced_with_fake_results(self):
        no_news = {**deepcopy(self.bundle), "news_events": []}
        spec = deepcopy(self.spec)
        spec["candidate"] = "agent"
        spec["variants"].append({"id": "agent", "kind": "agent", "variant_fingerprint": "a" * 64})
        with subscription_workspace() as root:
            book, study = self.setup_book(root, spec, no_news)
            rows = {row["variant_id"]: row for row in book.run_phase(study["id"], "train")}
            self.assertEqual(rows["news"]["status"], "no_data")
            self.assertEqual(rows["agent"]["status"], "no_data")
            self.assertNotIn("metrics", rows["agent"])
            self.assertIn("没有调用替身", rows["agent"]["reason"])

    def test_runtime_failures_are_preserved_in_both_registries(self):
        with subscription_workspace() as root:
            book, study = self.setup_book(root)
            with patch("evals.research._technical_signal", side_effect=RuntimeError("test failure")):
                rows = book.run_phase(study["id"], "train")
            self.assertTrue(all(row["status"] == "failed" for row in rows))
            self.assertTrue(all(row["status"] == "failed" for row in list_experiments(book.path)))
            self.assertEqual(book.review(study["id"])["attempt_count"], 3)

    def test_child_research_requires_review_and_new_test_window(self):
        with subscription_workspace() as root:
            book, study = self.setup_book(root)
            child = {**deepcopy(self.spec), "parent_id": study["id"]}
            with self.assertRaises(ValueError): book.create(child)
            book.review(study["id"])
            with self.assertRaisesRegex(ValueError, "独立"):
                book.create(child)
            child["splits"]["test"] = ["2026-10-01", "2026-10-20"]
            new = book.create(child)
            self.assertEqual(new["family_id"], study["family_id"])
            self.assertEqual(book.review(study["id"])["family_study_count"], 2)

    def test_code_changes_prevent_claiming_an_identical_replay(self):
        with subscription_workspace() as root:
            book, study = self.setup_book(root)
            rows = book.run_phase(study["id"], "train")
            with patch("evals.research.code_identity", return_value="changed"):
                with self.assertRaisesRegex(ValueError, "变化"):
                    book.replay(study["id"], rows[0]["attempt_id"])

    def agent_book(self, root):
        example = model_log(self.bundle, self.days[30])
        spec = deepcopy(self.spec)
        spec["candidate"] = "agent"
        spec["variants"].append({"id": "agent", "kind": "agent", "variant_fingerprint": canonical_hash(decision_log_variant(example))})
        book, study = self.setup_book(root, spec)
        return book, study, example

    def test_model_attachment_is_first_output_only_and_price_bound(self):
        with subscription_workspace() as root:
            book, study, log = self.agent_book(root)
            book.attach_log(study["id"], "agent", log)
            self.assertEqual(book.attach_log(study["id"], "agent", log)["status"], "already_attached")
            changed = deepcopy(log)
            changed["predictions"][0]["direction"] = "down"
            with self.assertRaisesRegex(ValueError, "首次"):
                book.attach_log(study["id"], "agent", changed)
            changed = model_log(self.bundle, self.days[31])
            changed["input_dataset"]["snapshot"][0]["close"] += 1
            changed["dataset_fingerprint"] = canonical_hash(changed["input_dataset"])
            with self.assertRaisesRegex(ValueError, "收盘价"):
                book.attach_log(study["id"], "agent", changed)

    def test_attached_model_cannot_read_future_or_late_collected_news(self):
        day = self.days[30]
        example = model_log(self.bundle, day)
        spec = deepcopy(self.spec)
        spec["variants"].append({"id": "agent", "kind": "agent", "variant_fingerprint": canonical_hash(decision_log_variant(example))})
        with subscription_workspace() as root:
            book, study = self.setup_book(root, spec)
            future = next(row for row in self.bundle["news_events"] if row["published_at"][:10] > day)
            example["input_dataset"]["news_screen"] = {"candidates": [{"top_events": [future]}]}
            example["dataset_fingerprint"] = canonical_hash(example["input_dataset"])
            with self.assertRaisesRegex(ValueError, "尚不可见"):
                book.attach_log(study["id"], "agent", example)
        with subscription_workspace() as root:
            spec["mode"] = "prospective"
            bundle = deepcopy(self.bundle)
            bundle["news_events"][0]["first_seen_at"] = "2026-10-27T10:00:00+08:00"
            book = ResearchBook(root / "research.db")
            with patch("evals.research.now", return_value=datetime(2026, 1, 1, tzinfo=timezone.utc)):
                study = book.create(spec)
            book.freeze(study["id"], bundle)
            example["input_dataset"]["news_screen"] = {"candidates": [{"top_events": [bundle["news_events"][0]]}]}
            example["dataset_fingerprint"] = canonical_hash(example["input_dataset"])
            with self.assertRaisesRegex(ValueError, "尚不可见"):
                book.attach_log(study["id"], "agent", example)

    def test_same_day_agent_variants_cannot_change_frozen_account(self):
        with subscription_workspace() as root:
            log = model_log(self.bundle, self.days[30])
            second = deepcopy(log)
            second["agent_sdk"]["role_models"]["manager"]["model"] = "second-fixture"
            second["input_dataset"]["account"]["cash"] = 999
            second["dataset_fingerprint"] = canonical_hash(second["input_dataset"])
            spec = deepcopy(self.spec)
            spec["variants"] += [{"id": "agent1", "kind": "agent", "variant_fingerprint": canonical_hash(decision_log_variant(log))},
                                 {"id": "agent2", "kind": "agent", "variant_fingerprint": canonical_hash(decision_log_variant(second))}]
            book, study = self.setup_book(root, spec)
            book.attach_log(study["id"], "agent1", log)
            with self.assertRaisesRegex(ValueError, "输入不同"):
                book.attach_log(study["id"], "agent2", second)

    def test_agent_replay_uses_saved_predictions_and_records_usage_without_model_calls(self):
        with subscription_workspace() as root:
            book, study, _ = self.agent_book(root)
            for day in self.days[30:49]:
                book.attach_log(study["id"], "agent", model_log(self.bundle, day))
            rows = {row["variant_id"]: row for row in book.run_phase(study["id"], "train")}
            agent = rows["agent"]
            self.assertEqual(agent["status"], "complete")
            self.assertEqual(agent["metrics"]["sample_count"], 30)
            self.assertEqual(agent["metrics"]["model_requests"], 19 * 4)
            self.assertEqual(agent["portfolio"]["strategy"], "frozen-prediction-allocation-v1")
            self.assertTrue(book.replay(study["id"], agent["attempt_id"])["reproduced"])

    def test_semantic_report_can_be_attached_later_and_cannot_reference_another_log(self):
        from evals.semantic_review import prepare_review
        with subscription_workspace() as root:
            book, study, log = self.agent_book(root)
            book.attach_log(study["id"], "agent", log)
            review = prepare_review(log)
            self.assertEqual(book.attach_log(study["id"], "agent", log, review)["status"], "semantic_attached")
            bad = deepcopy(review)
            bad["source_log_fingerprint"] = "wrong"
            with self.assertRaisesRegex(ValueError, "不属于"):
                book.attach_log(study["id"], "agent", log, bad)

    def test_dashboard_exposes_research_tasks_without_fetching_market_data(self):
        from dashboard.app import app
        with subscription_workspace() as root:
            book, study = self.setup_book(root)
            with patch("dashboard.app.EXPERIMENT_DB", book.path):
                response = app.test_client().get("/api/experiments")
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.get_json()["research"][0]["id"], study["id"])

    def predictor_book(self, root):
        from evals.research_agent import PROTOCOL
        example = model_log(self.bundle, self.days[30])
        example["strategy_version"] = PROTOCOL
        identity = decision_log_variant(example)
        spec = deepcopy(self.spec)
        spec["candidate"] = "agent"
        spec["variants"].append({"id": "agent", "kind": "agent", "variant_fingerprint": canonical_hash(identity)})
        book, study = self.setup_book(root, spec)
        return book, study, example, identity

    def test_frozen_predictor_checks_guards_before_spending_model_requests(self):
        from evals.research_agent import predict_frozen
        with subscription_workspace() as root:
            book, study, _, identity = self.predictor_book(root)
            with patch("evals.research_agent.variant_identity", return_value=identity), patch("evals.research_agent.run_agent_graph", new_callable=AsyncMock) as graph:
                with self.assertRaisesRegex(ValueError, "allow-retrospective"):
                    asyncio.run(predict_frozen(book, study["id"], "agent", self.days[30], output=root / "out.json"))
                (root / "out.json").write_text("prior output", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "已存在"):
                    asyncio.run(predict_frozen(book, study["id"], "agent", self.days[30], output=root / "out.json", allow_retrospective=True))
                graph.assert_not_awaited()
            self.assertEqual(book.attempts(study["id"]), [])

    def test_frozen_predictor_retains_failed_generation_attempt(self):
        from evals.research_agent import predict_frozen
        with subscription_workspace() as root:
            book, study, _, identity = self.predictor_book(root)
            with patch("evals.research_agent.variant_identity", return_value=identity), patch("evals.research_agent.run_agent_graph", new_callable=AsyncMock, side_effect=RuntimeError("fixture generation failed")):
                with self.assertRaisesRegex(RuntimeError, "fixture generation failed"):
                    asyncio.run(predict_frozen(book, study["id"], "agent", self.days[30], output=root / "out.json", allow_retrospective=True))
            attempt = book.attempts(study["id"])[0]
            self.assertEqual(attempt["status"], "failed")
            self.assertEqual(attempt["phase"], "predict:" + self.days[30])
            self.assertFalse((root / "out.json").exists())

    def test_frozen_predictor_attaches_zero_order_output_and_prevents_repeat(self):
        from evals.research_agent import predict_frozen
        with subscription_workspace() as root:
            book, study, example, identity = self.predictor_book(root)
            async def fake_graph(reader, market, day, *args, **kwargs):
                value = deepcopy(example)
                value["input_dataset"] = reader.dataset(day, {})
                return value
            with patch("evals.research_agent.variant_identity", return_value=identity), patch("evals.research_agent.run_agent_graph", side_effect=fake_graph) as graph:
                output = asyncio.run(predict_frozen(book, study["id"], "agent", self.days[30], output=root / "out.json", allow_retrospective=True))
                with self.assertRaisesRegex(ValueError, "首次输出"):
                    asyncio.run(predict_frozen(book, study["id"], "agent", self.days[30], output=root / "second.json", allow_retrospective=True))
                self.assertEqual(graph.await_count, 1)
            saved = json.loads((root / "out.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["execution"]["orders_submitted"], 0)
            self.assertEqual(book.attempts(study["id"])[0]["metrics"]["model_requests"], 4)
            with self.assertRaises(ValueError): book.replay(study["id"], output["attempt_id"])

    def test_review_decisions_use_fixed_pair_and_require_coverage_and_sample_size(self):
        with subscription_workspace() as root:
            book, study = self.setup_book(root)
            for phase in PHASES: book.run_phase(study["id"], phase)
            latest = {(row["phase"], row["variant_id"]): row for row in book.attempts(study["id"])}
            study["plan"]["mode"] = "retrospective"
            for row in latest.values(): row["synthetic_inputs"] = False  # 单元分支替身，不作真实业绩报告
            criteria = study["plan"]["criteria"]
            criteria.update(min_gain=-100, max_drawdown_pct=100, max_total_cost_pct=100)
            self.assertEqual(review_results(study, latest)["verdict"], "retain_for_forward_validation")
            criteria["min_gain"] = 101
            self.assertEqual(review_results(study, latest)["verdict"], "reject")
            criteria["min_matured_predictions"] = 10000
            self.assertEqual(review_results(study, latest)["verdict"], "insufficient_data")
            latest[("test", "news")]["dataset_fingerprint"] = "different"
            self.assertEqual(review_results(study, latest)["verdict"], "not_comparable")
