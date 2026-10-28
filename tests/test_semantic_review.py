import asyncio
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import unittest

from agent.evidence import build_catalog
from agent.subscription import subscription_workspace
from evals.semantic_review import (ReviewCache, ReviewOptions, collect_claims, prepare_review,
    quantities, review_with_model, source_packets, summarize, validate_judgments, local_passages, _cache_key)


def log_fixture(text="据媒体报道，公司拟投资2亿元。", *, official=False, summary=None):
    context = {"date": "2026-10-26", "symbols": ["600519"], "snapshot": [], "fundamentals": [],
        "news": [{"symbol": "600519", "top_events": [{"symbol": "600519", "title": "项目投资计划",
            "summary": summary or "据媒体报道，公司2026年拟投资2亿元，营收同比增长10%。尚待审批。",
            "source": "fixture publisher", "data_quality": "synthetic", "published_at": "2026-10-26T10:00:00+08:00",
            "url": "https://static.cninfo.com.cn/fixture.pdf" if official else "https://example.org/fixture"}]}]}
    catalog = build_catalog(context, "cn", allow_synthetic=True)
    annotation = {"status": "supported", "supporting_ids": [catalog["records"][0]["evidence_id"]],
        "contradicting_ids": [], "invalidation_condition": "正式公告更正项目安排"}
    claims = [{"symbol": "600519", "claim": text, "evidence": annotation}]
    return {"kind": "real-news-single-analyst-acceptance", "analyst_result": {"claims": claims},
        "evidence_audit": {"catalog": catalog, "analyst_claims": {"fundamental-analyst": claims}}}


def response_for(rows, catalog, *, kind="media_statement", support="supported", inference="not_applicable"):
    reviews = []
    for row in rows:
        passage = source_packets(row, catalog)[0]["passages"][-1]
        reviews.append({"claim_id": row["claim_id"], "claim_kind": kind, "source_support": support,
            "inference_review": inference, "reason": "与保存片段对照后的测试结论", "needed_context": "",
            "quotes": [{"passage_id": passage["passage_id"], "quote": passage["text"]}]})
    return {"reviews": reviews}


class FakeClient:
    def __init__(self, model, *, callback=None, fail=False):
        self.model, self.callback, self.fail = model, callback, fail
        self.settings, self.calls = {}, []

    def report(self):
        return {"requests_started": len(self.calls), "completed_calls": 0 if self.fail else len(self.calls),
                "total_tokens": len(self.calls) * 100}

    async def complete(self, prompt, schema, *, role):
        self.calls.append((prompt, role))
        if self.fail:
            raise TimeoutError("bounded test timeout")
        packet = json.loads(prompt.splitlines()[-1])
        if self.callback:
            return self.callback(packet)
        source_by_id = {source["evidence_id"]: source for source in packet["sources"]}
        rows = []
        for claim in packet["claims"]:
            passage = source_by_id[claim["evidence"]["supporting_ids"][0]]["passages"][-1]
            rows.append({"claim_id": claim["claim_id"], "claim_kind": "media_statement", "source_support": "supported",
                "inference_review": "not_applicable", "reason": "test", "needed_context": "",
                "quotes": [{"passage_id": passage["passage_id"], "quote": passage["text"]}]})
        return {"reviews": rows}


class SemanticRuleTests(unittest.TestCase):
    def test_full_decision_log_collects_specialists_manager_forecast_and_risk(self):
        result = log_fixture()
        claims = result["analyst_result"]["claims"]
        annotation = deepcopy(claims[0]["evidence"])
        result.pop("kind")
        result.pop("analyst_result")
        result["evidence_audit"]["analyst_claims"]["technical-analyst"] = [{"symbol": "600519",
            "claim": "缺少可用技术资料", "evidence": {"status": "insufficient", "supporting_ids": [],
                "contradicting_ids": [], "invalidation_condition": "取得行情后重审"}}]
        result["decisions"] = [{"symbol": "600519", "action": "BUY", "reason": "建议小规模研究该机会", "evidence": annotation}]
        result["predictions"] = [{"symbol": "600519", "rationale": "未来趋势偏正，但存在不确定性", "evidence": annotation}]
        result["risk_review"] = {"assessments": [{"symbol": "600519", "action": "BUY", "reason": "限制仓位", "evidence": annotation}]}
        report = prepare_review(result, ReviewOptions(sample_rate=1))
        self.assertEqual({row["role"] for row in report["items"]},
                         {"technical-analyst", "fundamental-analyst", "investment-manager", "risk-manager"})
        self.assertTrue(all(row["kind_hint"] == "agent_inference" for row in report["items"] if row["prediction"]))
        self.assertEqual(sum(row["trade_impact"] for row in report["items"]), 2)

    def test_rejects_us_legacy_and_altered_logs(self):
        with self.assertRaisesRegex(ValueError, "旧版"):
            prepare_review({})
        result = log_fixture()
        result["evidence_audit"]["catalog"]["market"] = "us"
        with self.assertRaisesRegex(ValueError, "A股"):
            prepare_review(result)
        result = log_fixture()
        result["evidence_audit"]["catalog"]["records"][0]["payload"]["summary"] = "altered"
        with self.assertRaises(ValueError):
            prepare_review(result)

    def test_atomic_claims_keep_parent_context_and_exact_offsets(self):
        result = log_fixture("媒体报道公司拟投资；但金额尚未确认。")
        _, claims = collect_claims(result)
        self.assertEqual(len(claims), 2)
        for row in claims:
            self.assertEqual(row["parent_text"][row["start"]:row["end"]].strip(), row["text"])
            self.assertIn("金额尚未确认", row["parent_text"])
        self.assertEqual([row["claim_id"] for row in claims],
                         [row["claim_id"] for row in collect_claims(result)[1]])

    def test_gap_review_sees_actual_input_and_cache_tracks_uncited_text_changes(self):
        first = log_fixture("缺少业绩信息。")
        second = log_fixture("缺少业绩信息。", summary="公司2026年已披露营业收入20亿元，净利润3亿元。")
        for result in (first, second):
            result["analyst_result"]["claims"][0]["evidence"].update(status="insufficient", supporting_ids=[])
        catalog1, rows1 = collect_claims(first)
        catalog2, rows2 = collect_claims(second)
        self.assertEqual(rows1[0]["claim_id"], rows2[0]["claim_id"])
        self.assertNotEqual(_cache_key(rows1[0] | {"rule_findings": []}, catalog1, "test", ReviewOptions()),
                            _cache_key(rows2[0] | {"rule_findings": []}, catalog2, "test", ReviewOptions()))
        self.assertIn("净利润3亿元", local_passages(rows2[0], catalog2)[1]["text"])

    def test_risk_role_reporting_manager_plan_is_not_automatically_an_impersonation_error(self):
        from evals.semantic_review import rule_check
        catalog, rows = collect_claims(log_fixture("经理：建议HOLD。"))
        rows[0]["role"] = "risk-manager"
        codes = {issue["code"] for issue in rule_check(rows[0], source_packets(rows[0], catalog))}
        self.assertNotIn("role_scope_error", codes)
        self.assertIn("foreign_role_attribution_needs_log", codes)

    def test_surface_number_match_does_not_become_semantic_approval(self):
        result = prepare_review(log_fixture("媒体报道公司拟投资20000万元。"), ReviewOptions(sample_rate=1))
        row = result["items"][0]
        self.assertIn("numeric_surface_match_only", [issue["code"] for issue in row["rule_findings"]])
        self.assertIsNone(row["review"])
        self.assertFalse(result["semantic_truth_verified"])
        self.assertEqual(quantities("2亿元"), quantities("20000万元"))
        self.assertNotEqual(quantities("10%"), quantities("10个百分点"))

    def test_unit_period_comparison_and_modality_mismatches_are_flagged(self):
        result = prepare_review(log_fixture("据媒体报道，公司2025年已完成2万元投资，营收环比增长10个百分点。"))
        codes = {issue["code"] for row in result["items"] for issue in row["rule_findings"]}
        self.assertTrue({"numeric_value_or_unit_unmatched", "reporting_period_unmatched",
                         "comparison_basis_risk", "modality_upgrade_risk"} <= codes)
        self.assertTrue(all(row["review"] is None for row in result["items"]))

    def test_role_scope_error_does_not_treat_explicit_suggestion_as_approval(self):
        report = prepare_review(log_fixture("经理：HOLD；风控：审批通过。"))
        self.assertEqual(summarize(report)["rule_findings"]["role_scope_error"], 2)
        report = prepare_review(log_fixture("建议经理考虑HOLD，仍需独立风控审批。"))
        self.assertNotIn("role_scope_error", summarize(report)["rule_findings"])

    def test_sampling_is_stable_and_reports_unreviewed_critical_overflow(self):
        result = log_fixture("经理：HOLD；风控：审批通过；媒体报道拟投资2亿元。")
        options = ReviewOptions(sample_rate=0, max_claims=1)
        first, second = prepare_review(result, options), prepare_review(result, options)
        self.assertEqual(first, second)
        summary = summarize(first)
        self.assertGreater(summary["strata"]["critical"]["available"], summary["strata"]["critical"]["selected"])
        self.assertEqual(summary["reviewed_count"], 0)

    def test_literal_quotes_receive_host_offsets_and_reject_paraphrases_or_wrong_sources(self):
        catalog, rows = collect_claims(log_fixture())
        valid = response_for(rows, catalog)
        reviewed = validate_judgments(valid, rows, catalog)[rows[0]["claim_id"]]
        quote = reviewed["quotes"][0]
        passage = source_packets(rows[0], catalog)[0]["passages"][-1]["text"]
        self.assertEqual(passage[quote["start"]:quote["end"]], quote["quote"])
        for change in ({"quote": "公司已完成投资"}, {"passage_id": "other-news/summary"}):
            bad = deepcopy(valid)
            bad["reviews"][0]["quotes"][0].update(change)
            with self.assertRaises(ValueError):
                validate_judgments(bad, rows, catalog)

    def test_review_must_cover_each_claim_once_and_have_quotes_for_support(self):
        catalog, rows = collect_claims(log_fixture())
        valid = response_for(rows, catalog)
        cases = [{"reviews": []}, {"reviews": valid["reviews"] * 2}]
        no_quote = deepcopy(valid)
        no_quote["reviews"][0]["quotes"] = []
        cases.append(no_quote)
        for response in cases:
            with self.subTest(response=response), self.assertRaises(ValueError):
                validate_judgments(response, rows, catalog)

    def test_model_can_select_passage_id_and_host_restores_exact_text(self):
        catalog, rows = collect_claims(log_fixture())
        response = response_for(rows, catalog)
        del response["reviews"][0]["quotes"][0]["quote"]
        checked = validate_judgments(response, rows, catalog)[rows[0]["claim_id"]]
        self.assertEqual(checked["quotes"][0]["quote"], source_packets(rows[0], catalog)[0]["passages"][-1]["text"])
        self.assertEqual(checked["quotes"][0]["scope"], "source_excerpt")

    def test_local_role_record_can_explain_workflow_but_cannot_prove_news(self):
        catalog, rows = collect_claims(log_fixture("经理：HOLD。"))
        response = response_for(rows, catalog, kind="workflow_assertion", support="insufficient")
        response["reviews"][0]["quotes"] = [{"passage_id": "local/" + rows[0]["claim_id"] + "/role"}]
        checked = validate_judgments(response, rows, catalog)[rows[0]["claim_id"]]
        self.assertIn("role_scope_error", checked["host_adjustments"])
        response["reviews"][0].update(claim_kind="media_statement", source_support="supported")
        with self.assertRaises(ValueError):
            validate_judgments(response, rows, catalog)

    def test_media_does_not_become_confirmed_company_disclosure(self):
        catalog, rows = collect_claims(log_fixture("公司公告披露拟投资2亿元。"))
        response = response_for(rows, catalog, kind="company_disclosure")
        reviewed = validate_judgments(response, rows, catalog)[rows[0]["claim_id"]]
        self.assertEqual(reviewed["source_support"], "context_needed")
        self.assertIn("primary_disclosure_missing", reviewed["host_adjustments"])
        official, official_rows = collect_claims(log_fixture("公司公告披露拟投资2亿元。", official=True))
        reviewed = validate_judgments(response_for(official_rows, official, kind="company_disclosure"),
                                     official_rows, official)[official_rows[0]["claim_id"]]
        self.assertEqual(reviewed["source_support"], "supported")

    def test_forecast_is_assessed_as_inference_without_requiring_literal_future_outcome(self):
        catalog, rows = collect_claims(log_fixture("这可能改善未来业务布局，但收益规模尚不确定。"))
        response = response_for(rows, catalog, kind="agent_inference", support="not_applicable", inference="weak")
        response["reviews"][0]["quotes"] = []
        checked = validate_judgments(response, rows, catalog)[rows[0]["claim_id"]]
        self.assertEqual(checked["inference_review"], "weak")
        response["reviews"][0]["source_support"] = "contradicted"
        with self.assertRaises(ValueError):
            validate_judgments(response, rows, catalog)

    def test_options_reject_invalid_sampling_or_budget(self):
        for params in ({"sample_rate": -1}, {"sample_rate": 2}, {"max_requests": 0}, {"timeout": 0}):
            with self.subTest(params=params), self.assertRaises(ValueError):
                ReviewOptions(**params)

    def test_rules_cli_is_read_only_and_needs_no_model(self):
        with subscription_workspace() as root:
            source, output = root / "input.json", root / "report.json"
            source.write_text(json.dumps(log_fixture(), ensure_ascii=False), encoding="utf-8")
            original = source.read_bytes()
            run = subprocess.run([sys.executable, "scripts/review_evidence.py", str(source),
                "--sample-rate", "1", "--output", str(output)], capture_output=True, encoding="utf-8", timeout=15)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            self.assertEqual(source.read_bytes(), original)
            self.assertEqual(json.loads(output.read_text(encoding="utf-8"))["model_usage"]["requests_started"], 0)


class SemanticModelTests(unittest.IsolatedAsyncioTestCase):
    async def test_cache_reuse_and_invalidation_on_text_or_model_changes(self):
        result = log_fixture()
        options = ReviewOptions(sample_rate=1)
        clients = []
        def factory(model):
            client = FakeClient(model)
            clients.append(client)
            return client
        with subscription_workspace() as root:
            cache = ReviewCache(root / "cache.db")
            first = await review_with_model(result, options=options, cache=cache, client_factory=factory)
            second = await review_with_model(result, options=options, cache=cache, client_factory=factory)
            self.assertEqual(first["status"], "completed")
            self.assertEqual(second["items"][0]["review_status"], "cached")
            self.assertEqual(second["items"][0]["review"], first["items"][0]["review"])
            self.assertEqual(second["model_usage"]["requests_started"], 0)
            self.assertEqual(len(clients), 1)
            await review_with_model(log_fixture("据媒体报道，公司计划投资2亿元。"), options=options, cache=cache, client_factory=factory)
            await review_with_model(result, model="different-test-model", options=options, cache=cache, client_factory=factory)
            self.assertEqual(len(clients), 3)

    async def test_separate_budget_and_critical_claims_first(self):
        client = FakeClient("test")
        report = await review_with_model(log_fixture("据媒体报道，公司发布计划；经理：HOLD。"),
            options=ReviewOptions(sample_rate=1, batch_size=1, max_requests=1), client_factory=lambda _: client)
        self.assertEqual(report["status"], "budget_exhausted")
        packet = json.loads(client.calls[0][0].splitlines()[-1])
        self.assertEqual(packet["claims"][0]["kind_hint"], "workflow_assertion")
        self.assertEqual(client.settings["request_limit"], 1)
        self.assertEqual(client.settings["request_timeout"], 75)
        self.assertEqual(summarize(report)["reviewed_count"], 1)
        self.assertEqual(sum(row["review_status"] == "not_reviewed" for row in report["items"]), 1)
        self.assertEqual(next(row for row in report["items"] if row["review"])["review"]["source_support"], "insufficient")

    async def test_timeout_and_invalid_quotes_fail_without_poisoning_cache(self):
        for client in (FakeClient("test", fail=True), FakeClient("test", callback=lambda packet: {"reviews": [{
            "claim_id": packet["claims"][0]["claim_id"], "claim_kind": "media_statement", "source_support": "supported",
            "inference_review": "not_applicable", "reason": "test", "needed_context": "",
            "quotes": [{"passage_id": "invented", "quote": "not in source"}]}]})):
            with self.subTest(fail=client.fail), subscription_workspace() as root:
                cache = ReviewCache(root / "cache.db")
                report = await review_with_model(log_fixture(), options=ReviewOptions(sample_rate=1),
                    cache=cache, client_factory=lambda _: client)
                self.assertEqual(report["status"], "failed")
                self.assertEqual(summarize(report)["reviewed_count"], 0)
                self.assertTrue(report["errors"])
                good = FakeClient("test")
                retried = await review_with_model(log_fixture(), options=ReviewOptions(sample_rate=1),
                    cache=cache, client_factory=lambda _: good)
                self.assertEqual(retried["items"][0]["review_status"], "reviewed")
                self.assertEqual(len(good.calls), 1)

    async def test_small_input_budget_does_not_truncate_sources_or_invoke_model(self):
        client = FakeClient("test")
        report = await review_with_model(log_fixture(), options=ReviewOptions(sample_rate=1, input_char_limit=10),
                                        client_factory=lambda _: client)
        self.assertEqual(report["status"], "input_limit")
        self.assertEqual(client.calls, [])
