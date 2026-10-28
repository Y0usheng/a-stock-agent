import asyncio
from copy import deepcopy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from agent.opportunities import (OpportunityBook, compact_context, discover, normalize_event,
                                 refresh_feed, route_feed, validate_review)
from agent.subscription import subscription_workspace
from evals.experiments import canonical_hash

DAY = "2026-10-26"
OBSERVED = DAY + "T12:00:00+08:00"


def event(title="数据中心算力建设长期需求观察", excerpt="公开测试片段：数据中心服务器扩容仍需核对投入、客户订单和现金流。", suffix="1"):
    return normalize_event({"title": title, "excerpt": excerpt, "url": "https://finance.eastmoney.com/a/test" + suffix + ".html",
                            "published_at": DAY + "T10:00:00+08:00"}, OBSERVED)


def feed(events=None):
    return {"as_of_date": DAY, "events": [event()] if events is None else events, "coverage": "unit_fixture", "window_days": 30}


def judgment(leads, assessment="research_priority"):
    return {"themes": [{"theme_id": row["theme_id"], "assessment": assessment, "durability": "structural",
        "horizon_months": 36, "priority": 1, "mechanism": "仅测试：需求可能形成业务收入，需核验", "counterargument": "可能没有利润转化",
        "validation_needed": "核实公司业务、订单、现金流、竞争和估值", "invalidation_condition": "业务暴露或利润转化不成立",
        "beneficiary_symbols": row["possible_symbols"][:1], "supporting_event_ids": [row["events"][0]["event_id"]],
        "contradicting_event_ids": [], "source_observations": [{"event_id": row["events"][0]["event_id"],
            "quote": row["events"][0]["excerpt"][:30], "source_kind": row["events"][0]["source_kind"]}]} for row in leads]}


class FakeClient:
    model = "unit-fixture-model"
    def __init__(self, result):
        self.settings = {}
        self.result, self.calls = result, 0

    async def complete(self, prompt, schema, *, role):
        self.calls += 1
        return deepcopy(self.result)

    def report(self):
        return {"requests_started": self.calls, "total_tokens": 10 * self.calls}


class OpportunityTests(unittest.TestCase):
    def run_discovery(self, source, **kwargs):
        with subscription_workspace() as root:
            return asyncio.run(discover(source, book=OpportunityBook(root / "opportunities.db"), **kwargs))

    def test_broad_theme_does_not_require_seed_company_in_article(self):
        leads, excluded = route_feed(feed())
        self.assertFalse(excluded)
        self.assertEqual(leads[0]["theme_id"], "computing")
        self.assertNotIn("300750", leads[0]["possible_symbols"])
        client = FakeClient(judgment(leads))
        result = self.run_discovery(feed(), client=client)
        self.assertEqual(result["selected_symbols"], ["601138"])
        self.assertEqual(result["orders_submitted"], 0)

    def test_rules_only_cannot_select_stock_or_claim_long_term_merit(self):
        result = self.run_discovery(feed(), use_model=False)
        self.assertEqual(result["selected_symbols"], [])
        self.assertEqual(result["usage"]["requests_started"], 0)
        self.assertEqual(result["themes"][0]["status"], "unreviewed_keyword_lead")

    def test_reprints_are_not_independent_corroboration(self):
        one, two = event(), event(suffix="2")
        leads, excluded = route_feed(feed([one, two]))
        self.assertEqual(len(excluded), 1)
        self.assertEqual(leads[0]["origin_group_count"], 1)

    def test_unknown_media_origins_stay_one_group_even_different_headlines(self):
        leads, _ = route_feed(feed([event(), event(title="服务器需求另一条转述", suffix="2")]))
        self.assertEqual(len(leads[0]["events"]), 2)
        self.assertEqual(leads[0]["origin_group_count"], 1)

    def test_late_first_seen_never_backfills_old_decision(self):
        row = event()
        row["first_seen_at"] = DAY + "T16:00:00+08:00"
        leads, excluded = route_feed(feed([row]))
        self.assertFalse(leads)
        self.assertEqual(excluded[0]["reason"], "not_observed_by_price_cutoff")

    def test_future_publication_and_unknown_timestamp_rejected(self):
        raw = {"url": "https://finance.eastmoney.com/a/fixture.html", "title": "title", "excerpt": "excerpt", "published_at": ""}
        with self.assertRaises(ValueError):
            normalize_event(raw, OBSERVED)
        raw["published_at"] = DAY + "T18:00:00+08:00"
        with self.assertRaises(ValueError):
            normalize_event(raw, OBSERVED)

    def test_domain_and_title_only_rejected(self):
        for url in ("https://evil.example/article", "http://finance.eastmoney.com/a/test.html", "https://finance.eastmoney.com/"):
            with self.assertRaises(ValueError):
                normalize_event({"url": url, "title": "title", "excerpt": "body", "published_at": DAY}, OBSERVED)
        with self.assertRaises(ValueError):
            normalize_event({"url": "https://finance.eastmoney.com/a/test.html", "title": "title", "published_at": DAY}, OBSERVED)

    def test_changed_excerpt_is_detected(self):
        row = event()
        row["excerpt"] += "changed"
        self.assertEqual(route_feed(feed([row]))[1][0]["reason"], "content_hash_mismatch")

    def test_original_first_seen_and_revisions_are_preserved(self):
        with subscription_workspace() as root:
            book = OpportunityBook(root / "test.db")
            raw = {k: event()[k] for k in ("title", "excerpt", "url", "published_at")}
            book.ingest([raw], OBSERVED)
            book.ingest([raw], DAY + "T13:00:00+08:00")
            self.assertEqual(len(book.feed(DAY)["events"]), 1)
            self.assertEqual(book.feed(DAY)["events"][0]["first_seen_at"], OBSERVED)
            raw["excerpt"] += "revision"
            book.ingest([raw], DAY + "T13:00:00+08:00")
            self.assertEqual(len(book.feed(DAY)["events"]), 2)
            leads, excluded = route_feed(book.feed(DAY))
            self.assertTrue(leads[0]["events"][0]["excerpt"].endswith("revision"))
            self.assertEqual(excluded[0]["reason"], "duplicate_headline_same_day")

    def test_uncaptured_quote_and_media_to_disclosure_laundering_rejected(self):
        leads, _ = route_feed(feed())
        review = judgment(leads)
        for key, value in (("quote", "原文没有说净利润增长百分之五十"), ("source_kind", "company_disclosure")):
            bad = deepcopy(review)
            bad["themes"][0]["source_observations"][0][key] = value
            with self.assertRaises(ValueError):
                validate_review(bad, leads)

    def test_unknown_evidence_and_unmapped_company_rejected(self):
        leads, _ = route_feed(feed())
        review = judgment(leads)
        for key, value in (("supporting_event_ids", ["invented"]), ("beneficiary_symbols", ["300750"])):
            bad = deepcopy(review)
            bad["themes"][0][key] = value
            with self.assertRaises(ValueError):
                validate_review(bad, leads)

    def test_transient_or_short_horizon_cannot_be_priority(self):
        leads, _ = route_feed(feed())
        for key, value in (("durability", "transient"), ("horizon_months", 3)):
            bad = judgment(leads)
            bad["themes"][0][key] = value
            with self.assertRaises(ValueError):
                validate_review(bad, leads)

    def test_uncovered_grid_theme_not_mapped_to_hydropower(self):
        leads, _ = route_feed(feed([event(title="特高压电网建设", excerpt="测试公开片段：电网输配电建设需核对设备受益方和收入转化。")]))
        self.assertEqual(leads[0]["possible_symbols"], [])
        self.assertEqual(leads[0]["mapping_status"], "outside_controlled_universe")

    def test_no_news_retains_held_and_prior_research_without_seed_fallback(self):
        result = self.run_discovery(feed([]), held=["600276"], previous_symbols=["601899"], use_model=False)
        self.assertEqual(result["active_symbols"], ["600276", "601899"])
        self.assertNotIn("300750", result["active_symbols"])

    def test_bad_model_result_fails_closed_and_keeps_research(self):
        result = self.run_discovery(feed(), client=FakeClient({"themes": []}), previous_symbols=["600276"])
        self.assertTrue(result["review_error"])
        self.assertEqual(result["selected_symbols"], [])
        self.assertEqual(result["active_symbols"], ["600276"])

    def test_repeated_snapshot_reuses_review_zero_new_calls(self):
        with subscription_workspace() as root:
            book = OpportunityBook(root / "test.db")
            leads, _ = route_feed(feed())
            client = FakeClient(judgment(leads))
            first = asyncio.run(discover(feed(), book=book, client=client))
            second_client = FakeClient(judgment(leads))
            second = asyncio.run(discover(feed(), book=book, client=second_client))
            self.assertEqual(client.calls, 1)
            self.assertEqual(second_client.calls, 0)
            self.assertTrue(second["review_cached"])
            self.assertEqual(first["selected_symbols"], second["selected_symbols"])
            self.assertEqual(client.settings["request_limit"], 1)
            self.assertEqual(client.settings["request_timeout"], 75)

    def test_future_observation_is_not_selection_day_or_portfolio_profit(self):
        with subscription_workspace() as root:
            book = OpportunityBook(root / "test.db")
            leads, _ = route_feed(feed())
            asyncio.run(discover(feed(), book=book, client=FakeClient(judgment(leads))))
            history = {"601138": [{"date": DAY, "close": 1}, {"date": "2026-10-27", "close": 10}, {"date": "2026-10-28", "close": 11}]}
            observation = book.view(history)["forward_observations"][0]
            self.assertEqual(observation["observed_return_pct"], 10)
            self.assertEqual(observation["observation_start"], "2026-10-27")
            self.assertEqual(book.view({})["forward_observations"][0]["status"], "awaiting_future_prices")

    def test_read_only_page_does_not_create_database(self):
        with subscription_workspace() as root:
            book = OpportunityBook(root / "absent.db")
            self.assertEqual(book.view()["status"], "no_data")
            self.assertFalse(book.path.exists())

    def test_network_failure_is_recorded_and_cached_events_survive(self):
        with subscription_workspace() as root:
            book = OpportunityBook(root / "test.db")
            raw = {k: event()[k] for k in ("title", "excerpt", "url", "published_at")}
            book.ingest([raw], OBSERVED)
            with patch("requests.get", side_effect=TimeoutError("unit fixture timeout")):
                self.assertEqual(refresh_feed(book)["status"], "failed")
            self.assertEqual(len(book.feed(DAY)["events"]), 1)

    def test_real_feed_contract_keeps_required_query_and_normalizes_article_link(self):
        from unittest.mock import Mock
        with subscription_workspace() as root:
            book = OpportunityBook(root / "test.db")
            response = Mock()
            response.json.return_value = {"data": {"fastNewsList": [{"title": "数据中心测试快讯", "summary": "测试原文片段，业务收入需进一步核实。", "showTime": DAY + " 10:00:00", "code": "20261026123456"}]}}
            with patch("requests.get", return_value=response) as request:
                receipt = refresh_feed(book)
            self.assertEqual(receipt["status"], "ok")
            self.assertEqual(receipt["accepted"], 1)
            self.assertIn("req_trace", request.call_args.kwargs["params"])
            self.assertEqual(request.call_args.kwargs["timeout"], (5, 15))

    def test_hypothesis_context_cannot_become_company_document_evidence(self):
        from tests.test_investment import example
        from agent.subscription_roles import export_paper_context
        from agent.evidence import build_catalog
        context, _ = example(ready=False)
        leads, _ = route_feed(feed())
        snapshot = self.run_discovery(feed(), client=FakeClient(judgment(leads)))
        context["news"]["theme_discovery"] = snapshot
        projected = export_paper_context(context, "cn", True)
        self.assertFalse([row for row in build_catalog(projected, "cn")["records"] if row["kind"] == "company_document"])
        self.assertEqual(compact_context(snapshot, ["601138"])["status"], "past_model_hypotheses_not_company_evidence")


if __name__ == "__main__":
    unittest.main()
