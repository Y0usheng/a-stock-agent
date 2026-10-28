from copy import deepcopy
import unittest

from agent.evidence import (build_catalog, validate_catalog, validate_claims, validate_links,
                            validate_result_evidence)
from evals.experiments import canonical_hash
from evals.news_store import get_stored_event, ingest_events, query_events
from agent.subscription import subscription_workspace


def context():
    return {"date": "2026-10-26", "symbols": ["AAPL"],
        "snapshot": [{"symbol": "AAPL", "date": "2026-10-26", "close": 100, "data_source": "yfinance"}],
        "fundamentals": [{"symbol": "AAPL", "report_date": "2026-09-30", "roe_pct": 20, "data_source": "yfinance"}],
        "news": [{"symbol": "AAPL", "top_events": [{"symbol": "AAPL", "event_id": "event-1",
            "title": "Company releases results", "summary": "Reported revenue rose", "source": "test publisher",
            "url": "https://example.org/article/1", "published_at": "2026-10-26T10:00:00-04:00",
            "first_seen_at": "2026-10-27T10:00:00+00:00", "data_quality": "live"}]}]}


def links(ids=(), status=None):
    return {"status": status or ("supported" if ids else "insufficient"), "supporting_ids": list(ids),
            "contradicting_ids": [], "invalidation_condition": "A corrected filing or newer observation reverses the claim"}


class EvidenceTests(unittest.TestCase):
    def test_catalog_freezes_excerpt_and_flags_late_observation(self):
        catalog = build_catalog(context(), "us")
        validate_catalog(catalog)
        news = next(row for row in catalog["records"] if row["kind"] == "news")
        self.assertTrue(news["eligible"])
        self.assertIn("observed_after_as_of_close", news["issues"])
        self.assertEqual(news["content_scope"], "title_and_excerpt")
        self.assertFalse(catalog["semantic_support_verified"])
        fundamentals = next(row for row in catalog["records"] if row["kind"] == "fundamentals")
        self.assertFalse(fundamentals["eligible"])
        self.assertIn("report_date_is_not_publication_time", fundamentals["issues"])

    def test_future_after_close_date_only_and_missing_metadata_are_not_citable(self):
        cases = [{"published_at": "2026-10-27T09:00:00-04:00"},
                 {"published_at": "2026-10-26T16:01:00-04:00"},
                 {"published_at": "2026-10-26"}, {"published_at": ""},
                 {"url": ""}, {"source": ""}]
        for changes in cases:
            packet = context()
            packet["news"][0]["top_events"][0].update(changes)
            catalog = build_catalog(packet, "us")
            row = next(row for row in catalog["records"] if row["kind"] == "news")
            with self.subTest(changes=changes):
                self.assertFalse(row["eligible"])
                with self.assertRaises(ValueError):
                    validate_links(links([row["evidence_id"]]), catalog, "AAPL")

    def test_date_only_becomes_usable_next_day(self):
        packet = context()
        packet["news"][0]["top_events"][0]["published_at"] = "2026-10-25"
        row = next(row for row in build_catalog(packet, "us")["records"] if row["kind"] == "news")
        self.assertTrue(row["eligible"])
        self.assertIn("date_only_usable_after_publication_day", row["issues"])

    def test_hash_changes_with_text_and_detects_rewritten_catalog(self):
        packet = context()
        before = build_catalog(packet, "us")
        packet["news"][0]["top_events"][0]["summary"] = "Corrected: revenue fell"
        after = build_catalog(packet, "us")
        self.assertNotEqual(before["fingerprint"], after["fingerprint"])
        before["records"][0]["payload"]["close"] = 999
        before["fingerprint"] = canonical_hash(before["records"])
        with self.assertRaises(ValueError):
            validate_catalog(before)

    def test_ids_stable_across_collection_times_but_audit_retains_them(self):
        packet = context()
        first = build_catalog(packet, "us")
        packet["news"][0]["top_events"][0]["first_seen_at"] = "2026-10-28T10:00:00+00:00"
        second = build_catalog(packet, "us")
        self.assertEqual([row["evidence_id"] for row in first["records"]],
                         [row["evidence_id"] for row in second["records"]])
        self.assertNotEqual(first["fingerprint"], second["fingerprint"])

    def test_unknown_cross_stock_cross_role_and_conflicting_refs_are_rejected(self):
        catalog = build_catalog(context(), "us")
        quote = next(row["evidence_id"] for row in catalog["records"] if row["kind"] == "market_snapshot")
        for refs, symbol, kinds in [(["made-up"], "AAPL", None), ([quote], "MSFT", None),
                                    ([quote], "AAPL", {"news", "fundamentals"})]:
            with self.subTest(symbol=symbol, kinds=kinds), self.assertRaises(ValueError):
                validate_links(links(refs), catalog, symbol, kinds)
        annotation = links([quote])
        annotation["contradicting_ids"] = [quote]
        with self.assertRaises(ValueError):
            validate_links(annotation, catalog, "AAPL")

    def test_insufficient_evidence_can_hold_but_cannot_trade_or_approve(self):
        catalog = build_catalog(context(), "us")
        result = {"decisions": [{"symbol": "AAPL", "action": "HOLD", "evidence": links()}],
                  "predictions": [{"symbol": "AAPL", "evidence": links()}]}
        validate_result_evidence(result, catalog, {"AAPL"})
        result["decisions"][0]["action"] = "BUY"
        with self.assertRaises(ValueError):
            validate_result_evidence(result, catalog, {"AAPL"})
        with self.assertRaises(ValueError):
            validate_result_evidence({"assessments": [{"symbol": "AAPL", "action": "BUY", "verdict": "APPROVE",
                                       "evidence": links()}]}, catalog, {"AAPL"}, risk=True)

    def test_synthetic_data_never_becomes_real_evidence(self):
        packet = context()
        packet["snapshot"][0]["data_source"] = "mock(synthetic fixture)"
        live = build_catalog(packet, "us")
        offline = build_catalog(packet, "us", allow_synthetic=True)
        self.assertFalse(next(row for row in live["records"] if row["kind"] == "market_snapshot")["eligible"])
        self.assertTrue(next(row for row in offline["records"] if row["kind"] == "market_snapshot")["eligible"])
        self.assertEqual(next(row for row in offline["records"] if row["kind"] == "market_snapshot")["data_quality"], "synthetic")

    def test_news_storage_returns_first_seen_hash_and_cutoff_checked_lookup(self):
        event = context()["news"][0]["top_events"][0]
        with subscription_workspace() as root:
            db = root / "news.db"
            ingest_events([event], "us", db_path=db)
            first = query_events("us", "AAPL", "2026-10-26", db_path=db)[0]
            ingest_events([event], "us", db_path=db)
            second = get_stored_event("event-1", "us", "2026-10-26", db)
            self.assertEqual(first["first_seen_at"], second["first_seen_at"])
            self.assertEqual(len(second["content_hash"]), 64)
            self.assertIsNone(get_stored_event("event-1", "us", "2026-10-25", db))
            modified = deepcopy(event)
            modified["summary"] = "changed"
            with self.assertRaises(ValueError):
                ingest_events([modified], "us", db_path=db)
