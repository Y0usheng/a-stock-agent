import tempfile
import unittest
from pathlib import Path

from evals.news_store import ingest_events, load_news_scores, query_events


class NewsStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.db = Path(self.temp.name) / "news.db"

    def tearDown(self):
        self.temp.cleanup()

    @staticmethod
    def event(event_id="e1", day="2026-08-10", score=1.0, title="业绩增长"):
        return {"event_id": event_id, "symbol": "AAPL", "published_at": day,
                "title": title, "source": "provider", "sentiment_score": score,
                "event_tags": ["earnings"]}

    def test_ingest_is_idempotent_and_history_is_immutable(self):
        first = ingest_events([self.event()], "us", "imported", self.db)
        second = ingest_events([self.event()], "us", "imported", self.db)
        self.assertEqual(first["inserted"], 1)
        self.assertEqual(second["duplicates"], 1)
        with self.assertRaises(ValueError):
            ingest_events([self.event(title="内容被改写")], "us", "imported", self.db)

    def test_query_rejects_future_events_at_database_boundary(self):
        ingest_events([self.event("past", "2026-08-09"),
                       self.event("future", "2026-08-11")], "us", "live", self.db)
        rows = query_events("us", "AAPL", "2026-08-10", 3, db_path=self.db)
        self.assertEqual([row["event_id"] for row in rows], ["past"])

    def test_dataset_fingerprint_and_daily_scores_are_stable(self):
        ingest_events([self.event("a", "2026-08-09", 1),
                       self.event("b", "2026-08-09", -0.25)], "us", "imported", self.db)
        scores1, manifest1 = load_news_scores("us", "2026-08-01", "2026-08-10", self.db)
        scores2, manifest2 = load_news_scores("us", "2026-08-01", "2026-08-10", self.db)
        self.assertEqual(scores1[("2026-08-09", "AAPL")], 0.75)
        self.assertEqual(scores1[("2026-08-10", "AAPL")], 0.75)
        self.assertEqual(manifest1["fingerprint"], manifest2["fingerprint"])
        self.assertEqual(manifest1["data_quality"], "historical_news")


if __name__ == "__main__":
    unittest.main()
