import unittest
from unittest.mock import patch

from config import MARKETS
from servers import news_server


class NewsServerTests(unittest.TestCase):
    def setUp(self):
        news_server._cache.clear()

    def test_lexical_score_distinguishes_positive_and_negative_events(self):
        positive, _ = news_server._score_text("公司宣布回购并上调业绩预期")
        negative, _ = news_server._score_text("公司受到处罚并提示退市风险")
        self.assertGreater(positive, 0)
        self.assertLess(negative, 0)

    def test_offline_candidate_screen_is_reproducible_and_bounded(self):
        with patch.object(news_server, "FORCE_OFFLINE", True):
            first = news_server.screen_news_candidates("2026-08-19", limit=5, days=3)
            news_server._cache.clear()
            second = news_server.screen_news_candidates("2026-08-19", limit=5, days=3)
        self.assertEqual(first, second)
        self.assertEqual(len(first["candidates"]), 5)
        allowed = {item["symbol"] for item in MARKETS["cn"]["candidate_universe"]}
        self.assertTrue({item["symbol"] for item in first["candidates"]} <= allowed)
        self.assertTrue(all(item["top_events"][0]["untrusted_text"]
                            for item in first["candidates"]))

    def test_news_window_rejects_future_events(self):
        current = {"published_at": "2026-08-19T09:00:00"}
        future = {"published_at": "2026-08-20T09:00:00"}
        self.assertTrue(news_server._within_window(current, "2026-08-19", 3))
        self.assertFalse(news_server._within_window(future, "2026-08-19", 3))

    def test_live_provider_failure_does_not_fall_back_to_mock_news(self):
        with patch.object(news_server, "FORCE_OFFLINE", False), \
             patch.object(news_server, "_cn_live_news", return_value=[]):
            result = news_server.screen_news_candidates("2026-08-19", limit=5, days=3)
        self.assertFalse(result["ok"])
        self.assertEqual(result["data_sources"], [])
        self.assertTrue(all(item["article_count"] == 0 for item in result["candidates"]))


if __name__ == "__main__":
    unittest.main()
