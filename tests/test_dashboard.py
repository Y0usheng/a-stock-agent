import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

import dashboard.app as dashboard


class DashboardOverviewTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.db = root / "account_cn.db"
        self.logs = root / "logs"
        self.logs.mkdir()
        conn = sqlite3.connect(self.db)
        conn.executescript("""
        CREATE TABLE meta (key TEXT PRIMARY KEY, value REAL);
        CREATE TABLE positions (symbol TEXT PRIMARY KEY, name TEXT, shares INTEGER,
                                avg_cost REAL, last_buy_date TEXT);
        CREATE TABLE trades (id INTEGER PRIMARY KEY, trade_date TEXT, symbol TEXT,
                             name TEXT, action TEXT, shares INTEGER, price REAL,
                             fee REAL, reason TEXT);
        INSERT INTO meta VALUES ('cash', 900);
        INSERT INTO positions VALUES ('AAA', '测试股票', 10, 10, '2026-02-01');
        """)
        conn.commit()
        conn.close()
        self.old_markets = dashboard.MARKETS
        self.old_log_dir = dashboard.LOG_DIR
        self.old_experiment_db = dashboard.EXPERIMENT_DB
        self.old_news_db = dashboard.NEWS_DB
        dashboard.MARKETS = {"cn": {
            "name": "A股", "currency": "CNY", "initial_cash": 1000.0,
            "db": self.db, "watchlist": [{"symbol": "AAA", "name": "测试股票"}],
        }}
        dashboard.LOG_DIR = self.logs
        dashboard.EXPERIMENT_DB = root / "experiments.db"
        dashboard.NEWS_DB = root / "news.db"

    def tearDown(self):
        dashboard.MARKETS = self.old_markets
        dashboard.LOG_DIR = self.old_log_dir
        dashboard.EXPERIMENT_DB = self.old_experiment_db
        dashboard.NEWS_DB = self.old_news_db
        self.temp.cleanup()

    def _write_log(self, day, price):
        data = {
            "market": "cn", "date": day, "as_of_date": day,
            "decisions": [{"symbol": "AAA", "name": "测试股票", "action": "HOLD",
                           "shares": 0, "price": price, "reason": "test"}],
            "account_after": {"total_assets": 1000.0, "initial_cash": 1000.0},
        }
        (self.logs / f"{day}.cn.json").write_text(
            json.dumps(data, ensure_ascii=False), encoding="utf-8")

    def test_price_older_than_position_falls_back_to_cost(self):
        self._write_log("2026-01-01", 1.0)
        payload = dashboard.app.test_client().get("/api/overview").get_json()["cn"]
        self.assertEqual(payload["positions"][0]["last_price"], 10.0)
        self.assertEqual(payload["total_assets"], 1000.0)
        self.assertIn("AAA", payload["data_warning"])

    def test_price_after_position_is_used_for_current_valuation(self):
        self._write_log("2026-03-01", 12.0)
        payload = dashboard.app.test_client().get("/api/overview").get_json()["cn"]
        self.assertEqual(payload["positions"][0]["last_price"], 12.0)
        self.assertEqual(payload["positions"][0]["pnl_amount"], 20.0)
        self.assertIsNone(payload["data_warning"])

    def test_experiment_endpoint_is_empty_but_well_formed_before_first_run(self):
        payload = dashboard.app.test_client().get("/api/experiments").get_json()
        self.assertEqual(payload["runs"], [])
        self.assertEqual(payload["comparisons"], [])
        self.assertEqual(payload["news_dataset"]["event_count"], 0)


if __name__ == "__main__":
    unittest.main()
