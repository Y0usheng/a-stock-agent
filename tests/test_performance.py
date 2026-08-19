import unittest
import sqlite3
import tempfile
from pathlib import Path

import pandas as pd

from evals.performance import (equal_weight_curve, indexed_curve, load_valuation_snapshots,
                               performance_metrics, performance_report,
                               time_varying_equal_weight_curve)


class PerformanceTests(unittest.TestCase):
    def test_return_drawdown_and_small_sample_guard(self):
        points = [
            {"date": "2026-01-01", "total": 100},
            {"date": "2026-01-02", "total": 110},
            {"date": "2026-01-03", "total": 88},
            {"date": "2026-01-04", "total": 105},
        ]
        metrics = performance_metrics(points, initial_value=100)
        self.assertEqual(metrics["total_return_pct"], 5.0)
        self.assertEqual(metrics["max_drawdown_pct"], -20.0)
        self.assertIsNone(metrics["annualized_return_pct"])
        self.assertIsNotNone(metrics["annualized_volatility_pct"])

    def test_equal_weight_curve_uses_asof_prices(self):
        frames = {
            "A": pd.DataFrame({"date": ["2026-01-01", "2026-01-03"], "close": [100, 110]}),
            "B": pd.DataFrame({"date": ["2026-01-01", "2026-01-02"], "close": [200, 180]}),
        }
        curve = equal_weight_curve(frames, ["2026-01-01", "2026-01-02", "2026-01-03"])
        self.assertAlmostEqual(curve[1]["total"], 0.95)
        self.assertAlmostEqual(curve[2]["total"], 1.0)

    def test_indexed_curve_exposes_change_return_and_drawdown(self):
        curve = indexed_curve([
            {"date": "2026-01-01", "total": 100},
            {"date": "2026-01-02", "total": 110},
            {"date": "2026-01-03", "total": 88},
        ], initial_value=100)
        self.assertEqual(curve[1]["index"], 110.0)
        self.assertEqual(curve[1]["period_change"], 10.0)
        self.assertEqual(curve[2]["period_return_pct"], -20.0)
        self.assertEqual(curve[2]["drawdown_pct"], -20.0)

    def test_future_constituent_cannot_change_earlier_benchmark_return(self):
        frames = {
            "A": pd.DataFrame({"date": ["2026-01-01", "2026-01-02", "2026-01-03"],
                               "close": [100, 110, 110]}),
            "FUTURE": pd.DataFrame({"date": ["2026-01-01", "2026-01-02", "2026-01-03"],
                                    "close": [1, 1000, 1000]}),
        }
        schedule = [
            {"date": "2026-01-01", "symbols": ["A"]},
            {"date": "2026-01-02", "symbols": ["A", "FUTURE"]},
            {"date": "2026-01-03", "symbols": ["A", "FUTURE"]},
        ]
        curve = time_varying_equal_weight_curve(frames, schedule)
        self.assertAlmostEqual(curve[1]["total"], 1.1)
        self.assertAlmostEqual(curve[2]["total"], 1.1)

    def test_loads_durable_valuation_snapshots(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "account.db"
            conn = sqlite3.connect(path)
            conn.execute("CREATE TABLE valuation_snapshots (snapshot_date TEXT, run_id TEXT, cash REAL, total_assets REAL, positions_json TEXT, prices_json TEXT, data_sources_json TEXT)")
            conn.execute("INSERT INTO valuation_snapshots VALUES (?,?,?,?,?,?,?)",
                         ("2026-01-02", "run", 50, 110, "[]", '{\"A\":110}', '[\"mock\"]'))
            conn.commit()
            conn.close()
            rows = load_valuation_snapshots(path)
        self.assertEqual(rows[0]["total_assets"], 110)
        self.assertEqual(rows[0]["prices"]["A"], 110)

    def test_report_keeps_synthetic_benchmark_same_source(self):
        frame = pd.DataFrame({"date": ["2026-01-01", "2026-01-02"], "close": [100, 105]})

        class Data:
            @staticmethod
            def _mock_history(_symbol, _days):
                return frame

            @staticmethod
            def _fetch_history(_symbol, _days):
                raise AssertionError("不应为离线日志拉取真实行情")

        logs = [
            {"market": "us", "date": "2026-01-01", "offline": True,
             "account_after": {"total_assets": 100, "initial_cash": 100},
             "decisions": []},
            {"market": "us", "date": "2026-01-02", "offline": True,
             "account_after": {"total_assets": 110, "initial_cash": 100},
             "decisions": [{"action": "BUY", "shares": 1, "price": 100}]},
        ]
        report = performance_report(logs, Data, [{"symbol": "A"}, {"symbol": "B"}])
        self.assertEqual(report["portfolio"]["total_return_pct"], 10.0)
        self.assertEqual(report["benchmark"]["total_return_pct"], 5.0)
        self.assertEqual(report["excess_return_pct"], 5.0)
        self.assertEqual(report["portfolio_curve"][-1]["index"], 110.0)
        self.assertEqual(report["benchmark_curve"][-1]["index"], 105.0)
        self.assertTrue(report["data_quality"]["synthetic"])


if __name__ == "__main__":
    unittest.main()
