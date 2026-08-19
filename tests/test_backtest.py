import math
import unittest

import pandas as pd

from evals.backtest import news_ab_eval, walk_forward_backtest


def _history(days=180):
    dates = pd.bdate_range("2025-01-01", periods=days).strftime("%Y-%m-%d")
    closes = [100 + index * 0.08 + math.sin(index / 3) * 3 for index in range(days)]
    return pd.DataFrame({
        "date": dates,
        "open": [value * 1.001 for value in closes],
        "high": [value * 1.01 for value in closes],
        "low": [value * 0.99 for value in closes],
        "close": closes,
        "volume": [2_000_000] * days,
    })


class WalkForwardBacktestTests(unittest.TestCase):
    def test_window_must_be_60_to_120_days(self):
        with self.assertRaises(ValueError):
            walk_forward_backtest({"AAPL": _history()}, "us", 59)
        with self.assertRaises(ValueError):
            walk_forward_backtest({"AAPL": _history()}, "us", 121)

    def test_orders_execute_on_next_open_with_slippage(self):
        frame = _history()
        result = walk_forward_backtest({"AAPL": frame}, "us", 90, use_news=False)
        self.assertTrue(result["trades"])
        indexed = frame.set_index("date")
        for trade in result["trades"]:
            self.assertGreater(trade["trade_date"], trade["signal_date"])
            market_open = float(indexed.loc[trade["trade_date"]]["open"])
            factor = 0.9997 if trade["action"] == "SELL" else 1.0003
            self.assertAlmostEqual(trade["price"], round(market_open * factor, 4))

    def test_future_price_change_cannot_change_past_decisions(self):
        original = _history()
        changed = original.copy()
        changed.loc[changed.index[-1], "close"] *= 10
        first = walk_forward_backtest({"AAPL": original}, "us", 90, use_news=False)
        second = walk_forward_backtest({"AAPL": changed}, "us", 90, use_news=False)
        self.assertEqual(first["decisions"], second["decisions"])

    def test_news_ab_returns_comparable_reports(self):
        report = news_ab_eval({"AAPL": _history()}, "us", 60)
        self.assertEqual(report["data_quality"], "synthetic_news")
        self.assertIn("with_news", report)
        self.assertIn("without_news", report)
        self.assertIsNotNone(report["return_delta_pct"])


if __name__ == "__main__":
    unittest.main()
