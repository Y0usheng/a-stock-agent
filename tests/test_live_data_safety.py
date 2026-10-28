import os
import unittest
from unittest.mock import patch

from agent.run_daily import _assert_live_data
from servers import market_server, us_market_server


class LiveDataSafetyTests(unittest.TestCase):
    def test_cn_live_market_failure_raises_instead_of_mocking(self):
        market_server._cache.clear()
        with patch.object(market_server, "_FORCE_OFFLINE", False), \
             patch.object(market_server, "_HAS_AKSHARE", False):
            with self.assertRaises(RuntimeError):
                market_server._fetch_history("600519", 30)

    def test_us_live_market_failure_raises_instead_of_mocking(self):
        us_market_server._cache.clear()
        with patch.object(us_market_server, "_FORCE_OFFLINE", False), \
             patch.object(us_market_server, "_HAS_YF", False):
            with self.assertRaises(RuntimeError):
                us_market_server._fetch_history("AAPL", 30)

    def test_explicit_offline_mode_still_allows_reproducible_data(self):
        market_server._cache.clear()
        with patch.object(market_server, "_FORCE_OFFLINE", True):
            first = market_server._fetch_history("600519", 30)
            market_server._cache.clear()
            second = market_server._fetch_history("600519", 30)
        self.assertTrue(first.equals(second))

    def test_host_rejects_synthetic_snapshot_in_live_mode(self):
        snapshot = [{"symbol": "AAPL", "data_source": "mock(模拟数据)"}]
        with patch.dict(os.environ, {"ASTOCK_OFFLINE": ""}):
            with self.assertRaises(RuntimeError):
                _assert_live_data(snapshot, {"ok": True})

    def test_host_accepts_synthetic_snapshot_only_offline(self):
        snapshot = [{"symbol": "AAPL", "data_source": "mock(模拟数据)"}]
        with patch.dict(os.environ, {"ASTOCK_OFFLINE": "1"}):
            _assert_live_data(snapshot, {"ok": True})

    def test_live_dates_must_be_same_fresh_and_prices_finite(self):
        from datetime import date, timedelta
        snapshot = [{"symbol": "AAPL", "date": date.today().isoformat(),
                     "close": 100., "volume": 1000, "data_source": "yfinance"}]
        with patch.dict(os.environ, {"ASTOCK_OFFLINE": ""}):
            _assert_live_data(snapshot, {"ok": True})
            snapshot[0]["close"] = float("inf")
            with self.assertRaises(RuntimeError):
                _assert_live_data(snapshot, {"ok": True})
            snapshot[0]["close"] = 100.
            snapshot[0]["date"] = (date.today() - timedelta(days=15)).isoformat()
            with self.assertRaises(RuntimeError):
                _assert_live_data(snapshot, {"ok": True})


if __name__ == "__main__":
    unittest.main()
