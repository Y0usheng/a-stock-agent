import unittest

import pandas as pd

from servers.synthetic_data import _seed_for_symbol, random_walk_history


class SyntheticDataTests(unittest.TestCase):
    def test_seed_is_stable_and_symbol_specific(self):
        self.assertEqual(_seed_for_symbol("AAPL"), _seed_for_symbol("AAPL"))
        self.assertNotEqual(_seed_for_symbol("AAPL"), _seed_for_symbol("MSFT"))

    def test_history_is_reproducible(self):
        args = dict(symbol="AAPL", days=20, base_prices={"AAPL": 200},
                    drift=0.0005, volatility=0.02,
                    volume_low=100, volume_high=1000, high_low_sigma=0.009)
        first = random_walk_history(**args)
        second = random_walk_history(**args)
        pd.testing.assert_frame_equal(first, second)

    def test_overlapping_dates_do_not_depend_on_query_window(self):
        args = dict(symbol="AAPL", base_prices={"AAPL": 200},
                    drift=0.0005, volatility=0.02,
                    volume_low=100, volume_high=1000, high_low_sigma=0.009)
        short = random_walk_history(days=20, **args)
        long = random_walk_history(days=200, **args)
        pd.testing.assert_frame_equal(short, long.tail(20).reset_index(drop=True))

    def test_later_end_date_does_not_rewrite_old_ohlcv(self):
        args = dict(symbol="AAPL", days=100, base_prices={"AAPL": 200},
                    drift=.0005, volatility=.02, volume_low=100, volume_high=1000, high_low_sigma=.009)
        first = random_walk_history(end_date="2026-09-01", **args)
        later = random_walk_history(end_date="2026-09-02", **args)
        overlap = sorted(set(first["date"]) & set(later["date"]))
        pd.testing.assert_frame_equal(first[first["date"].isin(overlap)].reset_index(drop=True),
                                      later[later["date"].isin(overlap)].reset_index(drop=True))


if __name__ == "__main__":
    unittest.main()
