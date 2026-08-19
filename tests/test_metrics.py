import unittest

import pandas as pd

from evals.metrics import actual_direction_from_frame, prediction_start_date, uses_synthetic_data


class MetricsTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame({
            "date": [f"2026-01-{day:02d}" for day in range(1, 8)],
            "close": [100, 101, 102, 103, 104, 106, 99],
        })

    def test_requires_full_trading_day_horizon(self):
        self.assertEqual(
            actual_direction_from_frame(self.frame, "2026-01-01", 5), "up")
        self.assertIsNone(
            actual_direction_from_frame(self.frame, "2026-01-03", 5))

    def test_flat_threshold_is_symmetric(self):
        frame = pd.DataFrame({"date": ["a", "b"], "close": [100, 100.5]})
        self.assertEqual(actual_direction_from_frame(frame, "a", 1), "flat")

    def test_as_of_date_takes_precedence(self):
        self.assertEqual(prediction_start_date({
            "date": "2026-01-02", "as_of_date": "2026-01-01"
        }), "2026-01-01")

    def test_invalid_horizon_is_rejected(self):
        with self.assertRaises(ValueError):
            actual_direction_from_frame(self.frame, "2026-01-01", 0)

    def test_synthetic_source_is_preserved_for_scoring(self):
        self.assertTrue(uses_synthetic_data({"offline": True}))
        self.assertTrue(uses_synthetic_data({"data_sources": ["mock(离线模拟数据)"]}))
        self.assertFalse(uses_synthetic_data({"data_sources": ["akshare(真实数据)"]}))


if __name__ == "__main__":
    unittest.main()
