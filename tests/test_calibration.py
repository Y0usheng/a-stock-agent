import unittest

from evals.calibration import calibration_metrics, chronological_calibration


class CalibrationTests(unittest.TestCase):
    def test_brier_penalizes_confident_errors(self):
        careful = calibration_metrics([
            {"confidence": 0.9, "correct": True},
            {"confidence": 0.1, "correct": False},
        ])
        wrong = calibration_metrics([
            {"confidence": 0.9, "correct": False},
            {"confidence": 0.1, "correct": True},
        ])
        self.assertLess(careful["brier_score"], wrong["brier_score"])

    def test_calibration_uses_strictly_earlier_dates_for_training(self):
        rows = []
        for day, correct in (("2026-01-01", True), ("2026-01-02", False),
                             ("2026-01-03", True), ("2026-01-04", False)):
            rows.extend({"date": day, "symbol": str(i), "confidence": 0.7,
                         "correct": correct} for i in range(2))
        report = chronological_calibration(rows, train_fraction=0.5)
        self.assertEqual(report["status"], "complete")
        self.assertLess(report["train_end"], report["test_start"])
        self.assertEqual(report["train_count"] + report["test_count"], 8)

    def test_small_samples_are_labeled_not_overclaimed(self):
        report = chronological_calibration([
            {"date": "2026-01-01", "confidence": 0.6, "correct": True}
        ])
        self.assertEqual(report["status"], "insufficient_data")


if __name__ == "__main__":
    unittest.main()
