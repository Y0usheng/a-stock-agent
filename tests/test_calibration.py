import unittest

from evals.calibration import calibration_metrics, chronological_calibration, variant_calibration


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

    def test_different_model_samples_are_never_pooled_for_calibration(self):
        rows = [{"date": day, "market": "us", "symbol": str(index),
                 "confidence": 0.6, "correct": True, "variant": {"model": model}}
                for model in ("first", "second")
                for day in ("2026-01-01", "2026-01-02") for index in range(2)]
        report = variant_calibration(rows)
        self.assertEqual(report["calibration"]["status"], "multiple_variants")
        self.assertEqual(len(report["calibration_by_variant"]), 2)
        self.assertTrue(all(entry["calibration"]["status"] == "insufficient_data"
                            for entry in report["calibration_by_variant"]))

    def test_single_variant_keeps_existing_summary_shape(self):
        report = variant_calibration([{"date": "2026-01-01", "market": "us",
                                      "confidence": 0.6, "correct": True}])
        self.assertEqual(report["calibration"]["status"], "insufficient_data")
        self.assertEqual(len(report["calibration_by_variant"]), 1)


if __name__ == "__main__":
    unittest.main()
