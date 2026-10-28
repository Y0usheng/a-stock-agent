from datetime import datetime, timezone
from pathlib import Path
import tempfile
import unittest
import pandas as pd

from evals.price_data import load_price_dataset, export_price_dataset, normalize_history
from servers.data_quality import completed_history, snapshot_date


class PriceDataTests(unittest.TestCase):
    def test_frozen_price_dataset_replays_with_identical_content_fingerprint(self):
        frames, manifest = load_price_dataset("us", end_date="2026-09-01", window=60)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "prices.json"
            export_price_dataset(path, "us", frames, manifest)
            replay, other = load_price_dataset("us", "file", end_date="2026-09-01", path=path, window=60)
            self.assertEqual(manifest["fingerprint"], other["fingerprint"])
            self.assertEqual(other["data_quality"], "replayed_prices")
            for symbol in frames:
                pd.testing.assert_frame_equal(frames[symbol], replay[symbol])
            with self.assertRaises(ValueError):
                load_price_dataset("cn", "file", path=path, window=60)

    def test_csv_keeps_cn_leading_zero_codes_and_clips_future_prices(self):
        frames, _ = load_price_dataset("cn", end_date="2026-09-02", window=60)
        frame = frames["002594"].copy()
        frame["symbol"] = "002594"
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "cn.csv"
            frame.to_csv(path, index=False)
            loaded, manifest = load_price_dataset("cn", "file", end_date="2026-09-01", path=path, window=60)
        self.assertEqual(list(loaded), ["002594"])
        self.assertEqual(manifest["end_date"], "2026-09-01")
        self.assertEqual(manifest["data_quality"], "imported_prices_unverified")

    def test_changed_price_changes_dataset_and_invalid_bars_are_rejected(self):
        frames, manifest = load_price_dataset("us", end_date="2026-09-01", window=60)
        frames["AAPL"].loc[0, "open"] *= 1.1
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "prices.json"
            export_price_dataset(path, "us", frames, manifest)
            with self.assertRaisesRegex(ValueError, "指纹不符"):
                load_price_dataset("us", "file", end_date="2026-09-01", path=path, window=60)
        for value in (-1, float("nan"), float("inf")):
            bad = frames["AAPL"].copy()
            bad.loc[0, "close"] = value
            with self.assertRaises(ValueError):
                normalize_history("AAPL", bad)
        with self.assertRaises(ValueError):
            normalize_history("AAPL", pd.concat([frames["AAPL"], frames["AAPL"].head(1)]))

    def test_completed_bars_respect_us_timezone_and_daylight_saving(self):
        frame = pd.DataFrame({"date": ["2026-09-01", "2026-09-02"]})
        before_close = datetime(2026, 9, 2, 19, 59, tzinfo=timezone.utc)
        after_close = datetime(2026, 9, 2, 20, 1, tzinfo=timezone.utc)
        self.assertEqual(completed_history(frame, "us", before_close)["date"].tolist(), ["2026-09-01"])
        self.assertEqual(len(completed_history(frame, "us", after_close)), 2)
        with self.assertRaises(RuntimeError):
            snapshot_date([{"date": "2026-09-01"}, {"date": "2026-09-02"}])
