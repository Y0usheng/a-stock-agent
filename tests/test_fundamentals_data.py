import unittest

from servers.fundamentals_data import mock_fundamentals


class FundamentalsDataTests(unittest.TestCase):
    def test_offline_fundamentals_are_labeled(self):
        data = mock_fundamentals("AAPL")
        self.assertEqual(data["symbol"], "AAPL")
        self.assertIn("mock", data["data_source"])
        self.assertIsNotNone(data["roe_pct"])


if __name__ == "__main__":
    unittest.main()
