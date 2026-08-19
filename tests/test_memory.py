import json
import tempfile
import unittest
from pathlib import Path

from agent.memory import build_memory_snapshot, memory_prompt


class MemoryTests(unittest.TestCase):
    def test_selection_is_past_only_compact_and_stable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for day, action in [("2026-01-01", "BUY"), ("2026-01-02", "HOLD"),
                                ("2026-01-04", "SELL")]:
                data = {
                    "market": "cn", "date": day,
                    "decisions": [{"symbol": "600519", "action": action,
                                   "shares": 100, "price": 10, "reason": "past reason"}],
                    "predictions": [{"symbol": "600519", "direction": "up",
                                     "confidence": 0.6, "horizon_days": 5}],
                }
                (root / f"{day}.cn.json").write_text(json.dumps(data), encoding="utf-8")
            trades = [
                {"trade_date": "2026-01-03", "symbol": "600519", "action": "BUY",
                 "shares": 100, "price": 11, "fee": 5, "reason": "old"},
                {"trade_date": "2026-01-04", "symbol": "600519", "action": "SELL",
                 "shares": 100, "price": 12, "fee": 5, "reason": "future"},
            ]
            first = build_memory_snapshot("cn", "2026-01-04", root, trades)
            second = build_memory_snapshot("cn", "2026-01-04", root, trades)
            self.assertEqual(first["selected_dates"], ["2026-01-01", "2026-01-02"])
            self.assertEqual(len(first["recent_non_hold_decisions"]), 1)
            self.assertEqual(len(first["recent_trades"]), 1)
            self.assertEqual(first["memory_id"], second["memory_id"])
            rendered = memory_prompt(first)
            self.assertIn("<MEMORY_JSON>", rendered)
            self.assertIn("只读历史记忆", rendered)


if __name__ == "__main__":
    unittest.main()
