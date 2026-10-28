import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

from agent.analysis import AnalysisHost, shadow_log_path, write_log
from agent.contracts import capture_risk_review
from agent.run_daily import run_shadow
from agent.sdk_runner import build_prompt_manifest


class Host:
    market = "us"
    active_symbols = ["AAPL"]
    news_screen = {"as_of_date": "2026-09-01", "candidates": []}

    def __init__(self):
        self.calls = []
        self.account = {"cash": 1000.0, "total_assets": 1000.0,
                        "initial_cash": 1000.0, "positions": []}
        self.fundamentals = [{"symbol": "AAPL", "roe_pct": 20.0}]

    async def call(self, name, args):
        self.calls.append((name, deepcopy(args)))
        if name == "get_account": return deepcopy(self.account)
        if name == "get_trades": return []
        if name == "candidate_fundamentals": return deepcopy(self.fundamentals)
        if name == "get_history": return [{"date": "2026-09-01", "close": 100.0}]
        raise AssertionError(f"unexpected tool {name}")


SNAPSHOT = [{"symbol": "AAPL", "name": "Apple", "date": "2026-09-01",
             "close": 100.0, "volume": 1000000, "data_source": "test-source"}]


class AnalysisTests(unittest.IsolatedAsyncioTestCase):
    async def test_history_is_cut_off_at_frozen_market_date_and_saved_in_dataset(self):
        host = Host()
        reader = AnalysisHost(host, SNAPSHOT, host.fundamentals, host.account, [])
        with patch.object(host, "call", return_value=[{"date": "2026-09-01", "close": 100},
                                                     {"date": "2026-09-02", "close": 200}]):
            rows = await reader.call("get_history", {"symbol": "AAPL", "days": 60})
        self.assertEqual([row["date"] for row in rows], ["2026-09-01"])
        dataset = reader.dataset("2026-09-01", {})
        self.assertEqual(dataset["tool_data"][0]["result"], rows)

    async def test_reader_blocks_all_mutations_and_returns_frozen_copies(self):
        host = Host()
        reader = AnalysisHost(host, SNAPSHOT, host.fundamentals, host.account, [])
        for tool in ("buy", "sell", "authorize_run", "record_valuation_snapshot"):
            with self.subTest(tool=tool), self.assertRaises(ValueError):
                await reader.call(tool, {})
        first = await reader.call("get_account", {})
        first["cash"] = 0
        self.assertEqual((await reader.call("get_account", {}))["cash"], 1000)
        for _ in range(2): await reader.call("get_history", {"symbol": "AAPL", "days": 60})
        self.assertEqual(len(host.calls), 1)
        with self.assertRaises(ValueError):
            await reader.call("get_history", {"symbol": "META"})

    async def test_same_day_prompt_variants_share_dataset_and_preserve_both_logs(self):
        host = Host()

        async def fake_sdk(reader, market, day, memory, manager_provider, risk_provider, version):
            if version == "manager-v2":
                await reader.call("get_history", {"symbol": "AAPL", "days": 60})
            decisions = [{"symbol": "AAPL", "name": "Apple", "action": "HOLD",
                          "shares": 0, "price": 100.0, "reason": version}]
            review = {"summary": "hold", "assessments": [{"symbol": "AAPL", "action": "HOLD",
                      "verdict": "APPROVE", "max_shares": 0, "reason": "hold"}]}
            return {"summary": version, "analyst_views": {"technical": "t", "fundamental": "f", "risk": "r"},
                    "memory_assessment": "none", "decisions": decisions,
                    "predictions": [{"symbol": "AAPL", "direction": "flat", "horizon_days": 5,
                                     "confidence": 0.5, "rationale": "test"}],
                    "risk_review": capture_risk_review(decisions, review, market, {"AAPL"}),
                    "agent_sdk": {"prompt": build_prompt_manifest(market, day, ["AAPL"], version)}}

        with tempfile.TemporaryDirectory() as temp, \
             patch("agent.run_daily.LOG_DIR", Path(temp)), \
             patch("agent.sdk_runner.run_agent_sdk", side_effect=fake_sdk) as sdk, \
             patch("builtins.print"):
            versions = ["manager-v1", "manager-v2"]
            await run_shadow(host, SNAPSHOT, "2026-09-01", False, None, None, versions)
            paths = sorted((Path(temp) / "shadow").glob("*.us.json"))
            self.assertEqual(len(paths), 2)
            logs = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
            self.assertEqual(len({log["dataset_fingerprint"] for log in logs}), 1)
            self.assertTrue(all(log["input_dataset"]["tool_data"] for log in logs))
            self.assertEqual({log["summary"] for log in logs}, set(versions))
            for log in logs:
                self.assertNotIn("account_after", log)
                self.assertEqual(log["execution"]["orders_submitted"], 0)
            self.assertEqual(list(Path(temp).glob("*.us.json")), [])
            await run_shadow(host, SNAPSHOT, "2026-09-01", False, None, None, versions)
            self.assertEqual(sdk.call_count, 2)
            self.assertFalse(any(name in {"buy", "sell", "authorize_run", "record_valuation_snapshot"}
                                 for name, _ in host.calls))


class AnalysisLogTests(unittest.TestCase):
    def test_filename_identity_ignores_rendered_prompt_hash_but_keeps_models(self):
        meta = {"strategy_version": "x", "execution_mode": "shadow", "agent_sdk": {
            "prompt": {"version": "manager-v1", "sha256": "day1", "template_sha256": "template"},
            "role_models": {"manager": {"model": "first"}}}}
        first = shadow_log_path(Path("logs"), "2026-09-01", "us", meta)
        meta["agent_sdk"]["prompt"]["sha256"] = "day2"
        self.assertEqual(first, shadow_log_path(Path("logs"), "2026-09-01", "us", meta))
        meta["agent_sdk"]["role_models"]["manager"]["model"] = "second"
        self.assertNotEqual(first, shadow_log_path(Path("logs"), "2026-09-01", "us", meta))

    def test_failed_json_write_preserves_previous_log(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "decision.json"
            write_log(path, {"version": "valid"})
            with self.assertRaises(ValueError):
                write_log(path, {"value": float("nan")})
            self.assertEqual(json.loads(path.read_text()), {"version": "valid"})
            self.assertEqual(list(Path(temp).glob("*.tmp")), [])
