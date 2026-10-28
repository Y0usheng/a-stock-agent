import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from agent.operations import monitored_markets, operation_summary, preflight, safe_error, market_lock


class OperationsTests(unittest.IsolatedAsyncioTestCase):
    async def test_failed_market_does_not_prevent_other_market_and_status_is_persisted(self):
        seen = []
        async def run(market, **kwargs):
            seen.append(market)
            if market == "cn":
                raise RuntimeError("source unavailable")
            return {"status": "complete", "log_path": "test.json"}
        with tempfile.TemporaryDirectory() as temp, patch("builtins.print"):
            db = Path(temp) / "ops.db"
            result = await monitored_markets(["cn", "us"], run, {}, db_path=db)
            persisted = operation_summary(db)
            self.assertEqual(seen, ["cn", "us"])
            self.assertEqual([item["status"] for item in result], ["failed", "complete"])
            self.assertEqual(persisted["latest"]["cn"]["status"], "failed")
            self.assertEqual(persisted["latest"]["us"]["log_path"], "test.json")

    def test_process_lock_blocks_duplicate_market_and_releases_after_exit(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with market_lock("cn", root):
                with self.assertRaises(RuntimeError):
                    with market_lock("cn", root):
                        self.fail("duplicate market acquired lock")
                with market_lock("us", root):
                    pass
            with market_lock("cn", root):
                pass

    def test_preflight_does_not_reveal_credentials_and_errors_are_redacted(self):
        key = "sk-test-secret-token"
        with patch.dict(os.environ, {"OPENAI_API_KEY": key}, clear=True):
            report = preflight(framework="langgraph", model_access="api")
            self.assertTrue(report["ok"])
            self.assertNotIn(key, str(report))
            self.assertNotIn(key, safe_error(RuntimeError("error " + key)))
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(preflight(model_access="api")["ok"])
            self.assertTrue(preflight(mock=True)["ok"])
