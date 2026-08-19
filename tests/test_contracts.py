import unittest

from agent.contracts import reconcile_executions, validate_decision_result
from agent.run_daily import execute_sdk_plan, run_mock
from config import MARKETS


def _valid_result():
    decisions = []
    predictions = []
    for item in MARKETS["us"]["watchlist"]:
        decisions.append({"symbol": item["symbol"], "name": item["name"], "action": "HOLD",
                          "shares": 0, "price": 100.0, "reason": "no edge"})
        predictions.append({"symbol": item["symbol"], "direction": "flat", "horizon_days": 5,
                            "confidence": 0.5, "rationale": "mixed"})
    return {"summary": "hold", "analyst_views": {"technical": "x", "fundamental": "y", "risk": "z"},
            "memory_assessment": "none", "decisions": decisions, "predictions": predictions}


class ContractTests(unittest.TestCase):
    def test_valid_complete_result(self):
        validate_decision_result(_valid_result(), "us")

    def test_incomplete_predictions_are_rejected(self):
        result = _valid_result()
        result["predictions"].pop()
        with self.assertRaises(ValueError):
            validate_decision_result(result, "us")

    def test_claimed_trade_must_exist_in_ledger(self):
        result = _valid_result()
        result["decisions"][0].update(action="BUY", shares=5)
        with self.assertRaises(ValueError):
            reconcile_executions(result, [])
        audit = reconcile_executions(result, [{"symbol": "AAPL", "action": "BUY", "shares": 5}])
        self.assertTrue(audit["ok"])

    def test_dynamic_expected_symbols_may_use_controlled_universe(self):
        result = _valid_result()
        result["decisions"] = [{"symbol": "META", "name": "Meta Platforms",
                                "action": "HOLD", "shares": 0, "price": 100.0,
                                "reason": "news candidate"}]
        result["predictions"] = [{"symbol": "META", "direction": "flat",
                                  "horizon_days": 5, "confidence": 0.5,
                                  "rationale": "mixed"}]
        validate_decision_result(result, "us", expected_symbols={"META"})


class ExecutionTests(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def _snapshot(market):
        return [{"symbol": item["symbol"], "name": item["name"], "close": 100.0,
                 "above_ma20": True, "macd_dif": 1.0, "macd_dea": 0.0,
                 "rsi14": 50.0}
                for item in MARKETS[market]["watchlist"]]

    async def test_mock_covers_watchlist_when_budget_is_below_one_lot(self):
        class Host:
            market = "cn"
            news_screen = {}

            async def call(self, name, _args):
                if name == "candidate_fundamentals":
                    return []
                if name == "get_account":
                    return {"cash": 10.0, "total_assets": 10.0, "positions": []}
                raise AssertionError(f"不应调用 {name}")

        result = await run_mock(Host(), {}, self._snapshot("cn"))
        validate_decision_result(result, "cn")
        self.assertEqual(len(result["decisions"]), len(MARKETS["cn"]["watchlist"]))
        self.assertTrue(all(item["action"] == "HOLD" and item["shares"] == 0
                            for item in result["decisions"]))

    async def test_mock_rejected_orders_become_zero_share_holds(self):
        class Host:
            market = "us"
            news_screen = {}

            async def call(self, name, _args):
                if name == "candidate_fundamentals":
                    return []
                if name == "get_account":
                    return {"cash": 1000.0, "total_assets": 1000.0, "positions": []}
                if name == "buy":
                    return {"ok": False, "error": "risk rejected"}
                raise AssertionError(name)

        result = await run_mock(Host(), {}, self._snapshot("us"))
        validate_decision_result(result, "us")
        self.assertTrue(all(item["action"] == "HOLD" and item["shares"] == 0
                            for item in result["decisions"]))

    async def test_host_locks_execution_price_and_records_rejection(self):
        result = _valid_result()
        result["decisions"][0].update(action="BUY", shares=5, price=1.0)

        class Host:
            async def call(self, _name, args):
                self.args = args
                return {"ok": False, "error": "risk rejected"}

        host = Host()
        errors = await execute_sdk_plan(host, result, {item["symbol"]: 123.0
                                                       for item in MARKETS["us"]["watchlist"]})
        self.assertEqual(host.args["price"], 123.0)
        self.assertEqual(result["decisions"][0]["action"], "HOLD")
        self.assertEqual(len(errors), 1)

    def test_fragmented_ledger_trades_are_aggregated(self):
        result = _valid_result()
        result["decisions"][0].update(action="BUY", shares=5)
        audit = reconcile_executions(result, [
            {"symbol": "AAPL", "action": "BUY", "shares": 2},
            {"symbol": "AAPL", "action": "BUY", "shares": 3},
        ])
        self.assertTrue(audit["ok"])


if __name__ == "__main__":
    unittest.main()
