import unittest
from copy import deepcopy

from agent.contracts import (capture_risk_review, reconcile_executions,
                             validate_decision_result, validate_risk_approval)
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


def _approve(result, verdict="APPROVE", maximum=None):
    proposed = deepcopy(result["decisions"])
    assessments = [{"symbol": item["symbol"], "action": item["action"],
                    "verdict": verdict if item["action"] != "HOLD" else "APPROVE",
                    "max_shares": (maximum if maximum is not None else item["shares"])
                                  if item["action"] != "HOLD" else 0,
                    "reason": "reviewed"} for item in proposed]
    result["risk_review"] = capture_risk_review(
        proposed, {"summary": "reviewed", "assessments": assessments},
        "us", {item["symbol"] for item in proposed})


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

    def test_duplicate_or_non_object_rows_are_rejected(self):
        for key in ("decisions", "predictions"):
            for extra in ({"symbol": "AAPL"}, None):
                with self.subTest(key=key, extra=extra):
                    result = _valid_result()
                    result[key].append(extra)
                    with self.assertRaises(ValueError):
                        validate_decision_result(result, "us")


class RiskApprovalTests(unittest.TestCase):
    def setUp(self):
        self.result = _valid_result()
        self.result["decisions"][0].update(action="BUY", shares=10)
        self.symbols = {item["symbol"] for item in self.result["decisions"]}

    def test_missing_approval_is_rejected_even_when_text_says_approve(self):
        self.result["analyst_views"]["risk"] = "APPROVE"
        with self.assertRaises(ValueError):
            validate_risk_approval(self.result, "us", self.symbols)

    def test_rejected_trade_and_unreviewed_direction_are_blocked(self):
        _approve(self.result, "REJECT", 0)
        with self.assertRaises(ValueError):
            validate_risk_approval(self.result, "us", self.symbols)
        _approve(self.result)
        self.result["decisions"][0]["action"] = "SELL"
        with self.assertRaises(ValueError):
            validate_risk_approval(self.result, "us", self.symbols)

    def test_resize_is_binding_and_manager_can_choose_hold(self):
        _approve(self.result, "RESIZE", 5)
        with self.assertRaises(ValueError):
            validate_risk_approval(self.result, "us", self.symbols)
        self.result["decisions"][0]["shares"] = 5
        validate_risk_approval(self.result, "us", self.symbols)
        self.result["decisions"][0].update(action="HOLD", shares=0)
        validate_risk_approval(self.result, "us", self.symbols)

    def test_approval_must_cover_exact_plan_with_valid_limits(self):
        _approve(self.result)
        valid = deepcopy(self.result["risk_review"])
        for change in ("missing", "duplicate", "oversize", "fractional", "wrong-action"):
            with self.subTest(change=change):
                review = deepcopy(valid)
                if change == "missing": review["assessments"].pop()
                if change == "duplicate": review["assessments"].append(review["assessments"][0])
                if change == "oversize": review["assessments"][0]["max_shares"] = 11
                if change == "fractional": review["assessments"][0]["max_shares"] = 1.5
                if change == "wrong-action": review["assessments"][0]["action"] = "SELL"
                with self.assertRaises(ValueError):
                    capture_risk_review(review["proposed_decisions"], review, "us", self.symbols)

    def test_cn_resize_obeys_lot_size(self):
        proposed = [{"symbol": "600519", "name": "贵州茅台", "action": "BUY",
                     "shares": 200, "price": 100.0, "reason": "test"}]
        review = {"summary": "resize", "assessments": [{"symbol": "600519", "action": "BUY",
                  "verdict": "RESIZE", "max_shares": 50, "reason": "test"}]}
        with self.assertRaises(ValueError):
            capture_risk_review(proposed, review, "cn", {"600519"})


class ExecutionTests(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def _snapshot(market):
        return [{"symbol": item["symbol"], "name": item["name"], "close": 100.0,
                 "above_ma20": True, "macd_dif": 1.0, "macd_dea": 0.0,
                 "rsi14": 50.0, "volume": 1_000_000}
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
        _approve(result)

        class Host:
            market = "us"
            async def call(self, _name, args):
                self.args = args
                return {"ok": False, "error": "risk rejected"}

        host = Host()
        errors = await execute_sdk_plan(host, result, {item["symbol"]: 123.0
                                                       for item in MARKETS["us"]["watchlist"]},
                                        {item["symbol"]: 1_000_000
                                         for item in MARKETS["us"]["watchlist"]})
        self.assertEqual(host.args["price"], 123.0)
        self.assertEqual(result["decisions"][0]["action"], "HOLD")
        self.assertEqual(len(errors), 1)

    async def test_whole_plan_is_rejected_before_any_order_is_submitted(self):
        result = _valid_result()
        result["decisions"][0].update(action="BUY", shares=5)
        result["decisions"][1].update(action="BUY", shares=5)
        _approve(result)
        result["decisions"][1]["shares"] = 6

        class Host:
            market = "us"
            calls = []

            async def call(self, name, args):
                self.calls.append(name)
                return {"ok": True}

        host = Host()
        prices = {item["symbol"]: 100.0 for item in result["decisions"]}
        with self.assertRaises(ValueError):
            await execute_sdk_plan(host, result, prices, {symbol: 1000000 for symbol in prices})
        self.assertEqual(host.calls, [])

    async def test_mock_shadow_never_submits_orders(self):
        class Host:
            market = "us"
            news_screen = {}

            async def call(self, name, args):
                if name == "candidate_fundamentals": return []
                if name == "get_account":
                    return {"cash": 1000.0, "total_assets": 1000.0, "positions": []}
                raise AssertionError(f"shadow 不应调用 {name}")

        result = await run_mock(Host(), {}, self._snapshot("us"), execute=False)
        validate_decision_result(result, "us")
        self.assertTrue(any(item["action"] == "BUY" for item in result["decisions"]))

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
