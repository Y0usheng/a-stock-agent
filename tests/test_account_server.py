import importlib
import os
import sqlite3
import sys
import tempfile
import types
import unittest
from datetime import date, timedelta
from pathlib import Path


class _FakeFastMCP:
    def __init__(self, *_args, **_kwargs):
        pass

    def tool(self):
        return lambda function: function

    def run(self):
        pass


def _load_account_server():
    # 单元测试账户规则无需启动 MCP 进程，以轻量 stub 隔离传输层。
    mcp_module = types.ModuleType("mcp")
    server_module = types.ModuleType("mcp.server")
    fastmcp_module = types.ModuleType("mcp.server.fastmcp")
    fastmcp_module.FastMCP = _FakeFastMCP
    names = ["mcp", "mcp.server", "mcp.server.fastmcp"]
    originals = {name: sys.modules.get(name) for name in names}
    try:
        sys.modules.update({
            "mcp": mcp_module,
            "mcp.server": server_module,
            "mcp.server.fastmcp": fastmcp_module,
        })
        os.environ["ASTOCK_MARKET"] = "cn"
        sys.modules.pop("servers.account_server", None)
        return importlib.import_module("servers.account_server")
    finally:
        for name, module in originals.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


account = _load_account_server()


class AccountRulesTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        account.DB_PATH = Path(self.tempdir.name) / "account.db"
        self.trade_date = "2026-08-19"
        self.order_number = 0
        approved = list(account.WATCHLIST)
        self.assertTrue(account.authorize_run("test-run", self.trade_date, approved)["ok"])

    def _buy(self, symbol, name, shares, price, reason, *, run_id="test-run",
             trade_date=None, order_id=None, current_prices=None):
        self.order_number += 1
        return account.buy(
            symbol, name, shares, price, reason, run_id,
            order_id or f"{run_id}:order-{self.order_number}",
            trade_date or self.trade_date, current_prices,
        )

    def _sell(self, symbol, shares, price, reason, *, run_id="test-run",
              trade_date=None, order_id=None):
        self.order_number += 1
        return account.sell(
            symbol, shares, price, reason, run_id,
            order_id or f"{run_id}:order-{self.order_number}",
            trade_date or self.trade_date,
        )

    def tearDown(self):
        self.tempdir.cleanup()

    def test_rejects_unknown_symbol_and_invalid_inputs(self):
        self.assertFalse(self._buy("000001", "x", 100, 10, "test")["ok"])
        self.assertFalse(self._buy("600519", "x", 1.5, 10, "test")["ok"])
        self.assertFalse(self._buy("600519", "x", 100, float("nan"), "test")["ok"])

    def test_buy_uses_canonical_name_and_includes_fee_in_cost(self):
        result = self._buy("600519", "wrong", 100, 10, "test")
        self.assertTrue(result["ok"])
        position = account.get_account()["positions"][0]
        self.assertEqual(position["name"], "贵州茅台")
        self.assertAlmostEqual(position["avg_cost"], 10.05)

    def test_market_value_position_limit(self):
        result = self._buy("600519", "贵州茅台", 3100, 100, "too large",
                           current_prices={"600519": 100})
        self.assertFalse(result["ok"])
        self.assertIn("最大仓位", result["error"])

    def test_t_plus_one_only_freezes_shares_bought_today(self):
        self.assertTrue(self._buy("600519", "贵州茅台", 1000, 100, "old")["ok"])
        yesterday = (date.today() - timedelta(days=1)).isoformat()
        conn = sqlite3.connect(account.DB_PATH)
        try:
            conn.execute("UPDATE trades SET trade_date=?", (yesterday,))
            conn.execute("UPDATE positions SET last_buy_date=?", (yesterday,))
            conn.commit()
        finally:
            conn.close()
        self.assertTrue(self._buy("600519", "贵州茅台", 100, 100, "today")["ok"])
        self.assertTrue(self._sell("600519", 1000, 110, "sell old")["ok"])
        blocked = self._sell("600519", 100, 110, "sell today")
        self.assertFalse(blocked["ok"])
        self.assertIn("今日可卖 0 股", blocked["error"])

    def test_duplicate_order_is_idempotent(self):
        first = self._buy("600519", "贵州茅台", 100, 10, "test", order_id="fixed-order")
        second = self._buy("600519", "贵州茅台", 100, 10, "test", order_id="fixed-order")
        self.assertTrue(first["ok"])
        self.assertTrue(second["ok"])
        self.assertTrue(second["duplicate"])
        self.assertEqual(account.get_account()["positions"][0]["shares"], 100)
        self.assertEqual(len(account.get_trades()), 1)

    def test_daily_approved_list_is_enforced_by_account(self):
        self.assertTrue(account.authorize_run(
            "narrow-run", self.trade_date, ["600519"]
        )["ok"])
        blocked = self._buy(
            "300750", "宁德时代", 100, 10, "test", run_id="narrow-run"
        )
        self.assertFalse(blocked["ok"])
        self.assertIn("每日批准名单", blocked["error"])


if __name__ == "__main__":
    unittest.main()
