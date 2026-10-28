from copy import deepcopy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from agent.subscription import subscription_workspace
from evals.experiments import canonical_hash
from evals.trading_view import analysis_logs, build_view, read_ledger, render_html


def fixture():
    # 单元测试价格替身，不输出到真实日报或真实价格缓存。
    rows = [{"date": day, "open": price, "high": price + 1, "low": price - 1, "close": price, "volume": 10000}
            for day, price in [("2026-10-20", 10), ("2026-10-21", 11), ("2026-10-22", 12), ("2026-10-23", 11), ("2026-10-26", 12)]]
    histories = {"300750": rows}
    payload = {"market": "cn", "data_quality": "live_prices", "source": "unit test fixture", "price_basis": "qfq",
               "histories": histories, "fingerprint": canonical_hash(histories), "fetched_at": "2026-10-26T16:00:00+08:00"}
    ledger = {"cash": 1000., "positions": [], "trades": []}
    log = {"market": "cn", "as_of_date": "2026-10-26", "execution_mode": "shadow", "offline": False,
           "decisions": [{"symbol": "300750", "action": "HOLD", "shares": 0}], "predictions": []}
    return payload, ledger, [log]


class TradingViewTests(unittest.TestCase):
    def test_empty_book_does_not_invent_historical_pnl_or_buys(self):
        payload, ledger, logs = fixture()
        logs[0]["decisions"][0].update(action="BUY", shares=100)
        view = build_view(payload, ledger, logs, 1000)
        self.assertEqual(len(view["stocks"][0]["series"]), 5)
        self.assertEqual([p["date"] for p in view["portfolio_curve"]], ["2026-10-26"])
        self.assertEqual(view["portfolio"]["net_pnl"], 0)
        self.assertEqual(view["stocks"][0]["trades"], [])
        self.assertEqual(view["stocks"][0]["tracking_benchmark"], [{"date": "2026-10-26", "return_pct": 0.}])

    def test_actual_buy_and_sell_fees_and_slippage_are_not_counted_twice(self):
        payload, ledger, logs = fixture()
        ledger.update(cash=945., positions=[{"symbol": "300750", "shares": 5, "avg_cost": 10.5, "last_buy_date": "2026-10-20"}],
            trades=[{"id": 1, "symbol": "300750", "trade_date": "2026-10-20", "action": "BUY", "shares": 10, "price": 10., "fee": 5., "slippage_cost": 1.},
                    {"id": 2, "symbol": "300750", "trade_date": "2026-10-21", "action": "SELL", "shares": 5, "price": 11., "fee": 5., "slippage_cost": 1.}])
        view = build_view(payload, ledger, logs, 1000)
        self.assertEqual(view["portfolio"]["total_assets"], 1005.)
        self.assertEqual(view["portfolio"]["net_pnl"], 5.)
        self.assertEqual(view["portfolio"]["realized_pnl"], -2.5)
        self.assertEqual(view["portfolio"]["unrealized_pnl"], 7.5)
        self.assertEqual(view["portfolio"]["total_fees"], 10.)
        self.assertEqual(view["portfolio"]["total_slippage"], 2.)
        self.assertEqual(view["portfolio_curve"][0]["total"], 995.)

    def test_unreconciled_cash_is_not_reported_as_investment_profit(self):
        payload, ledger, logs = fixture()
        ledger["cash"] = 1200.
        view = build_view(payload, ledger, logs, 1000)
        self.assertFalse(view["portfolio"]["journal_reconciled"])
        self.assertEqual(view["portfolio"]["total_assets"], 1200.)
        self.assertIsNone(view["portfolio"]["net_pnl"])
        self.assertEqual(view["portfolio_curve"], [])

    def test_missing_price_is_not_filled_at_cost(self):
        payload, ledger, logs = fixture()
        ledger.update(cash=895., positions=[{"symbol": "600036", "shares": 10, "avg_cost": 10.5, "last_buy_date": "2026-10-20"}],
            trades=[{"symbol": "600036", "trade_date": "2026-10-20", "action": "BUY", "shares": 10, "price": 10., "fee": 5.}])
        view = build_view(payload, ledger, logs, 1000)
        self.assertIsNone(view["portfolio"]["total_assets"])
        self.assertIsNone(view["positions"][0]["last_price"])
        self.assertTrue(view["warnings"])

    def test_trade_after_price_cutoff_cannot_be_valued_with_old_price(self):
        payload, ledger, logs = fixture()
        ledger.update(cash=895., positions=[{"symbol": "300750", "shares": 10, "avg_cost": 10.5, "last_buy_date": "2026-10-27"}],
            trades=[{"symbol": "300750", "trade_date": "2026-10-27", "action": "BUY", "shares": 10, "price": 10., "fee": 5.}])
        view = build_view(payload, ledger, logs, 1000)
        self.assertIsNone(view["portfolio"]["net_pnl"])
        self.assertIsNone(view["positions"][0]["last_price"])

    def test_changed_price_cache_is_rejected(self):
        payload, ledger, logs = fixture()
        payload["histories"]["300750"][0]["close"] = 999
        with self.assertRaisesRegex(ValueError, "指纹"):
            build_view(payload, ledger, logs, 1000)

    def test_shadow_logs_are_read_but_synthetic_logs_are_excluded(self):
        with subscription_workspace() as root:
            (root / "shadow").mkdir()
            for index, offline in enumerate([False, True]):
                (root / "shadow" / f"{index}.cn.json").write_text(json.dumps({"market": "cn", "offline": offline, "date": "2026-10-26"}), encoding="utf-8")
            self.assertEqual(len(analysis_logs(root)), 1)
            self.assertFalse(read_ledger(root / "not-created.db")["exists"])
            self.assertFalse((root / "not-created.db").exists())

    def test_page_uses_embedded_data_and_escapes_script_content(self):
        payload, ledger, logs = fixture()
        view = build_view(payload, ledger, logs, 1000)
        view["warnings"].append("</script><script>alert(1)</script>")
        html = render_html(view)
        self.assertNotIn("<script>alert(1)", html)
        self.assertIn("\\u003c/script>", html)
        self.assertNotIn("cdn.jsdelivr", html)
        self.assertIn('id="priceChart"', html)

    def test_routes_use_saved_data_without_model_or_network(self):
        from dashboard.app import app
        payload, ledger, logs = fixture()
        view = build_view(payload, ledger, logs, 1000)
        with patch("evals.trading_view.load_view", return_value=view):
            client = app.test_client()
            self.assertEqual(client.get("/api/trading-view").get_json()["portfolio"]["net_pnl"], 0.)
            self.assertEqual(client.get("/trading").status_code, 200)

    def test_sina_refresh_preserves_share_units_and_source(self):
        import pandas as pd
        from scripts.refresh_trading_view import main
        payload, ledger, logs = fixture()
        view = build_view(payload, ledger, logs, 1000)
        frame = pd.DataFrame(payload["histories"]["300750"])
        with subscription_workspace() as root:
            argv = ["refresh", "--provider", "sina", "--symbols", "300750", "--cache", str(root / "prices.json"), "--output", str(root / "view.html")]
            with patch("sys.argv", argv), patch("scripts.refresh_trading_view.analysis_logs", return_value=logs), patch("scripts.refresh_trading_view.read_ledger", return_value=ledger), patch("akshare.stock_zh_a_daily", return_value=frame), patch("scripts.refresh_trading_view.load_view", return_value=view), patch("builtins.print"):
                self.assertEqual(main(), 0)
            saved = json.loads((root / "prices.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["histories"]["300750"][0]["volume"], 10000)
            self.assertIn("新浪", saved["source"])
            self.assertTrue((root / "view.html").exists())

    def test_failed_refresh_preserves_previous_price_cache(self):
        from scripts.refresh_trading_view import main
        payload, ledger, logs = fixture()
        with subscription_workspace() as root:
            path = root / "prices.json"
            path.write_text("previous saved data", encoding="utf-8")
            argv = ["refresh", "--symbols", "300750", "--cache", str(path), "--output", str(root / "view.html")]
            with patch("sys.argv", argv), patch("scripts.refresh_trading_view.analysis_logs", return_value=logs), patch("scripts.refresh_trading_view.read_ledger", return_value=ledger), patch("akshare.stock_zh_a_hist", side_effect=OSError("provider offline")), patch("builtins.print"):
                self.assertEqual(main(), 2)
            self.assertEqual(path.read_text(encoding="utf-8"), "previous saved data")
