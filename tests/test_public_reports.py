from copy import deepcopy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from agent.opportunities import OpportunityBook
from agent.subscription import subscription_workspace
from evals.experiments import canonical_hash
from evals.public_reports import (build_site, filename, ledger_as_of, make_report, sanitize_view,
                                 validate_report, write_report)
from tests.test_trading_view import fixture


def rehash(report):
    report["content_hash"] = canonical_hash({key: value for key, value in report.items() if key not in {"content_hash", "published_at"}})


class PublicReportTests(unittest.TestCase):
    def test_learning_and_ai_policy_whitelist_excludes_code_parameters_and_order_ids(self):
        view = sanitize_view({'status': 'ready', 'execution_policy': 'autonomous-paper-execution-v1',
            'learning': {'version': 'paper-outcome-learning-v1', 'status': 'insufficient_data', 'mature_samples': 0,
                'pending_samples': 2, 'strategy_status': 'waiting_for_mature_samples',
                'active_parameters': {'private': True}, 'strategy_code': {'source': 'private'}},
            'pending_orders': [{'execution_policy': 'autonomous-paper-execution-v1', 'status': 'confirmed', 'plan_hash': 'private'}]})
        self.assertEqual(view['learning']['pending_samples'], 2)
        self.assertNotIn('private', json.dumps(view))
        self.assertEqual(view, sanitize_view(view))

    def setup_report(self, root, day="2026-10-26", *, ledger=None, logs=None):
        payload, original, original_logs = fixture()
        original["cash"] = 1_000_000.
        cache = root / "prices.json"
        cache.write_text(json.dumps(payload), encoding="utf-8")
        return make_report(day, cache=cache, ledger=ledger or original, logs=logs or original_logs, book=OpportunityBook(root / "absent.db"))

    def test_projection_never_exports_raw_inputs_ledger_identifiers_or_news_excerpt(self):
        payload = {"status": "ready", "input_dataset": {"credentials": "private"}, "account_snapshot": {"private": True},
            "trades": [{"symbol": "300750", "run_id": "private", "order_id": "private", "reason": "private", "price": 10}],
            "opportunities": {"status": "ready", "themes": [{"theme_id": "x", "name": "test", "events": [{"title": "public", "url": "https://finance.eastmoney.com/a/test.html", "excerpt": "private", "content_hash": "private"}], "judgment": None}]}}
        view = sanitize_view(payload)
        encoded = json.dumps(view)
        self.assertNotIn("private", encoded)
        self.assertNotIn("input_dataset", encoded)
        self.assertNotIn("excerpt", encoded)
        self.assertEqual(view, sanitize_view(view))

    def test_historical_report_excludes_future_prices_and_model_outputs(self):
        with subscription_workspace() as root:
            report = self.setup_report(root, "2026-10-23")
            self.assertEqual(report["price_date"], "2026-10-23")
            self.assertIsNone(report["analysis_date"])
            self.assertIsNone(report["view"]["stocks"][0]["decision"])
            self.assertEqual(report["view"]["stocks"][0]["series"][-1]["date"], "2026-10-23")

    def test_after_close_decision_is_not_backfilled_to_its_older_price_date(self):
        _, _, logs = fixture()
        logs[0]["timing"] = {"version": "after-close-next-session-v1", "price_as_of_date": "2026-10-26",
            "information_cutoff_at": "2026-10-27T10:00:00+08:00", "decision_at": "2026-10-27T10:05:00+08:00",
            "execution_session": "2026-10-27", "approval_deadline": "2026-10-27T14:55:00+08:00", "max_price_deviation_pct": .02}
        with subscription_workspace() as root:
            old = self.setup_report(root, "2026-10-26", logs=logs)
            self.assertIsNone(old["analysis_date"])
            self.assertNotIn("timing", old["view"])
            current = self.setup_report(root, "2026-10-27", logs=logs)
            self.assertEqual(current["analysis_date"], "2026-10-27")
            self.assertEqual(current["price_date"], "2026-10-26")
            self.assertNotIn("queue_plan_hash", json.dumps(current))

    def test_old_date_replays_only_then_existing_trades_and_fees(self):
        ledger = {"cash": 999950., "positions": [{"symbol": "300750", "shares": 5}], "exists": True,
            "trades": [{"symbol": "300750", "trade_date": "2026-10-20", "action": "BUY", "shares": 10, "price": 10., "fee": 5.},
                       {"symbol": "300750", "trade_date": "2026-10-26", "action": "SELL", "shares": 5, "price": 12., "fee": 5.}]}
        old = ledger_as_of(ledger, "2026-10-23")
        self.assertEqual(old["cash"], 999895.)
        self.assertEqual(old["positions"][0]["shares"], 10)
        self.assertEqual(old["positions"][0]["avg_cost"], 10.5)
        with subscription_workspace() as root:
            report = self.setup_report(root, "2026-10-23", ledger=ledger)
            self.assertEqual(report["view"]["portfolio"]["net_pnl"], 5.)
            self.assertEqual(len(report["view"]["trades"]), 1)

    def test_unreconciled_ledger_cannot_be_published_as_profit(self):
        with self.assertRaises(ValueError):
            ledger_as_of({"cash": 1., "positions": [], "trades": []}, "2026-10-26")

    def test_credentials_in_model_text_stop_publication_without_echoing_them(self):
        with subscription_workspace() as root:
            report = self.setup_report(root)
            secret = "sk-" + "A" * 40
            report["view"]["stocks"][0]["decision"]["reason"] = secret
            rehash(report)
            with self.assertRaises(ValueError) as raised:
                validate_report(report)
            self.assertNotIn(secret, str(raised.exception))

    def test_local_path_and_raw_fields_cannot_enter_public_report(self):
        with subscription_workspace() as root:
            report = self.setup_report(root)
            report["view"]["stocks"][0]["decision"]["reason"] = "C:\\Users\\private\\file"
            rehash(report)
            with self.assertRaises(ValueError):
                validate_report(report)
            report = self.setup_report(root)
            report["view"]["account_snapshot"] = {"secret": "private"}
            rehash(report)
            with self.assertRaises(ValueError):
                validate_report(report)

    def test_script_in_numeric_field_and_nonfinite_values_are_rejected(self):
        with subscription_workspace() as root:
            for value in ("<img src=x onerror=alert(1)>", float("nan"), True):
                report = self.setup_report(root)
                report["view"]["stocks"][0]["decision"]["shares"] = value
                rehash(report)
                with self.assertRaises(ValueError):
                    validate_report(report)

    def test_same_content_keeps_first_publication_and_changes_create_revision(self):
        with subscription_workspace() as root:
            report = self.setup_report(root)
            path = write_report(report, root / "public")
            second = deepcopy(report)
            second["published_at"] = "2026-10-28T18:00:00+08:00"
            self.assertEqual(write_report(second, root / "public"), path)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["published_at"], report["published_at"])
            second["view"]["stocks"][0]["decision"]["reason"] = "new research view"
            rehash(second)
            self.assertNotEqual(write_report(second, root / "public"), path)
            self.assertEqual(len(list((root / "public" / "reports").glob("*.json"))), 2)

    def test_empty_archive_build_has_no_fabricated_daily_records(self):
        with subscription_workspace() as root:
            manifest = build_site(root / "none", root / "site")
            self.assertEqual(manifest["reports"], [])
            self.assertIn("还没有发布真实研究报告", (root / "site" / "index.html").read_text(encoding="utf-8"))

    def test_date_archive_uses_relative_paths_and_labels_shadow_results(self):
        with subscription_workspace() as root:
            source, site = root / "public", root / "site"
            write_report(self.setup_report(root, "2026-10-23"), source)
            report = self.setup_report(root)
            report["view"]["stocks"][0]["decision"]["reason"] = "</script><script>alert(1)</script>"
            rehash(report)
            write_report(report, source)
            manifest = build_site(source, site)
            self.assertEqual([row["date"] for row in manifest["reports"]], ["2026-10-26", "2026-10-23"])
            html = (site / "reports" / "2026-10-26.html").read_text(encoding="utf-8")
            self.assertIn("只分析 · 未成交建议", html)
            self.assertIn("\\u003c/script", html)
            self.assertNotIn('<a href="/">', html)
            self.assertFalse((site / "auth.json").exists())

    def test_altered_report_and_mismatched_filename_fail_build(self):
        with subscription_workspace() as root:
            source = root / "public"
            report = self.setup_report(root)
            path = write_report(report, source)
            report["view"]["portfolio"]["cash"] += 1
            path.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaises(ValueError):
                build_site(source, root / "site")

    def test_authenticated_and_script_urls_are_removed(self):
        from evals.public_reports import public_url
        for url in ("javascript:alert(1)", "https://secret@finance.eastmoney.com/a/test.html", "https://finance.eastmoney.com/a/test.html?token=private"):
            self.assertEqual(public_url(url), "")

    def test_publisher_binds_destination_before_any_push(self):
        from scripts.publish_reports import publish
        with patch("scripts.publish_reports.command", return_value=b"https://github.com/another/repo.git\n") as command:
            with self.assertRaises(ValueError):
                publish(Path("unused"), "Y0usheng/a-stock-agent")
            self.assertEqual(command.call_count, 1)

    def test_output_directory_with_private_files_cannot_be_uploaded_by_accident(self):
        with subscription_workspace() as root:
            output = root / "site"
            output.mkdir()
            (output / "auth.json").write_text("private", encoding="utf-8")
            with self.assertRaises(ValueError):
                build_site(root / "empty", output)
            self.assertEqual((output / "auth.json").read_text(encoding="utf-8"), "private")

    def test_historical_report_cannot_reuse_later_theme_discovery(self):
        with subscription_workspace() as root:
            payload, ledger, logs = fixture()
            ledger["cash"] = 1_000_000.
            cache = root / "prices.json"
            cache.write_text(json.dumps(payload), encoding="utf-8")
            future_book = type("FutureBook", (), {"_read": lambda self, *args: [{"as_of_date": "2026-10-26", "selected_symbols": ["300750"]}]})()
            report = make_report("2026-10-23", cache=cache, ledger=ledger, logs=logs, book=future_book)
            self.assertFalse(report["view"]["opportunities"].get("selected_symbols"))


if __name__ == "__main__":
    unittest.main()
