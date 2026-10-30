import asyncio
import json
import os
import sys
import threading
from time import monotonic
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from agent.model_access import resolve_model_access
from agent.subscription import (SubscriptionClient, _run_codex, check_subscription_login,
    parse_codex_events, protocol_progress, stderr_progress, subscription_environment, subscription_workspace)
from agent.subscription_roles import export_paper_context


def events(result=None, usage=None):
    return "\n".join(json.dumps(item) for item in [
        {"type": "thread.started", "thread_id": "test"},
        {"type": "item.completed", "item": {"type": "agent_message", "text": json.dumps(result or {"ok": True})}},
        {"type": "turn.completed", "usage": usage or {"input_tokens": 20, "cached_input_tokens": 10, "output_tokens": 5}},
    ])


class SubscriptionTests(unittest.TestCase):
    def test_recovered_connection_notice_requires_valid_terminal_and_rejects_auth_or_quota(self):
        def notice(text):
            return json.dumps({'type':'error', 'message':text})+'\n'
        message = 'Reconnecting... 1/5 (stream disconnected before completion: connection reset)'
        self.assertEqual(parse_codex_events(notice(message)+events())[0], {'ok':True})
        self.assertEqual(parse_codex_events(notice('Reconnecting... 1/5')+events())[0], {'ok':True})
        for suffix in (' HTTP 401', ' HTTP 403', ' HTTP 429', ' quota exceeded', ' authentication failed', ' permission denied'):
            with self.assertRaises(RuntimeError):
                parse_codex_events(notice(message+suffix)+events())
        with self.assertRaises(RuntimeError):
            parse_codex_events(notice(message))
        with self.assertRaises(RuntimeError):
            parse_codex_events(notice(message)*6+events())
        with self.assertRaises(RuntimeError):
            parse_codex_events(events()+'\n'+notice(message))

    def test_cli_errors_and_actual_tools_are_still_rejected(self):
        for kind in ('warning', 'error', 'command_execution', 'mcp_tool_call', 'file_change', 'unrecognized_tool'):
            activity = json.dumps({'type': 'item.completed', 'item': {'type': kind}})
            with self.assertRaises(RuntimeError):
                parse_codex_events(activity + '\n' + events())

    def test_stderr_diagnostics_only_export_fixed_categories(self):
        with subscription_workspace() as root:
            path = root / 'stderr.txt'
            path.write_text('otel exporter timeout private-account-field sqlite database locked', encoding='utf-8')
            result = stderr_progress(path)
            self.assertEqual(set(result['categories']), {'timeout', 'local_state', 'telemetry'})
            self.assertNotIn('private-account-field', json.dumps(result))

    def test_default_subscription_and_explicit_api(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(resolve_model_access(), "subscription")
            self.assertEqual(resolve_model_access("api"), "api")
            with self.assertRaises(ValueError):
                resolve_model_access("unknown")

    def test_environment_uses_system_proxy_without_forwarding_api_credentials(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "secret", "CODEX_API_KEY": "secret",
                "OPENAI_BASE_URL": "https://example.invalid", "CODEX_ACCESS_TOKEN": "secret"}, clear=True), \
                patch("agent.subscription.getproxies", return_value={"https": "http://127.0.0.1:7897"}):
            env = subscription_environment()
        self.assertEqual(env["HTTPS_PROXY"], "http://127.0.0.1:7897")
        self.assertNotIn("secret", str(env))
        self.assertNotIn("OPENAI_BASE_URL", env)
        with patch.dict(os.environ, {"https_proxy": "http://explicit:80"}, clear=True), \
                patch("agent.subscription.getproxies", return_value={"https": "http://system:80"}):
            env = subscription_environment()
            self.assertEqual(next(value for key, value in env.items() if key.upper() == "HTTPS_PROXY"),
                             "http://explicit:80")

    def test_login_rejects_api_key_auth_and_bounds_wait(self):
        with patch("agent.subscription.codex_binary", return_value="codex"), \
                patch("agent.subscription.subprocess.run") as run:
            run.return_value = SimpleNamespace(returncode=0, stdout="", stderr="Logged in using ChatGPT")
            self.assertEqual(check_subscription_login()["auth_mode"], "chatgpt")
            self.assertEqual(run.call_args.kwargs["timeout"], 15)
            run.return_value.stderr = "Logged in using an API key"
            with self.assertRaises(RuntimeError):
                check_subscription_login()

    def test_complete_stream_records_real_usage(self):
        result, usage = parse_codex_events(events())
        self.assertEqual(result, {"ok": True})
        self.assertEqual(usage["cached_input_tokens"], 10)
        self.assertIsNone(usage["reasoning_output_tokens"])

    def test_incomplete_error_tool_and_invalid_usage_never_produce_result(self):
        bad = ["[]", "not-json", '{"type":"item.completed","item":[]}',
               '{"type":"item.completed","item":{"type":"agent_message","text":"{}"}}',
               '{"type":"error","message":"secret"}',
               '{"type":"item.started","item":{"type":"command_execution"}}',
               events(usage={"input_tokens": 1, "output_tokens": 1, "cached_input_tokens": 2}),
               events(usage={"input_tokens": True, "output_tokens": 1})]
        for raw in bad:
            with self.subTest(raw=raw), self.assertRaises(RuntimeError) as error:
                parse_codex_events(raw)
            self.assertNotIn("secret", str(error.exception))

    def test_transport_runs_and_removes_its_workspace(self):
        with subscription_workspace() as root:
            output = _run_codex([sys.executable, "-c", "print('test-output')"],
                                "fixed-test-text", root, 2, threading.Event())
            self.assertEqual(output.strip(), "test-output")
        self.assertFalse(root.exists())

    def test_timeout_terminates_process_and_returns_promptly(self):
        started = monotonic()
        with subscription_workspace() as root:
            with self.assertRaises(TimeoutError):
                _run_codex([sys.executable, "-u", "-c", "import json,time;print(json.dumps({'type':'thread.started'}));print(json.dumps({'type':'turn.started'}));time.sleep(60)"],
                           "fixed-test-text", root, 1, threading.Event())
            diagnostic = json.loads((root / "transport.json").read_text(encoding="utf-8"))
            self.assertEqual(diagnostic["phase"], "turn.started")
            self.assertEqual(diagnostic["event_counts"]["turn.started"], 1)
        self.assertLess(monotonic() - started, 5)
        self.assertFalse(root.exists())

    def test_valid_terminal_result_does_not_wait_for_process_exit(self):
        payload = events({"ok": True}) + "\n"
        code = "import sys,time;sys.stdout.write(" + repr(payload) + ");sys.stdout.flush();time.sleep(60)"
        started = monotonic()
        with subscription_workspace() as root:
            output = _run_codex([sys.executable, "-u", "-c", code], "fixture", root, 8, threading.Event())
            result, usage = parse_codex_events(output)
            self.assertEqual(result, {"ok": True})
            self.assertEqual(usage["input_tokens"], 20)
            diagnostic = json.loads((root / "transport.json").read_text(encoding="utf-8"))
            self.assertTrue(diagnostic["validated_terminal_result"])
        self.assertLess(monotonic() - started, 5)
        self.assertFalse(root.exists())

    def test_message_without_terminal_usage_is_not_adopted(self):
        raw = json.dumps({"type": "item.completed", "item": {"type": "agent_message", "text": json.dumps({"ok": True})}}) + "\n"
        code = "import sys,time;sys.stdout.write(" + repr(raw) + ");sys.stdout.flush();time.sleep(60)"
        with subscription_workspace() as root:
            with self.assertRaises(TimeoutError):
                _run_codex([sys.executable, "-u", "-c", code], "fixture", root, 1, threading.Event())

    def test_terminal_event_preserves_error_and_usage_validation(self):
        for raw in (events({"ok": True}, {"input_tokens": -1, "output_tokens": 1}),
                    json.dumps({"type": "error", "message": "PRIVATE-CANARY"}) + "\n" + events({"ok": True})):
            code = "import sys,time;sys.stdout.write(" + repr(raw + "\n") + ");sys.stdout.flush();time.sleep(60)"
            with subscription_workspace() as root:
                with self.assertRaises(RuntimeError) as error:
                    _run_codex([sys.executable, "-u", "-c", code], "fixture", root, 3, threading.Event())
                self.assertNotIn("PRIVATE-CANARY", str(error.exception))

    def test_protocol_diagnostic_contains_no_messages_ids_or_unknown_fields(self):
        with subscription_workspace() as root:
            path = root / "output.jsonl"
            entries = [{"type": "thread.started", "thread_id": "PRIVATE-CANARY"},
                       {"type": "error", "message": "PRIVATE-CANARY", "token": "PRIVATE-CANARY"},
                       {"type": "PRIVATE-CANARY"}, {"type": ["PRIVATE-CANARY"]}]
            path.write_text("\n".join(json.dumps(row) for row in entries), encoding="utf-8")
            result = protocol_progress(path)
        self.assertEqual(result, {"phase": "error", "event_counts": {"thread.started": 1, "error": 1}})
        self.assertNotIn("PRIVATE-CANARY", json.dumps(result))

    def test_paper_export_filters_free_text_and_preserves_recent_predictions(self):
        context = {"date": "2026-10-26", "symbols": ["AAPL"],
            "snapshot": [{"symbol": "AAPL", "close": 100, "private": "secret"}],
            "fundamentals": [], "account": {"market": "us", "currency": "USD",
                "initial_cash": 150000., "cash": 150000., "positions": [], "note": "secret"},
            "memory": {"recent_predictions": [{"symbol": "AAPL", "direction": "up", "reason": "secret"}]}}
        packet = export_paper_context(context, "us")
        self.assertEqual(packet["past_predictions"], [{"symbol": "AAPL", "direction": "up"}])
        self.assertNotIn("secret", json.dumps(packet))
        context["account"]["initial_cash"] = 1
        with self.assertRaises(ValueError):
            export_paper_context(context, "us")


class SubscriptionAsyncTests(unittest.IsolatedAsyncioTestCase):
    async def test_rejected_response_preserves_known_terminal_usage_and_budget(self):
        client = SubscriptionClient('fixture')
        client._login = {'binary': 'fixture'}
        client.settings['token_limit'] = 25
        def rejected(args, prompt, root, timeout, cancelled):
            (root / 'transport.json').write_text(json.dumps({'phase': 'turn.completed', 'terminal_usage': {
                'input_tokens': 20, 'output_tokens': 5, 'cached_input_tokens': 0, 'reasoning_output_tokens': None}}), encoding='utf-8')
            raise RuntimeError('rejected protocol item')
        with patch('agent.subscription._run_codex', side_effect=rejected) as transport:
            with self.assertRaises(RuntimeError):
                await client.complete('fixture', {}, role='test')
            self.assertEqual(client.report()['total_tokens'], 25)
            self.assertTrue(client.report()['usage_complete'])
            self.assertEqual(client.report()['validated_calls'], 0)
            self.assertEqual(client.report()['rejected_calls'], 1)
            with self.assertRaisesRegex(RuntimeError, '预算'):
                await client.complete('fixture', {}, role='test')
            self.assertEqual(transport.call_count, 1)

    async def test_timeout_diagnostic_survives_workspace_cleanup_without_claiming_usage(self):
        client = SubscriptionClient("fixture")
        client._login = {"binary": "fixture"}
        def transport(args, prompt, root, timeout, cancelled):
            (root / "transport.json").write_text(json.dumps({"phase": "turn.started", "event_counts": {"turn.started": 1}, "elapsed_seconds": 1}), encoding="utf-8")
            raise TimeoutError("fixture timeout")
        with patch("agent.subscription._run_codex", side_effect=transport):
            with self.assertRaises(TimeoutError):
                await client.complete("fixture", {}, role="opportunity-researcher")
        report = client.report()
        self.assertFalse(report["usage_complete"])
        self.assertEqual(report["completed_calls"], 0)
        self.assertEqual(report["transport_attempts"][0]["phase"], "turn.started")

    async def test_four_role_subscription_graph_and_resume_never_use_api(self):
        from agent.analysis import AnalysisHost
        from agent.graph_runner import run_agent_graph
        class Host:
            market = "us"
            active_symbols = ["AAPL"]
            news_screen = {"as_of_date": "2026-10-26", "candidates": []}
        reader = AnalysisHost(Host(), [{"symbol": "AAPL", "date": "2026-10-26", "close": 100, "data_source": "test-provider"}],
            [{"symbol": "AAPL"}], {"market": "us", "currency": "USD", "initial_cash": 150000.,
                                   "cash": 150000., "positions": []}, [])
        def transport(args, prompt, root, timeout, cancelled):
            schema = json.loads((root / "response.json").read_text(encoding="utf-8"))
            packet = json.loads(prompt.splitlines()[-1])
            ids = [row["evidence_id"] for row in packet["evidence"]]
            evidence = {"status": "supported" if ids else "insufficient", "supporting_ids": ids[:1],
                        "contradicting_ids": [], "invalidation_condition": "new data contradicts the cited observation"}
            if "decisions" in schema["properties"]:
                result = {"summary": "test", "analyst_views": {"technical": "t", "fundamental": "f", "risk": "pending"},
                    "memory_assessment": "none", "decisions": [{"symbol": "AAPL", "name": "Apple", "action": "BUY",
                        "shares": 10, "price": 100., "reason": "test"}],
                    "predictions": [{"symbol": "AAPL", "direction": "up", "confidence": .7,
                                     "horizon_days": 5, "rationale": "test"}]}
            elif "assessments" in schema["properties"]:
                result = {"summary": "review", "assessments": [{"symbol": "AAPL", "action": "BUY",
                          "verdict": "RESIZE", "max_shares": 5, "reason": "test"}]}
            else:
                result = {"summary": "test-analysis", "claims": [
                    {"symbol": "AAPL", "claim": "test observation or insufficient evidence", "evidence": evidence}]}
            for collection in ("decisions", "predictions", "assessments"):
                for row in result.get(collection, []):
                    row["evidence"] = evidence
            return events(result)
        with subscription_workspace() as root, \
                patch("agent.graph_runner.ModelFactory", side_effect=AssertionError("API must not start")), \
                patch("agent.subscription.check_subscription_login", return_value={"binary": "codex"}), \
                patch("agent.subscription._run_codex", side_effect=transport) as run:
            kwargs = {"model_access": "subscription", "require_confirmation": True,
                      "checkpoint_path": root / "graph.db"}
            result = await run_agent_graph(reader, "us", "2026-10-26", {}, **kwargs)
            self.assertEqual(result["graph"]["status"], "awaiting_confirmation")
            self.assertEqual(result["decisions"][0]["shares"], 5)
            self.assertEqual(result["agent_sdk"]["auth_mode"], "chatgpt")
            self.assertEqual(result["agent_sdk"]["usage"]["requests"], 4)
            self.assertFalse(result["evidence_audit"]["semantic_support_verified"])
            self.assertTrue(result["decisions"][0]["evidence"]["supporting_ids"])
            from scripts.inspect_evidence import inspect_log
            traced = inspect_log(result, "AAPL")
            self.assertEqual(traced["results"][0]["supporting_evidence"][0]["symbol"], "AAPL")
            result = await run_agent_graph(reader, "us", "2026-10-26", {},
                approve_plan=result["graph"]["plan_hash"], **kwargs)
            self.assertEqual(result["graph"]["status"], "approved")
            self.assertEqual(result["agent_sdk"]["usage"]["requests"], 4)
            self.assertEqual(run.call_count, 4)

    async def test_failed_request_counts_toward_limit_without_api_fallback(self):
        client = SubscriptionClient("gpt-6.1-sol")
        client.settings["request_limit"] = 1
        client._login = {"binary": "codex"}
        with patch("agent.subscription._run_codex", side_effect=RuntimeError("subscription failed")) as run:
            with self.assertRaisesRegex(RuntimeError, "subscription failed"):
                await client.complete("test", {}, role="test")
            with self.assertRaisesRegex(RuntimeError, "预算"):
                await client.complete("test", {}, role="test")
        self.assertEqual(run.call_count, 1)
        self.assertEqual(client.report()["requests_started"], 1)
        self.assertEqual(client.report()["completed_calls"], 0)

    async def test_budget_usage_and_valid_cli_configuration(self):
        client = SubscriptionClient("gpt-6.1-sol")
        client._login = {"binary": "codex"}
        with patch("agent.subscription._run_codex", return_value=events()) as run:
            self.assertEqual(await client.complete("test", {}, role="test"), {"ok": True})
            self.assertNotIn("model_providers.openai", " ".join(run.call_args.args[0]))
            self.assertIn('forced_login_method="chatgpt"', run.call_args.args[0])
            self.assertIn('model_verbosity="low"', run.call_args.args[0])
            self.assertNotIn('features.skip_host_skill_discovery=true', run.call_args.args[0])
            client.settings["token_limit"] = 25
            with self.assertRaisesRegex(RuntimeError, "预算"):
                await client.complete("test", {}, role="test")
        self.assertEqual(client.report()["total_tokens"], 25)

    async def test_cancellation_stops_worker_before_workspace_cleanup(self):
        started = threading.Event()
        stopped = threading.Event()
        def worker(args, prompt, root, timeout, cancelled):
            started.set()
            cancelled.wait(3)
            stopped.set()
            raise RuntimeError("cancelled")
        client = SubscriptionClient("gpt-6.1-sol")
        client._login = {"binary": "codex"}
        with patch("agent.subscription._run_codex", side_effect=worker):
            task = asyncio.create_task(client.complete("test", {}, role="test"))
            self.assertTrue(await asyncio.to_thread(started.wait, 2))
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await asyncio.wait_for(task, 2)
        self.assertTrue(stopped.is_set())


if __name__ == "__main__":
    unittest.main()
