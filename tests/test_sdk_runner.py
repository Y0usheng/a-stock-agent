import os
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from agent.sdk_runner import (
    RESULT_SCHEMA,
    SPECIALIST_TOOL_ORDER,
    build_agent_definitions,
    build_agents,
    build_read_only_tools,
    parse_agent_result,
    resolve_role_routing,
    validate_specialist_sequence,
    build_prompt_manifest,
    risk_output_extractor,
    run_agent_sdk,
    ModelFactory,
)


class AgentSdkConfigurationTests(unittest.TestCase):
    def test_default_role_routing_matches_approved_design(self):
        with patch.dict(os.environ, {}, clear=True):
            routing = resolve_role_routing()
        self.assertEqual(routing["investment-manager"].provider, "openai")
        self.assertTrue(all(role.provider == "openai" and role.model == "gpt-6.1-sol"
                            for role in routing.values()))

    def test_global_model_override_applies_to_all_roles(self):
        with patch.dict(os.environ, {"ASTOCK_OPENAI_MODEL": "test-model"}, clear=True):
            routing = resolve_role_routing()
        self.assertTrue(all(role.model == "test-model" for role in routing.values()))

    def test_legacy_provider_cannot_silently_mix_models(self):
        with patch.dict(os.environ, {"ASTOCK_RISK_PROVIDER": "deepseek"}, clear=True):
            with self.assertRaises(ValueError):
                resolve_role_routing()

    def test_invalid_provider_is_rejected(self):
        with self.assertRaises(ValueError):
            resolve_role_routing(manager_provider="claude")

    def test_roles_are_read_only_and_separated(self):
        definitions = build_agent_definitions(
            resolve_role_routing(manager_provider="openai", risk_provider="openai")
        )
        self.assertEqual(
            set(definitions),
            {"investment-manager", "technical-analyst", "fundamental-analyst", "risk-manager"},
        )
        for definition in definitions.values():
            self.assertFalse(any(name in {"buy", "sell"} for name in definition.tools))
        self.assertIn(
            "market_watchlist_fundamentals",
            definitions["fundamental-analyst"].tools,
        )
        self.assertIn("account_snapshot", definitions["risk-manager"].tools)
        self.assertIn("news_candidate_screen", definitions["investment-manager"].tools)
        self.assertIn("news_symbol", definitions["risk-manager"].tools)

    def test_output_schema_requires_auditable_role_views(self):
        required = RESULT_SCHEMA["required"]
        self.assertIn("analyst_views", required)
        self.assertIn("memory_assessment", required)

    def test_specialists_and_final_risk_review_are_mandatory(self):
        validate_specialist_sequence(list(SPECIALIST_TOOL_ORDER))
        with self.assertRaises(RuntimeError):
            validate_specialist_sequence(["technical_analysis", "risk_review"])
        with self.assertRaises(RuntimeError):
            validate_specialist_sequence([
                "technical_analysis", "fundamental_analysis", "risk_review",
                "technical_analysis",
            ])
        with self.assertRaises(RuntimeError):
            validate_specialist_sequence(["fundamental_analysis", "technical_analysis", "risk_review"])

    def test_template_identity_is_stable_across_dates_and_candidates(self):
        first = build_prompt_manifest("us", "2026-09-01", ["AAPL"], "manager-v1")
        second = build_prompt_manifest("us", "2026-09-02", ["META"], "manager-v1")
        changed = build_prompt_manifest("us", "2026-09-01", ["AAPL"], "manager-v2")
        self.assertNotEqual(first["sha256"], second["sha256"])
        self.assertEqual(first["template_sha256"], second["template_sha256"])
        self.assertNotEqual(first["template_sha256"], changed["template_sha256"])

    def test_json_parser_supports_both_provider_output_styles(self):
        expected = {"summary": "hold"}
        self.assertEqual(parse_agent_result('{"summary":"hold"}'), expected)
        self.assertEqual(parse_agent_result('```json\n{"summary":"hold"}\n```'), expected)

    def test_sdk_tools_wrap_only_read_calls(self):
        class Host:
            async def call(self, _name, _args):
                return {}

        tools = build_read_only_tools(Host())
        self.assertIn("account_snapshot", tools)
        self.assertIn("news_candidate_screen", tools)
        self.assertNotIn("buy", tools)
        self.assertNotIn("sell", tools)

    def test_manager_exposes_three_specialists_as_tools(self):
        class Host:
            async def call(self, _name, _args):
                return {}

        class ModelFactory:
            def model_for(self, selection):
                return selection.model

        manager = build_agents(
            Host(), "us", "2026-08-13", resolve_role_routing(), ModelFactory()
        )
        names = [tool.name for tool in manager.tools]
        self.assertEqual(
            names,
            [
                "market_watchlist_snapshot",
                "news_candidate_screen",
                "account_snapshot",
                "technical_analysis",
                "fundamental_analysis",
                "risk_review",
            ],
        )
        self.assertFalse(manager.model_settings.parallel_tool_calls)


class RiskToolCaptureTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.plan = [{"symbol": "AAPL", "name": "Apple", "action": "BUY",
                      "shares": 10, "price": 100.0, "reason": "test"}]
        self.review = {"summary": "approved", "assessments": [
            {"symbol": "AAPL", "action": "BUY", "verdict": "APPROVE",
             "max_shares": 10, "reason": "test"}]}

    async def test_invalid_later_review_cannot_reuse_old_approval(self):
        audit = {}
        extract = risk_output_extractor(audit, "us", {"AAPL"})
        await extract(SimpleNamespace(input=json.dumps({"decisions": self.plan}),
                                      final_output=json.dumps(self.review)))
        self.assertEqual(audit["source"], "risk-tool")
        with self.assertRaises(RuntimeError):
            await extract(SimpleNamespace(input="a free-form plan", final_output="APPROVE"))
        self.assertEqual(audit, {})

    async def test_real_sdk_tool_adapter_captures_review_and_overwrites_manager_field(self):
        from agents import RunContextWrapper, Runner

        class Host:
            market = "us"
            active_symbols = ["AAPL"]
            news_screen = {}

            async def call(self, name, args):
                raise AssertionError("SDK inference is replaced locally in this test")

        async def fake_run(starting_agent, input, **kwargs):
            if starting_agent.name == "risk-manager":
                return SimpleNamespace(input=input, final_output=json.dumps(self.review))
            risk_tool = next(tool for tool in starting_agent.tools if tool.name == "risk_review")
            captured = await risk_tool.on_invoke_tool(
                RunContextWrapper(context=None),
                json.dumps({"input": json.dumps({"decisions": self.plan})}),
            )
            self.assertEqual(json.loads(captured)["source"], "risk-tool")
            return SimpleNamespace(
                final_output={"decisions": self.plan, "risk_review": {"source": "manager-forgery"}},
                new_items=[SimpleNamespace(type="tool_call_item", raw_item={"name": name})
                           for name in SPECIALIST_TOOL_ORDER],
                context_wrapper=RunContextWrapper(context=None), raw_responses=[], last_response_id=None,
            )

        with patch.object(ModelFactory, "model_for", return_value="test-model"), \
             patch.object(Runner, "run", side_effect=fake_run):
            result = await run_agent_sdk(Host(), "us", "2026-09-01", {})
        self.assertEqual(result["risk_review"]["source"], "risk-tool")
        self.assertEqual(result["risk_review"]["proposed_decisions"], self.plan)


if __name__ == "__main__":
    unittest.main()
