import os
import unittest
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
)


class AgentSdkConfigurationTests(unittest.TestCase):
    def test_default_role_routing_matches_approved_design(self):
        with patch.dict(os.environ, {}, clear=True):
            routing = resolve_role_routing()
        self.assertEqual(routing["investment-manager"].provider, "openai")
        self.assertEqual(routing["technical-analyst"].provider, "deepseek")
        self.assertEqual(routing["fundamental-analyst"].provider, "deepseek")
        self.assertEqual(routing["risk-manager"].provider, "deepseek")

    def test_manager_and_risk_are_independently_switchable(self):
        routing = resolve_role_routing(manager_provider="deepseek", risk_provider="openai")
        self.assertEqual(routing["investment-manager"].provider, "deepseek")
        self.assertEqual(routing["risk-manager"].provider, "openai")
        self.assertEqual(routing["technical-analyst"].provider, "deepseek")

    def test_invalid_provider_is_rejected(self):
        with self.assertRaises(ValueError):
            resolve_role_routing(manager_provider="claude")

    def test_roles_are_read_only_and_separated(self):
        definitions = build_agent_definitions(
            resolve_role_routing(manager_provider="openai", risk_provider="deepseek")
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


if __name__ == "__main__":
    unittest.main()
