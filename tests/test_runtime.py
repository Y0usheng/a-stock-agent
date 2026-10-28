import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from agent.runtime import ModelBudget, model_settings


class RuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def test_experts_share_request_and_completed_token_budget(self):
        with patch.dict(os.environ, {"ASTOCK_MAX_MODEL_REQUESTS": "2", "ASTOCK_MAX_TOTAL_TOKENS": "100"}):
            budget = ModelBudget()
        await budget.on_llm_start(None, None, None, [])
        await budget.on_llm_end(None, None, SimpleNamespace(usage=SimpleNamespace(input_tokens=80, output_tokens=20)))
        with self.assertRaises(RuntimeError):
            await budget.on_llm_start(None, None, None, [])
        self.assertEqual(budget.report()["requests_started"], 1)
        with patch.dict(os.environ, {"ASTOCK_MAX_MODEL_REQUESTS": "1"}):
            budget = ModelBudget()
        await budget.on_llm_start(None, None, None, [])
        with self.assertRaises(RuntimeError):
            await budget.on_llm_start(None, None, None, [])

    def test_sol_rejects_unsupported_reasoning_and_invalid_output_cap(self):
        with patch.dict(os.environ, {"ASTOCK_REASONING_EFFORT": "none"}):
            with self.assertRaises(ValueError):
                model_settings()
        with patch.dict(os.environ, {"ASTOCK_MAX_OUTPUT_TOKENS": "0"}):
            with self.assertRaises(ValueError):
                model_settings()

    def test_settings_are_responses_compatible_and_do_not_store_outputs(self):
        with patch.dict(os.environ, {}, clear=True):
            settings = model_settings()
        self.assertEqual(settings.reasoning.effort, "medium")
        self.assertEqual(settings.max_tokens, 8192)
        self.assertFalse(settings.store)
        self.assertFalse(settings.parallel_tool_calls)
