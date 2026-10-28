"""真实 SDK + 本地 HTTP 替身，验证 Responses 请求和三位专家出口，无外网/计费。"""
import json
import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import httpx
from openai import AsyncOpenAI
from agent.sdk_runner import ModelFactory, run_agent_sdk


class ResponsesIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_all_four_roles_use_sol_responses_and_captured_risk(self):
        decisions = [{"symbol": "AAPL", "name": "Apple", "action": "BUY",
                      "shares": 10, "price": 100., "reason": "test"}]
        review = {"summary": "approved", "assessments": [{"symbol": "AAPL", "action": "BUY",
                  "verdict": "APPROVE", "max_shares": 10, "reason": "test"}]}
        manager_turns = 0
        seen = []
        async def respond(request):
            nonlocal manager_turns
            body = json.loads(request.content)
            seen.append(body)
            self.assertEqual(request.url.path, "/v1/responses")
            self.assertEqual(body["model"], "gpt-6.1-sol")
            self.assertEqual(body["reasoning"]["effort"], "medium")
            self.assertEqual(body["max_output_tokens"], 8192)
            self.assertFalse(body["store"])
            instructions = body["instructions"]
            if instructions.startswith("你是"):
                manager_turns += 1
                if manager_turns <= 3:
                    name = ["technical_analysis", "fundamental_analysis", "risk_review"][manager_turns - 1]
                    prompt = json.dumps({"decisions": decisions}) if name == "risk_review" else "analyze"
                    output = [{"type": "function_call", "id": f"fc_{manager_turns}", "call_id": f"call_{manager_turns}",
                               "name": name, "arguments": json.dumps({"input": prompt}), "status": "completed"}]
                else:
                    output = self.message(json.dumps({"summary": "test", "decisions": decisions,
                                                      "risk_review": {"source": "forged"}}))
            elif '"assessments"' in instructions:
                output = self.message(json.dumps(review))
            else:
                output = self.message("analysis fixture")
            return httpx.Response(200, json={"id": f"resp_{len(seen)}", "object": "response", "created_at": 1,
                "status": "completed", "model": "gpt-6.1-sol", "output": output,
                "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15,
                          "input_tokens_details": {"cached_tokens": 0}, "output_tokens_details": {"reasoning_tokens": 0}}})

        client = AsyncOpenAI(api_key="local-http-fixture", base_url="https://fixture.invalid/v1",
                             http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)))
        factory = ModelFactory()
        factory._clients["openai"] = client
        host = SimpleNamespace(market="us", active_symbols=["AAPL"], news_screen={})
        with patch.dict(os.environ, {"ASTOCK_OPENAI_MODEL": "gpt-6.1-sol", "ASTOCK_MANAGER_PROVIDER": "openai",
                                    "ASTOCK_RISK_PROVIDER": "openai", "ASTOCK_REASONING_EFFORT": "medium",
                                    "ASTOCK_MAX_OUTPUT_TOKENS": "8192", "ASTOCK_OPENAI_TRACING": "0"}), \
             patch("agent.sdk_runner.ModelFactory", return_value=factory):
            result = await run_agent_sdk(host, "us", "2026-09-01", {})
        self.assertEqual(len(seen), 7)
        self.assertEqual(result["risk_review"]["source"], "risk-tool")
        self.assertEqual(result["risk_review"]["proposed_decisions"], decisions)
        self.assertEqual(result["agent_sdk"]["budget"]["requests_started"], 7)
        self.assertEqual(result["agent_sdk"]["budget"]["total_tokens"], 105)
        self.assertTrue(client.is_closed())

    @staticmethod
    def message(text):
        return [{"id": "msg_local", "type": "message", "role": "assistant", "status": "completed",
                 "content": [{"type": "output_text", "text": text, "annotations": []}]}]
