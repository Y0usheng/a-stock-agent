import asyncio
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from agent.analysis import AnalysisHost
from agent.graph_runner import build_decision_graph, run_agent_graph
from agent.sdk_runner import ModelFactory
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command


def draft():
    return {"summary": "test", "analyst_views": {"technical": "t", "fundamental": "f", "risk": "pending"},
            "memory_assessment": "none", "decisions": [{"symbol": "AAPL", "name": "Apple", "action": "BUY",
                "shares": 10, "price": 100., "reason": "test"}],
            "predictions": [{"symbol": "AAPL", "direction": "up", "confidence": .7,
                             "horizon_days": 5, "rationale": "test"}]}


def review(verdict="RESIZE"):
    return {"summary": "review", "assessments": [{"symbol": "AAPL", "action": "BUY", "verdict": verdict,
             "max_shares": 5 if verdict == "RESIZE" else 0, "reason": "test"}]}


class GraphTests(unittest.IsolatedAsyncioTestCase):
    async def test_parallel_analysis_join_and_persistent_confirmation_without_repeating_models(self):
        calls = []
        both_started = asyncio.Event()
        async def call(name, input):
            calls.append(name)
            if name in {"technical-analyst", "fundamental-analyst"}:
                if len(calls) == 2:
                    both_started.set()
                await asyncio.wait_for(both_started.wait(), 2)
                return name
            if name == "investment-manager":
                self.assertTrue(both_started.is_set())
                self.assertIn("technical-analyst", input)
                return json.dumps(draft())
            self.assertEqual(json.loads(input)["decisions"][0]["shares"], 10)
            return json.dumps(review())

        with tempfile.TemporaryDirectory() as temp:
            path = str(Path(temp) / "checkpoints.db")
            config = {"configurable": {"thread_id": "test"}}
            async with AsyncSqliteSaver.from_conn_string(path) as saver:
                graph = build_decision_graph(call, "us", {"AAPL"}, checkpointer=saver)
                result = await graph.ainvoke({"context_fingerprint": "frozen"}, config)
                self.assertTrue(result["__interrupt__"])
                self.assertEqual(result["result"]["decisions"][0]["shares"], 5)
                plan_hash = result["plan_hash"]
            async with AsyncSqliteSaver.from_conn_string(path) as saver:
                graph = build_decision_graph(call, "us", {"AAPL"}, checkpointer=saver)
                result = await graph.ainvoke(Command(resume={"approve": True, "plan_hash": plan_hash}), config)
            self.assertEqual(result["status"], "approved")
            self.assertEqual(len(calls), 4)

    async def test_rejected_risk_never_enters_confirmation(self):
        async def call(name, input):
            return json.dumps(draft() if name == "investment-manager" else review("REJECT")) if name in {
                "investment-manager", "risk-manager"} else name
        result = await build_decision_graph(call, "us", {"AAPL"}).ainvoke({})
        self.assertNotIn("__interrupt__", result)
        self.assertEqual(result["result"]["decisions"][0]["action"], "HOLD")

    async def test_confirmation_must_bind_exact_plan_and_explicit_boolean(self):
        from langgraph.checkpoint.memory import InMemorySaver
        async def call(name, input):
            return json.dumps(draft() if name == "investment-manager" else review()) if name in {
                "investment-manager", "risk-manager"} else name
        graph = build_decision_graph(call, "us", {"AAPL"}, checkpointer=InMemorySaver())
        config = {"configurable": {"thread_id": "test"}}
        result = await graph.ainvoke({}, config)
        with self.assertRaises(RuntimeError):
            await graph.ainvoke(Command(resume={"approve": True, "plan_hash": "forged"}), config)
        # 用另一检查点验证显式拒绝。
        config = {"configurable": {"thread_id": "reject"}}
        result = await graph.ainvoke({}, config)
        result = await graph.ainvoke(Command(resume={"approve": False, "plan_hash": result["plan_hash"]}), config)
        self.assertEqual(result["status"], "rejected")
        self.assertEqual(result["result"]["decisions"][0]["shares"], 0)

    async def test_sdk_graph_wrapper_resumes_without_api_and_rejects_changed_account(self):
        from agents import Runner
        class Host:
            market = "us"
            active_symbols = ["AAPL"]
            news_screen = {"as_of_date": "2026-09-01"}
            async def call(self, *args):
                raise AssertionError("frozen tools only")
        def reader(cash=1000):
            return AnalysisHost(Host(), [{"symbol": "AAPL", "date": "2026-09-01", "close": 100}],
                                [{"symbol": "AAPL"}], {"cash": cash}, [])
        async def fake_run(agent, input, **kwargs):
            output = draft() if agent.name == "investment-manager" else review() if agent.name == "risk-manager" else agent.name
            return SimpleNamespace(final_output=output)
        with tempfile.TemporaryDirectory() as temp, patch.object(ModelFactory, "model_for", return_value="test"), \
             patch.object(Runner, "run", side_effect=fake_run) as run:
            path = Path(temp) / "checkpoints.db"
            initial = await run_agent_graph(reader(), "us", "2026-09-01", {}, require_confirmation=True, checkpoint_path=path)
            self.assertEqual(initial["graph"]["status"], "awaiting_confirmation")
            with self.assertRaisesRegex(RuntimeError, "已变化"):
                await run_agent_graph(reader(999), "us", "2026-09-01", {}, require_confirmation=True,
                    approve_plan=initial["graph"]["plan_hash"], checkpoint_path=path)
            approved = await run_agent_graph(reader(), "us", "2026-09-01", {}, require_confirmation=True,
                approve_plan=initial["graph"]["plan_hash"], checkpoint_path=path)
            self.assertEqual(approved["graph"]["status"], "approved")
            self.assertEqual(approved["input_dataset"], initial["input_dataset"])
            self.assertEqual(approved["agent_sdk"], initial["agent_sdk"])
            self.assertEqual(run.call_count, 4)

    async def test_force_replaces_stale_pending_plan_with_new_hash(self):
        from agents import Runner
        class Host:
            market = "us"
            active_symbols = ["AAPL"]
            news_screen = {"as_of_date": "2026-09-01"}
            async def call(self, *args):
                raise AssertionError("frozen tools only")
        def reader(cash):
            return AnalysisHost(Host(), [{"symbol": "AAPL", "date": "2026-09-01", "close": 100}],
                                [{"symbol": "AAPL"}], {"cash": cash}, [])
        async def fake_run(agent, input, **kwargs):
            return SimpleNamespace(final_output=draft() if agent.name == "investment-manager" else
                                   review() if agent.name == "risk-manager" else agent.name)
        with tempfile.TemporaryDirectory() as temp, patch.object(ModelFactory, "model_for", return_value="test"), \
             patch.object(Runner, "run", side_effect=fake_run) as run:
            path = Path(temp) / "graph.db"
            first = await run_agent_graph(reader(1000), "us", "2026-09-01", {}, require_confirmation=True, checkpoint_path=path)
            renewed = await run_agent_graph(reader(999), "us", "2026-09-01", {}, require_confirmation=True,
                                            checkpoint_path=path, restart=True)
            self.assertNotEqual(first["graph"]["plan_hash"], renewed["graph"]["plan_hash"])
            approved = await run_agent_graph(reader(999), "us", "2026-09-01", {}, require_confirmation=True,
                checkpoint_path=path, approve_plan=renewed["graph"]["plan_hash"])
            self.assertEqual(approved["graph"]["status"], "approved")
            self.assertEqual(run.call_count, 8)
