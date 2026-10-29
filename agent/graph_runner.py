"""并行分析 → 投资计划 → 宿主绑定风控 → 持久化人工确认的 LangGraph 流程。"""
from __future__ import annotations

import asyncio
from copy import deepcopy
from dataclasses import asdict
import os
from pathlib import Path
from time import perf_counter
from typing import TypedDict
from uuid import uuid4

from agent.contracts import capture_risk_review, validate_decision_result, validate_risk_approval
from agent.memory import memory_prompt
from agent.prompt_registry import prompt_manifest
from agent.runtime import ModelBudget, model_settings, positive_env
from agent.sdk_runner import (ModelFactory, _json_output, _system_prompt, build_agent_definitions,
                              build_read_only_tools, build_specialists, parse_agent_result,
                              resolve_role_routing)
from config import DATA_DIR
from evals.experiments import canonical_hash
from agent.evidence import validate_evidence_audit


class GraphState(TypedDict, total=False):
    technical: str
    fundamental: str
    draft: dict
    result: dict
    plan_hash: str
    status: str
    context_fingerprint: str
    input_dataset: dict
    run_metadata: dict


def manager_instructions(market, as_of_date, symbols, version=None):
    if version == "manager-long-v3":
        return manager_instructions(market, as_of_date, symbols, "manager-long-v2").replace("未确认不成交", "独立风控与宿主检查通过才自动排定模拟成交") + (
            "\n本轮为AI自主模拟：独立风控与宿主硬性规则通过后自动排定未来模拟计划，不要求用户逐笔确认。"
            "learning_context仅为已到期结果的历史反馈，不是当前公司外部证据；未到期、失败或缺价格的判断不能算赢输。"
            "参考历史误差与已验证校准信息更新判断，不机械照搬旧操作；策略校准代码由宿主经过测试、对照和回滚检查后更新，角色不得直接写代码或改风险上限、交易时点和证据门槛。")
    from agent.investment import is_long, manager_prompt
    if is_long(version):
        prompt = manager_prompt(market, as_of_date, symbols)
        if version == "manager-long-v2":
            from config import MARKETS, MAX_POSITION_PCT
            from agent.investment import LONG_INSTRUCTIONS
            prompt = (f"你是A股长期模拟投资经理，行情日{as_of_date}，研究池{','.join(symbols)}。" + LONG_INSTRUCTIONS +
                f"股数为{MARKETS[market]['lot_size']}的倍数，单股不超过{MAX_POSITION_PCT:.0%}，受现金及T+1约束。"
                "逐股论证、计划和252交易日预测；估值缺口保持空值并列补证任务，不编造数值。"
                "建仓须研究完成、有业务和财务原始披露引用及明确估值；否则记录核心缺口，7天内复核。"
                "计划参考价为冻结行情价。区分行情日与实际信息截止，只提出未来交易日收盘计划，未确认不成交。"
                "PDF须核对可见表头、单位及限定条件；提取成功不等于语义核验。"
                "引用状态由宿主按支持ID是否为空计算，不表示建仓研究完成；core_gaps另列未解决依据。"
                "只返回输出格式要求的JSON，不自行宣称风控审批或成交。")
        return prompt
    original = _system_prompt(market, as_of_date, symbols, version)
    return (original.split("严格按顺序工作：")[0] +
            "技术与基本面专家已经并行完成，输出在输入 JSON 中。忠实结合两者、账户与新闻，"
            "生成完整候选交易计划和预测，随后由宿主送交独立风控。不得自行声称审批已通过。\n\n硬约束：" +
            original.split("硬约束：", 1)[1])


def build_graph_manifest(market, day, symbols, version=None, model_access="api"):
    rendered = manager_instructions(market, day, symbols, version)
    template = manager_instructions(market, "<AS_OF_DATE>", ["<ACTIVE_SYMBOLS>"], version)
    roles = build_agent_definitions()
    manifest = prompt_manifest(rendered, version, _json_output({
        "graph": "parallel-analysts-plan-risk-confirm-v1", "manager": template,
        "specialists": {name: {"prompt": role.prompt, "tools": role.tools}
                        for name, role in roles.items() if name != "investment-manager"}}))
    if model_access == "subscription":
        from agent.subscription_roles import subscription_manifest
        manifest = subscription_manifest(manifest)
    return manifest


def build_decision_graph(call_role, market: str, symbols: set[str], *, checkpointer=None,
                         require_confirmation: bool = True, capture_metadata=None):
    """节点只生成决策，不拥有账户写入工具；可以注入角色调用做离线验收。"""
    from langgraph.graph import StateGraph, START, END
    from langgraph.types import interrupt

    async def technical(state):
        return {"technical": await call_role("technical-analyst", "分析当前候选池技术面。")}

    async def fundamental(state):
        return {"fundamental": await call_role("fundamental-analyst", "分析当前候选池基本面与新闻。")}

    async def plan(state):
        draft = parse_agent_result(await call_role("investment-manager", _json_output({
            "technical": state["technical"], "fundamental": state["fundamental"]})))
        validate_decision_result(draft, market, expected_symbols=symbols)
        return {"draft": draft}

    async def risk(state):
        submitted = deepcopy(state["draft"]["decisions"])
        payload = {"decisions": submitted}
        if state["draft"].get("investment_cases"):
            payload["investment_cases"] = state["draft"]["investment_cases"]
        review = parse_agent_result(await call_role("risk-manager", _json_output(payload)))
        audit = capture_risk_review(submitted, review, market, symbols)
        result = deepcopy(state["draft"])
        result["risk_review"] = audit
        assessments = {item["symbol"]: item for item in audit["assessments"]}
        for decision in result["decisions"]:
            approval = assessments[decision["symbol"]]
            if approval["max_shares"] == 0:
                decision.update(action="HOLD", shares=0)
            else:
                decision["shares"] = min(decision["shares"], approval["max_shares"])
            decision["reason"] += "；风控：" + approval["reason"]
        result["analyst_views"] = {"technical": state["technical"],
                                   "fundamental": state["fundamental"], "risk": review["summary"]}
        validate_risk_approval(result, market, symbols)
        validate_decision_result(result, market, expected_symbols=symbols)
        saved = capture_metadata() if capture_metadata else {}
        if "evidence_audit" in saved:
            result["evidence_audit"] = saved.pop("evidence_audit")
            validate_evidence_audit(result)
        plan_hash = canonical_hash({"result": result, "context": state.get("context_fingerprint")})
        return {"result": result, "plan_hash": plan_hash, "status": "risk_reviewed", **saved}

    def route(state):
        return ("confirmation" if require_confirmation and any(
            item["action"] != "HOLD" for item in state["result"]["decisions"]) else "finish")

    def confirmation(state):
        answer = interrupt({"status": "awaiting_confirmation", "plan_hash": state["plan_hash"],
                            "decisions": state["result"]["decisions"],
                            "risk_review": state["result"]["risk_review"]})
        if not isinstance(answer, dict) or answer.get("plan_hash") != state["plan_hash"]:
            raise RuntimeError("人工确认的计划哈希不匹配")
        if type(answer.get("approve")) is not bool:
            raise RuntimeError("人工确认必须包含 approve 布尔值")
        result = deepcopy(state["result"])
        if not answer["approve"]:
            for decision in result["decisions"]:
                decision.update(action="HOLD", shares=0, reason=decision["reason"] + "；人工拒绝执行")
        return {"result": result, "status": "approved" if answer["approve"] else "rejected"}

    def finish(state):
        return {"status": state["status"] if state["status"] in {"approved", "rejected"}
                else "analysis_complete"}

    builder = StateGraph(GraphState)
    for name, node in (("technical", technical), ("fundamental", fundamental), ("plan", plan),
                       ("risk", risk), ("confirmation", confirmation), ("finish", finish)):
        builder.add_node(name, node)
    builder.add_edge(START, "technical")
    builder.add_edge(START, "fundamental")
    builder.add_edge(["technical", "fundamental"], "plan")
    builder.add_edge("plan", "risk")
    builder.add_conditional_edges("risk", route, ["confirmation", "finish"])
    builder.add_edge("confirmation", "finish")
    builder.add_edge("finish", END)
    return builder.compile(checkpointer=checkpointer)


async def run_agent_graph(host, market, as_of_date, memory, prompt_version=None, *,
                          require_confirmation=False, approve_plan=None, reject_plan=None,
                          checkpoint_path: Path | None = None, restart: bool = False,
                          model_access: str = "api") -> dict:
    from agents import Agent, Runner, RunConfig
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
    from langgraph.types import Command
    from importlib.metadata import version

    started = perf_counter()
    routing = resolve_role_routing()
    from agent.investment import is_long, case_variant, prepare_context
    long_term = is_long(prompt_version)
    if long_term and (market != "cn" or model_access != "subscription"):
        raise ValueError("长期协议本阶段仅支持A股订阅LangGraph，其他入口请显式使用旧版本")
    manifest = build_graph_manifest(market, as_of_date, host.active_symbols, prompt_version, model_access)
    if model_access not in {"api", "subscription"}:
        raise ValueError("未知模型接入方式")
    if model_access == "subscription":
        from agent.subscription import subscription_settings
        subscription_config = subscription_settings()
        settings = None
        setting_identity = {"effort": subscription_config["reasoning_effort"],
                            "max_output_tokens": None, "input_char_limit": subscription_config["input_char_limit"]}
    else:
        settings = model_settings()
        setting_identity = {"effort": settings.reasoning.effort, "max_output_tokens": settings.max_tokens}
    thread_id = canonical_hash({"market": market, "date": as_of_date, "models": asdict(routing["investment-manager"]),
        "model_access": model_access, "prompt": manifest["template_sha256"],
        "settings": setting_identity, "confirmation": require_confirmation})
    if not require_confirmation:
        thread_id += ":" + uuid4().hex
    path = checkpoint_path or DATA_DIR / "graph_checkpoints.db"
    path.parent.mkdir(parents=True, exist_ok=True)
    # 绑定账户与行情，恢复时数据变化则拒绝旧审批；新闻取稳定事件内容，剔除抓取时刻。
    current_context = {"date": as_of_date, "symbols": host.active_symbols,
        "snapshot": await host.call("candidate_snapshot", {"symbols": host.active_symbols}),
        "account": await host.call("get_account", {}), "memory": memory,
        "news": host.news_screen,
        "fundamentals": await host.call("candidate_fundamentals", {"symbols": host.active_symbols})}
    if getattr(host, "timing", None):
        current_context["timing"] = deepcopy(host.timing)
    if getattr(host, "learning_context", None):
        current_context["learning_context"] = deepcopy(host.learning_context)
    if prompt_version == "manager-long-v3":
        from agent.order_queue import AI_PAPER_POLICY
        current_context["execution_policy"] = AI_PAPER_POLICY
    if long_term:
        identity = case_variant({"agent_sdk": {"prompt": manifest, "role_models": {name: asdict(role) for name, role in routing.items()}, "budget": subscription_config}})
        research = deepcopy(host.investment_context) if getattr(host, "investment_context", None) else await prepare_context(host, as_of_date, host.active_symbols, identity)
        quote_prices = {row["symbol"]: row["close"] for row in current_context["snapshot"]}
        for row in list(research.get("price_history", [])):
            if abs(row["recent_daily"][-1]["close"] - quote_prices[row["symbol"]]) > 1e-6:
                research["price_history"].remove(row)
                research["auxiliary_gaps"].append({"symbol": row["symbol"], "kind": "price_history", "severity": "auxiliary", "reason": "历史末日价格与冻结快照不一致，未提供给模型"})
        host.investment_context = research
        current_context["investment_context"] = research
    context_hash = canonical_hash(current_context)
    factory = ModelFactory() if model_access == "api" else None
    budget = ModelBudget() if model_access == "api" else None
    subscription_roles = None
    if model_access == "subscription":
        from agent.subscription_roles import SubscriptionRoles
        subscription_roles = SubscriptionRoles(current_context, market,
            manager_instructions(market, as_of_date, host.active_symbols, prompt_version),
            routing["investment-manager"].model,
            allow_synthetic=bool(host.news_screen.get("offline")) or
                os.environ.get("ASTOCK_OFFLINE", "").lower() in {"1", "true", "yes"}, long_term=long_term)
    agents_cache = {}

    async def call_role(name, input):
        if subscription_roles is not None:
            return await subscription_roles.call(name, input)
        if not agents_cache:
            agents_cache.update(build_specialists(host, routing, factory))
            tools = build_read_only_tools(host)
            agents_cache["investment-manager"] = Agent(name="investment-manager",
                instructions=manager_instructions(market, as_of_date, host.active_symbols, prompt_version),
                model=factory.model_for(routing["investment-manager"]), model_settings=settings,
                tools=[tools[key] for key in ("market_watchlist_snapshot", "news_candidate_screen", "account_snapshot")])
        result = await Runner.run(agents_cache[name], input + "\n\n" + memory_prompt(memory),
            hooks=budget, max_turns=positive_env("ASTOCK_MAX_TURNS", 24),
            run_config=RunConfig(tracing_disabled=True, trace_include_sensitive_data=False))
        return result.final_output if isinstance(result.final_output, str) else _json_output(result.final_output)

    config = {"configurable": {"thread_id": thread_id}}
    def capture_metadata():
        usage = subscription_roles.client.report() if subscription_roles is not None else budget.report()
        captured = {"input_dataset": host.dataset(as_of_date, memory), "run_metadata": {
            "framework": "langgraph", "sdk_version": version("langgraph"),
            "model_access": model_access, "auth_mode": "chatgpt" if subscription_roles is not None else "api_key",
            "transport": "codex-exec" if subscription_roles is not None else "responses-api",
            "duration_ms": round((perf_counter() - started) * 1000), "budget": usage,
            "role_models": {name: asdict(model) for name, model in routing.items()}, "prompt": manifest,
            "role_usage": deepcopy(subscription_roles.client.calls) if subscription_roles is not None else [],
            "usage": {"requests": usage["requests_started"], "input_tokens": usage["input_tokens"],
                      "output_tokens": usage["output_tokens"], "total_tokens": usage["total_tokens"],
                      "cached_input_tokens": usage.get("cached_input_tokens"),
                      "reasoning_output_tokens": usage.get("reasoning_output_tokens")}}}
        if subscription_roles is not None:
            captured["evidence_audit"] = subscription_roles.audit()
            captured["run_metadata"]["evidence_fingerprint"] = subscription_roles.catalog["fingerprint"]
            if current_context.get("timing"):
                captured["run_metadata"]["role_sources"] = subscription_roles.audit()["role_sources"]
        return captured
    try:
        async with AsyncSqliteSaver.from_conn_string(str(path)) as saver:
            graph = build_decision_graph(call_role, market, set(host.active_symbols),
                checkpointer=saver, require_confirmation=require_confirmation, capture_metadata=capture_metadata)
            previous = await graph.aget_state(config)
            if approve_plan or reject_plan:
                if not previous.interrupts:
                    raise RuntimeError("该行情日没有等待确认的图计划")
                if previous.values.get("context_fingerprint") != context_hash:
                    raise RuntimeError("行情、账户或记忆已变化，旧计划不能恢复；请重新分析")
                supplied = approve_plan or reject_plan
                if supplied != previous.values.get("plan_hash"):
                    raise RuntimeError("审批哈希不对应当前待确认计划")
                invocation = Command(resume={"approve": bool(approve_plan), "plan_hash": supplied})
            elif previous.interrupts and not restart:
                if previous.values.get("context_fingerprint") != context_hash:
                    raise RuntimeError("待确认计划的输入已变化；请用 --force 重新分析后再确认")
                invocation = None
            else:
                invocation = {"context_fingerprint": context_hash}
            if invocation is None:
                state = previous.values
                awaiting = True
            else:
                state = await asyncio.wait_for(graph.ainvoke(invocation, config),
                    timeout=positive_env("ASTOCK_ANALYSIS_TIMEOUT", 600))
                awaiting = bool(state.get("__interrupt__"))
    except Exception as exc:
        if prompt_version in {"manager-long-v2", "manager-long-v3"} and subscription_roles is not None:
            from agent.analysis import write_log
            from agent.trading_clock import now
            from config import LOG_DIR
            write_log(LOG_DIR / "failures" / (as_of_date + "." + uuid4().hex + ".cn.json"), {
                "status": "failed", "error_type": type(exc).__name__, "failed_at": now().isoformat(), "orders_submitted": 0,
                "timing": getattr(host, "timing", {}), "context_fingerprint": context_hash,
                "raw_role_outputs": deepcopy(subscription_roles.raw_outputs), **capture_metadata()})
        raise
    finally:
        if factory is not None:
            await factory.close()
    result = deepcopy(state["result"])
    validate_risk_approval(result, market, set(host.active_symbols))
    result["graph"] = {"status": "awaiting_confirmation" if awaiting else state["status"],
                       "plan_hash": state["plan_hash"], "thread_id": thread_id,
                       "context_fingerprint": context_hash}
    result["input_dataset"] = state["input_dataset"]
    result["agent_sdk"] = state["run_metadata"]
    if model_access == "subscription":
        validate_evidence_audit(result)
    result["graph"]["invocation_duration_ms"] = round((perf_counter() - started) * 1000)
    return result
