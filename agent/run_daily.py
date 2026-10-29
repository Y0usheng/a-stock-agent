# -*- coding: utf-8 -*-
"""
每日决策循环(MCP client + 订阅登录/LangGraph,双市场)
========================================================
流程:按市场连接数据与账户 server → GPT 6.1 Sol 多角色只读分析 →
宿主校验并执行计划 → 决策与预测写入 logs/decisions/ 供 evals 评分。

用法:
    python agent/run_daily.py                  # 默认A股，新闻主题发现+长期多角色研究
    python agent/run_daily.py --mock           # A股规则策略(无需 API key)
    python agent/run_daily.py --framework langgraph --shadow
"""
import os
import sys
import json
import asyncio
import argparse
import math
from datetime import date, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import ROOT, LOG_DIR, MAX_POSITION_PCT, MARKETS
from agent.memory import build_memory_snapshot
from agent.contracts import validate_decision_result, reconcile_executions, validate_risk_approval
from agent.analysis import AnalysisHost, shadow_log_path, write_log
from evals.experiments import canonical_hash
from servers.data_quality import snapshot_date
from agent.operations import monitored_markets, preflight
from agent.model_access import resolve_model_access

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

ACCOUNT_SERVER = ROOT / "servers" / "account_server.py"
STRATEGY_VERSION = "phase5-subscription-four-role-v1"


def _offline_enabled() -> bool:
    return os.environ.get("ASTOCK_OFFLINE", "").lower() in {"1", "true", "yes"}


def _assert_live_data(snapshot: list[dict], news_screen: dict, require_news=True) -> None:
    """实时模式必须使用可验证真实数据；模拟数据只允许显式离线运行。"""
    if _offline_enabled():
        return
    if require_news and news_screen.get("ok") is not True:
        raise RuntimeError(news_screen.get("error") or "实时新闻数据不可用，已停止交易")
    if not snapshot:
        raise RuntimeError("行情快照为空，已停止交易")
    sources = [str(item.get("data_source") or "").lower() for item in snapshot]
    if any(not source or any(tag in source for tag in ("mock", "offline", "synthetic", "模拟", "离线"))
           for source in sources):
        raise RuntimeError("实时模式检测到模拟、离线或未知行情数据，已停止交易")
    if any(not isinstance(item.get("volume"), int) or item["volume"] <= 0 for item in snapshot):
        raise RuntimeError("实时模式检测到无效成交量或停牌标的，已停止交易")
    if any(not isinstance(item.get("close"), (int, float)) or not math.isfinite(item["close"])
           or item["close"] <= 0 for item in snapshot):
        raise RuntimeError("实时行情含无效价格，已停止交易")
    day = date.fromisoformat(snapshot_date(snapshot))
    age = (date.today() - day).days
    max_age = int(os.environ.get("ASTOCK_MAX_QUOTE_AGE_DAYS", "10"))
    if age < 0 or max_age < 0 or age > max_age:
        raise RuntimeError("行情日期在未来或超过允许的陈旧天数，停止本轮")


def _order_id(run_id: str, symbol: str) -> str:
    return f"{run_id}:{symbol}"


class McpHost:
    """连接某市场的两个 MCP server(数据+账户),统一路由工具调用"""

    def __init__(self, market: str):
        self.market = market
        self.active_symbols = [item["symbol"] for item in MARKETS[market]["watchlist"]]
        self.news_screen: dict = {}
        self.sessions: dict[str, ClientSession] = {}
        self.tool_map: dict[str, str] = {}   # tool_name -> server_key
        self.tools_schema: list[dict] = []   # 兼容测试与工具数量审计
        self._stack = None

    async def __aenter__(self):
        from contextlib import AsyncExitStack
        self._stack = AsyncExitStack()
        servers = {
            # 显式传递完整环境，确保 ASTOCK_OFFLINE/数据目录等运行参数进入子进程。
            "market": (MARKETS[self.market]["market_server"], {**os.environ, **({"ASTOCK_CN_PRICE_PROVIDER": self.price_provider} if getattr(self, "price_provider", None) else {})}),
            "news": (MARKETS[self.market]["news_server"],
                     {**os.environ, "ASTOCK_MARKET": self.market}),
            "account": (ACCOUNT_SERVER, {**os.environ, "ASTOCK_MARKET": self.market}),
        }
        for key, (script, env) in servers.items():
            params = StdioServerParameters(command=sys.executable, args=[str(script)], env=env)
            read, write = await self._stack.enter_async_context(stdio_client(params))
            session = await self._stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
            self.sessions[key] = session
            for t in (await session.list_tools()).tools:
                self.tool_map[t.name] = key
                self.tools_schema.append({
                    "name": t.name,
                    "description": t.description or "",
                    "input_schema": t.inputSchema,
                })
        return self

    async def __aexit__(self, *exc):
        await self._stack.aclose()

    async def call(self, name: str, args: dict):
        session = self.sessions[self.tool_map[name]]
        result = await session.call_tool(name, args)
        structured = getattr(result, "structuredContent", None)
        texts = [c.text for c in result.content if getattr(c, "type", "") == "text"]
        if os.environ.get("ASTOCK_DEBUG_MCP") == "1":
            print(f"[mcp-debug] {name}: structured={structured!r} texts={texts!r}")
        # MCP 1.29 会把结构化函数结果放到 structuredContent["result"]；
        # 优先使用它可保留“单元素 list”类型，避免被 text fallback 误解为 dict。
        if isinstance(structured, dict) and "result" in structured:
            return structured["result"]
        if structured is not None:
            return structured
        if not texts:
            return {}

        def _parse(t):
            try:
                return json.loads(t)
            except json.JSONDecodeError:
                return t
        # 兼容没有 structuredContent 的旧 MCP server。
        if len(texts) == 1:
            return _parse(texts[0])
        return [_parse(t) for t in texts]


async def run_mock(host: McpHost, memory: dict, snapshot: list[dict] | None = None,
                   run_id: str = "", trade_date: str = "", execute: bool = True) -> dict:
    """规则策略:站上MA20+MACD金叉+RSI未超买 → 买;跌破MA20 → 卖。
    同样通过 MCP 工具执行,用于无 API key 跑通全流程"""
    lot = MARKETS[host.market]["lot_size"]
    snapshot = snapshot or await host.call("watchlist_snapshot", {})
    fundamentals = await host.call("candidate_fundamentals", {
        "symbols": [item["symbol"] for item in snapshot],
    })
    prices = {s["symbol"]: s["close"] for s in snapshot}
    account = await host.call("get_account", {"current_prices": prices})
    held = {p["symbol"]: p for p in account["positions"]}
    cash = account["cash"]
    total = account["total_assets"] or cash
    decisions, predictions = [], []
    news_by_symbol = {
        item["symbol"]: item for item in host.news_screen.get("candidates", [])
    }

    for s in snapshot:
        sym, px = s["symbol"], s["close"]
        bullish = s["above_ma20"] and s["macd_dif"] > s["macd_dea"] and s["rsi14"] < 70
        news_score = float(news_by_symbol.get(sym, {}).get("news_score") or 0)
        severe_negative_news = news_score <= -1.5
        news_confirmed = news_score >= 0
        predictions.append({
            "symbol": sym, "direction": "up" if bullish and news_confirmed else "down",
            "horizon_days": 5,
            "confidence": round(min(0.75, 0.55 + abs(news_score) * 0.04), 2),
            "rationale": (f"MA20{'上方' if s['above_ma20'] else '下方'}, "
                          f"RSI={s['rsi14']}, 新闻事件分={news_score}")})
        if sym in held and (not s["above_ma20"] or severe_negative_news):
            sell_reason = ("重大负面新闻触发风险退出" if severe_negative_news
                           else "跌破MA20,趋势走弱止损/止盈")
            r = await host.call("sell", {"symbol": sym, "shares": held[sym]["shares"],
                                         "price": px, "reason": sell_reason,
                                         "run_id": run_id, "order_id": _order_id(run_id, sym),
                                         "trade_date": trade_date,
                                         "daily_volume": int(s["volume"])}) if execute else {"ok": True, "price": px}
            executed = bool(r.get("ok"))
            decisions.append({"symbol": sym, "name": s["name"],
                              "action": "SELL" if executed else "HOLD",
                              "shares": held[sym]["shares"] if executed else 0,
                              "price": r.get("price", px) if executed else px,
                              "market_price": px,
                              "slippage_cost": r.get("slippage_cost", 0),
                              "reason": r.get("error") or sell_reason})
        elif sym not in held and bullish and news_confirmed:
            budget = min(cash * 0.3, total * (MAX_POSITION_PCT - 0.02))
            shares = int(budget / px / lot) * lot
            if shares >= lot:
                r = await host.call("buy", {"symbol": sym, "name": s["name"], "shares": shares,
                                            "price": px,
                                            "reason": f"技术趋势确认且新闻事件分={news_score}",
                                            "current_prices": prices, "run_id": run_id,
                                            "order_id": _order_id(run_id, sym),
                                            "trade_date": trade_date,
                                            "daily_volume": int(s["volume"])}) if execute else {
                                                "ok": True, "price": px, "cash_after": cash - shares * px}
                if r.get("ok"):
                    cash = r["cash_after"]
                executed = bool(r.get("ok"))
                decisions.append({"symbol": sym, "name": s["name"],
                                  "action": "BUY" if executed else "HOLD",
                                  "shares": shares if executed else 0,
                                  "price": r.get("price", px) if executed else px,
                                  "market_price": px,
                                  "slippage_cost": r.get("slippage_cost", 0),
                                  "reason": r.get("error") or
                                            f"站上MA20+MACD金叉且新闻事件分={news_score}"})
            else:
                decisions.append({"symbol": sym, "name": s["name"], "action": "HOLD",
                                  "shares": 0, "price": px,
                                  "reason": f"可用预算不足以买入最小交易单位{lot}股"})
        else:
            decisions.append({"symbol": sym, "name": s["name"], "action": "HOLD",
                              "shares": 0, "price": px,
                              "reason": ("持有中趋势与新闻风险未触发退出" if sym in held else
                                         f"技术或新闻未形成共振(新闻事件分={news_score})")})
    quality = [f for f in fundamentals if f.get("roe_pct") is not None]
    quality.sort(key=lambda item: item.get("roe_pct") or -999, reverse=True)
    memory_dates = memory.get("selected_dates", [])
    return {
        "summary": f"{MARKETS[host.market]['name']}多角色规则策略(mock)" +
                   ("执行完毕" if execute else "分析完毕（未提交订单）"),
        "analyst_views": {
            "technical": "按MA20、MACD和RSI逐只生成趋势信号；具体证据见 predictions。",
            "fundamental": ("离线基本面质量排序前列: " +
                            ", ".join(item["symbol"] for item in quality[:3])) if quality else "基本面数据不足",
            "risk": (f"新闻只用于候选与风险门控，仍按现金、{MAX_POSITION_PCT:.0%}单股上限"
                     "和市场交收规则执行；账户工具为最终闸门。"),
        },
        "memory_assessment": (f"选择性参考历史日期 {memory_dates}，不机械复用旧交易。"
                              if memory_dates else "没有早于当前截面日的可用历史记忆。"),
        "decisions": decisions, "predictions": predictions,
    }


async def execute_sdk_plan(host: McpHost, result: dict, prices: dict[str, float],
                           volumes: dict[str, int],
                           run_id: str = "", trade_date: str = "") -> list[dict]:
    """结构验证后由宿主统一执行 SDK 计划；失败订单降级为 HOLD 并保留原因。"""
    # 在第一笔订单之前检查整个计划，避免前面的交易执行后才发现后面的审批无效。
    validate_risk_approval(result, host.market, set(prices))
    errors = []
    for decision in result.get("decisions", []):
        action = decision["action"]
        if action == "HOLD":
            continue
        # 成交价由宿主锁定为本轮行情快照，不能采用模型自行生成的价格。
        decision["price"] = float(prices[decision["symbol"]])
        if action == "BUY":
            response = await host.call("buy", {
                "symbol": decision["symbol"], "name": decision["name"],
                "shares": decision["shares"], "price": decision["price"],
                "reason": decision["reason"], "current_prices": prices,
                "run_id": run_id, "order_id": _order_id(run_id, decision["symbol"]),
                "trade_date": trade_date,
                "daily_volume": volumes[decision["symbol"]],
            })
        else:
            response = await host.call("sell", {
                "symbol": decision["symbol"], "shares": decision["shares"],
                "price": decision["price"], "reason": decision["reason"],
                "run_id": run_id, "order_id": _order_id(run_id, decision["symbol"]),
                "trade_date": trade_date,
                "daily_volume": volumes[decision["symbol"]],
            })
        if not response.get("ok"):
            errors.append({"symbol": decision["symbol"], "action": action,
                           "error": response.get("error", "未知账户错误")})
            decision["action"] = "HOLD"
            decision["shares"] = 0
            decision["reason"] = f"计划未执行:{response.get('error', '未知账户错误')}"
        else:
            decision["market_price"] = float(prices[decision["symbol"]])
            decision["price"] = float(response["price"])
            decision["slippage_cost"] = float(response.get("slippage_cost") or 0)
    return errors


async def run_shadow(host: McpHost, snapshot: list[dict], as_of_date: str,
                     mock: bool, manager_provider: str | None, risk_provider: str | None,
                     prompt_versions: list[str | None], force: bool = False,
                     frameworks: list[str] | None = None, model_access: str = "api") -> dict:
    if not mock and model_access == "subscription" and any(
            framework != "langgraph" for framework in frameworks or ["sdk"]):
        raise ValueError("订阅模式只支持 langgraph，不能自动改用 API SDK")
    prices = {item["symbol"]: item["close"] for item in snapshot}
    account = await host.call("get_account", {"current_prices": prices})
    trades = await host.call("get_trades", {"limit": 20})
    memory = build_memory_snapshot(host.market, as_of_date, LOG_DIR, trades)
    fundamentals = await host.call("candidate_fundamentals", {"symbols": host.active_symbols})
    reader = AnalysisHost(host, snapshot, fundamentals, account, trades)
    completed = []
    from itertools import product
    for version, framework in product(dict.fromkeys(prompt_versions), frameworks or ["sdk"]):
        metadata = {
            "strategy_version": STRATEGY_VERSION, "execution_mode": "shadow",
            "mode": "mock_multi_role" if mock else ("langgraph" if framework == "langgraph" else "openai_agents_sdk"),
            "offline": _offline_enabled(),
        }
        if not mock:
            from dataclasses import asdict
            from agent.sdk_runner import build_prompt_manifest, resolve_role_routing, run_agent_sdk
            from agent.runtime import ModelBudget
            from agent.graph_runner import build_graph_manifest, run_agent_graph
            metadata["agent_sdk"] = {
                "framework": "langgraph" if framework == "langgraph" else "openai-agents",
                "model_access": model_access,
                "budget": {},
                "prompt": (build_graph_manifest if framework == "langgraph" else build_prompt_manifest)(
                    host.market, as_of_date, host.active_symbols, version),
                "role_models": {name: asdict(model) for name, model in
                                resolve_role_routing(manager_provider, risk_provider).items()},
            }
            if model_access == "subscription":
                from agent.subscription import subscription_settings
                metadata["agent_sdk"]["budget"] = subscription_settings()
                metadata["agent_sdk"]["prompt"] = build_graph_manifest(
                    host.market, as_of_date, host.active_symbols, version, model_access)
            else:
                metadata["agent_sdk"]["budget"] = ModelBudget().report()
        path = shadow_log_path(LOG_DIR, as_of_date, host.market, metadata)
        if path.exists() and version == "manager-long-v1" and force:
            raise ValueError("长期论证同版本同日保留首次输出；请使用新版本而非覆盖")
        if path.exists() and not force:
            print(f"本行情日该分析变体已有日志，已跳过: {path}")
            continue
        if mock:
            result = await run_mock(reader, memory, snapshot, execute=False)
        else:
            if framework == "langgraph":
                result = await run_agent_graph(reader, host.market, as_of_date, memory, version,
                                               model_access=model_access)
            else:
                result = await run_agent_sdk(reader, host.market, as_of_date, memory,
                                             manager_provider, risk_provider, version)
            validate_risk_approval(result, host.market, set(host.active_symbols))
        validate_decision_result(result, host.market, expected_symbols=set(host.active_symbols))
        dataset = reader.dataset(as_of_date, memory)
        result.update({key: value for key, value in metadata.items() if key != "agent_sdk"})
        result.update({
            "date": as_of_date, "as_of_date": as_of_date, "market": host.market,
            "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "run_id": f"{host.market}:{as_of_date}:shadow:{path.name.split('.')[1]}",
            "data_sources": sorted({item["data_source"] for item in snapshot}),
            "active_symbols": host.active_symbols, "news_screen": host.news_screen,
            "memory": memory, "account_snapshot": account,
            "input_dataset": dataset, "dataset_fingerprint": canonical_hash(dataset),
            "execution": {"status": "analysis_only", "orders_submitted": 0},
        })
        write_log(path, result)
        completed.append((path, result))
        print(f"只分析完成，未提交订单: {path}")
    # 最终指纹包含所有变体访问过的冻结工具数据；单次命令的变体共享这一完整数据集。
    dataset = reader.dataset(as_of_date, memory)
    for path, result in completed:
        result["input_dataset"] = dataset
        result["dataset_fingerprint"] = canonical_hash(dataset)
        write_log(path, result)
        if result.get("investment_policy"):
            from agent.investment import ResearchBook, case_variant
            ResearchBook().save(result, case_variant(result))
    return {"status": "complete" if completed else "skipped",
            "log_path": str(completed[-1][0]) if completed else ""}


async def run_market(
    market: str,
    mock: bool,
    manager_provider: str | None = None,
    risk_provider: str | None = None,
    prompt_version: str | None = None,
    force: bool = False,
    shadow: bool = False,
    compare_prompts: list[str] | None = None,
    framework: str = "sdk",
    compare_frameworks: list[str] | None = None,
    approve_plan: str | None = None,
    reject_plan: str | None = None,
    model_access: str | None = None,
    research_symbols: list[str] | None = None,
):
    model_access = resolve_model_access(model_access)
    from agent.investment import is_long, PROMPT_VERSION
    if prompt_version is None and market == "cn" and not mock and model_access == "subscription":
        prompt_version = compare_prompts[0] if compare_prompts else os.environ.get("ASTOCK_PROMPT_VERSION") or PROMPT_VERSION
    long_term = is_long(prompt_version) or any(is_long(value) for value in compare_prompts or [])
    if long_term and (market != "cn" or mock or model_access != "subscription" or framework != "langgraph"):
        raise ValueError("长期投资协议仅支持A股订阅LangGraph；旧模式须显式选择manager-v1/v2")
    if not mock and model_access == "subscription" and framework != "langgraph":
        raise ValueError("订阅模式只支持 langgraph，不能自动改用 API SDK")
    if research_symbols is not None:
        allowed = {item["symbol"] for item in MARKETS["cn"]["candidate_universe"]}
        if (market != "cn" or not shadow or not research_symbols or len(set(research_symbols)) != len(research_symbols)
                or not set(research_symbols) <= allowed):
            raise ValueError("固定研究池仅支持A股shadow，代码不得重复或越界")
    host_instance = McpHost(market)
    if prompt_version == "manager-long-v2":
        if _offline_enabled():
            raise ValueError("收盘后真实队列不支持offline；离线回归请显式选择旧协议或测试替身")
        host_instance.price_provider = "sina"
    async with host_instance as host:
        if prompt_version == "manager-long-v2":
            if compare_prompts or compare_frameworks:
                raise ValueError("新时序使用独立shadow首次输出，暂不混入旧协议同截面对照")
            from agent.after_close import run_after_close
            return await run_after_close(host, shadow=shadow, research_symbols=research_symbols,
                                         approve_plan=approve_plan, reject_plan=reject_plan, force=force)
        m = MARKETS[market]
        print(f"\n{'='*12} {m['name']} ({market}) {'='*12}")
        print(f"已连接 MCP servers,共 {len(host.tools_schema)} 个工具")
        # 先用核心池确定行情截面日，再让新闻 MCP 从更大的受控宇宙选出每日候选。
        stable_pool = research_symbols or m.get("investment_watchlist", [item["symbol"] for item in m["watchlist"]])
        account_before = None
        if long_term:
            account_before = await host.call("get_account", {})
            if not research_symbols:
                stable_pool = list(dict.fromkeys(stable_pool + [item["symbol"] for item in account_before.get("positions", [])]))
        core_snapshot = await host.call("candidate_snapshot", {"symbols": stable_pool[:1]}) if long_term else await host.call("watchlist_snapshot", {})
        # 使用行情截面日期而不是本机日期，避免亚洲时区运行美股任务时错一天。
        as_of_date = snapshot_date(core_snapshot)
        if long_term:
            _assert_live_data(core_snapshot, {}, require_news=False)
        discovery = None
        if long_term and not research_symbols:
            from agent.opportunities import OpportunityBook, discover
            book = OpportunityBook()
            prior = book.latest()
            if prior and prior["as_of_date"] > as_of_date:
                prior = None
            if approve_plan or reject_plan:
                if not prior or prior["as_of_date"] != as_of_date:
                    raise RuntimeError("缺少本日冻结的新闻发现快照，不能恢复审批")
                discovery = prior
            else:
                try:
                    feed = await asyncio.wait_for(host.call("market_discovery_feed", {"as_of_date": as_of_date}), timeout=25)
                except asyncio.TimeoutError:
                    feed = {**book.feed(as_of_date), "refresh_receipt": {"status": "failed", "error": "broad_feed_total_timeout_25s"}}
                print("发现行业主题：单独最多1次订阅判断，最长75秒；公司分析仍为原4次预算")
                discovery = await discover(feed,
                    held=[item["symbol"] for item in account_before.get("positions", [])],
                    previous_symbols=prior["active_symbols"] if prior else [], book=book,
                    use_model=not _offline_enabled())
            stable_pool = discovery["active_symbols"]
            if not stable_pool:
                print("没有通过长期研究筛选的新闻主题或已有持仓，发现结果已保存，本轮未提交订单")
                return {"status": "skipped", "reason": "no_qualified_opportunity", "discovery": discovery["fingerprint"]}
        screen_args = {
            "as_of_date": as_of_date,
            "limit": len(m["candidate_universe"]) if research_symbols else m.get("candidate_limit", len(m["watchlist"])),
            "days": 3,
        }
        if long_term:
            screen_args.update(symbols=stable_pool, limit=len(stable_pool), days=30)
        news_screen = await host.call("screen_news_candidates", screen_args)
        host.news_screen = news_screen
        if discovery is not None:
            host.news_screen = news_screen = {**news_screen, "theme_discovery": discovery}
        account_before = account_before or await host.call("get_account", {})
        selected = [item["symbol"] for item in news_screen.get("candidates", [])]
        held_symbols = [item["symbol"] for item in account_before.get("positions", [])]
        fallback = [item["symbol"] for item in m["watchlist"]]
        host.active_symbols = list(dict.fromkeys(selected + held_symbols)) or fallback
        if long_term:
            host.active_symbols = list(dict.fromkeys(stable_pool + held_symbols))
        if research_symbols:
            host.active_symbols = list(research_symbols)
            host.news_screen = {**news_screen, "research_fixed_pool": True,
                "candidates": [item for item in news_screen.get("candidates", []) if item["symbol"] in research_symbols]}
        snapshot = await host.call("candidate_snapshot", {"symbols": host.active_symbols})
        if snapshot_date(snapshot) != as_of_date:
            raise RuntimeError("候选池行情日与核心池不一致，停止本轮")
        _assert_live_data(snapshot, news_screen, require_news=not long_term)
        if shadow:
            return await run_shadow(host, snapshot, as_of_date, mock, manager_provider, risk_provider,
                                    compare_prompts or [prompt_version], force,
                                    compare_frameworks or [framework], model_access)
        prices = {s["symbol"]: s["close"] for s in snapshot}
        volumes = {s["symbol"]: int(s["volume"]) for s in snapshot}
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        log_file = LOG_DIR / f"{as_of_date}.{market}.json"
        if long_term and log_file.exists() and force:
            raise ValueError("长期模拟盘保留首次日结果；新版本比较请使用独立shadow日志")
        if log_file.exists() and not force:
            print(f"本行情日已存在决策日志,为避免重复下单已跳过: {log_file}")
            return {"status": "skipped", "log_path": str(log_file)}
        run_id = f"{market}:{as_of_date}:daily-v1"
        existing_run_trades = await host.call("get_trades", {"limit": 100, "run_id": run_id})
        if existing_run_trades:
            if not log_file.exists():
                raise RuntimeError(f"{run_id} 有成交但缺少决策日志，需核对账本后恢复；禁止重新分析或重复下单")
            print(f"账户已存在本轮成交，禁止重复决策/下单并保留原日志: {run_id}")
            return {"status": "skipped", "log_path": str(log_file)}
        recent_trades = await host.call("get_trades", {"limit": 20})
        memory = build_memory_snapshot(market, as_of_date, LOG_DIR, recent_trades)
        async def authorize():
            authorization = await host.call("authorize_run", {
                "run_id": run_id, "trade_date": as_of_date,
                "approved_symbols": host.active_symbols,
            })
            if not authorization.get("ok"):
                raise RuntimeError(f"账户运行授权失败:{authorization.get('error', '未知错误')}")
        if mock:
            await authorize()
            result = await run_mock(host, memory, snapshot, run_id, as_of_date)
            mode = "mock_multi_role"
        else:
            from agent.sdk_runner import run_agent_sdk
            # 正式模型运行也冻结账户、行情和基本面，审批与执行使用同一截面。
            account = await host.call("get_account", {"current_prices": prices})
            fundamentals = await host.call("candidate_fundamentals", {"symbols": host.active_symbols})
            reader = AnalysisHost(host, snapshot, fundamentals, account, recent_trades)
            if framework == "langgraph":
                from agent.graph_runner import run_agent_graph
                result = await run_agent_graph(reader, market, as_of_date, memory, prompt_version,
                    require_confirmation=True, approve_plan=approve_plan, reject_plan=reject_plan, restart=force,
                    model_access=model_access)
                mode = "langgraph"
                if result["graph"]["status"] == "awaiting_confirmation":
                    pending_file = LOG_DIR / "pending" / f"{as_of_date}.{market}.{result['graph']['plan_hash'][:16]}.json"
                    result.update({"date": as_of_date, "market": market, "execution_mode": "pending",
                                   "input_dataset": result.get("input_dataset") or reader.dataset(as_of_date, memory),
                                   "execution": {"status": "awaiting_confirmation", "orders_submitted": 0}})
                    write_log(pending_file, result)
                    print(f"风控复核完成，计划等待人工确认: {pending_file}")
                    options = (f" --model-access {model_access}" + (" --offline" if _offline_enabled() else "")
                               + (f" --prompt-version {prompt_version}" if prompt_version else ""))
                    print(f"确认命令: .venv\\Scripts\\python agent/run_daily.py --market {market} --framework langgraph{options} --approve-plan {result['graph']['plan_hash']}")
                    return {"status": "awaiting_confirmation", "log_path": str(pending_file)}
            else:
                result = await run_agent_sdk(reader, market, as_of_date, memory,
                    manager_provider=manager_provider, risk_provider=risk_provider,
                    prompt_version=prompt_version)
                mode = "openai_agents_sdk"
            dataset = result.get("input_dataset") or reader.dataset(as_of_date, memory)
            result.update(input_dataset=dataset, dataset_fingerprint=canonical_hash(dataset))

        # 记录决策日志(evals 的原始数据)
        expected_symbols = set(host.active_symbols)
        validate_decision_result(result, market, require_roles=True,
                                 expected_symbols=expected_symbols)
        if not mock:
            await authorize()
            result["execution_errors"] = await execute_sdk_plan(
                host, result, prices, volumes, run_id, as_of_date
            )
            validate_decision_result(result, market, require_roles=True,
                                     expected_symbols=expected_symbols)
        run_trades = await host.call("get_trades", {"limit": 100, "run_id": run_id})
        result["execution_reconciliation"] = reconcile_executions(result, run_trades)
        account = await host.call("get_account", {"current_prices": prices})
        data_sources = sorted({s.get("data_source") for s in snapshot if s.get("data_source")})
        valuation_snapshot = await host.call("record_valuation_snapshot", {
            "run_id": run_id, "snapshot_date": as_of_date,
            "current_prices": prices, "data_sources": data_sources,
        })
        if not valuation_snapshot.get("ok"):
            raise RuntimeError(f"每日资产快照写入失败:{valuation_snapshot.get('error', '未知错误')}")
        result["date"] = as_of_date  # 兼容已有日志和评分器
        result["as_of_date"] = as_of_date
        result["generated_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
        result["market"] = market
        result["mode"] = mode
        result["execution_mode"] = "paper"
        result["data_sources"] = data_sources
        result["news_screen"] = news_screen
        result["active_symbols"] = host.active_symbols
        result["offline"] = _offline_enabled()
        result["run_id"] = run_id
        result["strategy_version"] = STRATEGY_VERSION
        result["memory"] = memory
        result["account_after"] = account
        result["valuation_snapshot"] = valuation_snapshot
        write_log(log_file, result)
        if result.get("investment_policy"):
            from agent.investment import ResearchBook, case_variant
            ResearchBook().save(result, case_variant(result))

        print(f"--- {result['date']} {m['name']}决策总结 ---")
        print(result.get("summary", ""))
        for d in result.get("decisions", []):
            line = "  {:4s} {}({}) {}股 @{} | {}".format(
                d["action"], d["name"], d["symbol"], d["shares"], d["price"], d["reason"])
            print(line)
        print(f"总资产: {account['total_assets']} {m['currency']} (初始 {account['initial_cash']})")
        print(f"日志已写入: {log_file}")
        return {"status": "complete", "log_path": str(log_file)}


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mock", action="store_true", help="规则策略,无需 API key")
    parser.add_argument("--offline", action="store_true", help="强制使用可复现离线行情")
    parser.add_argument("--shadow", action="store_true", help="只分析不下单，单独保存各变体日志")
    parser.add_argument("--research-symbols", nargs="+", help="A股shadow研究用的固定受控股票池")
    parser.add_argument("--model-access", choices=["subscription", "api"], default=None,
                        help="默认 subscription，复用官方 ChatGPT 登录；API 必须显式选择")
    parser.add_argument("--framework", choices=["sdk", "langgraph"], default=None)
    parser.add_argument("--compare-frameworks", nargs="+", choices=["sdk", "langgraph"],
                        help="同截面比较编排框架，自动启用 --shadow")
    approval = parser.add_mutually_exclusive_group()
    approval.add_argument("--approve-plan", help="恢复并确认指定完整哈希的 LangGraph 计划")
    approval.add_argument("--reject-plan", help="恢复并拒绝指定完整哈希的 LangGraph 计划")
    parser.add_argument("--preflight", action="store_true", help="只检查配置、登录方式和依赖，不调用模型或下单")
    parser.add_argument("--compare-prompts", nargs="+", choices=["manager-v1", "manager-v2", "manager-long-v1", "manager-long-v2"],
                        help="共享只读截面依次运行多个 Prompt 版本，自动启用 --shadow")
    parser.add_argument(
        "--manager-provider", choices=["openai"], default=None,
        help="投资经理模型提供商;默认读取 ASTOCK_MANAGER_PROVIDER(openai)",
    )
    parser.add_argument(
        "--risk-provider", choices=["openai"], default=None,
        help="风险经理模型提供商;统一 openai",
    )
    parser.add_argument(
        "--prompt-version", choices=["manager-v1", "manager-v2", "manager-long-v1", "manager-long-v2"], default=None,
        help="不可变 Prompt 版本；默认读取 ASTOCK_PROMPT_VERSION(manager-v1)",
    )
    parser.add_argument("--force", action="store_true",
                        help="无成交时允许覆盖同日分析日志；已有成交时仍安全跳过")
    parser.add_argument("--market", choices=["cn", "us", "both"], default="cn")
    args = parser.parse_args()
    args.model_access = resolve_model_access(args.model_access)
    args.framework = args.framework or ("sdk" if args.mock or args.model_access == "api" else "langgraph")
    if not args.mock and args.model_access == "subscription" and (args.framework != "langgraph" or args.compare_frameworks):
        parser.error("订阅接入使用 --framework langgraph；SDK/框架对照需显式选择 --model-access api")
    if args.compare_prompts or args.compare_frameworks:
        if args.mock:
            parser.error("版本/框架对照需要模型分析，不能使用 --mock")
        args.shadow = True
    if args.research_symbols and (args.market != "cn" or not args.shadow):
        parser.error("--research-symbols需要--market cn --shadow")
    if args.mock and args.framework != "sdk":
        parser.error("--mock 是规则演示；LangGraph 需要模型，离线图验收请运行测试")
    if args.approve_plan or args.reject_plan:
        if args.framework != "langgraph" or args.shadow or args.mock or args.market == "both" or args.force:
            parser.error("审批需要 --framework langgraph、单市场，并且不能 --shadow/--mock")
    if args.offline:
        os.environ["ASTOCK_OFFLINE"] = "1"
    check = preflight(args.mock, args.framework, args.model_access)
    if args.preflight:
        print(json.dumps(check, ensure_ascii=False, indent=2))
        raise SystemExit(0 if check["ok"] else 2)
    if not check["ok"]:
        print(json.dumps(check, ensure_ascii=False, indent=2))
        raise SystemExit(2)
    markets = ["cn", "us"] if args.market == "both" else [args.market]
    results = await monitored_markets(markets, run_market, {
        "mock": args.mock, "manager_provider": args.manager_provider, "risk_provider": args.risk_provider,
        "prompt_version": args.prompt_version, "force": args.force, "shadow": args.shadow,
        "compare_prompts": args.compare_prompts, "framework": args.framework,
        "compare_frameworks": args.compare_frameworks, "approve_plan": args.approve_plan,
        "reject_plan": args.reject_plan, "model_access": args.model_access, "research_symbols": args.research_symbols})
    if any(item["status"] == "failed" for item in results):
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())
