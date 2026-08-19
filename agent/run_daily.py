# -*- coding: utf-8 -*-
"""
每日决策循环(MCP client + OpenAI Agents SDK,双市场)
========================================================
流程:按市场连接数据与账户 server → GPT/DeepSeek 多角色只读分析 →
宿主校验并执行计划 → 决策与预测写入 logs/decisions/ 供 evals 评分。

用法:
    python agent/run_daily.py                  # 双市场,混合 Provider 多角色决策
    python agent/run_daily.py --mock           # 双市场,规则策略(无需 API key)
    python agent/run_daily.py --manager-provider deepseek --risk-provider openai
"""
import os
import sys
import json
import asyncio
import argparse
from datetime import date, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import ROOT, LOG_DIR, MAX_POSITION_PCT, MARKETS
from agent.memory import build_memory_snapshot
from agent.contracts import validate_decision_result, reconcile_executions

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

ACCOUNT_SERVER = ROOT / "servers" / "account_server.py"
STRATEGY_VERSION = "phase3-news-dynamic-universe-v2-safety"


def _offline_enabled() -> bool:
    return os.environ.get("ASTOCK_OFFLINE", "").lower() in {"1", "true", "yes"}


def _assert_live_data(snapshot: list[dict], news_screen: dict) -> None:
    """实时模式必须使用可验证真实数据；模拟数据只允许显式离线运行。"""
    if _offline_enabled():
        return
    if not news_screen.get("ok", True):
        raise RuntimeError(news_screen.get("error") or "实时新闻数据不可用，已停止交易")
    if not snapshot:
        raise RuntimeError("行情快照为空，已停止交易")
    sources = [str(item.get("data_source") or "").lower() for item in snapshot]
    if any(not source or any(tag in source for tag in ("mock", "offline", "synthetic", "模拟", "离线"))
           for source in sources):
        raise RuntimeError("实时模式检测到模拟、离线或未知行情数据，已停止交易")


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
            "market": (MARKETS[self.market]["market_server"], {**os.environ}),
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
                   run_id: str = "", trade_date: str = "") -> dict:
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
                                         "trade_date": trade_date})
            executed = bool(r.get("ok"))
            decisions.append({"symbol": sym, "name": s["name"],
                              "action": "SELL" if executed else "HOLD",
                              "shares": held[sym]["shares"] if executed else 0, "price": px,
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
                                            "trade_date": trade_date})
                if r.get("ok"):
                    cash = r["cash_after"]
                executed = bool(r.get("ok"))
                decisions.append({"symbol": sym, "name": s["name"],
                                  "action": "BUY" if executed else "HOLD",
                                  "shares": shares if executed else 0, "price": px,
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
        "summary": f"{MARKETS[host.market]['name']}多角色规则策略(mock)执行完毕",
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
                           run_id: str = "", trade_date: str = "") -> list[dict]:
    """结构验证后由宿主统一执行 SDK 计划；失败订单降级为 HOLD 并保留原因。"""
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
            })
        else:
            response = await host.call("sell", {
                "symbol": decision["symbol"], "shares": decision["shares"],
                "price": decision["price"], "reason": decision["reason"],
                "run_id": run_id, "order_id": _order_id(run_id, decision["symbol"]),
                "trade_date": trade_date,
            })
        if not response.get("ok"):
            errors.append({"symbol": decision["symbol"], "action": action,
                           "error": response.get("error", "未知账户错误")})
            decision["action"] = "HOLD"
            decision["shares"] = 0
            decision["reason"] = f"计划未执行:{response.get('error', '未知账户错误')}"
    return errors


async def run_market(
    market: str,
    mock: bool,
    manager_provider: str | None = None,
    risk_provider: str | None = None,
    force: bool = False,
):
    async with McpHost(market) as host:
        m = MARKETS[market]
        print(f"\n{'='*12} {m['name']} ({market}) {'='*12}")
        print(f"已连接 MCP servers,共 {len(host.tools_schema)} 个工具")
        # 先用核心池确定行情截面日，再让新闻 MCP 从更大的受控宇宙选出每日候选。
        core_snapshot = await host.call("watchlist_snapshot", {})
        # 使用行情截面日期而不是本机日期，避免亚洲时区运行美股任务时错一天。
        as_of_dates = [s.get("date") for s in core_snapshot if s.get("date")]
        as_of_date = max(as_of_dates) if as_of_dates else date.today().isoformat()
        news_screen = await host.call("screen_news_candidates", {
            "as_of_date": as_of_date,
            "limit": m.get("candidate_limit", len(m["watchlist"])),
            "days": 3,
        })
        host.news_screen = news_screen
        account_before = await host.call("get_account", {})
        selected = [item["symbol"] for item in news_screen.get("candidates", [])]
        held_symbols = [item["symbol"] for item in account_before.get("positions", [])]
        fallback = [item["symbol"] for item in m["watchlist"]]
        host.active_symbols = list(dict.fromkeys(selected + held_symbols)) or fallback
        snapshot = await host.call("candidate_snapshot", {"symbols": host.active_symbols})
        _assert_live_data(snapshot, news_screen)
        prices = {s["symbol"]: s["close"] for s in snapshot}
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        log_file = LOG_DIR / f"{as_of_date}.{market}.json"
        if log_file.exists() and not force:
            print(f"本行情日已存在决策日志,为避免重复下单已跳过: {log_file}")
            return
        run_id = f"{market}:{as_of_date}:daily-v1"
        authorization = await host.call("authorize_run", {
            "run_id": run_id, "trade_date": as_of_date,
            "approved_symbols": host.active_symbols,
        })
        if not authorization.get("ok"):
            raise RuntimeError(f"账户运行授权失败:{authorization.get('error', '未知错误')}")
        existing_run_trades = await host.call("get_trades", {"limit": 100, "run_id": run_id})
        if existing_run_trades:
            print(f"账户已存在本轮成交，禁止重复决策/下单并保留原日志: {run_id}")
            return
        recent_trades = await host.call("get_trades", {"limit": 20})
        memory = build_memory_snapshot(market, as_of_date, LOG_DIR, recent_trades)
        if mock:
            result = await run_mock(host, memory, snapshot, run_id, as_of_date)
            mode = "mock_multi_role"
        else:
            from agent.sdk_runner import run_agent_sdk
            result = await run_agent_sdk(
                host,
                market,
                as_of_date,
                memory,
                manager_provider=manager_provider,
                risk_provider=risk_provider,
            )
            mode = "openai_agents_sdk"

        # 记录决策日志(evals 的原始数据)
        expected_symbols = set(host.active_symbols)
        validate_decision_result(result, market, require_roles=True,
                                 expected_symbols=expected_symbols)
        if mode == "openai_agents_sdk":
            result["execution_errors"] = await execute_sdk_plan(
                host, result, prices, run_id, as_of_date
            )
            validate_decision_result(result, market, require_roles=True,
                                     expected_symbols=expected_symbols)
        run_trades = await host.call("get_trades", {"limit": 100, "run_id": run_id})
        result["execution_reconciliation"] = reconcile_executions(result, run_trades)
        account = await host.call("get_account", {"current_prices": prices})
        result["date"] = as_of_date  # 兼容已有日志和评分器
        result["as_of_date"] = as_of_date
        result["generated_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
        result["market"] = market
        result["mode"] = mode
        data_sources = sorted({s.get("data_source") for s in snapshot if s.get("data_source")})
        result["data_sources"] = data_sources
        result["news_screen"] = news_screen
        result["active_symbols"] = host.active_symbols
        result["offline"] = _offline_enabled()
        result["run_id"] = run_id
        result["strategy_version"] = STRATEGY_VERSION
        result["memory"] = memory
        result["account_after"] = account
        log_file.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

        print(f"--- {result['date']} {m['name']}决策总结 ---")
        print(result.get("summary", ""))
        for d in result.get("decisions", []):
            line = "  {:4s} {}({}) {}股 @{} | {}".format(
                d["action"], d["name"], d["symbol"], d["shares"], d["price"], d["reason"])
            print(line)
        print(f"总资产: {account['total_assets']} {m['currency']} (初始 {account['initial_cash']})")
        print(f"日志已写入: {log_file}")


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mock", action="store_true", help="规则策略,无需 API key")
    parser.add_argument("--offline", action="store_true", help="强制使用可复现离线行情")
    parser.add_argument(
        "--manager-provider", choices=["openai", "deepseek"], default=None,
        help="投资经理模型提供商;默认读取 ASTOCK_MANAGER_PROVIDER(openai)",
    )
    parser.add_argument(
        "--risk-provider", choices=["openai", "deepseek"], default=None,
        help="风险经理模型提供商;默认读取 ASTOCK_RISK_PROVIDER(deepseek)",
    )
    parser.add_argument("--force", action="store_true",
                        help="无成交时允许覆盖同日分析日志；已有成交时仍安全跳过")
    parser.add_argument("--market", choices=["cn", "us", "both"], default="both")
    args = parser.parse_args()
    if args.offline:
        os.environ["ASTOCK_OFFLINE"] = "1"
    markets = ["cn", "us"] if args.market == "both" else [args.market]
    for market in markets:
        await run_market(
            market,
            args.mock,
            manager_provider=args.manager_provider,
            risk_provider=args.risk_provider,
            force=args.force,
        )


if __name__ == "__main__":
    asyncio.run(main())
