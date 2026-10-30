"""New long-term daily path: captured information -> future confirmed paper orders."""
import asyncio
import os
from dataclasses import asdict
from copy import deepcopy

from agent.analysis import AnalysisHost, shadow_log_path, write_log
from agent.contracts import validate_decision_result, validate_risk_approval
from agent.graph_runner import build_graph_manifest, run_agent_graph
from agent.investment import PROMPT_VERSION, ResearchBook, case_variant, prepare_context
from agent.memory import build_memory_snapshot
from agent.opportunities import OpportunityBook, discover
from agent.order_queue import AI_PAPER_POLICY, OrderQueue, execute_due
from agent.sdk_runner import resolve_role_routing
from agent.subscription import subscription_settings
from agent.trading_clock import make_timing, now
from config import LOG_DIR, MARKETS
from evals.experiments import canonical_hash
from servers.data_quality import snapshot_date


async def run_after_close(host, *, shadow=False, research_symbols=None, approve_plan=None, reject_plan=None, force=False, prompt_version=PROMPT_VERSION):
    queue = OrderQueue()
    if approve_plan or reject_plan:
        receipt = queue.confirm(approve_plan or reject_plan, approve=bool(approve_plan))
        print(receipt)
        return receipt
    if force:
        raise ValueError("收盘后协议保留首次输出和队列，不支持强制覆盖")
    if prompt_version == "manager-long-v3" and not shadow:
        from agent.learning import review_cached
        review_cached(now().isoformat())
    if not shadow:
        receipts = await execute_due(host, queue)
        if receipts:
            path = LOG_DIR / "executions" / (now().date().isoformat() + ".cn.json")
            write_log(path, {"processed_at": now().isoformat(), "receipts": receipts, "record_kind": "queued_execution_receipts"})
        if any(row["status"] in {"blocked", "execution_pending"} for row in receipts):
            raise RuntimeError("既有次交易日计划尚未结清，已保留成交与队列，停止新增计划")
        if any(row["status"] == "blocked" for row in queue.rows()):
            raise RuntimeError("既有计划因账户变化被冻结，请先在本机取消或复核")
    probe_symbols = research_symbols or MARKETS["cn"]["investment_watchlist"][:1]
    probe = await host.call("candidate_snapshot", {"symbols": probe_symbols})
    from agent.run_daily import _assert_live_data
    _assert_live_data(probe, {}, require_news=False)
    day = snapshot_date(probe)
    calendar = await host.call("trading_sessions", {})
    preliminary = make_timing(day, calendar["sessions"], calendar_hash=calendar["fingerprint"])
    formal_path = LOG_DIR / "after_close" / (day + (".long-v3" if prompt_version == "manager-long-v3" else "") + ".cn.json")
    if formal_path.exists() and not shadow:
        return {"status": "skipped", "log_path": str(formal_path)}
    account = await host.call("get_account", {})
    held = [row["symbol"] for row in account["positions"]]
    discovery = None
    if research_symbols:
        active = list(research_symbols)
    else:
        book = OpportunityBook()
        prior = book.latest() or {}
        watched = list(dict.fromkeys(prior.get("research_watchlist", []) + prior.get("active_symbols", []) + MARKETS["cn"]["investment_watchlist"]))
        from evals.trading_view import analysis_logs
        baseline = ResearchBook().review_order(watched, day, analysis_logs())
        resume = os.environ.get("ASTOCK_RESUME_NEWS") or None
        feed = {"as_of_date": day, "information_cutoff_at": preliminary["information_cutoff_at"]} if resume else await asyncio.wait_for(host.call("market_discovery_feed", {"as_of_date": day, "after_close": True}), timeout=25)
        discovery = await discover(feed, held=held, previous_symbols=prior.get("active_symbols", []), book=book, limit=2, resume_snapshot=resume, baseline_symbols=baseline)
        active = discovery["active_symbols"]
        if discovery.get("review_error"):
            raise RuntimeError("新闻主题评审失败，未形成有效候选判断；请检查发现记录，本轮未进入投资经理")
    if not active:
        if discovery and discovery.get("review_error"):
            raise RuntimeError("新闻主题评审失败，未形成有效候选判断；请检查发现记录，本轮未进入投资经理")
        print("没有通过长期筛选的新机会，保留发现记录，未产生计划或成交。")
        return {"status": "skipped", "reason": "no_qualified_opportunity"}
    host.active_symbols = active
    snapshot = probe if active == probe_symbols else await host.call("candidate_snapshot", {"symbols": active})
    if snapshot_date(snapshot) != day:
        raise ValueError("候选行情日不一致")
    _assert_live_data(snapshot, {}, require_news=False)
    from agent.disclosures import collect
    print("自动补证：本轮最多2家公司，每家最新年报和最新中期报告；总预算90秒，零模型调用。")
    collection = await asyncio.wait_for(asyncio.to_thread(collect, active[:2], budget_seconds=90), timeout=100)
    screen = await asyncio.wait_for(host.call("after_close_symbol_news", {"symbols": active, "as_of_date": day}), timeout=max(25, 15 * len(active)))
    if discovery:
        screen["theme_discovery"] = discovery
    host.news_screen = screen
    try:
        fundamentals = await asyncio.wait_for(host.call("candidate_fundamentals", {"symbols": active}), timeout=35)
    except (asyncio.TimeoutError, RuntimeError):
        fundamentals = [{"symbol": symbol, "ok": False, "data_source": "unavailable(bounded_fundamentals_request)"} for symbol in active]
    prices = {row["symbol"]: row["close"] for row in snapshot}
    account = await host.call("get_account", {"current_prices": prices})
    recent = await host.call("get_trades", {"limit": 20})
    memory = build_memory_snapshot("cn", day, LOG_DIR, recent, include_after_close=prompt_version == "manager-long-v3", cutoff_at=preliminary["information_cutoff_at"] if prompt_version == "manager-long-v3" else None)
    manifest = build_graph_manifest("cn", day, active, prompt_version, "subscription")
    metadata = {"agent_sdk": {"prompt": manifest, "role_models": {name: asdict(model) for name, model in resolve_role_routing().items()}, "budget": subscription_settings()}}
    path = shadow_log_path(LOG_DIR, day, "cn", metadata) if shadow else formal_path
    if path.exists():
        return {"status": "skipped", "log_path": str(path)}
    host.timing = make_timing(day, calendar["sessions"], calendar_hash=calendar["fingerprint"])
    reader = AnalysisHost(host, snapshot, fundamentals, account, recent)
    reader.investment_context = await prepare_context(reader, day, active, case_variant(metadata))
    reader.timing = make_timing(day, calendar["sessions"], calendar_hash=calendar["fingerprint"])
    learning = None
    if prompt_version == "manager-long-v3":
        from agent.learning import prepare, calibrate_predictions
        learning = await prepare(reader, day, calendar["sessions"], reader.timing["information_cutoff_at"], shadow=shadow)
        reader.learning_context = learning
    result = await run_agent_graph(reader, "cn", day, memory, prompt_version, require_confirmation=False, model_access="subscription")
    if learning:
        calibrate_predictions(result, learning)
        result["learning"] = learning
    validate_decision_result(result, "cn", expected_symbols=set(active))
    validate_risk_approval(result, "cn", set(active))
    dataset = reader.dataset(day, memory)
    result.update(date=day, as_of_date=day, market="cn", generated_at=now().isoformat(), timing=reader.timing,
        run_id="cn:" + day + ":" + prompt_version, active_symbols=active, input_dataset=dataset, dataset_fingerprint=canonical_hash(dataset),
        account_snapshot=account, memory=memory, news_screen=screen, data_sources=sorted({row["data_source"] for row in snapshot}),
        collection_receipt=collection, execution_mode="shadow" if shadow else "paper",
        execution={"status": "analysis_only" if shadow else "no_orders", "orders_submitted": 0})
    result["timing"] = {**reader.timing, "decision_at": result["generated_at"]}
    if not shadow:
        autonomous = prompt_version == "manager-long-v3"
        if autonomous:
            result["execution_policy"] = AI_PAPER_POLICY
        identifier = queue.stage(result, result["timing"], account, auto_confirm=autonomous)
        if identifier:
            result["queue_plan_hash"] = identifier
            if autonomous:
                result["execution"] = {"status": "ai_scheduled_for_future_session", "orders_submitted": 0}
                print("AI模拟计划已自动排定；目标交易日", reader.timing["execution_session"], "；当前未成交。")
            else:
                result.update(execution_mode="pending", execution={"status": "awaiting_confirmation_for_future_session", "orders_submitted": 0})
                pending = LOG_DIR / "pending" / (day + ".cn." + identifier[:16] + ".json")
                write_log(pending, result)
                print("未来收盘模拟计划待确认；执行日", reader.timing["execution_session"], "确认期限", reader.timing["approval_deadline"])
                print("本机确认：.venv\\Scripts\\python agent/run_daily.py --market cn --approve-plan " + identifier)
        else:
            result["execution"]["status"] = "analysis_complete_no_trade_plan"
    write_log(path, result)
    ResearchBook().save(result, case_variant(result))
    print("收盘后研究完成：", path, "；当日未按旧收盘价成交。")
    return {"status": "awaiting_confirmation" if result["execution_mode"] == "pending" else "complete", "log_path": str(path)}
