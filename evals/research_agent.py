"""从研究库的冻结价格/新闻生成只分析预测；不接触正式账本。"""
from dataclasses import asdict
from datetime import date, timedelta
import json
from uuid import uuid4

from agent.analysis import AnalysisHost, write_log
from agent.graph_runner import build_graph_manifest, run_agent_graph
from agent.sdk_runner import resolve_role_routing
from agent.subscription import subscription_settings
from config import MARKETS
from evals.experiments import canonical_hash, decision_log_variant
from evals.research import _available, now
from servers.data_quality import market_close, market_zone

PROTOCOL = "research-frozen-prices-news-no-financials-memory-v1"


def variant_identity(prompt_version="manager-v1", synthetic=False):
    return decision_log_variant({"strategy_version": PROTOCOL, "mode": "langgraph", "execution_mode": "shadow",
        "offline": synthetic, "agent_sdk": {"framework": "langgraph", "model_access": "subscription",
            "budget": subscription_settings(),
            "prompt": build_graph_manifest("cn", "2026-01-01", ["600519"], prompt_version, "subscription"),
            "role_models": {name: asdict(role) for name, role in resolve_role_routing().items()}}})


async def predict_frozen(book, identifier, variant_id, day, *, output, allow_retrospective=False):
    import pandas as pd
    study, frozen = book.get(identifier), book.latest_data(identifier)
    day = date.fromisoformat(day).isoformat()
    if output.suffix.lower() != ".json" or output.resolve() == book.path.resolve():
        raise ValueError("预测输出必须为独立JSON，不能覆盖研究库")
    if output.exists():
        raise ValueError("预测输出文件已存在，不覆盖历史结果")
    from evals.research import code_identity
    if study["code_hash"] != code_identity():
        raise ValueError("注册后的研究代码或配置有变化，未发起模型调用")
    plan = study["plan"]
    variant = next((row for row in plan["variants"] if row["id"] == variant_id), None)
    if variant is None or variant["kind"] != "agent" or frozen is None:
        raise ValueError("需要已冻结输入的预注册Agent变体")
    synthetic = "synthetic" in frozen["data"]["price_quality"] or any(
        row["data_quality"] == "synthetic" for row in frozen["data"]["news_events"])
    expected = variant_identity(variant.get("prompt_version", "manager-v1"), synthetic)
    if canonical_hash(expected) != variant["variant_fingerprint"]:
        raise ValueError("当前模型/Prompt/冻结输入协议与预注册变体不同，未发起模型调用")
    if not any(start <= day <= end for start, end in plan["splits"].values()):
        raise ValueError("预测日不在研究范围内")
    if now().astimezone(market_zone("cn")) < market_close(day, "cn"):
        raise ValueError("预测日尚未收盘")
    if plan["mode"] != "prospective" and not allow_retrospective:
        raise ValueError("历史模型重建需要显式--allow-retrospective；不能用它声称向前验证")
    with book.db() as conn:
        if conn.execute("SELECT 1 FROM research_logs WHERE study_id=? AND variant_id=? AND day=?", (identifier, variant_id, day)).fetchone():
            raise ValueError("该变体该日已有首次输出，不重复消耗订阅额度")
    names = {row["symbol"]: row["name"] for row in MARKETS["cn"]["candidate_universe"]}
    snapshot, candidates = [], []
    for symbol in plan["symbols"]:
        frame = pd.DataFrame([row for row in frozen["data"]["histories"][symbol] if row["date"] <= day]).tail(60)
        if len(frame) < 26 or frame.iloc[-1]["date"] != day:
            raise ValueError("冻结价格缺少当日或预热历史，未发起模型调用")
        close = frame["close"].astype(float)
        delta = close.diff()
        gain, loss = delta.clip(lower=0).rolling(14).mean().iloc[-1], (-delta.clip(upper=0)).rolling(14).mean().iloc[-1]
        dif = close.ewm(span=12).mean() - close.ewm(span=26).mean()
        snapshot.append({**frame.iloc[-1].to_dict(), "symbol": symbol, "name": names[symbol],
            "ma5": round(float(close.rolling(5).mean().iloc[-1]), 2),
            "ma20": round(float(close.rolling(20).mean().iloc[-1]), 2),
            "rsi14": round(float(100 - 100 / (1 + gain / loss) if loss > 0 else 100), 1),
            "macd_dif": round(float(dif.iloc[-1]), 3), "macd_dea": round(float(dif.ewm(span=9).mean().iloc[-1]), 3),
            "above_ma20": bool(close.iloc[-1] > close.rolling(20).mean().iloc[-1]),
            "data_source": "frozen-research:" + frozen["data"]["price_quality"]})
        events = [row for row in frozen["data"]["news_events"] if row["symbol"] == symbol
                  and _available(row, plan["visibility_policy"]) is not None
                  and market_close(day, "cn") - timedelta(days=3) <= _available(row, plan["visibility_policy"])
                  and _available(row, plan["visibility_policy"]) <= market_close(day, "cn")]
        events.sort(key=lambda row: row["published_at"], reverse=True)
        candidates.append({"symbol": symbol, "top_events": events[:2]})
    class Host:
        market = "cn"
        active_symbols = plan["symbols"]
        news_screen = {"as_of_date": day, "offline": synthetic, "candidates": candidates}
        async def call(self, *args):
            raise ValueError("冻结研究不允许额外工具或账户写入")
    account = {"market": "cn", "currency": "CNY", "initial_cash": MARKETS["cn"]["initial_cash"],
               "cash": MARKETS["cn"]["initial_cash"], "total_assets": MARKETS["cn"]["initial_cash"], "positions": []}
    fundamentals = [{"symbol": symbol, "data_source": "unavailable(研究未冻结财务数据)"} for symbol in plan["symbols"]]
    reader = AnalysisHost(Host(), snapshot, fundamentals, account, [])
    with book.db() as conn:
        feature = conn.execute("SELECT input_hash FROM research_inputs WHERE study_id=? AND day=?", (identifier, day)).fetchone()
        if feature and feature["input_hash"] != canonical_hash(reader.dataset(day, {})):
            raise ValueError("同日模型变体的冻结输入不同，未发起模型调用")
        attempt_id = "attempt_" + uuid4().hex
        conn.execute("INSERT INTO research_attempts VALUES(?,?,?,?,?,?,?)", (
            attempt_id, identifier, "predict:" + day, variant_id, now().isoformat(), "running", "{}"))
    try:
        result = await run_agent_graph(reader, "cn", day, {}, variant.get("prompt_version", "manager-v1"), model_access="subscription",
            checkpoint_path=book.path.parent / "research_checkpoints" / (identifier + ".db"))
        result.update(strategy_version=PROTOCOL, mode="langgraph", execution_mode="shadow", offline=synthetic,
            market="cn", date=day, as_of_date=day, generated_at=now().isoformat(),
            execution={"status": "analysis_only", "orders_submitted": 0}, research_id=identifier,
            research_limitations="固定初始虚拟账户；仅冻结价格/新闻，未回放财务或选择性记忆；不能冒充历史实时预测")
        result["dataset_fingerprint"] = canonical_hash(result["input_dataset"])
        write_log(output, result)
        attached = book.attach_log(identifier, variant_id, result)
        usage = result.get("agent_sdk", {}).get("usage", {})
        receipt = {"status": "complete", "log": str(output.resolve()), "receipt": attached,
                   "metrics": {"model_requests": usage.get("requests"), "model_tokens": usage.get("total_tokens")}}
    except BaseException as exc:
        from agent.errors import safe_error
        with book.db() as conn:
            conn.execute("UPDATE research_attempts SET status=?,result_json=? WHERE id=?",
                         ("failed", json.dumps({"reason": safe_error(exc)}, ensure_ascii=False), attempt_id))
        raise
    with book.db() as conn:
        conn.execute("UPDATE research_attempts SET status=?,result_json=? WHERE id=?",
                     ("complete", json.dumps(receipt, ensure_ascii=False), attempt_id))
    return {"status": "attached", "attempt_id": attempt_id, "log": str(output.resolve()), "receipt": attached}
