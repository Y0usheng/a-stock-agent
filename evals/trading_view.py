"""只读模拟账本与真实股价的展示数据；不生成交易或回填虚构收益。"""
from contextlib import closing
from copy import deepcopy
from datetime import date
import json
from pathlib import Path
import sqlite3

import pandas as pd

from config import DATA_DIR, LOG_DIR, MARKETS
from evals.experiments import canonical_hash
from evals.metrics import uses_synthetic_data
from evals.performance import indexed_curve, performance_metrics
from evals.price_data import normalize_history

PRICE_CACHE = DATA_DIR / "market_views" / "cn-prices.json"


def analysis_logs(root=LOG_DIR):
    rows = []
    for path in [*Path(root).glob("*.cn.json"), *(Path(root) / "shadow").glob("*.cn.json"), *(Path(root) / "after_close").glob("*.cn.json")]:
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
            if row.get("market") == "cn" and not uses_synthetic_data(row):
                rows.append(row)
        except (ValueError, OSError):
            continue
    return sorted(rows, key=lambda row: (row.get("as_of_date") or row.get("date", ""), row.get("generated_at", "")))


def read_ledger(path=MARKETS["cn"]["db"]):
    if not Path(path).exists():
        return {"cash": MARKETS["cn"]["initial_cash"], "positions": [], "trades": [], "exists": False}
    with closing(sqlite3.connect(f"file:{Path(path).resolve().as_posix()}?mode=ro", uri=True)) as conn:
        conn.row_factory = sqlite3.Row
        return {"cash": float(conn.execute("SELECT value FROM meta WHERE key='cash'").fetchone()[0]),
                "positions": [dict(row) for row in conn.execute("SELECT * FROM positions WHERE shares>0")],
                "trades": [dict(row) for row in conn.execute("SELECT * FROM trades ORDER BY trade_date,id")], "exists": True}


def _journal(ledger, initial):
    cash, shares, basis, realized = initial, {}, {}, 0.
    for row in ledger["trades"]:
        symbol, quantity, price = row["symbol"], int(row["shares"]), float(row["price"])
        fee = float(row.get("fee") or 0)
        if quantity <= 0 or price <= 0 or fee < 0:
            raise ValueError("无效成交记录")
        if row["action"] == "BUY":
            shares[symbol] = shares.get(symbol, 0) + quantity
            basis[symbol] = basis.get(symbol, 0.) + quantity * price + fee
            cash = round(cash - quantity * price - fee, 2)
        elif row["action"] == "SELL" and shares.get(symbol, 0) >= quantity:
            cost = basis[symbol] / shares[symbol] * quantity
            realized += quantity * price - fee - cost
            shares[symbol] -= quantity
            basis[symbol] -= cost
            cash = round(cash + quantity * price - fee, 2)
        else:
            raise ValueError("成交账本方向或持股数量不一致")
    return cash, {key: value for key, value in shares.items() if value}, realized


def build_view(payload, ledger, logs, initial=1_000_000):
    if payload.get("market") != "cn" or payload.get("data_quality") != "live_prices":
        raise ValueError("交易展示需要明确标注的真实A股价格缓存")
    if canonical_hash(payload["histories"]) != payload.get("fingerprint"):
        raise ValueError("价格缓存指纹不符")
    histories = {symbol: normalize_history(symbol, pd.DataFrame(rows)).to_dict("records")
                 for symbol, rows in payload["histories"].items()}
    if not histories or any(not rows for rows in histories.values()):
        raise ValueError("价格缓存为空")
    end = min(rows[-1]["date"] for rows in histories.values())
    if end > date.today().isoformat():
        raise ValueError("价格缓存包含未来日期")
    prices = {symbol: {row["date"]: row["close"] for row in rows} for symbol, rows in histories.items()}
    valid_logs = [row for row in logs if not uses_synthetic_data(row) and (row.get("as_of_date") or row.get("date", "")) <= end]
    latest = valid_logs[-1] if valid_logs else {}
    observations = [row.get("as_of_date") or row.get("date") for row in valid_logs]
    warnings = []
    if latest.get("timing") and latest.get("execution_mode") == "shadow" and not latest.get("news_screen", {}).get("theme_discovery"):
        warnings.append("本轮为固定受控公司样本验收，未由新闻发现流程自动入选；只分析，不代表建仓。")
    journal_ok = True
    actual_quantities = {row["symbol"]: row["shares"] for row in ledger["positions"]}
    try:
        cash, quantities, realized = _journal(ledger, initial)
        if abs(cash - ledger["cash"]) > .02 or quantities != actual_quantities:
            raise ValueError("成交记录无法与当前现金/持仓对账")
    except ValueError as exc:
        journal_ok, realized = False, None
        warnings.append(str(exc) + "；只展示当前估值，不回填历史组合曲线")
    trade_dates = [row["trade_date"] for row in ledger["trades"]]
    start = min(observations + trade_dates) if observations or trade_dates else end
    if not journal_ok:
        start = end
    if start < min(rows[0]["date"] for rows in histories.values()):
        warnings.append("价格窗口未覆盖全部跟踪期，曲线与回撤仅代表有完整价格的区间")
    days = sorted({row["date"] for rows in histories.values() for row in rows if start <= row["date"] <= end})
    points, missing = [], set()
    for day in days:
        if journal_ok:
            # 只有实际账本成交会改变持仓。HOLD/预测均不会成为买入。
            daily_ledger = {**ledger, "trades": [row for row in ledger["trades"] if row["trade_date"] <= day]}
            daily_cash, held, _ = _journal(daily_ledger, initial)
        else:
            daily_cash, held = ledger["cash"], actual_quantities
        missing_today = {symbol for symbol, quantity in held.items() if quantity and day not in prices.get(symbol, {})}
        if missing_today:
            missing |= missing_today
            continue
        value = daily_cash + sum(quantity * prices[symbol][day] for symbol, quantity in held.items())
        points.append({"date": day, "total": round(value, 2)})
    if missing:
        warnings.append("持仓缺少对应日期真实价格，跳过不完整估值：" + ", ".join(sorted(missing)))
    curve = indexed_curve(points, initial) if journal_ok else []
    current_complete = bool(points and points[-1]["date"] == end)
    if (trade_dates and max(trade_dates) > end) or any(str(row.get("last_buy_date") or "") > end for row in ledger["positions"]):
        current_complete = False
        warnings.append("存在晚于缓存价格的成交，当前盈亏尚不能估值")
    positions = []
    for row in ledger["positions"]:
        close = prices.get(row["symbol"], {}).get(end) if end >= str(row.get("last_buy_date") or "") else None
        value = close * row["shares"] if close is not None else None
        cost = row["avg_cost"] * row["shares"]
        positions.append({**row, "last_price": close, "market_value": round(value, 2) if value is not None else None,
                          "unrealized_pnl": round(value - cost, 2) if value is not None else None})
    names = {row["symbol"]: row["name"] for row in MARKETS["cn"]["candidate_universe"]}
    stocks = []
    decisions = {row["symbol"]: row for row in latest.get("decisions", [])}
    predictions = {row["symbol"]: row for row in latest.get("predictions", [])}
    cases = {row["symbol"]: row for row in latest.get("investment_cases", [])}
    for symbol, rows in histories.items():
        frame = pd.DataFrame(rows)
        ma20, ma60 = frame["close"].rolling(20).mean(), frame["close"].rolling(60).mean()
        series = [{**row, "ma20": round(float(ma20.iloc[index]), 4) if pd.notna(ma20.iloc[index]) else None,
                   "ma60": round(float(ma60.iloc[index]), 4) if pd.notna(ma60.iloc[index]) else None}
                  for index, row in enumerate(rows) if row["date"] <= end]
        last, previous = series[-1], series[-2] if len(series) > 1 else series[-1]
        base = prices[symbol].get(start)
        benchmark = [{"date": row["date"], "return_pct": round((row["close"] / base - 1) * 100, 4)}
                     for row in series if base is not None and row["date"] >= start]
        if base is None:
            warnings.append(symbol + "缺少跟踪起点价格，不补造买入持有对照")
        stocks.append({"symbol": symbol, "name": names.get(symbol, symbol), "series": series,
            "last_close": last["close"], "daily_change_pct": round((last["close"] / previous["close"] - 1) * 100, 3),
            "decision": decisions.get(symbol), "prediction": predictions.get(symbol),
            "investment_case": cases.get(symbol),
            "tracking_benchmark": benchmark,
            "trades": [row for row in ledger["trades"] if row["symbol"] == symbol and row["trade_date"] in prices[symbol]]})
    fees = sum(float(row.get("fee") or 0) for row in ledger["trades"])
    slippage = sum(float(row.get("slippage_cost") or 0) for row in ledger["trades"])
    total = points[-1]["total"] if current_complete else None
    from agent.investment import POLICY, ResearchBook
    try:
        research_tasks = ResearchBook().tasks()
    except sqlite3.Error:
        research_tasks = []
    return {"status": "ready", "market": "cn", "as_of_date": end, "fetched_at": payload.get("fetched_at"),
        "source": payload["source"], "price_basis": payload["price_basis"], "fingerprint": payload["fingerprint"],
        "tracking_start": start, "analysis_date": latest.get("as_of_date") or latest.get("date"),
        "investment_policy": POLICY, "analysis_policy": latest.get("investment_policy", "legacy-short-term"),
        "research_tasks": research_tasks,
        "stocks": stocks, "positions": positions, "trades": ledger["trades"], "portfolio_curve": curve,
        "portfolio": {"initial_cash": initial, "cash": ledger["cash"], "total_assets": total,
            "net_pnl": round(total - initial, 2) if total is not None and journal_ok else None,
            "net_return_pct": round((total / initial - 1) * 100, 4) if total is not None and journal_ok else None,
            "realized_pnl": round(realized, 2) if realized is not None else None,
            "unrealized_pnl": round(sum(row["unrealized_pnl"] for row in positions), 2) if all(row["unrealized_pnl"] is not None for row in positions) else None,
            "total_fees": round(fees, 2), "total_slippage": round(slippage, 2), "trade_count": len(ledger["trades"]),
            "position_count": len(positions), "journal_reconciled": journal_ok},
        "metrics": performance_metrics(points if journal_ok else [], initial), "warnings": sorted(set(warnings)),
        "limitations": ["组合只按实际模拟成交重建，历史股价不等于Agent历史收益；未成交建议不算买入。",
            "成交价已包含滑点，收益已扣实际费用，不重复减滑点。",
            "观察股票对照为不计费的价格涨跌，不代表模拟成交或可实现收益。",
            "价格为前复权，未重建分红、拆股及现金流；单日记录不足以评价策略表现。"]}


def load_view(cache=PRICE_CACHE, log_dir=LOG_DIR, db=None):
    from agent.opportunities import OpportunityBook
    if not Path(cache).exists():
        return {"status": "no_data", "message": "尚未缓存真实股价，请先运行 scripts/refresh_trading_view.py", "opportunities": OpportunityBook().view()}
    payload = json.loads(Path(cache).read_text(encoding="utf-8"))
    logs = analysis_logs(log_dir)
    result = build_view(payload, read_ledger(db or MARKETS["cn"]["db"]), logs)
    result["opportunities"] = OpportunityBook().view(payload.get("histories", {}))
    if logs and logs[-1].get("timing"):
        from agent.order_queue import OrderQueue
        from agent.trading_clock import now
        result["timing"] = deepcopy(logs[-1]["timing"])
        result["pending_orders"] = OrderQueue().public_state(now().date().isoformat())
        result["analysis_date"] = (logs[-1]["timing"].get("decision_at") or logs[-1]["timing"]["information_cutoff_at"])[:10]
    return result


def render_html(view):
    template = Path(__file__).resolve().parent.parent / "dashboard" / "templates" / "trading.html"
    encoded = json.dumps(view, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c")
    return template.read_text(encoding="utf-8").replace("__VIEW_DATA__", encoded)
