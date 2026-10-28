"""长周期阶段结果与空仓机会成本；零收益不会自动成为策略成功。"""
import pandas as pd
from agent.investment import VERSION
from evals.experiments import canonical_hash, decision_log_variant
from evals.metrics import actual_direction_from_frame


def evaluate_investments(logs, histories, portfolio_view=None):
    rows = []
    for log in logs:
        if log.get("investment_policy") != VERSION:
            continue
        day = log.get("as_of_date") or log.get("date")
        cases = {row["symbol"]: row for row in log["investment_cases"]}
        for prediction in log["predictions"]:
            frame = pd.DataFrame(histories.get(prediction["symbol"], [])).sort_values("date") if histories.get(prediction["symbol"]) else pd.DataFrame()
            actual = actual_direction_from_frame(frame, day, 252) if not frame.empty else None
            prices = frame[frame["date"] >= day] if not frame.empty else frame
            exact_start = not prices.empty and prices.iloc[0]["date"] == day
            price_return = round((float(prices.iloc[-1]["close"]) / float(prices.iloc[0]["close"]) - 1) * 100, 4) if exact_start else None
            case = cases[prediction["symbol"]]
            account = log.get("input_dataset", {}).get("account") or log.get("account_snapshot", {})
            empty_book = not any(item.get("shares", 0) > 0 for item in account.get("positions", [])) and bool(account.get("total_assets")) and account.get("cash", 0) / account["total_assets"] >= .9999
            rows.append({"date": day, "symbol": prediction["symbol"], "horizon_days": 252,
                "status": "matured" if actual is not None else "pending", "actual": actual,
                "correct": actual == prediction["direction"] if actual is not None else None,
                "research_status": case["research_status"], "position_intent": case["position_intent"],
                "observed_price_return_pct": price_return,
                "cash_vs_stock_opportunity_gap_pct": -price_return if empty_book and price_return is not None and case["position_intent"] in {"research_pending", "wait_price"} else None,
                "benchmark_note": "现金相对不计费股票价格的示意差值，并非实际账户收益归因",
                "variant": decision_log_variant(log), "source_log_fingerprint": canonical_hash(log)})
    mature = [row for row in rows if row["status"] == "matured"]
    portfolio = (portfolio_view or {}).get("portfolio", {})
    return {"policy": VERSION, "status": "awaiting_outcomes" if not mature else "descriptive_only",
        "rows": rows, "matured_count": len(mature), "pending_count": len(rows) - len(mature),
        "direction_accuracy_pct": sum(row["correct"] for row in mature) / len(mature) * 100 if mature else None,
        "actual_simulated_portfolio": portfolio,
        "current_cash_fraction": portfolio.get("cash", 0) / portfolio["total_assets"] if portfolio.get("total_assets") else None,
        "strategy_effect_verified": False,
        "interpretation": "252日方向仅为阶段观察；交易优劣以实际扣费净收益、回撤及同口径基准评价，空仓0收益不自动通过。"}
