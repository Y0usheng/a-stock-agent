# -*- coding: utf-8 -*-
"""组合收益、风险和等权基准评估。"""
from __future__ import annotations

import math
from statistics import mean, pstdev

from evals.metrics import EVALUATION_HISTORY_DAYS, prediction_start_date, uses_synthetic_data

TRADING_PERIODS_PER_YEAR = 252


def _finite(value) -> float | None:
    try:
        number = float(value)
        return number if math.isfinite(number) and number > 0 else None
    except (TypeError, ValueError):
        return None


def performance_metrics(points: list[dict], initial_value: float | None = None,
                        periods_per_year: int = TRADING_PERIODS_PER_YEAR) -> dict:
    """从按期资产点计算收益、波动、回撤、Sharpe 和 Sortino。

    年化收益至少需要20个收益观察，避免几天样本被夸张成年化结果。
    """
    dedup = {}
    for point in points:
        value = _finite(point.get("total"))
        if point.get("date") and value is not None:
            dedup[str(point["date"])] = value
    ordered = sorted(dedup.items())
    values = [value for _, value in ordered]
    start_value = _finite(initial_value) or (values[0] if values else None)
    if not values or start_value is None:
        return {
            "observations": 0, "start_date": None, "end_date": None,
            "total_return_pct": None, "annualized_return_pct": None,
            "annualized_volatility_pct": None, "max_drawdown_pct": None,
            "sharpe_ratio": None, "sortino_ratio": None,
            "positive_period_ratio_pct": None,
        }

    returns = [values[0] / start_value - 1]
    returns.extend(values[index] / values[index - 1] - 1 for index in range(1, len(values)))
    total_return = values[-1] / start_value - 1

    peak = start_value
    max_drawdown = 0.0
    for value in values:
        peak = max(peak, value)
        max_drawdown = min(max_drawdown, value / peak - 1)

    volatility = None
    sharpe = None
    sortino = None
    if len(returns) >= 2:
        period_std = pstdev(returns)
        volatility = period_std * math.sqrt(periods_per_year)
        if period_std > 0:
            sharpe = mean(returns) / period_std * math.sqrt(periods_per_year)
        downside = [min(value, 0.0) for value in returns]
        downside_deviation = math.sqrt(mean(value * value for value in downside))
        if downside_deviation > 0:
            sortino = mean(returns) / downside_deviation * math.sqrt(periods_per_year)

    annualized_return = None
    if len(returns) >= 20 and total_return > -1:
        annualized_return = (1 + total_return) ** (periods_per_year / len(returns)) - 1

    return {
        "observations": len(values), "start_date": ordered[0][0], "end_date": ordered[-1][0],
        "start_value": round(start_value, 4), "end_value": round(values[-1], 4),
        "total_return_pct": round(total_return * 100, 4),
        "annualized_return_pct": round(annualized_return * 100, 4) if annualized_return is not None else None,
        "annualized_volatility_pct": round(volatility * 100, 4) if volatility is not None else None,
        "max_drawdown_pct": round(max_drawdown * 100, 4),
        "sharpe_ratio": round(sharpe, 4) if sharpe is not None else None,
        "sortino_ratio": round(sortino, 4) if sortino is not None else None,
        "positive_period_ratio_pct": round(sum(r > 0 for r in returns) / len(returns) * 100, 2),
        "best_period_pct": round(max(returns) * 100, 4),
        "worst_period_pct": round(min(returns) * 100, 4),
    }


def indexed_curve(points: list[dict], initial_value: float | None = None) -> list[dict]:
    """把资产点转成适合前端绘图的100点指数、区间变动和回撤序列。"""
    dedup = {}
    for point in points:
        value = _finite(point.get("total"))
        if point.get("date") and value is not None:
            dedup[str(point["date"])] = value
    ordered = sorted(dedup.items())
    base = _finite(initial_value) or (ordered[0][1] if ordered else None)
    if base is None:
        return []

    previous = base
    peak = base
    curve = []
    for day, value in ordered:
        peak = max(peak, value)
        change = value - previous
        period_return = value / previous - 1 if previous else 0.0
        curve.append({
            "date": day,
            "total": round(value, 4),
            "index": round(value / base * 100, 4),
            "cumulative_return_pct": round((value / base - 1) * 100, 4),
            "period_change": round(change, 4),
            "period_return_pct": round(period_return * 100, 4),
            "drawdown_pct": round((value / peak - 1) * 100, 4),
        })
        previous = value
    return curve


def equal_weight_curve(frames: dict, dates: list[str]) -> list[dict]:
    """按每只股票首个评估日归一化，构建股票池等权基准。只使用目标日及之前价格。"""
    if not dates or not frames:
        return []
    ordered_dates = sorted(set(dates))
    series = {}
    for symbol, frame in frames.items():
        pairs = sorted((str(row["date"]), float(row["close"])) for _, row in frame.iterrows())
        if pairs:
            series[symbol] = pairs

    def asof(pairs, target):
        eligible = [price for day, price in pairs if day <= target]
        return eligible[-1] if eligible else None

    bases = {symbol: asof(pairs, ordered_dates[0]) for symbol, pairs in series.items()}
    curve = []
    for target in ordered_dates:
        relatives = []
        for symbol, pairs in series.items():
            base = bases.get(symbol)
            current = asof(pairs, target)
            if base and current:
                relatives.append(current / base)
        if relatives:
            curve.append({"date": target, "total": mean(relatives)})
    return curve


def performance_report(logs: list[dict], data_module, watchlist: list[dict]) -> dict:
    """生成组合相对股票池等权基准的收益风险报告。"""
    valid = [log for log in logs if prediction_start_date(log)
             and _finite(log.get("account_after", {}).get("total_assets"))]
    valid.sort(key=prediction_start_date)
    if not valid:
        return {"portfolio": performance_metrics([]), "benchmark": performance_metrics([]),
                "portfolio_curve": [], "benchmark_curve": [],
                "excess_return_pct": None, "trade_count": 0, "turnover_pct": None,
                "data_quality": {"status": "no_data", "message": "暂无有效资产快照"}}

    points = [{"date": prediction_start_date(log),
               "total": log["account_after"]["total_assets"]} for log in valid]
    initial = valid[0]["account_after"].get("initial_cash")
    portfolio = performance_metrics(points, initial)

    source_flags = {uses_synthetic_data(log) for log in valid}
    mixed_sources = len(source_flags) > 1
    synthetic = source_flags == {True}
    fetch = data_module._mock_history if synthetic else data_module._fetch_history
    frames = {}
    fetch_errors = []
    if not mixed_sources:
        for item in watchlist:
            try:
                frames[item["symbol"]] = fetch(item["symbol"], EVALUATION_HISTORY_DAYS)
            except Exception as exc:
                fetch_errors.append(f"{item['symbol']}:{exc}")
    benchmark_curve = equal_weight_curve(frames, [point["date"] for point in points])
    benchmark = performance_metrics(benchmark_curve, 1.0)

    notionals = []
    for log in valid:
        for decision in log.get("decisions", []):
            if decision.get("action") in {"BUY", "SELL"}:
                try:
                    notionals.append(abs(float(decision["shares"]) * float(decision["price"])))
                except (KeyError, TypeError, ValueError):
                    pass
    average_assets = mean(point["total"] for point in points)
    turnover = sum(notionals) / average_assets * 100 if average_assets > 0 else None
    p_return = portfolio.get("total_return_pct")
    b_return = benchmark.get("total_return_pct")
    excess = p_return - b_return if p_return is not None and b_return is not None else None
    status = "ok"
    message = "真实行情同源评估"
    if synthetic:
        message = "离线模拟行情同源评估，仅用于工程回归"
    if mixed_sources:
        status, message = "mixed_sources", "日志混合真实与模拟行情，基准不可比"
    elif fetch_errors:
        status, message = "partial_benchmark", "部分基准标的行情缺失"

    return {
        "portfolio": portfolio, "benchmark": benchmark,
        "portfolio_curve": indexed_curve(points, initial),
        "benchmark_curve": indexed_curve(benchmark_curve, 1.0),
        "excess_return_pct": round(excess, 4) if excess is not None else None,
        "trade_count": len(notionals),
        "turnover_pct": round(turnover, 4) if turnover is not None else None,
        "data_quality": {"status": status, "message": message,
                         "synthetic": synthetic, "fetch_errors": fetch_errors[:5]},
    }
