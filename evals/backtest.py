# -*- coding: utf-8 -*-
"""60–120交易日滚动回测与新闻开/关 A/B Evals。

信号只读取信号日及以前数据，订单统一在下一交易日开盘成交，并施加与模拟账户一致的
滑点、佣金和成交量参与率。默认 CLI 使用可复现离线行情/事件，只用于工程与策略比较。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import EXPERIMENT_DB, MARKETS, NEWS_DB, MAX_POSITION_PCT
from evals.price_data import normalize_history, load_price_dataset, export_price_dataset
from evals.experiments import record_experiment
from evals.news_store import load_news_scores
from evals.performance import indexed_curve, performance_metrics


def _fee(config: dict, amount: float, is_sell: bool) -> float:
    commission = (max(amount * config["commission_rate"], config["commission_min"])
                  if config["commission_rate"] > 0 else 0.0)
    tax = amount * config["stamp_tax_rate"] if is_sell else 0.0
    return round(commission + tax, 2)


def _fill_price(config: dict, market_price: float, is_sell: bool) -> float:
    direction = -1 if is_sell else 1
    return round(market_price * (1 + direction * config.get("slippage_bps", 0) / 10_000), 4)


def _offline_news_score(market: str, symbol: str, day: str) -> float:
    values = (-2.0, 0.0, 1.0, 2.0)
    digest = hashlib.sha256(f"{market}:{symbol}:{day}:news-ab-v1".encode()).digest()
    return values[digest[0] % len(values)]


def _technical_signal(frame: pd.DataFrame, day: str) -> dict | None:
    history = frame[frame["date"].astype(str) <= day].tail(60)
    if len(history) < 26:
        return None
    close = history["close"].astype(float)
    ma20 = close.rolling(20).mean().iloc[-1]
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean().iloc[-1]
    loss = (-delta.clip(upper=0)).rolling(14).mean().iloc[-1]
    rsi = 100 - 100 / (1 + gain / loss) if loss > 0 else 100.0
    dif = close.ewm(span=12).mean() - close.ewm(span=26).mean()
    dea = dif.ewm(span=9).mean()
    return {
        "bullish": bool(close.iloc[-1] > ma20 and dif.iloc[-1] > dea.iloc[-1] and rsi < 70),
        "above_ma20": bool(close.iloc[-1] > ma20),
        "rsi14": float(rsi),
    }


def walk_forward_backtest(histories: dict[str, pd.DataFrame], market: str,
                          window_days: int = 90, use_news: bool = True,
                          news_scores: dict[tuple[str, str], float] | None = None,
                          dynamic_candidates: bool = False,
                          attention_by_day: dict | None = None, *,
                          strategy: str = "technical", prediction_signals: dict | None = None,
                          start_date: str | None = None, end_date: str | None = None,
                          config_override: dict | None = None, target_weight: float = .28,
                          position_cap: float | None = None, signal_cache: dict | None = None) -> dict:
    """运行无未来数据的收盘信号/次日开盘成交回测。"""
    if not 60 <= int(window_days) <= 120:
        raise ValueError("window_days 必须在60到120之间")
    if strategy not in {"technical", "buy_hold", "agent_predictions"}:
        raise ValueError("未知回测策略")
    config = {**MARKETS[market], **(config_override or {})}
    cap = MAX_POSITION_PCT if position_cap is None else float(position_cap)
    if not 0 < target_weight <= cap <= 1:
        raise ValueError("目标权重必须为正且不超过仓位上限")
    normalized = {symbol: normalize_history(symbol, frame) for symbol, frame in histories.items()}
    common_dates = sorted(set.intersection(*(
        set(frame["date"]) for frame in normalized.values()
    ))) if normalized else []
    if start_date is not None or end_date is not None:
        if not start_date or not end_date or start_date > end_date:
            raise ValueError("自定义回测区间必须同时给出起止日期")
        evaluation_dates = [day for day in common_dates if start_date <= day <= end_date]
        if len(evaluation_dates) < 2 or common_dates.index(evaluation_dates[0]) < 26:
            raise ValueError("区间不足2个共同交易日或缺少26日指标预热")
    else:
        if len(common_dates) < window_days + 26:
            raise ValueError(f"共同历史仅{len(common_dates)}日，不足回测窗口和指标预热")
        evaluation_dates = common_dates[-window_days:]
    rows = {symbol: frame.set_index("date") for symbol, frame in normalized.items()}
    cash = float(config["initial_cash"])
    positions: dict[str, int] = {}
    pending: list[dict] = []
    trades = []
    decisions = []
    curve = []
    candidate_history = []
    lot = int(config["lot_size"])

    for index, day in enumerate(evaluation_dates):
        # 只执行上一交易日已经生成的订单，成交价取当日开盘。
        for order in sorted(pending, key=lambda item: item["action"] != "SELL"):
            symbol, action = order["symbol"], order["action"]
            row = rows[symbol].loc[day]
            market_price = float(row["open"])
            # 开盘时还不知道当天总成交量；使用信号日已知的量作保守限额。
            daily_volume = int(rows[symbol].loc[order["signal_date"]]["volume"])
            max_shares = int(daily_volume * config["max_volume_participation_pct"] / lot) * lot
            shares = min(int(order["shares"]), max_shares)
            shares = shares // lot * lot
            if shares < lot:
                continue
            is_sell = action == "SELL"
            price = _fill_price(config, market_price, is_sell)
            amount = shares * price
            fee = _fee(config, amount, is_sell)
            if action == "BUY":
                opening_total = cash + sum(held * float(rows[ticker].loc[day]["open"])
                                           for ticker, held in positions.items())
                held_value = positions.get(symbol, 0) * market_price
                capacity = max(0, (opening_total * cap - held_value)
                               / (price + cap * (price - market_price)))
                shares = min(shares, int(capacity / lot) * lot)
                affordable = int(cash / (price * lot)) * lot
                shares = min(shares, affordable)
                while shares >= lot:
                    amount = shares * price
                    fee = _fee(config, amount, False)
                    after_total = opening_total - (price - market_price) * shares - fee
                    if amount + fee <= cash and held_value + amount <= after_total * cap:
                        break
                    shares -= lot
                if shares < lot:
                    continue
                cash -= amount + fee
                positions[symbol] = positions.get(symbol, 0) + shares
            else:
                shares = min(shares, positions.get(symbol, 0))
                if shares < lot:
                    continue
                amount = shares * price
                fee = _fee(config, amount, True)
                cash += amount - fee
                positions[symbol] -= shares
                if positions[symbol] == 0:
                    del positions[symbol]
            trades.append({
                "signal_date": order["signal_date"], "trade_date": day,
                "symbol": symbol, "action": action, "shares": shares,
                "market_price": market_price, "price": price, "fee": fee,
                "liquidity_reference_date": order["signal_date"],
                "liquidity_reference_volume": daily_volume,
                "slippage_cost": round(abs(price - market_price) * shares, 2),
            })
        pending = []

        closes = {symbol: float(rows[symbol].loc[day]["close"]) for symbol in normalized}
        total = cash + sum(shares * closes[symbol] for symbol, shares in positions.items())
        curve.append({"date": day, "total": total})
        if index == len(evaluation_dates) - 1:
            continue

        scores = {symbol: (float(news_scores.get((day, symbol), 0.0)) if news_scores is not None
                          else _offline_news_score(market, symbol, day)) for symbol in normalized}
        core = {item["symbol"] for item in config["watchlist"]}
        attention = (attention_by_day or {}).get(day, {})
        candidates = (sorted(normalized, key=lambda symbol:
                      (-float(attention.get(symbol, abs(scores[symbol]) if attention_by_day is None else 0))
                       - (0.05 if symbol in core else 0), -abs(scores[symbol]), symbol))
                      [:config["candidate_limit"]] if dynamic_candidates else list(normalized))
        candidates = list(dict.fromkeys(candidates + list(positions)))
        candidate_history.append({"date": day, "symbols": candidates})
        for symbol in candidates:
            frame = normalized[symbol]
            key = (symbol, day)
            if signal_cache is not None and key in signal_cache:
                signal = signal_cache[key]
            else:
                signal = _technical_signal(frame, day)
                if signal_cache is not None:
                    signal_cache[key] = signal
            if signal is None and strategy == "technical":
                continue
            if not use_news:
                score = 0.0
            else:
                score = scores[symbol]
            held = positions.get(symbol, 0)
            action = "HOLD"
            shares = 0
            if strategy == "buy_hold":
                if index == 0 and not held:
                    shares = int(total * target_weight / closes[symbol] / lot) * lot
                    action = "BUY" if shares >= lot else "HOLD"
            elif strategy == "agent_predictions":
                prediction = (prediction_signals or {}).get((day, symbol))
                if prediction is None:
                    raise ValueError(f"缺少预先保存的模型预测：{day}/{symbol}")
                if held and prediction["direction"] == "down":
                    action, shares = "SELL", held
                elif not held and prediction["direction"] == "up":
                    shares = int(total * target_weight / closes[symbol] / lot) * lot
                    action = "BUY" if shares >= lot else "HOLD"
            elif held and (not signal["above_ma20"] or (use_news and score <= -1.5)):
                action, shares = "SELL", held
            elif not held and signal["bullish"] and (not use_news or score >= 0):
                budget = total * target_weight
                shares = int(budget / closes[symbol] / lot) * lot
                if shares >= lot:
                    action = "BUY"
            decision = {"signal_date": day, "symbol": symbol, "action": action,
                        "shares": shares, "news_score": score, "use_news": use_news}
            decisions.append(decision)
            if action != "HOLD":
                pending.append(decision)

    metrics = performance_metrics(curve, config["initial_cash"])
    return {
        "market": market, "window_days": len(evaluation_dates), "use_news": use_news,
        "execution": "signal close / next trading day open",
        "metrics": metrics, "curve": indexed_curve(curve, config["initial_cash"]),
        "trades": trades, "decisions": decisions,
        "candidate_history": candidate_history,
        "strategy": ("bounded-buy-hold-v1" if strategy == "buy_hold" else
                     "frozen-prediction-allocation-v1" if strategy == "agent_predictions" else
                     "technical-news-gate-v2" if dynamic_candidates else "technical-news-gate-fixed-v2"),
        "limitations": ("保存预测的统一仓位重放，不等同原Agent逐笔交易计划。" if strategy == "agent_predictions" else
                        "规则策略基线；未回放 GPT、多角色基本面或记忆。") + "成交量限额取信号日，未模拟开盘盘口。",
        "total_fees": round(sum(item["fee"] for item in trades), 2),
        "total_slippage_cost": round(sum(item["slippage_cost"] for item in trades), 2),
    }


def news_ab_eval(histories: dict[str, pd.DataFrame], market: str,
                 window_days: int = 90,
                 news_scores: dict[tuple[str, str], float] | None = None,
                 news_manifest: dict | None = None,
                 dynamic_candidates: bool = False) -> dict:
    attention = (news_manifest or {}).get("attention_by_day")
    with_news = walk_forward_backtest(histories, market, window_days, True, news_scores, dynamic_candidates, attention)
    without_news = walk_forward_backtest(histories, market, window_days, False, news_scores, dynamic_candidates, attention)
    left = with_news["metrics"].get("total_return_pct")
    right = without_news["metrics"].get("total_return_pct")
    return {
        "market": market, "window_days": window_days,
        "data_quality": ((news_manifest or {}).get("data_quality") if news_scores is not None
                         else "synthetic_news") or "provided_historical_news",
        "news_dataset": news_manifest,
        "with_news": with_news, "without_news": without_news,
        "return_delta_pct": round(left - right, 4) if left is not None and right is not None else None,
        "trade_count_delta": len(with_news["trades"]) - len(without_news["trades"]),
        "interpretation": "return_delta_pct>0 仅表示该时间窗新闻门控有增益，不代表未来有效",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--market", choices=["cn", "us", "both"], default="both")
    parser.add_argument("--window", type=int, default=90, help="60到120个交易日")
    parser.add_argument("--price-source", choices=["synthetic", "live", "file"], default="synthetic")
    parser.add_argument("--prices-file", type=Path, help="标准 CSV 或导出的价格 JSON；目录中按 cn.csv/us.csv 读取")
    parser.add_argument("--end-date", help="固定行情截至日期 YYYY-MM-DD")
    parser.add_argument("--export-prices", type=Path, help="将单市场输入冻结为可重放 JSON")
    parser.add_argument("--dynamic-candidates", action="store_true", help="用历史新闻关注度选候选，始终保留持仓")
    parser.add_argument("--historical-news", action="store_true",
                        help="从 SQLite 读取真实/导入历史新闻；无事件时拒绝伪装成真实实验")
    parser.add_argument("--news-db", type=Path, default=NEWS_DB)
    parser.add_argument("--record-experiment", action="store_true")
    parser.add_argument("--experiment-db", type=Path, default=EXPERIMENT_DB)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if args.export_prices and args.market == "both":
        parser.error("--export-prices 需要选择单个市场")
    if args.prices_file and args.price_source != "file":
        parser.error("--prices-file 需要 --price-source file")
    markets = ["cn", "us"] if args.market == "both" else [args.market]
    reports = {}
    for market in markets:
        histories, prices_manifest = load_price_dataset(market, args.price_source,
            end_date=args.end_date, path=args.prices_file, window=args.window)
        if args.export_prices:
            export_price_dataset(args.export_prices, market, histories, prices_manifest)
        news_scores = None
        manifest = None
        if args.historical_news:
            common_dates = sorted(set.intersection(*(
                set(frame["date"].astype(str)) for frame in histories.values()
            )))
            evaluation_dates = common_dates[-args.window:]
            news_scores, manifest = load_news_scores(
                market, evaluation_dates[0], evaluation_dates[-1], args.news_db
            )
            if not manifest["event_count"]:
                raise RuntimeError(
                    f"{market} 在 {evaluation_dates[0]}~{evaluation_dates[-1]} 没有真实历史新闻；"
                    "请先积累实时事件或导入供应商历史数据，未降级为合成新闻"
                )
        reports[market] = news_ab_eval(
            histories, market, args.window, news_scores, manifest, args.dynamic_candidates
        )
        reports[market]["price_dataset"] = prices_manifest
        reports[market]["data_quality"] = prices_manifest["data_quality"] + "/" + reports[market]["data_quality"]
        if args.record_experiment:
            report = reports[market]
            dataset = {
                "market": market, "window_days": args.window,
                "prices": {key: value for key, value in prices_manifest.items()
                           if key not in {"source", "data_quality", "limitations"}},
                "news": manifest or {"data_quality": "synthetic_news",
                                      "fingerprint": "synthetic-news-ab-v1"},
            }
            record_experiment(
                "news-on-vs-off", market, dataset,
                {"strategy": report["with_news"]["strategy"], "news_enabled_variants": [True, False]},
                {"return_delta_pct": report["return_delta_pct"],
                 "trade_count_delta": report["trade_count_delta"],
                 "with_news": report["with_news"]["metrics"],
                 "without_news": report["without_news"]["metrics"]},
                db_path=args.experiment_db,
                notes=prices_manifest["data_quality"] + "；" + report["with_news"]["limitations"],
            )
    if args.json:
        print(json.dumps(reports, ensure_ascii=False, indent=2))
        return
    for market, report in reports.items():
        on = report["with_news"]["metrics"]
        off = report["without_news"]["metrics"]
        print(f"\n[{market}] {report['window_days']}日滚动新闻 A/B（{report['data_quality']}）")
        print(f"  新闻开启收益: {on['total_return_pct']:.2f}% | 关闭: {off['total_return_pct']:.2f}% | 差值: {report['return_delta_pct']:.2f}%")
        print(f"  开启后最大回撤: {on['max_drawdown_pct']:.2f}% | 成交数: {len(report['with_news']['trades'])}")


if __name__ == "__main__":
    main()
