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
from config import MARKETS
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
                          news_scores: dict[tuple[str, str], float] | None = None) -> dict:
    """运行无未来数据的收盘信号/次日开盘成交回测。"""
    if not 60 <= int(window_days) <= 120:
        raise ValueError("window_days 必须在60到120之间")
    config = MARKETS[market]
    normalized = {}
    for symbol, frame in histories.items():
        required = {"date", "open", "close", "volume"}
        if not required <= set(frame.columns):
            raise ValueError(f"{symbol} 缺少回测字段:{sorted(required - set(frame.columns))}")
        clean = frame.copy()
        clean["date"] = clean["date"].astype(str)
        normalized[symbol] = clean.sort_values("date").drop_duplicates("date", keep="last")
    common_dates = sorted(set.intersection(*(
        set(frame["date"]) for frame in normalized.values()
    ))) if normalized else []
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
    lot = int(config["lot_size"])

    for index, day in enumerate(evaluation_dates):
        # 只执行上一交易日已经生成的订单，成交价取当日开盘。
        for order in pending:
            symbol, action = order["symbol"], order["action"]
            row = rows[symbol].loc[day]
            market_price = float(row["open"])
            daily_volume = int(row["volume"])
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
                affordable = int(cash / max(price * lot, 1)) * lot
                shares = min(shares, affordable)
                if shares < lot:
                    continue
                amount = shares * price
                fee = _fee(config, amount, False)
                if amount + fee > cash:
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
                "slippage_cost": round(abs(price - market_price) * shares, 2),
            })
        pending = []

        closes = {symbol: float(rows[symbol].loc[day]["close"]) for symbol in normalized}
        total = cash + sum(shares * closes[symbol] for symbol, shares in positions.items())
        curve.append({"date": day, "total": total})
        if index == len(evaluation_dates) - 1:
            continue

        for symbol, frame in normalized.items():
            signal = _technical_signal(frame, day)
            if signal is None:
                continue
            if not use_news:
                score = 0.0
            elif news_scores is None:
                score = _offline_news_score(market, symbol, day)
            else:
                score = float(news_scores.get((day, symbol), 0.0))
            held = positions.get(symbol, 0)
            action = "HOLD"
            shares = 0
            if held and (not signal["above_ma20"] or (use_news and score <= -1.5)):
                action, shares = "SELL", held
            elif not held and signal["bullish"] and (not use_news or score >= 0):
                budget = total * 0.28
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
        "market": market, "window_days": window_days, "use_news": use_news,
        "execution": "signal close / next trading day open",
        "metrics": metrics, "curve": indexed_curve(curve, config["initial_cash"]),
        "trades": trades, "decisions": decisions,
        "total_fees": round(sum(item["fee"] for item in trades), 2),
        "total_slippage_cost": round(sum(item["slippage_cost"] for item in trades), 2),
    }


def news_ab_eval(histories: dict[str, pd.DataFrame], market: str,
                 window_days: int = 90,
                 news_scores: dict[tuple[str, str], float] | None = None) -> dict:
    with_news = walk_forward_backtest(histories, market, window_days, True, news_scores)
    without_news = walk_forward_backtest(histories, market, window_days, False, news_scores)
    left = with_news["metrics"].get("total_return_pct")
    right = without_news["metrics"].get("total_return_pct")
    return {
        "market": market, "window_days": window_days,
        "data_quality": "synthetic_news" if news_scores is None else "provided_historical_news",
        "with_news": with_news, "without_news": without_news,
        "return_delta_pct": round(left - right, 4) if left is not None and right is not None else None,
        "trade_count_delta": len(with_news["trades"]) - len(without_news["trades"]),
        "interpretation": "return_delta_pct>0 仅表示该时间窗新闻门控有增益，不代表未来有效",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--market", choices=["cn", "us", "both"], default="both")
    parser.add_argument("--window", type=int, default=90, help="60到120个交易日")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    markets = ["cn", "us"] if args.market == "both" else [args.market]
    reports = {}
    for market in markets:
        module_name = "servers.market_server" if market == "cn" else "servers.us_market_server"
        module = __import__(module_name, fromlist=["_mock_history"])
        histories = {
            item["symbol"]: module._mock_history(item["symbol"], args.window + 80)
            for item in MARKETS[market]["watchlist"]
        }
        reports[market] = news_ab_eval(histories, market, args.window)
    if args.json:
        print(json.dumps(reports, ensure_ascii=False, indent=2))
        return
    for market, report in reports.items():
        on = report["with_news"]["metrics"]
        off = report["without_news"]["metrics"]
        print(f"\n[{market}] {report['window_days']}日滚动新闻 A/B（离线合成新闻）")
        print(f"  新闻开启收益: {on['total_return_pct']:.2f}% | 关闭: {off['total_return_pct']:.2f}% | 差值: {report['return_delta_pct']:.2f}%")
        print(f"  开启后最大回撤: {on['max_drawdown_pct']:.2f}% | 成交数: {len(report['with_news']['trades'])}")


if __name__ == "__main__":
    main()
