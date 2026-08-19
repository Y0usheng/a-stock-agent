# -*- coding: utf-8 -*-
"""收益风险 Evals 命令行报告。用法: python evals/report.py"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import LOG_DIR, MARKETS
from evals.performance import performance_report

sys.path.insert(0, str(Path(__file__).parent.parent / "servers"))
import market_server
import us_market_server

DATA = {"cn": market_server, "us": us_market_server}


def _logs(market: str) -> list[dict]:
    out = []
    for path in sorted(LOG_DIR.glob(f"*.{market}.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if data.get("market") == market:
                out.append(data)
        except (OSError, json.JSONDecodeError):
            continue
    return out


def _fmt(value, suffix=""):
    return "样本不足" if value is None else f"{value:.2f}{suffix}"


def _dynamic_universe(logs: list[dict], config: dict) -> list[dict]:
    symbols = {symbol for log in logs for symbol in log.get("active_symbols", [])}
    source = config.get("candidate_universe", config["watchlist"])
    return [item for item in source if not symbols or item["symbol"] in symbols]


def main():
    for market, config in MARKETS.items():
        logs = _logs(market)
        report = performance_report(logs, DATA[market], _dynamic_universe(logs, config))
        p, b = report["portfolio"], report["benchmark"]
        print(f"\n[{market}] {config['name']} 收益风险 Evals")
        print(f"  样本: {p['observations']} 个资产截面 | {report['data_quality']['message']}")
        print(f"  累计收益: {_fmt(p['total_return_pct'], '%')} | 等权基准: {_fmt(b['total_return_pct'], '%')} | 超额: {_fmt(report['excess_return_pct'], '%')}")
        print(f"  年化波动: {_fmt(p['annualized_volatility_pct'], '%')} | 最大回撤: {_fmt(p['max_drawdown_pct'], '%')}")
        print(f"  Sharpe: {_fmt(p['sharpe_ratio'])} | Sortino: {_fmt(p['sortino_ratio'])}")
        print(f"  交易次数: {report['trade_count']} | 累计换手: {_fmt(report['turnover_pct'], '%')}")


if __name__ == "__main__":
    main()
