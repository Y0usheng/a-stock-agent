"""刷新A股真实历史价格缓存，并导出无需外部脚本的交易表现HTML。"""
import argparse
from datetime import datetime, timedelta
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agent.analysis import write_log
from agent.errors import safe_error
from config import MARKETS
from evals.experiments import canonical_hash
from evals.price_data import normalize_history
from evals.trading_view import PRICE_CACHE, analysis_logs, load_view, read_ledger, render_html, tracking_symbols
from servers.data_quality import completed_history, market_zone


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", nargs="+")
    parser.add_argument("--days", type=int, choices=[60, 120, 180, 240], default=180)
    parser.add_argument("--direct", action="store_true", help="仅本进程直连公开行情源，不使用环境代理")
    parser.add_argument("--provider", choices=["eastmoney", "sina"], default="eastmoney")
    parser.add_argument("--cache", type=Path, default=PRICE_CACHE)
    parser.add_argument("--output", type=Path, default=Path("logs/live/trading-performance-cn.html"))
    args = parser.parse_args()
    cache_updated = False
    try:
        import akshare as ak
        from unittest.mock import patch
        import requests
        logs, ledger = analysis_logs(), read_ledger()
        latest = logs[-1] if logs else {}
        from agent.opportunities import OpportunityBook
        opportunity = OpportunityBook().view()
        symbols = args.symbols or tracking_symbols(logs, ledger, opportunity)
        symbols = sorted(set(symbols) | set(opportunity.get("active_symbols", [])))
        if not args.symbols:
            symbols = sorted(set(symbols) | {row["symbol"] for row in opportunity.get("forward_observations", [])})
        symbols = sorted(set(symbols) | {row["symbol"] for row in ledger["positions"]} | {row["symbol"] for row in ledger["trades"]})
        allowed = {row["symbol"] for row in MARKETS["cn"]["candidate_universe"]}
        if not symbols or not set(symbols) <= allowed:
            raise ValueError("需要受控A股代码，可先保存真实分析日志或指定--symbols")
        if args.output.resolve() == args.cache.resolve():
            raise ValueError("HTML输出不能覆盖价格缓存")
        local = datetime.now(market_zone("cn"))
        start = (local - timedelta(days=args.days * 2)).strftime("%Y%m%d")
        histories = {}
        original = requests.sessions.Session.__init__
        def init_session(instance, *params, **kwargs):
            original(instance, *params, **kwargs)
            if args.direct:
                instance.trust_env = False
        request = requests.sessions.Session.request
        def bounded_request(instance, *params, **kwargs):
            if kwargs.get("timeout") is None:
                kwargs["timeout"] = 25
            return request(instance, *params, **kwargs)
        with patch.object(requests.sessions.Session, "__init__", init_session), patch.object(requests.sessions.Session, "request", bounded_request):
            for symbol in symbols:
                if args.provider == "eastmoney":
                    frame = ak.stock_zh_a_hist(symbol=symbol, period="daily", start_date=start,
                        end_date=local.strftime("%Y%m%d"), adjust="qfq", timeout=25)
                    frame = frame.rename(columns={"日期": "date", "开盘": "open", "收盘": "close", "最高": "high", "最低": "low", "成交量": "volume"})
                    frame["volume"] = frame["volume"] * 100
                else:
                    prefix = "sh" if symbol.startswith("6") else "sz"
                    frame = ak.stock_zh_a_daily(symbol=prefix + symbol, start_date=start,
                        end_date=local.strftime("%Y%m%d"), adjust="qfq")
                    # 新浪日线成交量已是股，不再乘100。
                frame["date"] = frame["date"].astype(str)
                clean = normalize_history(symbol, completed_history(frame, "cn")).tail(args.days)
                if clean.empty:
                    raise ValueError(symbol + "无已收盘的真实价格")
                histories[symbol] = clean.to_dict("records")
        payload = {"market": "cn", "histories": histories, "price_basis": "qfq", "volume_unit": "shares",
            "source": ("AKShare / 东方财富 stock_zh_a_hist" if args.provider == "eastmoney" else "AKShare / 新浪财经 stock_zh_a_daily"), "data_quality": "live_prices",
            "fetched_at": local.isoformat(), "network_route": "direct" if args.direct else "configured_proxy",
            "fingerprint": canonical_hash(histories)}
        # 全部股票成功后才原子更新；失败保留上一份缓存。
        write_log(args.cache, payload)
        cache_updated = True
        view = load_view(args.cache)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(render_html(view), encoding="utf-8")
        write_log(args.output.with_suffix(".json"), view)
        print(json.dumps({"status": "ready", "as_of_date": view["as_of_date"], "symbols": symbols,
            "rows": {key: len(rows) for key, rows in histories.items()}, "portfolio": view["portfolio"],
            "output": str(args.output.resolve()), "warnings": view["warnings"]}, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        print(json.dumps({"status": "failed", "error": safe_error(exc), "prior_cache_preserved": not cache_updated}, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
