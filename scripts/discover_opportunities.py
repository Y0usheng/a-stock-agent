"""发现A股行业主题，保存原文片段和订阅判断；不触发公司四角色分析或交易。"""
import argparse
import asyncio
from datetime import datetime
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agent.analysis import write_log
from agent.opportunities import OpportunityBook, discover, refresh_feed
from servers.data_quality import market_zone


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--as-of", default=datetime.now(market_zone("cn")).date().isoformat())
    parser.add_argument("--refresh", action="store_true", help="一次采集最新200条东方财富全局快讯，不承诺完整历史")
    parser.add_argument("--import-file", type=Path, help="受控HTTPS来源事件数组，每项title/excerpt/url/published_at")
    parser.add_argument("--rules-only", action="store_true", help="只产生未评价线索，零模型调用，不选出新候选")
    parser.add_argument("--output", type=Path, default=Path("logs/live/opportunities-cn.json"))
    args = parser.parse_args()
    book = OpportunityBook()
    receipt = None
    if args.refresh:
        receipt = refresh_feed(book)
    if args.import_file:
        receipt = book.ingest(json.loads(args.import_file.read_text(encoding="utf-8")), datetime.now(market_zone("cn")).isoformat())
    prior = book.latest()
    from evals.trading_view import read_ledger
    held = [row["symbol"] for row in read_ledger()["positions"]]
    feed = {**book.feed(args.as_of), "refresh_receipt": receipt}
    snapshot = await discover(feed, held=held, previous_symbols=prior["active_symbols"] if prior and prior["as_of_date"] <= args.as_of else [], book=book, use_model=not args.rules_only)
    write_log(args.output, snapshot)
    print(json.dumps({"status": "saved", "as_of_date": args.as_of, "themes": len(snapshot["themes"]),
        "selection_mode": snapshot["selection_mode"], "selected_symbols": snapshot["selected_symbols"],
        "usage": snapshot["usage"], "refresh_receipt": receipt, "review_error": snapshot["review_error"],
        "excluded_events": len(snapshot["excluded_events"]), "output": str(args.output.resolve())}, ensure_ascii=False, indent=2))
    return 2 if snapshot["review_error"] else 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(asyncio.run(main()))
