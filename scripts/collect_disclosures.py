"""从巨潮官方获取年报/季报原件，保存PDF哈希与页码片段；不调用模型。"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agent.disclosures import collect, extract_pages
from agent.analysis import write_log


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", nargs="+")
    parser.add_argument("--budget-seconds", type=int, default=90)
    parser.add_argument("--extract-worker", nargs=2, metavar=("PDF", "JSON"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.extract_worker:
        source, target = map(Path, args.extract_worker)
        write_log(target, extract_pages(source))
        return 0
    if not args.symbols or not 5 <= args.budget_seconds <= 180:
        parser.error("需提供1-3家股票代码，预算5-180秒")
    result = collect(args.symbols, budget_seconds=args.budget_seconds)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["documents"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
