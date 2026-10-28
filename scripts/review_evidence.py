"""A股离线证据评审；默认只运行规则，--model-review 才消耗订阅额度。"""
import argparse
import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agent.analysis import write_log
from agent.errors import safe_error
from config import DATA_DIR, DEFAULT_OPENAI_MODEL
from evals.semantic_review import ReviewCache, ReviewOptions, prepare_review, review_with_model, summarize


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("--model-review", action="store_true")
    parser.add_argument("--model", default=DEFAULT_OPENAI_MODEL)
    parser.add_argument("--sample-rate", type=float, default=.1)
    parser.add_argument("--seed", default="cn-review-v1")
    parser.add_argument("--max-claims", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-requests", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=75)
    parser.add_argument("--token-limit", type=int, default=12000)
    parser.add_argument("--input-char-limit", type=int, default=24000)
    parser.add_argument("--reasoning-effort", choices=["low", "medium", "high", "xhigh", "max"], default="low")
    parser.add_argument("--cache", type=Path, default=DATA_DIR / "semantic_reviews.db")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    started = perf_counter()
    try:
        result = json.loads(args.log.read_text(encoding="utf-8"))
        options = ReviewOptions(**{key: getattr(args, key) for key in ReviewOptions.__dataclass_fields__})
        if args.model_review:
            report = asyncio.run(review_with_model(result, model=args.model, options=options,
                cache=None if args.no_cache else ReviewCache(args.cache)))
        else:
            report = prepare_review(result, options)
        report["source_log"] = str(args.log.resolve())
        report["generated_at"] = datetime.now(timezone.utc).isoformat()
        report["duration_ms"] = round((perf_counter() - started) * 1000)
        report["summary"] = summarize(report)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        output = args.output or Path("logs/reviews") / (args.log.stem + "." + report["mode"] + "." + stamp + ".json")
        if output.resolve() == args.log.resolve():
            raise ValueError("评审输出不能覆盖原决策日志")
        write_log(output, report)
        print(json.dumps({**report["summary"], "duration_ms": report["duration_ms"],
                          "output": str(output.resolve())}, ensure_ascii=False, indent=2))
        return 0 if report["status"] in {"rules_only", "completed"} else 2
    except Exception as exc:
        print(json.dumps({"status": "failed", "error": safe_error(exc)}, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
