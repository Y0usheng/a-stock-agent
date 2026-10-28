"""A股研究任务：plan → freeze → predict/attach → run → review → replay。"""
import argparse
import asyncio
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agent.analysis import write_log
from agent.errors import safe_error
from config import EXPERIMENT_DB
from evals.experiments import canonical_hash, decision_log_variant
from evals.research import PHASES, ResearchBook, bundle_from_files, render_review


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def demo(book, output_dir, prices_file=None):
    from servers.synthetic_data import random_walk_history
    symbols = ["600036", "600519"]
    if prices_file:
        bundle = bundle_from_files(prices_file, symbols=symbols)
        baseline, candidate, mode = "buy_hold", "technical", "retrospective"
    else:
        histories = {symbol: random_walk_history(symbol, 120, {}, .0003, .015,
            1000000, 3000000, .008, end_date="2026-09-30").to_dict("records") for symbol in symbols}
        bundle = {"market": "cn", "histories": histories, "price_quality": "synthetic_prices", "price_basis": "synthetic"}
        baseline, candidate, mode = "technical", "news_gate", "engineering"
    days = sorted(set.intersection(*(set(row["date"] for row in rows) for rows in bundle["histories"].values())))
    if len(days) < 100:
        raise ValueError("演示需要至少100个共同交易日")
    if not prices_file:
        bundle["news_events"] = [{"symbol": symbol, "title": "合成研究事件", "summary": "仅用于闭环工程验收",
            "source": "synthetic-research-fixture", "url": "https://example.org/synthetic",
            "published_at": day + "T10:00:00+08:00", "sentiment_score": 1 if index % 2 else -2,
            "data_quality": "synthetic"} for index, day in enumerate(days[26::3]) for symbol in symbols]
    spec = {"name": "A股冻结研究闭环验收" + ("-历史价格" if prices_file else "-合成工程"),
        "hypothesis": "固定股票池和费用后，候选策略能否改善基线的5日方向判断？", "market": "cn", "mode": mode,
        "symbols": symbols, "horizon_days": 5,
        "splits": {"train": [days[30], days[49]], "validation": [days[50], days[69]], "test": [days[70], days[-1]]},
        "baseline": baseline, "candidate": candidate,
        "variants": [{"id": "buy_hold", "kind": "buy_hold"}, {"id": "technical", "kind": "technical"}, {"id": "news_gate", "kind": "news_gate"}],
        "criteria": {"min_matured_predictions": 10, "min_nonoverlapping_dates": 2, "min_gain": 0}}
    study = book.create(spec)
    frozen = book.freeze(study["id"], bundle)
    for phase in PHASES:
        book.run_phase(study["id"], phase, days[-1])
    report = book.review(study["id"])
    first = next(row for row in book.attempts(study["id"]) if row["status"] == "complete")
    replay = book.replay(study["id"], first["attempt_id"])
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = "research-historical-cn" if prices_file else "research-engineering-cn"
    write_log(output_dir / (stem + ".json"), report)
    (output_dir / (stem + ".md")).write_text(render_review(report), encoding="utf-8")
    write_log(output_dir / (stem + ".plan.json"), spec)
    receipt = {"study_id": study["id"], "verdict": report["verdict"], "attempt_count": report["attempt_count"],
            "replay": replay, "data_fingerprint": frozen["fingerprint"], "report": str((output_dir / (stem + ".md")).resolve())}
    write_log(output_dir / (stem + ".receipt.json"), receipt)
    return receipt


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=EXPERIMENT_DB)
    sub = parser.add_subparsers(dest="command", required=True)
    plan = sub.add_parser("plan"); plan.add_argument("spec", type=Path)
    freeze = sub.add_parser("freeze"); freeze.add_argument("study"); freeze.add_argument("--prices", type=Path, required=True); freeze.add_argument("--news", type=Path)
    attach = sub.add_parser("attach"); attach.add_argument("study"); attach.add_argument("variant"); attach.add_argument("log", type=Path); attach.add_argument("--semantic", type=Path)
    run = sub.add_parser("run"); run.add_argument("study"); run.add_argument("--phase", choices=[*PHASES, "all"], default="all"); run.add_argument("--as-of")
    review = sub.add_parser("review"); review.add_argument("study"); review.add_argument("--output", type=Path)
    replay = sub.add_parser("replay"); replay.add_argument("study"); replay.add_argument("attempt")
    show = sub.add_parser("show"); show.add_argument("study")
    sub.add_parser("list")
    variant = sub.add_parser("variant"); variant.add_argument("--log", type=Path); variant.add_argument("--prompt-version", choices=["manager-v1", "manager-v2"], default="manager-v1"); variant.add_argument("--synthetic", action="store_true")
    predict = sub.add_parser("predict"); predict.add_argument("study"); predict.add_argument("variant"); predict.add_argument("--day", required=True); predict.add_argument("--output", type=Path); predict.add_argument("--allow-retrospective", action="store_true")
    example = sub.add_parser("demo"); example.add_argument("--output-dir", type=Path, default=Path("logs/demo")); example.add_argument("--prices", type=Path)
    args = parser.parse_args()
    book = ResearchBook(args.db)
    try:
        if args.command == "plan":
            output = book.create(read_json(args.spec))
        elif args.command == "freeze":
            plan = book.get(args.study)["plan"]
            data = book.freeze(args.study, bundle_from_files(args.prices, args.news, plan["symbols"]))
            output = {"data_id": data["id"], "fingerprint": data["fingerprint"]}
        elif args.command == "attach":
            output = book.attach_log(args.study, args.variant, read_json(args.log), read_json(args.semantic) if args.semantic else None)
        elif args.command == "run":
            output = []
            for phase in PHASES if args.phase == "all" else [args.phase]:
                rows = book.run_phase(args.study, phase, args.as_of)
                output += [{key: row.get(key) for key in ("attempt_id", "variant_id", "status", "reason", "metrics")} | {"phase": phase} for row in rows]
        elif args.command == "review":
            output = book.review(args.study)
            path = args.output or Path("logs/research") / (args.study + ".review.json")
            write_log(path, output)
            path.with_suffix(".md").write_text(render_review(output), encoding="utf-8")
            output = {"verdict": output["verdict"], "reason": output["reason"], "next_hypothesis": output["next_hypothesis"], "report": str(path.resolve())}
        elif args.command == "replay":
            output = book.replay(args.study, args.attempt)
        elif args.command == "show":
            output = book.get(args.study)
        elif args.command == "list":
            output = book.summaries()
        elif args.command == "variant":
            if args.log:
                identity = decision_log_variant(read_json(args.log))
            else:
                from evals.research_agent import variant_identity
                identity = variant_identity(args.prompt_version, args.synthetic)
            output = {"identity": identity, "variant_fingerprint": canonical_hash(identity)}
        elif args.command == "predict":
            from evals.research_agent import predict_frozen
            path = args.output or Path("logs/research") / (args.study + "." + args.variant + "." + args.day + ".json")
            output = asyncio.run(predict_frozen(book, args.study, args.variant, args.day, output=path,
                                              allow_retrospective=args.allow_retrospective))
        else:
            output = demo(book, args.output_dir, args.prices)
        print(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    except Exception as exc:
        print(json.dumps({"status": "failed", "error": safe_error(exc)}, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
