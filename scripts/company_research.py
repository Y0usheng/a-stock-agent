"""管理长期研究底稿、补证任务与可重放输入；除显式分析入口外不调用模型。"""
import argparse
import asyncio
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agent.analysis import write_log
from agent.errors import safe_error
from agent.investment import ResearchBook, prepare_context, case_variant


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path)
    sub = parser.add_subparsers(dest="command", required=True)
    material = sub.add_parser("import-material")
    material.add_argument("file", type=Path)
    snapshot = sub.add_parser("snapshot")
    snapshot.add_argument("--symbols", nargs="+", required=True)
    snapshot.add_argument("--day", required=True)
    sub.add_parser("tasks")
    prepare = sub.add_parser("prepare")
    prepare.add_argument("frozen_input", type=Path)
    prepare.add_argument("--output", type=Path, required=True)
    save = sub.add_parser("save-log")
    save.add_argument("log", type=Path)
    args = parser.parse_args()
    book = ResearchBook(args.db) if args.db else ResearchBook()
    try:
        if args.command == "import-material":
            raw = json.loads(args.file.read_text(encoding="utf-8"))
            output = [book.add_material(row) for row in raw] if isinstance(raw, list) else book.add_material(raw)
        elif args.command == "tasks":
            output = book.tasks()
        elif args.command == "snapshot":
            output = book.snapshot(args.symbols, args.day)
        elif args.command == "save-log":
            log = json.loads(args.log.read_text(encoding="utf-8"))
            book.save(log, case_variant(log))
            output = {"status": "saved", "source_log": str(args.log.resolve())}
        else:
            if args.output.resolve() == args.frozen_input.resolve():
                raise ValueError("不能覆盖原始冻结输入")
            inputs = json.loads(args.frozen_input.read_text(encoding="utf-8"))
            class FrozenHost:
                async def call(self, name, params):
                    for item in inputs.get("tool_data", []):
                        if item.get("tool") == name and item.get("args", {}).get("symbol") == params.get("symbol"):
                            return item["result"]
                    raise ValueError("该历史价格未在输入文件中，离线准备不会联网补写")
            output = asyncio.run(prepare_context(FrozenHost(), inputs["as_of_date"], inputs["active_symbols"], book=book))
            output["preparation_only"] = True
            output["model_requests"] = 0
            write_log(args.output, output)
            output = {"status": "prepared", "output": str(args.output.resolve()), "policy": output["policy"],
                "material_count": len(output["materials"]), "price_history": [{"symbol": row["symbol"], "observations": row["observations"]} for row in output["price_history"]],
                "tasks": output["tasks"], "model_requests": 0, "preparation_only": True}
        print(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    except Exception as exc:
        print(json.dumps({"status": "failed", "error": safe_error(exc)}, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
