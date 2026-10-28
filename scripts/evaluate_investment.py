"""用本地价格与原始日志评估长期投资，不调用模型或联网。"""
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agent.analysis import write_log
from evals.investment import evaluate_investments
from evals.trading_view import analysis_logs, load_view, PRICE_CACHE

if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    prices = json.loads(PRICE_CACHE.read_text(encoding="utf-8"))
    result = evaluate_investments(analysis_logs(), prices["histories"], load_view())
    output = Path("logs/research/long-term-evaluation.json")
    write_log(output, result)
    print(json.dumps({**result, "output": str(output.resolve())}, ensure_ascii=False, indent=2))
