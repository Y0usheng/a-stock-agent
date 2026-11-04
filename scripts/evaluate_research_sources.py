"""Zero-model acceptance against local, hash-bound official report originals."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATA_DIR
from evals.research_sources import evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, default=DATA_DIR / 'disclosures')
    parser.add_argument('--context-log', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    raw = json.loads(args.context_log.read_text(encoding='utf-8'))
    dataset = raw['input_dataset']
    context = {'date': dataset['as_of_date'], 'symbols': dataset['active_symbols'], 'snapshot': dataset['snapshot'],
               'fundamentals': dataset['fundamentals'], 'account': dataset['account'], 'memory': dataset['memory'],
               'news': dataset['news_screen'], 'investment_context': dataset['investment_context'],
               'timing': dataset['timing'], 'execution_policy': raw['execution_policy']}
    report = evaluate(args.source_root, context)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k:report[k] for k in ('status','facts_passed','facts_evaluated','unavailable_sources','visible_excerpt_chars','model_requests','orders_submitted')},ensure_ascii=False))
    return 0 if report['status'] == 'passed' else 2


if __name__ == '__main__':
    raise SystemExit(main())
