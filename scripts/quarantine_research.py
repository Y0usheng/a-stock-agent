"""登记已经人工或宿主原文核验确认的输入错误；不改写原始研究或模拟账本。"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agent.research_quality import QualityBook, REASONS
from config import LOG_DIR


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--log', type=Path, required=True)
    parser.add_argument('--reason', choices=sorted(REASONS), required=True)
    parser.add_argument('--material-page', action='append', default=[], help='该轮材料ID:PDF页序')
    args = parser.parse_args()
    path = args.log.resolve()
    if not path.is_relative_to(Path(LOG_DIR).resolve()) or path.suffix != '.json' or path.stat().st_size > 5_000_000:
        parser.error('仅接受本机决策目录内的有界原始JSON记录')
    proofs = []
    for item in args.material_page:
        identifier, page = item.rsplit(':', 1)
        proofs.append({'material_id': identifier, 'pdf_page': int(page)})
    result = QualityBook().exclude(json.loads(path.read_text(encoding='utf-8')), args.reason, proofs=proofs)
    print(json.dumps({'status': 'quarantined', 'log_hash': result['log_hash'], 'reason': result['reason'], 'recorded_at': result['recorded_at'], 'original_log_unchanged': True}, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
