"""Append-only, host-confirmed input faults; original research and ledgers stay intact."""
from datetime import datetime, timezone
import json
from pathlib import Path
import re

from config import DATA_DIR
from evals.experiments import canonical_hash
from servers.data_quality import event_time

HOME = DATA_DIR / 'research_quality'
REASONS = {'financial_column_error', 'source_retracted', 'input_corruption'}


class QualityBook:
    def __init__(self, root=None):
        self.root = Path(root) if root is not None else HOME

    def exclude(self, log, reason, *, proofs=(), recorded_at=None):
        if reason not in REASONS or log.get('market') != 'cn':
            raise ValueError('隔离需要明确的受控原因及A股研究记录')
        stamp = recorded_at or datetime.now(timezone.utc).isoformat()
        if event_time(stamp, 'cn') > datetime.now(timezone.utc):
            raise ValueError('隔离发现时间不能在未来')
        materials = {m['material_id']: m for m in log.get('input_dataset', {}).get('investment_context', {}).get('materials', []) if m.get('material_id')}
        evidence = []
        for proof in proofs:
            if proof.get('material_id') not in materials or not isinstance(proof.get('pdf_page'), int) or proof['pdf_page'] < 1:
                raise ValueError('错误证据须绑定该轮真实材料与PDF页序')
            if proof['pdf_page'] not in {s['pdf_page'] for s in materials[proof['material_id']].get('source_spans', [])}:
                raise ValueError('错误页序未出现在该轮冻结的材料跨度中')
            evidence.append({k: proof[k] for k in ('material_id', 'pdf_page')})
        if reason == 'financial_column_error' and not evidence:
            raise ValueError('财务列错误需要绑定原始材料与页序')
        value = {'version': 'research-quality-v1', 'log_hash': canonical_hash(log),
                 'dataset_fingerprint': log.get('dataset_fingerprint'), 'reason': reason,
                 'recorded_at': stamp, 'proofs': evidence,
                 'symbols': sorted({r['symbol'] for r in log.get('decisions', [])}),
                 'case_hashes': [canonical_hash(c) for c in log.get('investment_cases', [])]}
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / (canonical_hash([value['log_hash'], reason]) + '.json')
        try:
            with path.open('x', encoding='utf-8') as stream:
                json.dump(value, stream, ensure_ascii=False, indent=2)
                stream.write('\n')
        except FileExistsError:
            value = json.loads(path.read_text(encoding='utf-8'))
        return value

    def issues(self, cutoff=None):
        end = event_time(cutoff, 'cn') if cutoff else datetime.now(timezone.utc)
        rows = []
        for path in sorted(self.root.glob('*.json')):
            if path.is_symlink() or not re.fullmatch(r'[0-9a-f]{64}\.json', path.name) or path.stat().st_size > 100_000:
                raise ValueError('研究隔离登记损坏，停止使用历史研究')
            row = json.loads(path.read_text(encoding='utf-8'))
            if row.get('version') != 'research-quality-v1' or row.get('reason') not in REASONS or not re.fullmatch(r'[0-9a-f]{64}', row.get('log_hash', '')):
                raise ValueError('研究隔离登记格式无效')
            if event_time(row['recorded_at'], 'cn') <= end:
                rows.append(row)
        return rows

    def blocked(self, kind='log', cutoff=None):
        rows = self.issues(cutoff)
        if kind == 'log':
            return {r['log_hash'] for r in rows}
        if kind == 'case':
            return {h for r in rows for h in r['case_hashes']}
        if kind == 'material':
            return {p['material_id'] for r in rows for p in r['proofs']}
        raise ValueError('未知隔离对象')

    def notices(self, logs, cutoff=None):
        hashes = {canonical_hash(log) for log in logs}
        return [{'detected_at': r['recorded_at'], 'reason': r['reason'], 'symbols': r['symbols'],
                 'status': 'excluded_from_learning_and_memory'} for r in self.issues(cutoff) if r['log_hash'] in hashes]

    def annotate(self, view, logs, cutoff=None):
        notices = self.notices(logs, cutoff)
        if notices:
            view['research_quality'] = notices
        blocked = self.blocked(cutoff=cutoff)
        for log in logs:
            stamp = log.get('timing', {}).get('decision_at') or log.get('generated_at')
            if not stamp or canonical_hash(log) not in blocked:
                continue
            symbols = {r['symbol'] for r in log.get('decisions', [])}
            for stock in view.get('stocks', []):
                if stock['symbol'] in symbols and stock.get('decision_at') == stamp:
                    stock['tracking_status'] = 'input_quarantined'
