"""Frozen source-to-model acceptance, independent of trades and model profitability."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re

from evals.experiments import canonical_hash
from agent.research_retrieval import validate_windows

CASES = Path(__file__).parent / 'fixtures' / 'research_source_cases.json'


def normalized(text):
    return re.sub(r'\s+', '', text).replace('％', '%')


def present_sources(context, documents):
    """Use the production selection AND production export, never a parallel renderer."""
    from agent.disclosures import select_passages, VERSION
    from agent.financial_inputs import make_material
    from agent.investment import select_materials
    from agent.subscription_roles import export_paper_context
    materials = []
    for document, extracted in documents:
        rows = select_passages(document, extracted, context.get('investment_context', {}).get('tasks', []))
        financial = make_material(document, extracted)
        if financial:
            rows.append(financial)
        for row in rows:
            validate_windows(row, extracted)
            item = {**row, 'capture_protocol': VERSION, 'material_id': 'qa_' + canonical_hash(row)[:24],
                    'data_source': row.get('data_source') or '受控公司披露片段(导入)',
                    'first_seen_at': document['published_at'], 'expires_at': '2099-12-31'}
            materials.append(item)
    selected = [m for symbol in context['symbols'] for m in select_materials([r for r in materials if r['symbol'] == symbol], 8)]
    ctx = deepcopy(context)
    ctx.setdefault('investment_context', {})['materials'] = selected
    projected = export_paper_context(ctx, 'cn', True)
    from agent.evidence import build_catalog
    from agent.research_retrieval import eligible_coverage
    catalog = build_catalog(projected, 'cn')
    allowed = {r['payload'].get('material_id') for r in catalog['records'] if r['kind']=='company_document' and r['eligible']}
    projected['research_materials'] = [r for r in projected['research_materials'] if r['material_id'] in allowed]
    projected['retrieval_coverage'] = eligible_coverage(projected.get('retrieval_coverage', []), allowed)
    return projected


def fact_results(case, extracted, presented):
    results = []
    for fact in case['facts']:
        page = next((p for p in extracted['pages'] if p['pdf_page'] == fact['pdf_page']), None)
        expected = normalized(fact['text'])
        present = bool(page and expected in normalized(page['text']))
        matches = [r for r in presented if r.get('symbol') == case['symbol']
                   and r.get('pdf_sha256') == case['pdf_sha256']
                   and expected in normalized(r.get('excerpt', ''))]
        # Export removes spans from the prompt. The immutable material identity
        # binds it back to the source; a page locator remains visible to the model.
        results.append({'fact_id': fact['id'], 'pdf_page': fact['pdf_page'], 'source_present': present,
                        'model_visible': bool(matches), 'status': 'pass' if present and matches else 'fail'})
    return results


def evaluate(root, context, *, manifest=CASES):
    root = Path(root)
    cases = json.loads(Path(manifest).read_text(encoding='utf-8'))['cases']
    documents, unavailable, case_sources = [], [], []
    for case in cases:
        pdf = root / (case['pdf_sha256'] + '.pdf')
        text = root / (case['pdf_sha256'] + '.pymupdf-sorted-v1.pages.json')
        if not pdf.exists() or not text.exists():
            unavailable.append({'case_id': case['id'], 'reason': 'source_files_missing'})
            continue
        if hashlib.sha256(pdf.read_bytes()).hexdigest() != case['pdf_sha256']:
            unavailable.append({'case_id': case['id'], 'reason': 'pdf_hash_mismatch'})
            continue
        receipt = root / (case['announcement_id'] + '.capture.json')
        captured = json.loads(receipt.read_text(encoding='utf-8')) if receipt.exists() else {}
        text_hash = captured.get('extractions', {}).get('pymupdf-sorted-v1', {}).get('sha256')
        if captured.get('pdf_sha256') != case['pdf_sha256'] or hashlib.sha256(text.read_bytes()).hexdigest() != text_hash:
            unavailable.append({'case_id': case['id'], 'reason': 'extraction_receipt_mismatch'})
            continue
        extracted = json.loads(text.read_text(encoding='utf-8'))
        if any(p['text_hash'] != canonical_hash(p['text']) for p in extracted['pages']):
            unavailable.append({'case_id': case['id'], 'reason': 'extracted_page_hash_mismatch'})
            continue
        documents.append((case, extracted))
        case_sources.append((case, extracted))
    projected = present_sources(context, documents) if documents else {'research_materials': []}
    checks = [dict(case_id=case['id'], **row) for case, extracted in case_sources
              for row in fact_results(case, extracted, projected['research_materials'])]
    passed = sum(r['status'] == 'pass' for r in checks)
    prechecks = assertion_prechecks()
    return {'protocol': 'research-source-acceptance-v1', 'mode': 'engineering_replay_not_trading_or_backtest',
            'status': 'passed' if checks and passed == len(checks) and not unavailable and all(r['status']=='pass' for r in prechecks) else 'failed',
            'facts_passed': passed, 'facts_evaluated': len(checks), 'unavailable_sources': unavailable,
            'sampling': 'regression_cases_selected_from_observed_omissions_not_marketwide_recall',
            'checks': checks, 'model_requests': 0, 'orders_submitted': 0,
            'source_coverage': projected.get('retrieval_coverage', []),
            'assertion_prechecks': prechecks,
            'visible_excerpt_chars': sum(len(r['excerpt']) for r in projected['research_materials']),
            'boundary': 'Checks frozen facts, source spans and actual input delivery; not general semantic entailment or investment performance.'}


def assertion_prechecks():
    """Labelled semantic traps exercise existing prechecks; none certify entailment."""
    from evals.semantic_review import rule_check
    cases = [
        ('amount_unit', '合同总金额36.4亿元', '合同总金额36.4万元', 'company_disclosure', 'disclosure_reference', 'numeric_value_or_unit_unmatched'),
        ('period', '2026年第三季度收入10亿元', '2025年第三季度收入10亿元', 'company_disclosure', 'disclosure_reference', 'reporting_period_unmatched'),
        ('comparison', '营业收入同比增长10%', '营业收入环比增长10%', 'company_disclosure', 'disclosure_reference', 'comparison_basis_risk'),
        ('modality', '拟签署合作协议', '公司已签署合作协议', 'company_disclosure', 'disclosure_reference', 'modality_upgrade_risk'),
        ('media_origin', '媒体报道订单增长', '公司确认订单增长', 'company_disclosure', 'media', 'primary_disclosure_missing'),
        ('surface_is_not_entailment', '净利润10亿元', '净利润10亿元', 'company_disclosure', 'disclosure_reference', 'numeric_surface_match_only')]
    rows = []
    for key, source, claim, kind, tier, expected in cases:
        issues = rule_check({'text': claim, 'kind_hint': kind, 'role': 'fundamental-analyst'},
                            [{'source_tier': tier, 'passages': [{'text': source}]}])
        rows.append({'case_id': key, 'status': 'pass' if expected in {r['code'] for r in issues} else 'fail',
                     'expected_finding': expected, 'findings': [r['code'] for r in issues],
                     'semantic_verified': False})
    return rows
