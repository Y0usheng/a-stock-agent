"""Question-led source windows. A matched passage is a candidate, not an answer."""
import re

from evals.experiments import canonical_hash

PROTOCOL = 'question-source-windows-v1'
# Industry operating questions, not stock recommendations or numerical forecasts.
PROFILES = {
    'bank': (
        ('asset_quality', ('不良贷款率', '拨备覆盖率', '关注贷款率')),
        ('margin', ('净利息收益率', '净息差', '净利差')),
        ('capital', ('核心一级资本充足率',)),
        ('impairment', ('信用减值损失', '信用成本')),
        ('business_mix', ('零售金融业务', '零售金融', '财富管理', '存款结构'))),
    'pharma': (
        ('sales_mix', ('创新药销售收入', '创新药收入', '药品销售收入')),
        ('research_spend', ('费用化研发投入', '研发费用', '研发投入')),
        ('licensing', ('对外许可收入', '对外许可首付款', '授权收入')),
        ('earnings_quality', ('扣除非经常性损益的净利润', '非经常性损益'))),
    'consumer': (
        ('sales_channels', ('直销', '经销', '渠道')),
        ('product_mix', ('分产品', '主要产品', '产品结构')),
        ('working_capital', ('合同负债', '存货', '应收账款'))),
    'manufacturing': (
        ('business_mix', ('分产品', '主要业务', '主营业务')),
        ('customers', ('前五名客户', '前五大客户', '客户集中')),
        ('capacity', ('产能利用率', '产能', '产量')),
        ('working_capital', ('存货', '应收账款', '经营活动产生的现金流量净额'))),
    'resources': (
        ('production', ('矿产金', '矿产铜', '产量', '储量')),
        ('cost', ('单位销售成本', '单位成本', '成本')),
        ('investment', ('资本开支', '资本承诺', '在建工程'))),
    'power': (
        ('production', ('发电量', '来水', '利用小时')),
        ('pricing', ('上网电价', '电价')),
        ('investment', ('在建工程', '资本开支', '负债'))),
    'generic': (
        ('business_mix', ('主要业务', '主营业务', '分部信息')),
        ('operating_risk', ('主要风险', '风险因素')),
        ('cash_quality', ('经营活动产生的现金流量净额', '现金流'))),
}
INDUSTRIES = {'600036': 'bank', '600276': 'pharma', '600519': 'consumer',
              '300750': 'manufacturing', '002594': 'manufacturing', '601138': 'manufacturing',
              '002475': 'manufacturing', '300308': 'manufacturing', '002371': 'manufacturing',
              '601899': 'resources', '600900': 'power'}
LABELS = {'asset_quality':'资产质量', 'margin':'息差', 'capital':'资本', 'impairment':'信用减值',
          'business_mix':'业务结构', 'sales_mix':'创新药收入', 'research_spend':'研发投入',
          'licensing':'授权收入', 'earnings_quality':'盈利质量', 'sales_channels':'销售渠道',
          'product_mix':'产品结构', 'working_capital':'营运资金', 'customers':'客户集中',
          'capacity':'产能', 'production':'产量', 'cost':'成本', 'investment':'资本投入',
          'pricing':'价格', 'operating_risk':'经营风险', 'cash_quality':'现金流'}


def _label(question_id):
    return LABELS.get(question_id, '补证' + question_id[-8:])


def source_questions(symbol, tasks=()):
    from agent.research_tasks import FINAL, query_terms
    profile = INDUSTRIES.get(symbol, 'generic')
    rows = [{'question_id': key, 'terms': terms, 'origin': profile} for key, terms in PROFILES[profile]]
    eligible = [t for t in tasks if t.get('status') not in FINAL and t.get('symbol', symbol) == symbol]
    for task in eligible[:4]:
        terms = query_terms(task)
        if terms:
            rows.append({'question_id': task.get('task_id', 'task_' + canonical_hash(task)[:16]),
                         'terms': terms, 'origin': 'manager_question'})
    return rows


def _window(text, at, width=120):
    start = text.rfind('\n', 0, max(0, at - 30)) + 1
    if at - start > 100:
        start = max(0, at - 30)
    end = min(len(text), max(at + 90, start + width))
    # End at a sentence/line boundary; never cut a number or its unit in half.
    stops = [m.end() for m in re.finditer(r'[。；]', text[start:end])]
    stops = [start + n for n in stops if start + n > at + 30]
    if stops:
        end = stops[-1]
    else:
        following = re.search(r'[。；]', text[end:end + 200])
        if following:
            end += following.end()
        else:
            following = re.search(r'\n', text[end:end + 100])
            if following:
                end += following.end()
    # Plain table rows can have no sentence delimiter. Keep a trailing numeric
    # token and common financial units intact even in that fallback case.
    while end < min(len(text), start + width + 200) and re.match(r'[\d.,%％亿元万千百股瓦时吨]', text[end]):
        end += 1
    return start, end


def find_candidates(document, extracted, tasks=(), *, only_tasks=False):
    rows = []
    for question in source_questions(document['symbol'], tasks):
        if only_tasks and question['origin'] != 'manager_question':
            continue
        candidates = []
        for page in extracted['pages']:
            text = page['text']
            if '目录' in text[:180] or '董事长致辞' in text[:240] or '行长致辞' in text[:240]:
                continue
            for rank, term in enumerate(question['terms']):
                for match in re.finditer(re.escape(term), text):
                    start, end = _window(text, match.start())
                    excerpt = text[start:end]
                    score = 30 - min(rank, 20) + sum(t in excerpt for t in question['terms']) * 3
                    score += 12 if re.search(r'\d[\d,.]*\s*(?:%|％|亿元|万元|百万元)', excerpt) else 0
                    score -= 30 if re.search(r'应分别不低于|监管要求|本公司房地产业', excerpt) else 0
                    score -= 15 if re.search(r'适应症|产品图片|参见|详见|见第', excerpt) else 0
                    candidates.append((score, -page['pdf_page'], -start, page, start, end))
        if candidates:
            _, _, _, page, start, end = max(candidates, key=lambda row: row[:3])
            rows.append({**question, 'pdf_page': page['pdf_page'], 'start': start, 'end': end,
                         'page_text_hash': page['text_hash'], 'text': page['text'][start:end]})
    return rows


def question_material(document, extracted, tasks=()):
    """Bounded management-analysis excerpts; financial tables retain their own parser."""
    hits = find_candidates(document, extracted, tasks)
    if not hits:
        return None
    # Newest interim risk facts and annual business explanations are complementary.
    profile = INDUSTRIES.get(document['symbol'], 'generic')
    annual = document['document_type'] == 'annual'
    budget = (300 if annual else 600) if profile == 'bank' else (650 if annual else 250) if profile == 'pharma' else (500 if annual else 350)
    if profile == 'bank' and annual:
        hits.sort(key=lambda h: h['question_id'] != 'business_mix')
    chosen, segments, used = [], [], set()
    for hit in hits:
        key = (hit['pdf_page'], hit['start'], hit['end'])
        if key in used:
            chosen.append(hit)
            continue
        label = f"[PDF{hit['pdf_page']} {_label(hit['question_id'])}]\n"
        text = hit['text'].strip()
        # Keep each source window whole. Omitted candidates remain explicit below.
        if len('\n'.join(segments)) + len(label) + len(text) + 1 > budget:
            continue
        segments.append(label + text)
        chosen.append(hit)
        used.add(key)
    if not chosen:
        return None
    spans = [{k: h[k] for k in ('pdf_page', 'start', 'end', 'page_text_hash')} for h in chosen]
    from agent.financial_inputs import report_period
    return {**document, 'categories': ['business'], 'excerpt': '\n'.join(segments),
            'report_period': report_period(extracted),
            'pdf_page': chosen[0]['pdf_page'],
            'locator': 'PDF页序 ' + ','.join(str(n) for n in sorted({h['pdf_page'] for h in chosen})),
            'source_spans': spans, 'question_protocol': PROTOCOL,
            'question_candidates': [{k: h[k] for k in ('question_id', 'origin', 'pdf_page', 'start', 'end')} for h in hits],
            'question_unmatched': [q['question_id'] for q in source_questions(document['symbol'], tasks)
                                   if q['question_id'] not in {h['question_id'] for h in hits}],
            'question_visible': [h['question_id'] for h in chosen],
            'task_ids': [h['question_id'] for h in chosen if h['origin'] == 'manager_question'],
            'task_matches': [], 'unit_context_present': None, 'data_quality': 'imported',
            'data_source': '受控公告按经营问题定位的原文', 'source_truth_verified': False,
            'extraction_status': 'question_source_candidates_not_semantic_resolution', 'page_excerpt_complete': False}


def compact_question_excerpt(row, limit):
    # The pack is already composed of whole, span-bound source windows. Dropping
    # an entire trailing window is preferable to making a source claim incomplete.
    segments = re.split(r'(?=\[PDF\d+ )', row['excerpt'])
    output = ''
    for segment in segments:
        if len(output) + len(segment) > limit:
            break
        output += segment
    return output.rstrip()


def coverage(materials, presented):
    """Source presence, budget omission and semantic verification are separate states."""
    rows = []
    for material in materials:
        if material.get('question_protocol') != PROTOCOL:
            continue
        visible = next((r for r in presented if r.get('material_id') == material.get('material_id')), None)
        for candidate in material.get('question_candidates', []):
            marker = f"[PDF{candidate['pdf_page']} {_label(candidate['question_id'])}]"
            is_visible = bool(visible and (marker in visible['excerpt'] or
                candidate['question_id'] in material.get('question_visible', []) and visible['excerpt'] == material['excerpt']))
            rows.append({'symbol': material['symbol'], 'question_id': candidate['question_id'],
                         'material_id': material.get('material_id'), 'pdf_page': candidate['pdf_page'],
                         'status': 'visible_candidate_not_verified' if is_visible else 'omitted_by_input_budget',
                         'semantic_verified': False})
        rows.extend({'symbol': material['symbol'], 'question_id': key, 'material_id': material.get('material_id'),
                     'status': 'not_found_in_captured_text', 'semantic_verified': False}
                    for key in material.get('question_unmatched', []))
    return rows


def coverage_summary(rows):
    order = {'visible_candidate_not_verified': 0, 'omitted_by_input_budget': 1,
             'excluded_by_source_checks': 2, 'not_found_in_captured_text': 3}
    selected = {}
    for row in rows:
        key = (row['symbol'], row['question_id'])
        if key not in selected or order[row['status']] < order[selected[key]['status']]:
            selected[key] = row
    return [{'symbol': symbol, 'questions': {key: row['status'] for (s, key), row in selected.items() if s == symbol}}
            for symbol in dict.fromkeys(s for s, _ in selected)]


def eligible_coverage(rows, allowed_material_ids):
    return [dict(row, status='excluded_by_source_checks')
            if row['status'] == 'visible_candidate_not_verified' and row.get('material_id') not in allowed_material_ids
            else dict(row) for row in rows]


def validate_windows(material, extracted):
    pages = {p['pdf_page']: p for p in extracted['pages']}
    spans = material.get('source_spans', [])
    if not spans:
        raise ValueError('missing_source_spans')
    for span in spans:
        p = pages.get(span['pdf_page'])
        if (not p or p['text_hash'] != canonical_hash(p['text']) or p['text_hash'] != span['page_text_hash']
                or not 0 <= span['start'] < span['end'] <= len(p['text'])):
            raise ValueError('source_span_or_hash_mismatch')
    if material.get('question_protocol'):
        parts = re.split(r'\[PDF(\d+) [^\]]+\]\n', material['excerpt'])
        if parts[0] or len(parts) < 3:
            raise ValueError('rendered_excerpt_not_bound_to_source')
        originals = [(s['pdf_page'], pages[s['pdf_page']]['text'][s['start']:s['end']].strip()) for s in spans]
        if any((int(parts[i]), parts[i+1].strip()) not in originals for i in range(1, len(parts), 2)):
            raise ValueError('rendered_excerpt_not_bound_to_source')
