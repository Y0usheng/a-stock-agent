"""Conservative table-row extraction and arithmetic, never a valuation forecast."""
from decimal import Decimal, InvalidOperation
import re

from evals.experiments import canonical_hash

VERSION = 'source-financial-rows-v1'
LABELS = {
    'eps': ('基本每股收益',), 'bvps': ('每股净资产',),
    'roe': ('加权平均净资产收益率', '平均净资产收益率'),
    'revenue': ('营业收入',), 'operating_cashflow': ('经营活动产生的现金流量净额',),
    'capex_cash': ('购建固定资产、无形资产和其他长期资产支付的现金',),
    'cash': ('货币资金',), 'short_borrowings': ('短期借款',), 'long_borrowings': ('长期借款',),
    'bonds': ('应付债券',), 'npl_ratio': ('不良贷款率',), 'capital_ratio': ('资本充足率',),
    'net_interest_margin': ('净利息收益率', '净息差'),
    'provision_coverage': ('拨备覆盖率', '不良贷款拨备覆盖率'),
    'special_mention_ratio': ('关注类贷款占比', '关注贷款率'),
    'credit_cost': ('信用成本率', '信用成本'),
}
PRIORITY = ('eps', 'bvps', 'roe', 'npl_ratio', 'net_interest_margin', 'provision_coverage',
            'special_mention_ratio', 'credit_cost', 'capital_ratio', 'operating_cashflow',
            'capex_cash', 'cash', 'short_borrowings', 'long_borrowings', 'bonds', 'revenue')
MONEY = re.compile(r'(?:单位\s*[:：]\s*(?:人民币)?|[（(]人民币)(百万元|千元|万元|亿元|元)')


def report_period(extracted):
    text = re.sub(r'\s+', '', ''.join(p['text'] for p in extracted['pages'][:3]))
    match = re.search(r'(20\d{2})年?(半年度|年度|第[一二三四1-4]季度)报告', text)
    return (match[1] + match[2]) if match else None


def metric_name(label):
    label = re.sub(r'\s+|[（(]\d+[)）]', '', label)
    # Diluted, adjusted and sector-specific ratios cannot replace their headline measures.
    if any(word in label for word in ('稀释', '扣除', '调整', '核心一级', '一级资本', '房地产业')):
        return None
    label = re.sub(r'[（(].*?[)）]', '', label)
    for metric, names in LABELS.items():
        if any(label == name or (metric in {'eps', 'bvps', 'roe'} and label.endswith(name)) for name in names):
            return metric
    return None


def extract_facts(document, extracted):
    period = report_period(extracted)
    if not period:
        return []
    year = period[:4]
    found = {}
    statement_scope = None
    anchors = []
    for page in extracted['pages']:
        text = page['text']
        offset, history = 0, []
        lines = text.splitlines(keepends=True)
        for index, line in enumerate(lines):
            heading = re.sub(r'\s+', '', line)
            if re.fullmatch(r'合并(?:资产负债表|现金流量表|利润表|综合收益表)(?:[（(]续[)）])?', heading):
                statement_scope, anchors, history = 'consolidated', [], []
            elif re.fullmatch(r'(?:母公司|本行)(?:资产负债表|现金流量表|利润表|综合收益表)(?:[（(]续[)）])?', heading):
                statement_scope, anchors, history = 'parent', [], []
            if MONEY.search(line) or len(set(re.findall(r'20\d{2}', line))) >= 2:
                anchors.append({'text':line, 'pdf_page':page['pdf_page'], 'start':offset,
                                'end':offset+len(line), 'page_text_hash':page['text_hash']})
                anchors = anchors[-6:]
            row = re.match(r'^\s*(.*?)\s{2,}([-−(\d].*?)\s*$', line)
            label = row[1] if row else ''
            cells = row[2] if row else ''
            if row and re.fullmatch(r'[\d,().−-]+', label.strip()):
                label, cells = '', line.strip()
            source_start, source_end = offset, offset + len(line)
            metric = metric_name(label) if row else None
            if row and not metric and index and len(lines[index-1].strip()) < 65:
                combined = lines[index-1].strip() + label
                if metric_name(combined):
                    label, metric = combined, metric_name(combined)
                    source_start -= len(lines[index-1])
            if row and not metric and index+1 < len(lines) and len(lines[index+1].strip()) < 40:
                combined = label + lines[index+1].strip()
                if metric_name(combined):
                    label, metric = combined, metric_name(combined)
                    source_end += len(lines[index+1])
            if row and not metric and not label.strip() and index and index+1 < len(lines):
                combined = lines[index-1].strip() + lines[index+1].strip()
                if metric_name(combined):
                    label, metric = combined, metric_name(combined)
                    source_start -= len(lines[index-1])
                    source_end += len(lines[index+1])
            if metric and metric not in found:
                usable_anchors = [a for a in anchors if a['pdf_page']==page['pdf_page'] or
                                  statement_scope=='consolidated' and 0 < page['pdf_page']-a['pdf_page'] <= 3]
                context = ''.join(a['text'] for a in usable_anchors) + ''.join(history[-35:])
                headers = [a['text'].strip() for a in usable_anchors if len(set(re.findall(r'20\d{2}', a['text']))) >= 2]
                # No guessing which column belongs to which year, or switching columns.
                years = re.findall(r'20\d{2}', headers[-1]) if headers else []
                cell = re.match(r'\s*([−-]?\d[\d,]*(?:\.\d+)?)(%)?(?:\s|$)', cells)
                summary = page['pdf_page'] <= 40 and ('会计数据' in text or '财务指标' in text)
                scope_ok = summary or statement_scope == 'consolidated'
                if cell and years and years[0] == year and scope_ok:
                    if metric in {'eps', 'bvps'}:
                        unit = '元/股' if re.search(r'人民币元|元[／/]股|每股.*?[（(]元[)）]', context + label) else None
                    elif metric in {'roe', 'npl_ratio', 'capital_ratio', 'net_interest_margin',
                                    'provision_coverage', 'special_mention_ratio', 'credit_cost'}:
                        unit = '%' if cell[2] or '%' in context + row[1] or '％' in context + row[1] else None
                    else:
                        units = list(MONEY.finditer(context))
                        unit = units[-1][1] if units else None
                    if unit:
                        value = cell[1].replace(',', '').replace('−', '-')
                        fact = {'metric': metric, 'label': label.strip(), 'value': value, 'unit': unit,
                                'period': period, 'scope': 'group_summary' if summary else 'consolidated',
                                'column_header': headers[-1], 'pdf_page': page['pdf_page'],
                                'source_context': usable_anchors,
                                'source_line': text[source_start:source_end], 'start': source_start, 'end': source_end,
                                'page_text_hash': page['text_hash']}
                        fact['fact_id'] = 'fact_' + canonical_hash([document['symbol'], document.get('announcement_id'), fact])[:20]
                        found[metric] = fact
            history.append(line)
            offset += len(line)
    return [found[key] for key in PRIORITY if key in found]


def render_facts(facts, limit=700):
    if not facts:
        return ''
    text = '正文报告期间：' + facts[0]['period'] + '；以下逐行注明单位与PDF页序，均为首列当期数。'
    for fact in facts:
        line = '\n' + fact_line(fact)
        if len(text) + len(line) > limit:
            break  # A partial number, label or unit is never presented.
        text += line
    return text


def fact_line(fact):
    years = list(re.finditer(r'20\d{2}', fact['column_header']))
    column = fact['column_header'][years[0].start():years[1].start()].strip() if len(years)>1 else ''
    column = re.sub(r'\s+', '', column)
    # Retain explicit YTD or balance-sheet dates; the report title alone cannot
    # distinguish a quarterly flow from cumulative January-to-September figures.
    suffix = (column + '；') if column and column != fact['period'][:4]+'年' else ''
    return f"{fact['metric']}={fact['value']} {fact['unit']}（{suffix}PDF{fact['pdf_page']}）"


def make_material(document, extracted):
    facts = extract_facts(document, extracted)
    if not facts:
        return None
    return {**document, 'categories': ['financials'], 'numeric_facts': facts, 'financial_protocol': VERSION,
            'excerpt': render_facts(facts, 3000), 'report_period': facts[0]['period'],
            'locator': 'PDF页序 ' + ','.join(str(p) for p in sorted({f['pdf_page'] for f in facts})),
            'source_spans': [span for f in facts for span in
                             ([{'pdf_page': f['pdf_page'], 'start': f['start'], 'end': f['end'],
                                'page_text_hash': f['page_text_hash']}] +
                              [{k:a[k] for k in ('pdf_page','start','end','page_text_hash')} for a in f.get('source_context',[])])],
            'unit_context_present': True, 'source_truth_verified': False,
            'extraction_status': 'table_rows_units_period_bound_semantics_unverified',
            'page_excerpt_complete': False, 'data_quality': 'imported'}


def references(materials, snapshot):
    """Observed-price ratios only; no scenario multiples, annualisation or price targets."""
    rows = []
    for quote in snapshot:
        candidates = [m for m in materials if m['symbol'] == quote['symbol']]
        output = {'symbol': quote['symbol'], 'kind': 'host_arithmetic_not_forecast', 'items': []}
        for metric, ratio in (('eps', 'historical_annual_PE'), ('bvps', 'reported_PB')):
            available = [(m, f) for m in candidates for f in m.get('numeric_facts', [])
                         if f['metric'] == metric and f['unit'] == '元/股'
                         and (metric != 'eps' or m.get('document_type') == 'annual')]
            available.sort(key=lambda pair: pair[0]['published_at'], reverse=True)
            if not available:
                continue
            material, fact = available[0]
            try:
                value, price = Decimal(fact['value']), Decimal(str(quote['close']))
                if not value.is_finite() or value <= 0 or not price.is_finite() or price <= 0:
                    continue
            except InvalidOperation:
                continue
            output['items'].append({'calculation': ratio, 'value': round(float(price/value), 4),
                                    'price': float(price), 'denominator': fact['value'], 'period': fact['period'],
                                    'material_id': material['material_id'], 'fact_id': fact['fact_id']})
        if output['items']:
            rows.append(output)
    return rows
