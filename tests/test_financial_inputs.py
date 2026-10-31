from copy import deepcopy
import unittest

from agent.financial_inputs import extract_facts, make_material, references, render_facts
from agent.subscription_roles import current_materials, material_excerpt
from evals.experiments import canonical_hash


def pages(*texts):
    return {'pages':[{'pdf_page':i+1,'text':text,'text_hash':canonical_hash(text)} for i,text in enumerate(texts)]}


DOC = {'symbol':'600036','announcement_id':'test','document_type':'annual','published_at':'2026-03-28'}


class FinancialInputTests(unittest.TestCase):
    def bank(self):
        return pages('600036 招商银行2025年度报告\n主要会计数据和财务指标\n'
                     '（人民币百万元，特别注明除外）  2025年  2024年  2023年\n'
                     '营业收入    300,000    290,000    280,000\n'
                     '每股计（人民币元）\n'
                     '归属于本行普通股股东的基本每股收益(1)    5.70    5.66    5.63\n'
                     '归属于本行普通股股东的每股净资产\n'
                     '（人民币元）(1)    45.40    43.43    41.46\n'
                     '财务比率(%)\n'
                     '归属于本行普通股股东的加权平均净资产收益率    13.44    14.49    16.22\n')

    def test_currency_scale_and_per_share_units_are_separate(self):
        facts = {r['metric']:r for r in extract_facts(DOC,self.bank())}
        self.assertEqual(facts['revenue']['value'],'300000')
        self.assertEqual(facts['revenue']['unit'],'百万元')
        self.assertEqual(facts['eps']['value'],'5.70')
        self.assertEqual(facts['eps']['unit'],'元/股')
        self.assertEqual(facts['bvps']['value'],'45.40')
        self.assertEqual(facts['roe']['unit'],'%')
        source = self.bank()['pages'][0]['text']
        for fact in facts.values():
            self.assertEqual(source[fact['start']:fact['end']],fact['source_line'])

    def test_reversed_columns_unknown_units_and_diluted_eps_are_rejected(self):
        original = self.bank()['pages'][0]['text']
        reversed_years = original.replace('2025年  2024年','2024年  2025年')
        self.assertEqual(extract_facts(DOC,pages(reversed_years)),[])
        text = '600036 2025年度报告\n主要会计数据\n2025年  2024年\n'
        self.assertEqual(extract_facts(DOC,pages(text+'营业收入    100    90\n')),[])
        self.assertEqual(extract_facts(DOC,pages(text+'稀释每股收益（元/股）    5    4\n')),[])

    def test_wrapped_cash_capex_and_header_continuation_preserve_anchors(self):
        source = pages('601899 2026年第三季度报告\n合 并 现 金 流 量 表\n'
                       '单位：元\n项目  2026年1-9月  2025年1-9月\n',
                       '经营活动产生的现金流量净额    840    600\n'
                       '购建固定资产、无形资产和其他长期资\n'
                       '                180    150\n'
                       '产支付的现金\n母公司现金流量表\n单位：元\n2026年1-9月  2025年1-9月\n'
                       '经营活动产生的现金流量净额    99999    99\n')
        facts = {r['metric']:r for r in extract_facts({**DOC,'symbol':'601899','document_type':'interim'},source)}
        self.assertEqual(facts['operating_cashflow']['value'],'840')
        self.assertEqual(facts['capex_cash']['value'],'180')
        self.assertEqual(facts['capex_cash']['source_context'][0]['pdf_page'],1)
        f=facts['capex_cash']
        self.assertEqual(source['pages'][1]['text'][f['start']:f['end']],f['source_line'])

    def test_parent_statement_heading_at_bottom_does_not_hide_group_row_above(self):
        source=pages('601899 2025年度报告\n合并资产负债表\n单位：元\n2025年  2024年\n',
                     '长期借款    400    300\n母公司资产负债表\n单位：元\n2025年  2024年\n短期借款    999    99\n')
        facts=extract_facts({**DOC,'symbol':'601899'},source)
        self.assertEqual([(f['metric'],f['value']) for f in facts],[('long_borrowings','400')])

    def test_references_use_annual_eps_and_latest_book_value_without_annualisation(self):
        annual=make_material(DOC,self.bank());annual['material_id']='annual'
        interim=deepcopy(annual);interim.update(document_type='interim',published_at='2026-08-29',material_id='interim')
        for fact in interim['numeric_facts']:
            fact['period']='2026半年度'
            if fact['metric']=='eps':fact['value']='2.98'
        ratios=references([annual,interim],[{'symbol':'600036','close':43.55}])[0]['items']
        self.assertEqual(ratios[0]['denominator'],'5.70')
        self.assertEqual(ratios[1]['period'],'2026半年度')
        for fact in annual['numeric_facts']:
            if fact['metric']=='eps':fact['value']='0'
        self.assertNotIn('historical_annual_PE',[r['calculation'] for r in references([annual,interim],[{'symbol':'600036','close':43.55}])[0]['items']])

    def test_packet_selection_keeps_table_facts_and_never_slices_numbers(self):
        packet=make_material(DOC,self.bank())
        original={**packet,'numeric_facts':[],'excerpt':'普通财务段落'}
        self.assertEqual(current_materials([original,packet])[0],packet)
        text=material_excerpt(packet,170)
        self.assertEqual(text,render_facts(packet['numeric_facts'],170))
        self.assertLessEqual(len(text),170)
        for line in text.splitlines()[1:]:
            self.assertTrue(line.endswith('）'))

    def test_cashflow_column_period_is_visible_and_bank_risk_ratios_are_distinct(self):
        from agent.financial_inputs import fact_line
        text='600036 2026年半年度报告\n主要会计数据\n2026年1-6月  2025年1-6月\n财务比率(%)\n'
        text+='净利息收益率    1.83    1.87\n拨备覆盖率    385.10    391.79\n信用成本    0.69    0.60\n'
        facts=extract_facts({**DOC,'document_type':'interim'},pages(text))
        self.assertEqual({f['metric'] for f in facts},{'net_interest_margin','provision_coverage','credit_cost'})
        self.assertTrue(all('2026年1-6月' in fact_line(f) for f in facts))

    def test_new_extraction_keeps_legacy_text_cache_unchanged(self):
        import hashlib,json
        import pymupdf
        from unittest.mock import patch
        from agent.disclosures import collect, TEXT_PROTOCOL
        from agent.investment import ResearchBook
        from agent.subscription import subscription_workspace
        doc=pymupdf.open();page=doc.new_page()
        page.insert_text((40,40),'600036 2025年度报告\n'+('主要业务营业收入经营活动现金流及经营模式\n'*35),fontname='china-s',fontsize=9)
        data=doc.tobytes();doc.close();digest=hashlib.sha256(data).hexdigest()
        metadata={**DOC,'title':'测试年度报告','url':'https://static.cninfo.com.cn/fixture.pdf',
                  'pdf_sha256':digest,'first_downloaded_at':'2026-04-01T00:00:00+08:00'}
        with subscription_workspace() as root:
            (root/(digest+'.pdf')).write_bytes(data)
            legacy=root/(digest+'.pages.json');legacy.write_text('legacy-cache',encoding='utf-8')
            metadata['pages_sha256']=hashlib.sha256(legacy.read_bytes()).hexdigest()
            (root/'test.capture.json').write_text(json.dumps(metadata),encoding='utf-8')
            with patch('agent.disclosures.query_reports',return_value=[metadata]):
                result=collect(['600036'],book=ResearchBook(root/'research.db'),root=root)
            self.assertFalse(result['errors'])
            self.assertEqual(legacy.read_text(encoding='utf-8'),'legacy-cache')
            captured=json.loads((root/'test.capture.json').read_text(encoding='utf-8'))
            self.assertEqual(captured['pages_sha256'],metadata['pages_sha256'])
            self.assertIn(TEXT_PROTOCOL,captured['extractions'])
