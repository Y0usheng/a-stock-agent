import asyncio
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from agent.research_retrieval import (question_material, compact_question_excerpt, coverage, coverage_summary)
from agent.subscription_roles import export_paper_context, current_materials, SubscriptionRoles
from evals.experiments import canonical_hash
from evals.research_sources import validate_windows, fact_results, evaluate, assertion_prechecks, present_sources
from tests.test_investment import example


def pages(texts):
    return {'pages': [{'pdf_page': i+1, 'text': t, 'text_hash': canonical_hash(t)} for i, t in enumerate(texts)]}


class ResearchSourceTests(unittest.TestCase):
    def test_bank_question_aliases_select_actual_risk_facts_not_speech_or_regulatory_minimum(self):
        extracted = pages(['600036 2026年第三季度报告',
            '董事长致辞\n主要风险 净息差 不良贷款率 核心一级资本充足率。',
            '本集团不良贷款率0.94%；拨备覆盖率389.39%。\n本集团净利息收益率1.82%。',
            '监管要求核心一级资本充足率应分别不低于8.25%。\n截至报告期末，本集团核心一级资本充足率14.39%。',
            '2026年1-9月，本集团信用减值损失417.75亿元，同比增长25.26%。'])
        material = question_material({'symbol': '600036', 'document_type': 'interim'}, extracted)
        self.assertIn('不良贷款率0.94%', material['excerpt'])
        self.assertIn('净利息收益率1.82%', material['excerpt'])
        self.assertIn('资本充足率14.39%', material['excerpt'])
        self.assertNotIn('董事长致辞', material['excerpt'])
        self.assertEqual(material['report_period'], '2026第三季度')
        self.assertLessEqual(len(material['excerpt']), 600)
        validate_windows(material, extracted)

    def test_whole_source_windows_and_explicit_budget_omission_survive_actual_export(self):
        extracted = pages(['600276 2025年度报告', '创新药销售收入163.42亿元。\n费用化研发投入69.61亿元。\n对外许可收入确认10亿元。'])
        row = question_material({'symbol': '600276', 'document_type': 'annual'}, extracted)
        row.update(material_id='qa_one', title='测试年报', url='https://static.cninfo.com.cn/test.pdf',
                   published_at='2026-03-26', first_seen_at='2026-04-01', expires_at='2027-04-01')
        context, _ = example(False)
        context.update(symbols=['600276'], execution_policy='autonomous-paper-execution-v1')
        context['investment_context']['materials'] = [row]
        projected = export_paper_context(context, 'cn', True)
        self.assertEqual(projected['research_materials'][0]['excerpt'], row['excerpt'])
        self.assertTrue(all(not r['semantic_verified'] for r in projected['retrieval_coverage']))
        shorter = {**row, 'excerpt': compact_question_excerpt(row, 0)}
        inventory = coverage([row], [shorter])
        self.assertIn('omitted_by_input_budget', {r['status'] for r in inventory})
        self.assertIn('not_found_in_captured_text', {r['status'] for r in inventory})
        self.assertIn('visible_candidate_not_verified', {r['status'] for r in coverage([row], [row])})

    def test_modified_excerpt_or_page_hash_cannot_pass_source_binding(self):
        extracted = pages(['600276 2025年度报告', '创新药销售收入163.42亿元。'])
        row = question_material({'symbol': '600276', 'document_type': 'annual'}, extracted)
        validate_windows(row, extracted)
        bad = deepcopy(row); bad['excerpt'] = bad['excerpt'].replace('163.42', '1634.2')
        with self.assertRaises(ValueError):
            validate_windows(bad, extracted)
        bad = deepcopy(extracted); bad['pages'][1]['text'] += 'changed'
        with self.assertRaises(ValueError):
            validate_windows(row, bad)
        for excerpt in ('额外内容' + row['excerpt'], row['excerpt'].replace('[PDF2 ', '[PDF99 ')):
            with self.assertRaises(ValueError):
                validate_windows({**row, 'excerpt': excerpt}, extracted)

    def test_actual_role_prompt_and_delivery_audit_exclude_future_sources(self):
        context, _ = example(False)
        context.update(symbols=['600276'], execution_policy='autonomous-paper-execution-v1',
                       timing={'information_cutoff_at':'2026-10-28T18:00:00+08:00','price_as_of_date':'2026-10-28',
                               'execution_session':'2026-10-29','approval_deadline':'2026-10-29T14:55:00+08:00',
                               'max_price_deviation_pct':.02})
        context['snapshot'][0]['symbol'] = '600276'
        row = question_material({'symbol':'600276','document_type':'annual'},
                                pages(['600276 2025年度报告', '创新药销售收入163.42亿元。']))
        row.update(material_id='qa_one', title='测试年报', url='https://static.cninfo.com.cn/test.pdf',
                   published_at='2026-03-26', first_seen_at='2026-04-01', expires_at='2027-04-01')
        class Captured(Exception): pass
        class CaptureClient:
            async def complete(self, prompt, schema, **kwargs):
                self.prompt = prompt
                raise Captured()
        for future in (False, True):
            with self.subTest(future=future):
                context['investment_context']['materials'] = [{**row, 'first_seen_at':'2026-10-29' if future else '2026-04-01'}]
                roles = SubscriptionRoles(context, 'cn', 'fixture', 'fixture', long_term=True)
                client = CaptureClient(); roles.client = client
                with self.assertRaises(Captured):
                    asyncio.run(roles.call('fundamental-analyst', 'fixture'))
                self.assertIn('source_coverage', client.prompt)
                if future:
                    self.assertNotIn('创新药销售收入163.42', client.prompt)
                    self.assertIn('excluded_by_source_checks', client.prompt)
                else:
                    self.assertIn('创新药销售收入163.42', client.prompt)
                    self.assertIn('2025年度', client.prompt)
                states = {r['status'] for r in roles.audit()['source_delivery']}
                self.assertIn('excluded_by_source_checks' if future else 'visible_candidate_not_verified', states)

    def test_other_industries_use_operating_questions_without_stock_recommendation(self):
        for symbol, text, phrase in [
            ('600519', '直销收入100亿元。主要产品收入200亿元。合同负债50亿元。', '直销'),
            ('300750', '主要业务包括电池。前五名客户占比40%。产能利用率70%。', '前五名客户'),
            ('601899', '矿产铜产量100万吨。单位成本2000元。在建工程20亿元。', '矿产铜'),
            ('600900', '发电量100亿千瓦时。上网电价0.3元。在建工程10亿元。', '发电量')]:
            with self.subTest(symbol=symbol):
                row = question_material({'symbol': symbol, 'document_type': 'annual'}, pages([symbol+' 2025年度报告',text]))
                self.assertIn(phrase, row['excerpt'])
                self.assertFalse(row['source_truth_verified'])
                self.assertNotIn('target_weight', row)

    def test_original_problem_fact_visible_source_missing_or_wrong_document_is_not_passed(self):
        case = {'symbol': '600036', 'pdf_sha256': 'original',
                'facts': [{'id': 'margin', 'pdf_page': 2, 'text': '净利息收益率1.82%'}]}
        extracted = pages(['600036 2026年第三季度报告', '净利息收益率1.82%。'])
        self.assertEqual(fact_results(case, extracted, [])[0]['status'], 'fail')
        wrong = {'symbol':'600036','pdf_sha256':'other_report','excerpt':'净利息收益率1.82%'}
        self.assertEqual(fact_results(case, extracted, [wrong])[0]['status'], 'fail')
        correct = {**wrong, 'pdf_sha256':'original'}
        self.assertEqual(fact_results(case, extracted, [correct])[0]['status'], 'pass')

    def test_missing_source_files_fail_acceptance_instead_of_skipping_to_success(self):
        context, _ = example(False)
        with tempfile.TemporaryDirectory() as tmp:
            report = evaluate(Path(tmp), context)
        self.assertEqual(report['status'], 'failed')
        self.assertEqual(len(report['unavailable_sources']), 4)
        self.assertEqual(report['model_requests'], 0)
        self.assertEqual(report['orders_submitted'], 0)

    def test_acceptance_retains_eligible_financial_baselines_beside_operating_passages(self):
        context, _ = example(False)
        context.update(symbols=['600036'], execution_policy='autonomous-paper-execution-v1')
        context['snapshot'][0]['symbol'] = '600036'
        document = {'symbol':'600036','document_type':'annual','title':'测试年报','announcement_id':'qa',
                    'published_at':'2026-03-28','url':'https://static.cninfo.com.cn/qa.pdf','pdf_sha256':'qa'}
        extracted = pages(['600036 2025年度报告',
            '主要会计数据\n单位：人民币元\n项目  2025年  2024年\n基本每股收益（元/股）  5.70  5.20\n',
            '零售金融业务对本公司营收和利润的贡献占比在50%以上。'])
        projected = present_sources(context, [(document,extracted)])
        self.assertTrue(any(r.get('numeric_facts') for r in projected['research_materials']))
        self.assertTrue(any(r.get('question_protocol') for r in projected['research_materials']))

    def test_fixed_semantic_traps_remain_prechecks_not_automatic_entailment(self):
        rows = assertion_prechecks()
        self.assertEqual(len(rows), 6)
        self.assertTrue(all(r['status']=='pass' for r in rows))
        self.assertTrue(all(not r['semantic_verified'] for r in rows))

    def test_current_input_preserves_annual_and_latest_interim_question_packs(self):
        rows = [{'material_id':k,'categories':['business'],'question_protocol':'question-source-windows-v1',
                 'published_at':day,'document_type':kind} for k,day,kind in
                [('old','2025-03-01','annual'),('annual','2026-03-01','annual'),('latest','2026-10-31','interim')]]
        rows.append({'material_id':'speech','categories':['business'],'published_at':'2026-11-01'})
        self.assertEqual({r['material_id'] for r in current_materials(rows)}, {'annual','latest'})


if __name__ == '__main__':
    unittest.main()
