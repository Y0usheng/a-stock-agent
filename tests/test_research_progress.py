from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from agent.disclosures import select_passages
from agent.investment import ResearchBook
from agent.learning import past_logs
from agent.memory import _load_past_logs
from agent.research_quality import QualityBook
from agent.research_tasks import PROTOCOL, questions, question_id
from evals.experiments import canonical_hash
from evals.public_reports import sanitize_view
from tests.test_investment import example, material


class ResearchProgressTests(unittest.TestCase):
    def test_quarantine_is_dated_idempotent_and_excludes_learning_memory_and_cases(self):
        _, log = example(False)
        log.update(execution_mode='paper', timing={'decision_at':'2026-10-28T17:30:00+08:00'},
                   agent_sdk={'role_models':{'investment-manager':{'model':'gpt-6.1-sol'}}})
        before = canonical_hash(log)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); quality=QualityBook(root/'quality'); book=ResearchBook(root/'research.db',quality=quality)
            book.save(log,'variant')
            (root/'after_close').mkdir(); (root/'after_close'/'valid.cn.json').write_text(json.dumps(log),encoding='utf-8')
            with patch('agent.research_quality.HOME',root/'quality'):
                self.assertEqual(len(past_logs('2026-10-29T17:30:00+08:00',root)),1)
                issue=quality.exclude(log,'input_corruption',recorded_at='2026-10-28T18:00:00+08:00')
                self.assertEqual(quality.exclude(log,'input_corruption')['recorded_at'],issue['recorded_at'])
                self.assertEqual(len(past_logs('2026-10-28T17:45:00+08:00',root)),1)
                self.assertEqual(past_logs('2026-10-29T17:30:00+08:00',root),[])
                self.assertEqual(_load_past_logs(root,'cn','2026-10-29',True,'2026-10-29T17:30:00+08:00'),[])
                self.assertEqual(book.snapshot(['300750'],'2026-10-29','variant')['previous_cases'],[])
                view={'stocks':[{'symbol':'300750','decision_at':log['timing']['decision_at']},{'symbol':'300750','decision_at':'2026-10-29T17:30:00+08:00'}]}
                quality.annotate(view,[log]);self.assertEqual(view['stocks'][0]['tracking_status'],'input_quarantined')
                self.assertNotIn('tracking_status',view['stocks'][1])
                self.assertTrue(all(t['status']=='quarantined' for t in book.tasks() if t.get('task_protocol')==PROTOCOL))
            self.assertEqual(canonical_hash(log),before)
            self.assertEqual(canonical_hash(json.loads((root/'after_close'/'valid.cn.json').read_text())),before)

    def test_multiple_financial_gaps_are_distinct_and_omission_does_not_resolve(self):
        _, log=example(False)
        log['investment_cases'][0]['core_gaps']=[
            {'category':'financials','gap_type':'missing_disclosure','reason':'缺负债','what_to_fetch':'取得债务到期分布；取得受限资金披露。'},
            {'category':'financials','gap_type':'missing_disclosure','reason':'缺资本承诺','what_to_fetch':'取得资本承诺项目。'}]
        with tempfile.TemporaryDirectory() as tmp:
            book=ResearchBook(Path(tmp)/'research.db');book.save(log,'v')
            tasks=book.tasks();self.assertEqual(len(tasks),3);self.assertEqual(len({t['task_id'] for t in tasks}),3)
            with book.db() as conn:
                case={**log['investment_cases'][0],'core_gaps':[],'research_status':'pending'}
                book._save_tasks(conn,{**log,'as_of_date':'2026-10-29'},case)
            self.assertTrue(all(t['status']!='resolved' for t in book.tasks()))
            self.assertEqual(book.task_snapshot('2000-01-01T00:00:00+08:00'),[])

    def test_source_found_needs_ready_manager_and_citation_before_closure(self):
        _, log=example(False)
        with tempfile.TemporaryDirectory() as tmp:
            book=ResearchBook(Path(tmp)/'research.db');book.save(log,'v');task=book.tasks()[0]
            source=book.add_material({**material(),'task_ids':[task['task_id']],
                'source_spans':[{'pdf_page':1,'start':0,'end':5,'page_text_hash':'fixture'}]})
            book.record_capture([task],[source],cutoff=datetime.now(timezone.utc).isoformat())
            self.assertEqual(book.tasks()[0]['capture_status'],'source_found')
            self.assertNotEqual(book.tasks()[0]['status'],'resolved')
            case={**log['investment_cases'][0],'research_status':'ready','core_gaps':[],
                  'evidence':{'supporting_ids':[]}}
            fake={**log,'investment_cases':[case],'evidence_audit':{'catalog':{'records':[]}}}
            with book.db() as conn:book._save_tasks(conn,fake,case)
            self.assertNotEqual(book.tasks()[0]['status'],'resolved')
            case['evidence']['supporting_ids']=['ev_fixture']
            fake['evidence_audit']['catalog']['records']=[{'evidence_id':'ev_fixture','kind':'company_document','eligible':True,'payload':{'material_id':source['material_id']}}]
            with book.db() as conn:book._save_tasks(conn,fake,case)
            self.assertEqual(book.tasks()[0]['status'],'resolved')

    def test_unregistered_source_does_not_count_and_failed_capture_is_not_no_information(self):
        _, log=example(False)
        with tempfile.TemporaryDirectory() as tmp:
            book=ResearchBook(Path(tmp)/'research.db');book.save(log,'v');task=book.tasks()[0]
            source={**material(),'material_id':'unknown','task_ids':[task['task_id']]}
            book.record_capture([task],[source],cutoff=datetime.now(timezone.utc).isoformat(),had_error=True)
            result=book.tasks()[0];self.assertEqual(result['evidence'],[]);self.assertEqual(result['capture_status'],'capture_error')

    def test_legacy_migration_recovers_same_category_gaps_from_immutable_case(self):
        _, log=example(False);case=log['investment_cases'][0]
        case['core_gaps'].append({'category':'financials','gap_type':'missing_disclosure','reason':'另一缺口','what_to_fetch':'取得债务到期披露'})
        with tempfile.TemporaryDirectory() as tmp:
            book=ResearchBook(Path(tmp)/'research.db')
            legacy='gap_'+canonical_hash(['300750','financials'])[:24]
            payload={**case['core_gaps'][-1],'task_id':legacy,'symbol':'300750','review_by':'2026-11-04','status':'open','source_case_hash':canonical_hash(case)}
            with book.db() as conn:
                conn.execute('INSERT INTO cases VALUES(?,?,?,?,?,?)',('old','300750','2026-10-28','v',canonical_hash(log),json.dumps(case)))
                conn.execute('INSERT INTO tasks VALUES(?,?,?,?,?,?,?)',(legacy,'300750','financials','2026-10-28','2026-10-28','open',json.dumps(payload)))
            book.migrate_tasks();book.migrate_tasks()
            tasks=book.tasks();self.assertEqual(len(tasks),2)
            self.assertTrue(all(t['review_by']=='2026-11-04' and t['opened_at']=='2026-10-28' for t in tasks))

    def test_task_specific_new_phrase_binds_exact_page_span_without_claiming_resolution(self):
        gap={'category':'business','reason':'未知新问题','what_to_fetch':'取得海外分销网络披露'}
        task={**gap,'task_id':question_id('300750',gap),'task_protocol':PROTOCOL,'status':'open'}
        text='300750 2025年度报告\n海外分销网络披露\n'+('原始业务文字，尚未语义复核。'*20)
        extracted={'pages':[{'pdf_page':1,'text':text,'text_hash':canonical_hash(text)}]}
        rows=select_passages({'symbol':'300750','document_type':'annual'},extracted,[task])
        found=next(r for r in rows if task['task_id'] in r.get('task_ids',[]))
        span=found['source_spans'][0];self.assertEqual(text[span['start']:span['end']],found['excerpt'])
        self.assertFalse(found['source_truth_verified']);self.assertIn('not_semantic_resolution',found['extraction_status'])

    def test_public_progress_is_whitelisted_and_idempotent(self):
        raw={'research_tasks':[{'task_id':'gap_example','symbol':'300750','status':'source_found','what_to_fetch':'取得成本原文','review_by':'2026-11-04',
            'evidence':[{'url':'https://static.cninfo.com.cn/report.pdf','locator':'PDF第8页','source_spans':[{'secret':'private'}]}],
            'source_log_hash':'private','attempt_count':1}], 'research_quality':[{'symbols':['300750'],'reason':'financial_column_error','detected_at':'2026-11-02T19:00:00+08:00','status':'excluded_from_learning_and_memory','proofs':'private'}]}
        public=sanitize_view(raw);self.assertEqual(public,sanitize_view(public));self.assertNotIn('private',json.dumps(public))


if __name__=='__main__':unittest.main()
