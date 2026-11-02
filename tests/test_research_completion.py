from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from agent.contracts import capture_risk_review
from agent.evidence import validate_claims
from agent.investment import ResearchBook, validate_cases, RESEARCH_PROTOCOL
from agent.reference_slots import reference_slots, rewrite_references, restore_visible_references
from agent.subscription_roles import SubscriptionRoles
from tests.test_investment import example, DAY
from tests.test_after_close import timing


def rejected():
    context, log=example(True)
    log['execution_mode']='paper'
    decision=log['decisions'][0];decision.update(action='HOLD',shares=0)
    review=deepcopy(log['risk_review']);review['assessments'][0].update(action='HOLD',max_shares=0)
    log['risk_review']=capture_risk_review(log['decisions'],review,'cn',{'300750'})
    case=log['investment_cases'][0]
    case.update(research_status='rejected',position_intent='reject',target_weight=0,thesis_change='invalidated',entry_condition='若新公司披露推翻负面前提则重新研究')
    case['valuation'].update(bear_price=None,base_price=None,bull_price=None,method='测试披露否定投资前提，本次不估值')
    return context, log


class CompletionTests(unittest.TestCase):
    def test_evidenced_rejection_can_finish_without_invented_valuation(self):
        context,log=rejected();validate_cases(log,{'300750'},catalog=log['evidence_audit']['catalog'],as_of=DAY,account=context['account'])
        with tempfile.TemporaryDirectory() as temp:
            book=ResearchBook(Path(temp)/'research.db');book.save(log,'v')
            self.assertEqual(book.review_order(['300750'],DAY,[log]),[])
            self.assertEqual(book.review_order(['300750'],DAY,[{**log,'execution_mode':'shadow'}]),['300750'])
            self.assertEqual(book.review_order(['300750'],log['investment_cases'][0]['next_review_date'],[log]),['300750'])

    def test_rejection_cannot_hide_missing_facts_or_existing_holdings(self):
        context,log=rejected();case=log['investment_cases'][0]
        for change in ({'core_gaps':[{'category':'financials','gap_type':'missing_disclosure','reason':'缺资料','what_to_fetch':'取得原始数据'}]},
                       {'thesis_change':'unchanged'},{'target_weight':.1}):
            bad=deepcopy(log);bad['investment_cases'][0].update(change)
            with self.assertRaises(ValueError):validate_cases(bad,{'300750'},catalog=log['evidence_audit']['catalog'])
        with self.assertRaises(ValueError):validate_cases(log,{'300750'},catalog=log['evidence_audit']['catalog'],account={**context['account'],'positions':[{'symbol':'300750','shares':100}]})
        held=deepcopy(log);held['input_dataset']['account']['positions']=[{'symbol':'300750','shares':100}]
        with self.assertRaises(ValueError):validate_cases(held,{'300750'},catalog=log['evidence_audit']['catalog'])
        case['evidence']['supporting_ids']=[]
        with self.assertRaises(ValueError):validate_cases(log,{'300750'},catalog=log['evidence_audit']['catalog'])

    def test_pending_types_are_explicit_and_legacy_records_still_validate(self):
        _,log=example(False);gap=log['investment_cases'][0]['core_gaps'][0]
        gap['gap_type']='future_certainty'
        with self.assertRaises(ValueError):validate_cases(log,{'300750'})
        log.pop('research_protocol');gap.pop('gap_type')
        validate_cases(log,{'300750'})

    def test_empty_account_cannot_use_continue_holding_as_a_research_verdict(self):
        context,log=example(True);log['investment_cases'][0].update(position_intent='hold',target_weight=0)
        log['risk_review']['proposed_decisions'][0].update(action='HOLD',shares=0)
        with self.assertRaisesRegex(ValueError,'空仓'):validate_cases(log,{'300750'},account=context['account'])

    def test_rejection_closes_tasks_without_claiming_the_missing_facts_were_resolved(self):
        _,pending=example(False);_,log=rejected()
        with tempfile.TemporaryDirectory() as temp:
            book=ResearchBook(Path(temp)/'research.db');book.save(pending,'v')
            with book.db() as conn:book._save_tasks(conn,log,log['investment_cases'][0])
            self.assertTrue(all(t['status']=='closed_rejected' for t in book.tasks()))

    def test_reference_round_trip_preserves_source_text_numbers_and_unknown_ids(self):
        _,log=example();catalog=log['evidence_audit']['catalog'];slots=reference_slots(catalog)
        original={'evidence_id':catalog['records'][0]['evidence_id'],'claim':'E1','data':{'value':1.59,'text':'E2'},'evidence':{'supporting_ids':list(slots),'contradicting_ids':[]}}
        wire=rewrite_references(original,slots)
        self.assertEqual(wire['claim'],'E1');self.assertEqual(wire['data'],original['data'])
        self.assertEqual(rewrite_references(wire,{v:k for k,v in slots.items()}),original)
        self.assertEqual(rewrite_references({'supporting_ids':['E999']},{v:k for k,v in slots.items()})['supporting_ids'],['E999'])

    def test_known_but_unshown_reference_is_not_granted_by_guessing_a_short_slot(self):
        slots={'visible':'E1','hidden':'E2'}
        for ref in ('E2','hidden','E999'):
            with self.assertRaises(ValueError):restore_visible_references({'evidence':{'supporting_ids':[ref],'contradicting_ids':[]}},slots,{'visible'})


class ReferenceIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_short_references_restore_before_role_and_eligibility_checks(self):
        context,_=example();context.update(timing=timing(),execution_policy='autonomous-paper-execution-v1')
        roles=SubscriptionRoles(context,'cn','fixture','gpt-6.1-sol',long_term=True)
        quote=next(r for r in roles.catalog['records'] if r['kind']=='market_snapshot' and r['eligible'])
        slot=roles.slots[quote['evidence_id']]
        class Client:
            async def complete(self,prompt,schema,role):
                self.prompt=prompt
                return {'summary':'fixture','claims':[{'symbol':'300750','claim':'仅验证冻结报价引用','evidence':{'supporting_ids':[slot],'contradicting_ids':[],'invalidation_condition':'报价修订失效'}}]}
        roles.client=Client();result=json.loads(await roles.call('fundamental-analyst','fixture'))
        self.assertEqual(result['claims'][0]['evidence']['supporting_ids'],[quote['evidence_id']])
        self.assertEqual(roles.raw_outputs['fundamental-analyst']['claims'][0]['evidence']['supporting_ids'],[slot])
        self.assertIn(slot,roles.client.prompt)
        slot='E999'
        with self.assertRaises(ValueError):await roles.call('fundamental-analyst','fixture')


if __name__=='__main__':unittest.main()
