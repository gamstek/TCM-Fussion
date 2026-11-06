"""Closing judgement, provenance and shared budget; no live model requests."""
import asyncio
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from TCM_Fusion.lib.storage import IntegrationError,read_json,write_json
from TCM_Fusion.lib.workflow_runtime import WorkflowRuntime
from TCM_Fusion.lib.final_assessment import build_packet,finalize,validate_reply,request_model
from TCM_Fusion.lib.agent_tools import LocalToolbox,FixtureBackend
from TCM_Fusion.lib.react_graph import run_react,ScriptedPlanner
from TCM_Fusion.tests.test_workflow import StageFixture


class FinalAssessmentTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.image=self.root/'input.png';self.image.write_bytes(b'fixture')
        a={'status':'ok','fold_rankings':[[{'label':'秦皮','probability':.7},
                                          {'label':'桔梗','probability':.3}]]*5}
        self.runtime=WorkflowRuntime([self.image],root=self.root/'workflow',sample_id='s',a=a,
            simulation=True,backend=StageFixture(),workflow_config={'agent_policy':'adaptive_v2'})
        image=next(iter(self.runtime.images))
        self.observation={'accepted':True,'action':{'tool':'presence_check','action_id':'fixture',
            'args':{'image_id':image,'feature_id':'sand_crystal','morph_hint':''}},
            'result':{'ok':True,'present':True,'conf':.75}}
        self.result={'observations':[self.observation],'stop_reason':'override_exhausted',
                     'steps':2,'events':[],'needs_review':True}
        decision=self.runtime.score(self.result['observations'])['decision']
        self.verdict=copy.deepcopy(decision)
        self.verdict['ranking']=[r for r in decision['ranking'] if r['id'] in {'秦皮','桔梗'}]
        self.verdict['needs_review']=True

    def reply(self,packet):
        ref=next(f['id'] for f in packet['facts'] if f['kind']=='criterion'
                 and '秦皮' in f['supports_candidate_ids'])
        return {'candidate_id':'秦皮','support_refs':[ref],'counter_refs':[],
                'unresolved_refs':[],'explanation':'砂晶存在性支持秦皮，但未形成充分鉴别证据。'},None

    def test_every_stop_reason_still_produces_read_only_judgement(self):
        before=self.runtime.workflow.store.current()['version'];original=copy.deepcopy(self.verdict)
        for reason in ['override_exhausted','max_steps','workflow_deadline','request_budget',
                       'no_legal_action','planner_request_review']:
            result={**self.result,'stop_reason':reason}
            value=finalize(self.runtime,self.verdict,result,self.root/reason)
            self.assertTrue(value['judgement']);self.assertEqual(value['candidate_id'],'秦皮')
            self.assertTrue(value['supporting_evidence']);self.assertTrue(value['candidate_comparison'])
            self.assertEqual(value['new_image_calls'],0);self.assertTrue(value['review_required'])
        self.assertEqual(self.runtime.workflow.store.current()['version'],before)
        self.assertEqual(self.verdict,original)

    def test_injected_model_assessment_is_distinct_from_rule_confirmation(self):
        value=finalize(self.runtime,self.verdict,self.result,self.root/'model',responder=self.reply)
        self.assertEqual(value['method'],'injected_finalizer')
        self.assertEqual(value['status'],'tentative');self.assertTrue(value['simulation'])
        self.assertIsNone(value['confidence_basis']['calibrated_probability'])

    def test_snapshot_does_not_trigger_legacy_planning_or_review_calls(self):
        with patch.object(self.runtime,'summary',side_effect=AssertionError('no planning callbacks')):
            value=finalize(self.runtime,self.verdict,self.result,self.root/'readonly')
        self.assertTrue(value['judgement'])

    def test_text_request_uses_reserved_pool_and_records_actual_usage(self):
        from types import SimpleNamespace
        packet=build_packet(self.runtime,self.verdict,self.result)
        self.runtime.workflow.limits['finalization']=1
        reply,_=self.reply(packet)
        import json
        body={'choices':[{'message':{'content':json.dumps(reply,ensure_ascii=False)}}],
              'usage':{'total_tokens':123}}
        client=SimpleNamespace(model='fixture',api_key='fixture')
        with patch('adjudicator.AdjudicatorClient',return_value=client), \
             patch('TCM_Fusion.lib.hard_calls.post',return_value=body) as post:
            result,usage=request_model(self.runtime,packet)
        self.assertEqual(result,reply);self.assertEqual(usage,body['usage'])
        self.assertEqual(post.call_count,1)
        self.assertLessEqual(post.call_args.args[1],30)
        self.assertEqual(read_json(self.runtime.workflow.budget_path)['used']['finalization'],1)

    def test_unknown_candidate_or_fabricated_reference_is_rejected(self):
        packet=build_packet(self.runtime,self.verdict,self.result)
        good,_=self.reply(packet)
        for invalid in [{**good,'candidate_id':'imaginary'},
                        {**good,'support_refs':['nonexistent:1']}]:
            with self.assertRaises(IntegrationError):validate_reply(packet,invalid)

    def test_failed_observation_cannot_support_positive_judgement(self):
        result={**self.result,'observations':[{'accepted':False,'action':self.observation['action'],
                   'result':{'ok':False,'error_code':'TIMEOUT'}}]}
        packet=build_packet(self.runtime,self.verdict,result)
        failed=next(f for f in packet['facts'] if f['kind']=='observation')
        self.assertEqual(failed['supports_candidate_ids'],[])
        self.assertEqual(failed['counters_candidate_ids'],[])
        good,_=self.reply(build_packet(self.runtime,self.verdict,self.result))
        with self.assertRaises(IntegrationError):
            validate_reply(packet,{**good,'support_refs':[failed['id']]})

    def test_true_negative_is_scoped_counterevidence_not_global_absence(self):
        negative=copy.deepcopy(self.observation)
        negative['action']['args']['feature_id']='secretory_tissue'
        negative['result']['present']=False
        packet=build_packet(self.runtime,self.verdict,{**self.result,'observations':[negative]})
        obs=next(f for f in packet['facts'] if f['kind']=='observation')
        self.assertIn('桔梗',obs['counters_candidate_ids'])
        self.assertIn('bound field of view',obs['data']['scope'])

    def test_invalid_model_answer_falls_back_without_new_tools(self):
        def bad(packet):
            good,usage=self.reply(packet)
            return {**good,'support_refs':['made_up']},usage
        value=finalize(self.runtime,self.verdict,self.result,self.root/'invalid',responder=bad)
        self.assertEqual(value['method'],'deterministic')
        self.assertEqual(value['model_failure'],'ASSESSMENT_INVALID')
        self.assertEqual(value['candidate_id'],'秦皮')

    def test_no_model_request_after_deadline_or_close(self):
        budget=read_json(self.runtime.workflow.budget_path);budget['deadline']=0
        write_json(self.runtime.workflow.budget_path,budget)
        responder=unittest.mock.Mock(side_effect=AssertionError('must not call'))
        value=finalize(self.runtime,self.verdict,self.result,self.root/'expired',responder=responder)
        self.assertEqual(value['model_failure'],'WORKFLOW_TIMEOUT');responder.assert_not_called()
        self.assertTrue(value['judgement'])

    def test_budget_keeps_one_closing_request_after_planning_is_exhausted(self):
        w=self.runtime.workflow;w.limits.update(planning=10,network=2,finalization=1,total=20)
        w.charge('planning','p1')
        with self.assertRaises(IntegrationError):w.charge('planning','p2')
        w.charge('finalization','last')
        used=read_json(w.budget_path)['used']
        self.assertEqual(used['network'],2);self.assertEqual(used['finalization'],1)

    def test_injected_finalizer_is_denied_in_real_scope(self):
        self.runtime.simulation=False
        with self.assertRaises(IntegrationError):
            finalize(self.runtime,self.verdict,self.result,self.root/'denied',responder=self.reply)

    def test_reserves_time_then_exits_investigation_without_calling_tool(self):
        w=self.runtime.workflow;w.limits['finalization']=1
        budget=read_json(w.budget_path);import time
        budget['deadline']=time.time()+5;write_json(w.budget_path,budget)
        result=asyncio.run(run_react(self.runtime,LocalToolbox(self.runtime,
            ledger_path=self.root/'ledger.json',backend=FixtureBackend()),
            planner=ScriptedPlanner(),planner_mode='scripted'))
        self.assertEqual(result['stop_reason'],'finalization_time_reserve')
        self.assertEqual(result['steps'],0)
