"""检验先验领先者、竞争解释和实际合法工具入口在 ReAct 中保持可见。检查格式修复、候选输出一致性及充分性影子状态。HTTP 与视觉由可控替身提供。

主要定义：
- FairInvestigationTests。

关联：
- 直接依赖 「lib/agent/action_feedback.py」、「lib/agent/planning_choices.py」、「lib/agent/react_graph.py」、「lib/decision/candidate_audit.py」、「lib/decision/comparison_problems.py」、「lib/decision/final_assessment.py」、「lib/decision/output_projection.py」、「lib/runtime/storage.py」。
- 统一执行见 tests/README.md。
"""
import asyncio
import copy
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from TCM_Fusion.tests.support import workflow_fixtures as fixtures
from TCM_Fusion.tests.support.result_builders import _verdict, _result, _a_prediction
from TCM_Fusion.lib.tools.agent_tools import LocalToolbox, FixtureBackend
from TCM_Fusion.lib.decision.candidate_audit import build, readiness_shadow
from TCM_Fusion.lib.decision.comparison_problems import prior_summary, compact
from TCM_Fusion.lib.agent.planning_choices import decode, diagnose, model_view
from TCM_Fusion.lib.agent.react_graph import run_react, LLMPlanner
from TCM_Fusion.lib.decision.output_projection import project_output
from TCM_Fusion.lib.agent.action_feedback import snapshot, evaluate
from TCM_Fusion.lib.decision.final_assessment import finalize
from TCM_Fusion.lib.runtime.storage import IntegrationError
from TCM_Fusion.lib.runtime.storage import read_json


class FairInvestigationTests(fixtures.FairInvestigationFixture, unittest.TestCase):

    def observation(self, r, fid, present=True):
        return {'accepted':True, 'action':{'tool':'presence_check','action_id':'fixture:'+fid,
            'args':{'image_id':next(iter(r.images)), 'feature_id':fid, 'morph_hint':''}},
            'result':{'ok':True,'present':present,'conf':.9}}

    def run_agent(self, r, planner):
        box = LocalToolbox(r,ledger_path=self.root/'observations.json',backend=FixtureBackend(
            {'gelatinized_starch':{'ok':True,'present':True,'conf':.9}}))
        return asyncio.run(run_react(r,box,planner=planner,planner_mode='scripted',max_steps=6))

    def question_context(self):
        return {'execution_options':{'max_questions':2,'questions':[
            {'question_id':'Q1','canonical_question_id':'p:1','action_id':'a:1'},
            {'question_id':'Q2','canonical_question_id':'p:2','action_id':'a:2'}]}}

    def test_explanatory_fields_preserve_explicit_choice_without_mutating_reply(self):
        reply={'decision':{'kind':'observe_questions','question_ids':['Q1','Q2'],
                           'reason':'两个独立问题','override_reason':'当前比较仍有价值'}}
        original=copy.deepcopy(reply);value=decode(reply,self.question_context())
        self.assertEqual(reply,original);self.assertEqual(value['action_id'],'a:1')
        self.assertEqual(value['parallel_question_id'],'p:2')
        self.assertEqual(value['decision_summary'],'两个独立问题')
        self.assertFalse(value['choice_normalization']['selected_action_changed'])

    def test_unknown_execution_field_is_rejected_and_name_diagnosed_without_value(self):
        reply={'decision':{'kind':'observe_questions','question_ids':['Q1'],
                           'tool':'sensitive-value-must-not-be-logged'}}
        with self.assertRaises(IntegrationError):decode(reply,self.question_context())
        diagnostic=diagnose(reply,self.question_context())
        self.assertIn('tool',diagnostic['extra_field_names'])
        self.assertNotIn('sensitive-value',json.dumps(diagnostic))

    def test_nontext_metadata_and_mixed_or_duplicate_selectors_remain_invalid(self):
        for choice in [
            {'kind':'action','action_id':'a','reason':{'tool':'other'}},
            {'kind':'action','action_id':'a','goal_id':'g'},
            {'kind':'observe_questions','question_ids':['Q1','Q1'],'reason':'duplicate'}]:
            with self.subTest(choice=choice),self.assertRaises(IntegrationError):
                decode({'decision':choice},self.question_context())

    def test_one_paid_format_repair_uses_same_menu_and_never_executes_bad_choice(self):
        r=self.prepare();seen=[]
        class Planner:
            def choose(inner,c):
                seen.append(copy.deepcopy(c))
                if len(seen)==1:
                    return {'decision':{'kind':'action','action_id':'review','unknown_field':'x'}},None
                return {'decision':{'kind':'action','action_id':'review'}},None
        trace=self.run_agent(r,Planner())
        self.assertEqual(trace['planning_requests'],2);self.assertEqual(trace['steps'],0)
        self.assertFalse(seen[1]['format_feedback']['action_executed'])
        self.assertEqual(seen[0]['actions'],seen[1]['actions'])
        self.assertEqual(trace['stop_reason'],'planner_request_review')
        self.assertEqual(len([e for e in trace['events']if e.get('phase')=='planning_retry']),1)
        self.assertEqual(read_json(r.workflow.budget_path)['used']['planning'],2)

    def test_real_llm_serializer_preserves_metadata_choice_and_returns_usage_once(self):
        r=self.prepare();from TCM_Fusion.lib.agent.planning_choices import attach
        context=attach(r,{'actions':r.legal_actions({'observations':[],'attempted':[]}),
            'evidence':r.summary({'observations':[],'attempted':[]}),
            'stop_advisory':{},'parallel_observations':[]},can_pair=True)
        planner=LLMPlanner.__new__(LLMPlanner)
        planner.client=SimpleNamespace(model='fixture',timeout_s=1);planner.config={}
        captured={}
        def response(payload, timeout):
            c=json.loads(payload['messages'][1]['content']);captured.update(c)
            q=c['execution_options']['questions'][0]
            return {'choices':[{'message':{'content':json.dumps({'decision':{
                'kind':'observe_questions','question_ids':[q['question_id']], 'reason':'explicit choice'}})}}],
                'usage':{'total_tokens':51}}
        with patch('TCM_Fusion.lib.runtime.hard_calls.post',side_effect=response)as post:
            reply,usage=planner.choose(context)
        self.assertEqual(post.call_count,1);self.assertEqual(usage['total_tokens'],51)
        self.assertEqual(reply['decision_summary'],'explicit choice')
        cards=captured['evidence']['workflow']['comparison_problems']['questions']
        cluster=next(q for q in cards if q['feature_id']=='cluster_crystal')
        self.assertTrue(cluster['entry_paths'])
        self.assertIn('investigation',planner.last_request_menu)

    def test_repeated_bad_format_stops_after_two_attempts_with_no_tool_or_guess(self):
        r=self.prepare()
        class Planner:
            def choose(inner,c):
                return {'decision':{'kind':'action','action_id':'review','unknown_field':'x'}},None
        trace=self.run_agent(r,Planner())
        self.assertEqual(trace['planning_requests'],2);self.assertEqual(trace['steps'],0)
        self.assertEqual(trace['stop_reason'],'runtime_request_review')
        self.assertEqual(trace['observations'],[])

    def test_raw_image_cluster_question_reaches_actual_yolo_without_invented_instance(self):
        r=self.prepare();view=r.decision_view({'observations':[],'attempted':[]})
        row=next(c for c in view['candidate_audit']['candidates']if c['candidate_id']=='白芍')
        criterion=row['required'][0]
        self.assertEqual(criterion['availability_reason'],'reachable_via_prerequisite')
        self.assertTrue(criterion['action_ids'])
        paths=criterion['entry_paths'];self.assertTrue(all(p['tool']=='detect_fov_initial'for p in paths))
        self.assertTrue(all(not p['is_evidence']for p in paths))
        self.assertFalse(any(v.get('kind')=='crop'for v in r.workflow.targets().values()))
        qs=[q for q in view['comparison']['questions']if q['feature_id']=='cluster_crystal']
        self.assertTrue(all(q['capability']=='reachable_via_prerequisite'for q in qs))
        consumers={cid for q in qs for cid in q['required_by']}
        self.assertIn('白芍',consumers);self.assertIn('G1',consumers)
        # 共同发现入口可复用，不强行合并形态/量纲不同的候选判据问题。
        self.assertTrue(all(q['legal_action_ids']==qs[0]['legal_action_ids']for q in qs))

    def test_no_current_action_does_not_fabricate_a_discovery_path(self):
        r=self.prepare();audit=build(r,r.score([]),[],actions=[])
        row=next(c for c in audit['candidates']if c['candidate_id']=='白芍')['required'][0]
        self.assertEqual(row['action_ids'],[]);self.assertEqual(row['availability_reason'],'no_current_legal_action')

    def test_actual_model_cards_only_reference_current_short_action_ids(self):
        r=self.prepare();c={'actions':r.legal_actions({'observations':[],'attempted':[]}),
                           'evidence':r.summary({'observations':[],'attempted':[]})}
        view,ids=model_view(c)
        cards=view['evidence']['workflow']['comparison_problems']['questions']
        card=next(q for q in cards if q['feature_id']=='cluster_crystal')
        self.assertTrue(card['entry_paths'])
        self.assertTrue(all(p['action_id'] in ids for p in card['entry_paths']))
        self.assertEqual(view['evidence']['workflow']['investigation_focus']['a_leads'],['白芍'])

    def test_prior_fold_disagreement_survives_without_probability_averaging(self):
        r=self.prepare();prior=prior_summary(r.candidate_interest())
        self.assertEqual(prior['fold_top1_counts'],{'白芍':4})
        self.assertEqual(prior['unmapped_fold_top1'],{'玄明粉':1})
        self.assertEqual(prior['fold_observations'],5)

    def test_competing_positive_does_not_erase_main_gap_or_claim_exclusion(self):
        r=self.prepare();obs=[self.observation(r,'gelatinized_starch')]
        view=r.decision_view({'observations':obs,'attempted':[]});focus=view['investigation_focus']
        self.assertIn('cluster_crystal',focus['lead_required_gaps']['白芍'])
        self.assertIn('gelatinized_starch',focus['competing_support']['郁金'])
        cards=compact(view['comparison'])['questions']
        self.assertEqual(cards[0]['feature_id'],'cluster_crystal')

    def test_model_may_choose_competitor_before_yolo_and_then_return_to_yolo(self):
        r=self.prepare();calls=[]
        class Planner:
            def choose(inner,c):
                tools=['presence_check','detect_fov_initial'];index=len(calls)
                if index>=2:return {'action_id':'review'},None
                tool=tools[index];a=next(a for a in c['actions']if a['tool']==tool and
                    (tool!='presence_check' or a['args']['feature_id']=='gelatinized_starch'))
                calls.append(tool);return {'action_id':a['action_id']},None
        trace=self.run_agent(r,Planner())
        self.assertEqual(calls,['presence_check','detect_fov_initial'])
        self.assertEqual(trace['stop_reason'],'planner_request_review')

    def test_model_can_choose_yolo_immediately_without_automatic_presence(self):
        r=self.prepare();calls=[]
        class Planner:
            def choose(inner,c):
                calls.append(c)
                if len(calls)>1:return {'action_id':'review'},None
                return {'action_id':next(a['action_id']for a in c['actions']if a['tool']=='detect_fov_initial')},None
        trace=self.run_agent(r,Planner());self.assertEqual(trace['observation_requests'],0)
        self.assertEqual(trace['steps'],1)

    def test_supporting_audit_gaps_are_distinct_from_shadow_sufficiency(self):
        audit={'complete':False,'unmapped':[],'candidates':[{'candidate_id':'c',
               'required':[{'feature_id':'r','assessed':True}],
               'supporting':[{'feature_id':'s','assessed':False}]}]}
        shadow=readiness_shadow(audit,{'verdict':'confirmed','needs_review':False,'conclusion':{'id':'c'}})
        self.assertTrue(shadow['would_be_ready']);self.assertFalse(shadow['audit_complete'])
        self.assertFalse(shadow['production_authorization_changed'])

    def test_shadow_does_not_promote_incomplete_real_cohort(self):
        r=self.prepare();obs=[self.observation(r,'gelatinized_starch')]
        verdict=copy.deepcopy(r.score(obs)['decision']);verdict.update(
            verdict='confirmed',needs_review=False,conclusion={'id':'郁金'})
        final=finalize(r,verdict,{'observations':obs,'events':[],'stop_reason':'planner_request_review'},self.root/'final')
        self.assertTrue(final['review_required']);self.assertEqual(final['status'],'tentative')
        self.assertEqual(final['decision_readiness_shadow']['mode'],'shadow')

    def test_one_canonical_final_candidate_controls_output_without_rewriting_fusion(self):
        verdict=_verdict();original=copy.deepcopy(verdict);result=_result()
        result['final_assessment']={'schema':'final-assessment/v1','candidate_id':'菟丝子',
            'status':'tentative','review_required':True,'judgement':'暂定菟丝子，需复核。'}
        out=project_output('fixture',verdict,result,_a_prediction())
        self.assertEqual(out['top_candidate']['id'],'菟丝子');self.assertIn('菟丝子',out['judgement'])
        self.assertEqual(out['fusion_ranking'][0]['id'],'射干');self.assertEqual(verdict,original)
        self.assertIn('射干',[r['id']for r in out['alternatives']])

    def test_unknown_final_candidate_fails_closed_instead_of_contradictory_output(self):
        result={'final_assessment':{'schema':'final-assessment/v1','candidate_id':'made-up',
            'status':'confirmed','review_required':False,'judgement':'untrusted'}}
        out=project_output('fixture',_verdict(),result,_a_prediction())
        self.assertIsNone(out['top_candidate']);self.assertIsNone(out['judgement'])
        self.assertEqual(out['status'],'undetermined');self.assertTrue(out['review_required'])

    def test_competitor_support_is_not_reported_as_main_counter_or_discrimination(self):
        r=self.prepare();before=snapshot(r.score([]),primary_ids=['白芍'])
        after=snapshot(r.score([self.observation(r,'gelatinized_starch')]))
        f=evaluate(before,after,{'tool':'presence_check','args':{'feature_id':'gelatinized_starch'}},
                   accepted=True,result={'present':True,'conf':.9})
        self.assertIn('郁金',f['candidate_support_changes'])
        self.assertEqual(f['qualified_counter_changes'],[])
        self.assertFalse(f['primary_required_progress']);self.assertEqual(f['discrimination_progress'],'not_established')

    def test_true_quantitative_counter_is_separate_from_positive_support(self):
        before={'features':{},'checks':{'a:f':'NOT_DETECTED'},'ranking':['a'],
                'primary_ids':['a'],'required':{'a':['f']}}
        after={**before,'checks':{'a:f':'OUT_OF_RANGE'}}
        f=evaluate(before,after,{'tool':'measure_instances'},accepted=True)
        self.assertEqual(f['qualified_counter_changes'],['a:f']);self.assertTrue(f['primary_required_progress'])
        self.assertEqual(f['discrimination_progress'],'qualified_counter_added')
