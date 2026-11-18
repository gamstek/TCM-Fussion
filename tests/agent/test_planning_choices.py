"""核对规划选项到实际模型短编号的映射以及模型响应解码。用真实图和替身回复验证单动作、实例目标及双观察选择，检查坏提案不会进入工具调用。

主要定义：
- PlanningChoicesTests。

关联：
- 直接依赖 「lib/agent/planning_choices.py」、「lib/agent/react_graph.py」、「lib/agent/semantic_execution.py」、「lib/runtime/storage.py」、「lib/tools/agent_tools.py」、「tests/support/workflow_fixtures.py」。
- 统一执行见 tests/README.md。
"""
import asyncio
import copy
import json
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from TCM_Fusion.tests.support import workflow_fixtures as fixtures
from TCM_Fusion.lib.tools.agent_tools import LocalToolbox, FixtureBackend
from TCM_Fusion.lib.agent.planning_choices import attach, model_view, decode
from TCM_Fusion.lib.agent.react_graph import LLMPlanner, run_react
from TCM_Fusion.lib.agent.semantic_execution import compile_goal
from TCM_Fusion.lib.runtime.storage import IntegrationError, read_json
from TCM_Fusion.lib.runtime.storage import write_json


class PlanningChoicesTests(unittest.TestCase):
    setUp=fixtures.WorkflowFixture.setUp
    runtime=fixtures.WorkflowFixture.runtime

    def detected(self, **kw):
        r=self.runtime(agent_policy='adaptive_v2',semantic_execution=True,max_resource_units=200,**kw)
        r.workflow.execute(next(a for a in r.workflow.propose() if a['tool']=='detect_fov_initial'))
        r.investigate(['G3'],'explicit test scope',r.score([])['evidence_digest'])
        return r

    def context(self,r,*,observations=None,**kw):
        c={'actions':r.legal_actions({'observations':observations or [],'attempted':[]}),
           'stop_advisory':{},'parallel_observations':[],**kw}
        return attach(r,c,can_pair=True)

    def run_agent(self,r,p,backend=None):
        toolbox=LocalToolbox(r,ledger_path=self.root/'observations.json',backend=backend or FixtureBackend())
        return asyncio.run(run_react(r,toolbox,planner=p,planner_mode='scripted',max_steps=12))

    def test_model_view_is_nonmutating_and_links_each_goal_to_visible_action(self):
        c=self.context(self.detected());before=copy.deepcopy(c);view,ids=model_view(c)
        self.assertEqual(c,before)
        self.assertTrue(view['execution_options']['goals'])
        for g in view['execution_options']['goals']:
            self.assertIn(g['entry_action_id'],ids)
            self.assertIn(g['goal']['semantic_task'],{'verify_identity','measure_length'})
        self.assertLessEqual(len(view['execution_options']['goals']),12)

    def test_same_action_uses_separate_primary_and_companion_short_ids(self):
        shared = {'action_id': 'canonical', 'tool': 'presence_check'}
        view, lookup = model_view({'actions': [shared], 'parallel_observations': [shared]})
        self.assertEqual(view['actions'][0]['action_id'], 'A1')
        self.assertEqual(view['parallel_observations'][0]['action_id'], 'O1')
        self.assertEqual(lookup, {'A1': 'canonical', 'O1': 'canonical'})

    def test_actual_llm_payload_uses_primary_goal_selector_and_maps_back(self):
        c=self.context(self.detected(),semantic_options=[{'old':'duplicate'}],semantic_goal_contract={})
        p=LLMPlanner.__new__(LLMPlanner);p.client=SimpleNamespace(model='fixture',timeout_s=2);p.config={}
        captured={}
        def post(payload,timeout):
            ctx=json.loads(payload['messages'][1]['content']);captured.update(ctx)
            goal=next(g for g in ctx['execution_options']['goals'] if g['goal']['semantic_task']=='measure_length')
            return {'choices':[{'message':{'content':json.dumps({'decision':{'kind':'instance_goal','goal_id':goal['goal_id']},'decision_summary':'test'})}}]}
        with patch('TCM_Fusion.lib.runtime.hard_calls.post',side_effect=post):reply,usage=p.choose(c)
        self.assertEqual(reply['semantic_goal']['semantic_task'],'measure_length')
        self.assertIn(reply['action_id'],[a['action_id'] for a in c['actions']])
        self.assertEqual(p.last_request_menu['execution_options'],captured['execution_options'])
        self.assertNotIn('semantic_options',captured);self.assertNotIn('parallel_questions',captured)

    def test_actual_llm_payload_keeps_legal_yolo_distinct_from_unchecked_evidence(self):
        r=self.runtime(agent_policy='adaptive_v2')
        image=next(iter(r.images))
        observation={'accepted':True,'action':{'tool':'presence_check',
            'args':{'image_id':image,'feature_id':'sand_crystal'}},
            'result':{'ok':True,'present':True,'conf':.8}}
        audit={'candidates':[{'candidate_id':'test','required':{'vessel':'unchecked'}}]}
        c=self.context(r,observations=[observation],evidence={'workflow':{'candidate_audit':audit},
            'observation_states':r.score([observation])['observation_states']})
        p=LLMPlanner.__new__(LLMPlanner)
        p.client=SimpleNamespace(model='fixture',timeout_s=2)
        p.config={}

        def post(payload,timeout):
            ctx=json.loads(payload['messages'][1]['content'])
            yolo=next(a for a in ctx['actions'] if a['tool']=='detect_fov_initial')
            self.assertIn('scene_check',{a['tool'] for a in ctx['actions']})
            self.assertEqual(ctx['action_state'][yolo['action_id']],
                             {'image_id':image,'entry_kind':'discovery'})
            self.assertEqual(ctx['evidence']['workflow']['candidate_audit'],audit)
            return {'choices':[{'message':{'content':json.dumps({
                'decision':{'kind':'action','action_id':yolo['action_id']}})}}]}

        with patch('TCM_Fusion.lib.runtime.hard_calls.post',side_effect=post) as mock_post:
            reply,_=p.choose(c)
        mock_post.assert_called_once()
        self.assertEqual(p.last_request_menu['investigation']['candidate_audit'],audit)
        self.assertEqual(reply['action_id'],
                         next(a['action_id'] for a in c['actions'] if a['tool']=='detect_fov_initial'))

    def test_same_view_existing_yolo_crop_and_scan_are_both_available(self):
        class ReadableCrop(fixtures.StageFixture):
            def execute(self,session,tool,target,work,context):
                result=super().execute(session,tool,target,work,context)
                if tool=='detect_fov_initial':
                    from PIL import Image
                    pixels=Image.new('L',(8,8))
                    pixels.putdata([255 if (i + i//8)%2 else 0 for i in range(64)])
                    pixels.save(Path(result['crops_dir'])/'crop.png')
                return result

        r=self.runtime(backend=ReadableCrop(),agent_policy='adaptive_v2',
                       fast_recovery_selection='per_feature',fast_fallback_max_features=10)
        r.workflow.execute(next(a for a in r.workflow.propose() if a['tool']=='detect_fov_initial'))
        r.investigate(['G3'],'explicit test scope',r.score([])['evidence_digest'])
        c=self.context(r)
        scan=next(a for a in c['actions'] if a['tool']=='feature_scan'
                  and c['action_state'][a['action_id']]['feature_id']=='single_crystal')
        crop=next(a for a in c['actions'] if a['tool']=='classify_target')
        hint=c['action_state'][scan['action_id']]
        self.assertIn(crop['action_id'],hint['existing_crop_actions'])
        self.assertGreater(hint['max_scan_tiles'],0)
        self.assertEqual(hint['presence_states'],[])
        view,ids=model_view(c)
        short=lambda action: next(k for k,v in ids.items() if v==action['action_id'])
        self.assertIn(short(crop),view['action_state'][short(scan)]['existing_crop_actions'])
        # A legal entry hidden by menu compaction must not show up as an offered choice.
        actions=[a for a in c['actions'] if a['action_id']!=crop['action_id']]
        shortened=attach(r,{**c,'actions':actions},can_pair=True)
        self.assertNotIn(crop['action_id'],
                         shortened['action_state'][scan['action_id']].get('existing_crop_actions',[]))
        explicit=decode({'decision':{'kind':'action','action_id':short(scan)}},view)
        self.assertEqual(ids[explicit['action_id']],scan['action_id'])

    def test_explicit_goal_runs_dependencies_and_returns_without_replanning_each_tool(self):
        r=self.detected(auto_measure_after_segment=False);seen=[]
        class Planner:
            def choose(inner,c):
                seen.append(c)
                if len(seen)>1:return {'decision':{'kind':'action','action_id':'review'}},None
                g=next(g for g in c['execution_options']['goals'] if g['goal']['semantic_task']=='measure_length')
                return {'decision':{'kind':'instance_goal','goal_id':g['goal_id']}},None
        t=self.run_agent(r,Planner())
        self.assertEqual(r.workflow.backend.calls,['detect_fov_initial','classify_target','segment_instance','measure_instances'])
        self.assertEqual(t['planning_requests'],2)
        self.assertTrue(any(e['phase']=='semantic_return' and e['reason']=='local_goal_satisfied' for e in t['events']))

    def test_single_action_stays_single_and_does_not_infer_a_goal(self):
        r=self.detected();seen=[]
        class Planner:
            def choose(inner,c):
                seen.append(c)
                a=next(a for a in c['actions'] if a['tool']==('classify_target' if len(seen)==1 else 'request_review'))
                return {'decision':{'kind':'action','action_id':a['action_id']}},None
        self.run_agent(r,Planner())
        self.assertNotIn('segment_instance',r.workflow.backend.calls)

    def test_review_available_even_with_goal_and_observation_offers(self):
        r=self.detected();before=list(r.workflow.backend.calls)
        class Planner:
            def choose(inner,c):return {'decision':{'kind':'action','action_id':'review'}},None
        t=self.run_agent(r,Planner());self.assertEqual(t['planning_requests'],1)
        self.assertEqual(r.workflow.backend.calls,before)

    def test_explicit_scene_presence_set_overlaps_and_preserves_partial_failure(self):
        r=self.detected(max_parallel=2,max_observations=4,max_api_calls=30)
        r.workflow.execute(next(a for a in r.workflow.propose() if a['tool']=='classify_target'))
        r.workflow.execute(next(a for a in r.workflow.propose() if a['tool']=='segment_instance'))
        r.workflow.execute(next(a for a in r.workflow.propose() if a['tool']=='measure_instances'))
        r.a={'status':'ok','fold_rankings':[[{'label':'白芍','probability':.4},
            {'label':'秦皮','probability':.3},{'label':'茯苓','probability':.3}]]*5}
        seen=[];barrier=threading.Barrier(2);called=[]
        class Backend:
            def call(inner,runtime,a,path):
                called.append(a['tool']);barrier.wait(timeout=5)
                if a['tool']=='presence_check':raise IntegrationError('UPSTREAM_REMOTE_DISCONNECTED','fixture')
                return FixtureBackend().call(runtime,a,path)
        class Planner:
            def choose(inner,c):
                seen.append(c)
                if len(seen)>1:return {'action_id':'review'},None
                q=c['execution_options']['questions']
                pair=[next(x for x in q if x['tool']==tool)['question_id'] for tool in ('scene_check','presence_check')]
                return {'decision':{'kind':'observe_questions','question_ids':pair}},None
        t=self.run_agent(r,Planner(),Backend())
        self.assertEqual(set(called),{'scene_check','presence_check'})
        self.assertEqual(len(t['observations']),2)
        self.assertEqual(sum(o['accepted'] for o in t['observations']),1)
        self.assertTrue(any(e['phase']=='parallel_join' for e in t['events']))
        self.assertIsNone(next(o for o in t['observations'] if not o['accepted'])['result'].get('present'))

    def test_no_pair_is_chosen_implicitly_and_small_budget_hides_pair_offers(self):
        r=self.runtime(agent_policy='adaptive_v2',max_observations=1);seen=[]
        class Planner:
            def choose(inner,c):
                seen.append(c)
                self.assertEqual(c['execution_options']['max_questions'],1)
                self.assertEqual(c['execution_options']['questions'],[])
                return {'decision':{'kind':'action','action_id':'review'}},None
        t=self.run_agent(r,Planner());self.assertEqual(t['observation_requests'],0)

    def test_unknown_duplicate_or_mixed_selector_never_guesses_first(self):
        c={'actions':[],'execution_options':{'questions':[{'question_id':'q','action_id':'a'}],'max_questions':2}}
        values=[{'decision':{'kind':'instance_goal','goal_id':'missing'}},
                {'decision':{'kind':'observe_questions','question_ids':['q','q']}},
                {'decision':{'kind':'observe_questions','question_ids':['future']}},
                {'decision':{'kind':'action','action_id':'a'},'action_id':'different'}]
        for v in values:
            with self.subTest(v=v),self.assertRaises(IntegrationError):decode(v,c)

    def test_goal_failure_returns_to_react_without_bypassing_quality(self):
        class Review(fixtures.StageFixture):
            def execute(inner,s,tool,target,work,context):
                result=super().execute(s,tool,target,work,context)
                if tool=='segment_instance':
                    result['quality']='review'
                    write_json(Path(result['segmentation'])/'instances.json',
                               {'instances':[{'igm_status':'review'}]})
                return result
        r=self.detected(backend=Review());seen=[]
        class Planner:
            def choose(inner,c):
                seen.append(c)
                if len(seen)>1:return {'action_id':'review'},None
                g=next(g for g in c['execution_options']['goals'] if g['goal']['semantic_task']=='measure_length')
                return {'decision':{'kind':'instance_goal','goal_id':g['goal_id']}},None
        t=self.run_agent(r,Planner());self.assertEqual(t['planning_requests'],2)
        self.assertNotIn('measure_instances',r.workflow.backend.calls)
        self.assertTrue(any(e['phase']=='semantic_return' and e['reason']=='mask_review' for e in t['events']))

    def test_identity_only_goal_cannot_authorize_segmentation_entry(self):
        r=self.detected();a=next(a for a in r.workflow.propose() if a['tool']=='classify_target')
        c=self.context(r);g=next(g for g in c['execution_options']['goals'] if g['goal']['semantic_task']=='verify_identity')
        r.workflow.execute(a);seg=next(a for a in r.workflow.propose() if a['tool']=='segment_instance')
        with self.assertRaises(IntegrationError):compile_goal(r,g['goal'],seg)

    def test_actual_offer_and_requested_mode_are_audited(self):
        r=self.detected()
        class Planner:
            def choose(inner,c):return {'decision':{'kind':'action','action_id':'review'}},None
        t=self.run_agent(r,Planner())
        offer=next(e for e in t['events'] if e['phase']=='planner_offer')
        self.assertIn('execution_options',offer['menu'])
        thought=next(e for e in t['events'] if e['phase']=='thought')
        self.assertEqual(thought['execution_choice']['kind'],'action')

    def test_thin_verification_row_still_links_its_visible_goal(self):
        r=self.detected();r.workflow.execute(next(a for a in r.workflow.propose() if a['tool']=='classify_target'))
        a=next(a for a in r.workflow.propose() if a['tool']=='segment_instance')
        thin={'action_id':a['action_id'],'tool':a['tool'],'target_id':a['args']['target_id']}
        c=attach(r,{'actions':[thin]},can_pair=False)
        goals=c['execution_options']['goals'];self.assertTrue(goals)
        self.assertTrue(all(g['entry_action_id']==a['action_id'] for g in goals))

    def test_existing_automatic_measurement_is_not_counted_as_new_saving(self):
        r=self.detected(auto_measure_after_segment=True)
        goals=self.context(r)['execution_options']['goals']
        self.assertEqual(next(g for g in goals if g['goal']['semantic_task']=='measure_length')['planning_saved_upper_bound'],1)
        r.workflow.execute(next(a for a in r.workflow.propose() if a['tool']=='classify_target'))
        goals=self.context(r)['execution_options']['goals']
        self.assertEqual(next(g for g in goals if g['goal']['semantic_task']=='measure_length')['planning_saved_upper_bound'],0)

    def test_alias_and_current_canonical_question_ids_have_same_expansion(self):
        c=self.context(self.detected());view,ids=model_view(c)
        questions=view['execution_options']['questions'];self.assertTrue(questions)
        q=questions[0]
        for identifier in (q['question_id'],q['canonical_question_id']):
            reply=decode({'decision':{'kind':'observe_questions','question_ids':[identifier]}},view)
            self.assertEqual(reply['action_id'],q['action_id'])
        self.assertNotIn('choice_id',q)

    def test_partially_stale_independent_set_executes_only_explicit_current_selection(self):
        c={'execution_options':{'max_questions':2,'questions':[{'question_id':'Q2','canonical_question_id':'physical:known','action_id':'A7'}]}}
        reply=decode({'decision':{'kind':'observe_questions','question_ids':['old-invalid-value','physical:known']}},c)
        self.assertEqual(reply['action_id'],'A7');self.assertNotIn('parallel_question_id',reply)
        self.assertEqual(reply['partial_choice_rejection']['replacement_questions_chosen'],0)
        self.assertNotIn('old-invalid-value',str(reply));self.assertEqual(reply['partial_choice_rejection']['extra_planning_requests'],0)

    def test_llm_advertises_current_modes_in_system_prompt_and_retry(self):
        actions = [
            {'action_id': 'one', 'tool': 'detect_fov_initial'},
            {'action_id': 'two', 'tool': 'detect_fov_initial'},
            {'action_id': 'presence', 'tool': 'presence_check'},
            {'action_id': 'identity', 'tool': 'classify_target'},
            {'action_id': 'review', 'tool': 'request_review'},
        ]
        offers = {
            'questions': [{'question_id': 'question:local', 'action_id': 'presence'}], 'max_questions': 2,
            'goals': [{'goal_id': 'goal:identity', 'entry_action_id': 'identity',
                       'goal': {'semantic_task': 'verify_identity'}}],
            'physical_groups': [{'tool': 'detect_fov_initial', 'views': [['one'], ['two']], 'limit': 2}],
        }
        planner = LLMPlanner.__new__(LLMPlanner)
        planner.client = SimpleNamespace(model='fixture', timeout_s=1)
        planner.config = {}
        body = {'choices': [{'message': {'content': '{"decision":{"kind":"action","action_id":"A5"}}'}}]}
        cases = ((offers, {'action', 'instance_goal', 'observe_questions', 'physical_group'}),
                 ({'questions': [], 'max_questions': 2, 'goals': []}, {'action'}))
        for menu, enabled in cases:
            with self.subTest(modes=enabled):
                context = {'actions': actions, 'stop_advisory': {}, 'execution_options': menu,
                           'format_feedback': {'reason': 'PLANNER_QUESTION_UNAVAILABLE'}}
                with patch('TCM_Fusion.lib.runtime.hard_calls.post', return_value=body) as post:
                    reply, _ = planner.choose(context)
                self.assertEqual(reply['action_id'], 'review')
                system, user = post.call_args.args[0]['messages']
                text = system['content']
                sent = json.loads(user['content'])
                self.assertEqual(set(sent['format_feedback']['allowed_decision_keys']), enabled)
                for mode in ('action', 'instance_goal', 'observe_questions', 'physical_group'):
                    self.assertEqual(f'"kind":"{mode}"' in text, mode in enabled)
                self.assertEqual([a['tool'] for a in sent['actions']], [a['tool'] for a in actions])
                if 'observe_questions' in enabled:
                    self.assertIn('选1项', text)

    def test_non_observation_action_ids_in_question_group_are_rejected(self):
        ctx = {'actions': [{'action_id': 'A1', 'tool': 'feature_scan'},
                           {'action_id': 'A2', 'tool': 'feature_scan'}],
               'execution_options': {'max_questions': 2, 'questions': []}}
        reply = {'decision': {'kind': 'observe_questions', 'question_ids': ['A1', 'A2']}}
        with self.assertRaises(IntegrationError) as caught:
            decode(reply, ctx)
        self.assertEqual(caught.exception.code, 'PLANNER_QUESTION_UNAVAILABLE')
        decoded = decode({'decision': {'kind': 'action', 'action_id': 'A1'}}, ctx)
        self.assertEqual(decoded['action_id'], 'A1')
        self.assertNotIn('physical_partner_ids', decoded)

    def test_invalid_llm_choice_retains_real_usage_and_safe_diagnostic(self):
        c=self.context(self.detected());p=LLMPlanner.__new__(LLMPlanner)
        p.client=SimpleNamespace(model='fixture',timeout_s=1);p.config={}
        body={'choices':[{'message':{'content':json.dumps({'decision':{'kind':'observe_questions','question_ids':['Q999']}})}}],
              'usage':{'prompt_tokens':12,'completion_tokens':3,'total_tokens':15}}
        with patch('TCM_Fusion.lib.runtime.hard_calls.post',return_value=body),self.assertRaises(IntegrationError)as caught:p.choose(c)
        self.assertEqual(caught.exception.usage,body['usage'])
        self.assertEqual(caught.exception.choice_diagnostic['unmatched_short_ids'],['Q999'])

    def test_invalid_json_retains_returned_usage_without_inventing_count(self):
        p=LLMPlanner.__new__(LLMPlanner);p.client=SimpleNamespace(model='fixture',timeout_s=1);p.config={}
        body={'choices':[{'message':{'content':'not json'}}],'usage':{'total_tokens':19}}
        with patch('TCM_Fusion.lib.runtime.hard_calls.post',return_value=body),self.assertRaises(ValueError)as caught:p.choose(self.context(self.detected()))
        self.assertEqual(caught.exception.usage,{'total_tokens':19})

    def test_partial_group_feedback_reaches_next_normal_plan_without_paid_correction(self):
        r=self.runtime(agent_policy='adaptive_v2',max_observations=4)
        r.a={'status':'ok','fold_rankings':[[{'label':'秦皮','probability':1}]]*5};seen=[]
        class Planner:
            def choose(inner,c):
                seen.append(c)
                if len(seen)>1:
                    self.assertEqual(c['selection_feedback'][-1]['rejected_questions'],1)
                    return {'action_id':'review'},None
                q=c['execution_options']['questions'][0]
                return {'decision':{'kind':'observe_questions','question_ids':['stale',q['question_id']]}},None
        t=self.run_agent(r,Planner());self.assertEqual(t['planning_requests'],2)
        self.assertEqual(t['observation_requests'],1)
        self.assertFalse(any(e['phase']=='rejected_proposal' for e in t['events']))
