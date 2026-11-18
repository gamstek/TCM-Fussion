"""文档20回归：反思问题稳定性、同轮策略/后验和上下文对拍；无外部API。"""
import copy
import unittest
import json
from pathlib import Path
from unittest.mock import patch

from TCM_Fusion.tests import test_d3 as fixtures
from TCM_Fusion.tests import test_fair_investigation as fair
from TCM_Fusion.lib.reflection_controller import ReflectionController
from TCM_Fusion.lib.comparison_problems import range_relation, build
from TCM_Fusion.lib.planning_context import compact, deduplicate_notes
from TCM_Fusion.lib.planning_choices import audit, model_view, decode
from TCM_Fusion.lib.react_graph import compact_evidence, LLMPlanner
from TCM_Fusion.lib.semantic_execution import options
from TCM_Fusion.lib.storage import IntegrationError, read_json, write_json


class ReflectionStrategyTests(unittest.TestCase):
    setUp = fixtures.D3Tests.setUp
    runtime = fixtures.D3Tests.runtime
    detected = fixtures.D3Tests.detected
    controller = fixtures.D3Tests.controller
    run_agent = fixtures.D3Tests.run_agent

    def test_same_failure_does_not_refresh_card_when_ledger_grows(self):
        r, c, action, card = self.controller()
        for _ in range(3):
            c.record(action, progress=False, feedback={'problem_id':'same_question','accepted':True},
                     evidence_version=r.workflow.store.current()['version'])
            self.assertEqual(c.propose({'no_gain':5})['reflection_id'], card['reflection_id'])
        self.assertEqual(len(c.cards),1)

    def test_same_issue_new_input_gets_revision_without_new_issue(self):
        r,c,action,card=self.controller()
        bound=c.binding(action);bound['input_fingerprint']='new-input'
        c.record(action,progress=False,feedback={'accepted':True},binding=bound,evidence_version='v2')
        newer=c.propose({'no_gain':2})
        self.assertEqual(newer['issue_id'],card['issue_id'])
        self.assertEqual(newer['revision'],card['revision']+1)

    def test_real_capability_advancement_does_not_trigger_score_plateau(self):
        r=self.detected();c=ReflectionController(r);decision=r.score([])['decision']
        for i in range(4):
            c.record(r.action('read_scale',{},'evidence_inspection'), progress=True,
                feedback={'accepted':True,'progress_axes':{'capability':True}},
                evidence_version='v'+str(i),candidate_state=decision)
        self.assertIsNone(c.propose({'no_gain':0}));self.assertEqual(c.cards,[])

    def test_first_service_failure_is_not_hidden_by_diagnostic_file_progress(self):
        r=self.detected();c=ReflectionController(r)
        c.record(r.action('read_scale',{},'evidence_inspection'), progress=True,
            feedback={'accepted':False,'failure_code':'UPSTREAM_REMOTE_DISCONNECTED',
                      'progress_axes':{'diagnostic':True}}, evidence_version='failure-log')
        card=c.propose({'no_gain':0})
        self.assertEqual(card['failure_type'],'service')
        self.assertEqual(len(c.cards),1)

    def test_parallel_child_positive_and_local_negative_are_progress_without_join_axes(self):
        r=self.detected();c=ReflectionController(r);decision=r.score([])['decision']
        for i in range(4):
            feedback={'accepted':True,'new_features':['gelatinized_starch'] if i%2 else [],
                      'new_counterevidence':['stoma'] if not i%2 else [],'useful':True}
            c.record(r.action('read_scale',{},'evidence_inspection'),progress=True,feedback=feedback,
                     evidence_version='v'+str(i),candidate_state=decision)
        self.assertIsNone(c.propose({'no_gain':0}))
        self.assertTrue(all(row['progress_axes']['evidence'] for row in c.rows))

    def test_failed_child_does_not_borrow_successful_partner_progress(self):
        r=self.detected();c=ReflectionController(r)
        action=r.action('read_scale',{},'evidence_inspection')
        c.record(action,progress=True,feedback={'accepted':True,'new_features':['fiber']},evidence_version='v1')
        c.record(action,progress=False,feedback={'accepted':False,
            'failure_code':'UPSTREAM_REMOTE_DISCONNECTED'},evidence_version='v1')
        self.assertFalse(c.rows[-1]['progress_axes']['evidence'])
        self.assertEqual(c.propose({'no_gain':0})['failure_type'],'service')

    def response(self, card, disposition='maintain'):
        return {'issue_id':card['issue_id'],'revision':card['revision'],
                'disposition':disposition,'reason':'当前证据不足，改变有效条件后再查'}

    def assert_acknowledgement(self, disposition):
        r,c,action,card=self.controller()
        before=r.workflow.store.current()['version']
        c.respond(self.response(card,disposition),action,{})
        self.assertIsNone(c.pending);self.assertIsNone(c.propose({'no_gain':3}))
        self.assertEqual(r.workflow.store.current()['version'],before)
        self.assertEqual(c.applications,[])

    def test_maintain_acknowledgement(self):
        self.assert_acknowledgement('maintain')

    def test_reject_acknowledgement(self):
        self.assert_acknowledgement('reject')

    def test_missing_response_keeps_card_and_never_requests_extra_planning(self):
        r,c,action,card=self.controller()
        row=c.respond(None,action,{})
        self.assertEqual(row['disposition'],'not_acknowledged')
        self.assertEqual(row['extra_planning_calls'],0);self.assertEqual(c.pending,card)

    def strategy_fixture(self, expected='qualified_evidence'):
        r,c,action,card=self.controller()
        selected=r.action('list_fovs',{},'evidence_inspection')
        context={'evidence':{'workflow':{'comparison_problems':{'questions':[
            {'problem_id':'question:current','legal_action_ids':[selected['action_id']],
             'consumers':[{'capability_gap':None}]}]}}}}
        reply=self.response(card,'apply')
        reply['strategy']={'problem_id':'question:current','change_condition':'different_method',
                           'expected':expected,'stop_when':'no_progress'}
        return r,c,selected,context,reply

    def test_apply_tracks_strategy_delta_and_qualified_outcome(self):
        r,c,selected,context,reply=self.strategy_fixture()
        c.respond(reply,selected,context)
        self.assertTrue(c.applications[-1]['policy_delta']['strategy_target_changed'])
        self.assertFalse(c.applications[-1]['policy_delta']['changes_current_menu_or_scope'])
        c.record(selected,progress=True,feedback={'accepted':True,'candidate_support_changes':{'白芍':['cluster_crystal']},
            'resolved_criteria':['白芍:cluster_crystal']},evidence_version='v3')
        post=c.applications[-1]
        self.assertTrue(post['expected_outcome_observed']);self.assertTrue(post['goal_gap_resolved'])
        self.assertFalse(post['causal_improvement_validated'])

    def test_new_file_or_diagnostic_progress_is_not_qualified_evidence(self):
        r,c,selected,context,reply=self.strategy_fixture()
        c.respond(reply,selected,context)
        c.record(selected,progress=True,feedback={'accepted':True,'progress_axes':{'diagnostic':True}},
                 evidence_version='new-file-new-version')
        self.assertFalse(c.applications[-1]['qualified_fact_added'])
        self.assertFalse(c.applications[-1]['effective'])

    def test_strategy_cannot_reference_future_problem_or_unrelated_action(self):
        r,c,selected,context,reply=self.strategy_fixture()
        reply['strategy']['problem_id']='future:invented'
        with self.assertRaises(IntegrationError):c.respond(reply,selected,context)
        self.assertIsNone(c.active)

    def test_stop_intent_cannot_hide_unknown_problem_reference(self):
        r,c,selected,context,reply=self.strategy_fixture(expected='stop')
        selected={'tool':'request_review','action_id':'review','args':{}}
        reply['strategy'].update(problem_id='future:invented',change_condition='stop')
        with self.assertRaises(IntegrationError):c.respond(reply,selected,context)
        reply['strategy']['problem_id']=reply['issue_id']
        c.respond(reply,selected,context);c.finalize('planner_request_review')
        self.assertTrue(c.applications[-1]['expected_outcome_observed'])
        self.assertFalse(c.applications[-1]['qualified_fact_added'])

    def test_knowledge_reference_boundary_is_information_not_action_ban(self):
        r,c,selected,context,reply=self.strategy_fixture()
        context['evidence']['workflow']['comparison_problems']['questions'][0]['consumers']=[
            {'capability_gap':'knowledge_reference'}]
        c.respond(reply,selected,context)
        self.assertIn('cannot_be_generated',c.active['policy_delta']['applicability'])
        self.assertTrue(c.guard(selected))

    def test_goal_posterior_waits_for_dependent_evidence_then_closes(self):
        r,c,selected,context,reply=self.strategy_fixture()
        c.respond(reply,selected,context,goal={'plan_id':'current-goal'})
        c.record(selected,progress=True,feedback={'accepted':True,'progress_axes':{'capability':True}},
                 evidence_version='v3')
        self.assertIsNotNone(c.active);self.assertTrue(c.applications[-1]['posterior_pending'])
        c.close_goal('mask_review');self.assertIsNone(c.active)
        self.assertFalse(c.applications[-1]['expected_outcome_observed'])

    def test_optional_bad_response_preserves_valid_main_choice_without_paid_retry(self):
        r=self.detected();seen=[]
        class Planner:
            def choose(inner,context):
                seen.append(context)
                return {'action_id':'review','reflection_response':{'quality':'valid'}},None
        trace=self.run_agent(r,Planner())
        self.assertEqual(trace['planning_requests'],1)
        self.assertEqual(trace['stop_reason'],'planner_request_review')
        self.assertTrue(any(e['phase']=='reflection_response_rejected' for e in trace['events']))

    def test_compact_card_exposure_and_audit_have_real_issue_ids(self):
        r,c,selected,context,reply=self.strategy_fixture()
        context['task_reflection']=c.compact();context['actions']=[selected]
        c.offered(4,context);value=audit(context)
        self.assertEqual(value['reflection_offer']['issue_id'],reply['issue_id'])
        self.assertEqual(read_json(c.path)['exposures'][0]['request_number'],4)
        self.assertNotIn('facts',context['task_reflection']['pending'])

    def test_range_relation_never_turns_missing_reference_into_disjoint(self):
        a={'dim':'feret_max','min':11,'max':35}
        self.assertEqual(range_relation([a,None]),'not_comparable')
        self.assertEqual(range_relation([a]),'not_comparable')
        self.assertEqual(range_relation([a,dict(a,min=20,max=40)]),'overlap')
        self.assertEqual(range_relation([a,dict(a,min=40,max=50)]),'disjoint')
        self.assertEqual(range_relation([a,dict(a,dim='mean_width')]),'not_comparable')

    def test_shared_measurement_has_one_fact_and_distinct_candidate_reference_states(self):
        r=self.detected()
        r.a={'status':'ok','fold_rankings':[[{'label':'白芍','probability':.7},
                                          {'label':'牡丹皮','probability':.3}]]*5}
        scored=r.score([])
        # 显式评分输入替身，检验投影去重；不将fixture文件冒充真实簇晶测量。
        for candidate in scored['decision']['ranking']:
            if candidate['id'] in {'白芍','G1'}:
                for check in candidate['checks']:
                    if check['feature_id']=='cluster_crystal':
                        check.update(instance_refs=['fixture:one-instance'],stats={'median':17.2,'n_measured':1},
                                     state='NO_REF' if candidate['id']=='G1' else 'IN_RANGE')
        eligibility=r.decision_view({'observations':[]})['eligibility']
        comparison=build(r,scored,eligibility,r.legal_actions({'observations':[]}))
        rows=[c for q in comparison['questions'] if q['feature_id']=='cluster_crystal' for c in q['consumers']]
        by_id={c['candidate_id']:c for c in rows}
        self.assertIsNotNone(by_id['白芍']['fact_ref'])
        self.assertEqual(by_id['白芍']['fact_ref'],by_id['G1']['fact_ref'])
        self.assertEqual(len(comparison['physical_facts']),1)
        self.assertEqual(by_id['G1']['capability_gap'],'knowledge_reference')
        self.assertEqual(by_id['G1']['state'],'NO_REF')

    def test_compaction_parity_preserves_cohort_evidence_and_valid_menu(self):
        r=fair.FairInvestigationTests.prepare(self);state={'observations':[],'events':[],'attempted':[]}
        source={'evidence':compact_evidence(r.summary(state)),'actions':r.legal_actions(state)}
        original=copy.deepcopy(source);out,_,prof=compact(source)
        self.assertEqual(source,original)
        self.assertEqual(out['evidence']['candidate_interest'],source['evidence']['candidate_interest'])
        self.assertEqual(out['evidence']['workflow']['eligibility_projection'],
                         source['evidence']['workflow']['eligibility_projection'])
        self.assertEqual([a['action_id'] for a in out['actions']],[a['action_id'] for a in source['actions']])
        self.assertLess(prof['after']['bytes'],prof['before']['bytes'])
        model,aliases=model_view(out)
        for q in model['evidence']['workflow']['comparison_problems']['questions']:
            self.assertTrue(all(p['action_id'] in aliases for p in q['entry_paths']))

    def test_shared_notes_roundtrip_and_protected_inputs(self):
        text='可追溯的共同观察不能自动成为排他鉴别依据；'*8
        value={'one':{'interpretation':text},'two':{'interpretation':text},
               'actions':[{'args':{'scope':text}}],
               'evidence':{'eligibility_projection':{'scope':text}}}
        protected=copy.deepcopy(value['evidence'])
        deduplicate_notes(value)
        ref=value['one']['interpretation_note_ref']
        self.assertEqual(value['context_notes'][ref],text)
        self.assertEqual(value['two']['interpretation_note_ref'],ref)
        self.assertEqual(value['actions'][0]['args']['scope'],text)
        self.assertEqual(value['evidence'],protected)

    def test_observation_accepts_its_explicit_current_action_alias_without_guess(self):
        context={'execution_options':{'max_questions':2,'questions':[
            {'question_id':'Q1','action_id':'A11','canonical_question_id':'physical:one'}]}}
        result=decode({'decision':{'kind':'observe_questions','question_ids':['A11']}},context)
        self.assertEqual(result['action_id'],'A11')
        self.assertNotIn('parallel_question_id',result)

    def test_mixed_scan_and_observation_alias_preserves_only_explicit_legal_observation(self):
        context={'execution_options':{'max_questions':2,'questions':[
            {'question_id':'Q1','action_id':'A11','canonical_question_id':'physical:one'}]}}
        result=decode({'decision':{'kind':'observe_questions','question_ids':['A5','A11']}},context)
        self.assertEqual(result['action_id'],'A11')
        self.assertEqual(result['partial_choice_rejection']['rejected_questions'],1)
        self.assertNotIn('parallel_question_id',result)

    def test_unavailable_question_gets_one_budgeted_format_repair_without_tool_guess(self):
        r=self.detected(context_compact=True);bodies=[];before=list(r.workflow.backend.calls)
        def http(payload,timeout):
            bodies.append(payload)
            context=json.loads(payload['messages'][-1]['content'])
            choice=({'kind':'observe_questions','question_ids':['Q999']} if len(bodies)==1 else
                    {'kind':'action','action_id':next(a['action_id'] for a in context['actions']
                                                     if a['tool']=='request_review')})
            return {'choices':[{'message':{'content':json.dumps({'decision':choice})}}],
                    'usage':{'total_tokens':31+len(bodies)}}
        with patch('TCM_Fusion.lib.hard_calls.post',side_effect=http):
            trace=self.run_agent(r,LLMPlanner())
        self.assertEqual(trace['planning_requests'],2)
        self.assertEqual(trace['stop_reason'],'planner_request_review')
        errors=[e for e in trace['events'] if e['phase']=='planning_error']
        self.assertEqual(errors[0]['reason'],'PLANNER_QUESTION_UNAVAILABLE')
        self.assertTrue(errors[0]['retry_scheduled']);self.assertEqual(errors[0]['usage']['total_tokens'],32)
        self.assertEqual(r.workflow.backend.calls,before)
        self.assertIn('format_feedback',json.loads(bodies[-1]['messages'][-1]['content']))

    def test_identity_mismatch_reaches_next_planner_and_binds_model_choice(self):
        backend = fixtures.StageFixture(classification_feature='fiber')
        self.assert_identity_feedback(backend, 'measure_width', 'feature_mismatch', 'fiber',
                                      'SEMANTIC_BINDING_AMBIGUOUS')

    def test_required_crop_mismatch_blocks_measurement_with_explicit_feedback(self):
        backend = fixtures.StageFixture(classification_feature='fiber')
        self.assert_identity_feedback(backend, 'measure_width', 'feature_mismatch', 'fiber',
                                      'identity_rejected', identity_required=True)

    def test_unreadable_identity_returns_unknown_to_next_planner(self):
        class Unreadable(fixtures.StageFixture):
            def execute(self, session, tool, target, work, context):
                result = super().execute(session, tool, target, work, context)
                if tool == 'classify_target':
                    write_json(Path(result['classification']) / 'classification.json', {
                        'label': '图像不可判', 'feature_id': '', 'confidence': 'medium', 'reject': 'unreadable'})
                return result

        self.assert_identity_feedback(Unreadable(), 'verify_identity', 'unreadable', None, 'identity_rejected')

    def assert_identity_feedback(self, backend, semantic_task, outcome, reported_feature, return_reason,
                                 *, identity_required=False):
        runtime = self.detected(backend=backend, semantic_execution=True, context_compact=True)
        if identity_required:
            snapshot = runtime.workflow.store.current()
            crop = next(a for a in snapshot['artifacts'].values() if a['kind'] == 'crop')
            runtime.workflow.store.commit(snapshot['version'], [{**crop, 'identity_required': True}])
            runtime.a = {'status': 'ok', 'fold_rankings': [[{'label': '白芍', 'probability': 0.9},
                                                           {'label': '香附', 'probability': 0.1}]]}
        instance = options(runtime, runtime.workflow.propose())[0]
        contexts = []

        class Planner:
            def choose(inner, context):
                contexts.append(context)
                if len(contexts) == 1:
                    action = next(a for a in context['actions'] if a['tool'] == 'classify_target')
                    return {'action_id': action['action_id'], 'semantic_goal': {
                        'semantic_task': semantic_task, 'instance_ref': instance['instance_ref'],
                        'problem_id': 'fixture_identity_goal'}}, None
                return {'action_id': 'review', 'reflection_response': 'apply',
                        'decision_summary': '该实例身份没有通过，当前先复核'}, None

        trace = self.run_agent(runtime, Planner())
        self.assertEqual(trace['planning_requests'], 2)
        self.assertEqual(trace['stop_reason'], 'planner_request_review')
        self.assertEqual(backend.calls, ['detect_fov_initial', 'classify_target'])
        failure = contexts[1]['task_reflection']['pending']['identity_failure']
        self.assertEqual(failure, {'semantic_task': semantic_task, 'instance_ref': instance['instance_ref'],
                                  'outcome': outcome, 'expected_feature': instance['feature_id'],
                                  'reported_feature': reported_feature})
        ledger = read_json(runtime.workflow.root / 'reflection_controller.json')
        self.assertTrue(ledger['rows'][0]['accepted'])
        self.assertEqual(ledger['goal_failures'][0]['return_reason'], return_reason)
        self.assertEqual(ledger['exposures'][0]['request_number'], 2)
        self.assertEqual(ledger['responses'][0]['transition'], 'stop')
        self.assertTrue(ledger['applications'][-1]['expected_outcome_observed'])
        self.assertFalse(ledger['applications'][-1]['qualified_fact_added'])

    def test_short_apply_tracks_model_switch_to_a_real_followup_action(self):
        runtime, controller, failed, _ = self.controller()
        # 从原失败实例转查当前编排器已提供的分类动作，主选择仍由调用方决定。
        chosen = next(a for a in runtime.workflow.propose() if a['tool'] == 'classify_target')
        context = {'task_reflection': controller.compact(), 'actions': [chosen]}
        row = controller.respond('apply', chosen, context)
        self.assertEqual(row['transition'], 'switch_question')
        controller.record(chosen, progress=True, feedback={'accepted': True, 'capability_progress': True},
                          evidence_version='followup')
        posterior = controller.applications[-1]
        self.assertTrue(posterior['condition_observed'])
        self.assertFalse(posterior['qualified_fact_added'])
        self.assertFalse(posterior['causal_improvement_validated'])

    def test_short_apply_cannot_count_retry_of_unchanged_failed_input(self):
        _, controller, failed, _ = self.controller()
        with self.assertRaises(IntegrationError):
            controller.respond('apply', failed, {'task_reflection': controller.compact(), 'actions': [failed]})
        self.assertEqual(controller.applications, [])
