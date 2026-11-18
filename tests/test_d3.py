"""D3离线验收：真实LangGraph/工具守卫，视觉响应为显式fixture；不联网、不评价识别率。"""
import asyncio
import copy
import json
import unittest
from pathlib import Path

from TCM_Fusion.lib.agent_tools import LocalToolbox, FixtureBackend
from TCM_Fusion.lib.storage import IntegrationError, read_json, write_json, digest
from TCM_Fusion.lib.planning_context import compact
from TCM_Fusion.lib.react_graph import run_react, compact_evidence
from TCM_Fusion.lib.reflection_controller import ReflectionController
from TCM_Fusion.lib.semantic_execution import bind, compile_goal, next_action, options
from TCM_Fusion.tests import test_workflow as fixtures

StageFixture = fixtures.StageFixture


class D3StageFixture(StageFixture):
    """显式模拟几何返回宽度；原StageFixture仅有长度，用于缺量纲反例。"""
    def execute(self,s,tool,target,work,context):
        result=super().execute(s,tool,target,work,context)
        if tool=='measure_instances':
            p=Path(result['measurements']);row=json.loads(p.read_text(encoding='utf-8'))
            row.update(mean_width_um=5.5,area_um2=30)
            p.write_text(json.dumps(row)+'\n',encoding='utf-8')
        return result


class D3Tests(unittest.TestCase):
    setUp = fixtures.WorkflowTests.setUp
    runtime = fixtures.WorkflowTests.runtime

    def detected(self, backend=None, **config):
        r = self.runtime(backend=backend or D3StageFixture(), agent_policy='adaptive_v2', max_planning=20,
                         max_resource_units=200, **config)
        r.workflow.execute(next(a for a in r.workflow.propose() if a['tool']=='detect_fov_initial'))
        return r

    def state(self):
        return {'observations': [], 'attempted': [], 'events': []}

    def action(self, r, tool):
        return next(a for a in r.workflow.propose() if a['tool']==tool)

    def goal(self, r):
        option = options(r, r.workflow.propose())[0]
        return {'semantic_task':'measure_width','instance_ref':option['instance_ref'],'problem_id':'width_question'}

    def run_agent(self, r, planner, **opts):
        tools = LocalToolbox(r, ledger_path=self.root/'observations.json', backend=FixtureBackend())
        return asyncio.run(run_react(r, tools, planner=planner, planner_mode='scripted',
                                     max_steps=opts.pop('max_steps',12), **opts))

    def test_compaction_preserves_all_fold_candidates_actions_and_qualification(self):
        r = self.detected()
        ids = [row['id'] for row in r.kb.candidates()][:15]
        r.a={'status':'ok','fold_rankings':[[{'label':cid,'probability':.1} for cid in ids[i:i+3]] for i in range(0,15,3)]}
        evidence=compact_evidence(r.summary(self.state())); context={'evidence':evidence,'actions':r.legal_actions(self.state())}
        original=copy.deepcopy(context); out,cursor,audit=compact(context)
        self.assertEqual(context,original)
        self.assertEqual(out['evidence']['candidate_interest']['fold_top_k'],evidence['candidate_interest']['fold_top_k'])
        self.assertEqual(out['evidence']['candidate_interest']['ids'],evidence['candidate_interest']['ids'])
        self.assertEqual([a['action_id'] for a in out['actions']],[a['action_id'] for a in context['actions']])
        self.assertEqual(out['evidence']['workflow']['eligibility_projection'],evidence['workflow']['eligibility_projection'])
        self.assertGreater(len(out['evidence']['hypothesis_catalog_index']['ids']),6)
        self.assertLess(audit['after']['bytes'],audit['before']['bytes'])
        _,_,second=compact(context,cursor)
        self.assertEqual(second['after']['unit'],'utf8_json_bytes_not_tokens')

    def test_explicit_current_detail_read_is_not_silently_removed(self):
        full={'rows':[{'reason':'x'*9000}], 'kind':'provisional'}
        context={'evidence':{},'actions':[],'read_results':[{'stale':False,'result':full}]}
        out,_,_=compact(context)
        self.assertEqual(out['read_results'][0]['result'],full)
        context['read_results'][0]['stale']=True
        out,_,_=compact(context)
        self.assertTrue(out['read_results'][0]['result']['summary_only'])

    def test_catalog_provisional_and_observation_detail_readers(self):
        r=self.detected();state=self.state()
        r.workflow.execute(self.action(r,'classify_target'))
        for kind in ['candidate_catalog','provisional','observations','operations','classifications']:
            result=r.tools.read(r,'read_evidence_detail',{'kind':kind,'offset':0,'limit':2},state)
            self.assertIn('total',result);self.assertLessEqual(len(result['rows']),2)

    def controller(self):
        r=self.detected();c=ReflectionController(r)
        action=r.action('read_scale',{},'evidence_inspection')
        for _ in range(2): c.record(action,progress=False,feedback={'problem_id':'same_question','accepted':True},
                                  evidence_version=r.workflow.store.current()['version'])
        card=c.propose({'no_gain':2})
        return r,c,action,card

    def patch(self,card,**extra):
        return {'reflection_id':card['reflection_id'],'change_condition':'different_method',
                'avoid_action_keys':card['proposed_patch']['avoid_action_keys'],
                'activate_candidates':[],'recommended_exit':'continue',**extra}

    def test_reflection_has_patch_and_verifiable_posterior_without_evidence_write(self):
        r,c,action,card=self.controller(); before=r.workflow.store.current()['version']
        c.apply(self.patch(card));self.assertFalse(c.guard(action))
        other=r.action('list_fovs',{},'evidence_inspection');self.assertTrue(c.guard(other))
        c.record(other,progress=True,feedback={'problem_id':'new_question','accepted':True},evidence_version=before)
        posterior=read_json(c.path)['applications'][-1]
        self.assertTrue(posterior['condition_observed']);self.assertTrue(posterior['effective'])
        self.assertFalse(posterior['causal_improvement_validated'])
        self.assertEqual(r.workflow.store.current()['version'],before)

    def test_reflection_cannot_change_quality_or_ban_unrelated_keys(self):
        r,c,action,card=self.controller()
        for change in [{'quality':'valid'},{'avoid_action_keys':['invented']},{'activate_candidates':['not_in_kb']}]:
            with self.subTest(change=change),self.assertRaises(IntegrationError):c.apply({**self.patch(card),**change})
        self.assertIsNone(c.active)

    def test_reflection_stale_patch_and_current_action_conflict_rejected(self):
        r,c,action,card=self.controller()
        for patch in [self.patch(card,reflection_id='old'),self.patch(card)]:
            with self.assertRaises(IntegrationError):c.apply(patch,selected=action)
        self.assertIsNone(c.active)

    def test_reflection_checks_promised_change_not_just_progress(self):
        r,c,action,card=self.controller()
        c.apply(self.patch(card,change_condition='different_view'))
        c.record(r.action('list_fovs',{},'evidence_inspection'),progress=True,
                 feedback={'problem_id':'new','accepted':True},evidence_version=r.workflow.store.current()['version'])
        self.assertFalse(c.applications[-1]['condition_observed'])
        self.assertFalse(c.applications[-1]['effective'])

    def test_stop_patch_gets_posterior_without_another_tool_call(self):
        r,c,action,card=self.controller();c.apply(self.patch(card,change_condition='stop',recommended_exit='review'))
        c.finalize('planner_request_review');self.assertTrue(c.applications[-1]['condition_observed'])

    def test_long_flat_candidate_gap_is_advice_not_false_negative_or_hard_stop(self):
        r=self.detected();c=ReflectionController(r);decision=r.score([])['decision']
        for i in range(4):
            c.record(r.action('read_evidence',{'offset':i,'limit':1},'evidence_inspection'),
                     progress=True,feedback={'problem_id':'q'+str(i),'accepted':True},
                     evidence_version=r.workflow.store.current()['version'],candidate_state=decision)
        state={'no_gain':0};card=c.propose(state)
        self.assertEqual(card['failure_type'],'candidate_stagnation')
        self.assertEqual(card['proposed_patch']['avoid_action_keys'],[])
        self.assertEqual(state,{'no_gain':0});self.assertEqual(card['evidence_effect'],'none')

    def test_graph_reflection_is_merged_into_next_normal_planning_call(self):
        r=self.detected();seen=[]
        class Planner:
            def choose(inner,ctx):
                seen.append(ctx)
                if len(seen)==4:return {'action_id':'review'},None
                tool=['read_scale','list_fovs','list_features'][len(seen)-1]
                action=next(a for a in ctx['actions'] if a['tool']==tool)
                reply={'action_id':action['action_id']}
                card=ctx['task_reflection']['pending']
                if card:reply['reflection_patch']=self.patch(card)
                return reply,None
        trace=self.run_agent(r,Planner())
        self.assertEqual(trace['planning_requests'],4)
        self.assertTrue(any(e['phase']=='reflection_patch_applied' for e in trace['events']))
        self.assertTrue(read_json(r.workflow.root/'reflection_controller.json')['applications'])

    def test_invalid_reflection_does_not_reject_valid_stop(self):
        r=self.detected()
        class Planner:
            def choose(inner,c):return {'action_id':'review','reflection_patch':{'quality':'valid'}},None
        trace=self.run_agent(r,Planner());self.assertEqual(trace['planning_requests'],1)
        self.assertEqual(trace['stop_reason'],'planner_request_review')
        self.assertTrue(any(e['phase']=='reflection_patch_rejected' for e in trace['events']))

    def test_serial_goal_default_off_and_selected_action_is_preserved(self):
        r=self.detected();goal=self.goal(r);count=[]
        class Planner:
            def choose(inner,c):
                count.append(1)
                if len(count)>1:return {'action_id':'review'},None
                a=next(a for a in c['actions'] if a['tool']=='classify_target')
                return {'action_id':a['action_id'],'semantic_goal':goal},None
        trace=self.run_agent(r,Planner())
        self.assertNotIn('segment_instance',r.workflow.backend.calls)
        self.assertTrue(any(e['phase']=='semantic_goal_rejected' for e in trace['events']))

    def test_serial_goal_late_binds_full_chain_then_returns_to_react(self):
        r=self.detected(semantic_execution=True,auto_measure_after_segment=False);goal=self.goal(r);seen=[]
        class Planner:
            def choose(inner,c):
                seen.append(c)
                if len(seen)>1:return {'action_id':'review'},None
                a=next(a for a in c['actions'] if a['tool']=='classify_target')
                return {'action_id':a['action_id'],'semantic_goal':goal},None
        trace=self.run_agent(r,Planner())
        self.assertEqual(r.workflow.backend.calls,['detect_fov_initial','classify_target','segment_instance','measure_instances'])
        self.assertEqual(trace['planning_requests'],2)
        plan=read_json(r.workflow.root/'semantic_plan.json')
        self.assertEqual(plan['status'],'completed')
        self.assertTrue(all(n['status']=='succeeded' for n in plan['nodes']))
        self.assertTrue(all('target_id' not in n and 'tool' not in n for n in plan['nodes']))
        self.assertTrue(all(n['required_facts'] for n in plan['nodes']))
        self.assertTrue(any(e['phase']=='semantic_node_bound' for e in trace['events']))

    def test_serial_identity_rejection_returns_before_segmentation(self):
        r=self.detected(backend=StageFixture(classification_feature='fiber'),semantic_execution=True)
        goal=self.goal(r);seen=[]
        class Planner:
            def choose(inner,c):
                seen.append(1)
                if len(seen)>1:return {'action_id':'review'},None
                a=next(a for a in c['actions'] if a['tool']=='classify_target')
                return {'action_id':a['action_id'],'semantic_goal':goal},None
        trace=self.run_agent(r,Planner());self.assertNotIn('segment_instance',r.workflow.backend.calls)
        self.assertTrue(any(e['phase']=='semantic_return' for e in trace['events']))

    def test_semantic_contract_rejects_free_tool_sequence_and_future_targets(self):
        r=self.detected();a=self.action(r,'classify_target');goal=self.goal(r)
        for extra in [{'tool':'segment_instance'},{'target_id':'future_seg'},{'semantic_task':'arbitrary_python'}]:
            with self.subTest(extra=extra),self.assertRaises(IntegrationError):compile_goal(r,{**goal,**extra},a)

    def test_binding_and_use_revalidated_after_mask_before_measurement(self):
        r=self.detected();goal=self.goal(r);plan=compile_goal(r,goal,self.action(r,'classify_target'))
        r.workflow.execute(self.action(r,'classify_target'));a,reason=next_action(r,plan,self.state())
        self.assertEqual(a['tool'],'segment_instance');r.workflow.execute(a)
        a,reason=next_action(r,plan,self.state());self.assertEqual(a['tool'],'measure_instances')
        self.assertEqual(a['args']['evidence_version'],r.workflow.store.current()['version'])
        snap=r.workflow.store.current();image=next(iter(r.images))
        scale=snap['artifacts']['scale:'+image]
        path=self.root/'changed_scale.json';write_json(path,{'status':'OK','px_per_um':4})
        replacement=r.workflow._art(scale['id'],'scale',path,scale['parents'],image_id=image,px_per_um=4)
        r.workflow.store.commit(snap['version'],[replacement])
        a,reason=next_action(r,plan,self.state());self.assertIsNone(a)
        self.assertEqual(reason,'qualification_revoked')

    def test_semantic_input_change_and_task_close_stop_successors(self):
        r=self.detected();goal=self.goal(r);plan=compile_goal(r,goal,self.action(r,'classify_target'))
        crop,_=bind(r,goal['instance_ref']);import contracts
        Path(contracts.CropMeta.from_json(crop['path']).png).write_bytes(b'changed-pixels')
        a,reason=next_action(r,plan,self.state());self.assertIsNone(a);self.assertEqual(reason,'bound_input_changed')
        r.workflow.close('fixture_close');a,reason=next_action(r,plan,self.state());self.assertEqual(reason,'task_closed')

    def test_semantic_step_cap_prevents_followup_calls(self):
        r=self.detected(semantic_execution=True);goal=self.goal(r)
        class Planner:
            def choose(inner,c):
                a=next(a for a in c['actions'] if a['tool']=='classify_target')
                return {'action_id':a['action_id'],'semantic_goal':goal},None
        trace=self.run_agent(r,Planner(),max_steps=1)
        self.assertEqual(trace['stop_reason'],'max_steps');self.assertNotIn('segment_instance',r.workflow.backend.calls)

    def test_semantic_review_mask_returns_before_measurement(self):
        class Review(StageFixture):
            def execute(inner,s,tool,target,work,context):
                result=super().execute(s,tool,target,work,context)
                if tool=='segment_instance':
                    result['quality']='review'
                    write_json(Path(result['segmentation'])/'instances.json',{'instances':[{'igm_status':'review'}]})
                return result
        r=self.detected(backend=Review(),semantic_execution=True);goal=self.goal(r);seen=[]
        class Planner:
            def choose(inner,c):
                seen.append(1)
                if len(seen)>1:return {'action_id':'review'},None
                a=next(a for a in c['actions'] if a['tool']=='classify_target')
                return {'action_id':a['action_id'],'semantic_goal':goal},None
        trace=self.run_agent(r,Planner());self.assertNotIn('measure_instances',r.workflow.backend.calls)
        self.assertIn('mask_review',[e.get('reason') for e in trace['events'] if e['phase']=='semantic_return'])

    def test_graph_context_profile_records_bytes_not_estimated_tokens(self):
        r=self.detected(context_compact=True);seen=[]
        class Planner:
            def choose(inner,c):seen.append(c);return {'action_id':'review'},None
        trace=self.run_agent(r,Planner())
        event=next(e for e in trace['events'] if e['phase']=='context_profile')
        self.assertEqual(event['after']['unit'],'utf8_json_bytes_not_tokens')
        self.assertIn('round_delta',seen[0]);self.assertNotIn('semantic_options',seen[0])
        self.assertEqual(trace['planning_requests'],1)

    def test_unrelated_commit_does_not_invalidate_semantic_goal(self):
        r=self.detected();goal=self.goal(r);plan=compile_goal(r,goal,self.action(r,'classify_target'))
        snap=r.workflow.store.current();p=self.root/'note.json';write_json(p,{'unrelated':True})
        r.workflow.store.commit(snap['version'],[r.workflow._art('note','diagnostic',p,[next(iter(r.images))])])
        action,reason=next_action(r,plan,self.state())
        self.assertEqual(reason,'action_ready');self.assertEqual(action['tool'],'classify_target')
        self.assertEqual(action['args']['evidence_version'],r.workflow.store.current()['version'])

    def test_cancelled_semantic_goal_has_no_next_action(self):
        r=self.detected();plan=compile_goal(r,self.goal(r),self.action(r,'classify_target'))
        r.workflow.cancel_event.set();action,reason=next_action(r,plan,self.state())
        self.assertIsNone(action);self.assertEqual(reason,'task_cancelled')

    def test_geometry_mapping_change_is_not_mistaken_for_same_input(self):
        r=self.detected();goal=self.goal(r);plan=compile_goal(r,goal,self.action(r,'classify_target'))
        crop,_=bind(r,goal['instance_ref']);side=read_json(crop['path'])
        side['tight_box_local']=[1,0,7,8];write_json(crop['path'],side)
        action,reason=next_action(r,plan,self.state())
        self.assertIsNone(action);self.assertEqual(reason,'bound_input_changed')

    def test_parallel_success_does_not_hide_other_tool_failure_in_reflection(self):
        r=self.runtime(agent_policy='adaptive_v2',max_parallel=2,max_planning=12,max_resource_units=200)
        r.a={'status':'ok','ranking':[{'label':'广藿香','probability':.6},{'label':'淫羊藿','probability':.4}]}
        seen=[]
        class FailingPresence(FixtureBackend):
            def call(inner,runtime,action,path):
                raise IntegrationError('UPSTREAM_HTTP_404','controlled fixture failure')
        class Planner:
            def choose(inner,c):
                seen.append(1)
                if len(seen)>1:return {'action_id':'review'},None
                a=next(a for a in c['actions'] if a['tool']=='detect_fov_initial')
                b=next(a for a in c['parallel_observations'] if a['tool']=='presence_check')
                return {'action_id':a['action_id'],'parallel_observation_id':b['action_id']},None
        tools=LocalToolbox(r,ledger_path=self.root/'parallel_ledger.json',backend=FailingPresence())
        trace=asyncio.run(run_react(r,tools,planner=Planner(),planner_mode='scripted'))
        rows=read_json(r.workflow.root/'reflection_controller.json')['rows']
        self.assertEqual(len(rows),2)
        self.assertTrue(next(row for row in rows if row['tool']=='detect_fov_initial')['progress'])
        failed=next(row for row in rows if row['tool']=='presence_check')
        self.assertFalse(failed['progress']);self.assertFalse(failed['accepted'])
        self.assertEqual(trace['planning_requests'],2)

    def test_measurement_file_without_requested_dimension_is_not_goal_success(self):
        r=self.detected(backend=StageFixture(),semantic_execution=True);goal=self.goal(r);seen=[]
        class Planner:
            def choose(inner,c):
                seen.append(1)
                if len(seen)>1:return {'action_id':'review'},None
                a=next(a for a in c['actions'] if a['tool']=='classify_target')
                return {'action_id':a['action_id'],'semantic_goal':goal},None
        trace=self.run_agent(r,Planner())
        self.assertIn('requested_dimension_unavailable',[e.get('reason') for e in trace['events']
                                                        if e['phase']=='semantic_return'])
        self.assertEqual(r.workflow.backend.calls.count('measure_instances'),1)
        self.assertNotIn('mean_width_um',r.workflow.store.verified_records()[0])


    def test_reflection_rejection_is_audited_and_available_to_next_context(self):
        r,c,action,card=self.controller();bad=self.patch(card,avoid_action_keys=['A1_PRIVATE_UNKNOWN'])
        with self.assertRaises(IntegrationError) as caught:c.apply(bad)
        diagnosis=c.reject(bad,caught.exception)
        self.assertEqual(diagnosis['unknown_action_keys_count'],1)
        self.assertEqual(diagnosis['extra_planning_calls'],0)
        self.assertTrue(diagnosis['matches_current_reflection']);self.assertIsNone(c.active)
        saved=read_json(c.path);self.assertEqual(saved['rejections'],[diagnosis])
        self.assertNotIn('A1_PRIVATE_UNKNOWN',json.dumps(saved))
        self.assertEqual(c.compact()['last_rejection'],[diagnosis])
        self.assertEqual(card['proposed_patch']['reflection_id'],card['reflection_id'])
        self.assertIn('change_condition',card['proposed_patch'])

    def test_automatic_measurement_failure_enters_reflection_separately_from_mask_success(self):
        r=self.detected(backend=StageFixture(fail='measure_instances'),auto_measure_after_segment=True)
        class Planner:
            calls=0
            def choose(inner,ctx):
                inner.calls+=1
                if inner.calls>2:return {'action_id':'review'},None
                tool='classify_target' if inner.calls==1 else 'segment_instance'
                action=next(a for a in ctx['actions'] if a['tool']==tool)
                return {'action_id':action['action_id']},None
        trace=self.run_agent(r,Planner())
        rows=read_json(r.workflow.root/'reflection_controller.json')['rows']
        failed=next(row for row in rows if row['tool']=='measure_instances')
        self.assertFalse(failed['progress']);self.assertEqual(failed['failure_code'],'SIMULATED_FAILURE')
        self.assertTrue(next(row for row in rows if row['tool']=='segment_instance')['accepted'])
        self.assertTrue(any(e['phase']=='automatic_measurement_feedback' for e in trace['events']))


    def test_reflection_schema_diagnostics_keep_field_path_without_model_values(self):
        r,c,action,card=self.controller()
        for kind in ['enum','missing']:
            bad=self.patch(card)
            if kind=='enum':bad['change_condition']='PRIVATE_INVALID_CONDITION'
            else:bad.pop('recommended_exit')
            with self.assertRaises(IntegrationError) as caught:c.apply(bad)
            row=c.reject(bad,caught.exception)
            if kind=='enum':
                self.assertEqual(row['schema_error']['validator'],'enum')
                self.assertEqual(row['schema_error']['field_path'],['change_condition'])
            else:
                self.assertIn('recommended_exit',row['schema_error']['missing_fields'])
            self.assertNotIn('PRIVATE_INVALID_CONDITION',json.dumps(row))
