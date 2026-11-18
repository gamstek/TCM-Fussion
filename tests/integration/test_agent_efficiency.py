"""Agent效率回归：语义问题、分轴反馈、上下文/计时和模型授权并发；无真实API。

由现有selftest自动发现。使用显式可控后端；网络失败不当阴性，记录不等于质量提升。
该组用例由项目根 selftest 发现。涉及模型回复的用例使用可控输入，涉及进程的用例以各断言规定的执行边界为准。

主要定义：
- EfficiencyTests。

关联：
- 直接依赖 「lib/agent/investigation_control.py」、「lib/agent/planning_context.py」、「lib/agent/react_graph.py」、「lib/agent/reflection_controller.py」、「lib/agent/semantic_execution.py」、「lib/decision/comparison_problems.py」、「lib/mcp/mcp_bridge.py」、「lib/runtime/controlled_process.py」。
- 统一执行见 tests/README.md。
"""
import asyncio
import copy
import json
import sys
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from TCM_Fusion.tests.support import workflow_fixtures as fixtures
from TCM_Fusion.lib.runtime.storage import read_json, write_json, IntegrationError
from TCM_Fusion.lib.decision.comparison_problems import action_question
from TCM_Fusion.lib.agent.investigation_control import progress_state, progress_axes
from TCM_Fusion.lib.agent.planning_context import compact
from TCM_Fusion.lib.agent.reflection_controller import ReflectionController
from TCM_Fusion.lib.tools.agent_tools import LocalToolbox, FixtureBackend
from TCM_Fusion.lib.agent.react_graph import run_react, compact_evidence


class EfficiencyTests(unittest.TestCase):
    setUp=fixtures.WorkflowFixture.setUp
    runtime=fixtures.WorkflowFixture.runtime

    def detected(self, backend=None, **config):
        r=self.runtime(backend=backend or fixtures.StageFixture(),agent_policy='adaptive_v2',**config)
        r.workflow.execute(next(a for a in r.workflow.propose() if a['tool']=='detect_fov_initial'))
        return r

    def test_scan_to_classification_keeps_same_question(self):
        r=self.detected();targets=r.workflow.targets();crop=next(v for v in targets.values() if v['kind']=='crop')
        scan={'id':'scan','feature_id':'single_crystal','image_id':crop['image_id']}
        with patch.object(r.workflow,'targets',return_value={**targets,'scan':scan}):
            q1=action_question(r,{'tool':'feature_scan','args':{'target_id':'scan'}})
            q2=action_question(r,{'tool':'classify_target','args':{'target_id':crop['id']}})
        self.assertEqual(q1['question_id'],q2['question_id'])
        other=action_question(r,{'tool':'presence_check','args':{'image_id':crop['image_id'],'feature_id':'fiber'}})
        self.assertNotEqual(q1['question_id'],other['question_id'])

    def test_new_crop_is_capability_without_formal_evidence(self):
        r=self.runtime(agent_policy='adaptive_v2');before=progress_state(r,r.score([]))
        r.workflow.execute(next(a for a in r.workflow.propose() if a['tool']=='detect_fov_initial'))
        axes=progress_axes(before,progress_state(r,r.score([])))
        self.assertTrue(axes['capability']);self.assertFalse(axes['evidence'])
        self.assertEqual(r.workflow.store.verified_records(),[])

    def test_unreadable_classification_is_diagnostic_not_negative(self):
        class Unreadable(fixtures.StageFixture):
            def execute(inner,s,tool,target,work,context):
                result=super().execute(s,tool,target,work,context)
                if tool=='classify_target':
                    write_json(Path(result['classification'])/'classification.json',
                        {'label':'图像不可判','feature_id':'','confidence':'high','reject':'unreadable',
                         'reason':'fixture','granularity':'instance','source':'vlm'})
                return result
        r=self.detected(backend=Unreadable());before=progress_state(r,r.score([]))
        result=r.workflow.execute(next(a for a in r.workflow.propose() if a['tool']=='classify_target'))
        self.assertTrue(result['accepted'])
        axes=progress_axes(before,progress_state(r,r.score([])))
        self.assertTrue(axes['diagnostic']);self.assertFalse(axes['evidence'])
        self.assertEqual(r.score([])['observation_states'],[])

    def reflection(self):
        r=self.detected();c=ReflectionController(r);action=r.action('read_scale',{},'evidence_inspection')
        for _ in range(2):c.record(action,progress=False,feedback={'problem_id':'old','accepted':True},evidence_version=r.workflow.store.current()['version'])
        card=c.propose({'no_gain':2})
        value=copy.deepcopy(card['proposed_patch']);value['recommended_exit']='continue'
        return r,c,action,value

    def test_noop_activation_and_already_unavailable_key_are_not_policy_gain(self):
        r,c,a,value=self.reflection();value['activate_candidates']=['G3']
        with patch.object(r,'candidate_interest',return_value={'ids':['G3'],'hypotheses':[{'candidate_id':'G3','status':'investigating'}]}):
            c.apply(value,legal_actions=[])
        self.assertFalse(c.applications[-1]['policy_delta']['changes_current_menu_or_scope'])

    def test_recommended_exit_is_separate_from_evidence_gain(self):
        r,c,a,value=self.reflection();value.update(change_condition='switch_question',recommended_exit='review')
        c.apply(value,legal_actions=[]);c.finalize('planner_request_review');row=c.applications[-1]
        self.assertTrue(row['exit_recommendation_followed']);self.assertFalse(row['declared_exit_condition_consistent'])
        self.assertFalse(row['evidence_added']);self.assertFalse(row['effective'])

    def test_context_deduplicates_state_without_mutating_inputs_or_candidates(self):
        r=self.detected();r.a={'status':'ok','fold_rankings':[[{'label':'白芍','probability':1.}]]*5}
        evidence=compact_evidence(r.summary({'observations':[]}));gap=evidence['workflow']['supporting_gaps']
        ctx={'evidence':evidence,'actions':r.legal_actions({'observations':[],'attempted':[]}), 'supporting_gaps':copy.deepcopy(gap)}
        before=copy.deepcopy(ctx);out,cursor,audit=compact(ctx)
        self.assertEqual(ctx,before);self.assertNotIn('supporting_gaps',out)
        self.assertEqual(out['evidence']['candidate_interest'],before['evidence']['candidate_interest'])
        self.assertIn('criterion_states',out['evidence']['workflow'])
        self.assertLess(audit['after']['bytes'],audit['before']['bytes'])
        self.assertIn('白芍',out['evidence']['candidate_window']['ids'])
        different=copy.deepcopy(ctx);different['supporting_gaps']=[{'unique':'different scope'}]
        self.assertEqual(compact(different)[0]['supporting_gaps'],different['supporting_gaps'])

    def run_agent(self,r,planner,backend=None,**opts):
        toolbox=LocalToolbox(r,ledger_path=self.root/'observations.json',backend=backend or FixtureBackend())
        return asyncio.run(run_react(r,toolbox,planner=planner,planner_mode='scripted',**opts))

    def test_bad_optional_hypotheses_keep_legal_action_without_replanning(self):
        r=self.runtime(agent_policy='adaptive_v2');calls=[]
        class Planner:
            def choose(inner,context):
                calls.append(context)
                if len(calls)>1:return {'action_id':'review'},None
                action=next(a for a in context['actions'] if a['tool']=='detect_fov_initial')
                return {'action_id':action['action_id'],'hypothesis_updates':[{'bad':'fixture'}]},None
        trace=self.run_agent(r,Planner())
        self.assertEqual(len(calls),2);self.assertEqual(r.workflow.backend.calls,['detect_fov_initial'])
        self.assertTrue(any(e['phase']=='hypothesis_update_rejected' and e['selected_action_preserved'] for e in trace['events']))
        self.assertFalse(any(e['phase']=='rejected_proposal' for e in trace['events']))

    def test_planning_and_execution_have_monotonic_elapsed_measurements(self):
        r=self.runtime(agent_policy='adaptive_v2');count=[]
        class Planner:
            def choose(inner,c):
                count.append(1)
                if len(count)>1:return {'action_id':'review'},None
                a=next(a for a in c['actions'] if a['tool']=='detect_fov_initial')
                return {'action_id':a['action_id']},None
        trace=self.run_agent(r,Planner())
        spans=[e for e in trace['events'] if e['phase']=='planning_timing']
        self.assertEqual(len(spans),2);self.assertTrue(all(e['elapsed_seconds']>=0 for e in spans))
        self.assertEqual(len([e for e in trace['events'] if e['phase']=='execution_timing']),1)
        self.assertTrue(all('started_at' in e and 'ended_at' in e for e in spans))

    def test_semantic_goal_options_explain_identity_vs_physical_quantity(self):
        from TCM_Fusion.lib.agent.semantic_execution import GOAL_CONTRACT
        self.assertEqual(GOAL_CONTRACT['verify_identity']['may_require'],['classify_target'])
        self.assertIn('measure_instances',GOAL_CONTRACT['measure_length']['may_require'])
        r=self.detected(semantic_execution=True,context_compact=True)
        class Planner:
            def choose(inner,c):
                self.assertIn('semantic_goal_contract',c)
                return {'action_id':'review'},None
        trace=self.run_agent(r,Planner())
        self.assertNotIn('segment_instance',r.workflow.backend.calls)

    def paired(self, *, budget=4, fail=False, invalid=False):
        r=self.runtime(agent_policy='adaptive_v2',max_parallel=2,max_observations=budget,max_api_calls=30,max_resource_units=100)
        r.a={'status':'ok','fold_rankings':[[{'label':'秦皮','probability':.4},{'label':'茯苓','probability':.3},{'label':'郁金','probability':.3}]]*5}
        entered=[];barrier=threading.Barrier(2);calls=[];active=[0];lock=threading.Lock();seen=[]
        class Backend:
            def call(inner,runtime,action,path):
                with lock:active[0]+=1;calls.append(action['args']['feature_id'])
                try:
                    if budget>=2 and not invalid:barrier.wait(timeout=5)
                    if fail and action['args']['feature_id']==entered[0]:raise IntegrationError('UPSTREAM_REMOTE_DISCONNECTED','fixture')
                    return {'ok':True,'present':False,'conf':.8},None
                finally:
                    with lock:active[0]-=1
        class Planner:
            def choose(inner,c):
                seen.append(c)
                if len(seen)>1:return {'action_id':'review'},None
                actions=[a for a in c['actions'] if a['tool']=='presence_check']
                self.assertGreaterEqual(len(actions),2)
                first=actions[0];entered.append(first['args']['feature_id'])
                other=next(q for q in c['parallel_questions'] if q['feature_id']!=entered[0])
                return {'action_id':first['action_id'],'parallel_question_id':('unknown' if invalid else other['question_id'])},None
        commits=[]
        def confirm(action,result):
            self.assertEqual(active[0],0);commits.append(action['action_id'])
        with patch.object(r.workflow,'confirm_identity',side_effect=confirm):
            trace=self.run_agent(r,Planner(),Backend())
        return r,trace,calls,commits

    def test_model_authorized_questions_overlap_and_parent_commits_results(self):
        r,t,calls,commits=self.paired()
        self.assertEqual(len(calls),2);self.assertEqual(len(commits),2)
        self.assertEqual(t['observation_requests'],2);self.assertEqual(t['planning_requests'],2)
        self.assertTrue(all(o['accepted'] for o in t['observations']))
        self.assertTrue(any(e['phase']=='parallel_join' and e['policy']=='independent_semantic_questions' for e in t['events']))
        self.assertEqual(read_json(r.workflow.budget_path)['used']['observation'],2)

    def test_one_failed_question_does_not_drop_successful_sibling(self):
        r,t,calls,commits=self.paired(fail=True)
        self.assertEqual(len(t['observations']),2);self.assertEqual(sum(o['accepted'] for o in t['observations']),1)
        failed=next(o for o in t['observations'] if not o['accepted'])
        self.assertIsNone(failed['result'].get('present'));self.assertEqual(len(commits),1)

    def test_unaffordable_or_unknown_parallel_intent_preserves_primary(self):
        for budget,invalid in [(1,False),(4,True)]:
            with self.subTest(budget=budget,invalid=invalid):
                # 每个子例独立任务/账本。
                if invalid:self.root=self.root/'second';self.root.mkdir();self.image=self.root/'fov.png';self.image.write_bytes(b'fixture')
                r,t,calls,commits=self.paired(budget=budget,invalid=invalid)
                self.assertEqual(len(calls),1);self.assertEqual(t['observation_requests'],1)
                self.assertTrue(any(e['phase']=='parallel_question_rejected' for e in t['events']))

    def test_local_worker_timing_does_not_require_external_api(self):
        from TCM_Fusion.lib.runtime.controlled_process import run
        r=self.detected();work=self.root/'worker';work.mkdir();ctx=r.workflow.worker_context(work,'timing')
        run([sys.executable,'-c','print("fixture")'],context_path=ctx,timeout=5,log_path=work/'log.txt')
        span=read_json(work/'process_timing.json')
        self.assertGreaterEqual(span['elapsed_seconds'],0);self.assertEqual(span['returncode'],0)

    def test_http_worker_exports_phase_durations_without_credentials(self):
        from TCM_Fusion.lib.runtime.hard_calls import worker
        from adjudicator import AdjudicatorClient
        inp=self.root/'request.json';out=self.root/'response.json'
        write_json(inp,{'payload':{'model':'fixture'},'timeout':1})
        def init(client,*args,**kwargs):client.timeout_s=1
        with patch.object(AdjudicatorClient,'__init__',init),patch.object(AdjudicatorClient,'_post',return_value={'ok':True}):worker(inp,out)
        value=read_json(out);self.assertGreaterEqual(value['http_timing']['send_to_completion_seconds'],0)
        self.assertIsNone(value['http_timing']['send_to_headers_seconds'])


    def test_stage_transition_does_not_satisfy_reflection_switch_question(self):
        r=self.detected();targets=r.workflow.targets();crop=next(v for v in targets.values() if v['kind']=='crop')
        scan={'id':'scan','feature_id':'single_crystal','image_id':crop['image_id']}
        first={'tool':'feature_scan','args':{'target_id':'scan'}}
        second={'tool':'classify_target','args':{'target_id':crop['id']}}
        c=ReflectionController(r)
        with patch.object(r.workflow,'targets',return_value={**targets,'scan':scan}):
            for _ in range(2):c.record(first,progress=False,feedback={'problem_id':'scan','accepted':True},evidence_version=r.workflow.store.current()['version'])
            card=c.propose({'no_gain':2});value=copy.deepcopy(card['proposed_patch']);value['change_condition']='switch_question'
            c.apply(value,selected=second,legal_actions=[second])
            c.record(second,progress=True,feedback={'problem_id':'crop','accepted':True,'progress_axes':{'evidence':False,'capability':False,'diagnostic':True}},evidence_version=r.workflow.store.current()['version'])
        last=c.applications[-1];self.assertFalse(last['condition_observed']);self.assertFalse(last['evidence_added'])
        self.assertTrue(last['progress_axes']['diagnostic'])

    def test_equivalent_crop_alias_is_not_new_capability(self):
        from TCM_Fusion.lib.runtime.storage import file_hash
        r=self.detected();before=progress_state(r,r.score([]));snap=r.workflow.store.current()
        crop=next(v for v in snap['artifacts'].values() if v['kind']=='crop');path=Path(crop['path']).with_name('alias.json')
        data=read_json(crop['path']);data['crop_id']='alias';write_json(path,data)
        alias={k:v for k,v in crop.items() if k!='valid'};alias.update(id='crop:alias',path=str(path),sha256=file_hash(path))
        r.workflow.store.commit(snap['version'],[alias]);axes=progress_axes(before,progress_state(r,r.score([])))
        self.assertFalse(any(axes.values()))

    def test_real_stdio_parallel_requests_cannot_bypass_network_permission(self):
        from TCM_Fusion.lib.workflow.workflow_runtime import from_task
        from TCM_Fusion.lib.mcp.mcp_bridge import connect_stdio
        config={'sample_id':'pair_protocol','allow_network':False,'b':{'input_mode':'images'},
                'agent':{'image_roots':[str(self.image)],'workflow':{'python':sys.executable,
                'execution':'active','agent_policy':'adaptive_v2','max_seconds':30}}}
        directory=self.root/'mcp_pair';directory.mkdir();a={'status':'disabled','ranking':[]};write_json(directory/'a.json',a)
        runtime=from_task(config,a,directory)
        runtime.investigate(['秦皮','茯苓','郁金'],'explicit offline protocol audit',runtime.score([])['evidence_digest'])
        actions=[x for x in runtime.legal_actions({'observations':[],'attempted':[]}) if x['tool']=='presence_check']
        self.assertGreaterEqual(len(actions),2)
        context=directory/'context.json';write_json(context,{'workflow_config':config,'a_prediction':str(directory/'a.json'),
            'directory':str(directory),'allow_network':False,'max_requests':4})
        before=runtime.workflow.store.current()['version']
        async def probe():
            async with connect_stdio(context) as tools:
                values=await asyncio.gather(*(tools.observe(a) for a in actions[:2]),return_exceptions=True)
                for value in values:
                    self.assertIsInstance(value,dict);self.assertFalse(value['accepted'])
                    self.assertEqual(value['result']['error_code'],'NETWORK_DISABLED');self.assertEqual(value['new_requests'],0)
        asyncio.run(asyncio.wait_for(probe(),20))
        self.assertEqual(runtime.workflow.store.current()['version'],before)
        self.assertEqual(read_json(runtime.workflow.budget_path)['used'].get('network',0),0)


    def test_reflection_keeps_preexecution_binding_after_target_retired(self):
        r=self.detected();action=next(a for a in r.workflow.propose() if a['tool']=='classify_target');c=ReflectionController(r)
        key=c.key(action);binding=c.binding(action)
        with patch.object(r.workflow,'targets',return_value={}):
            c.record(action,progress=False,feedback={'accepted':True},evidence_version=r.workflow.store.current()['version'],
                     action_key=key,binding=binding)
        self.assertEqual(c.rows[-1]['question_id'],binding['question_id'])
        self.assertEqual(c.rows[-1]['input_fingerprint'],binding['input_fingerprint'])


    def test_mcp_preserves_explicit_resource_environment_without_global_mutation(self):
        import os
        from contextlib import asynccontextmanager
        from TCM_Fusion.lib.mcp.mcp_bridge import connect_stdio
        captured={};context=self.root/'resource_context.json';write_json(context,{})
        @asynccontextmanager
        async def transport(parameters):
            captured.update({k:parameters.env.get(k) for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','TEMP']})
            yield object(),object()
        class Session:
            def __init__(inner,*a,**k):pass
            async def __aenter__(inner):return inner
            async def __aexit__(inner,*a):pass
            async def initialize(inner):return None
        async def probe():
            async with connect_stdio(context):pass
        original=os.environ.get('OPENBLAS_NUM_THREADS')
        with patch.dict(os.environ,{'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','TEMP':str(self.root)}), \
             patch('mcp.client.stdio.stdio_client',transport),patch('mcp.ClientSession',Session):
            asyncio.run(probe())
        self.assertEqual(captured,{'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','TEMP':str(self.root)})
        self.assertEqual(os.environ.get('OPENBLAS_NUM_THREADS'),original)
