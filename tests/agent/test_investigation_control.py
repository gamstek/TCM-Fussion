"""建议/硬约束分离回归：真实LangGraph与明确fixture，验证自主选择及无收益收尾。

由现有selftest发现，不联网；断言来源/身份/尺寸资格仍由原有工具和规则测试保护。
该组用例由项目根 selftest 发现。涉及模型回复的用例使用可控输入，涉及进程的用例以各断言规定的执行边界为准。

主要定义：
- InvestigationControlTests。

关联：
- 直接依赖 「lib/agent/investigation_control.py」、「lib/agent/react_graph.py」、「lib/runtime/storage.py」、「lib/tools/agent_tools.py」、「tests/support/workflow_fixtures.py」。
- 统一执行见 tests/README.md。
"""
import asyncio
import copy
import io
import json
import os
import unittest
from unittest.mock import patch

from TCM_Fusion.lib.tools.agent_tools import LocalToolbox, FixtureBackend
from TCM_Fusion.lib.agent.investigation_control import progress_signature, continuation_count
from TCM_Fusion.lib.agent.react_graph import run_react
from TCM_Fusion.lib.runtime import storage
from TCM_Fusion.tests.support import workflow_fixtures as fixtures


class InvestigationControlTests(unittest.TestCase):
    setUp = fixtures.WorkflowFixture.setUp
    runtime = fixtures.WorkflowFixture.runtime

    def run_agent(self, runtime, planner, **options):
        toolbox=LocalToolbox(runtime,ledger_path=self.root/'obs.json',backend=FixtureBackend())
        return asyncio.run(run_react(runtime,toolbox,planner=planner,planner_mode='scripted',**options))

    def test_model_can_observe_before_initial_detection_despite_legacy_auto_flags(self):
        r=self.runtime(agent_policy='adaptive_v2',auto_prepare_stage=True,
                       auto_required_measurement=True,early_verification=False)
        r.a={'status':'ok','fold_rankings':[[{'label':'秦皮','probability':1.}]]*5}
        contexts=[]
        class Planner:
            def choose(self,c):
                contexts.append(copy.deepcopy(c))
                if len(contexts)==1:
                    a=next(a for a in c['actions'] if a['tool']=='presence_check')
                    return {'action_id':a['action_id']},None
                return {'action_id':'review'},None
        trace=self.run_agent(r,Planner())
        self.assertEqual(trace['stop_reason'],'planner_request_review')
        self.assertEqual(r.workflow.backend.calls,[])
        self.assertEqual(trace['observations_executed'],1)
        self.assertFalse(any(e['phase']=='auto_prepare' for e in trace['events']))

    def test_review_is_always_available_with_no_evidence(self):
        r=self.runtime(agent_policy='adaptive_v2',agent_driven_coverage=True)
        class Planner:
            def choose(self,c):
                assert any(a['tool']=='request_review' for a in c['actions'])
                return {'action_id':'review'},None
        trace=self.run_agent(r,Planner())
        self.assertEqual(trace['planning_requests'],1)
        self.assertEqual(trace['steps'],0)
        self.assertTrue(trace['needs_review'])

    def test_insufficient_finish_does_not_trigger_another_paid_planning_round(self):
        r=self.runtime(agent_policy='adaptive_v2');calls=[]
        class Planner:
            def choose(self,c):
                calls.append(c)
                return {'action_id':'finish','hypothesis_updates':[{'invalid':'fixture'}]},None
        trace=self.run_agent(r,Planner())
        self.assertEqual(len(calls),1)
        self.assertEqual(trace['stop_reason'],'planner_request_review')
        self.assertEqual(trace['planning_requests'],1)
        self.assertFalse(any(e['phase']=='rejected_stop' for e in trace['events']))
        self.assertTrue(any(e['phase']=='hypothesis_update_rejected' for e in trace['events']))
        self.assertNotEqual(trace['decision']['verdict'],'confirmed')

    def test_planner_failure_never_silently_picks_first_investigation_action(self):
        r=self.runtime(agent_policy='adaptive_v2',auto_prepare_stage=True)
        class Broken:
            def choose(self,c):raise RuntimeError('fixture planner failure')
        trace=self.run_agent(r,Broken())
        self.assertEqual(trace['stop_reason'],'runtime_request_review')
        self.assertTrue(trace['planner_degraded'])
        self.assertEqual(r.workflow.backend.calls,[])

    def test_no_gain_continuations_stop_without_repeated_stop_requests(self):
        r=self.runtime(agent_policy='adaptive_v2',max_planning=20)
        chosen=[]
        class Planner:
            def choose(self,c):
                # 每次都是不同只读动作，但没有新证据或能力；不能靠改动作名无限继续。
                a=next(a for a in c['actions'] if a['tool'] in
                       {'read_scale','list_fovs','list_features','criteria_diff'})
                chosen.append(a['tool']);return {'action_id':a['action_id']},None
        trace=self.run_agent(r,Planner(),max_steps=20,no_gain_limit=1)
        self.assertEqual(trace['stop_reason'],'no_progress_after_advice')
        self.assertEqual(len(chosen),3)
        self.assertEqual(trace['no_progress_after_advice'],2)
        self.assertFalse(any(e['phase']=='rejected_stop' for e in trace['events']))
        self.assertEqual(r.workflow.backend.calls,[])

    def test_useful_continuations_can_exceed_two_disagreements(self):
        r=self.runtime(agent_policy='adaptive_v2',max_planning=20,max_resource_units=200,
                       auto_prepare_stage=True,auto_measure_after_segment=False)
        r.workflow.execute(next(a for a in r.workflow.propose() if a['tool']=='detect_fov_initial'))
        schedule=['read_scale','classify_target','list_features','segment_instance',
                  'criteria_diff','measure_instances','review'];seen=[]
        class Planner:
            def choose(self,c):
                tool=schedule[len(seen)];seen.append(tool)
                if tool=='review':return {'action_id':'review'},None
                a=next(a for a in c['actions'] if a['tool']==tool)
                return {'action_id':a['action_id']},None
        trace=self.run_agent(r,Planner(),max_steps=12,no_gain_limit=1)
        self.assertEqual(trace['stop_reason'],'planner_request_review')
        overrides=[e for e in trace['events'] if e.get('phase')=='thought' and e.get('overrode_advisory')]
        self.assertGreaterEqual(len(overrides),3)
        self.assertTrue(r.workflow.store.current()['records'])
        self.assertEqual(trace['no_progress_after_advice'],0)

    def test_unrelated_file_commit_is_not_progress(self):
        r=self.runtime(agent_policy='adaptive_v2');before=progress_signature(r,r.score([]))
        path=self.root/'marker.json';storage.write_json(path,{'changed_file':True})
        snap=r.workflow.store.current();image=next(iter(r.images))
        r.workflow.store.commit(snap['version'],[r.workflow._art('marker','diagnostic',path,[image])])
        self.assertEqual(before,progress_signature(r,r.score([])))

    def test_hypothesis_or_advisory_changes_do_not_reset_no_progress_count(self):
        self.assertEqual(continuation_count(1,progress=False,continued_after_advice=False),1)
        self.assertEqual(continuation_count(1,progress=False,continued_after_advice=True),2)
        self.assertEqual(continuation_count(2,progress=True,continued_after_advice=True),0)

    def test_uninvestigated_repair_is_advice_not_formal_veto(self):
        r=self.runtime(agent_policy='adaptive_v2')
        # 仅隔离调度权限：以明确规则替身给出已授权结果，不冒充真实鉴别正确性。
        scored=r.score([]);scored['decision'].update(verdict='confirmed',needs_review=False,
                                                    conclusion={'id':'G3'})
        a=r.action('segment_instance',{'target_id':'fixture-unused',
                   'evidence_version':r.workflow.store.current()['version']},'workflow_repair')
        class Planner:
            def choose(self,c):
                pending=next(b for b in c['stop_decision']['blockers'] if b['code']=='pending_repair')
                assert pending['advisory_only'] and not pending['hard']
                assert c['stop_decision']['can_finish']
                return {'action_id':'finish'},None
        with patch.object(r,'score',return_value=scored),\
             patch.object(r,'legal_actions',return_value=[a]),\
             patch.object(r.workflow,'unresolved',return_value=True):
            trace=self.run_agent(r,Planner())
        self.assertEqual(trace['stop_reason'],'planner_finish')
        self.assertEqual(r.workflow.backend.calls,[])

    def test_planning_budget_remains_hard_and_does_not_start_any_tool(self):
        r=self.runtime(agent_policy='adaptive_v2',max_planning=0)
        class Planner:
            def choose(self,c):raise AssertionError('budget must prevent this call')
        trace=self.run_agent(r,Planner())
        self.assertEqual(trace['planning_requests'],0)
        self.assertEqual(trace['stop_reason'],'runtime_request_review')
        self.assertEqual(r.workflow.backend.calls,[])

    @unittest.skipUnless(os.name=='nt','Windows atomic-replace reader guard')
    def test_transient_local_read_retry_never_retries_an_api(self):
        with patch('TCM_Fusion.lib.runtime.storage.Path.open',side_effect=[PermissionError(13,'busy'),io.StringIO('{"ok":true}')]) as opened,\
             patch('TCM_Fusion.lib.runtime.storage.time.sleep'):
            self.assertEqual(storage.read_json('fixture.json'),{'ok':True})
        self.assertEqual(opened.call_count,2)

    @unittest.skipUnless(os.name=='nt','Windows atomic-replace reader guard')
    def test_permanent_read_denial_is_not_hidden_or_returned_as_empty_data(self):
        with patch('TCM_Fusion.lib.runtime.storage.Path.open',side_effect=PermissionError(13,'denied')) as opened,\
             patch('TCM_Fusion.lib.runtime.storage.time.sleep'):
            with self.assertRaises(PermissionError):storage.read_json('fixture.json')
        self.assertEqual(opened.call_count,4)

    def test_malformed_json_is_not_treated_as_transient_or_valid(self):
        with patch('TCM_Fusion.lib.runtime.storage.Path.open',return_value=io.StringIO('bad json')) as opened:
            with self.assertRaises(json.JSONDecodeError):storage.read_json('fixture.json')
        self.assertEqual(opened.call_count,1)
