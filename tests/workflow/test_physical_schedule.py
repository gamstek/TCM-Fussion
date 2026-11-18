"""用同一样品的两个视野验证模型授权、双阶段预约、并发准备与串行证据提交。
真实文件锁、预算、证据库和 ReAct 图参与运行，阶段替身用同步信号控制完成顺序。

主要定义：
- PhysicalScheduleTests 覆盖相反顺序、重复回执、中断恢复、关闭、额度和图入口。

关联：
- 被测 lib/workflow/physical_schedule、lib/agent/physical_execution 及 react_graph。
"""
import asyncio
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch
import contracts
from TCM_Fusion.lib.runtime.resource_budget import reserve
from TCM_Fusion.lib.runtime.storage import IntegrationError, read_json, write_json
from TCM_Fusion.lib.workflow import physical_schedule as schedule
from TCM_Fusion.lib.workflow.workflow_runtime import WorkflowRuntime
from TCM_Fusion.lib.agent.react_graph import run_react
from TCM_Fusion.lib.tools.agent_tools import LocalToolbox
from TCM_Fusion.tests.support.workflow_fixtures import StageFixture, WorkflowFixture


class TwoViews(StageFixture):
    def execute(self,session,tool,target,work,context):
        if tool != 'detect_fov_initial':
            return super().execute(session,tool,target,work,context)
        image = session.runtime.images[target['image_id']]['path']
        self.calls.append(image.name)
        folder = work/'crops'
        folder.mkdir()
        (folder/'crop.png').write_bytes(image.read_bytes())
        side = contracts.crop_sidecar('crop','crop.png',str(image),image.stem,'single_crystal',
            '方晶','',2,[0,0,8,8],[0,0,8,8],[0,0,8,8],'8x8')
        write_json(folder/'crop.json',side)
        write_json(work/'done.json',{'count':1})
        return {'completion':str(work/'done.json'),'crops_dir':str(folder)}


class PhysicalScheduleTests(WorkflowFixture,unittest.TestCase):
    def setup_pair(self, **config):
        other = self.root/'other.png'
        other.write_bytes(b'second-view')
        self.backend = TwoViews()
        self.runtime = WorkflowRuntime([self.image,other],root=self.root/'schedule',sample_id='pair',
            workflow_config={'physical_parallel':True,'agent_policy':'adaptive_v2',**config},
            simulation=True,backend=self.backend)
        self.s = self.runtime.workflow
        return [a for a in self.s.propose() if a['tool']=='detect_fov_initial']

    def test_reverse_order_duplicate_and_independent_classifications(self):
        actions = self.setup_pair()
        for selected in [actions,None]:
            if selected is None:
                selected = [a for a in self.s.propose() if a['tool']=='classify_target']
            tickets = schedule.reserve_pair(self.s,selected)
            rendezvous = threading.Barrier(2,timeout=10)
            original = self.backend.execute
            def concurrent(*args):
                rendezvous.wait()
                return original(*args)
            with patch.object(self.backend,'execute',side_effect=concurrent), ThreadPoolExecutor(2) as pool:
                futures = [pool.submit(schedule.prepare,self.s,*t) for t in tickets]
                self.assertTrue(all(f.result()['phase']=='ready' for f in futures))
            head = self.s.store.current()['version']
            for ticket in reversed(tickets):
                row = schedule.publish(self.s,*ticket)
                self.assertTrue(row['accepted'],row)
                self.assertEqual(self.s.store.current()['parent'],head)
                head = row['evidence_version']
                self.assertEqual(schedule.publish(self.s,*ticket)['evidence_version'],head)
                self.assertEqual(schedule.prepare(self.s,*ticket)['status'],'done')
            self.assertEqual(self.s.store.current()['version'],head)
        self.assertEqual(read_json(self.s.budget_path)['used'],{'initialization':2,'observation':2})
        self.assertEqual(len(self.backend.calls),4)

    def test_close_fences_private_result_and_unknown_http(self):
        actions = self.setup_pair()
        a,b = schedule.reserve_pair(self.s,actions)
        schedule.prepare(self.s,*a)
        schedule.publish(self.s,*a)
        schedule.prepare(self.s,*b)
        reserve(self.s.budget_path,'network','wire',self.s.limits,metadata={'operation_key':b[0]})
        head = self.s.store.current()['version']
        self.s.close('review')
        self.assertEqual(schedule.publish(self.s,*b)['error_code'],'BRANCH_REVOKED')
        self.assertEqual(read_json(self.s.budget_path)['entries']['wire']['status'],'unknown')
        self.assertEqual(self.s.store.current()['version'],head)

    def test_reconcile_crash_after_head_move(self):
        actions = self.setup_pair()
        a,b = schedule.reserve_pair(self.s,actions)
        schedule.prepare(self.s,*a)
        original = self.s.accept
        def interrupted(*args,**kwargs):
            original(*args,**kwargs)
            raise SystemExit('injected')
        with patch.object(self.s,'accept',side_effect=interrupted),self.assertRaises(SystemExit):
            schedule.publish(self.s,*a)
        schedule.prepare(self.s,*b)
        schedule.publish(self.s,*b)
        head = self.s.store.current()['version']
        row = schedule.publish(self.s,*a)
        self.assertTrue(row['accepted'] and row['repair_completed'])
        self.assertEqual(row['evidence_version'],self.s.store.current()['parent'])
        self.assertEqual(self.s.store.current()['version'],head)
        self.assertEqual(len(self.backend.calls),2)

    def test_scope_change_and_sibling_failure(self):
        actions = self.setup_pair()
        a,b = schedule.reserve_pair(self.s,actions)
        schedule.prepare(self.s,*a)
        schedule.prepare(self.s,*b)
        image = actions[0]['args']['target_id']
        side = self.root/'new.json'
        write_json(side,{'new':True})
        self.s.store.commit(self.s.store.current()['version'],[self.s._art('new','detection',side,[image],image_id=image)])
        self.assertEqual(schedule.publish(self.s,*a)['error_code'],'EVIDENCE_STALE')
        self.assertTrue(schedule.publish(self.s,*b)['accepted'])

    def test_denials_leave_no_reservations(self):
        actions = self.setup_pair(max_initializations=1)
        for selected,code in ((actions,'RESOURCE_BUDGET'),([actions[0]]*2,'PAIR_OVERLAP')):
            with self.assertRaises(IntegrationError) as caught:
                schedule.reserve_pair(self.s,selected)
            self.assertEqual(caught.exception.code,code)
        self.s.config['physical_parallel'] = False
        with self.assertRaises(IntegrationError):
            schedule.reserve_pair(self.s,actions)
        self.assertFalse(read_json(self.s.operations)['entries'])
        self.assertFalse(read_json(self.s.budget_path)['entries'])

    def test_tampered_bundle_and_premature_receipt(self):
        actions = self.setup_pair()
        a,b = schedule.reserve_pair(self.s,actions)
        with self.assertRaises(IntegrationError) as premature:
            schedule.publish(self.s,*a)
        self.assertEqual(premature.exception.code,'PAIR_NOT_READY')
        schedule.prepare(self.s,*a)
        path = self.s.root/'ops'/a[0][:16]/'prepared.json'
        payload = read_json(path)
        payload['bundle']['files'] = {}
        write_json(path,payload)
        self.assertEqual(schedule.publish(self.s,*a)['error_code'],'ARTIFACT_CHANGED')
        with patch.object(self.backend,'execute',side_effect=IntegrationError('CLASSIFY_FAILED','模拟错误')):
            schedule.prepare(self.s,*b)
        self.assertEqual(schedule.publish(self.s,*b)['error_code'],'CLASSIFY_FAILED')
        self.assertEqual(self.s.store.current()['version'],actions[0]['args']['evidence_version'])

    def test_expiry_at_publication_keeps_head_and_budget(self):
        actions = self.setup_pair()
        a,_ = schedule.reserve_pair(self.s,actions)
        schedule.prepare(self.s,*a)
        original = self.s.store._write
        def expire(value,**kw):
            book = read_json(self.s.budget_path)
            book['deadline'] = 0
            write_json(self.s.budget_path,book)
            return original(value,**kw)
        with patch.object(self.s.store,'_write',side_effect=expire):
            self.assertEqual(schedule.publish(self.s,*a)['error_code'],'WORKFLOW_TIMEOUT')
        self.assertEqual(self.s.store.current()['version'],actions[0]['args']['evidence_version'])
        self.assertEqual(read_json(self.s.budget_path)['used'],{'initialization':2})

    def test_graph_authorizes_pair_and_coalesces(self):
        self.setup_pair()
        class Planner:
            def choose(self,ctx):
                ids = ctx['execution_options']['physical_pairs'][0]
                return {'decision':{'kind':'physical_pair','action_ids':ids}},None
        tools = LocalToolbox(self.runtime,ledger_path=self.root/'observations.json')
        out = asyncio.run(run_react(self.runtime,tools,planner=Planner(),planner_mode='scripted',max_steps=2))
        self.assertEqual(out['steps'],2)
        self.assertEqual(out['planning_requests'],1)
        self.assertEqual(sum(e.get('accepted',False) for e in out['events'] if e['phase']=='recompute'),2)
        self.assertEqual(sum(e['phase']=='physical_join' for e in out['events']),1)

    def test_default_on_and_four_view_graph_with_step_cap(self):
        for width,steps in [(3,3),(4,3),(4,4)]:
            with self.subTest(width=width,steps=steps):
                folder = self.root/f'w{width}s{steps}'
                folder.mkdir()
                images = [folder/f'v{i}.png' for i in range(4)]
                for image in images:
                    image.write_bytes(image.name.encode())
                runtime = WorkflowRuntime(images,root=folder/'task',sample_id='group',
                    workflow_config={'agent_policy':'adaptive_v2','max_parallel':width},
                    simulation=True,backend=TwoViews())
                class Planner:
                    def choose(self,ctx):
                        group = next(g for g in ctx['execution_options']['physical_groups']
                                     if g['tool']=='detect_fov_initial')
                        ids = [v[0] for v in group['views'][:group['limit']]]
                        return {'decision':{'kind':'physical_group','action_ids':ids}},None
                tools = LocalToolbox(runtime,ledger_path=folder/'ledger.json')
                out = asyncio.run(run_react(runtime,tools,planner=Planner(),planner_mode='scripted',max_steps=steps))
                self.assertEqual(out['steps'],steps)
                self.assertEqual(out['planning_requests'],1)
                rows = read_json(runtime.workflow.operations)['entries'].values()
                self.assertEqual(len({r['image_id'] for r in rows if r['accepted']}),steps)
                self.assertEqual(read_json(runtime.workflow.budget_path)['used']['initialization'],steps)

    def test_group_menu_rejects_duplicate_view_or_excess_width(self):
        from TCM_Fusion.lib.agent.planning_choices import decode,model_view
        actions = [{'tool':'classify_target','action_id':str(i)} for i in range(5)]
        ctx = {'actions':actions,'execution_options':{'physical_groups':[
            {'tool':'classify_target','limit':3,'views':[['0','1'],['2'],['3'],['4']]}]}}
        short,ids = model_view(ctx)
        def select(keys):
            return decode({'decision':{'kind':'physical_group','action_ids':keys}},short)
        good = select(['A4','A1','A3'])
        self.assertEqual([ids[k] for k in [good['action_id'],*good['physical_partner_ids']]],['3','0','2'])
        for invalid in (['A1','A2'],['A1','A1'],['A1','A3','A4','A5'],['A1','A6']):
            with self.assertRaises(IntegrationError):
                select(invalid)

    def test_runtime_defaults_match_template_and_explicit_off_is_respected(self):
        from pathlib import Path
        import paths
        from TCM_Fusion.lib.application.workflow_entry import WORKFLOW_SCHEMA
        from TCM_Fusion.lib.tools.tool_registry import validate_arguments
        template = read_json(Path(paths.PROJECT_ROOT)/'TCM_Fusion/config/multi_view.json')['agent']['workflow']
        runtime = self.runtime(agent_policy='adaptive_v2')
        session = runtime.workflow
        self.assertTrue(template['physical_parallel'])
        self.assertTrue(schedule.enabled(session))
        self.assertEqual(session.config['max_parallel'],template['max_parallel'])
        for width in [1,2,3,4]:
            validate_arguments(WORKFLOW_SCHEMA,{'max_parallel':width})
        with self.assertRaises(IntegrationError):
            validate_arguments(WORKFLOW_SCHEMA,{'max_parallel':5})
        session.config['physical_parallel'] = False
        self.assertFalse(schedule.enabled(session))
        session.config.update(physical_parallel=True,max_parallel=1)
        self.assertFalse(schedule.enabled(session))

    def partial_run(self, *, slow, review):
        actions = self.setup_pair()
        signal = threading.Event()
        original = self.backend.execute
        def delayed(session,tool,target,work,context):
            if tool=='detect_fov_initial' and target['id']==actions[slow]['args']['target_id']:
                if not signal.wait(20):
                    raise AssertionError('快支未能推进')
            if tool=='classify_target':
                signal.set()
            return original(session,tool,target,work,context)
        class Planner:
            def choose(self,ctx):
                offers = ctx['execution_options']
                if offers.get('physical_pairs'):
                    return {'decision':{'kind':'physical_pair','action_ids':offers['physical_pairs'][0]}},None
                tool = 'request_review' if review else 'classify_target'
                action = next(a for a in ctx['actions'] if a['tool']==tool)
                return {'action_id':action['action_id']},None
        original_fence = schedule.fence_locked
        def fence_then_signal(session,reason):
            original_fence(session,reason)
            if reason=='planner_request_review':
                signal.set()
        tools = LocalToolbox(self.runtime,ledger_path=self.root/'observations.json')
        with patch.object(self.backend,'execute',side_effect=delayed),patch.object(schedule,'fence_locked',side_effect=fence_then_signal):
            out = asyncio.run(run_react(self.runtime,tools,planner=Planner(),planner_mode='scripted',max_steps=3))
        self.assertTrue(signal.is_set())
        self.assertEqual(out['planning_requests'],2)
        self.assertEqual(sum(e['phase']=='physical_partial' for e in out['events']),1)
        entries = list(read_json(self.s.operations)['entries'].values())
        late = next(r for r in entries if r['target_id']==actions[slow]['args']['target_id'])
        if review:
            self.assertEqual(out['stop_reason'],'planner_request_review')
            self.assertEqual(late['error_code'],'BRANCH_REVOKED')
            self.assertIn('late_return_at',late)
            self.assertNotIn('detected:'+late['target_id'],self.s.store.current()['artifacts'])
        else:
            self.assertEqual([r['status'] for r in entries],['done']*3)
            self.assertEqual(out['steps'],3)

    def test_fast_first_view_can_start_classification_before_slow_return(self):
        self.partial_run(slow=1,review=False)

    def test_fast_second_view_can_review_and_fence_slow_return(self):
        self.partial_run(slow=0,review=True)
