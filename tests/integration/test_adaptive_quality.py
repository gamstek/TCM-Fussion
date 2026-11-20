"""分层 QA、按用途采信、真值反馈、并发与硬截止的离线回归；不伪造在线收益。
该组用例由项目根 selftest 发现。涉及模型回复的用例使用可控输入，涉及进程的用例以各断言规定的执行边界为准。

主要定义：
- QualityTests（用途与完整性）: 验证用途化 QA 的 uses 授权、截断/映射失败、歧义组件、多段保留与空前景/极性/标定缺失。
- QualityTests（契约与审计）: 校验 quality_assessment 拒绝非法授权，legacy 后处理审计不改变结果。
- QualityTests（反馈与硬截止）: 阴性观察有用但不作阳性证据、重复同视野缺失不再增益；真实进程超时被杀。
- QualityTests（测量复用）: 真实测量绑定合格 mask，原 B 交付文件不被改写。
- AdaptiveWorkflowTests（v2 启动）: 单图先 YOLO 且与并行观察重叠，多图逐视野初始化。
- AdaptiveWorkflowTests（early 菜单）: 覆盖未满时仍开放 segment/feature_scan，建议不提升无关实例或发明动作。
- AdaptiveWorkflowTests（假设与否决）: finish_veto 重规划、假设撤销/暂停只改范围、参数不匹配与未知 action_id 留痕。
- AdaptiveWorkflowTests（并发与终态）: 独立测量与观察并发、重复预约只计一次、关闭或超时后不再提交。

关联：
- 直接依赖 「lib/agent/action_feedback.py」、「lib/agent/react_graph.py」、「lib/runtime/hard_calls.py」、「lib/runtime/storage.py」、「lib/tools/agent_tools.py」、「lib/workflow/workflow_backend.py」、「lib/workflow/workflow_runtime.py」、「tests/support/workflow_fixtures.py」。
- 统一执行见 tests/README.md。
"""
import asyncio
import copy
import csv
import json
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch
import cv2
import numpy as np
import contracts
import paths
from quality_eval import assess,replay
from TCM_Fusion.lib.runtime.storage import write_json,read_json,file_hash,IntegrationError,digest
from TCM_Fusion.lib.agent.action_feedback import snapshot,evaluate
from TCM_Fusion.lib.workflow.workflow_runtime import WorkflowRuntime
from TCM_Fusion.lib.tools.agent_tools import LocalToolbox,FixtureBackend
from TCM_Fusion.lib.agent.react_graph import run_react
from TCM_Fusion.tests.support.workflow_fixtures import StageFixture


class QualityTests(unittest.TestCase):
    def test_fusion_backend_passes_region_policy_and_supports_legacy_baseline(self):
        from types import SimpleNamespace
        from TCM_Fusion.lib.workflow.workflow_backend import WorkflowBackend
        captured = []
        def stage(argv, **kwargs):
            captured.append(list(map(str, argv)))
            write_json(Path(kwargs['log_path']).parent/'segmentation'/'instances.json',
                       {'instances': [{'igm_status': 'review'}]})
        for index, options in enumerate(({}, {'box_association': 'center_point'})):
            work = self.root/('op'+str(index));work.mkdir()
            session = SimpleNamespace(python=sys.executable, config=options, remaining_seconds=lambda: 10,
                                      cancel_event=None)
            with patch('TCM_Fusion.lib.workflow.workflow_backend.run', side_effect=stage):
                result = WorkflowBackend().execute(session, 'segment_instance', {'path': str(self.side)},
                                                   work, work/'context.json')
            self.assertEqual(result['quality'], 'review')
        self.assertEqual(captured[0][-2:], ['--box-association', 'region_component'])
        self.assertEqual(captured[1][-2:], ['--box-association', 'center_point'])

    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        self.image=self.root/'image.png';cv2.imwrite(str(self.image),np.full((100,100,3),220,np.uint8))
        self.side=self.root/'crop.json'
        write_json(self.side,contracts.crop_sidecar('crop','image.png',str(self.image),'image','fiber','纤维','',2,
            [10,10,90,90],[10,10,90,90],[0,0,100,100],'100x100'))
        self.crop=contracts.CropMeta.from_json(str(self.side))
        self.mask=np.zeros((100,100),np.uint8)
        cv2.polylines(self.mask,[np.array([[20,20],[20,80],[80,80]],np.int32)],False,255,9)

    def qa(self,mask=None,**kwargs):
        m=self.mask if mask is None else mask
        return assess(m,self.crop,bindings={'crop':file_hash(self.image),'crop_meta':file_hash(self.side)},
            stages={'binary':m,'pre_box':m},**kwargs)[1]

    def test_curved_target_center_is_auxiliary_not_whole_rejection(self):
        qa=self.qa(identity_verified=True)
        self.assertTrue(qa['integrity']);self.assertTrue(qa['uses']['length']['allowed'])
        self.assertIn('CENTER_POINT_MISS',[f['code'] for f in qa['findings']])
        self.assertFalse(qa['risk_validated'])

    def test_review_mask_does_not_create_positive_presence(self):
        qa=self.qa();self.assertFalse(any(u['allowed'] for u in qa['uses'].values()))

    def test_truncation_blocks_all_unvalidated_dimensions(self):
        m=self.mask.copy();m[0:22,15:25]=255
        qa=self.qa(m,identity_verified=True)
        self.assertTrue(qa['uses']['existence']['allowed'])
        self.assertFalse(any(qa['uses'][k]['allowed'] for k in ('length','width','area')))

    def test_mapping_and_invalid_numeric_are_hard_failures(self):
        for m in (np.zeros((90,100),np.uint8),np.full((100,100),np.nan)):
            qa=self.qa(m,identity_verified=True)
            self.assertFalse(qa['integrity']);self.assertFalse(any(u['allowed'] for u in qa['uses'].values()))

    def test_ambiguous_components_not_largest_component_fallback(self):
        m=np.zeros((100,100),np.uint8);cv2.circle(m,(30,30),10,255,-1);cv2.circle(m,(70,70),12,255,-1)
        qa=self.qa(m,identity_verified=True)
        self.assertIn('TARGET_ASSOCIATION_AMBIGUOUS',[f['code'] for f in qa['findings']])
        self.assertFalse(qa['uses']['area']['allowed'])

    def test_multipart_retains_valid_parts(self):
        s=self.crop.raw.copy();s['feature_id']='needle_crystal';write_json(self.side,s)
        self.crop=contracts.CropMeta.from_json(str(self.side))
        m=np.zeros((100,100),np.uint8);m[20:80,20:25]=255;m[20:80,65:70]=255
        qa=self.qa(m,identity_verified=True)
        self.assertEqual(qa['metrics']['associated_components'],2)
        self.assertTrue(qa['uses']['length']['allowed'])

    def test_empty_polarity_unknown_and_calibration_missing(self):
        for m,kwargs,code in [(np.zeros((100,100),np.uint8),{},'EMPTY_FOREGROUND'),
                (self.mask,{'polarity_ambiguous':True},'POLARITY_AMBIGUOUS')]:
            qa=self.qa(m,identity_verified=True,**kwargs)
            self.assertIn(code,[f['code'] for f in qa['findings']]);self.assertFalse(qa['uses']['length']['allowed'])
        self.crop.px_per_um=None
        qa=self.qa(identity_verified=True)
        self.assertTrue(qa['uses']['existence']['allowed']);self.assertFalse(qa['uses']['width']['allowed'])

    def test_contract_blocks_incomplete_and_illegal_grants(self):
        qa=self.qa();qa['integrity']=False;qa['uses']['area']['allowed']=True
        with self.assertRaises(contracts.ContractError):contracts.quality_assessment(qa)

    def test_legacy_postprocess_audit_does_not_change_result(self):
        from igm_segment import generated_to_mask,letterbox_to
        image=cv2.cvtColor(self.mask,cv2.COLOR_GRAY2BGR)
        sent,meta=letterbox_to(image,100,100)
        stages={}
        a,ia=generated_to_mask(sent,None,(100,100),meta,box=[10,10,90,90])
        b,ib=generated_to_mask(sent,None,(100,100),meta,box=[10,10,90,90],audit_stages=stages)
        self.assertEqual(a.tobytes(),b.tobytes());self.assertEqual(ia,ib)
        self.assertEqual(set(stages),{'binary','filled','resized','pre_box','legacy_final'})

    def test_negative_observation_is_useful_but_not_positive_evidence(self):
        before = {'features': {}, 'ranking': []}
        after = {**before, 'observed_absence': [('fov', 'fiber')]}
        action = {'tool': 'presence_check', 'args': {'image_id': 'fov', 'feature_id': 'fiber'}}
        result = {'present': False, 'conf': .9}
        value = evaluate(before, after, action, accepted=True, result=result)
        self.assertTrue(value['useful'])
        self.assertEqual(value['new_local_absence'], ['fiber'])
        self.assertEqual(value['new_features'], [])
        self.assertEqual(value['qualified_counter_changes'], [])
        self.assertFalse(evaluate(before, before, action, accepted=True, result=result)['useful'])

    def test_repeated_same_fov_absence_is_not_new_gain(self):
        state={'features':{},'checks':{},'ranking':[],'observed_absence':[('fov','fiber')]}
        value=evaluate(state,state,{'tool':'presence_check','args':{'feature_id':'fiber','image_id':'fov'}},
            accepted=True,result={'present':False,'conf':.9})
        self.assertFalse(value['useful']);self.assertEqual(value['new_local_absence'],[])

    def test_real_process_hard_deadline_kills_hanging_worker(self):
        import subprocess
        from TCM_Fusion.lib.runtime.hard_calls import encode_image, post
        original=subprocess.Popen
        def sleeping_worker(command,**kwargs):
            if 'TCM_Fusion.cli.api_worker' in command:
                self.assertEqual(kwargs['stdin'], subprocess.DEVNULL)
                command=[sys.executable,'-c','import time;time.sleep(20)']
            return original(command,**kwargs)
        with patch('TCM_Fusion.lib.runtime.hard_calls.output_root',return_value=self.root),patch('TCM_Fusion.lib.runtime.hard_calls.subprocess.Popen',side_effect=sleeping_worker):
            for call, argument, code in ((post, {'messages': []}, 'WORKFLOW_TIMEOUT'),
                                         (encode_image, self.image, 'IMAGE_ENCODE_TIMEOUT')):
                with self.subTest(code=code):
                    start = time.monotonic()
                    with self.assertRaises(IntegrationError) as failure:
                        call(argument, .3)
                    self.assertEqual(failure.exception.code, code)
                    self.assertLess(time.monotonic()-start,4)
        self.assertEqual(list((self.root/'request_work').iterdir()),[])

    def test_observation_image_worker_preserves_small_png_and_bounds_large_image(self):
        import base64
        import io
        from PIL import Image
        from TCM_Fusion.lib.runtime.hard_calls import encode_image

        small = self.root/'small.png'
        large = self.root/'large.png'
        Image.new('RGB', (20, 10)).save(small)
        Image.new('RGB', (2048, 1024)).save(large)
        small_url = encode_image(small, 15)
        self.assertEqual(base64.b64decode(small_url.split(',', 1)[1]), small.read_bytes())
        large_url = encode_image(large, 15)
        self.assertTrue(large_url.startswith('data:image/jpeg;base64,'))
        with Image.open(io.BytesIO(base64.b64decode(large_url.split(',', 1)[1]))) as image:
            self.assertEqual(image.size, (1536, 768))

    def test_real_measurement_uses_bound_qualified_mask_without_mutating_legacy(self):
        import shutil
        from igm_segment import letterbox_to
        source=self.root/'source';crops=source/'crops';crops.mkdir(parents=True)
        shutil.copy2(self.image,crops/'image.png')
        write_json(crops/'crop.json',self.crop.raw)
        masks=source/'segmentation/masks';masks.mkdir(parents=True)
        name='image__fiber__inst01.png';cv2.imwrite(str(masks/name),self.mask)
        raw={k:'' for k in contracts.MANIFEST_COLUMNS};raw.update(instance_id=1,inst_mask=name,crop_id='crop',
            source_image_path=str(self.image),feature_id='fiber',object_label='纤维',px_per_um=2,
            igm_status='review',granularity='instance',tight_bbox='[10,10,90,90]')
        with (masks/('image'+contracts.MANIFEST_SUFFIX)).open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=contracts.MANIFEST_COLUMNS);writer.writeheader();writer.writerow(raw)
        gen=source/'segmentation/igm/crop';gen.mkdir(parents=True)
        _,letter=letterbox_to(cv2.cvtColor(self.mask,cv2.COLOR_GRAY2BGR),100,100)
        cv2.imwrite(str(gen/'gen_raw.png'),self.mask)
        write_json(gen/'meta.json',{'letterbox':letter,'image_hw':[100,100],'gen_hw':[100,100],
            'size_match':True,'threshold':127,'invert':False,'polarity_conflict':False,'max_hole_ratio':0})
        before={str(p):file_hash(p) for p in source.rglob('*') if p.is_file()}
        r=WorkflowRuntime([self.image],root=self.root/'task',sample_id='real_cpu',
            workflow_config={'quality_evaluation':'purpose_v1','agent_policy':'adaptive_v1','enabled_tools':['measure_instances'],
                             'python':sys.executable,'max_api_calls':0},source_run=source,
            a={'status':'ok','ranking':[{'label':'黄芪','probability':1}]})
        self.assertFalse(any(a['tool']=='measure_instances' for a in r.workflow.propose()))
        action=next(a for a in r.legal_actions({'observations':[],'attempted':[]}) if a['tool']=='presence_check' and a['args']['feature_id']=='fiber')
        value={'ok':True,'present':True,'conf':.9}
        self.assertTrue(r.validate_observation(action,value))
        r.workflow.confirm_identity(action,value)  # 受信回调的程序单测；视觉身份响应为明确fixture。
        measurement=next(a for a in r.workflow.propose() if a['tool']=='measure_instances')
        result=r.workflow.execute(measurement)
        self.assertTrue(result['accepted'],result)
        records=r.workflow.store.current()['records']
        self.assertIn('igm_review',records[0]['data']['flags'])
        from TCM_Fusion.lib.evidence.purpose_evidence import project
        scope = records[0]['scope']
        self.assertEqual(project(r, scope_id=scope, measurement_id='measurement:' + scope), project(r))
        self.assertEqual(project(r, scope_id=scope, measurement_id='missing'), [])
        scored=r.score([])
        self.assertTrue(scored['cards']['fiber'].present);self.assertGreater(scored['cards']['fiber'].n_instances,0)
        self.assertEqual(before,{str(p):file_hash(p) for p in source.rglob('*') if p.is_file()})
        qa=next(v for v in r.workflow.store.current()['artifacts'].values() if v['kind']=='quality_assessment' and v['valid'])
        Path(qa['path']).write_text('{}',encoding='utf-8')
        with self.assertRaises(IntegrationError):r.score([])


class AdaptiveWorkflowTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        self.image=self.root/'fov.png';self.image.write_bytes(b'fixture')

    def runtime(self,**cfg):
        return WorkflowRuntime([self.image],root=self.root/'task',sample_id='s',a={},simulation=True,
            workflow_config={'agent_policy':'adaptive_v1',**cfg},backend=StageFixture())

    async def test_v2_single_view_starts_with_yolo_action(self):
        runtime = self.runtime(agent_policy='adaptive_v2', targeted_recovery=True, p0_priority=False)
        runtime.a = {'status':'ok','ranking':[{'label':'广藿香','probability':.6},
                                             {'label':'淫羊藿','probability':.4}]}
        menus = []
        intervals = {}
        import threading
        entered = threading.Barrier(2, timeout=10)

        class SlowDetection(StageFixture):
            def execute(self, *args):
                if args[1] != 'detect_fov_initial':
                    return super().execute(*args)
                intervals['yolo_start'] = time.monotonic()
                entered.wait()  # Both tools must enter; sequential scheduling fails.
                result = super().execute(*args)
                intervals['yolo_end'] = time.monotonic()
                return result

        class SlowObservation(FixtureBackend):
            def call(self, *args):
                intervals['glm_start'] = time.monotonic()
                entered.wait()
                result = super().call(*args)
                intervals['glm_end'] = time.monotonic()
                return result

        class Planner:
            def choose(self, context):
                menus.append([a['tool'] for a in context['actions']])
                if len(menus) == 1:
                    observation = next(a for a in context['parallel_observations'] if a['tool']=='presence_check')
                    return {'action_id': context['actions'][0]['action_id'],
                            'parallel_observation_id': observation['action_id']}, None
                return {'action_id': 'review'}, None

        runtime.workflow.backend = SlowDetection()
        toolbox = LocalToolbox(runtime, ledger_path=self.root/'ledger.json', backend=SlowObservation())
        trace = await run_react(runtime, toolbox, planner=Planner(), planner_mode='scripted')
        self.assertIn('detect_fov_initial', menus[0])
        self.assertFalse(entered.broken)  # Serial execution cannot pass the two-party barrier.
        self.assertLessEqual(max(intervals['yolo_start'], intervals['glm_start']),
                             min(intervals['yolo_end'], intervals['glm_end']))
        self.assertEqual(trace['observation_requests'], 1)
        self.assertTrue(any(e.get('policy')=='yolo_initial_and_raw_observation' for e in trace['events']))
        self.assertEqual(trace['stop_reason'], 'planner_request_review')

    def _early_runtime(self, **options):
        class MatchingStage(StageFixture):
            def execute(self, session, tool, target, work, context):
                if tool == 'classify_target':
                    folder = work / 'classification'
                    write_json(folder / 'classification.json', {
                        'crop_id': target.get('crop_id'), 'feature_id': 'single_crystal',
                        'label': '草酸钙方晶', 'confidence': 'high', 'reason': '明确替身',
                        'reject': None})
                    return {'classification': str(folder)}
                if tool == 'feature_scan':
                    folder = work / 'scan/crops'
                    folder.mkdir(parents=True)
                    completion = work / 'scan/scan_result.json'
                    write_json(completion, {'feature_id': target['feature_id'], 'count': 0})
                    return {'crops_dir': str(folder), 'completion': str(completion)}
                return super().execute(session, tool, target, work, context)
        config = {'agent_policy': 'adaptive_v2', 'fast_yolo_stage2': True,
                  'fast_recovery_selection': 'per_feature', 'agent_driven_coverage': True,
                  'targeted_recovery': True, 'early_verification': True,
                  'max_planning': 8, **options}
        runtime = self.runtime(**config)
        runtime.workflow.backend = MatchingStage()
        runtime.a = {'status': 'ok', 'fold_rankings': [[
            {'label': '枳壳', 'probability': .5}, {'label': '白芍', 'probability': .3},
            {'label': '知母', 'probability': .2}]] * 5}
        for tool in ('detect_fov_initial', 'classify_target'):
            action = next(a for a in runtime.workflow.propose() if a['tool'] == tool)
            self.assertEqual(runtime.workflow.execute(action)['status'], 'done')
        return runtime

    async def test_early_menu_keeps_segment_while_other_required_are_missing(self):
        runtime = self._early_runtime()
        menus = []
        class Planner:
            def choose(self, context):
                menus.append(copy.deepcopy(context))
                selected = next((a for a in context['actions'] if a['tool'] == 'segment_instance'),
                                context['actions'][0])
                return {'action_id': selected['action_id'], 'decision_summary': '核验现有判据'}, None
        toolbox = LocalToolbox(runtime, ledger_path=self.root/'early_ledger.json', backend=FixtureBackend())
        trace = await run_react(runtime, toolbox, planner=Planner(), planner_mode='scripted', max_steps=1)
        tools = {a['tool'] for a in menus[0]['actions']}
        self.assertIn('segment_instance', tools)
        self.assertIn('feature_scan', tools)
        self.assertFalse(menus[0]['coverage']['required_all'])
        self.assertTrue(any(e.get('tool') == 'segment_instance' for e in trace['events']))
        self.assertTrue(any(e.get('phase') == 'planning_menu' for e in trace['events']))

    async def test_early_menu_still_allows_model_to_choose_scan(self):
        runtime = self._early_runtime()
        observed = []
        class Planner:
            def choose(self, context):
                observed.append(copy.deepcopy(context))
                selected = next(a for a in context['actions'] if a['tool'] == 'feature_scan')
                return {'action_id': selected['action_id'],
                        'decision_summary': '当前补查竞争候选有价值，暂缓现有实例核验'}, None
        trace = await run_react(runtime, LocalToolbox(runtime, ledger_path=self.root/'scan_ledger.json',
                               backend=FixtureBackend()), planner=Planner(), planner_mode='scripted', max_steps=1)
        self.assertTrue(observed[0]['verification_opportunities'])
        self.assertTrue(any(e.get('tool') == 'feature_scan' for e in trace['events']))
        self.assertFalse(any(e.get('tool') == 'segment_instance' for e in trace['events']))

    async def test_early_measurement_is_offered_without_clearing_scan_queue(self):
        runtime = self._early_runtime()
        segment = next(a for a in runtime.workflow.propose() if a['tool'] == 'segment_instance')
        self.assertEqual(runtime.workflow.execute(segment)['status'], 'done')
        menus = []
        class Planner:
            def choose(self, context):
                menus.append(copy.deepcopy(context))
                selected = next(a for a in context['actions'] if a['tool'] == 'measure_instances')
                return {'action_id': selected['action_id'], 'decision_summary': '完成真实量化'}, None
        trace = await run_react(runtime, LocalToolbox(runtime, ledger_path=self.root/'measure_ledger.json',
                               backend=FixtureBackend()), planner=Planner(), planner_mode='scripted', max_steps=1)
        opportunities = menus[0]['verification_opportunities']
        self.assertEqual(opportunities[0]['tool'], 'measure_instances')
        self.assertIn('G3', {c['candidate_id'] for c in opportunities[0]['criteria']})
        self.assertTrue(any(a['tool'] == 'feature_scan' for a in menus[0]['actions']))
        self.assertTrue(any(e.get('tool') == 'measure_instances' for e in trace['events']))

    async def test_early_advice_does_not_promote_unrelated_instance(self):
        runtime = self._early_runtime()
        runtime.a = {'status': 'ok', 'fold_rankings': [[{'label': '知母', 'probability': 1.0}]] * 5}
        actions = runtime.legal_actions({'observations': [], 'attempted': []})
        self.assertTrue(any(a['tool'] == 'segment_instance' for a in actions))
        self.assertEqual(runtime.verification_opportunities(actions), [])

    async def test_early_advice_does_not_invent_unavailable_action(self):
        runtime = self._early_runtime()
        actions = [a for a in runtime.legal_actions({'observations': [], 'attempted': []})
                   if a['tool'] not in {'segment_instance', 'measure_instances'}]
        self.assertEqual(runtime.verification_opportunities(actions), [])

    async def test_early_advice_checks_full_quantification_budget(self):
        runtime = self._early_runtime()
        actions = runtime.legal_actions({'observations': [], 'attempted': []})
        self.assertTrue(runtime.verification_opportunities(actions, {'recompute': 2, 'total': 2, 'network': 0}))
        self.assertEqual(runtime.verification_opportunities(actions, {'recompute': 1, 'total': 1, 'network': 0}), [])

    async def test_early_initial_detection_remains_available(self):
        runtime = self.runtime(agent_policy='adaptive_v2', early_verification=True,
                               agent_driven_coverage=True, max_planning=4)
        menus = []
        class Planner:
            def choose(self, context):
                menus.append([a['tool'] for a in context['actions']])
                return {'action_id': context['actions'][0]['action_id']}, None
        await run_react(runtime, LocalToolbox(runtime, ledger_path=self.root/'init_ledger.json',
                        backend=FixtureBackend()), planner=Planner(), planner_mode='scripted', max_steps=1)
        self.assertIn('detect_fov_initial', menus[0])
        self.assertIn('request_review', menus[0])

    async def test_early_mode_off_does_not_force_autonomous_coverage_route(self):
        runtime = self._early_runtime(early_verification=False)
        menus = []
        class Planner:
            def choose(self, context):
                menus.append([a['tool'] for a in context['actions']])
                return {'action_id': context['actions'][0]['action_id']}, None
        await run_react(runtime, LocalToolbox(runtime, ledger_path=self.root/'old_ledger.json',
                        backend=FixtureBackend()), planner=Planner(), planner_mode='scripted', max_steps=1)
        self.assertIn('segment_instance', menus[0])
        self.assertIn('feature_scan', menus[0])

    async def test_early_resolved_dimensions_do_not_recommend_repeat_measurement(self):
        runtime = self._early_runtime()
        actions = runtime.legal_actions({'observations': [], 'attempted': []})
        decision = {'ranking': [{'id': 'G3', 'checks': [
            {'feature_id': 'single_crystal', 'present': True, 'state': 'IN_RANGE'}]}]}
        before = runtime.workflow.store.current()['version']
        self.assertEqual(runtime.verification_opportunities(actions, decision=decision), [])
        self.assertEqual(runtime.workflow.store.current()['version'], before)

    async def test_early_opportunity_uses_model_menu_short_id(self):
        from types import SimpleNamespace
        from TCM_Fusion.lib.agent.react_graph import LLMPlanner
        planner = LLMPlanner.__new__(LLMPlanner)
        planner.client = SimpleNamespace(model='fixture', timeout_s=10)
        planner.config = {}
        action = {'action_id': 'long-action-identifier', 'tool': 'segment_instance',
                  'args': {'target_id': 'crop:fixture'}}
        context = {'actions': [action], 'stop_advisory': {}, 'parallel_observations': [],
                   'action_objectives': {}, 'verification_opportunities': [
                       {'action_id': action['action_id'], 'tool': action['tool']} ]}
        def post(payload, timeout):
            body = json.loads(payload['messages'][1]['content'])
            self.assertEqual(body['actions'][0]['action_id'], 'A1')
            self.assertEqual(body['verification_opportunities'][0]['action_id'], 'A1')
            return {'choices': [{'message': {'content': '{"action_id":"A1"}'}}]}
        with patch('TCM_Fusion.lib.runtime.hard_calls.post', side_effect=post):
            reply, usage = planner.choose(context)
        self.assertEqual(reply['action_id'], action['action_id'])

    async def test_v2_p0_gap_is_advisory_and_keeps_independent_checks_available(self):
        runtime = self.runtime(agent_policy='adaptive_v2', targeted_recovery=True)
        runtime.a = {'status': 'ok', 'ranking': [{'label': '龙胆', 'probability': .99},
                                                  {'label': '桔梗', 'probability': .01}]}
        runtime.score([])
        focus = runtime.p0_focus()
        self.assertIsNotNone(focus)
        self.assertEqual(focus['candidate_id'], '龙胆')
        self.assertEqual(focus['feature_id'], 'needle_crystal')
        self.assertTrue(focus['advisory_only'])
        self.assertFalse(focus['mandatory'])
        tools = [a['tool'] for a in runtime.legal_actions({'observations': [], 'attempted': []})]
        self.assertIn('detect_fov_initial', tools)
        self.assertIn('presence_check', tools)

    async def test_v2_failure_can_withdraw_one_hypothesis_and_switch_next_round(self):
        r=self.runtime(agent_policy='adaptive_v2',targeted_recovery=True,max_planning=8,
                       early_verification=True)
        r.workflow.execute(next(a for a in r.workflow.propose() if a['tool']=='detect_fov_initial'))
        r.a={'status':'ok','fold_rankings':[[{'label':'秦皮','probability':1.0}]]*5}
        r.investigate(['G1'],'待核验簇晶假设',r.score([])['evidence_digest'])
        contexts=[]
        class Planner:
            def choose(self,context):
                contexts.append(copy.deepcopy(context))
                if len(contexts)==1:
                    action=next(a for a in context['actions'] if a['tool']=='presence_check'
                                and a['args']['feature_id']=='sand_crystal')
                    return {'action_id':action['action_id'],'decision_summary':'核验砂晶'},None
                if len(contexts)==2:
                    action=next(a for a in context['actions'] if a['tool']=='read_evidence')
                    return {'action_id':action['action_id'],'decision_summary':'失败后重新检查假设',
                            'hypothesis_updates':[
                                {'candidate_id':'G1','status':'withdrawn','reason':'原路线未取得可用观察，撤销调查优先'},
                                {'candidate_id':'龙胆','status':'investigating','reason':'尝试其他知识库假设'}]},None
                return {'action_id':'review','decision_summary':'保留证据不足'},None
        class Failure(FixtureBackend):
            def call(self,*args):raise IntegrationError('TEST_FAILURE','fixture')
        trace=await run_react(r,LocalToolbox(r,ledger_path=self.root/'ledger.json',backend=Failure()),
                              planner=Planner(),planner_mode='scripted',max_steps=6)
        self.assertTrue(contexts[1]['task_reflection'])
        interest=contexts[2]['evidence']['candidate_interest']
        self.assertIn('G1',interest['inactive_ids']);self.assertNotIn('G1',interest['ids'])
        self.assertIn('龙胆',interest['ids'])
        self.assertTrue(any(e['phase']=='hypotheses_updated' for e in trace['events']))
        self.assertNotEqual(trace['decision']['verdict'],'confirmed')
        self.assertEqual(trace['stop_reason'],'planner_request_review')

    async def test_v2_unqualified_finish_exits_to_review_without_replanning(self):
        r = self.runtime(agent_policy="adaptive_v2", early_verification=True)
        contexts = []
        class Planner:
            def choose(self, context):
                contexts.append(copy.deepcopy(context))
                return {"action_id":"finish"}, None
        trace = await run_react(r, LocalToolbox(r, ledger_path=self.root/"ledger.json", backend=FixtureBackend()),
                                planner=Planner(), planner_mode="scripted")
        self.assertEqual(len(contexts),1)
        self.assertEqual(trace['stop_reason'],'planner_request_review')
        self.assertTrue(trace['needs_review'])
        self.assertTrue(any(e['phase']=='stop_projected_to_review' for e in trace['events']))
        self.assertFalse(any(e['phase']=='rejected_stop' for e in trace['events']))

    async def test_hypothesis_update_keeps_decision_basis_and_preserves_new_observation(self):
        r=self.runtime(agent_policy='adaptive_v2',max_planning=4)
        r.workflow.execute(next(a for a in r.workflow.propose() if a['tool']=='detect_fov_initial'))
        r.a={'status':'ok','fold_rankings':[[{'label':'秦皮','probability':1.0}]]*5}
        contexts=[]
        class Planner:
            def choose(self,context):
                contexts.append(copy.deepcopy(context))
                if len(contexts)==1:
                    action=next(a for a in context['actions'] if a['tool']=='presence_check'
                                and a['args']['feature_id']=='sand_crystal')
                    return {'action_id':action['action_id'],'hypothesis_updates':[
                        {'candidate_id':'G1','status':'paused','reason':'本轮后暂不沿簇晶调查'}]},None
                return {'action_id':'review','decision_summary':'保留不确定性'},None
        backend=FixtureBackend({'sand_crystal':{'ok':True,'present':True,'conf':.9}})
        trace=await run_react(r,LocalToolbox(r,ledger_path=self.root/'ledger.json',backend=backend),
                              planner=Planner(),planner_mode='scripted',max_steps=4)
        change=next(e for e in trace['events'] if e['phase']=='hypotheses_updated')
        self.assertEqual(change['evidence_digest'],contexts[0]['evidence']['evidence_digest'])
        self.assertNotEqual(change['applied_evidence_digest'],change['evidence_digest'])
        self.assertEqual(change['decision_source'],'injected_planner')
        self.assertTrue(change['simulation'])
        self.assertTrue(r.score(trace['observations'])['cards']['sand_crystal'].present)

    async def test_calibrated_context_marks_old_scale_read_stale_and_allows_refresh(self):
        r=self.runtime(agent_policy='adaptive_v2',max_planning=6)
        r.workflow.backend=StageFixture(scale=None)
        r.workflow.execute(next(a for a in r.workflow.propose() if a['tool']=='detect_fov_initial'))
        r.a={'status':'ok','fold_rankings':[[{'label':'秦皮','probability':1.0}]]*5}
        contexts=[];read_ids=[]
        class Planner:
            def choose(self,context):
                contexts.append(copy.deepcopy(context));n=len(contexts)
                if n in (1,3):
                    if n==3:
                        assert context['read_results'][0]['stale'] is True
                        assert context['evidence']['workflow']['calibration']['scales'][0]['px_per_um']==2
                    action=next(a for a in context['actions'] if a['tool']=='read_scale')
                    read_ids.append(action['action_id'])
                    return {'action_id':action['action_id']},None
                if n==2:
                    action=next(a for a in context['actions'] if a['tool']=='calibrate_fov')
                    return {'action_id':action['action_id']},None
                return {'action_id':'review','decision_summary':'标定已更新，其他证据仍缺失'},None
        trace=await run_react(r,LocalToolbox(r,ledger_path=self.root/'ledger.json',backend=FixtureBackend()),
                              planner=Planner(),planner_mode='scripted',max_steps=6)
        self.assertEqual(len(read_ids),2);self.assertNotEqual(read_ids[0],read_ids[1])
        self.assertEqual(trace['stop_reason'],'planner_request_review')

    async def test_v2_multiview_review_does_not_force_remaining_views(self):
        second = self.root / "second.png"
        second.write_bytes(b"distinct fixture")
        runtime = WorkflowRuntime([self.image, second], root=self.root / "task",
            sample_id="s", a={}, simulation=True,
            workflow_config={"agent_policy": "adaptive_v2", "max_planning": 8},
            backend=StageFixture())
        offered = []

        class Planner:
            def choose(self, context):
                offered.append([a["tool"] for a in context["actions"]])
                return {"action_id": "review"}, None

        toolbox = LocalToolbox(runtime, ledger_path=self.root / "ledger.json", backend=FixtureBackend())
        trace = await run_react(runtime, toolbox, planner=Planner(), planner_mode="scripted")
        self.assertEqual(len(offered),1)
        self.assertIn('request_review',offered[0])
        self.assertEqual(runtime.workflow.backend.calls,[])
        self.assertEqual(trace["stop_reason"], "planner_request_review")

    async def test_v2_records_argument_mismatch_and_retries_by_action_id(self):
        r = self.runtime(agent_policy="adaptive_v2", max_planning=2)
        r.workflow.execute(next(a for a in r.workflow.propose() if a['tool']=='detect_fov_initial'))
        contexts = []

        class Planner:
            def choose(self, context):
                contexts.append(copy.deepcopy(context))
                if len(contexts) == 1:
                    action = next(a for a in context["actions"] if a["tool"] == "segment_instance")
                    return {"proposal": {"tool": action["tool"],
                                         "args": {**action["args"], "unexpected": "private-value"}}}, None
                return {"action_id": "stale-or-invalid"}, None

        toolbox = LocalToolbox(r, ledger_path=self.root/"ledger.json", backend=FixtureBackend())
        trace = await run_react(r, toolbox, planner=Planner(), planner_mode="scripted")
        detail = contexts[1]["rejection_details"][0]
        self.assertEqual(detail["kind"], "argument_mismatch")
        self.assertEqual(detail["tool"], "segment_instance")
        self.assertEqual(detail["variants"][0]["unexpected_field_count"], 1)
        self.assertNotIn("private-value", str(detail))
        self.assertEqual(trace["stop_reason"], "runtime_request_review")
        self.assertEqual(trace["events"][-1]["remaining_budget"]["planning"], 0)
        self.assertEqual(trace["events"][-1]["last_rejection"]["kind"], "unknown_action_id")

    async def test_v2_selects_action_beyond_first_three_and_bounds_advisory_overrides(self):
        r = self.runtime(agent_policy="adaptive_v2", max_observations=1, early_verification=True)
        r.workflow.execute(next(a for a in r.workflow.propose() if a['tool']=='detect_fov_initial'))
        r.a = {"status": "ok", "ranking": [{"label": "广藿香", "probability": .6},
                                           {"label": "淫羊藿", "probability": .4}]}
        selected_indices = []
        class Planner:
            def choose(self, context):
                actions = context["actions"]
                if not selected_indices:
                    action = next(a for a in actions if a["tool"] == "presence_check")
                else:
                    assert context["stop_advisory"]["recommend"] == "review"
                    action = next(a for a in actions[3:] if a["tool"] in {"list_fovs", "criteria_diff", "list_artifacts"})
                selected_indices.append(actions.index(action))
                return {"action_id": action["action_id"], "override_reason": "核对独立证据索引"}, None
        class FailedBackend(FixtureBackend):
            def call(self, *args):
                raise IntegrationError("TEST_FAILURE", "fixture")
        toolbox = LocalToolbox(r, ledger_path=self.root/"ledger.json", backend=FailedBackend())
        # 隔离本测试的软停止场景：不要让新版 supporting 优先建议覆盖 no_gain 建议。
        # supporting 缺口行为由独立测试覆盖；这里仍断言三次真实决策及两次越过建议。
        with patch.object(r, 'supporting_states', return_value=[]):
            trace = await run_react(r, toolbox, planner=Planner(), planner_mode="scripted", no_gain_limit=1)
        self.assertEqual(trace["stop_reason"], "no_progress_after_advice")
        self.assertEqual(len(selected_indices), 3)
        self.assertTrue(all(i >= 3 for i in selected_indices[1:]))

    async def test_v2_model_expansion_changes_next_round_scope_not_evidence(self):
        r=self.runtime(agent_policy='adaptive_v2',max_planning=4)
        r.workflow.execute(next(a for a in r.workflow.propose() if a['tool']=='detect_fov_initial'))
        r.a={'status':'ok','fold_rankings':[[{'label':'秦皮','probability':1.0}]]*5}
        contexts=[]
        class Planner:
            def choose(self,context):
                contexts.append(copy.deepcopy(context))
                if len(contexts)==1:
                    action=next(a for a in context['actions'] if a['tool']=='read_evidence')
                    return {'action_id':action['action_id'],'investigate_candidates':['龙胆'],
                            'investigation_reason':'核对A候选外假设的判据差异'},None
                return {'action_id':'review','decision_summary':'测试终止'},None
        trace=await run_react(r,LocalToolbox(r,ledger_path=self.root/'ledger.json',backend=FixtureBackend()),
                              planner=Planner(),planner_mode='scripted',max_steps=4)
        self.assertIn('龙胆',contexts[1]['evidence']['candidate_interest']['ids'])
        self.assertTrue(any(e['phase']=='investigation_expanded' for e in trace['events']))
        self.assertNotIn('needle_crystal',r.present_features)

    async def test_adaptive_has_legal_presence_without_measurement(self):
        r=self.runtime();r.a={'status':'ok','ranking':[{'label':'广藿香','probability':.6},{'label':'淫羊藿','probability':.4}]}
        actions=r.legal_actions({'observations':[],'attempted':[]})
        self.assertIn('presence_check',{a['tool'] for a in actions});self.assertFalse(r.records)

    async def test_adaptive_feedback_and_real_graph(self):
        r=self.runtime();t=LocalToolbox(r,ledger_path=self.root/'ledger.json',backend=FixtureBackend())
        trace=await run_react(r,t,planner_mode='heuristic',max_steps=12,max_requests=12)
        self.assertTrue(any('feedback' in e for e in trace['events']))
        self.assertTrue(all(not e['feedback'].get('uses_reference_label',False) for e in trace['events'] if 'feedback' in e))

    async def test_closed_task_cannot_reserve_or_commit_new_action(self):
        r=self.runtime();before=r.workflow.store.current()['version'];r.workflow.close('test_stop')
        action=r.workflow.propose()[0]
        with self.assertRaises(IntegrationError) as failure:r.workflow.execute(action)
        self.assertEqual(failure.exception.code,'WORKFLOW_CLOSED')
        self.assertEqual(before,r.workflow.store.current()['version'])

    async def test_late_physical_result_is_not_committed(self):
        r=self.runtime();a=r.workflow.propose()[0];before=r.workflow.store.current()['version']
        original=r.workflow.backend
        class ExpiresAfterBackend:
            def execute(inner,*args):
                result=original.execute(*args)
                budget=read_json(r.workflow.budget_path);budget['deadline']=0;write_json(r.workflow.budget_path,budget)
                return result
        r.workflow.backend=ExpiresAfterBackend()
        result=r.workflow.execute(a)
        self.assertFalse(result['accepted']);self.assertEqual(result['error_code'],'WORKFLOW_TIMEOUT')
        self.assertEqual(before,r.workflow.store.current()['version'])

    async def test_independent_measurement_and_observation_are_concurrent(self):
        import threading
        r=self.runtime(max_parallel=2)
        intervals={}
        measurement_started=threading.Event();observation_started=threading.Event()
        class SlowStage(StageFixture):
            def execute(self,s,tool,target,work,context):
                if tool=='measure_instances':
                    intervals['m_start']=time.monotonic();measurement_started.set()
                    if not observation_started.wait(5):
                        raise AssertionError('measurement ran without a concurrent observation')
                    time.sleep(.03)  # Keep the interval wider than a Windows clock tick.
                result=super().execute(s,tool,target,work,context)
                if tool=='measure_instances':intervals['m_end']=time.monotonic()
                return result
        r.workflow.backend=SlowStage()
        # 阶段fixture用于并发协议，不冒充真实分割效果。
        for tool in ('detect_fov_initial','classify_target','segment_instance'):
            a=next(a for a in r.workflow.propose() if a['tool']==tool);r.workflow.execute(a)
        r.a={'status':'ok','ranking':[{'label':'广藿香','probability':.6},{'label':'淫羊藿','probability':.4}]}
        class SlowFixture(FixtureBackend):
            def call(self,*args):
                intervals['o_start']=time.monotonic();observation_started.set()
                if not measurement_started.wait(5):
                    raise AssertionError('observation ran without concurrent measurement')
                time.sleep(.03)
                result=super().call(*args);intervals['o_end']=time.monotonic();return result
        tools=LocalToolbox(r,ledger_path=self.root/'ledger.json',backend=SlowFixture())
        trace=await run_react(r,tools,planner_mode='heuristic',max_steps=2,max_requests=5)
        self.assertTrue(any(e['phase']=='parallel_join' for e in trace['events']))
        self.assertTrue(r.workflow.store.current()['records'])
        self.assertLess(max(intervals['m_start'],intervals['o_start']),min(intervals['m_end'],intervals['o_end']))

    async def test_duplicate_action_reserves_only_once_without_holding_lock_over_http(self):
        r=self.runtime();r.a={'status':'ok','ranking':[{'label':'广藿香','probability':.6},{'label':'淫羊藿','probability':.4}]}
        a=next(a for a in r.legal_actions({'observations':[],'attempted':[]}) if a['tool']=='presence_check')
        backend=FixtureBackend();x=LocalToolbox(r,ledger_path=self.root/'shared.json',backend=backend)
        y=LocalToolbox(r,ledger_path=self.root/'shared.json',backend=backend)
        values=await asyncio.gather(x.observe(a),y.observe(a))
        self.assertEqual(len(backend.calls),1);self.assertEqual(sum(v['new_requests'] for v in values),1)
