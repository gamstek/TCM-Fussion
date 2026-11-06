"""分层QA、按用途采信、真值反馈、并发和硬截止；全部离线，不伪造在线收益。"""
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
from TCM_Fusion.lib.storage import write_json,read_json,file_hash,IntegrationError,digest
from TCM_Fusion.lib.action_feedback import snapshot,evaluate,reflection_card
from TCM_Fusion.lib.outcome_evaluation import evaluate as outcome
from TCM_Fusion.lib.ablation_evaluation import wilson,compare
from TCM_Fusion.lib.workflow_runtime import WorkflowRuntime
from TCM_Fusion.lib.agent_tools import LocalToolbox,FixtureBackend
from TCM_Fusion.lib.react_graph import run_react
from TCM_Fusion.tests.test_workflow import StageFixture


class QualityTests(unittest.TestCase):
    def test_fusion_backend_passes_region_policy_and_supports_legacy_baseline(self):
        from types import SimpleNamespace
        from TCM_Fusion.lib.workflow_backend import WorkflowBackend
        captured = []
        def stage(argv, **kwargs):
            captured.append(list(map(str, argv)))
            write_json(Path(kwargs['log_path']).parent/'segmentation'/'instances.json',
                       {'instances': [{'igm_status': 'review'}]})
        for index, options in enumerate(({}, {'box_association': 'center_point'})):
            work = self.root/('op'+str(index));work.mkdir()
            session = SimpleNamespace(python=sys.executable, config=options, remaining_seconds=lambda: 10,
                                      cancel_event=None)
            with patch('TCM_Fusion.lib.workflow_backend.run', side_effect=stage):
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

    def test_true_reference_is_not_exposed_by_task_feedback(self):
        trace={'decision':{'conclusion':{'id':'correct'},'verdict':'confirmed'},'needs_review':False,'events':[]}
        ref={'label':'correct','label_source':'expert','group_id':'g','split':'training','label_scope':'sample_identity'}
        value=outcome(trace,ref,a_label='wrong')
        self.assertTrue(value['corrected_a']);self.assertNotIn('label',value)
        self.assertFalse(reflection_card([])['uses_reference_label'])
        trace['needs_review']=True
        value=outcome(trace,ref,a_label='correct')
        self.assertTrue(value['abstained']);self.assertFalse(value['broke_correct_a'])

    def test_negative_observation_is_useful_but_not_positive_evidence(self):
        s={'features':{},'checks':{},'ranking':[],'verdict':'insufficient_evidence'}
        action={'tool':'presence_check','args':{'feature_id':'fiber'}}
        v=evaluate(s,s,action,accepted=True,result={'present':False,'conf':.9})
        self.assertTrue(v['useful']);self.assertEqual(v['new_features'],[])

    def test_repeated_same_fov_absence_is_not_new_gain(self):
        state={'features':{},'checks':{},'ranking':[],'observed_absence':[('fov','fiber')]}
        value=evaluate(state,state,{'tool':'presence_check','args':{'feature_id':'fiber','image_id':'fov'}},
            accepted=True,result={'present':False,'conf':.9})
        self.assertFalse(value['useful']);self.assertEqual(value['new_counterevidence'],[])

    def test_image_label_is_not_silently_sample_identity_truth(self):
        trace={'decision':{'conclusion':{'id':'x'}},'needs_review':False}
        result=outcome(trace,{'label':'x','label_source':'expert','group_id':'g','split':'test','label_scope':'image_class'})
        self.assertFalse(result['scope_compatible']);self.assertIsNone(result['correct'])

    def test_zero_errors_does_not_imply_zero_risk(self):
        self.assertGreater(wilson(0,5)[1],0)
        with self.assertRaises(IntegrationError):compare({'split':'test','cases':[]})

    def test_real_process_hard_deadline_kills_hanging_worker(self):
        import subprocess
        from TCM_Fusion.lib.hard_calls import encode_image, post
        original=subprocess.Popen
        def sleeping_worker(command,**kwargs):
            if 'TCM_Fusion.cli.api_worker' in command:
                self.assertEqual(kwargs['stdin'], subprocess.DEVNULL)
                command=[sys.executable,'-c','import time;time.sleep(20)']
            return original(command,**kwargs)
        with patch('TCM_Fusion.lib.hard_calls.output_root',return_value=self.root),patch('TCM_Fusion.lib.hard_calls.subprocess.Popen',side_effect=sleeping_worker):
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
        from TCM_Fusion.lib.hard_calls import encode_image

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
        self.assertEqual(menus[0], ['detect_fov_initial'])
        self.assertFalse(entered.broken)  # Serial execution cannot pass the two-party barrier.
        self.assertLessEqual(max(intervals['yolo_start'], intervals['glm_start']),
                             min(intervals['yolo_end'], intervals['glm_end']))
        self.assertEqual(trace['observation_requests'], 1)
        self.assertTrue(any(e.get('policy')=='yolo_initial_and_raw_observation' for e in trace['events']))
        self.assertEqual(trace['stop_reason'], 'planner_request_review')

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
        r=self.runtime(agent_policy='adaptive_v2',targeted_recovery=True,max_planning=8)
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

    async def test_v2_veto_replans_with_one_observation_and_carries_failure(self):
        r = self.runtime(agent_policy="adaptive_v2", max_observations=1)
        r.workflow.execute(next(a for a in r.workflow.propose() if a['tool']=='detect_fov_initial'))
        r.a = {"status": "ok", "ranking": [{"label": "广藿香", "probability": .6},
                                           {"label": "淫羊藿", "probability": .4}]}
        contexts = []
        class Planner:
            def choose(self, context):
                contexts.append(copy.deepcopy(context))
                if len(contexts) == 1:
                    return {"action_id": "finish", "override_reason": "核对确认边界"}, None
                if len(contexts) == 2:
                    action = next(a for a in context["actions"] if a["tool"] == "presence_check")
                    return {"proposal": {"tool": action["tool"], "args": action["args"]}}, None
                return {"action_id": "review", "decision_summary": "视觉失败"}, None
        class FailedBackend(FixtureBackend):
            def call(self, *args):
                raise IntegrationError("TEST_FAILURE", "fixture")
        toolbox = LocalToolbox(r, ledger_path=self.root/"ledger.json", backend=FailedBackend())
        trace = await run_react(r, toolbox, planner=Planner(), planner_mode="scripted", max_requests=1)
        self.assertEqual(contexts[1]["rejected_proposals"], ["finish_veto"])
        self.assertEqual(contexts[1]["remaining_budget"]["observation"], 1)
        self.assertEqual(contexts[2]["remaining_budget"]["observation"], 0)
        self.assertEqual(contexts[2]["previous_observations"][0]["result"]["error_code"], "TEST_FAILURE")
        self.assertIsNotNone(contexts[2]["task_reflection"])
        self.assertEqual(trace["stop_reason"], "planner_request_review")
        self.assertEqual(trace["events"][-1]["override_reason"], "视觉失败")
        self.assertTrue(trace["needs_review"])

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

    async def test_v2_multiview_initializes_remaining_view_after_rejected_stop(self):
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
        self.assertEqual(offered[:4], [["detect_fov_initial"] * 2] * 2 + [["detect_fov_initial"]] * 2)
        detections = [event for event in trace["events"] if event.get("tool") == "detect_fov_initial"]
        self.assertEqual(len({e["result"]["target_id"] for e in detections}), 2)
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
        r = self.runtime(agent_policy="adaptive_v2", max_observations=1)
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
        trace = await run_react(r, toolbox, planner=Planner(), planner_mode="scripted", no_gain_limit=1)
        self.assertEqual(trace["stop_reason"], "override_exhausted")
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
        r=self.runtime(max_parallel=2)
        intervals={}
        class SlowStage(StageFixture):
            def execute(self,s,tool,target,work,context):
                if tool=='measure_instances':intervals['m_start']=time.monotonic();time.sleep(.2)
                result=super().execute(s,tool,target,work,context)
                if tool=='measure_instances':intervals['m_end']=time.monotonic()
                return result
        r.workflow.backend=SlowStage()
        # 阶段fixture用于并发协议，不冒充真实分割效果。
        for _ in range(2):
            a=r.workflow.propose()[0];r.workflow.execute(a)
        r.a={'status':'ok','ranking':[{'label':'广藿香','probability':.6},{'label':'淫羊藿','probability':.4}]}
        class SlowFixture(FixtureBackend):
            def call(self,*args):
                intervals['o_start']=time.monotonic();time.sleep(.15)
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
