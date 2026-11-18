"""在真实会话和注册表上检查工件提交、身份门、预算、重试与去重。多数阶段使用显式替身，几何回归实际启动 MCP 子进程和原 measure.py 处理合成掩码。

主要定义：
- StageFixture: 以显式替身产出检测、标定、分割、分类和测量工件，保证离线可重复。
- WorkflowTests（输入与快路径）: feature_scan 缺 feature_id 在执行前拒绝；fast 路径先 YOLO 后 GLM，hybrid 复用同一 crop。
- WorkflowTests（通用补扫）: per_feature 候选判据绑定、tile/请求预算、空结果不作证据、跨特征结果被拒绝。
- WorkflowTests（证据与缓存）: EvidenceStore 的 CAS 失效、环检测与篡改拒绝，operation 缓存和录制回放不重复发图。
- WorkflowTests（故障与预算）: 服务 4xx 熔断、空掩膜与未知失败区分、recompute 为 0 不调后端、网络守卫和 worker 超时。
- WorkflowTests（上下文与菜单）: action 菜单截断仍保留 finish/request_review，重复 feedback 被去重。
- WorkflowTests（假设与准入）: investigate/update_hypotheses 只改变调查范围，证据不足时不调用模型。
- WorkflowTests（真实几何链路）: 复用原 measure.py 侧车与 mask，验证量纲与规则排除不被改写。

关联：
- 直接依赖 「lib/agent/react_graph.py」、「lib/application/pipeline.py」、「lib/decision/decision_llm.py」、「lib/evidence/artifact_index.py」、「lib/evidence/evidence_store.py」、「lib/offline/workflow_replay.py」、「lib/runtime/controlled_process.py」、「lib/runtime/resource_budget.py」。
- 统一执行见 tests/README.md。
"""
import asyncio
import copy
import csv
import json
import sys
import tempfile
import subprocess
import time
import unittest
from pathlib import Path
from unittest.mock import patch
import contracts
import paths
from TCM_Fusion.lib.runtime.storage import IntegrationError, digest, file_hash, read_json, write_json
from TCM_Fusion.lib.tools.tool_registry import ToolSpec, validate_arguments
from TCM_Fusion.lib.evidence.evidence_store import EvidenceStore
from TCM_Fusion.lib.workflow.workflow_runtime import WorkflowRuntime
from TCM_Fusion.lib.workflow.workflow_backend import WorkflowBackend
from TCM_Fusion.lib.runtime.controlled_process import kill_tree
from TCM_Fusion.lib.tools.agent_tools import LocalToolbox, FixtureBackend
from TCM_Fusion.lib.agent.react_graph import run_react, compact_action_menu, compact_feedback
from TCM_Fusion.lib.evidence.artifact_index import ArtifactIndex
from TCM_Fusion.lib.runtime.resource_budget import install_http_guard
from TCM_Fusion.tests.support.workflow_fixtures import StageFixture, WorkflowFixture


class WorkflowTests(WorkflowFixture, unittest.TestCase):
    def test_feature_scan_rejects_missing_feature_before_work(self):
        from TCM_Fusion.lib.workflow.workflow_backend import WorkflowBackend

        for target in ({}, {'feature_id': ''}, {'feature_id': '   '}):
            with self.subTest(target=target):
                with self.assertRaises(IntegrationError) as raised:
                    WorkflowBackend().execute(self.runtime().workflow, 'feature_scan',
                                              target, self.root / 'scan', None)
                self.assertEqual(raised.exception.code, 'FEATURE_SCAN_INPUT')
                self.assertFalse((self.root / 'scan').exists())

    def test_fast_path_keeps_yolo_and_delays_glm_until_classified(self):
        fast = WorkflowRuntime([self.image], root=self.root / 'fast_task',
                               sample_id='sample', a={},
                               workflow_config={'max_recompute': 8,
                                                'fast_yolo_stage2': True,
                                                'hybrid_scan_enabled': True,
                                                'feature_scan_features': ['needle_crystal']},
                               simulation=True, backend=StageFixture())
        image = next(iter(fast.images))
        self.assertNotIn('hybrid_scan:' + image, fast.workflow.targets())
        detect = next(a for a in fast.legal_actions({'observations': []})
                      if a['tool'] == 'detect_fov_initial')
        self.assertTrue(fast.tools.invoke(fast, detect, {'action': detect})['accepted'])
        # YOLO crop 尚未分类，不允许提前走 GLM 兜底。
        self.assertNotIn('hybrid_scan:' + image, fast.workflow.targets())
        classify = next(a for a in fast.legal_actions({'observations': []})
                        if a['tool'] == 'classify_target')
        self.assertTrue(fast.tools.invoke(fast, classify, {'action': classify})['accepted'])
        self.assertNotIn('hybrid_scan:' + image, fast.workflow.targets())
        cls = [v for v in fast.workflow.store.current()['artifacts'].values()
               if v.get('valid') and v.get('kind') == 'classification']
        self.assertTrue(cls)
        self.assertTrue(any(v.get('identity_verified') for v in cls))

    def test_hybrid_reuses_same_yolo_crop_and_keeps_classification_valid(self):
        fast = WorkflowRuntime([self.image], root=self.root / 'reuse_task', sample_id='sample',
                               a={}, simulation=True, backend=StageFixture(),
                               workflow_config={'fast_yolo_stage2': True, 'hybrid_scan_enabled': True})
        image = next(iter(fast.images))
        detect = next(a for a in fast.legal_actions({'observations': []})
                      if a['tool'] == 'detect_fov_initial')
        detected = fast.tools.invoke(fast, detect, {'action': detect})
        self.assertTrue(detected['accepted'])
        classify = next(a for a in fast.legal_actions({'observations': []})
                        if a['tool'] == 'classify_target')
        self.assertTrue(fast.tools.invoke(fast, classify, {'action': classify})['accepted'])
        before = next(v for v in fast.workflow.store.current()['artifacts'].values()
                      if v.get('kind') == 'classification' and v.get('valid'))
        source = fast.workflow.store.current()['artifacts']
        crop = next(v for v in source.values() if v.get('kind') == 'crop' and v.get('valid'))
        detect_folder = Path(detected['bundle']['root']) / 'detection' / 'crops'
        feature_folder = self.root / 'empty_feature_crops'
        feature_folder.mkdir()
        completion = self.root / 'empty_feature_done.json'
        write_json(completion, {'count': 0})
        target = {'id': 'hybrid_scan:' + image, 'kind': 'hybrid_scan', 'image_id': image,
                  'source': str(self.image), 'parents': [image], 'box_quality_retry': False}
        fast.workflow.accept('hybrid_scan', target,
                             {'detections': {'crops_dir': str(detect_folder),
                                             'completion': str(Path(detected['bundle']['root']) / 'done.json')},
                              'feature_scan': {'crops_dir': str(feature_folder),
                                               'completion': str(completion)}},
                             fast.workflow.store.current()['version'])
        after = next(v for v in fast.workflow.store.current()['artifacts'].values()
                     if v.get('id') == before['id'])
        self.assertTrue(after['valid'])
        self.assertEqual(crop['id'], next(v['id'] for v in fast.workflow.store.current()['artifacts'].values()
                                          if v.get('kind') == 'crop' and v.get('valid')))

    def _generic_recovery_runtime(self, labels=('白芍',), **config):
        runtime = self.runtime(fast_yolo_stage2=True, targeted_recovery=True,
                               fast_recovery_selection='per_feature',
                               fast_fallback_max_features=3,
                               fast_fallback_max_locate_calls=12,
                               feature_scan_max_tiles=4, **config)
        runtime.a = {'status': 'ok', 'fold_rankings': [
            [{'label': label, 'probability': 1 / len(labels)} for label in labels]] * 5}
        for image in runtime.images:
            done = self.root / (image + '_detected.json')
            write_json(done, {'count': 0})
            workflow = runtime.workflow
            artifact = workflow._art('detected:' + image, 'detection', done,
                                     [image], image_id=image)
            workflow.store.commit(workflow.store.current()['version'], [artifact])
        return runtime

    def test_generic_white_peony_exposes_supporting_without_needle(self):
        runtime = self._generic_recovery_runtime()
        workflow = runtime.workflow
        before = workflow.store.current()['version']
        targets = workflow.targets()
        scans = [t for t in targets.values() if t.get('kind') == 'feature_scan']
        self.assertEqual({t['feature_id'] for t in scans},
                         {'cluster_crystal', 'starch_grain', 'vessel', 'fiber'})
        self.assertFalse(any(t.get('kind') == 'hybrid_scan' for t in targets.values()))
        self.assertTrue(all(len(t['criteria']) == 1 for t in scans))
        self.assertEqual(workflow.store.current()['version'], before)
        offered = [a for a in workflow.propose() if a['tool'] == 'feature_scan']
        self.assertEqual(len(offered), 4)  # menu options are not truncated to the spend cap
        self.assertFalse(workflow.coverage()['required_all'])

    def test_generic_competitor_needle_has_candidate_basis(self):
        runtime = self._generic_recovery_runtime(('白芍', '知母'))
        options = {o['feature_id']: o for o in runtime.workflow.recovery_options()}
        self.assertEqual(options['needle_crystal']['criteria'],
                         [{'candidate_id': '知母', 'role': 'required'}])
        self.assertEqual(options['vessel']['criteria'],
                         [{'candidate_id': '白芍', 'role': 'supporting'}])

    def test_generic_non_needle_candidate_uses_its_own_required(self):
        runtime = self._generic_recovery_runtime(('连翘',))
        options = runtime.workflow.recovery_options()
        self.assertTrue(any(o['feature_id'] == 'fiber'
                            and {'candidate_id': '连翘', 'role': 'required'} in o['criteria']
                            for o in options))
        self.assertNotIn('needle_crystal', {o['feature_id'] for o in options})

    def test_generic_roles_can_limit_supporting_without_needle_fallback(self):
        runtime = self._generic_recovery_runtime(fast_recovery_roles=['supporting'])
        features = {t['feature_id'] for t in runtime.workflow.targets().values()
                    if t.get('kind') == 'feature_scan'}
        self.assertEqual(features, {'starch_grain', 'vessel', 'fiber'})

    def test_generic_no_candidate_does_not_invent_feature(self):
        runtime = self._generic_recovery_runtime()
        runtime.a = {}
        self.assertEqual(runtime.workflow.recovery_options(), [])
        self.assertFalse(any(t.get('kind') == 'feature_scan'
                             for t in runtime.workflow.targets().values()))

    def test_generic_paused_candidate_removes_exclusive_needle(self):
        runtime = self._generic_recovery_runtime(('白芍', '知母'))
        interest = runtime.candidate_interest()
        with patch.object(runtime, 'candidate_interest',
                          return_value={**interest, 'ids': ['白芍'], 'inactive_ids': ['知母']}):
            self.assertNotIn('needle_crystal',
                             {o['feature_id'] for o in runtime.workflow.recovery_options()})

    def test_generic_spend_caps_count_pending_and_failed_attempts(self):
        runtime = self._generic_recovery_runtime()
        workflow = runtime.workflow
        image = next(iter(runtime.images))
        prefix = 'feature_scan:' + image + ':'
        entries = {'a': {'tool': 'feature_scan', 'target_id': prefix + 'fiber',
                         'status': 'failed', 'scan_tiles_reserved': 4},
                   'b': {'tool': 'feature_scan', 'target_id': prefix + 'vessel',
                         'status': 'pending', 'scan_tiles_reserved': 4}}
        write_json(workflow.operations, {'entries': entries})
        scans = [t for t in workflow.targets().values() if t.get('kind') == 'feature_scan']
        self.assertEqual({t['feature_id'] for t in scans}, {'cluster_crystal', 'starch_grain'})
        self.assertTrue(all(t['max_tiles'] == 4 for t in scans))
        entries['c'] = {'tool': 'feature_scan', 'target_id': prefix + 'cluster_crystal',
                        'status': 'done', 'scan_tiles_reserved': 4}
        write_json(workflow.operations, {'entries': entries})
        self.assertFalse(any(t.get('kind') == 'feature_scan'
                             for t in workflow.targets().values()))

    def test_generic_reserves_only_remaining_locate_budget(self):
        runtime = self._generic_recovery_runtime()
        workflow = runtime.workflow
        workflow.config['fast_fallback_max_locate_calls'] = 5
        image = next(iter(runtime.images))
        write_json(workflow.operations, {'entries': {'a': {
            'tool': 'feature_scan', 'target_id': 'feature_scan:' + image + ':fiber',
            'status': 'done', 'scan_tiles_reserved': 4}}})
        scans = [t for t in workflow.targets().values() if t.get('kind') == 'feature_scan']
        self.assertTrue(scans)
        self.assertTrue(all(t['max_tiles'] == 1 for t in scans))

    def test_generic_feature_backend_forwards_requested_id_and_hint(self):
        from PIL import Image
        from TCM_Fusion.lib.workflow.workflow_backend import WorkflowBackend
        runtime = self._generic_recovery_runtime()
        Image.new('RGB', (300, 300), 'white').save(self.image)
        workflow = runtime.workflow
        workflow.config['feature_scan_tile_policy'] = 'full'
        seen = []
        def worker(cmd, **kwargs):
            ctx = read_json(cmd[cmd.index('--context') + 1])
            seen.append(ctx)
            (Path(ctx['output']) / 'crops').mkdir(parents=True)
        with patch('TCM_Fusion.lib.workflow.workflow_backend.run', side_effect=worker):
            for fid in ('fiber', 'vessel', 'stone_cell', 'starch_grain', 'needle_crystal'):
                target = {'source': str(self.image), 'feature_id': fid, 'max_tiles': 1}
                result = WorkflowBackend().execute(workflow, 'feature_scan', target,
                                                    self.root / ('scan_' + fid), None)
                self.assertEqual(read_json(result['completion'])['feature_id'], fid)
                self.assertEqual(seen[-1]['feature_id'], fid)
                self.assertTrue(seen[-1]['feature_hint'])
                if fid != 'needle_crystal':
                    self.assertNotIn('针晶可散在', seen[-1]['feature_hint'])

    def test_generic_unknown_feature_is_rejected_before_image_work(self):
        from TCM_Fusion.lib.workflow.workflow_backend import WorkflowBackend
        with self.assertRaises(IntegrationError) as raised:
            WorkflowBackend().execute(self.runtime().workflow, 'feature_scan',
                                      {'feature_id': 'not_supported'}, self.root / 'bad_scan', None)
        self.assertEqual(raised.exception.code, 'FEATURE_SCAN_INPUT')
        self.assertFalse((self.root / 'bad_scan').exists())

    def test_generic_selected_feature_is_executed_and_empty_is_not_evidence(self):
        runtime = self._generic_recovery_runtime()
        workflow = runtime.workflow
        selected = []
        class EmptyScan:
            def execute(self, session, tool, target, work, context):
                self_tool = tool
                if self_tool != 'feature_scan':
                    raise AssertionError(self_tool)
                selected.append(target['feature_id'])
                folder = work / 'scan/crops'
                folder.mkdir(parents=True)
                completion = work / 'scan/scan_result.json'
                write_json(completion, {'feature_id': target['feature_id'], 'count': 0})
                return {'crops_dir': str(folder), 'completion': str(completion)}
        workflow.backend = EmptyScan()
        for fid in ('vessel', 'fiber'):
            action = next(a for a in workflow.propose() if a['tool'] == 'feature_scan'
                          and a['args']['target_id'].endswith(':' + fid))
            result = workflow.execute(action)
            self.assertEqual(result['status'], 'done')
            self.assertEqual(result['feature_id'], fid)
            self.assertEqual(result['scan_tiles_reserved'], 4)
        self.assertEqual(selected, ['vessel', 'fiber'])
        artifacts = workflow.store.current()['artifacts']
        image = next(iter(runtime.images))
        self.assertTrue(artifacts['scanned:' + image + ':vessel']['valid'])
        self.assertTrue(artifacts['scanned:' + image + ':fiber']['valid'])
        self.assertEqual(workflow.store.verified_records(), [])
        self.assertFalse(workflow.coverage()['required_all'])

    def test_generic_rejects_result_for_a_different_feature(self):
        runtime = self._generic_recovery_runtime()
        workflow = runtime.workflow
        target = next(t for t in workflow.targets().values()
                      if t.get('kind') == 'feature_scan' and t['feature_id'] == 'vessel')
        folder = self.root / 'wrong_feature/crops'
        folder.mkdir(parents=True)
        completion = folder.parent / 'scan_result.json'
        write_json(completion, {'feature_id': 'needle_crystal', 'count': 0})
        before = workflow.store.current()['version']
        with self.assertRaises(IntegrationError) as raised:
            workflow.accept('feature_scan', target,
                            {'crops_dir': str(folder), 'completion': str(completion)}, before)
        self.assertEqual(raised.exception.code, 'FEATURE_SCAN_SCOPE')
        self.assertEqual(workflow.store.current()['version'], before)

    def test_generic_classification_cap_does_not_make_recovery_unreachable(self):
        runtime = self._generic_recovery_runtime()
        workflow = runtime.workflow
        snapshot = workflow.store.current()
        image = next(iter(runtime.images))
        arts = snapshot['artifacts']
        for i in range(4):
            cid = 'crop:ordinary:' + str(i)
            arts[cid] = {'valid': True, 'kind': 'crop', 'image_id': image}
            if i < 3:
                arts['classification:' + cid] = {
                    'valid': True, 'kind': 'classification', 'image_id': image,
                    'feature_id': 'cluster_crystal'}
        with patch.object(workflow.store, 'current', return_value=snapshot):
            self.assertTrue(workflow.fast_recovery_targets(arts))
            arts['crop:ordinary:3']['identity_required'] = True
            self.assertEqual(workflow.fast_recovery_targets(arts), {})
            arts['crop:ordinary:3'].pop('identity_required')
            arts['classification:crop:ordinary:2']['valid'] = False
            self.assertEqual(workflow.fast_recovery_targets(arts), {})
            arts['classification:crop:ordinary:2']['valid'] = True
            arts['crop:ordinary:0']['identity_required'] = True
            # Match propose()'s existing per-view counter, which also includes
            # already-classified identity_required crops; unresolved ones still block.
            self.assertTrue(workflow.fast_recovery_targets(arts))

    def test_classify_crops_batch_preserves_success_when_one_item_fails(self):
        original = WorkflowBackend.execute
        def routed(backend, session, tool, target, work, context):
            if tool == 'classify_target':
                if target['id'] == 'bad':
                    raise IntegrationError('CLASSIFY_FAILED', 'simulated one-crop failure')
                return {'classification': 'ok'}
            return original(backend, session, tool, target, work, context)
        with tempfile.TemporaryDirectory() as tmp, patch.object(WorkflowBackend, 'execute', routed):
            session = type('Session', (), {'python': 'python'})()
            result = WorkflowBackend().execute(
                session, 'classify_crops_batch',
                {'crops': [{'id': 'ok'}, {'id': 'bad'}]}, Path(tmp), {})
        self.assertEqual([row['crop_id'] for row in result['classifications']], ['ok'])
        self.assertEqual([row['crop_id'] for row in result['failures']], ['bad'])
        self.assertEqual(result['failures'][0]['error_code'], 'CLASSIFY_FAILED')

    def test_kill_tree_is_bounded_when_child_ignores_termination(self):
        child = subprocess.Popen(['python', '-c', 'import time; time.sleep(30)'])
        started = time.monotonic()
        kill_tree(child)
        self.assertLess(time.monotonic() - started, 10)
        self.assertIsNotNone(child.poll())

    def test_context_action_menu_is_bounded_and_keeps_stops(self):
        actions=[{'action_id':f'a{i}','tool':'feature_scan' if i else 'detect_fov_initial',
                  'args':{'target_id':str(i)}} for i in range(80)]
        actions += [{'action_id':'finish','tool':'finish'},{'action_id':'review','tool':'request_review'}]
        visible,meta=compact_action_menu(actions,20)
        self.assertLessEqual(len(visible),22)
        self.assertEqual({a['tool'] for a in visible if a['tool'] in {'finish','request_review'}},
                         {'finish','request_review'})
        self.assertEqual(meta['total'],82)
        self.assertGreater(meta['omitted'],0)

    def test_context_feedback_deduplicates_same_problem(self):
        events=[{'feedback':{'problem_id':'x','failure_code':None,'new_features':[],
                             'new_counterevidence':[]}} for _ in range(6)]
        self.assertEqual(len(compact_feedback(events)),1)

    def test_classify_crops_batch_is_declared_in_workflow_repair_skill(self):
        registry = json.loads((Path(paths.PROJECT_ROOT) / 'TCM_Fusion' / 'skills' / 'registry.json')
                              .read_text(encoding='utf-8'))
        skill = next(row for row in registry if row['id'] == 'workflow_repair')
        self.assertIn('classify_crops_batch', skill['tools'])

    def test_action_timeline_records_offer_select_finish(self):
        runtime = self.runtime()
        action = next(a for a in runtime.legal_actions({'observations': []})
                      if a['tool'] == 'detect_fov_initial')
        self.assertTrue(runtime.tools.invoke(runtime, action, {'action': action})['accepted'])
        path = runtime.workflow.timeline_path
        self.assertTrue(path.is_file())
        doc = json.loads(path.read_text(encoding='utf-8'))
        row = next(v for v in doc['actions'].values()
                   if v.get('tool') == action['tool'] and v.get('target_id') == action['args']['target_id'])
        self.assertIsNotNone(row['first_offered_at'])
        self.assertIsNotNone(row['selected_at'])
        self.assertIsNotNone(row['started_at'])
        self.assertIsNotNone(row['ended_at'])
        self.assertEqual(row['status'], 'done')
        self.assertGreaterEqual(row['offer_to_select_s'], 0)
        self.assertGreaterEqual(row['execution_s'], 0)

    def test_fast_required_coverage_combines_views(self):
        second = self.root / 'second.png'
        second.write_bytes(b'second-view')
        runtime = WorkflowRuntime(
            [self.image, second], root=self.root / 'multiview_task',
            sample_id='sample', a={}, simulation=True, backend=StageFixture(),
            workflow_config={'fast_yolo_stage2': True, 'hybrid_scan_enabled': True,
                             'feature_scan_features': ['fiber']},
        )
        workflow = runtime.workflow
        first_id, second_id = runtime.images
        snapshot = workflow.store.current()
        artifacts = snapshot['artifacts']
        for image_id in runtime.images:
            artifacts['detected:' + image_id] = {
                'valid': True, 'kind': 'detection', 'image_id': image_id,
            }
        artifacts['classification:first'] = {
            'valid': True, 'kind': 'classification', 'image_id': first_id,
            'feature_id': 'fiber',
        }
        artifacts['classification:second'] = {
            'valid': False, 'kind': 'classification', 'image_id': second_id,
            'feature_id': 'stone_cell',
        }
        with patch.object(workflow.store, 'current', return_value=snapshot), \
                patch.object(workflow, 'recovery_features', return_value=['fiber', 'stone_cell']):
            self.assertFalse(workflow.coverage()['required_all'])
            for image_id in runtime.images:
                self.assertEqual(workflow.targets()['hybrid_scan:' + image_id]['features'],
                                 ['stone_cell'])

            artifacts['classification:second']['valid'] = True
            self.assertTrue(workflow.coverage()['required_all'])
            self.assertEqual(workflow.coverage()['missing_required'], [])
            self.assertFalse(any(key.startswith('hybrid_scan:') for key in workflow.targets()))

    def test_nested_jsonschema_and_external_ref(self):
        schema={'type':'object','required':['x'],'additionalProperties':False,'properties':{
            'x':{'type':'array','items':{'type':'integer','minimum':1},'minItems':2}}}
        validate_arguments(schema,{'x':[1,2]})
        for obj in ({'x':[0,2]},{'x':[True,2]},{'x':[1]}, {'x':[1,2],'shell':'bad'}):
            with self.assertRaises(IntegrationError):validate_arguments(schema,obj)
        with self.assertRaises(IntegrationError):validate_arguments({'$ref':'https://example.com/schema'}, {})

    def test_valid_absence_is_observed_but_failure_stays_unknown(self):
        r=self.runtime();image=next(iter(r.images))
        obs={'accepted':True,'action':{'tool':'presence_check','args':{
            'image_id':image,'feature_id':'epidermis_cell'}},
            'result':{'ok':True,'present':False,'conf':.9}}
        scored=r.score([obs])
        self.assertIn((image,'epidermis_cell'),r._observed_fovs)
        self.assertEqual(scored['observation_states'][0]['state'],'absent')
        self.assertNotIn('epidermis_cell',r.present_features)
        r.score([{**obs,'accepted':False}])
        self.assertEqual(r._observed_features,set())

    def test_model_investigation_extends_tools_without_adding_evidence(self):
        r=self.runtime(targeted_recovery=True)
        r.a={'status':'ok','fold_rankings':[[{'label':'秦皮','probability':1.0}]]*5}
        before=r.workflow.store.current()['version']
        self.assertNotIn('龙胆',r.candidate_interest()['ids'])
        r.investigate(['龙胆'],'检查A候选外针晶假设',r.score([])['evidence_digest'])
        self.assertIn('龙胆',r.candidate_interest()['ids'])
        self.assertIn('needle_crystal',r.workflow.recovery_features())
        self.assertEqual(r.workflow.store.current()['version'],before)
        self.assertFalse(r.present_features)
        self.assertFalse(read_json(r.workflow.root/'investigation.json')['is_evidence'])
        self.assertIn('龙胆',self.runtime(targeted_recovery=True).candidate_interest()['ids'])
        with self.assertRaises(IntegrationError):r.investigate(['arbitrary_shell'],'bad','hash')
        r.workflow.close('test')
        with self.assertRaises(IntegrationError):r.investigate(['桔梗'],'late','hash')

    def test_admission_does_not_call_model_for_insufficient_evidence(self):
        r=self.runtime()
        with patch.object(r,'_decision_client',side_effect=AssertionError('must not call')):
            result=r.admission_decision({'verdict':'insufficient_evidence','needs_review':True})
        self.assertEqual(result['decision'],'verify')

    def test_hypothesis_pause_and_resume_does_not_change_evidence_or_a(self):
        r=self.runtime(agent_policy='adaptive_v2',targeted_recovery=True)
        r.a={'status':'ok','fold_rankings':[[{'label':'秦皮','probability':1.0}]]*5}
        original_a=copy.deepcopy(r.a);before=r.score([])
        r.investigate(['G1'],'新假设',before['evidence_digest'])
        version=r.workflow.store.current()['version']
        result=r.update_hypotheses([
            {'candidate_id':'G1','status':'paused','reason':'定位没有支持'},
            {'candidate_id':'秦皮','status':'withdrawn','reason':'暂时离开A先验方向'}],before['evidence_digest'])
        self.assertTrue(result['scope_changed'])
        self.assertNotIn('G1',r.candidate_interest()['ids'])
        self.assertNotIn('秦皮',r.candidate_interest()['ids'])
        self.assertEqual(r.a,original_a)
        self.assertEqual(r.score([])['decision'],before['decision'])
        self.assertEqual(r.workflow.store.current()['version'],version)
        r.update_hypotheses([{'candidate_id':'G1','status':'investigating','reason':'新观察值得重查'}],before['evidence_digest'])
        self.assertIn('G1',r.candidate_interest()['ids'])
        # A separately constructed MCP-side runtime consumes the same lifecycle.
        other=self.runtime(agent_policy='adaptive_v2',targeted_recovery=True);other.a=original_a
        self.assertIn('G1',other.candidate_interest()['ids'])
        self.assertNotIn('秦皮',other.candidate_interest()['ids'])

    def test_invalid_hypothesis_update_is_atomic(self):
        r=self.runtime();d=r.score([])['evidence_digest'];r.investigate(['G1'],'initial',d)
        before=file_hash(r.workflow.root/'investigation.json')
        with self.assertRaises(IntegrationError):
            r.update_hypotheses([{'candidate_id':'G1','status':'paused','reason':'wait'},
                                 {'candidate_id':'unknown','status':'investigating','reason':'bad'}],d)
        self.assertEqual(file_hash(r.workflow.root/'investigation.json'),before)

    def test_calibration_changes_read_revision_even_without_rule_evidence_change(self):
        r=self.runtime(backend=StageFixture(scale=None),agent_policy='adaptive_v2')
        r.workflow.execute(next(a for a in r.workflow.propose() if a['tool']=='detect_fov_initial'))
        before=r.score([]);state={'observations':[],'evidence_digest':before['evidence_digest']}
        revision=r.read_revision(state)
        self.assertEqual(r.tools.read(r,'read_scale',{},state)['scales'],[])
        r.workflow.execute(next(a for a in r.workflow.propose() if a['tool']=='calibrate_fov'))
        after=r.score([])
        self.assertEqual(after['evidence_digest'],before['evidence_digest'])
        self.assertNotEqual(r.read_revision(state),revision)
        current=r.summary(state)['workflow']['calibration']
        self.assertTrue(any(row.get('valid') and row.get('px_per_um')==2 for row in current['scales']))

    def test_paused_exclusive_crop_branch_is_hidden_but_shared_feature_survives(self):
        r=self.runtime(agent_policy='adaptive_v2',targeted_recovery=True)
        r.a={'status':'ok','fold_rankings':[[{'label':'秦皮','probability':1.0}]]*5}
        image=next(iter(r.images));d=r.score([])['evidence_digest'];r.investigate(['G1'],'probe',d)
        side=self.root/'cluster.json'
        write_json(side,contracts.crop_sidecar('cluster',str(self.image),str(self.image),self.image.stem,
                   'cluster_crystal','草酸钙簇晶','',2,[0,0,8,8],[0,0,8,8],[0,0,8,8],'8x8'))
        art=r.workflow._art('crop:probe','crop',side,[image],image_id=image,identity_required=True)
        r.workflow.store.commit(r.workflow.store.current()['version'],[art])
        def offers():return [a for a in r.workflow.propose() if a['tool']=='classify_target' and a['args']['target_id']=='crop:probe']
        self.assertTrue(offers())
        r.update_hypotheses([{'candidate_id':'G1','status':'paused','reason':'no gain'}],d)
        self.assertFalse(offers())
        shared=next(row['id'] for row in r.kb.candidates() if row['id']!='G1'
                    and any(c['feature']=='cluster_crystal' for c in row['criteria']))
        r.investigate([shared],'shared feature still matters',d)
        self.assertTrue(offers())

    def test_review_component_plan_is_bound_to_one_segmentation(self):
        r=self.runtime()
        r.workflow.execute(next(a for a in r.workflow.propose() if a['tool']=='detect_fov_initial'))
        r.workflow.execute(next(a for a in r.workflow.propose() if a['tool']=='segment_instance'))
        target=next(t for t in r.workflow.targets().values() if t['kind']=='segmentation')
        focus={'status':'crop_segmentation_review','segmentation_id':target['id'],
               'review':{'components':[{'id':1}]},
               'review_decision':{'repair':'diagnostic_measure','component_ids':[1]}}
        with patch.object(r,'p0_focus',return_value=focus):
            self.assertEqual(r.workflow.review_measure_plan(target)['target_sha256'],target['sha256'])
            self.assertIsNone(r.workflow.review_measure_plan({**target,'id':'different_mask'}))
            self.assertIsNone(r.workflow.review_measure_plan({**target,'sha256':'changed'}))

    def test_decision_model_shares_budget_and_does_not_call_after_close(self):
        from types import SimpleNamespace
        from TCM_Fusion.lib.decision.decision_llm import DecisionLLM
        r=self.runtime(max_planning=2,max_api_calls=2)
        llm=DecisionLLM.__new__(DecisionLLM)
        llm.cache={};llm.workflow=r.workflow;llm.config={}
        llm.client=SimpleNamespace(timeout_s=60,model='fixture')
        body={'choices':[{'message':{'content':'{"decision":"verify","reason":"missing"}'}}],
              'usage':{'total_tokens':10}}
        with patch('TCM_Fusion.lib.runtime.hard_calls.post',return_value=body) as call:
            result=llm.admission({'id':'x'},{},[])
            self.assertEqual(result['decision'],'verify')
            self.assertEqual(call.call_count,1)
            self.assertEqual(read_json(r.workflow.budget_path)['used']['planning'],1)
            self.assertTrue(read_json(r.workflow.root/'decision_calls.json')['calls'])
            r.workflow.close('test')
            result=llm.admission({'id':'different'},{},[])
            self.assertFalse(result['available'])
            self.assertEqual(call.call_count,1)

    def test_relaxed_support_policy_is_explicit_and_does_not_mutate_baseline(self):
        from rule_engine import MIN_SUPPORT_FEATURES,decide,pool_evidence
        self.assertEqual(MIN_SUPPORT_FEATURES,2)
        records=[{'feature':'cluster_crystal','image_stem':'fov','instance_id':1,
                  'calibration_status':'ok','calibration_px_per_um':2,'conf':.9,
                  'primary_dimension':'feret_max_um','primary_value_um':18}]
        cards=pool_evidence(records);r=self.runtime()
        baseline=decide(r.kb,cards)
        decide(r.kb,cards,min_support_features=1)
        self.assertEqual(decide(r.kb,cards),baseline)

    def test_independent_handler_precondition_and_output(self):
        r=self.runtime();reg=r.tools
        reg.register_handler('trusted_test',lambda r,a,s:{'value':a['n']*2})
        reg.register_precondition('positive_test',lambda r,a,s:a['n']>0)
        reg.register(ToolSpec('trusted_test','read','1',{'type':'object','required':['n'],'properties':{'n':{'type':'integer'}}},
            {'type':'object','required':['value'],'properties':{'value':{'type':'integer'}}},
            ('evidence_inspection',),'read',False,'trusted_test','positive_test'))
        for row in r.skills.registry:
            if row['id']=='evidence_inspection':row['tools'].append('trusted_test')
        self.assertEqual(reg.read(r,'trusted_test',{'n':2},{}),{'value':4})
        with self.assertRaises(IntegrationError):reg.read(r,'trusted_test',{'n':-1},{})
        with self.assertRaises(IntegrationError):reg.register(ToolSpec('unbound','read','1',{}, {},(), 'read',False,'arbitrary.import'))

    def test_registry_proposer_is_not_hardcoded(self):
        r=self.runtime();r.tools.register_proposer('test_proposals',lambda r,s:[r.action('trusted_query',{},'evidence_inspection')])
        r.tools.register_handler('trusted_query',lambda r,a,s:{})
        r.tools.register(ToolSpec('trusted_query','read','1',{'type':'object'}, {'type':'object'},
            ('evidence_inspection',),'read',False,'trusted_query',proposal_id='test_proposals'))
        for row in r.skills.registry:
            if row['id']=='evidence_inspection':row['tools'].append('trusted_query')
        self.assertTrue(any(a['tool']=='trusted_query' for a in r.tools.propose(r,{})))

    def test_raw_images_bootstrap_actual_langgraph(self):
        backend=StageFixture();r=self.runtime(backend);tools=LocalToolbox(r,ledger_path=self.root/'observations.json',backend=FixtureBackend())
        result=asyncio.run(run_react(r,tools,planner_mode='heuristic',max_requests=12,max_steps=12))
        self.assertEqual(backend.calls[:4],['detect_fov_initial','classify_target','segment_instance','measure_instances'])
        self.assertTrue(r.workflow.store.current()['records']);self.assertTrue(any(e['phase']=='recompute' for e in result['events']))
        self.assertEqual(self.image.read_bytes(),b'simulation-fov')

    def test_label_conflict_offers_classification_and_relabels_crop(self):
        """观察到fiber与crop类别single_crystal相左：应提供复核，并把结论落成可重分割的新crop。"""
        backend=StageFixture(classification_feature='fiber');r=self.runtime(backend)
        detect=next(a for a in r.workflow.propose() if a['tool']=='detect_fov_initial')
        r.workflow.execute(detect)
        image_id=next(iter(r.images))
        r.score([{'accepted':True,'action':{'tool':'presence_check',
                                            'args':{'image_id':image_id,'feature_id':'fiber'}},
                  'result':{'present':True,'conf':0.9,'reason':'模拟观察'}}])
        offered=[a for a in r.workflow.propose() if a['tool']=='classify_target']
        self.assertEqual(len(offered),1,'类别冲突应恰好提供一次复核动作')
        record=r.workflow.execute(offered[0])
        self.assertEqual(record['status'],'done')
        arts=r.workflow.store.current()['artifacts']
        relabeled=[k for k,v in arts.items() if v['valid'] and v['kind']=='crop' and '__as_fiber' in k]
        self.assertEqual(len(relabeled),1,'复核出不同类别应产出可重分割的新crop')
        meta=contracts.CropMeta.from_json(arts[relabeled[0]]['path'])
        original=contracts.CropMeta.from_json(arts['crop:'+image_id+':crop']['path'])
        self.assertEqual((meta.feature_id,meta.object_label),('fiber','纤维'))
        self.assertEqual(Path(meta.png).resolve(),Path(original.png).resolve(),'只改类别，不复制或改写像素')
        self.assertTrue(any(a['tool']=='segment_instance' and a['args']['target_id']==relabeled[0]
                            for a in r.workflow.propose()),'新crop应被提议重新分割')
        self.assertEqual(backend.calls.count('classify_target'),1)

    def test_missing_scale_is_repaired_before_segmentation(self):
        backend=StageFixture(scale=None);r=self.runtime(backend)
        tools=LocalToolbox(r,ledger_path=self.root/'observations.json',backend=FixtureBackend())
        asyncio.run(run_react(r,tools,planner_mode='heuristic',max_requests=12,max_steps=12))
        self.assertEqual(backend.calls[:5],['detect_fov_initial','calibrate_fov','classify_target','segment_instance','measure_instances'])

    def test_segment_failure_is_visible_and_not_retried(self):
        backend=StageFixture(fail='segment_instance');r=self.runtime(backend)
        tools=LocalToolbox(r,ledger_path=self.root/'observations.json',backend=FixtureBackend())
        result=asyncio.run(run_react(r,tools,planner_mode='heuristic',max_steps=10))
        self.assertTrue(result['needs_review']);self.assertEqual(backend.calls.count('segment_instance'),1)
        self.assertEqual(r.workflow.store.current()['records'],[])
        self.assertTrue(any(v.get('error_code')=='SIMULATED_FAILURE' for v in read_json(r.workflow.operations)['entries'].values()))

    def test_service_404_blocks_segmentation_without_spatial_repair(self):
        class ServiceFailure(StageFixture):
            def execute(self,s,tool,target,work,context):
                if tool=='segment_instance':
                    self.calls.append(tool);raise IntegrationError('UPSTREAM_HTTP_404','simulated service failure')
                return super().execute(s,tool,target,work,context)
        backend=ServiceFailure();r=self.runtime(backend)
        tools=LocalToolbox(r,ledger_path=self.root/'service_obs.json',backend=FixtureBackend())
        trace=asyncio.run(run_react(r,tools,planner_mode='heuristic',max_steps=10))
        self.assertEqual(backend.calls,['detect_fov_initial','classify_target','segment_instance'])
        # 当前快路径多出分类观察；无增益优先收尾，但服务熔断仍须独立成立。
        self.assertEqual(trace['stop_reason'],'no_gain_twice');self.assertTrue(trace['needs_review'])
        self.assertEqual(r.workflow.blocked_tools(),{'segment_instance':'UPSTREAM_HTTP_404'})
        self.assertFalse(any(k.startswith('roi:auto:') for k in r.workflow.targets()))
        self.assertEqual(r.summary({'observations':[]})['workflow']['blocked_tools'],r.workflow.blocked_tools())
        # 新目标、新证据版本也不能越过同一任务内的服务熔断。
        crop=next(v for v in r.workflow.targets().values() if v.get('kind')=='crop')
        raw=read_json(crop['path']);raw.update(crop_id='second',png=str(Path(crop['path']).parent/'crop.png'))
        file=self.root/'second.json';write_json(file,raw)
        cid='crop:'+crop['image_id']+':second'
        art=r.workflow._art(cid,'crop',file,crop['parents'],image_id=crop['image_id'])
        snap=r.workflow.store.commit(r.workflow.store.current()['version'],[art])
        action=r.action('segment_instance',{'target_id':cid,'evidence_version':snap['version']},'workflow_repair')
        with self.assertRaises(IntegrationError) as caught:r.tools.invoke(r,action)
        self.assertEqual(caught.exception.code,'TOOL_SERVICE_BLOCKED')
        self.assertEqual(backend.calls.count('segment_instance'),1)
        self.assertFalse(any(a['tool']=='segment_instance' for a in r.workflow.propose()))
        fresh=WorkflowRuntime([self.image],root=self.root/'fresh',workflow_config={},sample_id='fresh',
                              simulation=True,backend=StageFixture())
        self.assertEqual(fresh.workflow.blocked_tools(),{})

    def test_empty_mask_still_proposes_local_image_repair(self):
        class EmptyMask(StageFixture):
            def execute(self,s,tool,target,work,context):
                if tool=='segment_instance':raise IntegrationError('EMPTY_MASK','simulated empty mask')
                return super().execute(s,tool,target,work,context)
        r=self.runtime(EmptyMask())
        r.tools.invoke(r,next(a for a in r.workflow.propose() if a['tool']=='detect_fov_initial'))
        r.tools.invoke(r,next(a for a in r.workflow.propose() if a['tool']=='segment_instance'))
        self.assertEqual(r.workflow.blocked_tools(),{})
        self.assertTrue(any(a['tool']=='detect_region' for a in r.workflow.propose()))

    def test_unknown_worker_failure_is_not_guessed_as_a_bad_region(self):
        class UnknownFailure(StageFixture):
            def execute(self,s,tool,target,work,context):
                if tool=='segment_instance':raise IntegrationError('WORKER_FAILED','unknown failure')
                return super().execute(s,tool,target,work,context)
        r=self.runtime(UnknownFailure())
        r.tools.invoke(r,r.workflow.propose()[0]);r.tools.invoke(r,r.workflow.propose()[0])
        self.assertEqual(r.workflow.blocked_tools(),{})
        self.assertFalse(any(k.startswith('roi:auto:') for k in r.workflow.targets()))
        self.assertTrue(r.workflow.unresolved())

    def test_review_mask_keeps_image_repair_available(self):
        class ReviewMask(StageFixture):
            def execute(self,s,tool,target,work,context):
                value=super().execute(s,tool,target,work,context)
                if tool=='segment_instance':value['quality']='review'
                return value
        r=self.runtime(ReviewMask())
        r.tools.invoke(r,next(a for a in r.workflow.propose() if a['tool']=='detect_fov_initial'))
        result=r.tools.invoke(r,next(a for a in r.workflow.propose() if a['tool']=='segment_instance'))
        self.assertTrue(result['accepted']);self.assertFalse(result['repair_completed'])
        self.assertTrue(any(a['tool']=='detect_region' for a in r.workflow.propose()))

    def test_recompute_budget_zero_never_calls_backend(self):
        backend=StageFixture();r=self.runtime(backend,max_recompute=0,max_initializations=0)
        tools=LocalToolbox(r,ledger_path=self.root/'observations.json',backend=FixtureBackend())
        result=asyncio.run(run_react(r,tools,planner_mode='heuristic',max_steps=3))
        self.assertEqual(backend.calls,[]);self.assertTrue(result['needs_review'])

    def test_operation_cache_does_not_execute_twice(self):
        backend=StageFixture();r=self.runtime(backend);action=r.workflow.propose()[0]
        first=r.tools.invoke(r,action);second=r.tools.invoke(r,action)
        self.assertTrue(first['accepted']);self.assertTrue(second['cache_hit']);self.assertEqual(len(backend.calls),1)

    def test_recorded_workflow_replay_has_no_new_image_calls(self):
        from TCM_Fusion.lib.offline.workflow_replay import replay_report,RecordedStages
        r=self.runtime();tools=LocalToolbox(r,ledger_path=self.root/'obs.json',backend=FixtureBackend(),max_requests=100)
        trace=asyncio.run(run_react(r,tools,planner_mode='heuristic',max_requests=100,max_steps=100))
        operations=read_json(r.workflow.operations)['entries']
        report={'route':'workflow_agent','a':{},'sample_id':'sample','workflow_options':r.workflow.config,
            'agent':{'operations':operations,'observations':trace['observations']},
            'b':{'verdict':{**trace['decision'],'source_images':[str(self.image)]}}}
        value=asyncio.run(replay_report(report,self.root/'replay'))
        self.assertTrue(value['covered']);self.assertTrue(value['matches_source_verdict'])
        self.assertEqual(value['new_image_api_calls'],0)
        missing=RecordedStages([])
        with self.assertRaises(IntegrationError):missing.execute(r.workflow,'detect_fov_initial',
            next(iter(r.workflow.targets().values())),self.root/'unused',None)
        self.assertEqual(missing.misses,1)

    def test_store_rejects_cycles_and_tampering(self):
        store=EvidenceStore(self.root/'evidence','sample');p=self.root/'p';p.write_bytes(b'data')
        with self.assertRaises(IntegrationError):store.commit(store.current()['version'],[
            {'id':'a','path':str(p),'sha256':file_hash(p),'parents':['a']}])
        store.commit(store.current()['version'],[{'id':'a','path':str(p),'sha256':file_hash(p),'parents':[]}])
        p.write_bytes(b'changed')
        with self.assertRaises(IntegrationError):store.verified_records()

    def test_failed_and_crop_index_has_no_truth(self):
        root=self.root/'b';(root/'detection/crops').mkdir(parents=True)
        write_json(root/'job.json',{'status':'failed','error_code':'EMPTY_MASK','secret':'not-for-agent'})
        write_json(root/'detection/crops/crop.json',{'source_image_path':str(self.image),'herb':'truth'})
        result=ArtifactIndex([root]).scan();text=json.dumps(result,ensure_ascii=False)
        self.assertIn('EMPTY_MASK',text);self.assertNotIn('not-for-agent',text);self.assertNotIn('truth',text)
        self.assertTrue(any(a['status']=='missing_pair' for a in result['artifacts']))

    def test_store_cas_invalidation_and_immutable_history(self):
        store=EvidenceStore(self.root/'evidence','sample');p=self.root/'p';p.write_text('first');m=self.root/'mask';m.write_text('mask')
        initial=store.current()['version'];one=store.commit(initial,[{'id':'crop','path':str(p),'sha256':file_hash(p),'parents':[]}])
        two=store.commit(one['version'],[{'id':'mask','path':str(m),'sha256':file_hash(m),'parents':['crop']}])
        new=self.root/'new';new.write_text('second')
        three=store.commit(two['version'],[{'id':'crop','path':str(new),'sha256':file_hash(new),'parents':[]}])
        self.assertFalse(three['artifacts']['mask']['valid']);self.assertTrue((store.root/'versions'/(one['version']+'.json')).is_file())
        with self.assertRaises(IntegrationError):store.commit(one['version'],[])

    def test_invalidation_traverses_resubmitted_unchanged_sidecar(self):
        store=EvidenceStore(self.root/'store','sample')
        pixel=self.root/'pixel';pixel.write_bytes(b'old')
        side=self.root/'side';side.write_bytes(b'same-sidecar')
        mask=self.root/'mask';mask.write_bytes(b'mask')
        artifacts=[{'id':'pixels','path':str(pixel),'sha256':file_hash(pixel),'parents':[]},
                   {'id':'crop','path':str(side),'sha256':file_hash(side),'parents':['pixels']},
                   {'id':'mask','path':str(mask),'sha256':file_hash(mask),'parents':['crop']}]
        first=store.commit(store.current()['version'],artifacts)
        new=self.root/'newpixel';new.write_bytes(b'changed')
        second=store.commit(first['version'],[
            {**artifacts[0],'path':str(new),'sha256':file_hash(new)},artifacts[1]])
        self.assertTrue(second['artifacts']['crop']['valid'])
        self.assertFalse(second['artifacts']['mask']['valid'])

    def test_network_guard_counts_attempt_and_blocks_retry(self):
        import urllib.request
        budget=self.root/'budget.json';ctx=self.root/'ctx.json'
        write_json(budget,{'identity':'test','deadline':10**12,'used':{},'entries':{}})
        write_json(ctx,{'identity':'test','budget_path':str(budget),'operation_key':'op','limits':{'network':2,'total':2}})
        original=urllib.request.urlopen;redirect=urllib.request.HTTPRedirectHandler.redirect_request
        try:
            with patch('urllib.request.urlopen',side_effect=OSError('network failure')) as remote:
                install_http_guard(ctx)
                with self.assertRaises(OSError):urllib.request.urlopen('https://example.com')
                with self.assertRaises(IntegrationError):urllib.request.urlopen('https://example.com')
                self.assertEqual(remote.call_count,1)
            self.assertEqual(read_json(budget)['used']['network'],1)
        finally:urllib.request.urlopen=original;urllib.request.HTTPRedirectHandler.redirect_request=redirect

    def test_real_worker_process_budget_and_timeout(self):
        from TCM_Fusion.lib.runtime.controlled_process import run
        budget=self.root/'budget.json';ctx=self.root/'ctx.json'
        write_json(budget,{'identity':'test','deadline':10**12,'used':{},'entries':{}})
        write_json(ctx,{'identity':'test','budget_path':str(budget),'operation_key':'op','limits':{'network':0,'total':0}})
        code = (
            "import urllib.request\n"
            "assert urllib.request.urlopen.__module__ == 'TCM_Fusion.lib.runtime.resource_budget'\n"
            "try:\n"
            "    urllib.request.urlopen('http://127.0.0.1:0')\n"
            "except Exception as exc:\n"
            "    assert getattr(exc, 'code', '') == 'RESOURCE_BUDGET'\n"
            "else:\n"
            "    raise AssertionError('network call should have been rejected by the budget')\n"
        )
        run([sys.executable, '-c', code], context_path=ctx, timeout=5,
            log_path=self.root/'worker.log')
        with self.assertRaises(IntegrationError):run([sys.executable,'-c','import time; time.sleep(10)'],context_path=ctx,timeout=.15,log_path=self.root/'timeout.log')

    def test_original_geometry_worker_with_partial_mask_and_real_mcp(self):
        self._geometry_reuse_case(False)

    def test_diagnostic_review_reuses_mask_and_preserves_rule_exclusion(self):
        self._geometry_reuse_case(True)

    def _geometry_reuse_case(self, diagnostic):
        """真正执行原measure.py与stdio；合成图仅用于接口/量纲验证，不联网。"""
        import subprocess
        from TCM_Fusion.lib.application.pipeline import run_pipeline
        py=Path(sys.executable)
        if not py.is_file():self.skipTest('没有可用的CV解释器')
        probe=subprocess.run([str(py),'-c','import cv2,numpy'],capture_output=True)
        if probe.returncode:self.skipTest('CV解释器缺numpy/opencv；其它回归仍运行')
        subprocess.run([str(py),'-c',
            "import cv2,numpy as np,sys; a=np.full((64,64,3),220,np.uint8); m=np.zeros((64,64),np.uint8); cv2.circle(m,(32,32),10,255,-1); cv2.imwrite(sys.argv[1],a); cv2.imwrite(sys.argv[2],m)",
            str(self.image),str(self.root/'mask.png')],check=True,capture_output=True)
        original=self.root/'original_b';crop=original/'detection/crops';crop.mkdir(parents=True)
        import shutil
        shutil.copy2(self.image,crop/'crop.png')
        side=contracts.crop_sidecar('crop','crop.png',str(self.image),'fov','single_crystal','方晶','',2,
                                   [20,20,44,44],[20,20,44,44],[0,0,64,64],'64x64')
        write_json(crop/'crop.json',side)
        masks=original/'segmentation/masks';masks.mkdir(parents=True)
        name='fov__single_crystal__inst01.png';shutil.copy2(self.root/'mask.png',masks/name)
        raw={k:'' for k in contracts.MANIFEST_COLUMNS};raw.update(instance_id=1,inst_mask=name,crop_id='crop',
             source_image_path=str(self.image),feature_id='single_crystal',object_label='方晶',
             px_per_um='2',igm_status='review' if diagnostic else 'ok',granularity='instance',tight_bbox='[20,20,44,44]')
        with (masks/('fov'+contracts.MANIFEST_SUFFIX)).open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=contracts.MANIFEST_COLUMNS);w.writeheader();w.writerow(raw)
        shutil.copytree(crop,masks/'crops')  # 原B交付的同源侧车副本不应造成工件ID冲突。
        before={str(p):file_hash(p) for p in original.rglob('*') if p.is_file()}
        config={'sample_id':'cpu_test','allow_network':False,'a':{'mode':'unavailable','required':False},
                'b':{'mode':'existing','run_dir':str(original),'enable_vlm':False},
                'agent':{'mode':'active','planner':'heuristic','transport':'mcp','image_roots':[str(self.image)],
                         'max_steps':6,'workflow':{'python':str(py),'max_api_calls':0,'max_seconds':15}}}
        if diagnostic:
            config['agent']['workflow'].update(quality_policy='diagnostic_review',enabled_tools=['measure_instances'])
        directory=run_pipeline(config,base=self.root/'runs')
        value=read_json(directory/'fusion.json');self.assertEqual(value['route'],'workflow_agent')
        self.assertTrue(any(o['tool']=='measure_instances' and o['accepted'] for o in value['agent']['operations'].values()),
                        {'stop':value['agent'].get('stop_reason'), 'operations':[
                            {k:o.get(k) for k in ('tool','status','error_code')} for o in value['agent']['operations'].values()]})
        snap=read_json(directory/'workflow/evidence/head.json')
        content=read_json(directory/'workflow/evidence/versions'/(snap['version']+'.json'))
        self.assertTrue(content['records']);self.assertEqual(content['records'][0]['data']['calibration_px_per_um'],2)
        self.assertEqual(value['agent']['resources']['used'].get('network',0),0)
        self.assertEqual(before,{str(p):file_hash(p) for p in original.rglob('*') if p.is_file()})
        if diagnostic:
            self.assertTrue(value['diagnostic_only']);self.assertTrue(value['needs_review'])
            self.assertIn('igm_review',content['records'][0]['data']['flags'])
            from rule_engine import pool_evidence
            cards=pool_evidence([r['data'] for r in content['records']])
            self.assertTrue(all(not c.present and c.n_instances==0 for c in cards.values()))
            self.assertEqual(value['agent']['resources']['used'].get('observation',0),0)

    def test_offline_missing_view_still_requires_review(self):
        r=self.runtime();self.assertTrue(r.workflow.unresolved())
        r.simulation=False;r.workflow.allow_network=False
        self.assertEqual(r.workflow.propose(),[])
        self.assertTrue(r.workflow.unresolved())

    def test_read_scale_exposes_contract_and_version(self):
        r=self.runtime();r.tools.invoke(r,r.workflow.propose()[0])
        result=r.tools.read(r,'read_scale',{}, {})
        self.assertEqual(result['scales'][0]['px_per_um'],2)
        self.assertIn('evidence_version',result['scales'][0])

    def test_initialization_and_api_pools_are_separate_but_globally_bounded(self):
        r=self.runtime(max_recompute=0,max_api_calls=1)
        self.assertTrue(r.tools.invoke(r,r.workflow.propose()[0])['accepted'])
        used=read_json(r.workflow.budget_path)['used']
        self.assertEqual(used.get('initialization'),1);self.assertNotIn('recompute',used)
        r.workflow.charge('planning','plan')
        with self.assertRaises(IntegrationError):r.workflow.charge('observation','observe')
        self.assertEqual(read_json(r.workflow.budget_path)['used']['network'],1)

    def test_shadow_execution_is_removed(self):
        from TCM_Fusion.lib.application.pipeline import validate_config
        config={'sample_id':'shadow-removed','allow_network':False,
                'a':{'mode':'unavailable','required':False},
                'b':{'mode':'run','input_mode':'images','input':str(self.root),'enable_vlm':False},
                'agent':{'mode':'active','planner':'heuristic','image_roots':[str(self.image)],
                         'workflow':{'execution':'shadow'}}}
        with self.assertRaises(IntegrationError):
            validate_config(config)

    def test_transport_failure_retains_valid_state_and_requires_review(self):
        class BrokenTransport:
            async def recompute(self,action):raise RuntimeError('simulated transport loss')
        r=self.runtime()
        result=asyncio.run(run_react(r,BrokenTransport(),planner_mode='heuristic',max_steps=3))
        self.assertTrue(result['needs_review']);self.assertEqual(r.records,[])
        self.assertEqual(result['events'][-1]['error_code'],'TOOL_TRANSPORT_FAILED')

    def test_measure_cache_and_replay_bind_transitive_mask_content(self):
        r=self.runtime()
        for tool in ('detect_fov_initial','segment_instance'):
            r.tools.invoke(r,next(a for a in r.workflow.propose() if a['tool']==tool))
        target=next(v for v in r.workflow.targets().values() if v.get('kind')=='segmentation')
        before=r.workflow.operation_key('measure_instances',target)
        original=r.tools.invoke(r,next(a for a in r.workflow.propose() if a['tool']=='measure_instances'))
        from TCM_Fusion.lib.offline.workflow_replay import RecordedStages
        recorded=RecordedStages([original])
        snap=r.workflow.store.current();mask=next(v for v in snap['artifacts'].values()
                                                if v['kind']=='segmentation_file' and v['path'].endswith('.png'))
        new=self.root/'changed-mask.png';new.write_bytes(b'new-mask-content')
        replacement={k:v for k,v in mask.items() if k!='valid'}
        replacement.update(path=str(new),sha256=file_hash(new))
        marker={k:v for k,v in snap['artifacts'][target['id']].items() if k!='valid'}
        r.workflow.store.commit(snap['version'],[replacement,marker])
        current=r.workflow.targets()[target['id']]
        self.assertNotEqual(before,r.workflow.operation_key('measure_instances',current))
        self.assertEqual(r.workflow.store.verified_records(),[])
        with self.assertRaises(IntegrationError):recorded.execute(r.workflow,'measure_instances',current,self.root/'unused',None)
        self.assertEqual(recorded.misses,1)


    def test_automatic_measurement_uses_real_context_file(self):
        from TCM_Fusion.lib.runtime.resource_budget import operation_failure
        class ContextChecked(StageFixture):
            def execute(inner,s,tool,target,work,context):
                if tool == 'measure_instances':
                    self.assertIsInstance(context,Path)
                    doc=read_json(context)
                    self.assertEqual(doc['budget_path'],str(s.budget_path))
                    self.assertTrue(doc['operation_key'].startswith('auto:'))
                    self.assertIsNone(operation_failure(context))
                return super().execute(s,tool,target,work,context)
        r=self.runtime(backend=ContextChecked(),agent_policy='adaptive_v2',auto_measure_after_segment=True)
        for tool in ['detect_fov_initial','classify_target','segment_instance']:
            r.workflow.execute(next(a for a in r.workflow.propose() if a['tool']==tool))
        ops=list(read_json(r.workflow.operations)['entries'].values())
        auto=next(o for o in ops if o.get('automatic'))
        self.assertEqual(auto['status'],'done');self.assertTrue(auto['accepted'])
        self.assertEqual(len(r.workflow.store.verified_records()),1)

    def test_automatic_measurement_real_original_geometry_worker(self):
        import cv2
        import numpy as np
        from TCM_Fusion.lib.workflow.workflow_backend import WorkflowBackend
        cv2.imwrite(str(self.image),np.full((64,64,3),220,np.uint8))
        class Geometry(StageFixture):
            def execute(inner,s,tool,target,work,context):
                if tool == 'measure_instances':
                    return WorkflowBackend().execute(s,tool,target,work,context)
                result=super().execute(s,tool,target,work,context)
                if tool == 'segment_instance':
                    masks=Path(result['segmentation'])/'masks'
                    mask=np.zeros((64,64),np.uint8);cv2.circle(mask,(32,32),10,255,-1)
                    name=self.image.stem+'__single_crystal__inst01.png'
                    cv2.imwrite(str(masks/name),mask)
                    raw={k:'' for k in contracts.MANIFEST_COLUMNS}
                    raw.update(instance_id=1,inst_mask=name,crop_id='crop',source_image_path=str(self.image),
                               feature_id='single_crystal',object_label='方晶',px_per_um='2',
                               igm_status='ok',granularity='instance',tight_bbox='[20,20,44,44]')
                    with (masks/(self.image.stem+contracts.MANIFEST_SUFFIX)).open('w',encoding='utf-8-sig',newline='') as f:
                        writer=csv.DictWriter(f,fieldnames=contracts.MANIFEST_COLUMNS);writer.writeheader();writer.writerow(raw)
                return result
        r=self.runtime(backend=Geometry(),agent_policy='adaptive_v2',auto_measure_after_segment=True,
                       python=sys.executable,max_seconds=40,tool_timeout_seconds=20)
        for tool in ['detect_fov_initial','classify_target','segment_instance']:
            result=r.workflow.execute(next(a for a in r.workflow.propose() if a['tool']==tool))
            self.assertTrue(result['accepted'])
        auto=next(o for o in read_json(r.workflow.operations)['entries'].values() if o.get('automatic'))
        self.assertEqual(auto['status'],'done',auto)
        rows=r.workflow.store.verified_records();self.assertEqual(len(rows),1)
        self.assertGreater(rows[0]['feret_max_um'],0)
        self.assertEqual(read_json(r.workflow.budget_path)['used'].get('network',0),0)
