"""框内容质量评估回归：只阻断空白内容，不因成簇/相邻而拒绝分割。

主要定义：
- BoxQualityTests.test_blank_is_flagged_but_invalid_is_not_blocked: 纯色空白图判为 blank 并阻断，不可读文件判为 invalid 但不阻断。
- BoxQualityTests.test_textured_crop_is_usable: 有纹理的裁图判为 usable 且不阻断。
- LowDetailWorkflowTests: 低细节提示与工具准入贯通，身份/空白/标定/QA/去重不被绕过。

关联：
- TCM_Fusion.lib.box_quality.assess: 被测试的裁图质量判定与 block_segment 建议。
- PIL.Image/ImageDraw: 生成临时空白与纹理裁图。
- 位于 tests/ 下，随 selftest 递归发现。
"""
import tempfile
import unittest
from pathlib import Path
import shutil

from PIL import Image, ImageDraw

import contracts
from TCM_Fusion.lib.box_quality import assess
from TCM_Fusion.lib.storage import IntegrationError, read_json, write_json
from TCM_Fusion.lib.tool_registry import validate_arguments
from TCM_Fusion.lib.workflow_entry import WORKFLOW_SCHEMA
from TCM_Fusion.lib.workflow_runtime import WorkflowRuntime
from TCM_Fusion.tests.test_workflow import StageFixture


class BoxQualityTests(unittest.TestCase):
    def test_blank_is_flagged_but_invalid_is_not_blocked(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            blank = root / 'blank.png'
            Image.new('RGB', (100, 80), (128, 128, 128)).save(blank)
            result = assess(blank)
            self.assertEqual(result['status'], 'blank')
            self.assertTrue(result['block_segment'])
            invalid = root / 'bad.png'
            invalid.write_bytes(b'not-an-image')
            result = assess(invalid)
            self.assertEqual(result['status'], 'invalid')
            self.assertFalse(result['block_segment'])

    def test_textured_crop_is_usable(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'texture.png'
            image = Image.new('RGB', (200, 120), 'white')
            draw = ImageDraw.Draw(image)
            for x in range(10, 190, 8):
                draw.line((x, 10, x + 30, 110), fill='black', width=2)
            image.save(path)
            result = assess(path)
            self.assertEqual(result['status'], 'usable')
            self.assertFalse(result['block_segment'])


class LowDetailStageFixture(StageFixture):
    """显式离线后端：真实低对比度PNG，分类/mask结果为受控替身，不调视觉API。"""
    def __init__(self, *, quality='valid', **kwargs):
        super().__init__(**kwargs)
        self.quality = quality

    def execute(self, session, tool, target, work, context):
        result = super().execute(session, tool, target, work, context)
        if tool in {'detect_fov_initial', 'detect_region'}:
            folder = Path(result['crops_dir'])
            shutil.copyfile(next(iter(session.runtime.images.values()))['path'], folder / 'crop.png')
            side = read_json(folder / 'crop.json')
            side.update(tight_box=[10, 10, 90, 90], tight_box_local=[10, 10, 90, 90],
                        loose_box=[0, 0, 100, 100], crop_size='100x100')
            write_json(folder / 'crop.json', side)
        elif tool == 'segment_instance' and self.quality == 'review':
            result['quality'] = 'review'
            write_json(Path(result['segmentation']) / 'instances.json',
                       {'instances': [{'igm_status': 'review'}]})
        return result


class LowDetailWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.image = self.root / 'fov.png'
        # 可见灰度渐变：不是空白，但平均相邻差低；不用黑白条纹假装显微QA。
        image = Image.new('RGB', (100, 100))
        image.putdata([(110 + x * 120 // 99,) * 3 for y in range(100) for x in range(100)])
        image.save(self.image)
        self.assertEqual(assess(self.image)['status'], 'low_detail')

    def runtime(self, backend=None, **config):
        return WorkflowRuntime([self.image], root=self.root / 'task', sample_id='sample',
            a={}, simulation=True, backend=backend or LowDetailStageFixture(),
            workflow_config={'agent_policy': 'adaptive_v2', 'max_recompute': 8,
                             'quality_policy': 'strict', **config})

    def execute(self, runtime, tool):
        return runtime.workflow.execute(next(a for a in runtime.workflow.propose() if a['tool'] == tool))

    def crop(self, runtime):
        return next(v for v in runtime.workflow.store.current()['artifacts'].values()
                    if v.get('valid') and v.get('kind') == 'crop')

    def classified(self, **config):
        runtime = self.runtime(**config)
        self.execute(runtime, 'detect_fov_initial')
        self.execute(runtime, 'classify_target')
        return runtime

    def test_low_detail_requires_identity_and_forged_action_is_denied(self):
        runtime = self.runtime()
        self.execute(runtime, 'detect_fov_initial')
        crop = self.crop(runtime)
        admission = runtime.workflow.crop_segmentation_admission(crop)
        self.assertEqual(admission['reason'], 'identity_confirmation_required')
        self.assertFalse(admission['allowed'])
        action = runtime.action('segment_instance', {'target_id': crop['id'],
            'evidence_version': runtime.workflow.store.current()['version']}, 'workflow_repair')
        with self.assertRaises(IntegrationError) as error:
            runtime.workflow.execute(action)
        self.assertEqual(error.exception.code, 'ACTION_DENIED')
        self.assertNotIn('segment_instance', runtime.workflow.backend.calls)

    def test_verified_low_detail_is_offered_without_automatic_segmentation(self):
        runtime = self.classified()
        crop = self.crop(runtime)
        self.assertEqual(crop['box_quality'], 'low_detail')
        admission = runtime.workflow.crop_segmentation_admission(crop)
        self.assertTrue(admission['allowed'])
        self.assertTrue(admission['quality_advisory_only'])
        self.assertFalse(admission['grants_dimension_evidence'])
        self.assertIn('segment_instance', [a['tool'] for a in runtime.workflow.propose()])
        self.assertNotIn('segment_instance', runtime.workflow.backend.calls)
        self.assertEqual(runtime.workflow.store.verified_records(), [])

    def test_removed_option_is_rejected_and_absent_from_live_template(self):
        import paths
        key = 'box_quality_block_low_detail'
        for value in (True, False):
            with self.subTest(value=value), self.assertRaises(IntegrationError):
                validate_arguments(WORKFLOW_SCHEMA, {key: value})
        template = read_json(Path(paths.PROJECT_ROOT) / 'TCM_Fusion/config/multi_view.json')
        self.assertNotIn(key, template['agent']['workflow'])

    def test_blank_remains_blocked_even_after_identity_confirmation(self):
        Image.new('RGB', (100, 100), (128, 128, 128)).save(self.image)
        runtime = self.classified()
        self.assertEqual(self.crop(runtime)['box_quality'], 'blank')
        self.assertNotIn('segment_instance', [a['tool'] for a in runtime.workflow.propose()])

    def test_feature_identity_mismatch_does_not_grant_old_crop(self):
        runtime = self.runtime(backend=LowDetailStageFixture(classification_feature='fiber'))
        self.execute(runtime, 'detect_fov_initial')
        old = self.crop(runtime)
        self.execute(runtime, 'classify_target')
        # 分类改判会使原类别身份不成立；不得要求运行时把它当已确认身份处理。
        admission = runtime.workflow.crop_segmentation_admission(old)
        self.assertFalse(admission['allowed'])
        self.assertEqual(admission['reason'], 'identity_confirmation_required')
        self.assertNotIn(old['id'], [a['args']['target_id'] for a in runtime.workflow.propose()
                                    if a['tool'] == 'segment_instance'])

    def test_missing_scale_still_requires_calibration(self):
        runtime = self.runtime(backend=LowDetailStageFixture(scale=None))
        self.execute(runtime, 'detect_fov_initial')
        self.execute(runtime, 'classify_target')
        tools = [a['tool'] for a in runtime.workflow.propose()]
        self.assertNotIn('segment_instance', tools)
        self.assertIn('calibrate_fov', tools)

    def test_review_mask_is_not_auto_measured_in_strict_mode(self):
        runtime = self.runtime(backend=LowDetailStageFixture(quality='review'),
                               auto_measure_after_segment=True)
        self.execute(runtime, 'detect_fov_initial')
        self.execute(runtime, 'classify_target')
        self.execute(runtime, 'segment_instance')
        self.assertNotIn('measure_instances', runtime.workflow.backend.calls)
        self.assertNotIn('measure_instances', [a['tool'] for a in runtime.workflow.propose()])
        self.assertEqual(runtime.workflow.store.verified_records(), [])

    def test_selected_segment_keeps_dedup_and_same_object_auto_measure(self):
        runtime = self.classified(auto_measure_after_segment=True)
        action = next(a for a in runtime.workflow.propose() if a['tool'] == 'segment_instance')
        runtime.workflow.execute(action)
        runtime.workflow.execute(action)
        calls = runtime.workflow.backend.calls
        self.assertEqual(calls.count('segment_instance'), 1)
        self.assertEqual(calls.count('measure_instances'), 1)

    def test_quality_advisory_is_visible_in_existing_planner_summary(self):
        from TCM_Fusion.lib.react_graph import compact_evidence
        runtime = self.classified()
        summary = runtime.summary({'observations': []})
        rows = summary['workflow']['classifications']
        self.assertTrue(rows[0]['segmentation_admission']['quality_advisory_only'])
        self.assertTrue(rows[0]['segmentation_admission']['allowed'])
        self.assertFalse(rows[0]['segmentation_admission']['grants_dimension_evidence'])
        compacted = compact_evidence(summary)['workflow']['classification_summary']
        self.assertEqual(compacted['verified'][0]['segmentation_admission'],
                         rows[0]['segmentation_admission'])
        pending = {'crop_id': 'pending', 'feature_id': 'fiber', 'identity_verified': False,
                   'segmentation_admission': {'allowed': False, 'reason': 'identity_confirmation_required'}}
        compacted = compact_evidence({'workflow': {'classifications': [pending],
            'required_states': [{'feature_id': 'fiber'}]}})['workflow']['classification_summary']
        self.assertEqual(compacted['active_unverified'][0]['segmentation_admission'],
                         pending['segmentation_admission'])


if __name__ == '__main__':
    unittest.main()
