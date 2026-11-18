"""presence 可选定位与影子线索的边界回归：确定性与兼容性验证，不宣称真实 API 或真值。

主要定义：
- PresenceLocalizationTests（映射与用途）: 校验模型框逆变换到源图 tight bbox，且身份保持 pending、不授予尺寸资格。
- PresenceLocalizationTests（旧观察区分）: 无框阳性、阴性、服务失败与低置信分别落到 present_unlocalized/not_observed_local/unknown。
- PresenceLocalizationTests（非法坐标）: 坏数值、歧义空间、越界/反向/零面积、未知变换与源图不符都不静默夹取或擦除观察。
- PresenceLocalizationTests（诊断项）: 小目标、细长、贴边、整图、空间重叠与 EXIF 旋转只作诊断，不用于去重或身份验证。
- PresenceLocalizationTests（硬阻断）: 空白裁图与不可读质量即使旧 block_segment 为假也阻断新工件。
- PresenceLocalizationTests（发布与取消）: 任务取消、关闭、源图删除或替换时不发布诊断，并落失败/取消终态、不自动重放。
- PresenceLocalizationTests（主链与传输）: 图执行只让原始观察改变规则，local/MCP 传输均保留诊断与旧正式结果。
- PresenceLocalizationTests（进程清理）: taskkill 失败时立即回退并回收真实子进程。

关联：
- TCM_Fusion.lib.presence_localization: 被测的 assess_presence / observation_context / sanitize_proposal。
- TCM_Fusion.lib.agent_tools / controlled_process: 观察请求侧车与整树终止。
- TCM_Fusion.lib.evidence_runtime: 诊断工件发布与终态记录。
- 位于 tests/ 下，随 selftest 递归发现。
"""
import asyncio
import base64
import copy
import io
import json
import subprocess
import sys
import threading
import time
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
import paths

from TCM_Fusion.lib.agent_tools import LiveBackend, LocalToolbox
from TCM_Fusion.lib.controlled_process import kill_tree
from TCM_Fusion.lib.evidence_runtime import EvidenceRuntime
from TCM_Fusion.lib.presence_localization import (
    assess_presence, observation_context, sanitize_proposal,
)
from TCM_Fusion.lib.storage import file_hash, read_json, write_json


class PresenceLocalizationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.image = self.root / 'view.png'
        image = Image.new('RGB', (200, 120))
        image.putdata([((x * 41 + y * 73) % 256,) * 3 for y in range(120) for x in range(200)])
        image.save(self.image)
        self.sha = file_hash(self.image)
        with Image.open(self.image) as source:
            sent = source.resize((100, 60))
            buffer = io.BytesIO()
            sent.save(buffer, format='PNG')
        self.url = 'data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode('ascii')
        self.context = observation_context(self.image, self.url)
        self.result = {'ok': True, 'present': True, 'conf': .9, 'reason': 'test',
                       'localization_proposal': {'bbox': [10, 10, 40, 30],
                                                 'coordinate_space': 'model_input_pixels'}}

    def evaluate(self, result=None, **kwargs):
        return assess_presence(result if result is not None else self.result,
            path=self.image, expected_sha256=self.sha, feature_id='hypha',
            context=kwargs.pop('context', self.context), **kwargs)

    def test_mapping_context_crop_and_no_identity_or_dimensions_granted(self):
        directory = self.root / 'diagnostic'
        row = self.evaluate(output_dir=directory)
        self.assertEqual(row['source_tight_bbox'], [20, 20, 80, 60])
        self.assertEqual(row['localization'], 'valid')
        self.assertEqual(row['state'], 'present_pending')
        self.assertEqual(row['identity'], 'pending')
        self.assertFalse(any(row['usage_eligibility'].values()))
        self.assertNotEqual(row['crop_bbox'], row['source_tight_bbox'])
        self.assertTrue(Path(row['artifacts']['target_png']).is_file())
        self.assertTrue(Path(row['artifacts']['context_png']).is_file())
        self.assertEqual(read_json(directory / 'diagnostic.json'), row)
        self.assertEqual(file_hash(self.image), self.sha)
        self.assertNotIn('px_per_um', json.dumps(row))

    def test_legacy_no_box_positive_negative_failure_low_conf_are_distinct(self):
        positive = {k: v for k, v in self.result.items() if k != 'localization_proposal'}
        for result, expected in ((positive, 'present_unlocalized'),
                ({**positive, 'present': False}, 'not_observed_local'),
                ({'ok': False, 'present': False}, 'unknown'),
                ({**positive, 'conf': .2}, 'unknown')):
            with self.subTest(expected=expected):
                self.assertEqual(self.evaluate(result)['state'], expected)

    def test_schema_bad_numeric_or_ambiguous_coordinates_never_erase_observation(self):
        for box in ([True, 1, 2, 3], [float('nan'), 1, 2, 3], [1, 2, 3],
                    ['1', 1, 2, 3], [1, 1, float('inf'), 3]):
            with self.subTest(box=box):
                result = copy.deepcopy(self.result)
                result['localization_proposal']['bbox'] = box
                row = self.evaluate(result)
                self.assertEqual(row['localization'], 'invalid')
                self.assertEqual(row['state'], 'present_unlocalized')
                self.assertTrue(result['present'])
        self.assertEqual(sanitize_proposal({'bbox': [0, 0, 1, 1], 'coordinate_space': 'unknown'}),
                         {'error_code': 'BOX_SCHEMA_INVALID'})

    def test_out_of_bounds_reversed_zero_area_are_not_silently_clamped(self):
        for box in ([-1, 1, 10, 10], [0, 0, 101, 60], [10, 10, 9, 20], [1, 1, 1, 20]):
            result = copy.deepcopy(self.result)
            result['localization_proposal']['bbox'] = box
            row = self.evaluate(result)
            self.assertIn('BOX_BOUNDS_INVALID', row['findings'])
            self.assertEqual(row['localization'], 'invalid')

    def test_unknown_transform_and_mismatched_source_are_rejected(self):
        for edit in ({'source_image_sha256': 'wrong'}, {'source_size': [201, 120]},
                     {'transform': {'kind': 'letterbox'}}, {'model_input_size': [0, 60]}):
            row = self.evaluate(context={**self.context, **edit})
            self.assertEqual(row['localization'], 'invalid')
        self.image.write_bytes(b'changed')
        self.assertIn('SOURCE_CHANGED', self.evaluate()['findings'])

    def test_small_thin_border_and_full_image_are_diagnostics_not_coordinate_rejections(self):
        for box, flag in (([0, 0, 1, 50], 'small_box'),
                          ([0, 0, 100, 60], 'large_box')):
            result = copy.deepcopy(self.result)
            result['localization_proposal']['bbox'] = box
            row = self.evaluate(result)
            self.assertEqual(row['localization'], 'valid')
            self.assertIn(flag, row['findings'])
            self.assertIn('possible_border_truncation', row['findings'])
            self.assertFalse(row['usage_eligibility']['length'])

    def test_blank_target_not_hidden_by_textured_expanded_context(self):
        with Image.open(self.image) as original:
            original.paste((255, 255, 255), (20, 20, 80, 60))
            original.save(self.image)
        self.sha = file_hash(self.image)
        self.context['source_image_sha256'] = self.sha
        row = self.evaluate()
        self.assertEqual(row['quality'], 'blank')
        self.assertEqual(row['identity'], 'pending')

    def test_unreadable_quality_hard_blocks_even_legacy_block_segment_false(self):
        with patch('TCM_Fusion.lib.presence_localization.assess', return_value={
                'status': 'invalid', 'flags': ['invalid_image'], 'block_segment': False}):
            row = self.evaluate()
        self.assertEqual(row['quality'], 'unreadable')
        self.assertEqual(row['localization'], 'invalid')
        self.assertIn('CROP_UNREADABLE', row['findings'])

    def test_spatial_overlap_does_not_deduplicate_or_verify_identity(self):
        row = self.evaluate(existing=[{'id': 'same', 'feature_id': 'hypha', 'box': [20, 20, 80, 60]},
            {'id': 'nested', 'feature_id': 'cell', 'box': [20, 20, 80, 60]}])
        self.assertTrue(row['spatial_agreement'][0]['possible_duplicate'])
        self.assertTrue(row['spatial_agreement'][1]['possible_identity_conflict'])
        self.assertFalse(any(x['identity_inference'] for x in row['spatial_agreement']))
        self.assertEqual(row['identity'], 'pending')

    def test_exif_rotated_mapping_not_guessed(self):
        path = self.root / 'rotated.jpg'
        exif = Image.Exif()
        exif[274] = 6
        Image.new('RGB', (40, 60)).save(path, exif=exif)
        self.assertEqual(observation_context(path, self.url)['error_code'],
                         'ORIENTATION_MAPPING_UNCONFIRMED')

    def runtime_and_backend(self):
        fixture = read_json(Path(paths.PROJECT_ROOT) / 'TCM_Fusion/tests/fixtures/react_case.json')
        runtime = EvidenceRuntime([], [self.image], sample_id='presence-test',
                                  a=fixture['a'], simulation=True)
        write_json(self.root / 'vlm_config.json', {'vlm': {'base_url': 'https://invalid.example',
                   'model': 'fixture', 'api_key': 'fake'}})
        with patch.object(paths, 'DETECTION_CONFIG_DIR', str(self.root)):
            backend = LiveBackend()
        criterion = next(c for r in runtime.kb.candidates() for c in r['criteria']
                         if c['feature'] == 'hypha')
        action = runtime.action('presence_check', {'image_id': next(iter(runtime.images)),
            'feature_id': 'hypha', 'morph_hint': criterion.get('morph') or ''}, 'differential_presence')
        return runtime, backend, action

    def test_one_request_preserves_optional_box_and_planner_receives_diagnostic(self):
        runtime, backend, action = self.runtime_and_backend()
        body = {'choices': [{'message': {'content': json.dumps({
            'present': True, 'conf': .9, 'reason': 'test', 'box': self.result['localization_proposal']})}}]}
        toolbox = LocalToolbox(runtime, ledger_path=self.root / 'ledger.json', backend=backend)
        with patch('TCM_Fusion.lib.hard_calls.encode_image', return_value=self.url), \
                patch('TCM_Fusion.lib.hard_calls.post', return_value=body) as post:
            record = asyncio.run(toolbox._run_observation(action, {
                'action': action, 'action_id': action['action_id'], 'new_requests': 1}))
        post.assert_called_once()
        self.assertIn('model_input_pixels', json.dumps(post.call_args.args[0]))
        self.assertTrue(record['accepted'])
        self.assertEqual(record['presence_localization']['state'], 'present_pending')
        self.assertTrue(runtime.summary({'observations': [record]})['presence_localizations'])

    def test_bad_optional_box_preserves_legacy_formal_result_field_by_field(self):
        runtime, backend, action = self.runtime_and_backend()
        expected = {'ok': True, 'present': True, 'conf': .9, 'reason': 'test'}
        results = []
        for extra in ({}, {'box': {'bbox': [1, 2], 'coordinate_space': 'model_input_pixels'}}):
            body = {'choices': [{'message': {'content': json.dumps({
                'present': True, 'conf': .9, 'reason': 'test', **extra})}}]}
            with patch('TCM_Fusion.lib.hard_calls.encode_image', return_value=self.url), \
                    patch('TCM_Fusion.lib.hard_calls.post', return_value=body):
                value, _ = backend.call(runtime, action, self.image)
            self.assertEqual({k: value[k] for k in expected}, expected)
            results.append(runtime.score([{'action': action, 'result': value, 'accepted': True}]))
        self.assertEqual(results[0]['decision'], results[1]['decision'])
        self.assertEqual(results[0]['evidence_digest'], results[1]['evidence_digest'])

    def test_invalid_observation_confidence_remains_unknown(self):
        for conf in (True, None, float('nan'), float('inf'), '0.9', -1, 2):
            with self.subTest(conf=conf):
                self.assertEqual(self.evaluate({**self.result, 'conf': conf})['state'], 'unknown')

    def test_cancelled_diagnostic_does_not_publish(self):
        directory = self.root / 'cancelled'
        row = self.evaluate(output_dir=directory, can_publish=lambda: False)
        self.assertEqual(row['state'], 'unknown')
        self.assertFalse(directory.exists())

    def test_cancellation_during_quality_check_prevents_publishing_diagnostic(self):
        cancelled = threading.Event()
        directory = self.root / 'cancel-during-quality'
        from TCM_Fusion.lib.box_quality import assess as original
        def cancel_after_assess(*args, **kwargs):
            result = original(*args, **kwargs)
            cancelled.set()
            return result
        with patch('TCM_Fusion.lib.presence_localization.assess', side_effect=cancel_after_assess):
            row = self.evaluate(output_dir=directory, can_publish=lambda: not cancelled.is_set())
        self.assertEqual(row['state'], 'unknown')
        self.assertFalse((directory / 'diagnostic.json').exists())
        self.assertFalse(any(row['usage_eligibility'].values()))

    def test_existing_artifacts_never_overwritten(self):
        directory = self.root / 'immutable'
        original = self.evaluate(output_dir=directory)
        before = file_hash(directory / 'diagnostic.json')
        repeated = self.evaluate(output_dir=directory)
        self.assertEqual(repeated['localization'], 'invalid')
        self.assertEqual(file_hash(directory / 'diagnostic.json'), before)
        self.assertEqual(original, read_json(directory / 'diagnostic.json'))

    def test_closed_workflow_response_is_not_accepted_or_written_as_new_crop(self):
        from TCM_Fusion.lib.workflow_runtime import WorkflowRuntime
        from TCM_Fusion.tests.test_workflow import StageFixture
        from types import SimpleNamespace
        runtime = WorkflowRuntime([self.image], root=self.root / 'workflow',
            workflow_config={}, sample_id='closed-response', simulation=True, backend=StageFixture())
        criterion = next(c for r in runtime.kb.candidates() for c in r['criteria'] if c['feature'] == 'hypha')
        action = runtime.action('presence_check', {'image_id': next(iter(runtime.images)),
            'feature_id': 'hypha', 'morph_hint': criterion.get('morph') or ''}, 'differential_presence')
        owner = self
        class ClosingBackend:
            client = SimpleNamespace(observation_context=self.context, diagnostics={}, last_usage=None)
            def call(self, runtime, action, path):
                runtime.workflow.close('test-close')
                return owner.result, None
        toolbox = LocalToolbox(runtime, ledger_path=self.root / 'closed.json', backend=ClosingBackend())
        record = asyncio.run(toolbox._run_observation(action, {'action': action,
            'action_id': action['action_id'], 'new_requests': 1}))
        self.assertFalse(record['accepted'])
        self.assertEqual(record['error_code'], 'WORKFLOW_CLOSED')
        self.assertFalse((self.root / 'presence_localizations').exists())

    def test_cancelled_observation_has_terminal_ledger_and_no_automatic_replay(self):
        from TCM_Fusion.lib.agent_tools import FixtureBackend
        runtime, _, action = self.runtime_and_backend()
        class CancelledToolbox(LocalToolbox):
            async def _run_observation(self, action, pending):
                raise asyncio.CancelledError()
        toolbox = CancelledToolbox(runtime, ledger_path=self.root / 'cancel-ledger.json',
                                   backend=FixtureBackend())
        with patch.object(runtime, 'legal_actions', return_value=[action]):
            with self.assertRaises(asyncio.CancelledError):
                asyncio.run(toolbox.observe(action))
        entry = read_json(toolbox.ledger_path)['entries'][action['action_id']]
        self.assertEqual(entry['status'], 'cancelled')
        self.assertFalse(entry['accepted'])
        self.assertEqual(read_json(toolbox.ledger_path)['requests'], 1)
        # Re-open the same task's ledger: do not resend a request with unknown billing.
        reopened = LocalToolbox(runtime, ledger_path=toolbox.ledger_path, backend=FixtureBackend())
        cached = asyncio.run(reopened.observe(action))
        self.assertTrue(cached['cache_hit'])
        self.assertEqual(cached['new_requests'], 0)

    def test_source_deleted_during_observation_has_failed_terminal_ledger(self):
        runtime, _, action = self.runtime_and_backend()
        class DeletingBackend:
            def call(self, runtime, action, path):
                path.unlink()
                return {'ok': True, 'present': True, 'conf': .9, 'reason': 'test'}, None
        toolbox = LocalToolbox(runtime, ledger_path=self.root / 'deleted.json', backend=DeletingBackend())
        with patch.object(runtime, 'legal_actions', return_value=[action]):
            record = asyncio.run(toolbox.observe(action))
        self.assertFalse(record['accepted'])
        self.assertEqual(record['status'], 'failed')
        self.assertEqual(record['error_code'], 'INPUT_CHANGED')
        self.assertEqual(read_json(toolbox.ledger_path)['entries'][action['action_id']]['status'], 'failed')

    def test_source_changed_between_processing_and_commit_is_rejected(self):
        from TCM_Fusion.lib.agent_tools import FixtureBackend
        runtime, _, action = self.runtime_and_backend()
        class MutatingToolbox(LocalToolbox):
            async def _run_observation(self, action, pending):
                record = await super()._run_observation(action, pending)
                self.runtime.images[action['args']['image_id']]['path'].write_bytes(b'changed-before-commit')
                return record
        toolbox = MutatingToolbox(runtime, ledger_path=self.root / 'changed.json', backend=FixtureBackend())
        with patch.object(runtime, 'legal_actions', return_value=[action]):
            record = asyncio.run(toolbox.observe(action))
        self.assertFalse(record['accepted'])
        self.assertEqual(record['error_code'], 'INPUT_CHANGED')
        self.assertEqual(record['presence_localization']['state'], 'unknown')

    def test_graph_preserves_box_diagnostic_and_only_original_observation_changes_rules(self):
        from TCM_Fusion.lib.react_graph import run_react, ScriptedPlanner
        runtime, backend, _ = self.runtime_and_backend()
        fixture = read_json(Path(paths.PROJECT_ROOT) / 'TCM_Fusion/tests/fixtures/react_case.json')
        records = copy.deepcopy(fixture['records'])
        records[0]['image_stem'] = self.image.stem
        runtime = EvidenceRuntime(records, [self.image], sample_id='graph-hint',
                                  a=fixture['a'], simulation=True)
        toolbox = LocalToolbox(runtime, ledger_path=self.root / 'graph-ledger.json', backend=backend)
        def post(*args, **kwargs):
            if backend.client.expected_tool == 'presence_check':
                value = {'present': True, 'conf': .95, 'reason': 'test',
                         'box': self.result['localization_proposal']}
            else:
                value = {'conf': .9, 'cell_wall_visible': False, 'crystal_location': 'free',
                         'arrangement': 'scattered', 'abundance': 'many', 'reason': 'test'}
            return {'choices': [{'message': {'content': json.dumps(value)}}]}
        with patch('TCM_Fusion.lib.hard_calls.encode_image', return_value=self.url), \
                patch('TCM_Fusion.lib.hard_calls.post', side_effect=post):
            trace = asyncio.run(run_react(runtime, toolbox, planner_mode='scripted', planner=ScriptedPlanner()))
        presence = next(o for o in trace['observations'] if o['action']['tool'] == 'presence_check')
        self.assertEqual(presence['presence_localization']['state'], 'present_pending')
        self.assertFalse(any(presence['presence_localization']['usage_eligibility'].values()))
        old = [{k: v for k, v in o.items() if k != 'presence_localization'} for o in trace['observations']]
        for o in old:
            o['result'] = {k: v for k, v in o['result'].items() if k != 'localization_proposal'}
        self.assertEqual(runtime.score(old)['decision'], trace['decision'])
        self.assertEqual(runtime.score(old)['cards']['hypha'].dims, {})

    def test_real_mcp_transport_preserves_diagnostic_and_legacy_result(self):
        from TCM_Fusion.lib.mcp_bridge import connect_stdio
        from TCM_Fusion.lib.react_graph import run_react, ScriptedPlanner
        fixture = read_json(Path(paths.PROJECT_ROOT) / 'TCM_Fusion/tests/fixtures/react_case.json')
        fixture['records'][0]['image_stem'] = self.image.stem
        fixture['images'] = [str(self.image)]
        fixture['replies']['hypha']['localization_proposal'] = self.result['localization_proposal']
        runtime = EvidenceRuntime(fixture['records'], [self.image], sample_id='mcp-hint',
                                  a=fixture['a'], config_identity='mcp-hint', simulation=True)
        context = self.root / 'mcp-context.json'
        write_json(context, {'sample_id': 'mcp-hint', 'simulation': True,
                             'config_identity': runtime.config_identity, 'fixture': fixture, 'allow_network': False})
        async def check():
            async with connect_stdio(context) as toolbox:
                trace = await run_react(runtime, toolbox, planner_mode='scripted', planner=ScriptedPlanner())
                presence = next(o for o in trace['observations'] if o['action']['tool'] == 'presence_check')
                self.assertTrue(presence['accepted'])
                # Fixture supplied no trusted encoder transform: do not guess coordinates.
                self.assertEqual(presence['presence_localization']['state'], 'present_unlocalized')
                self.assertEqual(presence['presence_localization']['localization'], 'invalid')
                summary = await toolbox.call('read_evidence')
                self.assertTrue(summary['presence_localizations'])
        asyncio.run(check())

    @unittest.skipUnless(sys.platform == 'win32', 'Windows taskkill fallback')
    def test_failed_taskkill_falls_back_immediately_and_reaps_real_child(self):
        child = subprocess.Popen([sys.executable, '-c', 'import time;time.sleep(30)'])
        try:
            with patch('TCM_Fusion.lib.controlled_process.subprocess.run',
                       return_value=subprocess.CompletedProcess([], 1)):
                start = time.monotonic()
                kill_tree(child)
            self.assertLess(time.monotonic() - start, 2)
            self.assertIsNotNone(child.poll())
        finally:
            if child.poll() is None:
                child.kill()
                child.wait(timeout=2)

