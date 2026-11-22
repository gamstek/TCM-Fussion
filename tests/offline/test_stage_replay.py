"""验证批量分类和示例审计的录制与离线回放。

以临时工作目录中的固定结果验证逐实例输出、路径重绑定和摘要检查。

主要定义：
- StageReplayTests 覆盖混合成功/失败、旧录制缺件和篡改。

关联：
- stage_recording 记录文件，RecordedStages 读取副本。
"""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from TCM_Fusion.lib.evidence.stage_recording import bundle
from TCM_Fusion.lib.offline.workflow_replay import RecordedStages
from TCM_Fusion.lib.runtime.storage import IntegrationError, file_hash, write_json


class StageReplayTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / 'stage'
        self.source.mkdir()
        self.session = SimpleNamespace(dependency_fingerprint=lambda *_: 'fixed-dependencies')

    def replay(self, tool, recording, name='replay'):
        backend = RecordedStages([{'tool': tool, 'target_id': 'target', 'status': 'done',
                                   'dependency_fingerprint': 'fixed-dependencies', 'bundle': recording}])
        work = self.root / name
        work.mkdir()
        try:
            return backend.execute(self.session, tool, {'id': 'target'}, work, None)
        except IntegrationError:
            self.assertEqual(backend.misses, 1)
            raise

    def test_batch_round_trip_preserves_per_crop_success_and_failure(self):
        directory = self.source/'classify_0'/'classification'
        classification = directory/'classification.json'
        write_json(classification, {'label': 'fiber', 'source': 'cnn'})
        result = {
            'classifications': [{'crop_id': 'crop:one', 'status': 'done',
                                 'classification': str(directory)}],
            'failures': [{'crop_id': 'crop:two', 'status': 'failed',
                          'error_code': 'CNN_FAILED', 'error': 'discard raw message'}],
        }
        replayed = self.replay('classify_crops_batch', bundle(result, self.source))
        entry, = replayed['classifications']
        self.assertEqual((entry['crop_id'], entry['status']), ('crop:one', 'done'))
        copy = Path(entry['classification'])/'classification.json'
        self.assertTrue(copy.is_relative_to(self.root/'replay'/'recorded'))
        self.assertEqual(copy.read_bytes(), classification.read_bytes())
        self.assertEqual(replayed['failures'], [
            {'crop_id': 'crop:two', 'status': 'failed', 'error_code': 'CNN_FAILED'},
        ])

    def test_exemplar_round_trip_and_tamper_counts_as_replay_miss(self):
        audit = self.source/'audit.json'
        write_json(audit, {'schema': 'image-exemplar-audit/v2', 'non_blocking': True})
        sheet = self.source/'contact.jpg'
        sheet.write_bytes(b'fixture contact sheet bytes')
        recording = {
            'root': str(self.source),
            'paths': {'audit': audit.name, 'contact_sheet': sheet.name},
            'files': {path.name: file_hash(path) for path in (audit, sheet)},
        }
        replayed = self.replay('image_exemplar_audit', recording)
        self.assertEqual(Path(replayed['audit']).read_bytes(), audit.read_bytes())
        self.assertEqual(Path(replayed['contact_sheet']).read_bytes(), sheet.read_bytes())
        sheet.write_bytes(b'changed')
        with self.assertRaises(IntegrationError) as caught:
            self.replay('image_exemplar_audit', recording, name='tampered')
        self.assertEqual(caught.exception.code, 'REPLAY_ARTIFACT')

    def test_old_incomplete_bundles_report_missing_recording(self):
        for tool in ('classify_crops_batch', 'image_exemplar_audit'):
            with self.subTest(tool=tool), self.assertRaises(IntegrationError) as caught:
                self.replay(tool, bundle({}, self.source), name=tool)
            self.assertEqual(caught.exception.code, 'REPLAY_MISS')
