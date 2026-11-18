"""旁路针晶审计与 mask 拆分的离线回归，只验证非阻断输出，不证明识别准确率。

主要定义：
- NeedleObservationTests.test_split_mask_separates_components_without_inventing_instances: 校验 `split_mask` 按连通域拆分实例，不虚构实例且每个实例落盘。
- NeedleObservationTests.test_build_audit_is_sidecar_and_does_not_touch_operations: 校验 `build_needle_audit` 只写旁路 sidecar，`operations.json` 字节不变且标记 non_blocking。

关联：
- TCM_Fusion.lib.needle_observation: 被测的 `build_needle_audit` 与 `split_mask`。
- TCM_Fusion.lib.storage: 测试用 `read_json`/`write_json` 构造运行目录与操作账本。
"""
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from TCM_Fusion.lib.needle_observation import build_needle_audit, split_mask
from TCM_Fusion.lib.storage import read_json, write_json


class NeedleObservationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.mask = self.root / 'mask.png'
        image = np.zeros((160, 240), dtype=np.uint8)
        cv2.rectangle(image, (20, 40), (190, 48), 255, -1)
        cv2.rectangle(image, (40, 100), (210, 108), 255, -1)
        cv2.imwrite(str(self.mask), image)

    def test_split_mask_separates_components_without_inventing_instances(self):
        result = split_mask(self.mask, out_dir=self.root / 'instances')
        self.assertEqual(result['n_components'], 2)
        self.assertEqual(result['n_instances'], 2)
        self.assertEqual(result['status'], 'split')
        self.assertFalse(result['merged'])
        self.assertTrue(all(row['length_px'] > 0 for row in result['instances']))
        self.assertTrue(all(Path(row['mask']).is_file() for row in result['instances']))

    def test_build_audit_is_sidecar_and_does_not_touch_operations(self):
        run = self.root / 'run'
        op_root = run / 'workflow/ops/detect'
        crop_dir = op_root / 'detection/crops'
        crop_dir.mkdir(parents=True)
        crop = {'crop_id': 'needle_test_01', 'feature_id': 'needle_crystal',
                'det_score': 0.65, 'source_image_path': str(self.root / 'view_01.jpg')}
        write_json(crop_dir / 'needle_test_01.json', crop)
        classify = run / 'workflow/ops/classify/classification'
        classify.mkdir(parents=True)
        write_json(classify / 'classification.json', {
            'crop_id': 'needle_test_01', 'feature_id': 'fiber', 'label': '纤维',
            'confidence': 'medium', 'confidence_score': 0.65})
        seg = run / 'workflow/ops/segment/segmentation/masks'
        seg.mkdir(parents=True)
        cv2.imwrite(str(seg / 'mask.png'), cv2.imread(str(self.mask), cv2.IMREAD_GRAYSCALE))
        operations = {
            'detect': {'tool': 'locate_feature', 'status': 'done',
                       'target_id': 'locate:test', 'bundle': {'root': str(op_root)}},
            'classify': {'tool': 'classify_target', 'status': 'done',
                         'target_id': 'crop:needle_test_01',
                         'bundle': {'root': str(run / 'workflow/ops/classify')}},
            'segment': {'tool': 'segment_instance', 'status': 'done',
                        'target_id': 'crop:needle_test_01',
                        'bundle': {'root': str(run / 'workflow/ops/segment')}},
        }
        op_path = run / 'workflow/operations.json'
        write_json(op_path, {'entries': operations})
        before = op_path.read_bytes()
        audit = build_needle_audit(run)
        self.assertTrue(audit['non_blocking'])
        self.assertEqual(audit['summary']['classified_non_target'], 1)
        self.assertEqual(audit['summary']['n_instances'], 2)
        self.assertTrue((run / 'needle_audit/needle_audit.json').is_file())
        self.assertTrue((run / 'needle_audit/feature_audit.json').is_file())
        self.assertEqual(op_path.read_bytes(), before)
        self.assertTrue(audit['candidates'][0]['non_blocking'])


if __name__ == '__main__':
    unittest.main()
