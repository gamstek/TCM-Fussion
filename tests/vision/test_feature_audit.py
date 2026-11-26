"""验证特征旁路审计从有效快照读裁图、分类和 mask，并只写审计目录。

使用临时工件和真实工作流提交逻辑复现初检、补扫、逐裁图分类和依赖失效。
视觉阶段输入固定或合成，无模型调用。

主要定义：
- FeatureAuditTests 检查通用默认范围、分类绑定、工件失效和 mask 落点。

关联：
- 直接依赖 「lib/vision/feature_audit.py」、「tests/support/workflow_fixtures.py」。
- 统一执行见 tests/README.md。
"""
import shutil
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from TCM_Fusion.lib.runtime.storage import read_json, write_json
from TCM_Fusion.lib.decision.candidate_audit import feature_receipts
from TCM_Fusion.lib.vision.feature_audit import build_feature_audit, split_mask
from TCM_Fusion.tests.support.workflow_fixtures import WorkflowFixture


class FeatureAuditTests(WorkflowFixture, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.mask = self.root / 'mask.png'
        image = np.zeros((160, 240), dtype=np.uint8)
        cv2.rectangle(image, (20, 40), (190, 48), 255, -1)
        cv2.rectangle(image, (40, 100), (210, 108), 255, -1)
        cv2.imwrite(str(self.mask), image)

    def test_split_mask_preserves_both_components(self):
        result = split_mask(self.mask, out_dir=self.root / '实例')
        self.assertEqual(result['n_instances'], 2)
        self.assertEqual(result['status'], 'split')
        self.assertFalse(result['merged'])
        self.assertTrue(all(row['length_px'] > 0 for row in result['instances']))
        self.assertTrue(all(Path(row['mask']).is_file() for row in result['instances']))

    def _crops(self):
        runtime = self.runtime()
        workflow = runtime.workflow
        image_id = next(iter(runtime.images))
        target = {'id': image_id, 'image_id': image_id}
        initial = workflow.backend.execute(workflow, 'detect_fov_initial', target, self.root/'detect', {})
        version = workflow.accept('detect_fov_initial', target, initial, workflow.store.current()['version'])
        scan = workflow.backend.execute(workflow, 'detect_fov_initial', target, self.root/'scan', {})
        directory = Path(scan['crops_dir'])
        sidecar = directory / 'crop.json'
        doc = read_json(sidecar)
        sidecar.unlink()
        write_json(directory/'fiber.json', {**doc, 'feature_id': 'fiber', 'crop_id': 'fiber', 'det_score': 0.9})
        write_json(scan['completion'], {'feature_id': 'fiber', 'count': 1})
        workflow.accept('feature_scan',
                        {**target, 'parents': [image_id], 'feature_id': 'fiber', 'fast_recovery': True},
                        scan, version)
        crops = {row['box_source']: row for row in workflow.store.current()['artifacts'].values()
                 if row['kind'] == 'crop'}
        return workflow, crops['yolo'], crops['glm']

    def _classification(self, crop, feature, name, reject=None):
        folder = self.root / name
        write_json(folder/'classification.json', {'feature_id': feature, 'crop_id': 'bare_sidecar_id',
                   'label': feature or '图像不可判', 'reject': reject, 'confidence': 'high',
                   'confidence_source': 'cnn', 'confidence_score': 0.91})
        return {'crop_id': crop['id'], 'classification': str(folder), 'status': 'done'}

    def test_projects_initial_scan_and_batch_from_same_snapshot(self):
        workflow, initial, scan = self._crops()
        batch = [self._classification(initial, 'single_crystal', 'initial_class'),
                 self._classification(scan, 'starch_grain', 'scan_class')]
        workflow.accept('classify_crops_batch',
                        {'image_id': initial['image_id'], 'crops': [initial, scan]},
                        {'classifications': batch}, workflow.store.current()['version'])
        snapshot = workflow.store.current()
        before = {p: p.read_bytes() for p in workflow.root.rglob('*.json')}
        audit = build_feature_audit(snapshot, out_dir=self.root/'audit', feature_ids=[])
        self.assertEqual(audit['schema'], 'feature-audit/v2')
        self.assertEqual(audit['summary']['n_input_views'], 1)
        self.assertEqual(audit['summary']['n_with_examples'], 0)
        self.assertEqual(audit['evidence_version'], snapshot['version'])
        self.assertTrue(audit['non_blocking'])
        self.assertEqual(audit['feature_ids'], ['fiber', 'single_crystal'])
        rows = {row['feature_id']: row for row in audit['candidates']}
        self.assertEqual(rows['single_crystal']['box_source'], 'yolo')
        self.assertEqual(rows['single_crystal']['status'], 'classified_target')
        self.assertEqual(rows['fiber']['box_source'], 'glm')
        self.assertEqual(rows['fiber']['status'], 'classified_non_target')
        self.assertEqual(rows['fiber']['crop_id'], scan['id'])
        self.assertEqual(rows['fiber']['negative_margin'], -0.91)
        self.assertTrue(all(row['confidence_source'] == 'cnn' for row in rows.values()))
        self.assertTrue(all('needle_score' not in row for row in rows.values()))
        self.assertEqual(read_json(self.root/'audit/feature_audit.json'), audit)
        self.assertEqual({p: p.read_bytes() for p in workflow.root.rglob('*.json')}, before)
        subset = build_feature_audit(snapshot, out_dir=self.root/'subset', feature_ids=['fiber'])
        self.assertEqual([row['crop_id'] for row in subset['candidates']], [scan['id']])

    def test_batch_audit_only_credits_individual_receipts(self):
        workflow, initial, scan = self._crops()
        action = next(a for a in workflow.propose() if a['tool'] == 'classify_crops_batch')

        def batch_reply(session, tool, target, work, context):
            write_json(work/'good'/'classification.json',
                       {'label': '方晶', 'feature_id': 'single_crystal', 'confidence': 'high'})
            return {'classifications': [{'crop_id': initial['id'], 'status': 'done', 'classification': str(work/'good')}],
                    'failures': [{'crop_id': scan['id'], 'status': 'failed', 'error_code': 'CLASSIFY_FAILED'}]}

        with patch.object(workflow.backend, 'execute', side_effect=batch_reply):
            op = workflow.execute(action)
        self.assertEqual(op['batch_outcomes'], {initial['id']: 'completed', scan['id']: 'unknown'})
        for legacy in (False, True):
            if legacy:
                entries = read_json(workflow.operations)
                del entries['entries'][op['operation_key']]['batch_outcomes']
                write_json(workflow.operations, entries)
            rows = feature_receipts(workflow.runtime, [], workflow.targets())
            self.assertEqual([(r['crop_id'],r['outcome']) for r in rows['single_crystal']], [(initial['id'],'completed')])
            self.assertEqual([(r['crop_id'],r['outcome']) for r in rows['fiber']], [(scan['id'],'unknown')])
            self.assertNotIn('secretory_tissue', rows)

    def test_excludes_stale_classification_and_reports_rejected_or_changed_results(self):
        workflow, initial, scan = self._crops()
        for crop in (initial, scan):
            row = self._classification(crop, '', crop['box_source'], reject='unreadable')
            workflow.accept('classify_target', crop, row, workflow.store.current()['version'])
        snapshot = workflow.store.current()
        filtered = build_feature_audit(snapshot, out_dir=self.root/'rejected')
        self.assertEqual(filtered['summary']['classification_unresolved'], 2)
        self.assertTrue(all(row['target_score'] is None for row in filtered['candidates']))

        sidecar = self.root/'replacement.json'
        write_json(sidecar, {**read_json(initial['path']), 'det_score': 0.95})
        replacement = {**initial, **workflow._art(initial['id'], 'crop', sidecar, initial['parents']),
                       'det_score': 0.95}
        workflow.store.commit(snapshot['version'], [replacement])
        snapshot = workflow.store.current()
        self.assertFalse(snapshot['artifacts']['classification:'+initial['id']]['valid'])
        write_json(self.root/'glm/classification.json', {'feature_id': 'fiber'})
        audit = build_feature_audit(snapshot, out_dir=self.root/'audit')
        rows = {row['crop_id']: row for row in audit['candidates']}
        self.assertEqual(rows[initial['id']]['status'], 'high_score_unclassified')
        self.assertEqual(rows[scan['id']]['status'], 'classification_unresolved')
        self.assertEqual(audit['issues'], [{'artifact_id': 'classification:'+scan['id'], 'error': 'ARTIFACT_CHANGED'}])

    def test_only_registered_masks_are_split_and_same_names_do_not_collide(self):
        workflow, initial, scan = self._crops()
        classified = self._classification(scan, 'fiber', 'scan_class')
        workflow.accept('classify_target', scan, classified, workflow.store.current()['version'])
        for crop in (initial, scan):
            folder = self.root / crop['box_source']
            masks = folder/'masks'
            (masks/'crops').mkdir(parents=True)
            shutil.copyfile(self.mask, masks/'delivery.png')
            shutil.copyfile(self.mask, masks/'crops/copied_crop.png')
            write_json(folder/'instances.json', {'instances': [{'igm_status': 'ok'}]})
            workflow.accept('segment_instance', crop,
                            {'segmentation': str(folder), 'quality': 'valid'}, workflow.store.current()['version'])
            shutil.copyfile(self.mask, masks/'not_registered.png')
        audit = build_feature_audit(workflow.store.current(), out_dir=self.root/'audit')
        self.assertEqual(audit['summary']['n_masks'], 2)
        self.assertEqual(audit['summary']['n_instances'], 4)
        names = {item['mask'] for row in audit['instances'] for item in row['instances']}
        self.assertEqual(len(names), 4)
        self.assertTrue(all(Path(path).is_file() for path in names))
        snapshot = workflow.store.current()
        sidecar = self.root/'replacement.json'
        write_json(sidecar, {**read_json(initial['path']), 'det_score': 0.95})
        replacement = workflow._art(initial['id'], 'crop', sidecar, initial['parents'],
                                    image_id=initial['image_id'])
        workflow.store.commit(snapshot['version'], [replacement])
        audit = build_feature_audit(workflow.store.current(), out_dir=self.root/'updated_audit')
        self.assertEqual(audit['summary']['n_masks'], 1)
        self.assertEqual(audit['instances'][0]['crop_id'], scan['id'])


if __name__ == '__main__':
    unittest.main()
