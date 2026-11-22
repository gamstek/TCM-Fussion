"""验证检测原像素视窗、分割交付和实测坐标的绑定。"""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

import contracts
import paths
from igm_segment import apply_box_prior
from TCM_Fusion.cli.workflow_worker import main as detect_worker
from TCM_Fusion.lib.runtime.storage import file_hash, read_json, write_json
from TCM_Fusion.lib.vision.feature_window import feature_window
from 图像分割.cli.igm_segment_batch import assemble, load_crops


def write_png(path, image):
    ok, data = cv2.imencode('.png', image)
    if not ok:
        raise ValueError(f'无法编码本地用例 {path}')
    path.write_bytes(data.tobytes())


class FeatureWindowTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.source = self.root / 'view.png'
        self.image = np.full((360, 450, 3), (50, 20, 140), np.uint8)
        self.image[:, 210:] = (40, 170, 25)
        write_png(self.source, self.image)

    def test_detector_keeps_original_box_and_native_pixels(self):
        box = [70.4, 100.5, 340.7, 140.4]
        dets = self.root / 'yolo.json'
        write_json(dets, {'image_size': [450, 360], 'roi': None,
                         'source_sha256': file_hash(self.source),
                         'dets': [{'label': '导管', 'score': .9, 'tight_bbox': box}]})
        context = self.root / 'context.json'
        out = self.root / 'detection'
        write_json(context, {'tool': 'detect_fov_initial', 'source': str(self.source),
                            'output': str(out), 'boxes': str(dets), 'feature_crop_policy': 'structure_v1'})
        with patch('detect_core.lookup_px_per_um', return_value='4'):
            self.assertEqual(detect_worker(['--context', str(context)]), 0)
        crop = contracts.load_crop_metas(str(out / 'crops'))[0]
        self.assertEqual(crop.tight_box, box)
        self.assertEqual(crop.raw['feature_window']['axis_basis'], 'box_axis')
        self.assertEqual(crop.raw['feature_window']['source_sha256'], file_hash(self.source))
        x0, y0, x1, y1 = crop.loose_box
        self.assertAlmostEqual(crop.tight_box_local[0] + x0, box[0])
        self.assertGreater(box[0] - x0, box[1] - y0)
        decoded = cv2.imdecode(np.fromfile(crop.png, np.uint8), cv2.IMREAD_COLOR)
        np.testing.assert_array_equal(decoded, self.image[y0:y1, x0:x1])
        from geom import loose_box
        a, b, c, d = loose_box(box, 450, 360)
        p, q, r, s = crop.raw['feature_window']['classification_box_local']
        np.testing.assert_array_equal(decoded[q:s, p:r], self.image[b:d, a:c])

    def test_glm_minimum_size_only_expands_observation_window(self):
        seed = [112, 124, 131, 147]
        reply = {'dets': [{'label': '淀粉粒', 'score': .9, 'tight_bbox': seed}]}
        for policy in ('legacy', 'structure_v1'):
            with self.subTest(policy=policy):
                out = self.root / policy
                context = self.root / 'context.json'
                write_json(context, {'tool': 'locate_feature', 'source': str(self.source),
                           'output': str(out), 'feature_id': 'starch_grain', 'roi': [40, 40, 300, 300],
                           'px_per_um': 4, 'locate_expand': 1.5, 'locate_min_box_side': 90,
                           'feature_crop_policy': policy})
                # 独立回执在两个配置中复用，避免把 worker 对 det 的改写传给下一轮。
                result = {'dets': [dict(reply['dets'][0], tight_bbox=seed.copy())]}
                with patch('detect_core.detect_once', return_value=(result, None)) as locate, \
                     patch('ls_common.load_config', return_value={'vlm': {}}), \
                     patch('ls_common.resolve_key', return_value='offline'):
                    self.assertEqual(detect_worker(['--context', str(context)]), 0)
                    locate.assert_called_once()
                crop = contracts.load_crop_metas(str(out / 'crops'))[0]
                if policy == 'structure_v1':
                    self.assertEqual(crop.tight_box, seed)
                    self.assertEqual(crop.raw['feature_window']['profile'], 'outline')
                    self.assertGreaterEqual(crop.loose_box[2] - crop.loose_box[0], 90)
                else:
                    self.assertEqual(crop.tight_box[2] - crop.tight_box[0], 90)
                    self.assertNotIn('feature_window', crop.raw)

    def test_diagonal_texture_is_used_only_to_choose_axis_padding(self):
        y, x = np.indices((220, 220))
        stripes = np.where((x-y) % 24 < 11, 45, 150).astype(np.uint8)
        box = [40, 40, 180, 180]
        oriented, meta = feature_window(stripes, box, 'fiber')
        self.assertEqual(meta['axis_basis'], 'texture_axis')
        self.assertAlmostEqual(meta['axis_deg'], 45, delta=5)
        flat, meta = feature_window(np.full_like(stripes, 70), box, 'fiber')
        self.assertIsNone(meta['axis_deg'])
        self.assertLess(oriented[0], flat[0])
        self.assertLess(oriented[1], flat[1])
        self.assertIsNone(feature_window(stripes, box, 'epidermis_cell'))

    def test_region_component_reports_cut_pixels_of_the_selected_target(self):
        mask = np.zeros((100, 120), np.uint8)
        mask[38:53, 20:109] = 255
        final, meta = apply_box_prior(mask, [24, 30, 87, 60], association='region_component',
                                      feature_id='vessel')
        self.assertEqual(meta['associated_clipped_px'], 240)
        self.assertFalse(final[:, -1].any())

    def run_measurement(self, fixtures, *, remove_crop=None):
        crops, delivery = self.root / 'crops', self.root / 'segmentation'
        crops.mkdir()
        results = {}
        for cid, x, y, ppu, mask, clipped, shift_error in fixtures:
            crop = self.image[y:y+72, x:x+72]
            write_png(crops / f'{cid}.png', crop)
            side = contracts.crop_sidecar(cid, cid+'.png', str(self.source), 'view',
                     'starch_grain', '淀粉粒', '', ppu, [x+20, y+13, x+52, y+49],
                     [20+shift_error, 13, 52, 49], [x, y, x+72, y+72], '72x72')
            side['feature_window'] = {'schema': contracts.FEATURE_WINDOW_SCHEMA,
                        'source_size_wh': [450, 360], 'source_sha256': file_hash(self.source)}
            write_json(crops / f'{cid}.json', side)
            raw = delivery / 'igm' / cid
            raw.mkdir(parents=True)
            write_png(raw / 'mask.png', mask)
            results[cid] = {'ok': True, 'meta': {'status': 'ok', 'associated_clipped_px': clipped}}
        assemble(load_crops(str(crops)), results, str(delivery))
        masks = delivery / 'masks'
        if remove_crop:
            (masks / 'crops' / f'{remove_crop}.json').unlink()
        output, overlays = self.root / 'measurement', self.root / 'overlays'
        proc = subprocess.run([sys.executable, '-X', 'utf8', '-B', str(Path(paths.GEOM_DIR) / 'measure.py'),
                               '--masks', str(masks), '--out-prefix', str(output), '--overlay-dir', str(overlays)],
                               env=paths.python_environment(), capture_output=True, text=True, encoding='utf-8',
                               timeout=25)
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        rows = [json.loads(line) for line in (self.root / 'measurement_measurements.jsonl').read_text(
                encoding='utf-8').splitlines()]
        return overlays, rows

    def test_measurement_binds_each_same_size_mask_to_its_own_crop_and_scale(self):
        mask = np.zeros((72, 72), np.uint8)
        cv2.circle(mask, (33, 28), 10, 255, -1)
        overlays, (a, b) = self.run_measurement([
            ('a', 35, 65, 4, mask, 0, 0), ('b', 250, 105, 8, mask, 0, 0)])
        self.assertEqual([r['crop_frame']['status'] for r in (a, b)], ['valid', 'valid'])
        self.assertEqual([r['crop_frame']['crop_id'] for r in (a, b)], ['a', 'b'])
        self.assertEqual([r['calibration_px_per_um'] for r in (a, b)], [4.0, 8.0])
        self.assertAlmostEqual(a['primary_value_um'], 2 * b['primary_value_um'])
        self.assertEqual(a['crop_frame']['mask_bbox_source_xyxy'], [58, 83, 79, 104])
        self.assertEqual(b['crop_frame']['mask_bbox_source_xyxy'], [273, 123, 294, 144])
        for row, offset in ((a, (35, 65)), (b, (250, 105))):
            self.assertEqual(row['crop_frame']['feret_p1_px'],
                             [x+y for x,y in zip(row['feret_p1_px'], offset)])
        outs = sorted(overlays.glob('*.jpg'))
        left, right = [cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_COLOR)[69, 3] for p in outs]
        np.testing.assert_allclose(left, self.image[0, 0], atol=4)
        np.testing.assert_allclose(right, self.image[0, 250], atol=4)

    def test_measurement_marks_crop_source_and_association_edge_cases(self):
        intact = np.zeros((72, 72), np.uint8)
        cv2.circle(intact, (34, 28), 9, 255, -1)
        edge = np.zeros_like(intact)
        cv2.circle(edge, (2, 28), 9, 255, -1)
        _, rows = self.run_measurement([
            ('a_interior_edge', 30, 20, 4, edge, 0, 0),
            ('b_source_edge', 0, 20, 4, edge, 0, 0),
            ('c_association_cut', 75, 30, 4, intact, 5, 0),
            ('d_bad_offset', 75, 30, 4, intact, 0, 4),
            ('e_missing_crop', 75, 30, 4, intact, 0, 0),
        ], remove_crop='e_missing_crop')
        a, b, c, d, e = [r['crop_frame'] for r in rows]
        self.assertEqual([f['status'] for f in (a,b,c,d,e)],
                         ['possible_truncation']*3 + ['invalid_mapping', 'not_bound'])
        self.assertEqual(a['crop_edges'], ['left'])
        self.assertEqual(a['source_edges'], [])
        self.assertEqual(b['source_edges'], ['left'])
        self.assertEqual(c['associated_clipped_px'], 5)
        self.assertTrue(all(r['primary_value_um'] is None and 'igm_review' in r['flags'] for r in rows))
