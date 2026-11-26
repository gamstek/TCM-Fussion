"""验证同次视觉复核的对照选择、图幅、可见结构契约和终止状态。

主要定义：IdentityReviewTests 覆盖实际审核库、当前 worker 和模拟下游接纳。
关联：lib/vision/identity_review.py、cli/classify_worker.py、test_stage2.py。
"""
import base64
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

import paths
from detect_core import FEATURE_ID
from prompts import VALID_LABELS
from TCM_Fusion.cli.classify_worker import main as classify_main
from TCM_Fusion.lib.runtime.storage import read_json, write_json
from TCM_Fusion.lib.vision.identity_review import CUES, finish_review, load_review_bank, review_plan
from TCM_Fusion.lib.workflow.workflow_session import WorkflowSession
from 目标检测.lib.classify_crop import _prep


def predictions(first, second, first_prob=.50, second_prob=.42):
    return {'topk': [{'feature_id': feature, 'prob': score}
                    for feature, score in ((first, first_prob), (second, second_prob))]}


class IdentityReviewTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.bank_path = Path(paths.PROJECT_ROOT) / 'config/identity_review_bank.json'
        self.bank = load_review_bank(self.bank_path)
        self.assertIsNotNone(self.bank)
        self.enterContext(patch('ls_common.load_config', return_value={'vlm': {}}))
        self.enterContext(patch('ls_common.resolve_key', return_value='offline-test'))

    def plan(self, first='vessel', second='fiber', detected='vessel', exclude=(), **probs):
        return review_plan(self.bank, predictions(first, second, **probs), detected, 'yolo', set(exclude),
                           margin_limit=.15)

    def test_reference_selection_obeys_review_status_conflict_and_margin(self):
        self.assertEqual([row['feature_id'] for row in self.plan()['references']], ['vessel', 'fiber'])
        conflict = self.plan(first='vessel', second='stone_cell', detected='fiber',
                             first_prob=.7, second_prob=.18)
        self.assertEqual([row['feature_id'] for row in conflict['references']], ['fiber', 'vessel'])
        self.assertTrue(conflict['detector_conflict'])
        self.assertEqual(conflict['selection_basis'], 'detector_conflict')
        far = self.plan(first_prob=.7, second_prob=.18)
        self.assertEqual(far['fallback'], 'no_close_alternative')
        missing = self.plan(first='stone_cell', second='vessel', detected='stone_cell')
        self.assertEqual(missing['fallback'], 'no_reviewed_pair')
        self.assertFalse(self.plan(first='other', second='vessel')['references'])

    def test_excludes_reference_from_current_input(self):
        for row in self.plan()['references']:
            with self.subTest(excluded=row['id']):
                result = self.plan(exclude=[row['source_sha256']])
                self.assertEqual(result['pair_id'], '01')
                self.assertEqual(result['fallback'], 'no_reviewed_pair')
                self.assertEqual(result['references'], [])

    def test_incomplete_approved_pair_does_not_borrow_from_other_groups(self):
        original = read_json(self.bank_path)
        path = self.root / 'bank.json'
        for invalid_ids in (None, ['pair_01_p', 'pair_04_p'], ['pair_01_p', 'missing']):
            with self.subTest(example_ids=invalid_ids):
                bank = {**original, 'pairs': [{**pair, 'example_ids': invalid_ids} if pair['id'] == '01' else pair
                                               for pair in original['pairs']]}
                write_json(path, bank)
                plan = review_plan(load_review_bank(path), predictions('vessel', 'fiber'), 'vessel', 'yolo',
                                   set(), margin_limit=.15)
                self.assertEqual(plan['fallback'], 'no_reviewed_pair')
                self.assertEqual(plan['references'], [])
        for row in original['examples']:
            if row['id'] == 'pair_01_n':
                row['review_status'] = 'pending_human_review'
        write_json(path, original)
        plan = review_plan(load_review_bank(path), predictions('vessel', 'fiber'), 'vessel', 'yolo',
                           set(), margin_limit=.15)
        self.assertEqual(plan['fallback'], 'no_reviewed_pair')

    def test_historical_bank_keeps_category_based_lookup(self):
        doc = read_json(self.bank_path)
        doc['schema'] = 'identity-review-bank/v1'
        doc['examples'] = [row for row in doc['examples'] if not row['id'].startswith('pair_')]
        path = self.root / 'history.json'
        write_json(path, doc)
        plan = review_plan(load_review_bank(path), predictions('vessel', 'fiber'), 'vessel', 'yolo',
                           set(), margin_limit=.15)
        self.assertEqual([r['id'] for r in plan['references']], ['vessel_changshan_8', 'fiber_muxiang_w20170172'])

    def run_worker(self, crop, plan, reply, *, brief=None):
        path = self.root / 'target.png'
        ok, encoded = cv2.imencode('.png', crop)
        self.assertTrue(ok)
        path.write_bytes(encoded.tobytes())
        ctx = self.root / 'worker.json'
        result = self.root / 'out'
        write_json(ctx, {'crop_png': str(path), 'source': str(path), 'crop_id': 'fixture',
                         'tight_box': None, 'output': str(result), 'identity_review': plan,
                         **({'visual_brief':brief} if brief else {})})
        with patch('classify_crop.call_vlm', return_value=(json.dumps(reply, ensure_ascii=False), None)) as call:
            self.assertEqual(classify_main(['--context', str(ctx)]), 0)
        call.assert_called_once()
        _cfg, _key, encoded, prompt = call.call_args.args
        image = cv2.imdecode(np.frombuffer(base64.b64decode(encoded), np.uint8), cv2.IMREAD_COLOR)
        return read_json(result / 'classification.json'), image, prompt

    def test_free_question_reaches_target_with_examples_and_preserves_identity_gate(self):
        brief = '目标下方能看清哪些纹理？'
        report = {'finding':'只见局部轮廓', 'location':'下方', 'readability':'partial'}
        reply = {'label':'导管','confidence':'high','target_part':'中央','visible_cue':'invalid',
                 'visual_observation':report}
        crop = np.full((146, 180, 3), 75, dtype=np.uint8)
        doc, _, prompt = self.run_worker(crop, self.plan(), reply, brief=brief)
        self.assertIn(brief, prompt)
        self.assertEqual(doc['visual_observation']['finding'], report['finding'])
        self.assertEqual(doc['visual_identity']['status'], 'pending')

    def test_worker_accepts_all_seven_reviewed_pairs_in_both_orders(self):
        groups = [
            ('fiber', 'vessel'), ('secretory_tissue', 'stone_cell'), ('fiber', 'stone_cell'),
            ('secretory_tissue', 'fiber'), ('cluster_crystal', 'starch_grain'),
            ('stone_cell', 'starch_grain'), ('single_crystal', 'starch_grain'),
        ]
        labels = {feature: label for label, feature in FEATURE_ID.items() if label in VALID_LABELS}
        crop = np.full((146, 180, 3), 75, dtype=np.uint8)
        for number, (positive, negative) in enumerate(groups, 1):
            group_id = f'{number:02d}'
            approved_ids = [f'pair_{group_id}_p', f'pair_{group_id}_n']
            for first, second, ids in [(positive, negative, approved_ids), (negative, positive, approved_ids[::-1])]:
                with self.subTest(group=group_id, first=first):
                    plan = self.plan(first=first, second=second, detected=first)
                    self.assertEqual(plan['pair_id'], group_id)
                    self.assertEqual([r['id'] for r in plan['references']], ids)
                    response = {'label': labels[first], 'confidence': 'high',
                                'target_part': '中央', 'visible_cue': CUES[first][0]}
                    doc, image, prompt = self.run_worker(crop, plan, response)
                    self.assertEqual(image.shape, _prep(crop).shape)
                    self.assertEqual(doc['visual_input']['mode'], 'target_with_examples')
                    self.assertEqual(doc['visual_input']['pair_id'], group_id)
                    self.assertEqual([r['id'] for r in doc['visual_input']['references']], ids)
                    self.assertEqual(doc['visual_identity']['status'], 'observed')
                    self.assertIn(f'P 是{labels[first]}正例，N 是{labels[second]}对照', prompt)

    def test_worker_sends_one_same_size_image_with_large_target_and_reports_part(self):
        crop = np.full((146, 180, 3), 75, dtype=np.uint8)
        response = {'label': '导管', 'confidence': 'high', 'target_part': '中央',
                    'visible_cue': 'vessel_pattern', 'reason': '中央管壁可见规则横纹'}
        doc, image, prompt = self.run_worker(crop, self.plan(), response)
        self.assertEqual(image.shape, _prep(crop).shape)
        audit = doc['visual_input']
        self.assertEqual(audit['mode'], 'target_with_examples')
        self.assertEqual([row['feature_id'] for row in audit['references']], ['vessel', 'fiber'])
        self.assertNotIn('path', audit['references'][0])
        x0, y0, x1, y1 = audit['target_box_px']
        self.assertGreaterEqual(x1 - x0, crop.shape[1])
        self.assertGreaterEqual(y1 - y0, crop.shape[0])
        self.assertGreater((x1 - x0) * (y1 - y0), image.shape[0] * image.shape[1] / 2)
        self.assertTrue(np.all(np.abs(image[y0+5:y1-5, x0+5:x1-5].astype(int) - 75) <= 2))
        self.assertIn('P 是导管正例', prompt)
        self.assertIn('环纹、网纹', prompt)
        self.assertEqual(doc['visual_identity'],
                         {'status': 'observed', 'target_part': '中央', 'visible_cue': 'vessel_pattern'})

    def test_fallback_keeps_target_pixels_when_detail_or_asset_is_insufficient(self):
        response = {'label': '导管', 'confidence': 'high', 'reason': '只见一条模糊长影'}
        empty_asset = self.root / 'empty.png'
        empty_asset.touch()
        empty_plan = self.plan()
        empty_plan['references'][0] = {**empty_plan['references'][0], 'path': str(empty_asset),
                                       'sha256': hashlib.sha256(b'').hexdigest()}
        for crop, fallback, plan in (
                (np.full((700, 720, 3), 80, dtype=np.uint8), 'target_detail', self.plan()),
                (np.full((3, 240, 3), 80, dtype=np.uint8), 'reference_detail', self.plan()),
                (np.full((160, 190, 3), 80, dtype=np.uint8), 'reference_unavailable', self.plan()),
                (np.full((160, 190, 3), 80, dtype=np.uint8), 'reference_unavailable', empty_plan)):
            with self.subTest(fallback=fallback):
                if fallback == 'reference_unavailable' and plan is not empty_plan:
                    plan['references'][0] = {**plan['references'][0], 'sha256': '0' * 64}
                doc, actual, prompt = self.run_worker(crop, plan, response)
                self.assertEqual(doc['visual_input']['fallback'], fallback)
                self.assertEqual(doc['visual_input']['references'], [])
                expected = _prep(crop)
                self.assertEqual(actual.shape, expected.shape)
                np.testing.assert_array_equal(actual, expected)
                self.assertNotIn('P 是', prompt)
                self.assertIn('环纹、网纹', prompt)
                self.assertEqual((doc['label'], doc['reject'], doc['feature_id']), ('身份待定', 'unreadable', ''))

    def test_classifier_demands_location_and_cue_from_target(self):
        valid = {'label': '纤维', 'feature_id': 'fiber', 'target_part': '左中',
                 'visible_cue': 'fiber_wall_lumen', 'reason': '左中连续纵壁及窄腔'}
        self.assertEqual(finish_review(dict(valid))['visual_identity']['status'], 'observed')
        for invalid in ({'target_part': 'N'}, {'visible_cue': 'vessel_pattern'}, {'target_part': None},
                        {'reason': '只见模糊长影'}):
            with self.subTest(invalid=invalid):
                self.assertEqual(finish_review({**valid, **invalid})['visual_identity']['status'], 'pending')
        self.assertEqual(finish_review({'reject': 'not_feature'})['visual_identity']['status'], 'not_feature')
        self.assertEqual(finish_review({'error': 'offline service error'}), {'error': 'offline service error'})

    def test_visual_pending_cannot_open_ordinary_crop_segmentation(self):
        session = object.__new__(WorkflowSession)
        session.config = {'identity_review_bank': 'fixture'}
        session.store = SimpleNamespace(current=lambda: {'artifacts': {'classification:c1': {
            'valid': True, 'visual_identity_status': 'pending', 'review_required': False}}})
        result = session.crop_segmentation_admission({'id': 'c1', 'box_quality': 'ok'})
        self.assertFalse(result['allowed'])
        self.assertEqual(result['reason'], 'visual_identity_pending')
