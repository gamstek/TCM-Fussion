"""图片示例审计：contact sheet 与分数规范化；不调用真实 VLM。"""
import tempfile
import unittest
import json
from pathlib import Path

from PIL import Image

import paths
from TCM_Fusion.lib.image_exemplar_audit import build_contact_sheet, normalize_audit


class ImageExemplarAuditTests(unittest.TestCase):
    def test_contact_sheet_and_normalization(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / 'target.jpg'
            pos = root / 'pos.jpg'
            neg = root / 'neg.jpg'
            Image.new('RGB', (64, 48), 'white').save(target)
            Image.new('RGB', (64, 48), 'red').save(pos)
            Image.new('RGB', (64, 48), 'blue').save(neg)
            sheet = build_contact_sheet(target, [pos], [neg], root / 'contact.jpg')
            self.assertTrue(sheet.is_file())
            self.assertGreater(sheet.stat().st_size, 0)
            result = normalize_audit({'needle_score': 0.8, 'negative_similarity': 0.3,
                                      'audit_label': 'needle_crystal', 'confidence': 'medium'},
                                     target_id='crop:x', positive_count=1, negative_count=1)
            self.assertEqual(result['needle_score'], 0.8)
            self.assertEqual(result['negative_margin'], 0.5)
            self.assertEqual(result['review_status'], 'auto_unreviewed')
            self.assertTrue(result['non_blocking'])

    def test_scores_are_clamped(self):
        result = normalize_audit({'needle_score': 2, 'negative_similarity': -1,
                                  'negative_margin': 9}, positive_count=1, negative_count=1)
        self.assertEqual(result['needle_score'], 1.0)
        self.assertEqual(result['negative_similarity'], 0.0)
        self.assertEqual(result['negative_margin'], 1.0)

    def test_generic_feature_uses_target_score(self):
        result = normalize_audit({'target_score': 0.7, 'negative_similarity': 0.2,
                                  'audit_label': 'fiber', 'confidence': 'medium'},
                                 feature_id='fiber', positive_count=1, negative_count=1)
        self.assertEqual(result['feature_id'], 'fiber')
        self.assertEqual(result['target_score'], 0.7)
        self.assertNotIn('needle_score', result)

    def test_generic_bank_has_14_targets_and_fiber_entry(self):
        path = Path(paths.PROJECT_ROOT) / 'TCM_Fusion/config/feature_exemplar_bank.json'
        bank = json.loads(path.read_text(encoding='utf-8'))
        self.assertEqual(len(bank['targets']), 14)
        self.assertIn('fiber', bank['targets'])
        self.assertTrue(bank['targets']['fiber']['positives'])
        self.assertTrue(bank['targets']['fiber']['negatives'])


if __name__ == '__main__':
    unittest.main()
