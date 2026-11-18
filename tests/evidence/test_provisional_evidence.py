"""暂定证据排序回归：分类只更新 provisional 视图，不改变正式判决与支持计数。
该组用例由项目根 selftest 发现。涉及模型回复的用例使用可控输入，涉及进程的用例以各断言规定的执行边界为准。

主要定义：
- ProvisionalEvidenceTests（pending 分类）: 分类命中只标 DETECTED_PENDING，正式分数与支持计数保持不变。
- ProvisionalEvidenceTests（状态边界）: UNCAL/NO_REF 不算已核验；低置信且无部分线索记为 UNKNOWN。
- ProvisionalEvidenceTests（冲突判定）: 同一 crop 被分成互斥特征时标记 CONFLICT，越界结论同样是冲突。
- ProvisionalEvidenceTests（快照优先级）: 当前有效证据快照优先于历史操作记录。

关联：
- 直接依赖 「lib/evidence/provisional_evidence.py」。
- 统一执行见 tests/README.md。
"""

import json
import tempfile
import unittest
from pathlib import Path

from TCM_Fusion.lib.evidence.provisional_evidence import build


class ProvisionalEvidenceTests(unittest.TestCase):
    def test_classification_marks_pending_without_changing_formal(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            bundle = root / 'op'
            (bundle / 'classification').mkdir(parents=True)
            (bundle / 'classification' / 'classification.json').write_text(json.dumps({
                'crop_id': 'fiber_01', 'feature_id': 'fiber', 'label': '纤维',
                'confidence': 'high', 'confidence_source': 'cnn', 'reason': '可见纤维'}), encoding='utf-8')
            verdict = {'ranking': [{
                'id': '连翘', 'display': '连翘', 'a_probability': 0.6,
                'score': -0.06, 'checks': [
                    {'feature_id': 'fiber', 'role': 'required', 'tier': 'P1',
                     'present': False, 'state': 'NOT_DETECTED', 'weight': 0.5}]}]}
            result = {'operations': {'op': {'tool': 'classify_target', 'status': 'done',
                                             'bundle': {'root': str(bundle)}}}}
            out = build(verdict, result)
            self.assertEqual(out['formal_top_candidate'], '连翘')
            row = out['ranking'][0]
            self.assertEqual(row['formal_score'], -0.06)
            self.assertEqual(row['formal_support_count'], 0)
            self.assertEqual(row['criterion_states'][0]['state'], 'DETECTED_PENDING')
            self.assertEqual(row['criterion_states'][0]['instance_ids'], ['fiber_01'])
            self.assertAlmostEqual(row['provisional_score'], 0.6 * 0.6 + 0.4 * 0.5)


    def test_uncal_and_no_ref_are_not_verified(self):
        for state in ('UNCAL', 'NO_REF'):
            verdict = {'ranking': [{'id': '连翘', 'a_probability': 0.6, 'score': 0.1, 'support_count': 1,
                                    'checks': [{'feature_id': 'fiber', 'role': 'required', 'tier': 'P1',
                                                'present': True, 'state': state, 'weight': 0.5}]}]}
            out = build(verdict, {'operations': {}})
            self.assertEqual(out['ranking'][0]['criterion_states'][0]['state'], 'DETECTED_PENDING')
            self.assertEqual(out['ranking'][0]['formal_support_count'], 1)

    def test_low_confidence_without_partial_clue_is_unknown(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            bundle = root / 'op'
            (bundle / 'classification').mkdir(parents=True)
            (bundle / 'classification' / 'classification.json').write_text(json.dumps({
                'crop_id': 'fiber_01', 'feature_id': 'fiber', 'label': '纤维',
                'confidence': 'low', 'reason': '分类置信不足'}), encoding='utf-8')
            verdict = {'ranking': [{'id': '连翘', 'a_probability': 0.6, 'score': 0.0, 'support_count': 0,
                                    'checks': [{'feature_id': 'fiber', 'role': 'required', 'tier': 'P1',
                                                'present': False, 'state': 'NOT_DETECTED', 'weight': 0.5}]}]}
            out = build(verdict, {'operations': {'op': {'tool': 'classify_target', 'status': 'done',
                                                        'bundle': {'root': str(bundle)}}}})
            self.assertEqual(out['ranking'][0]['criterion_states'][0]['state'], 'UNKNOWN')

    def test_same_crop_conflicting_features_is_conflict(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            bundle = root / 'op'
            (bundle / 'classification').mkdir(parents=True)
            (bundle / 'classification' / 'classification.json').write_text(json.dumps({
                'crop_id': 'c1', 'feature_id': 'fiber', 'label': '纤维',
                'confidence': 'high', 'reason': '可见纤维'}), encoding='utf-8')
            verdict = {'ranking': [{'id': '连翘', 'a_probability': 0.6, 'score': 0.0, 'support_count': 0,
                                    'checks': [{'feature_id': 'fiber', 'role': 'required', 'tier': 'P1',
                                                'present': False, 'state': 'NOT_DETECTED', 'weight': 0.5}]}]}
            result = {'operations': {'op': {'tool': 'classify_target', 'status': 'done',
                                             'bundle': {'root': str(bundle)}}}}
            classes = [{'crop_id': 'c1', 'feature_id': 'fiber', 'label': '纤维', 'confidence': 'high', 'reason': '可见纤维'},
                       {'crop_id': 'c1', 'feature_id': 'stone_cell', 'label': '石细胞', 'confidence': 'high', 'reason': '可见石细胞'}]
            from TCM_Fusion.lib.evidence import provisional_evidence as pe
            best, conflicts = pe._best_classes(classes)
            state, _ = pe._tier_for(verdict['ranking'][0]['checks'][0], best, conflicts)
            self.assertEqual(state, 'CONFLICT')


    def test_current_valid_snapshot_takes_precedence_over_operation_history(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            bundle = root / 'op'
            (bundle / 'classification').mkdir(parents=True)
            (bundle / 'classification' / 'classification.json').write_text(json.dumps({
                'crop_id': 'fiber_01', 'feature_id': 'fiber', 'label': '纤维',
                'confidence': 'high', 'reason': '可见纤维'}), encoding='utf-8')
            verdict = {'ranking': [{'id': '连翘', 'a_probability': 0.6, 'score': 0.0, 'support_count': 0,
                                    'checks': [{'feature_id': 'fiber', 'role': 'required', 'tier': 'P1',
                                                'present': False, 'state': 'NOT_DETECTED', 'weight': 0.5}]}]}
            result = {'operations': {'op': {'tool': 'classify_target', 'status': 'done',
                                             'bundle': {'root': str(bundle)}}},
                      'valid_classifications': []}
            out = build(verdict, result)
            self.assertEqual(out['ranking'][0]['criterion_states'][0]['state'], 'NOT_DETECTED')

    def test_explicit_out_of_range_is_conflict_not_partial(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            bundle = root / 'op'
            (bundle / 'classification').mkdir(parents=True)
            (bundle / 'classification' / 'classification.json').write_text(json.dumps({
                'crop_id': 'fiber_01', 'feature_id': 'fiber', 'label': '纤维',
                'confidence': 'high'}), encoding='utf-8')
            verdict = {'ranking': [{'id': '连翘', 'a_probability': 0.6, 'score': 0.0, 'checks': [
                {'feature_id': 'fiber', 'role': 'required', 'tier': 'P1',
                 'present': True, 'state': 'OUT_OF_RANGE', 'weight': 0.5}]}]}
            out = build(verdict, {'operations': {'op': {'tool': 'classify_target',
                                                        'bundle': {'root': str(bundle)}}}})
            self.assertEqual(out['ranking'][0]['criterion_states'][0]['state'], 'CONFLICT')


if __name__ == '__main__':
    unittest.main()
