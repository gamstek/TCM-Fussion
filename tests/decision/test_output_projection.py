# -*- coding: utf-8 -*-
"""always-output 投影的离线回归；不联网、不执行阶段。
该组用例由项目根 selftest 发现。涉及模型回复的用例使用可控输入，涉及进程的用例以各断言规定的执行边界为准。

主要定义：
- `_a_prediction`/`_verdict`/`_result`: 构造五视图 A 预测、B 排名（含 group 与 NO_REF）与观察记录夹具。
- OutputProjectionTest.test_tie_uses_a_majority_and_keeps_strong_observations: 平局按 A 折多数决，并保留强观察及其支持候选。
- test_confirmed_maps_to_confirmed、test_confirmed_with_review_downgrades_to_tentative: verdict 到对外状态的映射与复核时降级。
- test_single_view_a_ranking_fallback、test_a_majority_does_not_jump_an_untied_lower_candidate: 单视图 ranking 回退与 A 多数不得越过未平局候选。
- test_missing_required_is_reported: 缺失 required 特征写入 `missing_required`。

关联：
- 直接依赖 「lib/decision/output_projection.py」、「tests/support/result_builders.py」。
- 统一执行见 tests/README.md。
"""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import contracts
from TCM_Fusion.lib.decision.output_projection import project_output
from TCM_Fusion.lib.runtime.storage import write_json
from TCM_Fusion.tests.support.result_builders import _a_prediction, _verdict, _result


class OutputProjectionTest(unittest.TestCase):
    def test_tie_uses_a_majority_and_keeps_strong_observations(self):
        out = project_output("case", _verdict(), _result(), _a_prediction())
        self.assertEqual(out["status"], "tentative")
        self.assertEqual(out["top_candidate"]["id"], "知母")
        self.assertTrue(out["tie_break"]["applied"])
        self.assertEqual(out["tie_break"]["a_fold_top1"], 25)
        self.assertEqual(out["ambiguous_with"], ["射干", "G2"])
        self.assertTrue(out["review_required"])
        self.assertTrue(out["audit"]["internal_needs_review"])
        self.assertEqual(len(out["strong_observations"]), 2)
        self.assertEqual(out["strong_observations"][0]["view"], "view_01")
        self.assertIn("菟丝子", out["strong_observations"][0]["supporting_candidates"])
        self.assertEqual(out["strong_observations"][1]["scope"], "observed_fov_only")
        self.assertEqual(out["strong_observations"][1]["qualification"], "presence_only")
        self.assertEqual(out["strong_observations"][1]["supporting_candidates"], [])

    def test_projection_uses_current_crops_and_batch_identity_with_scan_coverage(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / 'crop.json'
            write_json(path, contracts.crop_sidecar('crop', str(root / 'crop.png'), str(root / 'view.png'),
                'view', 'fiber', '纤维', '', 2, [0, 0, 8, 8], [0, 0, 8, 8], [0, 0, 8, 8], '8x8'))
            tile_a, tile_b, tile_c = [0, 0, 8, 8], [8, 0, 16, 8], [16, 0, 24, 8]
            write_json(root / 'scan.json', {'tiles': [tile_a, tile_b], 'planned_tiles': [tile_a, tile_b, tile_c],
                        'errors': [{'tile': 1}], 'count': 1})
            result = _result()
            result['operations'] = {'scan': {'tool': 'feature_scan', 'status': 'done', 'image_id': 'image:1',
                'bundle': {'root': str(root), 'paths': {'completion': 'scan.json'}}}}
            result['valid_classifications'] = [{'crop_id': 'crop', 'tool': 'classify_crops_batch',
                                                'feature_id': 'fiber', 'identity_verified': True}]
            verdict = _verdict()
            verdict['ranking'] = [{**verdict['ranking'][0], 'score': 0.3, 'fused_score': 0.3, 'b_score': -0.24}]
            crop = {'id': 'crop:1', 'valid': True, 'kind': 'crop', 'image_id': 'image:1',
                    'path': str(path), 'detection_tool': 'feature_scan'}
            output = project_output('case', verdict, result, _a_prediction(),
                                    artifacts={'crop:1': crop, 'old': {**crop, 'valid': False}})
            self.assertEqual([row['tool'] for row in output['detections']], ['feature_scan'])
            self.assertEqual(output['classifications'], result['valid_classifications'])
            scan, = output['feature_scans']
            self.assertEqual(scan['status'], 'partial')
            self.assertEqual(scan['successful_tiles'], [tile_a])
            self.assertEqual(scan['failed_tiles'], [tile_b])
            self.assertEqual(scan['unsearched_tiles'], [tile_c])
            self.assertEqual(output['top_candidate']['raw_score'], -0.24)
            self.assertEqual(output['top_candidate']['rule_score'], verdict['ranking'][0]['confidence'])
            self.assertNotEqual(output['status'], 'partial')
            self.assertEqual(output['top_candidate']['fused_score'], 0.3)

    def test_confirmed_maps_to_confirmed(self):
        verdict = _verdict()
        verdict["verdict"] = "confirmed"
        verdict["needs_review"] = False
        verdict["ranking"] = [verdict["ranking"][2]]
        out = project_output("case", verdict, _result(), _a_prediction())
        self.assertEqual(out["status"], "confirmed")
        self.assertEqual(out["top_candidate"]["id"], "知母")
        self.assertFalse(out["audit"]["internal_needs_review"])

    def test_confirmed_with_review_downgrades_to_tentative(self):
        verdict = _verdict()
        verdict["verdict"] = "confirmed"
        verdict["needs_review"] = True
        verdict["ranking"] = [verdict["ranking"][2]]
        out = project_output("case", verdict, _result(), _a_prediction())
        self.assertEqual(out["status"], "tentative")
        self.assertTrue(out["audit"]["internal_needs_review"])
        self.assertIn("not promoted", out["audit"]["status_note"])

    def test_single_view_a_ranking_fallback(self):
        a = {"status": "ok", "ranking": [{"label": "知母", "probability": 0.99}]}
        verdict = _verdict()
        out = project_output("case", verdict, _result(), a)
        self.assertEqual(out["tie_break"]["a_majority"], "知母")
        self.assertEqual(out["top_candidate"]["id"], "知母")

    def test_missing_required_is_reported(self):
        verdict = _verdict()
        verdict["ranking"] = [{"id": "知母", "display": "知母", "kind": "herb", "score": 0.0,
                               "confidence": 0.0, "support_count": 0,
                               "checks": [{"feature_id": "needle_crystal", "role": "required",
                                           "present": False, "state": "NOT_DETECTED"}]}]
        out = project_output("case", verdict, _result(), _a_prediction())
        self.assertEqual(out["top_candidate"]["id"], "知母")
        self.assertEqual(out["missing_required"][0]["feature_id"], "needle_crystal")

    def test_a_majority_does_not_jump_an_untied_lower_candidate(self):
        verdict = _verdict()
        verdict["ranking"] = [
            {"id": "辣蓼", "display": "辣蓼", "kind": "herb", "score": 0.0, "confidence": 0.0,
             "support_count": 0, "checks": []},
            {"id": "青黛", "display": "青黛", "kind": "herb", "score": 0.0, "confidence": 0.0,
             "support_count": 0, "checks": []},
            {"id": "龙胆", "display": "龙胆", "kind": "herb", "score": -0.24, "confidence": 0.0,
             "support_count": 0, "checks": []},
        ]
        out = project_output("case", verdict, _result(), _a_prediction())
        self.assertEqual(out["top_candidate"]["id"], "辣蓼")
        self.assertFalse(out["tie_break"]["applied"])


if __name__ == "__main__":
    unittest.main()
