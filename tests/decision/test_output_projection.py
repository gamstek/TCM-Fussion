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

from TCM_Fusion.lib.decision.output_projection import project_output
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
