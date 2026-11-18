# -*- coding: utf-8 -*-
"""候选范围门回归：B 只能从 A Top-K 取候选，非 A 必须严格证明。
该组用例由项目根 selftest 发现。涉及模型回复的用例使用可控输入，涉及进程的用例以各断言规定的执行边界为准。

主要定义：
- `_a_dragon`/`_verdict`: 构造五视图 A 预测与含荜茇、龙胆的 B 排名夹具。
- CandidateScopeTest.test_a_topk_drops_non_a_candidate: `a_topk` 模式丢弃非 A Top-K 候选并记录到 `candidate_scope.dropped`。
- test_open_strict_drops_generic_only_non_a、test_out_of_range_non_a_is_not_strict_evidence: 开放严格模式下泛化或超范围证据不足以保留非 A 候选。
- test_preserves_confirmed_candidate_when_scope_does_not_change_top: 范围过滤未改变排名首位时保留 confirmed 结论。

关联：
- 直接依赖 「lib/decision/candidate_scope.py」。
- 统一执行见 tests/README.md。
"""
import unittest
from pathlib import Path

import paths
from rule_engine import KnowledgeBase

from TCM_Fusion.lib.decision.candidate_scope import apply_candidate_scope


def _a_dragon():
    views = []
    for i in range(1, 6):
        views.append({"status": "ok", "image": f"D:/case/view_0{i}.jpg", "input_sha256": f"sha{i}",
                      "fold_rankings": [[{"label": "龙胆", "probability": 0.99},
                                         {"label": "党参", "probability": 0.01}]] * 5})
    return {"status": "ok", "ranking": [], "view_predictions": views}


def _verdict():
    return {"verdict": "insufficient_evidence", "conclusion": None,
            "ranking": [
                {"id": "荜茇", "display": "荜茇", "score": 0.3825, "confidence": 0.3825,
                 "support_count": 1,
                 "checks": [{"feature_id": "epidermis_cell", "role": "required",
                             "present": True, "state": "NO_REF"}]},
                {"id": "龙胆", "display": "龙胆", "score": -0.0487, "confidence": 0.0,
                 "support_count": 1,
                 "checks": [{"feature_id": "needle_crystal", "role": "required",
                             "present": False, "state": "NOT_DETECTED"},
                            {"feature_id": "epidermis_cell", "role": "supporting",
                             "present": True, "state": "NO_REF"}]},
            ]}


class CandidateScopeTest(unittest.TestCase):
    def setUp(self):
        self.kb = KnowledgeBase(Path(paths.JUDGE_DIR) / "data/herb_criteria.json")
        self.a = _a_dragon()
        self.policy = {"top_k": 3}

    def test_a_topk_drops_non_a_candidate(self):
        out = apply_candidate_scope(_verdict(), self.a, self.kb, {}, self.policy, "a_topk")
        ids = [r["id"] for r in out["ranking"]]
        self.assertNotIn("荜茇", ids)
        self.assertIn("龙胆", ids)
        self.assertEqual(out["candidate_scope"]["dropped"][0]["id"], "荜茇")

    def test_open_strict_drops_generic_only_non_a(self):
        out = apply_candidate_scope(_verdict(), self.a, self.kb, {}, self.policy, "open_strict")
        self.assertNotIn("荜茇", [r["id"] for r in out["ranking"]])

    def test_preserves_confirmed_candidate_when_scope_does_not_change_top(self):
        v=_verdict();v['ranking']=[v['ranking'][1]]
        v.update(verdict='confirmed',conclusion={'id':'龙胆'},needs_review=False)
        out=apply_candidate_scope(v,self.a,self.kb,{},self.policy,'a_topk')
        self.assertEqual(out['verdict'],'confirmed')
        self.assertEqual(out['conclusion']['id'],'龙胆')

    def test_out_of_range_non_a_is_not_strict_evidence(self):
        v=_verdict();v['ranking'][0]['checks']=[{'feature_id':'needle_crystal',
            'role':'required','present':True,'state':'OUT_OF_RANGE'}]
        out=apply_candidate_scope(v,self.a,self.kb,{},self.policy,'open_strict')
        self.assertNotIn('荜茇',[r['id'] for r in out['ranking']])


if __name__ == "__main__":
    unittest.main()
