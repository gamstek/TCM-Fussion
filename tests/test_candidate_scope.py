# -*- coding: utf-8 -*-
"""候选范围门：B 只能从 A Top-K 取候选，非 A 必须严格证明。"""
import unittest
from pathlib import Path

import paths
from rule_engine import KnowledgeBase

from TCM_Fusion.lib.candidate_scope import apply_candidate_scope


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


if __name__ == "__main__":
    unittest.main()
