# -*- coding: utf-8 -*-
"""A/B 加权融合的离线回归。"""
import unittest
from pathlib import Path

import paths
from rule_engine import KnowledgeBase

from TCM_Fusion.lib.ab_fusion import apply_ab_fusion


def _a_dragon():
    views = []
    for i in range(1, 6):
        views.append({"status": "ok", "image": f"D:/case/view_0{i}.jpg", "input_sha256": f"sha{i}",
                      "fold_rankings": [[{"label": "龙胆", "probability": 0.99},
                                         {"label": "党参", "probability": 0.01}]] * 5})
    return {"status": "ok", "ranking": [], "view_predictions": views}


class ABFusionTest(unittest.TestCase):
    def test_reranking_cannot_keep_stale_confirmed_conclusion(self):
        kb=KnowledgeBase(Path(paths.JUDGE_DIR)/'data/herb_criteria.json')
        v={'verdict':'confirmed','needs_review':False,'conclusion':{'id':'党参'},
           'ranking':[{'id':'党参','confidence':.9,'score':.9},
                      {'id':'龙胆','confidence':.2,'score':.2}]}
        out=apply_ab_fusion(v,_a_dragon(),kb,{},.9,.1)
        self.assertEqual(out['ranking'][0]['id'],'龙胆')
        self.assertEqual(out['ranking'][0]['confidence'],.2)
        self.assertEqual(out['verdict'],'insufficient_evidence')
        self.assertIsNone(out['conclusion']);self.assertTrue(out['needs_review'])
    def test_a_prior_lifts_dragon_above_zero_b(self):
        kb = KnowledgeBase(Path(paths.JUDGE_DIR) / "data/herb_criteria.json")
        verdict = {"verdict": "insufficient_evidence", "conclusion": None,
                   "ranking": [
                       {"id": "龙胆", "display": "龙胆", "score": -0.0487, "confidence": 0.0,
                        "support_count": 1},
                       {"id": "党参", "display": "党参", "score": 0.0, "confidence": 0.0,
                        "support_count": 0},
                   ]}
        out = apply_ab_fusion(verdict, _a_dragon(), kb, {}, 0.6, 0.4)
        self.assertEqual(out["ranking"][0]["id"], "龙胆")
        self.assertAlmostEqual(out["ranking"][0]["a_probability"], 0.99, places=4)
        self.assertAlmostEqual(out["ranking"][0]["fused_score"], 0.594, places=4)
        self.assertEqual(out["fusion_weights"]["a"], 0.6)
        self.assertEqual(out["fusion_weights"]["b"], 0.4)


if __name__ == "__main__":
    unittest.main()
