# -*- coding: utf-8 -*-
"""A/B 加权融合的离线回归。
该组用例由项目根 selftest 发现。涉及模型回复的用例使用可控输入，涉及进程的用例以各断言规定的执行边界为准。

主要定义：
- `_a_dragon`: 构造五视图、每折均给龙胆高概率的 A 预测夹具。
- ABFusionTest.test_reranking_cannot_keep_stale_confirmed_conclusion: 重排后不得保留过期的 confirmed 结论，需降为 insufficient_evidence 并复核。
- ABFusionTest.test_a_prior_lifts_dragon_above_zero_b: A 先验可抬升 B 分为 0 的候选，并校验 `fusion_weights` 与 `fused_score`。

关联：
- 直接依赖 「lib/decision/ab_fusion.py」。
- 统一执行见 tests/README.md。
"""
import unittest
from pathlib import Path

import paths
from rule_engine import KnowledgeBase

from TCM_Fusion.lib.decision.ab_fusion import apply_ab_fusion


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
