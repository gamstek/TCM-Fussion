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
from rule_engine import EvidenceCard, KnowledgeBase, evaluate_candidate

from TCM_Fusion.lib.decision.ab_fusion import apply_ab_fusion
from TCM_Fusion.lib.decision.rule_scoring import (
    decide, disjoint_counterevidence, score_candidate, rank_and_decide,
)


def check(fid, role, contribution, state='NO_REF', bounds=None, weight=1, scene=1):
    return dict(feature_id=fid, role=role, score_contribution=contribution, weight=weight,
                state=state, present=state != 'NOT_DETECTED', quant_bounds=bounds,
                scene_factor=scene, penalty=.5 if state == 'OUT_OF_RANGE' else 0)


def scored(*checks, name='甲', **options):
    return score_candidate(dict(id=name, display=name, kind='herb', checks=checks,
        score=sum(c['score_contribution'] for c in checks), wrong_present=[], absent_required=[]),
        **options)


def _a_dragon():
    views = []
    for i in range(1, 6):
        views.append({"status": "ok", "image": f"D:/case/view_0{i}.jpg", "input_sha256": f"sha{i}",
                      "fold_rankings": [[{"label": "龙胆", "probability": 0.99},
                                         {"label": "党参", "probability": 0.01}]] * 5})
    return {"status": "ok", "ranking": [], "view_predictions": views}


class ABFusionTest(unittest.TestCase):
    def test_verified_support_is_bounded_and_independent(self):
        req = check('needle', 'required', .6)
        sup = check('skin', 'supporting', .255, weight=.5)
        base = scored(req)
        extra = scored(req, sup)
        self.assertEqual(base['rule_score'], .48)
        self.assertEqual(extra['rule_score'], .504382)
        self.assertEqual(extra['score_components']['supporting_bonus'], .024382)
        self.assertEqual(scored(req, sup, sup, check('needle','supporting',1))['rule_score'],
                         extra['rule_score'])
        self.assertGreater(scored(req,sup,check('fiber','supporting',.3))['rule_score'],extra['rule_score'])
        self.assertEqual(scored(sup)['rule_score'],0)
        uncal = check('uncal','supporting',.5,'UNCAL')
        self.assertEqual(scored(req,uncal)['rule_score'],base['rule_score'])

    def test_required_gate_rejects_quantitative_gaps_and_partial_confirmation(self):
        req = check('strong','required',1,'IN_RANGE')
        for state in ('NOT_DETECTED','UNCAL','NO_REF','OUT_OF_RANGE'):
            missing = check('weak','required',.15 if state in ('UNCAL','NO_REF') else 0,
                            state,bounds={'min':1,'max':3},weight=.3)
            support = check('aux','supporting',1)
            with self.subTest(state=state):
                self.assertEqual(scored(missing,support)['rule_score'],0)
                row=scored(req,missing,support)
                self.assertEqual(row['score_components']['supporting_bonus'],0)
                self.assertFalse(row['score_components']['required_complete'])
                self.assertIsNone(rank_and_decide([row],1)['conclusion'])

    def test_missing_evidence_penalty_shadow_keeps_required_gap(self):
        kb = KnowledgeBase(Path(paths.JUDGE_DIR) / 'data/herb_criteria.json')
        fiber = EvidenceCard('fiber')
        fiber.instance_refs = [('view', 1)]
        fiber.confs = [.9]
        fiber.dims = {'mean_width': {'values': [11.5], 'uncal_only': False}}
        fiber.shape = {'aspect_ratio': 6}
        cards = {'fiber': fiber}
        candidate = next(c for c in kb.candidates() if c['id'] == '大腹皮')
        actual = next(r for r in decide(kb, cards)['ranking'] if r['id'] == candidate['id'])
        raw = evaluate_candidate(candidate, cards, kb)
        shadow = score_candidate({**raw, 'absent_required': []})
        self.assertEqual(actual['absent_required'][0]['basis'], 'missing_admitted_evidence')
        self.assertEqual(actual['score_components']['missing_required_penalty'], .06)
        self.assertEqual(actual['rule_score'], .352)
        self.assertEqual(shadow['rule_score'], .4)
        self.assertIsNone(rank_and_decide([shadow], 1)['conclusion'])

    def test_disjoint_vessel_counter_reaches_b_and_fusion(self):
        kb = KnowledgeBase(Path(paths.JUDGE_DIR) / 'data/herb_criteria.json')
        cluster = EvidenceCard('cluster_crystal')
        cluster.instance_refs = [('view', 1)]
        cluster.confs = [1.0]
        cluster.dims = {'feret_max': {'values': [16]}}
        vessel = EvidenceCard('vessel')
        vessel.instance_refs = [('view', 2)]
        vessel.confs = [1.0]
        vessel.dims = {'mean_width': {'values': [20]}}
        cards = {'cluster_crystal': cluster, 'vessel': vessel}
        row = next(c for c in kb.candidates() if c['id'] == 'G1')
        before = score_candidate(evaluate_candidate(row, cards, kb))
        after = decide(kb, cards)
        self.assertEqual(after['score_version'], 'b-rule/4')
        apply_ab_fusion(after, {}, kb)
        g1 = next(r for r in after['ranking'] if r['id'] == 'G1')
        parts = g1['score_components']
        self.assertEqual(before['rule_score'], .408)
        self.assertEqual(g1['rule_score'], .341333)
        self.assertAlmostEqual(g1['fused_score'], .1706665, places=6)
        self.assertEqual(g1['raw_score'], before['raw_score'])
        self.assertEqual(parts['disjoint_counter_penalty'], .066667)
        counter, = parts['disjoint_counterevidence']
        self.assertEqual(counter['median'], 20)
        self.assertEqual(counter['instance_refs'], ['view#2'])
        self.assertEqual([peer['candidate_id'] for peer in counter['supporting_candidates']], ['G2'])

        # The gap between the two ranges has no supported disjoint peer.
        vessel.dims['mean_width']['values'] = [24.5]
        gap = next(r for r in decide(kb, cards)['ranking'] if r['id'] == 'G1')
        self.assertEqual(gap['rule_score'], before['rule_score'])
        self.assertEqual(gap['score_components']['disjoint_counterevidence'], [])

    def test_disjoint_counter_needs_the_same_measured_fact_and_counts_once(self):
        outside = check('vessel', 'supporting', 0, 'OUT_OF_RANGE',
                        bounds={'dim': 'mean_width', 'min': 25, 'max': 78})
        inside = check('vessel', 'supporting', .5, 'IN_RANGE',
                       bounds={'dim': 'mean_width', 'min': 10, 'max': 24})
        measured = {'stats': {'median': 20}, 'instance_refs': ['view#1']}
        loser = {'id': 'G1', 'checks': [{**outside, **measured}]}
        for replacement in (
            {'instance_refs': ['other#1']}, {'stats': None}, {'state': 'UNCAL'},
            {'quant_bounds': {'dim': 'feret_max', 'min': 10, 'max': 24}},
            {'quant_bounds': {'dim': 'mean_width', 'min': 10, 'max': 25}},
            {'quant_bounds': None, 'state': 'NO_REF'},
        ):
            with self.subTest(replacement=replacement):
                peer = {'id': 'G2', 'checks': [{**inside, **measured, **replacement}]}
                self.assertEqual(disjoint_counterevidence([loser, peer]), {})
        peers = [{'id': ident, 'checks': [{**inside, **measured}]} for ident in ('G2', '另一个候选')]
        counters, = disjoint_counterevidence([loser, *peers])['G1']
        self.assertEqual(len(counters['supporting_candidates']), 2)
        self.assertEqual(counters['strength'], .5)

    def test_caps_ignore_support_and_cannot_be_bypassed_by_scene(self):
        wide = check('wide','required',1,'IN_RANGE',{'min':10,'max':90},scene=1.1)
        row = scored(wide,check('support1','supporting',1),check('support2','supporting',1))
        self.assertTrue(row['cap_dom_wide'])
        self.assertEqual(row['score_components']['required_attainment'],.6)
        self.assertLess(row['rule_score'],.6)
        self.assertEqual(scored({**wide,'scene_factor':.6})['score_components']['required_attainment'],.36)
        self.assertEqual(scored(wide,non_official=True)['score_components']['required_attainment'],.49)

    def test_score_bands_margin_and_d8_are_separate(self):
        req = check('main','required',1)
        a=scored(req,check('aux','supporting',.1))
        b=scored(req,name='乙')
        decision=rank_and_decide([b,a])
        self.assertEqual(decision['ranking'][0]['id'],'甲')
        self.assertEqual(decision['verdict'],'probable')
        self.assertTrue(decision['needs_review'])
        self.assertLess(decision['margin'],.1)
        for score, gap, expected in ((.499999,.1,'insufficient_evidence'),(.5,.1,'probable'),
                                     (.749999,.1,'probable'),(.75,.099999,'probable'),(.75,.1,'confirmed')):
            top={**a,'rule_score':score,'confidence':score}
            rival={**b,'rule_score':round(score-gap,6)}
            self.assertEqual(rank_and_decide([top,rival])['verdict'],expected)
        self.assertTrue(rank_and_decide([b])['needs_review'])

    def test_fusion_uses_the_canonical_score_even_on_repeat(self):
        row=scored(check('needle','required',.6),check('skin','supporting',.255),name='龙胆')
        decision=rank_and_decide([row])
        for _ in range(2):
            apply_ab_fusion(decision,_a_dragon(),{})
            top=decision['ranking'][0]
            self.assertEqual(top['fused_score'],.747191)
            self.assertEqual(top['rule_score'],.504382)
            self.assertEqual(top['raw_score'],.855)

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
