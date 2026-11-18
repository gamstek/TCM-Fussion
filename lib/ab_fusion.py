# -*- coding: utf-8 -*-
"""A/B 候选融合：把 A 的逐折概率与 B 的规则置信度按权重合并后重排候选。

A 是候选先验（Plan A），B 是显微证据（Plan B）；这里只重排候选并记录分项，不改 B verdict、不把 A 概率当显微证据。

主要定义：
- a_probabilities: 按全部折/视野求 A 的平均概率，未进 Top-K 的候选记 0。
- apply_ab_fusion: 计算 fused_score 并重排 ranking；若 top 改变则清空结论并要求复核。
- _group_map / _candidate_id: 内部工具，把别名/药材标签映射到候选组 id。

关联：
- 独立计算模块：不 import 项目内其他模块。
- .workflow_entry: 在候选范围门之后调用 apply_ab_fusion。
- .output_projection: 输出 fusion_weights 与 alternatives。
- tests/test_ab_fusion.py: 回归权重归一与 top 变化处理。
"""
from __future__ import annotations


def _group_map(kb):
    if hasattr(kb, 'group_of'):
        return kb.group_of
    mapping = {}
    groups = kb.get('groups') if isinstance(kb, dict) else {}
    if isinstance(groups, dict):
        for gid, group in groups.items():
            for herb in (group or {}).get('members') or []:
                mapping[herb] = gid
    for row in (kb.get('herbs') if isinstance(kb, dict) else []) or []:
        herb = row.get('herb')
        if herb:
            mapping.setdefault(herb, row.get('group_id') or herb)
    return mapping


def _candidate_id(label, kb, aliases):
    label = str(label or '')
    herb = (aliases or {}).get(label, label)
    return _group_map(kb).get(herb, herb)


def a_probabilities(a, kb, aliases=None):
    """按全部折/视野求平均概率；候选未出现在 Top-K 时记 0。"""
    sums, count = {}, 0
    if a.get('view_predictions'):
        folds = [fold for view in a['view_predictions']
                 for fold in (view.get('fold_rankings') or [])]
    else:
        folds = a.get('fold_rankings') or ([] if not a.get('ranking') else [a['ranking']])
    for rows in folds:
        count += 1
        for row in rows or []:
            identifier = _candidate_id(row.get('label'), kb, aliases)
            try:
                prob = max(0.0, min(1.0, float(row.get('probability') or 0)))
            except (TypeError, ValueError):
                prob = 0.0
            sums[identifier] = sums.get(identifier, 0.0) + prob
    if count == 0:
        return {}, 0
    return {key: value / count for key, value in sums.items()}, count


def apply_ab_fusion(verdict, a, kb, aliases=None, a_weight=0.5, b_weight=0.5):
    aw = max(0.0, float(a_weight))
    bw = max(0.0, float(b_weight))
    total = aw + bw
    if total <= 0:
        aw = bw = 0.5
        total = 1.0
    aw, bw = aw / total, bw / total
    probs, fold_count = a_probabilities(a, kb, aliases)
    ranking = list(verdict.get('ranking') or [])
    original_top = ranking[0].get('id') if ranking else None
    for row in ranking:
        identifier = row.get('id')
        a_prob = probs.get(identifier, 0.0)
        try:
            b_conf = max(0.0, min(1.0, float(row.get('confidence') or 0)))
        except (TypeError, ValueError):
            b_conf = 0.0
        fused = aw * a_prob + bw * b_conf
        row['a_probability'] = round(a_prob, 6)
        row['b_confidence'] = round(b_conf, 6)
        row['fused_score'] = round(fused, 6)
        row['b_score'] = row.get('score')
        row['score'] = round(fused, 6)
        # Keep rule confidence distinct from this uncalibrated ranking score.
    ranking.sort(key=lambda row: ((row.get('fused_score') or 0), (row.get('a_probability') or 0),
                                  (row.get('support_count') or 0)), reverse=True)
    verdict['ranking'] = ranking
    verdict['alternatives'] = ranking[1:5]
    if ranking and ranking[0].get('id') != original_top:
        verdict['conclusion'] = None
        verdict['verdict'] = 'insufficient_evidence'
        verdict['needs_review'] = True
    verdict['fusion_weights'] = {'a': round(aw, 4), 'b': round(bw, 4), 'a_folds': fold_count}
    verdict['fusion_policy'] = 'A 先验 + B 证据加权；B verdict/证据状态不由 A 改写'
    return verdict
