# -*- coding: utf-8 -*-
"""消费 A 候选、原裁决和融合配置，生成允许交付的候选范围。保留 A 候选占位，记录处理过的 B 候选以及严格非 A 准入依据。该层不创造显微证据。

主要定义：
- apply_candidate_scope: 就地收窄 verdict ranking，记录 kept/dropped，并为缺失的 A 候选补零支持占位。
- GENERIC_FEATURES: 不具判别力的通用特征（表皮细胞、分泌组织）。
- _strict_non_a: 内部工具，要求所有 required/supporting 为 present 且至少一个特异性 required。

关联：
- 调用方 「lib/application/workflow_entry.py」、「tests/decision/test_candidate_scope.py」。
- 直接依赖 「lib/decision/candidate_policy.py」。
"""
from __future__ import annotations

from TCM_Fusion.lib.decision.candidate_policy import candidates as a_candidates
from TCM_Fusion.lib.evidence.evidence_projection import supports_criterion

GENERIC_FEATURES = {'epidermis_cell', 'secretory_tissue'}


def _strict_non_a(row):
    """非 A 候选：所有 required/supporting 必须为 present，且至少一个特异性 required。"""
    checks = row.get('checks') or []
    required = [c for c in checks if c.get('role') == 'required']
    supporting = [c for c in checks if c.get('role') == 'supporting']
    if not required:
        return False
    if not all(supports_criterion(c) for c in required + supporting):
        return False
    return any(c.get('feature_id') not in GENERIC_FEATURES for c in required)


def apply_candidate_scope(verdict, a, kb, aliases=None, policy=None, scope='a_topk'):
    """按 A Top-K 收窄 verdict ranking；返回同一个 verdict（就地更新）。"""
    if scope not in {'a_topk', 'open_strict'}:
        scope = 'a_topk'
    try:
        a_ids = list(a_candidates(a, kb, aliases, policy).get('ids') or [])
    except Exception:
        a_ids = []
    a_set = set(a_ids)
    ranking = list(verdict.get('ranking') or [])
    original_top = ranking[0].get('id') if ranking else None
    kept, dropped = [], []
    for row in ranking:
        identifier = row.get('id')
        if identifier in a_set:
            kept.append(row)
        elif scope == 'open_strict' and _strict_non_a(row):
            kept.append(row)
        else:
            dropped.append({'id': identifier,
                            'reason': 'not_in_a_topk' if identifier not in a_set else 'strict_non_a_failed'})
    # A Top-K 中若 B 排名缺项，补一个零支持占位，保证 A 候选仍是可见重点。
    present = {row.get('id') for row in kept}
    for identifier in a_ids:
        if identifier in present:
            continue
        kept.append({'id': identifier, 'display': identifier, 'kind': 'herb', 'score': 0.0,
                     'confidence': 0.0, 'rule_score': 0.0, 'raw_score': 0.0, 'support_count': 0, 'checks': [],
                     'scope_placeholder': True})
    verdict['ranking'] = kept
    verdict['alternatives'] = kept[1:5]
    verdict['candidate_scope'] = {
        'scope': scope,
        'a_topk_ids': a_ids,
        'kept_ids': [row.get('id') for row in kept],
        'dropped': dropped,
        'policy': 'A Top-K 为 Plan B 重点；非 A 仅在 open_strict 且 required+supporting 全部符合时保留',
    }
    # 范围改变后，原 top 的 conclusion 不再可信；结论由规则 verdict/ranking 重新呈现。
    conclusion = verdict.get('conclusion')
    conclusion_id = conclusion.get('id') if isinstance(conclusion, dict) else conclusion
    top = kept[0].get('id') if kept else None
    if top is None or top != original_top or (conclusion_id and conclusion_id != top):
        verdict['conclusion'] = None
        verdict['verdict'] = 'insufficient_evidence'
        verdict['needs_review'] = True
    return verdict
