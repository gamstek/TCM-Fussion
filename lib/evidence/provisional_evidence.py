"""把已产生的分类结果转成暂定候选排序，供超时场景保留中间证据。

本模块刻意位于正式规则引擎之外：只读当前有效分类快照，不改 formal score、support_count 或 verdict；一个 crop 出现多个非空 feature_id 时记为冲突，不能凭置信度静默合并。

主要定义：
- build: 返回 provisional-ranking/v1 排序与逐判据状态，正式裁决保持不变。
- VERIFIED / PENDING / PARTIAL / CONFLICT / NOT_DETECTED / UNKNOWN: 暂定状态枚举。
- TIER: 各暂定状态的权重（VERIFIED 1.0、PENDING 0.5、PARTIAL 0.25、其余 0）。
- _classifications / _best_classes / _tier_for: 内部工具，负责分类快照、逐 crop 择优与判据定级。

关联：
- 调用方 「lib/decision/final_assessment.py」、「lib/decision/output_projection.py」、「lib/workflow/workflow_runtime.py」、「tests/evidence/test_provisional_evidence.py」。
- 直接依赖 「lib/runtime/storage.py」。
"""
from __future__ import annotations

from pathlib import Path

from TCM_Fusion.lib.runtime.storage import read_json

from TCM_Fusion.lib.evidence.evidence_projection import supports_criterion

QUALITATIVE = 'QUALITATIVE_ADMITTED_NO_RANGE'
VERIFIED = 'VERIFIED_MATCH'
ADMITTED = {QUALITATIVE, VERIFIED}
PENDING = 'DETECTED_PENDING'
PARTIAL = 'PARTIAL_MATCH'
CONFLICT = 'CONFLICT'
NOT_DETECTED = 'NOT_DETECTED'
UNKNOWN = 'UNKNOWN'

TIER = {VERIFIED: 1.0, QUALITATIVE: 0.5, PENDING: 0.5, PARTIAL: 0.25, CONFLICT: 0.0,
        NOT_DETECTED: 0.0, UNKNOWN: 0.0}


def _classifications(result):
    valid = (result or {}).get('valid_classifications')
    if valid is not None:
        # Planning/reporting path: only the current valid evidence snapshot.
        return [{'crop_id': row.get('crop_id'), 'feature_id': str(row.get('feature_id') or ''),
                 'label': row.get('label'), 'reject': row.get('reject'),
                 'confidence': str(row.get('confidence') or '').lower(),
                 'confidence_source': row.get('confidence_source'),
                 'reason': str(row.get('reason') or ''),
                 'confidence_score': row.get('confidence_score'),
                 'identity_verified': bool(row.get('identity_verified'))}
                for row in valid]
    rows = []
    for op in ((result or {}).get('operations') or {}).values():
        if op.get('tool') != 'classify_target' or op.get('status') != 'done':
            continue
        root = Path((op.get('bundle') or {}).get('root') or '')
        path = root / 'classification' / 'classification.json'
        if not path.is_file():
            continue
        try:
            doc = read_json(path)
        except (OSError, ValueError):
            continue
        rows.append({'crop_id': doc.get('crop_id'), 'feature_id': str(doc.get('feature_id') or ''),
                     'label': doc.get('label'), 'reject': doc.get('reject'),
                     'confidence': str(doc.get('confidence') or '').lower(),
                     'confidence_source': doc.get('confidence_source'),
                     'reason': str(doc.get('reason') or ''),
                     'confidence_score': doc.get('confidence_score')})
    return rows


def _best_classes(rows):
    """One best classification per crop plus conflict crops.

    A crop with two different non-empty feature_ids is not silently resolved by
    confidence; it is reported as a conflict and cannot supply a full tier.
    """
    grouped = {}
    for row in rows:
        cid = str(row.get('crop_id') or '')
        if cid:
            grouped.setdefault(cid, []).append(row)
    rank = {'high': 3, 'medium': 2, 'low': 1, '': 0}
    best, conflicts = {}, set()
    for cid, group in grouped.items():
        features = sorted({str(x.get('feature_id') or '') for x in group
                           if not x.get('reject') and str(x.get('feature_id') or '')})
        if len(features) > 1:
            conflicts.add(cid)
        best[cid] = max(group, key=lambda x: rank.get(x.get('confidence'), 0))
    return list(best.values()), conflicts


def _tier_for(check, classes, conflicts):
    fid = check.get('feature_id')
    if check.get('present'):
        state = check.get('state')
        if state in {'IN_RANGE', 'EXTREME'}:
            return VERIFIED, []
        if state == 'NO_REF' and supports_criterion(check):
            return QUALITATIVE, list(check.get('instance_refs') or [])
        if state == 'OUT_OF_RANGE':
            return CONFLICT, []
        # 未采信的 NO_REF、UNCAL 等只保留检出或形态线索，
        # 不能凭 formal check 直接升级成 VERIFIED。
        if state != 'NOT_DETECTED':
            return PENDING, []
    hits = [c for c in classes if c.get('feature_id') == fid and not c.get('reject')]
    if not hits:
        return NOT_DETECTED, []
    ids = [c['crop_id'] for c in hits]
    if any(cid in conflicts for cid in ids):
        return CONFLICT, ids
    # 0.5 要求：分类可信，且有可见判别理由（不是仅凭 confidence 档位）。
    if any(c.get('confidence') in {'high', 'medium'}
           and (c.get('reason') or c.get('identity_verified')) for c in hits):
        return PENDING, ids
    # 0.25 要求：出现部分/疑似/边缘等形态线索；低置信本身不够。
    markers = ('部分', '疑似', '边缘', '可见', '可能')
    if any(any(marker in c.get('reason', '') for marker in markers) for c in hits):
        return PARTIAL, ids
    return UNKNOWN, ids


def build(verdict, result):
    """Return provisional ranking + criterion states; never mutate formal verdict."""
    ranking = list(verdict.get('ranking') or [])
    classes, conflicts = _best_classes(_classifications(result))
    rows = []
    for candidate in ranking:
        states = []
        formal_support = int(candidate.get('support_count') or 0)
        bonus_num = 0.0
        bonus_den = 0.0
        for check in candidate.get('checks') or []:
            state, instance_ids = _tier_for(check, classes, conflicts)
            weight = float(check.get('weight') or 0.0)
            bonus_num += weight * TIER[state]
            bonus_den += weight
            states.append({'feature_id': check.get('feature_id'),
                           'role': check.get('role'), 'tier': check.get('tier'),
                           'state': state, 'weight': weight,
                           'provisional_contribution': weight * TIER[state],
                           'formal_contribution': check.get('score_contribution'),
                           'instance_ids': sorted(set(instance_ids)),
                           'formal_state': check.get('state'),
                           'quant_note': check.get('quant_note'),
                           'n_measured': (check.get('stats') or {}).get('n_measured'),
                           'pending': state in {PENDING, PARTIAL}})
        bonus = bonus_num / bonus_den if bonus_den else 0.0
        a_prob = float(candidate.get('a_probability') or 0.0)
        provisional = round(0.6 * a_prob + 0.4 * bonus, 6)
        rows.append({'candidate_id': candidate.get('id'),
                     'display': candidate.get('display'),
                     'provisional_score': provisional,
                     'provisional_delta': round(provisional - float(candidate.get('score') or 0.0), 6),
                     'formal_score': candidate.get('score'),
                     'formal_support_count': formal_support,
                     'criterion_states': states,
                     'pending_verification': [s for s in states if s['pending']]})
    rows.sort(key=lambda r: (r['provisional_score'], r['formal_score'] or 0), reverse=True)
    return {'schema': 'provisional-ranking/v1',
            'policy': 'temporary only; formal verdict and formal support count unchanged',
            'source': 'classification artifacts, one best class per crop',
            'ranking': rows,
            'top_candidate': rows[0]['candidate_id'] if rows else None,
            'formal_top_candidate': ranking[0].get('id') if ranking else None}
