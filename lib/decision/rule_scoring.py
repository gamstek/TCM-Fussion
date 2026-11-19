"""从 B 引擎已核验的逐条判据生成融合项目的正式 B 分与裁决。生产和离线复算
共享同一计算，原始贡献只供核对。证据、知识库和旧运行记录均只读。

主要定义：score_candidate 计算 required 达成度与独立佐证加分，rank_and_decide
执行分差、必需判据和证据广度门槛，decide 从当前证据卡构造候选。

关联：workflow.evidence_runtime 调用 decide，react_graph 消费相同裁决门槛；
逐条证据来自 rule_engine，采信资格与 evidence_projection 共用。
"""
from rule_engine import (
    BAND_CONFIRMED, BAND_PROBABLE, DOM_MAX_FACTOR, DOM_SHARE_THRESHOLD,
    MARGIN_DELTA, MIN_SUPPORT_FEATURES, NONOFF_MAX_CONF, WIDE_SPAN_THRESHOLD,
    evaluate_candidate, scene_fit,
)
from TCM_Fusion.lib.evidence.evidence_projection import supports_criterion


SCORE_VERSION = 'b-rule/2'
SUPPORT_WEIGHT = 0.20


def score_candidate(row, *, non_official=False, support_weight=SUPPORT_WEIGHT):
    """输入一行完整判据核验明细。离线权重对拍也从这些明细重算，不读旧分作为 R。"""
    if not 0 <= support_weight <= 1:
        raise ValueError('support_weight must be in [0, 1]')
    checks = row['checks']
    required = [c for c in checks if c['role'] == 'required']
    verified = [c for c in checks if supports_criterion(c)]
    matched = [c for c in verified if c['role'] == 'required']
    complete = bool(required) and len(matched) == len(required)
    required_ids = {c['feature_id'] for c in required}
    supporting = {}
    for c in verified:
        fid = c['feature_id']
        if c['role'] == 'supporting' and fid not in required_ids:
            supporting[fid] = max(supporting.get(fid, 0), c['score_contribution'])
    positive = sum(c['score_contribution'] for c in matched)
    penalties = (sum(c['penalty'] for c in required)
                 + sum(c['penalty'] for c in row['absent_required'])
                 + sum(c['penalty'] for c in row['wrong_present']))
    denominator = sum(c['weight'] for c in required)
    attainment = max(0, min(1, (positive - penalties) / denominator)) if denominator else 0
    wide = False
    for c in matched:
        bounds = c.get('quant_bounds') or {}
        if (bounds.get('max') is not None and bounds.get('min') is not None
                and bounds['max'] - bounds['min'] >= WIDE_SPAN_THRESHOLD
                and c['score_contribution'] >= DOM_SHARE_THRESHOLD * positive):
            wide = True
            break
    cap = min(DOM_MAX_FACTOR if wide else 1, NONOFF_MAX_CONF if non_official else 1)
    scene = (sum(c['score_contribution'] * (c.get('scene_factor') or 1) for c in matched)
             / positive if positive else 1)
    attainment = min(cap, min(attainment, cap) * scene)
    strength = sum(supporting.values())
    base = (1 - support_weight) * attainment
    bonus = support_weight * attainment * strength / (1 + strength) if complete else 0
    score = round(base + bonus, 6)
    return {
        'id': row['id'], 'display': row['display'], 'kind': row['kind'],
        'checks': [{k: v for k, v in c.items() if k != 'counts_toward_conf'} for c in checks],
        'wrong_present': row['wrong_present'], 'absent_required': row['absent_required'],
        'rule_score': score, 'score': score, 'confidence': score,
        'raw_score': row.get('raw_score', row.get('b_score', row['score'])),
        'support_count': len({c['feature_id'] for c in verified}),
        'cap_dom_wide': wide, 'cap_non_official': non_official,
        'score_components': {
            'version': SCORE_VERSION, 'support_weight': support_weight,
            'required_total': len(required), 'required_matched': len(matched),
            'required_complete': complete, 'required_attainment': round(attainment, 6),
            'required_base': round(base, 6), 'supporting_contributions': supporting,
            'supporting_strength': round(strength, 6), 'supporting_bonus': round(bonus, 6),
        },
    }


def rank_and_decide(rows, min_support_features=None):
    threshold = MIN_SUPPORT_FEATURES if min_support_features is None else min_support_features
    if type(threshold) is not int or threshold not in (1, 2):
        raise ValueError('min_support_features must be 1 or 2')
    ranking = sorted(rows, key=lambda r: (-r['rule_score'], -r.get('scene_fit', 0), r['id']))
    top = ranking[0] if ranking else None
    score = top['rule_score'] if top else 0
    margin = round(score - (ranking[1]['rule_score'] if len(ranking) > 1 else 0), 6)
    count = top['support_count'] if top else 0
    out = {'score_version': SCORE_VERSION, 'ranking': ranking, 'verdict': 'insufficient_evidence',
           'conclusion': None, 'margin': margin, 'needs_review': True, 'notes': [],
           'support_feature_count': count, 'alternatives': []}
    if not top:
        return out
    def summary(row):
        return {k: row[k] for k in ('id', 'kind', 'display', 'rule_score', 'confidence')}
    out['alternatives'] = [summary(r) for r in ranking[:4]]
    if not top['score_components']['required_complete'] or score < BAND_PROBABLE:
        out['notes'].append('必需判据未全部核验或 B 规则分未达倾向门槛。')
        return out
    decisive = margin >= MARGIN_DELTA
    strong = score >= BAND_CONFIRMED and decisive
    if count < threshold and not (count == 1 and strong):
        out['notes'].append('D8 独立证据类数不足。')
        return out
    out['conclusion'] = summary(top)
    out['alternatives'] = [summary(r) for r in ranking[1:4]]
    out['verdict'] = ('group_level' if top['kind'] == 'group' else
                      'confirmed' if strong else 'probable')
    out['needs_review'] = not decisive or count < threshold
    if not decisive:
        out['notes'].append(f'B 规则分差 {margin:g} 小于 {MARGIN_DELTA:g}，需要复核。')
    if count < threshold:
        out['notes'].append('D8b 单类证据条件放行，仍需复核。')
    return out


def decide(kb, cards, *, scene=None, min_support_features=None):
    rows = []
    for candidate in kb.candidates():
        evaluated = evaluate_candidate(candidate, cards, kb, scene)
        row = score_candidate(evaluated, non_official=bool(candidate.get('non_official_basis')))
        if scene:
            features = {c['feature_id'] for c in row['checks']
                        if c['role'] == 'required' and supports_criterion(c)}
            row['scene_fit'], row['scene_reasons'] = scene_fit(candidate, scene, features) if features else (0, [])
        rows.append(row)
    return rank_and_decide(rows, min_support_features)
