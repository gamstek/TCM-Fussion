"""用当前 B 证据池解释下一次实例测量可能解决的判据问题，仅影响调查菜单。

measurement_menu 汇合本轮合法工具、已采信量纲与各候选的原规则检查。
dimension_gain 用同一知识库的区间解释同源实例分歧，输出可审计的复测理由。
调用方为 react_graph、planning_choices。规则计分、工件发布与质量授权仍由原组件负责。
"""
from collections import Counter, defaultdict
from statistics import median

from rule_engine import check_quant_state
from TCM_Fusion.lib.decision.comparison_problems import _action_feature
from TCM_Fusion.lib.evidence.purpose_evidence import dimension_use
from TCM_Fusion.lib.evidence.evidence_projection import supports_criterion


def dimension_gain(values, checks, seen_views, target_views, active):
    """比较已采信数值。并列的实例意见可由新实例打破，同源多数后需新条件。"""
    consumers = []
    for candidate, check in checks:
        bounds = check['quant_bounds']
        reasons = []
        states = [check_quant_state(bounds, {'median': v, 'max': v})[0] for v in values]
        counts = dict(sorted(Counter(states).items()))
        sides = Counter('below_minimum' if status == 'OUT_OF_RANGE' and bounds.get('min') is not None
                        and value < bounds['min'] else 'above_maximum' if status == 'OUT_OF_RANGE' else status
                        for value, status in zip(values, states))
        if not values and candidate in active:
            reasons.append('first_qualified_measurement')
        elif len(values) == 1 and candidate in active:
            reasons.append('within_sample_variation_unknown')
        elif len(sides) > 1:
            largest = max(sides.values())
            if list(sides.values()).count(largest) > 1:
                reasons.append('resolve_split_instances')
            elif target_views and seen_views and target_views.isdisjoint(seen_views):
                reasons.append('test_observed_disagreement_in_new_view')
        if candidate in active or reasons or len(sides) > 1:
            consumers.append({'candidate_id': candidate, 'role': check.get('role'),
                'bounds': bounds, 'state': check.get('state'), 'observed_states': counts,
                'reasons': reasons, **({'sides': dict(sides)} if len(sides) > len(counts) else {})})
    return {'n_measured': len(values), 'median': median(values) if values else None,
            'measured_views': sorted(seen_views), 'consumers': consumers,
            'worth_measuring': any(row['reasons'] for row in consumers)}


def measurement_menu(runtime, actions, scored, targets):
    """保留独立身份核验。对同特征的量纲目标和直接付费入口使用同一收益表。"""
    by_feature = defaultdict(lambda: defaultdict(list))
    qualitative = defaultdict(list)
    for row in scored['decision'].get('ranking', []):
        for check in row.get('checks', []):
            bounds = check.get('quant_bounds') or {}
            if bounds.get('dim') and any(bounds.get(k) is not None for k in ('min', 'max')):
                by_feature[check['feature_id']][bounds['dim']].append((row['id'], check))
            elif not bounds:
                qualitative[check['feature_id']].append((row['id'], check))
    active = set(runtime.candidate_interest()['ids'])
    kept, groups, by_action = [], {}, {}
    for action in actions:
        if action['tool'] not in {'classify_target', 'segment_instance', 'measure_instances'}:
            kept.append(action)
            continue
        target = targets.get((action.get('args') or {}).get('target_id'), {})
        crop = targets.get(target.get('crop_id'), target)
        fid = _action_feature(action, targets)
        image = crop.get('image_id')
        card = scored['cards'].get(fid)
        applicable = [check for cid, check in qualitative.get(fid, []) if cid in active]
        qualitative_complete = bool(not by_feature.get(fid) and card and card.n_instances and
                                    applicable and all(supports_criterion(check) for check in applicable))
        if not image or (not by_feature.get(fid) and not qualitative_complete):
            kept.append(action)
            continue
        key = fid + '@' + image
        if key not in groups:
            target_views = set(runtime.images.get(image, {}).get('stems', []))
            dimensions = {}
            for dim, checks in by_feature[fid].items():
                values = (card.dims.get(dim, {}) if card else {}).get('values', [])
                # instance_refs对应所有主判量纲。混合量纲时不可猜某个值来自哪个视野。
                views = {stem for stem, _ in card.instance_refs} if card and len(card.dims) == 1 else set()
                dimensions[dim] = dimension_gain(values, checks, views, target_views, active)
            groups[key] = {'feature_id': fid, 'image_id': image, 'dimensions': dimensions}
            if qualitative_complete:
                groups[key]['qualitative_only'] = {'n_measured': card.n_instances,
                    'reason': 'admitted_qualitative_criteria_without_reference'}
        dimensions = groups[key]['dimensions']
        by_action[action['action_id']] = key
        existing_seg = targets.get('seg:' + crop['id'], {}) if crop.get('id') else {}
        pending_repair = (existing_seg.get('quality') == 'review' and
                          action['tool'] in {'measure_instances', 'segment_instance'})
        if (action['tool'] == 'classify_target' or pending_repair
                or not qualitative_complete and not any(d['n_measured'] for d in dimensions.values())
                or any(d['worth_measuring'] for d in dimensions.values())):
            kept.append(action)
    return kept, groups, by_action


def goal_consumers(group, task):
    """None 沿用首测或身份契约，空列表表示复测缺少用途，非空列表指明受益候选。"""
    if not group or task == 'verify_identity':
        return None
    if group.get('qualitative_only'):
        return []
    rows = [row for dim, row in group['dimensions'].items() if task == 'measure_' + dimension_use(dim)]
    reasons = sorted({c['candidate_id'] for row in rows for c in row['consumers'] if c['reasons']})
    if reasons:
        return reasons
    # The repeat filter starts only after B has accepted a measurement.
    if not any(row['n_measured'] for row in group['dimensions'].values()):
        return None
    return []
