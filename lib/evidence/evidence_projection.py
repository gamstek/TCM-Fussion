"""D2：循环内与最终研判共用的只读证据资格投影。

消费原 score/decide 的检查及当前工件绑定，不重新评分，不授予 QA/量纲资格。
WorkflowRuntime 给规划器紧凑视图，final_assessment 复用同一事实与引用规则。
完整视图通过 read_evidence_detail 读取；事实 ID 不依赖列表位置或全局版本号。

主要定义：
- fact。内容及实际来源变动才改变 ID，无关操作不会使既有引用改号。
- reference_allowed。共享引用资格；局部未见只进入观察，反证必须来自正式判据。
- considered_candidates。
- build。返回事实、候选资格和缺口；只读取当前证据，不新增模型调用。
- compact。有界默认视图；完整候选/事实由既有只读工具按需取回。

关联：
- 调用方 「lib/decision/candidate_audit.py」、「lib/decision/final_assessment.py」、「lib/workflow/workflow_runtime.py」、「tests/integration/test_decision_view.py」。
- 直接依赖 「lib/runtime/storage.py」。
"""
import copy

from TCM_Fusion.lib.runtime.storage import digest, finite_number, read_json


def fact(kind, data, supports=(), counters=(), *, binding=None):
    """内容及实际来源变动才改变 ID，无关操作不会使既有引用改号。"""
    return {'id': kind + ':' + digest({'data': data, 'binding': binding})[:24],
            'kind': kind, 'data': copy.deepcopy(data),
            'supports_candidate_ids': sorted(set(supports)),
            'counters_candidate_ids': sorted(set(counters))}


def reference_allowed(row, candidate_id, relation='support'):
    """存在性阳性可以引用为线索；只有正式判据可引用为反证。"""
    if candidate_id is None:
        return False
    if relation == 'support' and row.get('kind') in {'criterion', 'observation'}:
        return candidate_id in row.get('supports_candidate_ids', [])
    return (relation == 'counter' and row.get('kind') == 'criterion'
            and candidate_id in row.get('counters_candidate_ids', []))


def bound_localization(observation):
    """在摘要化之前绑定实际登记动作和原始定位内容。缺少可靠绑定时独立展示。"""
    raw = observation.get('presence_localization')
    if not isinstance(raw, dict):
        return raw
    row = {k:v for k,v in raw.items() if k not in {'source_action_id','source_localization_sha256'}}
    action = observation.get('action') or {}
    args = action.get('args') or {}
    source = row.get('source_image_sha256')
    if (isinstance(source, str) and source and args.get('image_id') == 'img_' + source
            and args.get('feature_id') == row.get('feature_id') and row.get('feature_id')
            and isinstance(action.get('action_id'), str) and action['action_id']):
        row['source_localization_sha256'] = digest(raw)
        row['source_action_id'] = action['action_id']
    return row


def recent_localizations(observations):
    """目录保留最近八份有定位的原始观察，沿用实际提交顺序。"""
    recent = [obs for obs in observations if 'presence_localization' in obs][-8:]
    return [bound_localization(obs) for obs in recent]


def presence_state(observation, kb, images=None):
    """分类局部观察，缺结果、失败和低置信均为未知。"""
    action = observation.get('action') or {}
    result = observation.get('result') or {}
    args = action.get('args') or {}
    fid = args.get('feature_id')
    if (observation.get('accepted') and action.get('tool') == 'presence_check'
            and result.get('ok') is True and type(result.get('present')) is bool
            and (images is None or args.get('image_id') in images)
            and kb.tier(fid) in {'P1', 'C'}
            and finite_number(result.get('conf'), minimum=.5, maximum=1)):
        return 'present' if result['present'] else 'absent'
    return 'unknown'


MORPHOLOGY_UNVERIFIED = 'MORPHOLOGY_UNVERIFIED'
COUNTER_GUIDANCE = (
    '按来源和用途解释反证。合格实测经知识库判为OUT_OF_RANGE时，required该项当前不满足；'
    'supporting轻微偏离时说明测量变异，明显偏离时说明它削弱哪个候选的哪项佐证。'
    '说明差距和不确定性，未查或不可判保持未知。不可自行设扣分、否定整味药材或改写B分数。')


def supports_criterion(check):
    """判据级支持，兼容原 B 对无量纲要求的定性判据。"""
    state = check.get('state')
    return bool(check.get('present') and (check.get('score_contribution') or 0) > 0
                and (state in {'IN_RANGE', 'EXTREME'} or
                     state == 'NO_REF' and not check.get('quant_bounds')))


def counters_criterion(check):
    """合格物理量纲超出参考范围时才生成判据反证。"""
    return bool(check.get('present') and check.get('quant_bounds') and check.get('state') == 'OUT_OF_RANGE')


def considered_candidates(runtime, decision, verdict=None):
    interest = runtime.candidate_interest()
    ids = set(interest.get('a_topk_ids', interest.get('ids', []))) | set(interest.get('ids', []))
    ids.update(h['candidate_id'] for h in interest.get('hypotheses', []))
    ids.update(r['id'] for r in (verdict or {}).get('ranking', []))
    ids.update(r['id'] for r in decision.get('ranking', [])
               if r.get('support_count', 0) > 0 and r.get('confidence', 0) > 0)
    return ids


def _bindings(runtime, snapshot, observations):
    """工件依赖和观察身份仅用于引用绑定，不把诊断工件转成规则支持。"""
    import contracts
    rows = {}
    arts = snapshot['artifacts']
    feature_by_crop = {}
    for aid, item in arts.items():
        if item.get('valid') and item.get('kind') == 'crop':
            feature_by_crop[aid] = contracts.CropMeta.from_json(item['path']).feature_id
    def features(aid, seen=None):
        seen = set() if seen is None else seen
        if aid in seen:
            return set()
        seen.add(aid); item = arts.get(aid, {})
        own = item.get('feature_id') or feature_by_crop.get(aid) or feature_by_crop.get(item.get('crop_id'))
        found = {own} if own else set()
        for parent in item.get('parents', []):
            found.update(features(parent, seen))
        return found
    for aid, item in arts.items():
        if not item.get('valid'):
            continue
        for fid in features(aid):
            rows.setdefault(fid, []).append((aid, item['sha256']))
    for obs in observations:
        args = (obs.get('action') or {}).get('args') or {}
        fid = args.get('feature_id')
        if fid:
            image = runtime.images.get(args.get('image_id'), {})
            rows.setdefault(fid, []).append(('observation', digest({
                'source': image.get('sha256'), 'action': obs.get('action'),
                'accepted': obs.get('accepted'), 'result': obs.get('result')})))
    return {fid: sorted(values) for fid, values in rows.items()}


def build(runtime, scored, observations, *, verdict=None):
    """返回事实、候选资格和缺口；只读取当前证据，不新增模型调用。"""
    snap = runtime.workflow.store.current()
    decision = scored['decision']; considered = considered_candidates(runtime, decision, verdict)
    bindings = _bindings(runtime, snap, observations)
    facts = []; projections = []; catalog = []
    checks_by_candidate = {}
    for row in decision.get('ranking', []):
        cid = row['id']; checks_by_candidate[cid] = row.get('checks', [])
        catalog.append({'id': cid, 'display': row.get('display', cid),
                        'rule_score': row.get('rule_score', row.get('confidence')),
                        'rule_confidence': row.get('rule_score', row.get('confidence')),
                        'support_count': row.get('support_count'), 'check_refs': []})
        if cid not in considered:
            continue
        support = []; counter = []; diagnostic = []; invalid = []; gaps = []
        for check in row.get('checks', []):
            state = check.get('state'); quantitative = bool(check.get('quant_bounds'))
            countered = counters_criterion(check)
            positive = supports_criterion(check)
            data = {'candidate_id': cid, **copy.deepcopy(check),
                    'claim_scope': ('quantitative_criterion' if quantitative else
                        check.get('observation_scope', 'qualitative_rule')),
                    'dimension_eligible': bool(quantitative and state in {'IN_RANGE', 'EXTREME', 'OUT_OF_RANGE'})}
            entry = fact('criterion', data, [cid] if positive else [],
                         [cid] if countered else [],
                         binding=bindings.get(check.get('feature_id'), []))
            facts.append(entry); catalog[-1]['check_refs'].append(entry['id'])
            if positive: support.append(entry['id'])
            elif countered: counter.append(entry['id'])
            else: diagnostic.append(entry['id'])
            if quantitative and state not in {'IN_RANGE', 'EXTREME', 'OUT_OF_RANGE'}:
                invalid.append({'feature_id': check['feature_id'], 'dimension': check['quant_bounds'].get('dim'),
                                'reason': state, 'ref': entry['id']})
            if check.get('role') == 'required' and not positive:
                gaps.append({'feature_id': check['feature_id'], 'state': state, 'ref': entry['id'],
                             'observation_state': 'unknown' if not check.get('present') else 'present'})
        conclusion = decision.get('conclusion') or {}
        projections.append({'candidate_id': cid, 'formal_support': support, 'formal_counter': counter,
                            'diagnostic_only': diagnostic, 'invalid_dimensions': invalid, 'gaps': gaps,
                            'allowed_conclusion': decision.get('verdict', 'insufficient_evidence')
                                if conclusion.get('id') == cid else 'insufficient_evidence',
                            'needs_review': bool(decision.get('needs_review')),
                            'evidence_version': snap['version']})
    for obs in observations:
        action = obs.get('action') or {}; result = obs.get('result') or {}
        args = action.get('args') or {}; fid = args.get('feature_id')
        related = [cid for cid, checks in checks_by_candidate.items()
                   if any(c.get('feature_id') == fid for c in checks)]
        observed_state = presence_state(obs, runtime.kb, runtime.images)
        entry = fact('observation', {'accepted': bool(obs.get('accepted')),
            'action': {k: action.get(k) for k in ('tool', 'args', 'action_id')}, 'result': result,
            'observation_state': observed_state, 'dimension_eligible': False,
            'claim_scope': 'feature_presence' if action.get('tool') == 'presence_check' else 'scene_context',
            'related_candidate_ids': sorted(related),
            'scope': 'bound field of view only; failure is unknown, absence does not exclude a sample'},
            related if observed_state == 'present' else [],
            binding=runtime.images.get(args.get('image_id'), {}).get('sha256'))
        facts.append(entry)
        for row in projections:
            if row['candidate_id'] in related:
                # 观察可支持局部存在性解释，不替代定量 required；其缺口保留。
                row['diagnostic_only'].append(entry['id'])
                for gap in row['gaps']:
                    if gap['feature_id'] == fid:
                        prior = gap.get('observation_state', 'unknown')
                        gap['observation_state'] = ('present' if 'present' in {prior, observed_state} else
                            'absent_in_observed_fovs' if observed_state == 'absent' or prior == 'absent_in_observed_fovs'
                            else 'unknown')
    for item in snap['artifacts'].values():
        if not item.get('valid') or item.get('kind') != 'classification':
            continue
        doc = read_json(item['path']); fid = item.get('feature_id') or doc.get('feature_id')
        entry = fact('classification', {'artifact_id': item['id'],
                     'identity_verified': item.get('identity_verified'), 'result': doc}, binding=item['sha256'])
        facts.append(entry)
        for row in projections:
            if any(c.get('feature_id') == fid for c in checks_by_candidate[row['candidate_id']]):
                row['diagnostic_only'].append(entry['id'])
    for item in snap['artifacts'].values():
        if not item.get('valid') or item.get('kind') != 'segmentation' or not item.get('assessment_id'):
            continue
        assessment = runtime.workflow.assessment(item)
        if not assessment:
            continue
        entry = fact('quality_assessment', assessment,
                     binding=snap['artifacts'][item['assessment_id']]['sha256'])
        facts.append(entry)
        crop = snap['artifacts'].get(item.get('crop_id'), {})
        if crop:
            import contracts
            fid = contracts.CropMeta.from_json(crop['path']).feature_id
            for row in projections:
                if any(c.get('feature_id') == fid for c in checks_by_candidate[row['candidate_id']]):
                    row['diagnostic_only'].append(entry['id'])
                    for use in ('length','width','area'):
                        if not assessment['uses'][use]['allowed']:
                            row['invalid_dimensions'].append({'feature_id':fid,'dimension':use,
                                'reason':'quality_use_denied','ref':entry['id']})
    # 同一事实不因重复观测列表再次出现而计作独立支持。
    facts = list({f['id']: f for f in facts}.values())
    for candidate in catalog:
        cid = candidate['id']
        candidate['eligible_support_refs'] = [f['id'] for f in facts if reference_allowed(f, cid)]
        candidate['eligible_counter_refs'] = [f['id'] for f in facts if reference_allowed(f, cid, 'counter')]
    return {'schema': 'evidence-eligibility/d2', 'evidence_version': snap['version'],
            'observation_revision': digest(observations), 'considered': sorted(considered),
            'candidates': catalog, 'projections': projections, 'facts': facts,
            'a_prior_is_formal_evidence': False}


def compact(view, limit=8):
    """有界默认视图；完整候选/事实由既有只读工具按需取回。"""
    rows = []
    for row in view['projections'][:limit]:
        rows.append({**row, **{key: row[key][:3] for key in (
            'formal_support', 'formal_counter', 'diagnostic_only', 'invalid_dimensions', 'gaps')},
            'counts': {k: len(row[k]) for k in ('formal_support', 'formal_counter', 'diagnostic_only', 'gaps')}})
    return {'evidence_version': view['evidence_version'], 'candidates': rows,
            'candidate_ids': view['considered'], 'total': len(view['projections']),
            'omitted': max(0, len(view['projections'])-limit), 'a_prior_is_formal_evidence': False,
            'detail_access': 'read_evidence_detail(kind=eligibility)'}
