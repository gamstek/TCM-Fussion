"""逐项对照候选判据、当前用途资格、任务动作和观察账本。输出候选审查状态、可达入口及缺口，供 ReAct 和收尾共用。区分未查、失败、未映射和真实反证。

主要定义：
- build。
- compact。每候选仅发状态与缺口，不重复判据全文；完整账本落盘/收尾可查。
- investigation_opportunities。选择已获准的关键判据入口及预算需求，供有界规划菜单使用。
- protect_investigation_budget。保留一个可执行的关键入口，阻止弱探查耗尽所需余额。
- readiness_shadow。候选审计与判定充分性分离的影子评估；未经独立验证，不提升交付资格。

关联：
- 调用方 「lib/agent/react_graph.py」、「lib/decision/final_assessment.py」、「lib/workflow/workflow_runtime.py」、「tests/decision/test_candidate_audit.py」、「tests/decision/test_final_assessment.py」、「tests/integration/test_fair_investigation.py」。
- 直接依赖 「lib/decision/comparison_problems.py」、「lib/evidence/evidence_projection.py」、「lib/runtime/storage.py」。
"""
from TCM_Fusion.lib.decision.comparison_problems import _action_feature, evidence_paths
from TCM_Fusion.lib.runtime.storage import read_json


def build(runtime, scored, observations, *, eligibility=None, actions=None):
    if eligibility is None:
        from TCM_Fusion.lib.evidence.evidence_projection import build as project
        eligibility=project(runtime,scored,observations)
    interest=runtime.candidate_interest()
    requested=list(dict.fromkeys(interest.get('a_topk_ids',interest.get('ids',[]))+
        [h['candidate_id'] for h in interest.get('hypotheses',[])]))
    catalog={c['id']:c for c in runtime.kb.candidates()}
    checks={r['id']:{c['feature_id']:c for c in r.get('checks',[])}
            for r in scored['decision'].get('ranking',[])}
    status={h['candidate_id']:h.get('status') for h in interest.get('hypotheses',[])}
    by_criterion={(f['data']['candidate_id'],f['data']['feature_id']):f
                  for f in eligibility['facts'] if f['kind']=='criterion'}
    observed={};observed_views={}
    for f in eligibility['facts']:
        if f['kind']=='observation':
            fid=(f['data'].get('action',{}).get('args')or{}).get('feature_id')
            if fid:
                state=f['data']['observation_state']
                observed.setdefault(fid,[]).append(state)
                image=(f['data']['action'].get('args')or{}).get('image_id')
                observed_views.setdefault(fid,{}).setdefault(image,[]).append(state)
    provided=set(runtime.images)
    targets=runtime.workflow.targets()
    actions=actions if actions is not None else runtime.legal_actions({'observations':observations,'attempted':[]})
    paths=evidence_paths(runtime,actions,{c['feature'] for cid in requested if cid in catalog
                                         for c in catalog[cid]['criteria']})
    available={fid:[p['action_id'] for p in ps] for fid,ps in paths.items()}
    ops=list(read_json(runtime.workflow.operations)['entries'].values())
    attempted={}
    for op in ops:
        fid=op.get('feature_id') or _action_feature(
            {'tool':op.get('tool','unknown'),'args':{'target_id':op.get('target_id')}},targets)
        if fid:attempted.setdefault(fid,[]).append(op.get('status'))
    rows=[]
    for cid in requested:
        if cid not in catalog:continue
        required=[];supporting=[]
        for c in catalog[cid]['criteria']:
            fid=c['feature'];check=checks.get(cid,{}).get(fid,{})
            fact=by_criterion.get((cid,fid),{})
            states=observed.get(fid,[])
            supported=cid in fact.get('supports_candidate_ids',[])
            counter=cid in fact.get('counters_candidate_ids',[])
            # 当前视野阴性算已观察，但不代表全样本不存在或可跳过其他判据。
            full_local_absence=bool(provided) and all(
                observed_views.get(fid,{}).get(image) and
                all(s=='absent' for s in observed_views[fid][image]) for image in provided)
            partial_local_absence='absent'in states and 'present'not in states
            assessed=supported or counter or full_local_absence
            state=('supported' if supported else 'quantitative_counter' if counter else
                   'local_absent' if full_local_absence else 'local_absent_partial' if partial_local_absence else
                   'present_pending' if 'present'in states else
                   'failed_or_unknown' if 'failed'in attempted.get(fid,[]) or 'unknown'in states else
                   'investigated_unresolved' if attempted.get(fid) else 'unchecked')
            item={'feature_id':fid,'state':state,'rule_state':check.get('state','NOT_DETECTED'),
                'assessed':bool(assessed),'satisfies_criterion':supported,
                'observed_fov_count':len(observed_views.get(fid,{})), 'provided_fov_count':len(provided),
                'availability_reason':('reachable_via_prerequisite' if all(
                    p['kind']=='discovery_prerequisite' for p in paths[fid]) else 'available') if available.get(fid) else
                    'hypothesis_paused' if status.get(cid) in {'paused','withdrawn'} else
                    'prior_tool_failure' if 'failed'in attempted.get(fid,[]) else 'no_current_legal_action',
                'action_ids':available.get(fid,[]),'entry_paths':paths.get(fid,[]),'fact_ref':fact.get('id')}
            (required if c['role']=='required' else supporting).append(item)
        missing=[x['feature_id'] for x in required+supporting if not x['assessed']]
        rows.append({'candidate_id':cid,'display':catalog[cid].get('display',cid),
            'hypothesis_status':status.get(cid,'proposed'),'required':required,'supporting':supporting,
            'review_complete':not missing,'unassessed_features':missing,
            'paused_is_excluded':False})
    unmapped=[]
    for value in interest.get('unmapped',[]):
        label=value if isinstance(value,str) else value['label']
        unmapped.append({'label':label,'state':'kb_unmapped','review_complete':False,
                         'reason':'没有可用候选映射，未审查，不是阴性或已排除'})
    pending=[r['candidate_id']for r in rows if not r['review_complete']]
    actionable=any(x['action_ids'] for r in rows for x in r['required']+r['supporting'] if not x['assessed'])
    return {'schema':'candidate-audit/v1','candidates':rows,'unmapped':unmapped,
        'candidate_ids':requested,'complete':not pending and not unmapped,
        'incomplete_candidate_ids':pending,'actionable_gaps':actionable,
        'comparison_required':bool(requested or unmapped),
        'scope':'current provided fields of view; local absence never excludes a specimen',
        'policy':'review is always allowed; incomplete audit cannot authorize a complete differential conclusion'}


def compact(value):
    """每候选仅发状态与缺口，不重复判据全文；完整账本落盘/收尾可查。"""
    return {k:value[k] for k in ('schema','complete','candidate_ids','unmapped',
        'incomplete_candidate_ids','actionable_gaps','comparison_required','scope')} | {
        'candidates':[{'candidate_id':r['candidate_id'],'hypothesis_status':r['hypothesis_status'],
            'required':{x['feature_id']:x['state']for x in r['required']},
            'supporting':{x['feature_id']:x['state']for x in r['supporting']},
            'review_complete':r['review_complete']}for r in value['candidates']]}


def investigation_opportunities(runtime, audit, focus, actions, remaining):
    """Return at most two current alternatives per role, with one plausible path per feature."""
    interest = runtime.candidate_interest()
    active = set(interest['ids'])
    key_candidates = ({focus.get('current_rule_lead'), *focus.get('a_leads', []),
                       *focus.get('competing_support', {})} & active) or active
    if len(key_candidates) < 2:
        key_candidates |= set(interest.get('a_topk_ids', active)) & active
    profiles = {}
    for candidate in runtime.kb.candidates():
        if candidate['id'] in key_candidates:
            profiles[candidate['id']] = {c['feature']: c for c in candidate['criteria']}
    by_id = {a['action_id']: a for a in actions}
    targets = runtime.workflow.targets()
    opportunities = {'required': {}, 'supporting': {}}
    for row in audit['candidates']:
        cid = row['candidate_id']
        if cid not in profiles:
            continue
        for role, by_feature in opportunities.items():
            for criterion in row[role]:
                feature = criterion['feature_id']
                if criterion['assessed']:
                    continue
                profile = profiles[cid][feature]
                others = [p.get(feature) for id_, p in profiles.items() if id_ != cid]
                differs = any(other != profile for other in others)
                if role == 'supporting' and not differs:
                    continue
                quantitative = bool(profile.get('quant'))
                for path in criterion['entry_paths']:
                    aid, tool = path['action_id'], path['tool']
                    if aid not in by_id:
                        continue
                    if tool == 'presence_check' and quantitative:
                        continue  # a cheap observation cannot complete a dimension
                    target = targets.get(by_id[aid].get('args', {}).get('target_id'), {})
                    if tool == 'feature_scan':
                        # Budget for locating and checking one instance, not all returned crops.
                        if quantitative and not target.get('px_per_um'):
                            continue
                        tiles = target['max_tiles']
                        followup_plans = 1 + int(quantitative and not
                                                 runtime.workflow.config.get('semantic_execution'))
                        cost = {'recompute': 1 + 2 * quantitative, 'observation': 1,
                                'network': tiles + 1 + quantitative + followup_plans,
                                'planning': followup_plans}
                    elif tool == 'classify_target':
                        cost = {'observation': 1, 'recompute': 2 * quantitative,
                                'network': 1 + 2 * quantitative, 'planning': int(quantitative)}
                    elif tool == 'segment_instance':
                        cost = {'recompute': 2, 'network': 1}
                    elif tool == 'measure_instances':
                        cost = {'recompute': 1}
                    elif tool == 'presence_check' and role == 'required':
                        cost = {'observation': 1, 'network': 1}
                    else:
                        continue
                    cost['total'] = sum(cost.values())
                    if any(remaining.get(pool, 0) < count for pool, count in cost.items()):
                        continue
                    current = by_feature.get(feature)
                    if current and current['action_id'] == aid:
                        if cid not in current['candidate_ids']:
                            current['candidate_ids'].append(cid)
                        if cost['total'] > current['cost']['total']:
                            current['cost'] = cost
                    elif not current or cost['total'] < current['cost']['total']:
                        by_feature[feature] = {'action_id': aid, 'role': role, 'feature_id': feature,
                            'candidate_ids': [cid], 'cost': cost}
    return [r for rows in opportunities.values()
              for r in sorted(rows.values(), key=lambda r: r['cost']['total'])[:2]]


def protect_investigation_budget(runtime, actions, opportunities, remaining):
    """Do not sell the last feasible path to an unrelated supporting probe."""
    if not runtime.workflow.config.get('protect_investigation_budget', True):
        return actions, []
    protected = [r for r in opportunities if r['role'] == 'required'] or opportunities
    if not protected:
        return actions, []
    from TCM_Fusion.lib.runtime.probe_budget import check_reservation, describe, with_sponsor
    targets = runtime.workflow.targets()
    book = read_json(runtime.workflow.budget_path)
    selected = {a['action_id']: a for a in actions}
    kept, deferred = [], []
    for action in actions:
        tool = action['tool']
        if tool not in {'presence_check', 'feature_scan'}:
            kept.append(action)
            continue
        target = targets.get(action.get('args', {}).get('target_id'))
        probe = describe(runtime, action, target)
        if probe is None or any(r['action_id'] == action['action_id'] for r in opportunities):
            kept.append(action)
            continue
        if tool == 'presence_check':
            spend = {'observation': 1, 'network': 2, 'planning': 1, 'total': 4}
        else:
            # Include the next ordinary planning call in the reserve check.
            tiles = target['max_tiles']
            spend = {'recompute': 1, 'network': tiles + 1, 'planning': 1, 'total': tiles + 3}
        preview = {'supporting_probes': {**book.get('supporting_probes', {}),
                                        probe['key']: with_sponsor(book, probe)}}
        can_keep_path = False
        for row in protected:
            next_action = selected[row['action_id']]
            if row['feature_id'] == probe['feature_id'] and next_action['tool'] == tool:
                can_keep_path = True
                break
            if any(remaining.get(pool, 0) - spend.get(pool, 0) < count
                   for pool, count in row['cost'].items()):
                continue
            next_target = targets.get(next_action.get('args', {}).get('target_id'))
            if tool == next_action['tool'] == 'feature_scan':
                if not next_target or (target['image_id'] == next_target['image_id'] and
                        (target['remaining_scan_features'] <= 1 or target['remaining_scan_tiles'] <= tiles)):
                    continue
            next_probe = describe(runtime, next_action, next_target)
            if next_probe is None or check_reservation(preview, next_probe) is None:
                can_keep_path = True
                break
        if can_keep_path:
            kept.append(action)
        else:
            deferred.append({'tool': tool, 'feature_id': probe['feature_id'],
                             'reason': 'protect_remaining_investigation_budget'})
    return kept, deferred


def readiness_shadow(audit, formal):
    """候选审计与判定充分性分离的影子评估；未经独立验证，不提升交付资格。"""
    conclusion=formal.get('conclusion') or {}
    selected=conclusion.get('id') if isinstance(conclusion,dict) else conclusion
    if not selected:
        selected=next((r['id'] for r in formal.get('ranking',[])),None)
    required={r['candidate_id']:[x['feature_id'] for x in r['required'] if not x['assessed']]
              for r in audit['candidates']}
    supporting={r['candidate_id']:[x['feature_id'] for x in r['supporting'] if not x['assessed']]
                for r in audit['candidates']}
    authorized=bool(formal.get('verdict') in {'confirmed','probable','group_level'}
                    and not formal.get('needs_review'))
    return {'schema':'decision-readiness/shadow-v1','mode':'shadow','selected_candidate_id':selected,
            'audit_complete':audit['complete'],'formal_authorized':authorized,
            'selected_required_gaps':required.get(selected,[]),
            'competing_required_gaps':{cid:f for cid,f in required.items() if cid!=selected and f},
            'supporting_gaps':{cid:f for cid,f in supporting.items() if f},
            'unmapped':[x['label'] for x in audit['unmapped']],
            'would_be_ready':authorized and not any(required.values()) and not audit['unmapped'],
            'production_authorization_changed':False,
            'limit':'未查supporting不单独判否；required观察完成不等于排除竞争候选，仍需独立风险验证'}
