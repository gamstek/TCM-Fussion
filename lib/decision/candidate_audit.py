"""逐项对照候选判据、当前用途资格、任务动作和观察账本。输出候选审查状态、可达入口及缺口，供 ReAct 和收尾共用。区分未查、失败、未映射和真实反证。

主要定义：
- build。
- compact。每候选仅发状态与缺口，不重复判据全文；完整账本落盘/收尾可查。
- investigation_opportunities。选择已获准的关键判据入口及预算需求，供有界规划菜单使用。
- protect_investigation_budget。保留一个可执行的关键入口，阻止弱探查耗尽所需余额。
- readiness_shadow。候选审计与判定充分性分离的影子评估；未经独立验证，不提升交付资格。

关联：
- 调用方 「lib/agent/react_graph.py」、「lib/decision/final_assessment.py」、「lib/workflow/workflow_runtime.py」、「tests/decision/test_candidate_audit.py」、「tests/decision/test_final_assessment.py」、「tests/integration/test_fair_investigation.py」。
- 直接依赖 「lib/decision/comparison_problems.py」、「lib/evidence/evidence_projection.py」、「lib/evidence/search_ledger.py」、「lib/runtime/storage.py」。
"""
from pathlib import Path

from TCM_Fusion.lib.decision.comparison_problems import _action_feature, evidence_paths
from TCM_Fusion.lib.evidence.evidence_projection import presence_state
from TCM_Fusion.lib.evidence.search_ledger import scan_coverage
from TCM_Fusion.lib.runtime.storage import read_json


def _batch_outcomes(op, artifacts):
    if 'batch_outcomes' in op:
        return op['batch_outcomes']
    # 旧账本中成功的父批次也可能拒收某项。用该项提交工件核对原回执。
    bundle = op.get('bundle') or {}
    outcomes = {row['crop_id']: 'unknown' for row in bundle.get('failures', [])}
    for row in bundle.get('classifications', []):
        cid = row['crop_id']
        path = artifacts.get('classification:' + cid, {}).get('path')
        committed = path and bundle.get('root') and row.get('classification') and (
            Path(path) == Path(bundle['root']) / row['classification'] / 'classification.json')
        outcomes[cid] = 'completed' if committed else 'unknown'
    return outcomes


def feature_receipts(runtime, observations, targets, *, operations=None):
    """按特征汇集有结果的定向观察和操作。同一物理回执可由多个候选引用。"""
    receipts = {}
    for observation in observations:
        action = observation.get('action') or {}
        args = action.get('args') or {}
        fid, image = args.get('feature_id'), args.get('image_id')
        status = observation.get('status')
        if (action.get('tool') != 'presence_check' or not fid or image not in runtime.images
                or status not in (None, 'done', 'failed', 'cancelled', 'late')
                or not observation.get('result') and status not in ('failed', 'cancelled', 'late')):
            continue
        outcome = presence_state(observation, runtime.kb, runtime.images) if status in (None, 'done') else 'unknown'
        record = {'tool': 'presence_check', 'image_id': image, 'outcome': outcome,
                  'action_id': action.get('action_id')}
        if record not in receipts.setdefault(fid, []):
            receipts[fid].append(record)

    artifacts = None
    if operations is None:
        operations = read_json(runtime.workflow.operations)
    for key, op in operations['entries'].items():
        if op.get('status') not in ('done', 'failed', 'cancelled', 'late'):
            continue
        legacy_result = 'accepted' not in op and bool(op.get('bundle') or op.get('scan_coverage'))
        accepted = op.get('accepted') is True or legacy_result
        completed = op['status'] == 'done' and accepted
        if op['tool'] == 'classify_crops_batch':
            if artifacts is None:
                artifacts = runtime.workflow.store.current()['artifacts']
            for cid, outcome in _batch_outcomes(op, artifacts).items():
                crop = artifacts.get(cid, {})
                if crop.get('image_id') not in runtime.images:
                    continue
                fid = _action_feature({'args': {'target_id': cid}}, artifacts)
                if fid:
                    receipts.setdefault(fid, []).append({'tool': op['tool'], 'operation_key': key, 'crop_id': cid,
                        'image_id': crop['image_id'], 'outcome': outcome if completed else 'unknown'})
            continue
        if op['tool'] not in {'feature_scan', 'locate_feature', 'classify_target', 'inspect_crop',
                              'segment_instance', 'measure_instances'}:
            continue
        target = targets.get(op.get('target_id'), {})
        fid = op.get('feature_id') or _action_feature({'args': {'target_id': op.get('target_id')}}, targets)
        image = op.get('image_id') or target.get('image_id')
        if not fid or image not in runtime.images:
            continue
        record = {'tool': op['tool'], 'operation_key': key, 'image_id': image,
                  'outcome': 'completed' if completed else 'unknown'}
        if completed and op['tool'] == 'feature_scan':
            coverage, details = scan_coverage(op)
            count = details.get('crop_count')
            record['coverage'] = coverage
            record['outcome'] = 'unknown'
            if (coverage in ('shareable', 'partial') and type(count) is int and count >= 0
                    and (count > 0 or details.get('successful_tiles'))):
                record.update(outcome='crops_found' if count else 'no_crop', crop_count=count)
                for field in ('successful_tiles', 'failed_tiles', 'unsearched_tiles'):
                    if isinstance(details.get(field), list):
                        record[field] = len(details[field])
        if op.get('error_code'):
            record['error_code'] = op['error_code']
        receipts.setdefault(fid, []).append(record)
    return receipts


_INVESTIGATION_LABELS = {
    'unchecked': '未查', 'investigated_unresolved': '已查仍未解决', 'failed_or_unknown': '回执失败或不可判',
    'local_absent': '已提供视野均未见', 'local_absent_partial': '部分视野未见',
    'present_pending': '已见结构、判据待核验', 'supported': '已有合格支持',
    'quantitative_counter': '已有定量反证',
}


def required_investigation_summary(candidates, unmapped):
    """供规划和收尾共用的必需判据事实摘要，输入为主机生成的候选审查行。"""
    groups = {}
    for candidate in candidates:
        for criterion in candidate['required']:
            groups.setdefault(criterion['state'], []).append(f"{candidate['candidate_id']}/{criterion['feature_id']}")
    parts = [f"{_INVESTIGATION_LABELS[state]}（{'、'.join(items)}）" for state, items in groups.items()]
    if unmapped:
        parts.append(f"缺少知识库映射，未查（{'、'.join(row['label'] for row in unmapped)}）")
    return '。'.join(parts) or '当前没有已绑定的候选必需判据'


def build(runtime, scored, observations, *, eligibility=None, actions=None, operations=None):
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
    provided=set(runtime.images)
    targets=runtime.workflow.targets()
    actions=actions if actions is not None else runtime.legal_actions({'observations':observations,'attempted':[]})
    paths=evidence_paths(runtime,actions,{c['feature'] for cid in requested if cid in catalog
                                         for c in catalog[cid]['criteria']})
    available={fid:[p['action_id'] for p in ps] for fid,ps in paths.items()}
    receipts=feature_receipts(runtime, observations, targets, operations=operations)
    rows=[]
    for cid in requested:
        if cid not in catalog:continue
        required=[];supporting=[]
        for c in catalog[cid]['criteria']:
            fid=c['feature'];check=checks.get(cid,{}).get(fid,{})
            fact=by_criterion.get((cid,fid),{})
            issued=receipts.setdefault(fid,[])
            observed_views={}
            for receipt in issued:
                if receipt['tool']=='presence_check':
                    observed_views.setdefault(receipt['image_id'],[]).append(receipt['outcome'])
            states={r['outcome'] for r in issued}
            supported=cid in fact.get('supports_candidate_ids',[])
            counter=cid in fact.get('counters_candidate_ids',[])
            seen_positive=bool(check.get('present')) or 'present' in states
            # 合格工件或阳性观察仍在时保留待核验状态。
            full_local_absence=not seen_positive and bool(provided) and all(
                observed_views.get(image) and all(s=='absent' for s in observed_views[image]) for image in provided)
            partial_local_absence='absent' in states and not seen_positive
            assessed=supported or counter or full_local_absence
            state=('supported' if supported else 'quantitative_counter' if counter else
                   'local_absent' if full_local_absence else 'local_absent_partial' if partial_local_absence else
                   'present_pending' if seen_positive else
                   'investigated_unresolved' if states.intersection({'completed','crops_found','no_crop'}) else
                   'failed_or_unknown' if 'unknown' in states else 'unchecked')
            item={'feature_id':fid,'state':state,'rule_state':check.get('state','NOT_DETECTED'),
                'assessed':bool(assessed),'satisfies_criterion':supported,
                'observed_fov_count':len(observed_views), 'provided_fov_count':len(provided),
                'availability_reason':('reachable_via_prerequisite' if all(
                    p['kind']=='discovery_prerequisite' for p in paths[fid]) else 'available') if available.get(fid) else
                    'hypothesis_paused' if status.get(cid) in {'paused','withdrawn'} else
                    'prior_tool_failure' if state=='failed_or_unknown' else 'no_current_legal_action',
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
        'receipts_by_feature':receipts,'required_investigation_summary':required_investigation_summary(rows, unmapped),
        'candidate_ids':requested,'complete':not pending and not unmapped,
        'incomplete_candidate_ids':pending,'actionable_gaps':actionable,
        'comparison_required':bool(requested or unmapped),
        'scope':'current provided fields of view; local absence never excludes a specimen',
        'policy':'review is always allowed; incomplete audit cannot authorize a complete differential conclusion'}


def first_contrast(audit):
    """返回当前映射候选间首个合格判据差异及可复核的事实 ID。局部未见不作反证。"""
    candidates = audit['candidates']
    if len(candidates) < 2:
        return None
    by_feature = {}
    for row in candidates:
        for item in row['required'] + row['supporting']:
            by_feature.setdefault(item['feature_id'], {})[row['candidate_id']] = item
    for feature, items in sorted(by_feature.items()):
        supports = {cid: item['fact_ref'] for cid, item in items.items()
                    if item['state'] == 'supported' and item['fact_ref']}
        counters = {cid: item['fact_ref'] for cid, item in items.items()
                    if item['state'] == 'quantitative_counter' and item['fact_ref']}
        outside = sorted(row['candidate_id'] for row in candidates if row['candidate_id'] not in items)
        if supports and (counters or outside):
            return {'feature_id': feature, 'support_refs': supports, 'counter_refs': counters,
                    'other_candidates_without_criterion': outside,
                    'basis': 'opposing_criteria' if counters else 'candidate_specific_criterion'}
    return None


def compact(value):
    """每候选仅发状态与缺口，不重复判据全文；完整账本落盘/收尾可查。"""
    return {k:value[k] for k in ('schema','complete','candidate_ids','unmapped',
        'incomplete_candidate_ids','actionable_gaps','comparison_required','scope')} | {
        'receipts_by_feature':value.get('receipts_by_feature', {}),
        'required_investigation_summary':value.get('required_investigation_summary', ''),
        'candidates':[{'candidate_id':r['candidate_id'],'hypothesis_status':r['hypothesis_status'],
            'required':{x['feature_id']:x['state']for x in r['required']},
            'supporting':{x['feature_id']:x['state']for x in r['supporting']},
            'review_complete':r['review_complete']}for r in value['candidates']]}


def investigation_opportunities(runtime, audit, focus, actions, remaining):
    """从所有活动候选中推荐可改变判据判断的入口。仅给推荐排序，合法动作由工具层决定。"""
    interest = runtime.candidate_interest()
    active = set(interest['ids'])
    leads = {focus.get('current_rule_lead'), *focus.get('a_leads', []), *focus.get('competing_support', {})}
    profiles = {}
    for candidate in runtime.kb.candidates():
        if candidate['id'] in active:
            profiles[candidate['id']] = {c['feature']: c for c in candidate['criteria']}
    by_id = {a['action_id']: a for a in actions}
    targets = runtime.workflow.targets()
    opportunities = {'required': {}, 'supporting': {}}
    for row in audit['candidates']:
        cid = row['candidate_id']
        if cid not in profiles:
            continue
        required_complete = bool(row['required']) and all(c['satisfies_criterion'] for c in row['required'])
        for role, routes in opportunities.items():
            if role == 'supporting' and not required_complete:
                continue
            for criterion in row[role]:
                feature = criterion['feature_id']
                if criterion['assessed']:
                    continue
                profile = profiles[cid][feature]
                others = [p.get(feature) for id_, p in profiles.items() if id_ != cid]
                differs = any(other is None or any(
                    other.get(k) != profile.get(k) for k in ('role', 'quant', 'morph', 'abundance')
                ) for other in others)
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
                    elif tool == 'presence_check':
                        cost = {'observation': 1, 'network': 1}
                    else:
                        continue
                    cost['total'] = sum(cost.values())
                    if any(remaining.get(pool, 0) < count for pool, count in cost.items()):
                        continue
                    entry = routes.setdefault((feature, aid), {
                        'action_id': aid, 'role': role, 'feature_id': feature, 'candidate_ids': [],
                        'criteria_differ': False, 'cost': cost})
                    if cid not in entry['candidate_ids']:
                        entry['candidate_ids'].append(cid)
                    entry['criteria_differ'] |= differs
                    if cost['total'] > entry['cost']['total']:
                        entry['cost'] = cost
    recommended = {}
    for routes in opportunities.values():
        # 比较同一物理观察的各个消费者，再选可共用的入口。每类角色仍只展示两种特征。
        by_feature = {}
        for entry in sorted(routes.values(), key=lambda r: (
                not r['criteria_differ'], r['cost']['total'], -len(r['candidate_ids']),
                not leads.intersection(r['candidate_ids']))):
            by_feature.setdefault(entry['feature_id'], entry)
        for entry in list(by_feature.values())[:2]:
            aid = entry['action_id']
            if aid in recommended:
                # required 入口已列出，同一动作还可服务这些候选的 supporting。
                recommended[aid]['also_supporting_for'] = entry['candidate_ids']
            else:
                recommended[aid] = entry
    return list(recommended.values())


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
