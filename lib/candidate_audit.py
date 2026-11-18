"""候选审查账本：KB判据×真实证据/动作，不用Top1完成代替候选组审查。

被WorkflowRuntime/ReAct及final_assessment共同消费。只读当前工件、原资格投影和
逐折候选并集；不增加API、修改证据、评分或假设状态。暂停/失败/未映射不是排除。
"""
from .comparison_problems import _action_feature, evidence_paths
from .storage import read_json


def build(runtime, scored, observations, *, eligibility=None, actions=None):
    if eligibility is None:
        from .evidence_projection import build as project
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
