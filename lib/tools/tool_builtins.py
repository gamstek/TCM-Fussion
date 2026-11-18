"""读取会话提供的证据、质量、标定和候选状态。按受信工具名生成受分页约束的只读结果，供本地工具注册表、MCP 和 Agent 详情读取消费。

主要定义：
- read_builtin: 按工具名分派 read_quality、read_evidence、read_action_catalog、read_evidence_detail、list_features、query_criteria、read_scale、list_fovs、criteria_diff、list_artifacts 等只读处理器，输出附带证据摘要与版本。

关联：
- 调用方 「lib/tools/tool_registry.py」。
- 直接依赖 「lib/evidence/artifact_index.py」、「lib/runtime/storage.py」。
"""
from TCM_Fusion.lib.runtime.storage import IntegrationError, read_json


def read_builtin(runtime, name, args, state):
    scored = runtime.score(state.get('observations', []))
    common = {'sample_id': runtime.sample_id, 'evidence_digest': scored['evidence_digest'],
              'registry_sha256': runtime.tools.signature()}
    workflow=getattr(runtime,'workflow',None)
    if workflow is not None:
        common['evidence_version']=workflow.store.current()['version']
    if name == 'read_quality':
        workflow=getattr(runtime,'workflow',None)
        result={'assessments':[],'unobserved_state':'unknown'}
        if workflow is not None:
            result['assessments']=[{'target_id':k,'assessment':workflow.assessment(v)} for k,v in
                workflow.store.current()['artifacts'].items() if v['valid'] and v['kind']=='segmentation' and v.get('assessment_id')]
            result['segmentation_reviews']=workflow.segmentation_reviews()
    elif name == 'read_evidence':
        offset, limit = args['offset'], args['limit']
        # records在runtime构造时已去除真值herb；不读取attributes或labels。
        result = {'records': runtime.records[offset:offset+limit], 'total': len(runtime.records)}
    elif name == 'read_action_catalog':
        offset,limit=args.get('offset',0),args.get('limit',100)
        workflow=getattr(runtime,'workflow',None)
        actions=workflow.propose() if workflow is not None else runtime.legal_actions(state)
        items=[]
        for action in actions:
            a=action.get('args') or {}
            items.append({'action_id':action.get('action_id'),'tool':action.get('tool'),
                          'target_id':a.get('target_id'),'feature_id':a.get('feature_id'),
                          'image_id':a.get('image_id'),'cost':runtime.tools.get(action.get('tool')).budget_class
                          if action.get('tool') in runtime.tools.specs else None})
        return {**common,'total':len(items),'offset':offset,'limit':limit,
                'actions':items[offset:offset+limit],
                'note':'只读动作目录；选择仍受当前合法集合、前置条件和预算约束'}
    elif name == 'read_evidence_detail':
        kind=args.get('kind');offset=args.get('offset',0);limit=args.get('limit',20)
        workflow=getattr(runtime,'workflow',None)
        if workflow is None:
            raise IntegrationError('EVIDENCE_SCOPE','没有工作流证据可展开')
        if kind == 'candidate_catalog':
            rows = [{'id': row['id'], 'display': row.get('display', row['id']),
                     'required': [c for c in row['criteria'] if c['role']=='required']}
                    for row in runtime.kb.candidates()]
        elif kind == 'provisional':
            rows = runtime.provisional_state(scored['decision'])['ranking']
        elif kind == 'observations':
            rows = list(state.get('observations', []))
        elif kind in {'eligibility','eligibility_facts','comparison','instances'}:
            view = runtime.decision_view(state, scored=scored)
            if kind == 'eligibility':
                rows = view['eligibility']['projections']
                common['candidate_audit']=view['candidate_audit']
                common['investigation_focus']=view['investigation_focus']
            elif kind == 'eligibility_facts':
                rows = view['eligibility']['facts']
            elif kind == 'comparison':
                rows = view['comparison']['questions'] + view['comparison']['boundaries']
            else:
                rows = list(view['instances']['instances'].values()) + view['instances']['associations']
        elif kind=='classifications':
            rows=[]
            for v in workflow.store.current()['artifacts'].values():
                if v.get('valid') and v.get('kind')=='classification':
                    doc=read_json(v['path'])
                    rows.append({'crop_id':v.get('crop_id'),'feature_id':v.get('feature_id'),
                                 'label':doc.get('label'),'confidence':doc.get('confidence'),
                                 'reject':doc.get('reject'),'identity_verified':v.get('identity_verified')})
        elif kind=='operations':
            rows=list(read_json(workflow.operations)['entries'].values())
        elif kind=='quality':
            rows=[workflow.assessment(v) for v in workflow.store.current()['artifacts'].values()
                  if v.get('valid') and v.get('kind')=='segmentation' and v.get('assessment_id')]
        elif kind=='measurements':
            rows=[]
            for v in workflow.store.current()['artifacts'].values():
                if v.get('valid') and v.get('kind')=='measurement':
                    rows.append({'artifact_id':v['id'],'crop_id':v.get('crop_id'),
                                 'feature_id':v.get('feature_id'),'path':v.get('path')})
        else:
            raise IntegrationError('EVIDENCE_SCOPE','未知证据类型')
        return {**common,'kind':kind,'total':len(rows),'offset':offset,'limit':limit,
                'rows':rows[offset:offset+limit]}
    elif name == 'list_features':
        result = {'features': {fid: vars(card) for fid,card in scored['cards'].items()}}
    elif name == 'query_criteria':
        rows = {r['id']: r for r in runtime.kb.candidates()}
        if args['candidate_id'] not in rows:
            raise IntegrationError('ACTION_DENIED', '未知判据候选')
        result = {'criteria': rows[args['candidate_id']]}
    elif name == 'read_scale':
        scales = [{k:r[k] for k in ('image_stem','calibration_px_per_um','calibration_status') if k in r}
                  for r in runtime.records]
        workflow = getattr(runtime, 'workflow', None)
        if workflow is not None:
            snap = workflow.store.current()
            scales.extend({'image_id':v['image_id'],'px_per_um':v['px_per_um'],
                           'valid':v['valid'],'evidence_version':snap['version'],
                           'source':'trusted_pipeline_artifact'}
                          for v in snap['artifacts'].values() if v['kind']=='scale')
        result = {'scales': scales, 'note': '只读已有标定；缺失不是0；失效尺度不得用于测量'}
    elif name == 'list_fovs':
        result = {'fovs': [{'image_id': k, 'sha256': v['sha256'], 'stems': v['stems']}
                           for k,v in runtime.images.items()]}
    elif name == 'criteria_diff':
        result = {'criteria_diff': runtime.summary(state)['criteria_diff']}
    elif name in {'list_artifacts','read_artifact_status'}:
        from TCM_Fusion.lib.evidence.artifact_index import ArtifactIndex
        result=ArtifactIndex(getattr(runtime,'artifact_roots',[])).scan()
        offset=args.get('offset',0);limit=args.get('limit',50)
        result['total']=len(result['artifacts']);result['offset']=offset;result['limit']=limit
        result['artifacts']=result['artifacts'][offset:offset+limit]
    else:
        raise IntegrationError("TOOL_UNIMPLEMENTED", "工具没有已实现的只读handler")
    return {**common, **result}
