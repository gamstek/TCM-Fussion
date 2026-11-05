"""已审阅的工具实现；注册表只接受受信callable，不执行配置代码。"""
from .storage import IntegrationError


def read_builtin(runtime, name, args, state):
    scored = runtime.score(state.get('observations', []))
    common = {'sample_id': runtime.sample_id, 'evidence_digest': scored['evidence_digest'],
              'registry_sha256': runtime.tools.signature()}
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
        from .artifact_index import ArtifactIndex
        result=ArtifactIndex(getattr(runtime,'artifact_roots',[])).scan()
        offset=args.get('offset',0);limit=args.get('limit',50)
        result['total']=len(result['artifacts']);result['offset']=offset;result['limit']=limit
        result['artifacts']=result['artifacts'][offset:offset+limit]
    else:
        raise IntegrationError("TOOL_UNIMPLEMENTED", "工具没有已实现的只读handler")
    return {**common, **result}
