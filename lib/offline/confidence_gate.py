"""离线温度缩放与冻结策略的风险验证：真实放行只消费通过复验的版本，不自动造真值。

校准、开发、冻结测试三份数据必须在样品与图像层面隔离，真实模式下还须提供训练集
重叠审计；风险以 Clopper-Pearson 上界表述，样品独立性仍依赖数据负责人确认。

主要定义：
- dataset_rows。
- fit。
- upper_error。独立二项样品的一侧Clopper-Pearson上界；无放行样品时不可估计。
- validate。
- publish。
- save。

关联：
- 调用方 「cli/offline.py」、「tests/offline/test_confidence_gate.py」。
- 直接依赖 「lib/decision/collaboration.py」、「lib/decision/confidence_contract.py」、「lib/decision/routing_policy.py」、「lib/runtime/storage.py」。
"""
import copy
import math
import uuid
from TCM_Fusion.lib.runtime.storage import IntegrationError, digest, file_hash, project_path, read_json, write_json, output_root, finite_number
from TCM_Fusion.lib.decision.confidence_contract import seal, check, model_identity, calibrated
from TCM_Fusion.lib.decision.routing_policy import decide


def dataset_rows(dataset,split):
    if dataset.get('split')!=split or not isinstance(dataset.get('simulation'),bool) or not dataset.get('cases'):
        raise IntegrationError('GATE_DATA','数据须显式声明split/simulation且非空')
    groups=set();images=set();identity=None
    for row in dataset['cases']:
        group=row.get('group_id');image=row.get('image_sha256');a=row.get('prediction',{})
        if not isinstance(group,str) or not group or group in groups or not image or image in images:
            raise IntegrationError('GATE_SPLIT','重复/缺失物理样品或原图身份')
        if file_hash(project_path(row['image']))!=image:
            raise IntegrationError('INPUT_CHANGED','评估原图哈希不符')
        ranking=a.get('ranking') or [];names=[r['label'] for r in ranking];ps=[r['probability'] for r in ranking]
        if (a.get('status')!='ok' or len(ranking)<2 or any(not isinstance(n,str) for n in names) or len(set(names))!=len(names)
            or row.get('truth') not in names or any(not finite_number(p,maximum=1) for p in ps)
            or abs(sum(ps)-1)>len(ps)*.00005+1e-8):
            raise IntegrationError('GATE_DATA','标签/完整概率分布非法')
        if not dataset['simulation'] and a.get('provenance')!='original_cli':
            raise IntegrationError('SIMULATION_SCOPE','真实校准拒绝录制或替身预测')
        current=model_identity(a)
        if identity is not None and current!=identity:raise IntegrationError('GATE_MODEL','评估集混用模型')
        identity=current;groups.add(group);images.add(image)
    audit=dataset.get('training_audit')
    if not dataset['simulation']:
        if (not isinstance(audit,dict) or audit.get('verified') is not True or
            audit.get('model_identity')!=identity or not isinstance(audit.get('reference'),str) or not audit['reference'] or
            not isinstance(audit.get('group_ids'),list) or not audit['group_ids'] or
            not isinstance(audit.get('image_sha256s'),list) or not audit['image_sha256s']):
            raise IntegrationError('GATE_TRAIN_AUDIT','真实评估必须提供全部fold训练样品/图像并集的重叠审计')
        if (any(not isinstance(x,str) or not x for x in audit['group_ids']+audit['image_sha256s']) or
            groups & set(audit['group_ids']) or images & set(audit['image_sha256s'])):
            raise IntegrationError('GATE_TRAIN_OVERLAP','评估数据与A训练数据重叠或审计非法')
    return identity,groups,images


def fit(dataset):
    identity,groups,images=dataset_rows(dataset,'calibration')
    def loss(temp):
        return sum(-math.log(max(next(r['probability'] for r in calibrated(row['prediction'],temp)['ranking']
                                   if r['label']==row['truth']),1e-12)) for row in dataset['cases'])/len(groups)
    temperatures=[.5,.75,1,1.25,1.5,2,3,5]
    temp=min(temperatures,key=loss)
    return seal({'schema':'confidence-calibration/v1','temperature':temp,'model_identity':identity,
        'calibration_groups':sorted(groups),'calibration_images':sorted(images),'dataset_sha256':digest(dataset),
        'training_audit':copy.deepcopy(dataset.get('training_audit')),
        'simulation':dataset['simulation'],'nll_before':loss(1),'nll_after':loss(temp),
        'method':'grid_temperature_on_log_rounded_ensemble_probabilities','input_precision':4})


def upper_error(errors,total,alpha):
    """独立二项样品的一侧Clopper-Pearson上界；无放行样品时不可估计。"""
    if not total:return None
    if errors==total:return 1.0
    if errors==0:return 1-alpha**(1/total)
    low,high=0.,1.
    for _ in range(70):
        p=(low+high)/2
        logs=[math.lgamma(total+1)-math.lgamma(k+1)-math.lgamma(total-k+1)+k*math.log(p)+(total-k)*math.log1p(-p)
              for k in range(errors+1)]
        m=max(logs);cdf=math.exp(m)*sum(math.exp(v-m) for v in logs)
        if cdf>alpha:low=p
        else:high=p
    return (low+high)/2


def validate(calibration,policy,dataset):
    check(calibration,'confidence-calibration/v1');identity,groups,images=dataset_rows(dataset,'test')
    dev=policy.get('development_groups',[])
    dev_images=policy.get('development_images',[])
    if (policy.get('frozen') is not True or not isinstance(dev,list) or not dev or
        not isinstance(dev_images,list) or not dev_images or
        any(not isinstance(x,str) or not x for x in dev+dev_images) or
        groups & (set(dev)|set(calibration['calibration_groups'])) or
        images & (set(dev_images)|set(calibration['calibration_images'])) or
        set(dev)&set(calibration['calibration_groups']) or set(dev_images)&set(calibration['calibration_images'])):
        raise IntegrationError('GATE_SPLIT','校准/开发/冻结测试必须样品及图像隔离')
    if identity!=calibration['model_identity']:raise IntegrationError('GATE_MODEL','校准模型不匹配')
    if not calibration['simulation'] and not dataset['simulation']:
        audit=dataset['training_audit']
        if digest(audit)!=digest(calibration.get('training_audit')):
            raise IntegrationError('GATE_TRAIN_AUDIT','校准/测试必须使用同一训练并集审计')
        if set(dev)&set(audit['group_ids']) or set(dev_images)&set(audit['image_sha256s']):
            raise IntegrationError('GATE_TRAIN_OVERLAP','开发数据与A训练数据重叠')
    alpha=policy.get('alpha');risk=policy.get('risk_limit');minimum=policy.get('minimum_released')
    if not finite_number(alpha,minimum=1e-12,maximum=.999999) or not finite_number(risk,maximum=.999999) or type(minimum) is not int or minimum<1:
        raise IntegrationError('GATE_POLICY','风险/置信水平/最小样品数非法')
    from TCM_Fusion.lib.decision.routing_policy import route, validate_policy
    from TCM_Fusion.lib.decision.collaboration import knowledge_base
    cfg=validate_policy({**policy['routing'],'mode':'trial'});kb=knowledge_base()
    released=errors=0;class_counts={}
    for row in dataset['cases']:
        a=calibrated(row['prediction'],calibration['temperature']);decision=route(a,cfg,kb)
        if decision['skip']:
            released+=1;error=int(a['ranking'][0]['label']!=row['truth']);errors+=error
            counts=class_counts.setdefault(a['ranking'][0]['label'],{'released':0,'errors':0})
            counts['released']+=1;counts['errors']+=error
    bound=upper_error(errors,released,alpha)
    simulation=calibration['simulation'] or dataset['simulation']
    passed=(dataset.get('independence_confirmed') is True and released>=minimum and bound is not None and bound<=risk)
    return seal({'schema':'confidence-validation/v1','calibration':calibration,'policy':policy,'dataset':dataset,
        'model_identity':identity,'simulation':simulation,'released':released,'errors':errors,
        'coverage':released/len(groups),'error_rate':errors/released if released else None,
        'upper_error':bound,'passed':passed,'per_class_counts':class_counts,
        'scope':'aggregate_risk_under_declared_independent_sample_distribution',
        'limitations':['样品独立性和真值由数据负责人确认；哈希无法证明采集独立性或排除分布漂移']})


def publish(report):
    check(report,'confidence-validation/v1')
    repeated=validate(report['calibration'],report['policy'],report['dataset'])
    if not report.get('passed') or not repeated['passed']:
        raise IntegrationError('GATE_RELEASE','风险验证不通过，禁止发布可信放行')
    return seal({'schema':'confidence-release/v1','report':repeated,'version':uuid.uuid4().hex})


def save(kind,value):
    path=output_root()/'confidence_gate'/kind/(uuid.uuid4().hex+'.json');write_json(path,value);return path
