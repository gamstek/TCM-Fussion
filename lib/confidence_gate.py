"""离线温度缩放与冻结策略风险验证；真实放行只消费经过复验的版本，不自动造真值。"""
import copy
import math
import uuid
from .storage import IntegrationError, digest, file_hash, project_path, read_json, write_json, output_root, finite_number


def seal(value):return {**value,'sha256':digest(value)}


def check(value,kind):
    if value.get('schema')!=kind or value.get('sha256')!=digest({k:v for k,v in value.items() if k!='sha256'}):
        raise IntegrationError('GATE_ARTIFACT','门控工件格式或哈希错误')


def model_identity(a):
    s=a.get('signature') or {}
    if not s.get('weights') or not s.get('script') or not s.get('python_sha256'):
        raise IntegrationError('GATE_MODEL','预测缺模型身份')
    return digest({'weights':s['weights'],'script':s['script'],'python':s['python_sha256'],
                   'classes':sorted(r['label'] for r in a['ranking'])})


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


def calibrated(a,temperature):
    result=copy.deepcopy(a);rows=result['ranking']
    logs=[math.log(max(r['probability'],1e-12))/temperature for r in rows]
    m=max(logs);zs=[math.exp(v-m) for v in logs];z=sum(zs)
    for r,p in zip(rows,zs):r['probability']=p/z
    result['ranking']=sorted(rows,key=lambda r:r['probability'],reverse=True)
    return result


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
    from .routing_policy import route, validate_policy
    from .collaboration import knowledge_base
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


def decide(a,path,kb,aliases=None):
    release=read_json(project_path(path));check(release,'confidence-release/v1')
    report=release['report'];check(report,'confidence-validation/v1')
    if not report['passed'] or report['simulation'] or a.get('provenance')!='original_cli':
        raise IntegrationError('GATE_SCOPE','未通过或模拟版本不能用于真实放行')
    if aliases:raise IntegrationError('GATE_SCOPE','本版可信门只验证直接药材标签，不接受额外映射')
    if model_identity(a)!=report['model_identity']:raise IntegrationError('GATE_MODEL','权重/脚本/解释器/类别不匹配')
    cal=report['calibration'];check(cal,'confidence-calibration/v1')
    from .routing_policy import route
    # 放行规则必须与冻结测试完全相同；不能看过测试类别计数后再改选择规则。
    result=route(calibrated(a,cal['temperature']),{**report['policy']['routing'],'mode':'trial'},kb)
    return {**result,'mode':'validated','calibrated':True,'calibration_input_precision':4,
            'release_version':release['version'],'validation_upper_error':report['upper_error']}


def save(kind,value):
    path=output_root()/'confidence_gate'/kind/(uuid.uuid4().hex+'.json');write_json(path,value);return path
