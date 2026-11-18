"""B0—B3 既有报告的配对评估与用途资格误拦/误放统计，不生成观察、不泄漏答案。

只接受开发或冻结测试集，模拟/诊断报告与重复样品一律拒绝；不拟合测试阈值，
不宣称风险已验收，区间仅作描述性统计并用 Wilson 置信区间表达。

主要定义：
- wilson: 计算比例的二项 Wilson 置信区间。
- compare: 对每个样品的四组报告配对评估，输出分组摘要与错误输出风险区间。
- quality_compare: 用人工用途参考统计新旧资格的误拦/误放计数及置信区间。

关联：
- 调用方 「cli/offline.py」、「tests/offline/test_evaluation.py」。
- 直接依赖 「lib/offline/outcome_evaluation.py」、「lib/runtime/storage.py」。
"""
import math
from TCM_Fusion.lib.offline.outcome_evaluation import evaluate,summarize
from TCM_Fusion.lib.runtime.storage import read_json,project_path,IntegrationError


def wilson(errors,n):
    if not n:return None
    z=1.96;p=errors/n;d=1+z*z/n
    c=(p+z*z/(2*n))/d;half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return [max(0,c-half),min(1,c+half)]


def compare(dataset):
    if dataset.get('split') not in {'development','test'}:raise IntegrationError('EVAL_SPLIT','消融需要开发或冻结测试集')
    cases=dataset.get('cases',[])
    if not cases:raise IntegrationError('EVAL_EMPTY','没有独立样品，不能宣称风险验收')
    groups={k:[] for k in ('B0','B1','B2','B3')};seen=set();paired=[]
    for case in cases:
        group=case['group_id']
        if group in seen:raise IntegrationError('EVAL_GROUP','同一样品重复或视野未合并')
        seen.add(group)
        if set(case['reports'])!=set(groups):raise IntegrationError('EVAL_GROUP','缺少四组报告')
        row={'group_id':group,'results':{}}
        for key,path in case['reports'].items():
            report=read_json(project_path(path))
            if report.get('simulation') or report.get('diagnostic_only'):raise IntegrationError('EVAL_SCOPE','模拟/诊断报告不能作正式验收')
            trace=report.get('agent') or {'decision':report['b']['verdict'],'needs_review':report['needs_review']}
            result=evaluate(trace,{**case['reference'],'group_id':group,'split':dataset['split']},
                            a_label=(report.get('a',{}).get('ranking') or [{}])[0].get('label'))
            groups[key].append(result);row['results'][key]=result
        paired.append(row)
    summary={k:summarize(v) for k,v in groups.items()}
    for s in summary.values():s['wrong_output_risk_ci95']=wilson(s['wrong_determinate'],s['determinate'])
    return {'schema':'ablation-report/v1','split':dataset['split'],'paired':paired,'summary':summary,
            'risk_acceptance':'not_assessed_requires_preregistered_limits','thresholds_fitted_on_test':False}


def quality_compare(dataset):
    """人工用途参考对新旧资格的误拦/误放统计；不拟合测试阈值。"""
    import contracts
    if dataset.get('split') not in {'development','test'} or not dataset.get('cases'):
        raise IntegrationError('EVAL_EMPTY','用途评测需要独立开发/测试人工参考')
    counts={u:{'usable':0,'unusable':0,'legacy_false_reject':0,'legacy_false_accept':0,
               'new_false_reject':0,'new_false_accept':0} for u in contracts.QUALITY_USES}
    seen=set()
    for c in dataset['cases']:
        if not c.get('label_source') or not c.get('group_id') or c.get('case_id') in seen:
            raise IntegrationError('EVAL_REFERENCE','用途真值来源/分组/实例身份无效')
        seen.add(c.get('case_id'))
        qa=contracts.quality_assessment(read_json(project_path(c['assessment'])))
        for use,truth in c['reference_uses'].items():
            if use not in counts or type(truth) is not bool:raise IntegrationError('EVAL_REFERENCE','用途真值必须是明确布尔值')
            s=counts[use];s['usable' if truth else 'unusable']+=1
            old=c['legacy_status']=='ok';new=qa['uses'][use]['allowed']
            for label,allowed in [('legacy',old),('new',new)]:
                if truth and not allowed:s[label+'_false_reject']+=1
                if not truth and allowed:s[label+'_false_accept']+=1
    for s in counts.values():
        for label in ('legacy','new'):
            s[label+'_false_accept_ci95']=wilson(s[label+'_false_accept'],s['unusable'])
    return {'schema':'quality-comparison/v1','split':dataset['split'],'counts':counts,
            'thresholds_fitted_on_test':False,'risk_acceptance':'requires_preregistered_limits',
            'interval_scope':'descriptive_per_instance_wilson; clustered_samples_require_group_resampling'}
