"""工作流提示的开发集双回放门：重放原 B 物理工件，不新发任何图像请求。

只使用独立 development 回放集并要求报告与参考反馈哈希绑定；非真值模式下答案必须
来自独立参考裁决，不能直接把模型结果当真值。通过条件同时约束覆盖率、退化、
物理证据摘要一致与请求/阶段尝试不增加。

主要定义：
- validate_workflow: 对每个开发样品分别以 baseline/candidate 指令回放，输出是否放行。

关联：
- 调用方 「lib/offline/prompt_optimization.py」。
- 直接依赖 「lib/agent/react_graph.py」、「lib/offline/outcome_evaluation.py」、「lib/offline/workflow_replay.py」、「lib/runtime/storage.py」。
"""
import uuid
from TCM_Fusion.lib.runtime.storage import IntegrationError, digest, output_root,read_json
from TCM_Fusion.lib.offline.workflow_replay import replay_report


async def validate_workflow(candidate,dataset,*,allow_network=False,planner_factory=None):
    """由prompt_optimization.validate调用；发布时经同一入口重新双回放。"""
    from TCM_Fusion.lib.agent.react_graph import LLMPlanner
    if dataset.get('split')!='development' or not dataset.get('cases'):
        raise IntegrationError('PROMPT_SPLIT','工作流提示仅使用独立development回放集')
    if planner_factory is None and not allow_network:
        raise IntegrationError('NETWORK_DISABLED','回放仅规划文本调用也须显式允许联网')
    groups=set();rows=[];contexts=[]
    truth_mode=dataset.get('mode')=='workflow_truth'
    simulation=bool(candidate.get('simulation') or dataset.get('simulation',True) or planner_factory is not None)
    root=output_root()/'workflow_prompt_replay'/uuid.uuid4().hex[:12]
    for index,case in enumerate(dataset['cases']):
        group=case.get('group_id');report=case.get('report',{});expected=case.get('expected')
        if not group or group in groups or group in candidate['training_groups']:
            raise IntegrationError('PROMPT_LEAKAGE','工作流开发样品须唯一并与训练分组隔离')
        if case.get('report_sha256')!=digest(report):
            raise IntegrationError('PROMPT_DATA','工作流报告与参考反馈未绑定')
        if not truth_mode and (not isinstance(expected,dict) or set(expected)!={'verdict','conclusion_id'}):
            raise IntegrationError('REPLAY_TRUTH','需独立参考裁决和结论，不能直接把模型结果当真值')
        if truth_mode and case.get('reference',{}).get('label') is None:
            raise IntegrationError('REPLAY_TRUTH','纠错验证需要独立真值')
        groups.add(group);simulation |= report.get('simulation',True) is not False
        pair=[]
        for label,instruction in (('b',candidate['baseline']),('c',candidate['instruction'])):
            planner=planner_factory(instruction) if planner_factory else LLMPlanner()
            value=await replay_report(report,root/str(index)/label,planner=planner,planning_instruction=instruction)
            contexts.append(value.pop('runtime_context'));pair.append(value)
        base,new=pair
        equal=('signature','physical_evidence','evidence_digest','needs_review','mandatory_complete')
        passed=(base['covered'] and new['covered'] and not base['degraded'] and not new['degraded']
                and base['signature']==new['signature']==expected
                and all(base[k]==new[k] for k in equal)
                and new['requests']<=base['requests'] and new['stage_attempts']<=base['stage_attempts'])
        if truth_mode:
            from TCM_Fusion.lib.offline.outcome_evaluation import evaluate
            reference={**case['reference'],'group_id':group,'split':'development'}
            b=evaluate(read_json(base['trace']),reference);n=evaluate(read_json(new['trace']),reference)
            if b['correct'] is None or n['correct'] is None:raise IntegrationError('REPLAY_TRUTH','真值范围不适配药材样品裁决')
            extra=dataset.get('max_extra_requests',0)
            if type(extra) is not int or not 0<=extra<=10:raise IntegrationError('PROMPT_BUDGET','额外预算非法')
            base['outcome']=b;new['outcome']=n
            passed=(base['covered'] and new['covered'] and not base['degraded'] and not new['degraded']
                and base['mandatory_complete'] and new['mandatory_complete'] and
                not (b['correct'] and not n['correct']) and not (not n['abstained'] and not n['correct'] and (b['abstained'] or b['correct']))
                and new['requests']<=base['requests']+extra and new['stage_attempts']<=base['stage_attempts']+extra)
        rows.append({'group_id':group,'baseline':base,'candidate':new,'passed':passed})
    return {'schema':'prompt-validation/v1','mode':'workflow','candidate_sha256':digest(candidate),
            'dataset_sha256':digest(dataset),'dataset':dataset,'contexts':contexts,'simulation':simulation,
            'rows':rows,'passed':all(r['passed'] for r in rows) and any(
                (truth_mode and r['candidate']['outcome']['correct'] and not r['baseline']['outcome']['correct']) or
                r['candidate']['requests']<r['baseline']['requests'] or
                r['candidate']['stage_attempts']<r['baseline']['stage_attempts'] for r in rows),
            'metric':'paired_truth_with_bounded_cost' if truth_mode else 'paired_requests_and_stage_attempts_with_equal_physical_evidence_and_verdict',
            'real_accuracy_validated':False,'new_image_api_calls':0}
