"""带独立真值的提示策略双回放：新旧指令配对比较，允许有证据的纠错。

只允许独立开发集，禁止分组与训练重叠；真值未知、标签范围不适配或回放未覆盖时直接失败，
不把模拟回放结果当作已验证的真实准确率，也不新发图像请求。

主要定义：
- validate_truth: 对每个开发样品回放基线/候选两条指令，按配对正确性、弃权与有界成本判定是否通过。

关联：
- .storage: 使用 IntegrationError、digest 做输入校验与工件绑定。
- .outcome_evaluation: 复用 evaluate 与 summarize 统计配对结果。
- .evidence_runtime / .react_graph: 用录制证据构造运行时并驱动 ReAct 回放。
- .prompt_optimization: 提供 RecordedTools 与 runtime_context，并在发布前调用 validate_truth。
"""
from .storage import IntegrationError,digest,read_json
from .outcome_evaluation import evaluate,summarize


async def validate_truth(candidate,dataset,*,allow_network=False,planner_factory=None):
    from .prompt_optimization import RecordedTools,runtime_context
    from .react_graph import LLMPlanner,run_react
    from .evidence_runtime import EvidenceRuntime
    if dataset.get('split')!='development' or not dataset.get('cases'):
        raise IntegrationError('PROMPT_SPLIT','真值提示验证仅允许独立开发集')
    if planner_factory is None and not allow_network:raise IntegrationError('NETWORK_DISABLED','规划回放需要显式文本联网许可')
    groups=set();rows=[];contexts=[];all_base=[];all_new=[]
    simulation=bool(candidate.get('simulation') or dataset.get('simulation') or planner_factory is not None)
    extra=dataset.get('max_extra_requests',0)
    if type(extra) is not int or not 0<=extra<=10:raise IntegrationError('PROMPT_BUDGET','额外请求预算须显式在0～10内')
    for case in dataset['cases']:
        group=case.get('group_id')
        if not group or group in groups or group in candidate['training_groups']:raise IntegrationError('PROMPT_LEAKAGE','开发分组重复或与训练重叠')
        groups.add(group);reference=case['reference']
        if reference.get('label') is None:raise IntegrationError('REPLAY_TRUTH','真值未知不能优化正确率')
        pair=[];covered=True
        for instruction in (candidate['baseline'],candidate['instruction']):
            runtime=EvidenceRuntime(case['records'],case['images'],sample_id=group,a=case.get('a'),
                config_identity=case['config_identity'],simulation=simulation)
            contexts.append(runtime_context(runtime));tools=RecordedTools(case['observations'])
            planner=planner_factory(instruction) if planner_factory else LLMPlanner()
            trace=await run_react(runtime,tools,planner=planner,planner_mode='llm',allow_network=allow_network,
                                  planning_instruction=instruction)
            covered &= not tools.misses and not trace['planner_degraded'] and trace['mandatory_complete']
            pair.append(evaluate(trace,{**reference,'group_id':group,'split':'development'},
                a_label=(case.get('a',{}).get('ranking') or [{}])[0].get('label')))
        base,new=pair;all_base.append(base);all_new.append(new)
        if base['correct'] is None or new['correct'] is None:
            raise IntegrationError('REPLAY_TRUTH','参考标签范围不能评价本任务结论')
        passed=bool(covered and not (base['correct'] and not new['correct']) and
            not (not new['abstained'] and not new['correct'] and (base['abstained'] or base['correct'])) and
            new['requests']<=base['requests']+extra)
        rows.append({'group_id':group,'baseline':base,'candidate':new,'passed':passed})
    base_summary=summarize(all_base);new_summary=summarize(all_new)
    improved=(new_summary['correct_determinate']>base_summary['correct_determinate'] or
              sum(r['requests'] for r in all_new)<sum(r['requests'] for r in all_base))
    return {'schema':'prompt-validation/v1','mode':'truth','candidate_sha256':digest(candidate),
        'dataset_sha256':digest(dataset),'dataset':dataset,'contexts':contexts,'simulation':simulation,
        'rows':rows,'baseline_summary':base_summary,'candidate_summary':new_summary,
        'passed':bool(all(r['passed'] for r in rows) and improved),
        'metric':'paired_correctness_and_abstention_with_bounded_cost',
        'real_accuracy_validated':False,'new_image_api_calls':0}
