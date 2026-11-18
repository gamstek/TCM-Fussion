"""运行后的独立真值评估：把轨迹结论与外部参考标签配对，不用模型自评充当真值。

评估只在离线阶段进行，不向在线 planner 注入标签；结论范围与参考标签范围不一致时
判定为不可比（correct 为 None），失败归因保持未解决，不推测根因。

主要定义：
- evaluate: 用参考标签与划分评估单条轨迹，输出是否正确、是否弃权、与 A 的修正关系。
- summarize: 汇总多行评估结果，统计确定输出、弃权、纠正 A 与破坏正确 A 的计数及覆盖率。

关联：
- 调用方 「cli/offline.py」、「lib/offline/ablation_evaluation.py」、「lib/offline/truth_prompt_validation.py」、「lib/offline/workflow_prompt_validation.py」、「tests/offline/test_evaluation.py」。
- 直接依赖 「lib/runtime/storage.py」。
"""
from TCM_Fusion.lib.runtime.storage import IntegrationError,digest


def evaluate(trace,reference,*,a_label=None):
    if not reference.get('label_source') or not reference.get('group_id') or reference.get('split') not in {'training','development','test'}:
        raise IntegrationError('REFERENCE_INVALID','真值须有来源、分组和划分')
    truth=reference.get('label');decision=trace.get('decision',{})
    conclusion=decision.get('conclusion') or {}
    predicted=conclusion.get('id') if isinstance(conclusion,dict) else conclusion
    abstain=not predicted or trace.get('needs_review',True)
    label_scope=reference.get('label_scope','unspecified')
    compatible=label_scope==trace.get('decision_scope','sample_identity')
    correct=None if truth is None or not compatible else bool(not abstain and truth in {predicted,conclusion.get('display') if isinstance(conclusion,dict) else predicted})
    a_correct=None if truth is None or a_label is None else a_label==truth
    return {'schema':'outcome-feedback/v1','trace_sha256':digest(trace),
        'group_id':reference['group_id'],'split':reference['split'],'label_source':reference['label_source'],
        'label_scope':label_scope,'scope_compatible':compatible,'correct':correct,'abstained':abstain,
        'a_correct':a_correct,'corrected_a':a_correct is False and correct is True,
        'broke_correct_a':a_correct is True and correct is False and not abstain,
        'failure_attribution':'unresolved_requires_tool_evidence',
        'requests':trace.get('requests',0),
        'action_feedback':[e['feedback'] for e in trace.get('events',[]) if 'feedback' in e]}


def summarize(rows):
    known=[r for r in rows if r['correct'] is not None];outputs=[r for r in known if not r['abstained']]
    return {'known_cases':len(known),'determinate':len(outputs),'correct_determinate':sum(r['correct'] for r in outputs),
        'wrong_determinate':sum(not r['correct'] for r in outputs),'abstained':sum(r['abstained'] for r in known),
        'corrected_a':sum(r['corrected_a'] for r in known),'broke_correct_a':sum(r['broke_correct_a'] for r in known),
        'accuracy_among_outputs':sum(r['correct'] for r in outputs)/len(outputs) if outputs else None,
        'correct_coverage':sum(r['correct'] for r in known)/len(known) if known else None}
