"""比较动作目标与动作前后的证据、候选判据和可用能力。形成可核验的阶段反馈，供 ReAct 上下文和离线经验评估消费。单纯文件变动不代表识别收益。

主要定义：
- objective: 按工具名给出问题的目标、预期解除的阻塞与停止条件。
- snapshot: 从评分结果提取特征、缺席观察、判据状态与候选排序，作为前后对比基线。
- evaluate: 对比前后快照并判定动作是否带来有效证据、反证或能力进展。
- reflection_card: 把失败反馈汇总成只读反思卡片，只影响任务策略，不改变证据。

关联：
- 调用方 「lib/agent/react_graph.py」、「tests/integration/test_adaptive_quality.py」、「tests/integration/test_fair_investigation.py」、「tests/offline/test_evaluation.py」。
"""
from TCM_Fusion.lib.evidence.evidence_projection import counters_criterion, supports_criterion
from TCM_Fusion.lib.runtime.storage import finite_number


def objective(action):
    tool=action.get('tool','');args=action.get('args',{})
    return {'problem_id':args.get('feature_id') or args.get('target_id') or tool,
            'objective':{'presence_check':'核验合法候选差异特征','scene_check':'完成强制场景上下文',
                'measure_instances':'获取具有用途资格的真实量化证据','segment_instance':'恢复可用目标掩码',
                'detect_region':'局部 YOLO 重检',
                'feature_scan':'按 feature_id 做全视野分块的目标条件候选扫描',
                'hybrid_scan':'并发执行 YOLO 初检与多特征 GLM 全视野扫描，并一次提交汇合结果',
                'image_exemplar_audit':'用正负示例 contact sheet 对 crop 做非阻断图片示例审计',
                'locate_feature':'用 GLM 核验选定局部并提出待验证框'}.get(tool,'读取或恢复有效证据'),
            'expected_resolution':'增加有效证据/反证或解除实际前置阻塞',
            'stop_condition':'同依赖重复动作、服务失败、预算或截止，不以生成过门为成功'}


def snapshot(scored, *, primary_ids=()):
    d = scored['decision']
    ranking = d.get('ranking', [])
    return {'features':{k:{'present':v.present,'n':v.n_instances} for k,v in scored['cards'].items()},
            'observed_absence':[(r['image_id'],r['feature_id']) for r in scored.get('observation_states',[]) if r['state']=='absent'],
            'ranking':[r.get('id') for r in ranking],
            'verdict':d.get('verdict'), 'primary_ids':list(primary_ids),
            'criterion_support':{r['id']:[c['feature_id'] for c in r.get('checks',[]) if
                supports_criterion(c)] for r in ranking},
            'criterion_counters':[f"{r['id']}:{c['feature_id']}" for r in ranking
                                  for c in r.get('checks', []) if counters_criterion(c)],
            'required':{r['id']:[c['feature_id'] for c in r.get('checks',[]) if c.get('role')=='required']
                        for r in ranking}}


def evaluate(before,after,action,*,accepted,capability_changed=False,result=None,elapsed_ms=0):
    result=result or {};a=before['features'];b=after['features']
    new=[k for k,v in b.items() if v['present'] and (not a.get(k,{}).get('present') or v['n']>a[k]['n'])]
    args = action.get('args', {})
    observed_key = (args.get('image_id'), args.get('feature_id'))
    negative = bool(accepted and action.get('tool') == 'presence_check' and result.get('present') is False
                    and finite_number(result.get('conf'), minimum=.5, maximum=1)
                    and observed_key in after.get('observed_absence', [])
                    and observed_key not in before.get('observed_absence', []))
    support_changes={cid:sorted(set(fids)-set(before.get('criterion_support',{}).get(cid,[])))
                     for cid,fids in after.get('criterion_support',{}).items()}
    support_changes={cid:fids for cid,fids in support_changes.items() if fids}
    counters = sorted(set(after.get('criterion_counters', [])) - set(before.get('criterion_counters', [])))
    if not accepted:
        support_changes={};counters=[]
    resolved = [f'{cid}:{fid}' for cid, fids in support_changes.items() for fid in fids] + counters
    useful = bool(accepted and (new or resolved or negative or capability_changed))
    primary_progress=any(cid in before.get('primary_ids',[]) and
        bool(set(fids)&set(before.get('required',{}).get(cid,[]))) for cid,fids in support_changes.items())
    primary_progress=primary_progress or any(cid+':'+fid in counters for cid in before.get('primary_ids',[])
                                           for fid in before.get('required',{}).get(cid,[]))
    feedback = {**objective(action),'accepted':bool(accepted),'new_features':new,
            'new_local_absence':[args['feature_id']] if negative else [],
            'resolved_criteria':resolved,'candidate_order_changed':before['ranking']!=after['ranking'],
            'capability_progress':bool(capability_changed),'useful':useful,
            'candidate_support_changes':support_changes,'qualified_counter_changes':counters,
            'primary_required_progress':primary_progress,
            'discrimination_progress':'qualified_counter_added' if accepted and counters else 'not_established',
            'interpretation':'支持增加/顺序变化不自动代表候选已区分；局部阴性不是标本排除',
            'latency_ms':round(elapsed_ms,2),'failure_code':result.get('error_code'),
            'continue_reason':'有效证据或能力进展' if useful else '无已验证进展；换路线或停止'}
    if result.get('scan_coverage'):
        feedback['scan_coverage'] = result['scan_coverage']
    return feedback


def reflection_card(events):
    rows=[e['feedback'] for e in events if 'feedback' in e]
    failures=[r for r in rows if not r['useful']]
    return {'schema':'task-reflection/v1','facts':failures[-3:],
            'hypothesis':'当前路线未产生已验证进展' if failures else '未发现需反思的失败',
            'avoid_problem_ids':list(dict.fromkeys(r['problem_id'] for r in failures)),
            'questions':['哪些观察支持/反驳当前假设，哪些只是unknown？',
                         '重复调查同一特征是否仍有新的可验证依据？',
                         '是否应查知识库中A候选外的假设、换视野，或停止？'],
            'next_plan_requires':'说明假设、剩余缺口、预期增益和停止依据；建议不是强制路线',
            'scope':'task_policy_only','changes_evidence':False,'uses_reference_label':False}
