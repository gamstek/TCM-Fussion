"""ReAct可核验反馈：目标→实际证据/能力变化；不将哈希或规则分数当准确率。"""
def objective(action):
    tool=action.get('tool','');args=action.get('args',{})
    return {'problem_id':args.get('feature_id') or args.get('target_id') or tool,
            'objective':{'presence_check':'核验合法候选差异特征','scene_check':'完成强制场景上下文',
                'measure_instances':'获取具有用途资格的真实量化证据','segment_instance':'恢复可用目标掩码',
                'detect_region':'局部 YOLO 重检',
                'propose_regions':'从原图纹理提议局部核验区域',
                'roi_screen':'用 GLM 多图筛选 ROI，决定哪些局部值得继续框选',
                'roi_screen_batch':'并发执行多个视野的 ROI 筛选',
                'needle_scan':'全视野分块、目标条件的针晶候选扫描；输出 crop 供后续身份/测量',
                'feature_scan':'按 feature_id 做全视野分块的目标条件候选扫描',
                'locate_feature':'用 GLM 核验选定局部并提出待验证框'}.get(tool,'读取或恢复有效证据'),
            'expected_resolution':'增加有效证据/反证或解除实际前置阻塞',
            'stop_condition':'同依赖重复动作、服务失败、预算或截止，不以生成过门为成功'}


def snapshot(scored):
    d=scored['decision'];checks={}
    for r in d.get('ranking',[]):
        for c in r.get('checks',[]):
            checks[str(r.get('id'))+':'+str(c.get('feature_id'))]=c.get('state')
    return {'features':{k:{'present':v.present,'n':v.n_instances} for k,v in scored['cards'].items()},
            'observed_absence':[(r['image_id'],r['feature_id']) for r in scored.get('observation_states',[]) if r['state']=='absent'],
            'checks':checks,'ranking':[r.get('id') for r in d.get('ranking',[])],
            'verdict':d.get('verdict')}


def evaluate(before,after,action,*,accepted,capability_changed=False,result=None,elapsed_ms=0):
    result=result or {};a=before['features'];b=after['features']
    new=[k for k,v in b.items() if v['present'] and (not a.get(k,{}).get('present') or v['n']>a[k]['n'])]
    resolved=[k for k,v in after['checks'].items() if before['checks'].get(k) in {'NOT_DETECTED','UNCAL'} and v not in {'NOT_DETECTED','UNCAL'}]
    negative=bool(accepted and action.get('tool')=='presence_check' and result.get('present') is False and result.get('conf',0)>=0.5)
    if (action.get('args',{}).get('image_id'),action.get('args',{}).get('feature_id')) in before.get('observed_absence',[]):negative=False
    # 有效负观察是Observation进展，不伪装为规则已支持的阳性证据。
    useful=bool(accepted and (new or resolved or negative or capability_changed))
    return {**objective(action),'accepted':bool(accepted),'new_features':new,
            'new_counterevidence':[action.get('args',{}).get('feature_id')] if negative else [],
            'resolved_criteria':resolved,'candidate_order_changed':before['ranking']!=after['ranking'],
            'capability_progress':bool(capability_changed),'useful':useful,
            'latency_ms':round(elapsed_ms,2),'failure_code':result.get('error_code'),
            'continue_reason':'有效证据或能力进展' if useful else '无已验证进展；换路线或停止'}


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
