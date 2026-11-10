"""融合路由后的自主B工具入口；无需B完成清单，产物位于独立fusion任务。"""
import time
import copy
from .code_guard import CodeSignatureGuard
from .output_projection import project_output
from .storage import write_json, IntegrationError
from .tool_registry import validate_arguments
from .workflow_backend import WorkflowBackend
from .workflow_runtime import from_task
from .agent_tools import LocalToolbox
from .react_graph import run_react
from .collaboration import fuse, knowledge_base


WORKFLOW_SCHEMA={'type':'object','additionalProperties':False,'properties':{
    'python':{'type':'string'},
    'detector':{'enum':['yolo']},
    'targeted_recovery':{'type':'boolean'},
    'yolo':{'type':'object','additionalProperties':False,
            'properties':{'weights':{'type':'string','minLength':1},
                          'class_map':{'type':'string','minLength':1},
                          'detector_config':{'type':'string','minLength':1},
                          'python':{'type':'string','minLength':1},
                          'confidence':{'type':'number','exclusiveMinimum':0,'maximum':1}}},
    'execution':{'enum':['shadow','active']},
    'p0_priority':{'type':'boolean'},
    'min_support_features':{'type':'integer','enum':[1,2]},
    'final_assessment_llm':{'type':'boolean'},
    'final_assessment_seconds':{'type':'number','exclusiveMinimum':0,'maximum':120},
    'needle_audit':{'type':'boolean'},
    'feature_audit':{'type':'boolean'},
    'feature_audit_features':{'type':'array','items':{'type':'string'}},
    'image_exemplar_audit':{'type':'boolean'},
    'image_exemplar_bank':{'type':'string'},
    'image_exemplar_positive':{'type':'integer','minimum':1,'maximum':20},
    'image_exemplar_negative':{'type':'integer','minimum':1,'maximum':20},
    'image_exemplar_seconds':{'type':'number','exclusiveMinimum':0,'maximum':180},
    'needle_negative_hint':{'type':'string','maxLength':500},
    'needle_positive_hint':{'type':'string','maxLength':500},
    'identity_min_confidence':{'enum':['high','medium','low']},
    'candidate_scope':{'enum':['a_topk','open_strict']},
    'a_weight':{'type':'number','minimum':0,'maximum':1},
    'b_weight':{'type':'number','minimum':0,'maximum':1},
    'quality_policy':{'enum':['strict','diagnostic_review']},
    'box_association':{'enum':['center_point','region_component']},
    'agent_policy':{'enum':['legacy','adaptive_v1','adaptive_v2']},
    'quality_evaluation':{'enum':['legacy','purpose_v1']},
    'max_parallel':{'type':'integer','minimum':1,'maximum':2},
    'screen_top_k':{'type':'integer','minimum':1,'maximum':6},
    'screen_concurrency':{'type':'integer','minimum':1,'maximum':8},
    'needle_scan_enabled':{'type':'boolean'},
    'needle_scan_tile_size':{'type':'integer','minimum':256,'maximum':2048},
    'needle_scan_overlap':{'type':'number','minimum':0,'maximum':0.5},
    'needle_scan_max_tiles':{'type':'integer','minimum':1,'maximum':50},
    'needle_scan_concurrency':{'type':'integer','minimum':1,'maximum':8},
    'feature_scan_enabled':{'type':'boolean'},
    'feature_scan_features':{'type':'array','items':{'type':'string'}},
    'feature_scan_from_candidates':{'type':'boolean'},
    'feature_scan_max_features':{'type':'integer','minimum':1,'maximum':14},
    'feature_scan_tile_size':{'type':'integer','minimum':256,'maximum':2048},
    'feature_scan_overlap':{'type':'number','minimum':0,'maximum':0.5},
    'feature_scan_max_tiles':{'type':'integer','minimum':1,'maximum':50},
    'feature_scan_concurrency':{'type':'integer','minimum':1,'maximum':8},
    'hybrid_scan_enabled':{'type':'boolean'},
    'hybrid_scan_max_features':{'type':'integer','minimum':1,'maximum':14},
    'feature_profiles':{'type':'object'},
    'glm_identity_threshold':{'type':'number','minimum':0,'maximum':1},
    'glm_locate_threshold':{'type':'number','minimum':0,'maximum':1},
    'locate_expand':{'type':'number','minimum':1,'maximum':3},
    'enabled_tools':{'type':'array','uniqueItems':True,'items':{'enum':[
        'detect_fov_initial','calibrate_fov','segment_instance','classify_target','measure_instances','detect_region',
        'propose_regions','roi_screen','roi_screen_batch','locate_feature','needle_scan','feature_scan','hybrid_scan','image_exemplar_audit']}},
    **{k:{'type':'integer','minimum':0,'maximum':10000} for k in
       ('max_initializations','max_recompute','max_api_calls','max_planning','max_observations','max_resource_units')},
    'max_locate_attempts':{'type':'integer','minimum':0,'maximum':100},
    'locate_attempts_per_view':{'type':'integer','minimum':1,'maximum':20},
    'locate_attempt_reserve':{'type':'integer','minimum':0,'maximum':40},
    'initialization_reserve':{'type':'integer','minimum':0,'maximum':40},
    'max_tool_failures':{'type':'integer','minimum':0,'maximum':10},
    'max_seconds':{'type':'number','exclusiveMinimum':0,'maximum':86400},
    'max_observation_seconds':{'type':'number','exclusiveMinimum':0,'maximum':86400},
    'max_call_seconds':{'type':'number','exclusiveMinimum':0,'maximum':86400},
    'regions':{'type':'array','maxItems':100,'items':{'type':'object','additionalProperties':False,
        'required':['image_stem','box'],'properties':{'image_stem':{'type':'string','minLength':1},
        'box':{'type':'array','minItems':4,'maxItems':4,'items':{'type':'integer','minimum':0}}}}}}}


async def execute_workflow(config,a,directory,*,routing,start,job,status_path,code_baseline=None):
    workflow = config['agent']['workflow']
    if 'agent_policy' not in workflow and config['agent'].get('planner', 'llm') == 'llm':
        from copy import deepcopy
        config = deepcopy(config)
        config['agent']['workflow']['agent_policy'] = 'adaptive_v2'
    validate_arguments(WORKFLOW_SCHEMA,config['agent']['workflow'])
    from .runtime_readiness import planner_preflight
    preflight=planner_preflight(config)
    write_json(directory/'agent_preflight.json',preflight)
    if not preflight['ready']:
        job.update(stage='agent_preflight');write_json(status_path,job)
        failed=[c for c in preflight['checks'] if not c['ok']]
        raise IntegrationError(failed[0]['code'],'真实Agent启动条件不满足：'+'；'.join(c['detail'] for c in failed))
    runtime=from_task(config,a,directory);agent=config['agent']
    guard=CodeSignatureGuard(directory/'code_signature.json',WorkflowBackend.code_signature,
                             baseline=code_baseline if code_baseline is not None else runtime.workflow.code_signature)
    guard.check('workflow_start')
    write_json(directory/'b_snapshot.json',{'verdict':runtime.original,'provenance':'initial_rule_state_not_completed_b'})
    if agent['workflow'].get('execution')=='shadow':
        guard.check('shadow_start')
        actions=runtime.legal_actions({'observations':[],'attempted':[]})
        value={'schema':'ab-fusion/v2','route':'workflow_shadow','sample_id':runtime.sample_id,
               'simulation':False,'real_model_validation':None,'needs_review':True,
               'a':a,'routing':routing,'b':None,'b_executed':False,
               'agent':{'executed':False,'actions':actions,'context':runtime.summary({'observations':[]}),
                        'new_image_api_calls':0},'limitations':['仅计划，不执行B阶段或视觉补看；不能作为识别结果']}
        write_json(directory/'fusion.json',value)
        (directory/'report.md').write_text('# 工作流shadow计划\n\n只查询现有证据与合法动作；未执行视觉工具。\n',encoding='utf-8')
        guard.finish()
        job.update(status='completed',stage='done',needs_review=True,route='workflow_shadow')
        write_json(status_path,job);return directory
    options={'max_requests':agent.get('judgement_request_budget',6),'max_steps':agent.get('max_steps',6),
             'planner_mode':agent.get('planner','llm'),'allow_network':config.get('allow_network',False),
             'trace_path':directory/'agent_trace.json'}
    tool_request_limit = (min(100, runtime.workflow.limits["observation"])
                          if agent["workflow"].get("agent_policy") == "adaptive_v2"
                          else options["max_requests"])
    if agent.get('prompt_release'):options['prompt_release']=agent['prompt_release']
    job.update(stage='workflow_agent');write_json(status_path,job)
    guard.start(runtime.workflow.cancel_event)
    try:
        if agent.get('transport','local')=='mcp':
            from .mcp_bridge import connect_stdio
            context=directory/'mcp_context.json'
            write_json(context,{'workflow_config':config,'a_prediction':str(directory/'a_prediction.json'),
                                'directory':str(directory),'allow_network':config.get('allow_network',False),
                                'max_requests':tool_request_limit,'simulation':False})
            async with connect_stdio(context) as toolbox:result=await run_react(runtime,toolbox,**options)
        else:
            toolbox=LocalToolbox(runtime,ledger_path=directory/'agent_tool_ledger.json',
                                max_requests=tool_request_limit,allow_network=config.get('allow_network',False))
            result=await run_react(runtime,toolbox,**options)
        guard.check('agent_returned')
    finally:
        guard.stop()
    from .storage import read_json
    runtime.score(result['observations'])
    if workflow.get('feature_audit', False) or workflow.get('needle_audit', False):
        try:
            from .needle_observation import build_feature_audit
            feature_ids = workflow.get('feature_audit_features') or ['needle_crystal']
            result['feature_audit']=build_feature_audit(directory, feature_ids=feature_ids)
            if 'needle_crystal' in feature_ids:
                result['needle_audit']=result['feature_audit']
        except Exception as exc:
            # 旁路审计失败不得改变原裁决或关闭任务。
            result['feature_audit']={'schema':'feature-audit/v1','non_blocking':True,
                                     'error':f'{type(exc).__name__}: {exc}'}
            result['needle_audit']=result['feature_audit']
    guard.check('before_fusion')
    if runtime.workflow.purpose_quality():
        from .purpose_evidence import project
        write_json(directory/'accepted_evidence.json',{'records':project(runtime),'quality_policy':'purpose_v1',
            'risk_validated':False,'original_records_unchanged':True})
        write_json(directory/'quality_report.json',runtime.summary(result)['workflow']['quality_assessments'])
    diagnostic=runtime.workflow.diagnostic_quality()
    identity_review=any(v.get('identity_review') for v in runtime.workflow.store.current()['artifacts'].values()
                        if v['valid'] and v['kind']=='classification')
    result.update(executed=True,needs_review=diagnostic or result['needs_review'] or runtime.workflow.unresolved() or identity_review,
                  diagnostic_only=diagnostic,
                  identity_review=identity_review,
                  evidence_version=runtime.workflow.store.current()['version'],
                  resources=read_json(runtime.workflow.budget_path),
                  blocked_tools=runtime.workflow.blocked_tools(),
                  operations=read_json(runtime.workflow.operations)['entries'])
    import contracts
    verdict={'sample_id':runtime.sample_id,**copy.deepcopy(result['decision']),'needs_review':result['needs_review'],
             'source_images':[str(v['path']) for v in runtime.images.values()],
             'agent':{'mode':'workflow_react','vlm_calls':result['observation_requests'],
                      'planning_calls':result['planning_requests'],'identity_review':identity_review}}
    contracts.Verdict.from_obj(verdict)
    from .candidate_scope import apply_candidate_scope
    verdict=apply_candidate_scope(verdict,a,runtime.kb,config.get('aliases'),
                                  runtime.candidate_policy,
                                  runtime.workflow.config.get('candidate_scope','open_strict'))
    from .ab_fusion import apply_ab_fusion
    verdict=apply_ab_fusion(verdict,a,runtime.kb,config.get('aliases'),
                            runtime.workflow.config.get('a_weight',0.5),
                            runtime.workflow.config.get('b_weight',0.5))
    admission=runtime.admission_decision(verdict)
    result['candidate_admission']=admission
    if admission.get('decision') in {'verify','suppress'}:
        result['needs_review']=True
    result['needs_review'] = bool(result['needs_review'] or verdict.get('needs_review'))
    verdict['needs_review']=result['needs_review']
    verdict['candidate_admission']=admission
    from .final_assessment import finalize
    assessment=finalize(runtime,verdict,result,directory)
    guard.check('final_assessment_returned')
    result['final_assessment']=assessment
    result['needs_review']=bool(result['needs_review'] or assessment['review_required'])
    verdict['needs_review']=result['needs_review']
    result['events'].append({'phase':'final_assessment','decision_source':assessment['method'],
        'stop_reason':result['stop_reason'],'candidate_id':assessment['candidate_id'],
        'model_failure':assessment['model_failure'],'usage':assessment['usage'],
        'new_image_calls':0,'new_tool_calls':0})
    runtime.workflow.close(result['stop_reason'])
    result['resources']=read_json(runtime.workflow.budget_path)
    write_json(directory/'agent_trace.json',result)
    write_json(directory/'agent_verdict.json',verdict)
    fusion=fuse(a,verdict,knowledge_base(),config.get('aliases'));fusion['needs_review']=result['needs_review'] or fusion['needs_review'];fusion['candidate_admission']=admission
    output=project_output(runtime.sample_id,verdict,result,a)
    write_json(directory/'output.json',output)
    value={'schema':'ab-fusion/v2','route':'workflow_agent','simulation':False,'real_model_validation':None,
           'sample_id':runtime.sample_id,'routing':routing,'a':a,'diagnostic_only':diagnostic,
           'quality_risk_validated':False if runtime.workflow.purpose_quality() else None,
           'workflow_options':runtime.workflow.config,'aliases':config.get('aliases',{}),
           'workflow_source':str(config['b'].get('run_dir') or config['b'].get('input')) if config['b'].get('mode')=='existing' or config['b'].get('input_mode') in {'crops','masks'} else None,
           'b':{'provenance':'workflow_tools','verdict':verdict},'fusion':fusion,'agent':result,
           'needs_review':fusion['needs_review'],'agreement':fusion['agreement'],
           'b_executed':bool(result['operations']),'elapsed_seconds':time.perf_counter()-start,
           'limitations':['真实API及风险阈值需独立数据验收；资源计量为调用/任务次数，不等于费用']}
    write_json(directory/'fusion.json',value)
    comparison='\n'.join(
        f"| {row['display']} | {row['rule_confidence']} | {row['fused_score']} | "
        f"{len(row['supported_criteria'])} | {len(row['required_gaps'])} |"
        for row in assessment['candidate_comparison'])
    (directory/'report.md').write_text('# 自主B工具与A融合\n\n'
        f"- 裁决：{verdict['verdict']}\n- 交付输出：{output['status']} / "
        f"{(output['top_candidate'] or {}).get('display', '无')}\n"
        f"- 内部复核标记：{fusion['needs_review']}；交付层 review_required={output['review_required']}\n"
        f"- 强观察：{len(output['strong_observations'])} 条；备选：{len(output['alternatives'])} 个\n"
        f"- 停止原因：{result['stop_reason']}\n- 证据版本：{result['evidence_version']}\n"
        '- 原图及原B目录未覆盖；阶段失败/资源明细见agent_trace.json。\n\n'
        '## 收尾综合研判\n\n'
        f"{assessment['judgement']}\n\n{assessment['explanation']}\n\n"
        f"- 研判来源：{assessment['method']}；模型失败原因：{assessment['model_failure']}\n"
        f"- 支持证据：{[f['id'] for f in assessment['supporting_evidence']]}\n"
        f"- 反证：{[f['id'] for f in assessment['counterevidence']]}\n"
        f"- 未解决：{[f['id'] for f in assessment['unresolved']]}\n"
        '- 全部证据及采信边界：final_evidence.json；结论与依据：final_assessment.json。\n\n'
        '| 候选 | B规则分 | 融合排序分（非正确率） | 支持判据数 | 必需判据缺口数 |\n'
        '|---|---|---|---|---|\n'+comparison+'\n',encoding='utf-8')
    job.update(status='completed',stage='done',needs_review=fusion['needs_review'],route='workflow_agent',
               elapsed_seconds=value['elapsed_seconds'],output=str(directory/'output.json'),
               code_signature=guard.snapshot());write_json(status_path,job)
    guard.finish()
    return directory
