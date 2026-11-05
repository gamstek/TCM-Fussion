"""融合路由后的自主B工具入口；无需B完成清单，产物位于独立fusion任务。"""
import time
from .storage import write_json, IntegrationError
from .tool_registry import validate_arguments
from .workflow_runtime import from_task
from .agent_tools import LocalToolbox
from .react_graph import run_react
from .collaboration import fuse, knowledge_base


WORKFLOW_SCHEMA={'type':'object','additionalProperties':False,'properties':{
    'python':{'type':'string'},
    'execution':{'enum':['shadow','active']},
    'quality_policy':{'enum':['strict','diagnostic_review']},
    'box_association':{'enum':['center_point','region_component']},
    'agent_policy':{'enum':['legacy','adaptive_v1','adaptive_v2']},
    'quality_evaluation':{'enum':['legacy','purpose_v1']},
    'max_parallel':{'type':'integer','minimum':1,'maximum':2},
    'enabled_tools':{'type':'array','uniqueItems':True,'items':{'enum':[
        'detect_fov_initial','calibrate_fov','segment_instance','classify_target','measure_instances','detect_region']}},
    **{k:{'type':'integer','minimum':0,'maximum':10000} for k in
       ('max_initializations','max_recompute','max_api_calls','max_planning','max_observations','max_resource_units')},
    'max_seconds':{'type':'number','exclusiveMinimum':0,'maximum':86400},
    'max_observation_seconds':{'type':'number','exclusiveMinimum':0,'maximum':86400},
    'max_call_seconds':{'type':'number','exclusiveMinimum':0,'maximum':86400},
    'regions':{'type':'array','maxItems':100,'items':{'type':'object','additionalProperties':False,
        'required':['image_stem','box'],'properties':{'image_stem':{'type':'string','minLength':1},
        'box':{'type':'array','minItems':4,'maxItems':4,'items':{'type':'integer','minimum':0}}}}}}}


async def execute_workflow(config,a,directory,*,routing,start,job,status_path):
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
    write_json(directory/'b_snapshot.json',{'verdict':runtime.original,'provenance':'initial_rule_state_not_completed_b'})
    if agent['workflow'].get('execution')=='shadow':
        actions=runtime.legal_actions({'observations':[],'attempted':[]})
        value={'schema':'ab-fusion/v2','route':'workflow_shadow','sample_id':runtime.sample_id,
               'simulation':False,'real_model_validation':None,'needs_review':True,
               'a':a,'routing':routing,'b':None,'b_executed':False,
               'agent':{'executed':False,'actions':actions,'context':runtime.summary({'observations':[]}),
                        'new_image_api_calls':0},'limitations':['仅计划，不执行B阶段或视觉补看；不能作为识别结果']}
        write_json(directory/'fusion.json',value)
        (directory/'report.md').write_text('# 工作流shadow计划\n\n只查询现有证据与合法动作；未执行视觉工具。\n',encoding='utf-8')
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
    from .storage import read_json
    runtime.workflow.close(result['stop_reason'])
    runtime.score(result['observations'])
    if runtime.workflow.purpose_quality():
        from .purpose_evidence import project
        write_json(directory/'accepted_evidence.json',{'records':project(runtime),'quality_policy':'purpose_v1',
            'risk_validated':False,'original_records_unchanged':True})
        write_json(directory/'quality_report.json',runtime.summary(result)['workflow']['quality_assessments'])
    diagnostic=runtime.workflow.diagnostic_quality()
    result.update(executed=True,needs_review=diagnostic or result['needs_review'] or runtime.workflow.unresolved(),
                  diagnostic_only=diagnostic,
                  evidence_version=runtime.workflow.store.current()['version'],
                  resources=read_json(runtime.workflow.budget_path),
                  blocked_tools=runtime.workflow.blocked_tools(),
                  operations=read_json(runtime.workflow.operations)['entries'])
    import contracts
    verdict={'sample_id':runtime.sample_id,**result['decision'],'needs_review':result['needs_review'],
             'source_images':[str(v['path']) for v in runtime.images.values()],
             'agent':{'mode':'workflow_react','vlm_calls':result['observation_requests'],
                      'planning_calls':result['planning_requests']}}
    contracts.Verdict.from_obj(verdict);write_json(directory/'agent_verdict.json',verdict)
    fusion=fuse(a,verdict,knowledge_base(),config.get('aliases'));fusion['needs_review']=result['needs_review'] or fusion['needs_review']
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
    (directory/'report.md').write_text('# 自主B工具与A融合\n\n'
        f"- 裁决：{verdict['verdict']}\n- 需复核：{fusion['needs_review']}\n"
        f"- 停止原因：{result['stop_reason']}\n- 证据版本：{result['evidence_version']}\n"
        '- 原图及原B目录未覆盖；阶段失败/资源明细见agent_trace.json。\n',encoding='utf-8')
    job.update(status='completed',stage='done',needs_review=fusion['needs_review'],route='workflow_agent',
               elapsed_seconds=value['elapsed_seconds']);write_json(status_path,job)
    return directory
