"""重算闭环的封闭回放：复用录制的物理工件与视觉观察，重跑规划链做对照评估。

回放不发起新的看图请求；未知动作或依赖指纹不唯一的录制记录一律拒绝（REPLAY_MISS），
录制产物越界、缺失或哈希变化同样失败，因此回放结果不代表独立真值或策略发布。

主要定义：
- RecordedStages。
- RecordedObservations。
- replay_report。不发新图像请求；报告对照不代替独立识别真值或策略发布。

关联：
- 调用方 「cli/offline.py」、「lib/offline/workflow_prompt_validation.py」、「tests/workflow/test_workflow.py」。
- 直接依赖 「lib/agent/prompt_release.py」、「lib/agent/react_graph.py」、「lib/runtime/storage.py」、「lib/tools/agent_tools.py」、「lib/workflow/workflow_runtime.py」。
"""
import shutil
from pathlib import Path
from TCM_Fusion.lib.runtime.storage import IntegrationError, file_hash, read_json


class RecordedStages:
    def __init__(self,operations):self.operations=list(operations);self.misses=0;self.calls=0
    def execute(self,session,tool,target,work,context):
        self.calls+=1
        fingerprint=session.dependency_fingerprint(tool,target)
        matches=[r for r in self.operations if r.get('tool')==tool and r.get('target_id')==target['id']
                 and r.get('dependency_fingerprint')==fingerprint and r.get('status')=='done' and r.get('bundle')]
        if len(matches)!=1:
            self.misses+=1;raise IntegrationError('REPLAY_MISS','没有唯一录制动作/依赖结果')
        record=matches[0]['bundle'];root=Path(record['root']).resolve();copy=Path(work)/'recorded';copy.mkdir()
        for name,sha in record['files'].items():
            source=(root/name).resolve();dest=(copy/name).resolve()
            if not source.is_relative_to(root) or not dest.is_relative_to(copy) or file_hash(source)!=sha:
                raise IntegrationError('REPLAY_ARTIFACT','录制产物越界、缺失或变更')
            dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest)
        return {**{k:str(copy/v) for k,v in record['paths'].items()},'quality':record.get('quality')}


class RecordedObservations:
    def __init__(self,rows):self.rows=rows;self.misses=0
    def call(self,runtime,action,path):
        matches=[r for r in self.rows if r['action']['action_id']==action['action_id']]
        if len(matches)!=1:
            self.misses+=1;raise IntegrationError('REPLAY_MISS','没有唯一录制视觉观察')
        return matches[0]['result'],matches[0].get('usage')


async def replay_report(report,directory,planner=None,planning_instruction=None):
    """不发新图像请求；报告对照不代替独立识别真值或策略发布。"""
    from TCM_Fusion.lib.workflow.workflow_runtime import WorkflowRuntime
    from TCM_Fusion.lib.tools.agent_tools import LocalToolbox
    from TCM_Fusion.lib.agent.react_graph import run_react
    from TCM_Fusion.lib.runtime.storage import write_json
    if report.get('route')!='workflow_agent':raise IntegrationError('REPLAY_SCOPE','需要工作流Agent报告')
    directory=Path(directory);stages=RecordedStages(report['agent']['operations'].values())
    observations=RecordedObservations(report['agent']['observations'])
    runtime=WorkflowRuntime(report['b']['verdict']['source_images'],root=directory/'workflow',
        workflow_config=report['workflow_options'],sample_id=report['sample_id'],a=report['a'],
        aliases=report.get('aliases'),simulation=True,backend=stages,source_run=report.get('workflow_source'))
    toolbox=LocalToolbox(runtime,ledger_path=directory/'observations.json',backend=observations,max_requests=100)
    options={'planning_instruction':planning_instruction} if planning_instruction is not None else {}
    trace=await run_react(runtime,toolbox,planner_mode='scripted' if planner else 'heuristic',planner=planner,
                          max_requests=100,max_steps=100,trace_path=directory/'trace.json',**options)
    decision=trace['decision'];source=report['b']['verdict']
    from TCM_Fusion.lib.agent.prompt_release import runtime_context
    physical=[]
    import contracts
    for key,v in sorted(runtime.workflow.store.current()['artifacts'].items()):
        if not v['valid']:continue
        if v['kind']=='crop':
            m=contracts.CropMeta.from_json(v['path'])
            physical.append((key,m.feature_id,m.px_per_um,m.tight_box,m.loose_box,file_hash(m.png)))
        elif v['kind']=='scale':physical.append((key,v['px_per_um']))
        elif v['kind']=='quality_assessment':
            qa=contracts.quality_assessment(read_json(v['path']))
            physical.append((key,qa['bindings'],qa['uses'],qa['policy_version']))
        elif v['kind']=='segmentation_file' and v['path'].endswith('.png'):
            physical.append((key,v['sha256']))
    from TCM_Fusion.lib.runtime.storage import digest
    value={'simulation':True,'real_accuracy_validated':False,'new_image_api_calls':0,
           'runtime_context':runtime_context(runtime),'physical_evidence':digest(physical),
           'evidence_digest':runtime.score(trace['observations'])['evidence_digest'],
           'signature':{'verdict':decision['verdict'],'conclusion_id':(decision.get('conclusion') or {}).get('id')},
           'needs_review':trace['needs_review'] or runtime.workflow.unresolved(),
           'mandatory_complete':trace['mandatory_complete'],'degraded':trace['planner_degraded'],
           'requests':trace['requests'],'stage_attempts':stages.calls,
           'unknown_stage_actions':stages.misses,'unknown_observation_actions':observations.misses,
           'matches_source_verdict':(decision['verdict'],decision.get('conclusion'))==
                                   (source['verdict'],source.get('conclusion')),
           'trace':str(directory/'trace.json')}
    value['covered']=not stages.misses and not observations.misses
    write_json(directory/'report.json',value);return value
