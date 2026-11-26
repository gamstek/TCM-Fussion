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
    def __init__(self, operations):
        self.operations = list(operations)
        self.misses = 0
        self.calls = 0

    def execute(self, session, tool, target, work, context):
        self.calls += 1
        fingerprint = session.dependency_fingerprint(tool, target)
        matches = [
            row for row in self.operations
            if row.get('tool') == tool and row.get('target_id') == target['id']
            and row.get('visual_brief') == target.get('visual_brief')
            and row.get('observation_basis') == target.get('observation_basis')
            and row.get('dependency_fingerprint') == fingerprint
            and row.get('status') == 'done' and row.get('bundle')
        ]
        try:
            if len(matches) != 1:
                raise IntegrationError('REPLAY_MISS', '没有唯一录制动作/依赖结果')
            return self._restore(matches[0]['bundle'], tool, work)
        except IntegrationError:
            self.misses += 1
            raise
        except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
            self.misses += 1
            raise IntegrationError('REPLAY_ARTIFACT', '录制产物不可读或格式错误') from exc

    @staticmethod
    def _restore(record, tool, work):
        root = Path(record['root']).resolve()
        copy = (Path(work) / 'recorded').resolve()
        paths = record['paths']
        files = record['files']

        def within(base, name):
            rel = Path(name)
            path = (base / rel).resolve()
            if rel.is_absolute() or '..' in rel.parts or not path.is_relative_to(base):
                raise IntegrationError('REPLAY_ARTIFACT', '录制产物越界')
            return path

        if tool == 'classify_crops_batch' and not {'classifications', 'failures'} <= record.keys():
            raise IntegrationError('REPLAY_MISS', '旧录制缺少逐裁图分类结果')
        if tool == 'image_exemplar_audit' and not {'audit', 'contact_sheet'} <= paths.keys():
            raise IntegrationError('REPLAY_MISS', '旧录制缺少示例审计或拼图')

        restored = {key: str(within(copy, name)) for key, name in paths.items()}
        restored['quality'] = record.get('quality')
        for key in ('audit', 'contact_sheet', 'visual_observation_file'):
            if key in paths and Path(paths[key]).as_posix() not in files:
                raise IntegrationError('REPLAY_ARTIFACT', '阶段文件缺少录制摘要')
        if 'classifications' in record:
            restored['classifications'] = []
            for row in record['classifications']:
                directory = within(copy, row['classification'])
                relative = (Path(row['classification']) / 'classification.json').as_posix()
                if relative not in files:
                    raise IntegrationError('REPLAY_ARTIFACT', '分类文件缺少录制摘要')
                restored['classifications'].append({**row, 'classification': str(directory)})
            restored['failures'] = [
                {key: row[key] for key in ('crop_id', 'status', 'error_code')}
                for row in record['failures']
            ]

        transfers = []
        for name, sha in files.items():
            source = within(root, name)
            destination = within(copy, name)
            if file_hash(source) != sha:
                raise IntegrationError('REPLAY_ARTIFACT', '录制工件内容已改变')
            transfers.append((source, destination, sha))
        copy.mkdir()
        for source, destination, sha in transfers:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            if file_hash(destination) != sha:
                raise IntegrationError('REPLAY_ARTIFACT', '复制期间录制工件内容已改变')
        return restored


class RecordedObservations:
    def __init__(self,rows):self.rows=rows;self.misses=0
    def call(self,runtime,action,path):
        matches=[r for r in self.rows if r['action']['action_id']==action['action_id']
                 and r['action'].get('visual_brief') == action.get('visual_brief')]
        if len(matches)!=1:
            self.misses+=1;raise IntegrationError('REPLAY_MISS','没有唯一录制视觉观察')
        row = matches[0]
        value = row['result']
        if row.get('accepted') and row.get('visual_observation'):
            value = {**value, 'visual_observation': row['visual_observation']}
        return value, row.get('usage')


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
