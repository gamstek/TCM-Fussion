"""可从原图/部分B启动的领域运行时；物理证据仅由可信阶段提交，仍交原decide。"""
from pathlib import Path
import paths
from .evidence_runtime import EvidenceRuntime
from .workflow_session import WorkflowSession, TOOLS
from .tool_registry import ToolSpec, object_schema
from .storage import IntegrationError, project_path, file_hash, digest


class WorkflowRuntime(EvidenceRuntime):
    def __init__(self, images, *, root, workflow_config, sample_id, a=None, aliases=None,
                 allow_network=False, simulation=False, backend=None, source_run=None):
        self.workflow=None
        config_file=Path(paths.DETECTION_CONFIG_DIR)/'vlm_config.json'
        identity=file_hash(config_file) if config_file.exists() else 'unconfigured'
        super().__init__([],images,sample_id=sample_id,a=a,aliases=aliases,simulation=simulation,
                         config_identity=identity,candidate_policy=workflow_config.get('candidate_policy',{'top_k':3}))
        self.workflow=WorkflowSession(self,root,workflow_config,allow_network=allow_network,backend=backend)
        if source_run:self.workflow.seed(source_run)
        self.artifact_roots=[self.workflow.root]+([Path(source_run)] if source_run else [])
        # 供工作流判断“观察到存在的特征”与crop类别是否相左；只保存布尔结论，不保存原始观察。
        self.present_features=set()
        self.tools.register_precondition('workflow_target',lambda r,a,s:r.workflow.precondition(a))
        self.tools.register_proposer('workflow_actions',lambda r,s:r.workflow.propose())
        for tool in TOOLS:
            self.tools.register_handler(tool,lambda r,a,s:r.workflow.execute(s['action']))
            self.tools.register(ToolSpec(tool,'recompute','1',object_schema({'target_id':'string','evidence_version':'string'}),
                {'type':'object','required':['accepted','status','action_id'],
                 'properties':{'accepted':{'type':'boolean'},'status':{'enum':['pending','done','failed']},
                               'action_id':{'type':'string'}}},('workflow_repair',),'recompute',True,
                tool,'workflow_target',proposal_id='workflow_actions',
                default_allow=tool in workflow_config.get('enabled_tools',TOOLS)))

    def score(self, observations):
        if self.workflow is not None:self.records=self.workflow.store.verified_records()
        if self.workflow is not None and self.workflow.purpose_quality():
            from .purpose_evidence import project
            original=self.records;self.records=project(self)
            try:scored=super().score(observations)
            finally:self.records=original
        else:
            scored=super().score(observations)
        self.present_features={fid for fid,c in scored['cards'].items() if c.present}
        return scored

    def legal_actions(self,state):
        base=super().legal_actions(state)
        if base and base[0]['mandatory'] and self.workflow.config.get('agent_policy') != 'adaptive_v2':return base
        repair=[a for a in base if self.tools.get(a['tool']).budget_class=='recompute']
        rest=[a for a in base if self.tools.get(a['tool']).budget_class!='recompute']
        return repair+rest if self.records or self.workflow.diagnostic_quality() or self.workflow.config.get('agent_policy') in {'adaptive_v1','adaptive_v2'} else repair

    def validate_action(self,action):
        if self.tools.get(action.get('tool')).budget_class=='recompute':
            return self.tools.validate(action,self)
        return super().validate_action(action)

    def summary(self,state):
        result=super().summary(state)
        from .storage import read_json
        result['workflow']={'evidence_version':self.workflow.store.current()['version'],
                            'resources':read_json(self.workflow.budget_path)['used'],
                            'operations':list(read_json(self.workflow.operations)['entries'].values()),
                            'blocked_tools':self.workflow.blocked_tools(),
                            'unresolved':self.workflow.unresolved()}
        result['workflow']['quality_assessments']=[self.workflow.assessment(v) for v in
            self.workflow.store.current()['artifacts'].values() if v['valid'] and v['kind']=='segmentation' and v.get('assessment_id')]
        # 类别复核结论单独列出：规划器据此决定是否重分割/重检，规则分不受其直接改写。
        result['workflow']['classifications']=[{'crop_id':v.get('crop_id'),'feature_id':v.get('feature_id'),
                                                 'label':v.get('label'),
                                                 'identity_verified':v.get('identity_verified')}
            for v in self.workflow.store.current()['artifacts'].values()
            if v['valid'] and v['kind']=='classification']
        if self.workflow.config.get('targeted_recovery', False):
            result['workflow']['recovery_regions'] = [
                {'target_id': t['id'], 'image_stem': Path(t['source']).stem,
                 'box': t['roi'], 'line_score': t['line_score'], 'feature_id': t.get('feature_id')}
                for t in self.workflow.targets().values() if 'line_score' in t]
        return result


def from_task(config, a, directory):
    agent=config['agent'];images=[]
    for value in agent['image_roots']:
        p=project_path(value)
        if not p.exists():raise IntegrationError('INPUT_MISSING','指定Agent原图路径不存在')
        images.extend([p] if p.is_file() else sorted(x for x in p.rglob('*') if x.is_file() and x.suffix.lower() in {
            '.png','.jpg','.jpeg','.tif','.tiff','.bmp','.webp'}))
    if not images:raise IntegrationError('AGENT_INPUT_MISSING','工作流Agent需要绑定原图')
    if a.get('view_predictions'):
        bound = {(p.resolve(), file_hash(p)) for p in images}
        predicted = {(project_path(v['image']).resolve(), v.get('input_sha256'))
                     for v in a['view_predictions']}
        if predicted != bound or len(predicted) != len(a['view_predictions']):
            raise IntegrationError('SAMPLE_MISMATCH','多视野A结果与本次Agent原图不一致')
    if a.get('signature',{}).get('image') and a['signature']['image'] not in {file_hash(p) for p in images}:
        raise IntegrationError('SAMPLE_MISMATCH','A不属于本次Agent原图')
    source=(config['b'].get('run_dir') if config['b'].get('mode')=='existing' else
            config['b'].get('input') if config['b'].get('input_mode') in {'crops','masks'} else None)
    if config['b'].get('mode')=='recorded':raise IntegrationError('CONFIG_INVALID','真实workflow禁止录制B冒充部分工件')
    if source and not project_path(source).is_dir():
        raise IntegrationError('B_PARTIAL_MISSING','指定B部分工件目录不存在')
    options={**agent['workflow'],'candidate_policy':agent.get('candidate_policy',{'top_k':3})}
    return WorkflowRuntime(images,root=Path(directory)/'workflow',workflow_config=options,
                           sample_id=config.get('sample_id','sample'),a=a,aliases=config.get('aliases'),
                           allow_network=config.get('allow_network',False),source_run=project_path(source) if source else None)
