"""受信handler/precondition注册；Draft2020-12校验；配置不执行Python。"""
from dataclasses import dataclass, asdict
import json
from jsonschema import Draft202012Validator
from .storage import IntegrationError, digest

@dataclass(frozen=True)
class ToolSpec:
    name: str
    layer: str
    version: str
    input_schema: dict
    output_schema: dict
    skill_ids: tuple
    budget_class: str
    produces_evidence_version: bool
    handler_id: str = ''
    precondition_id: str = 'bound_sample_read'
    result_guard_id: str = 'none'
    default_allow: bool = True
    requires_human: bool = False
    proposal_id: str = ""

def object_schema(fields):
    return {'type':'object','properties':{k:{'type':v} for k,v in fields.items()},
            'required':list(fields),'additionalProperties':False}

def local_refs(schema):
    if isinstance(schema,dict):
        for k,v in schema.items():
            if k in {'$ref','$dynamicRef'} and (not isinstance(v,str) or not v.startswith('#')):
                raise IntegrationError('SCHEMA_EXTERNAL_REF','禁止外部schema引用')
            local_refs(v)
    elif isinstance(schema,list):
        for v in schema: local_refs(v)

def validate_arguments(schema,args):
    try:
        local_refs(schema); Draft202012Validator.check_schema(schema)
        Draft202012Validator(schema).validate(json.loads(json.dumps(args,allow_nan=False)))
    except IntegrationError: raise
    except Exception as exc: raise IntegrationError('ACTION_DENIED','工具schema校验失败') from exc

class ToolRegistry:
    def __init__(self):
        self.specs,self.handlers,self.preconditions,self.guards,self.proposers={},{},{},{},{}
        self.register_proposer("legacy_observations",lambda r,s:r._legacy_proposals(s))
        self.register_precondition('bound_sample_read',lambda r,a,s:bool(r.sample_id))
        self.register_precondition('legacy_observation',lambda r,a,s:r._validate_action_semantics(s['action']))
        self.register_guard('none',lambda r,a,v:True)
        self.register_guard('legacy_observation_v1',lambda r,a,v:r._validate_observation_v1(a,v))
        self.register_handler('scene_check',lambda r,a,s:s['backend'].client.scene_check(s['path']))
        def presence(r,a,s):
            fid=a['feature_id'];zh=r.kb.raw['feature_vocab'][r.kb.tier(fid)][fid]
            return s['backend'].client.presence_check(zh,a['morph_hint'],s['path'])
        self.register_handler('presence_check',presence)
        for name,fields,skill in [('scene_check',{'image_id':'string'},'scene_context'),
            ('presence_check',{'image_id':'string','feature_id':'string','morph_hint':'string'},'differential_presence')]:
            self.register(ToolSpec(name,'recompute','1',object_schema(fields),{'type':'object'},
                (skill,),'observation',True,name,'legacy_observation','legacy_observation_v1',proposal_id='legacy_observations'))
        from .tool_builtins import read_builtin
        for name,fields in [('read_evidence',{'offset':'integer','limit':'integer'}),('list_features',{}),
            ('query_criteria',{'candidate_id':'string'}),('read_scale',{}),('list_fovs',{}),('criteria_diff',{}),
            ('list_artifacts',{}),('read_artifact_status',{}),('read_quality',{})]:
            self.register_handler(name,lambda r,a,s,n=name:read_builtin(r,n,a,s))
            schema=object_schema(fields)
            if name=='read_evidence':
                schema['properties']={'offset':{'type':'integer','minimum':0},
                                      'limit':{'type':'integer','minimum':1,'maximum':50}}
            if name in {'list_artifacts','read_artifact_status'}:
                schema={'type':'object','properties':{'offset':{'type':'integer','minimum':0},'limit':{'type':'integer','minimum':1,'maximum':200}},'additionalProperties':False}
            self.register(ToolSpec(name,'read','1',schema,{'type':'object'},
                ('evidence_inspection',),'read',False,name))
    def _bind(self,table,name,fn):
        if name in table or not callable(fn):raise IntegrationError('TOOL_BINDING','重复/非法绑定')
        table[name]=fn
    def register_handler(self,n,f):self._bind(self.handlers,n,f)
    def register_precondition(self,n,f):self._bind(self.preconditions,n,f)
    def register_guard(self,n,f):self._bind(self.guards,n,f)
    def register_proposer(self,n,f):self._bind(self.proposers,n,f)
    def register(self,spec):
        if spec.name in self.specs:raise IntegrationError('TOOL_DUPLICATE','工具已注册')
        if (spec.handler_id not in self.handlers or spec.precondition_id not in self.preconditions
            or spec.result_guard_id not in self.guards or (spec.proposal_id and spec.proposal_id not in self.proposers) or spec.layer not in {'read','recompute','upstream'}):
            raise IntegrationError('TOOL_BINDING','缺少受信handler/precondition')
        for schema in (spec.input_schema,spec.output_schema):
            local_refs(schema);Draft202012Validator.check_schema(schema)
        self.specs[spec.name]=spec
    def get(self,name):
        if name not in self.specs:raise IntegrationError('ACTION_DENIED','未注册工具')
        return self.specs[name]
    def describe(self):return [asdict(s) for s in self.specs.values()]
    def signature(self):return digest(self.describe())
    def propose(self,runtime,state):
        result=[];seen=set()
        for proposal_id in dict.fromkeys(s.proposal_id for s in self.specs.values() if s.proposal_id):
            for action in self.proposers[proposal_id](runtime,state):
                spec=self.get(action.get('tool'))
                if spec.proposal_id!=proposal_id:raise IntegrationError('TOOL_BINDING','提案器越过注册归属')
                if action['action_id'] in seen or action['action_id'] in state.get('attempted',[]):continue
                try:self.validate(action,runtime,state)
                except IntegrationError as exc:
                    if exc.code in {'ACTION_DENIED','PRECONDITION_FAILED'}:continue
                    raise
                seen.add(action['action_id']);result.append(action)
        mandatory=[a for a in result if a.get('mandatory')]
        workflow = getattr(runtime, 'workflow', None)
        if workflow and workflow.config.get('agent_policy') == 'adaptive_v2':
            return result
        return mandatory or result

    def validate(self,action,runtime,state=None):
        spec=self.get(action.get('tool'));validate_arguments(spec.input_schema,action.get('args'))
        skills={s['id'] for s in runtime.skills.registry if spec.name in s['tools']}
        if action.get('skill') not in skills or action['skill'] not in spec.skill_ids:
            raise IntegrationError('ACTION_DENIED','技能不能扩权')
        if (not spec.default_allow or spec.requires_human) and spec.name not in getattr(runtime,'authorized_tools',set()):
            raise IntegrationError('ACTION_DENIED','工具未授权')
        if not self.preconditions[spec.precondition_id](runtime,action['args'],{**(state or {}),'action':action}):
            raise IntegrationError('PRECONDITION_FAILED','前置条件失败')
        return spec
    def validate_result(self,runtime,action,value):
        spec=self.get(action['tool'])
        try:validate_arguments(spec.output_schema,value)
        except IntegrationError:return False
        return bool(self.guards[spec.result_guard_id](runtime,action,value))
    def invoke(self,runtime,action,state=None):
        spec=self.validate(action,runtime,state)
        result=self.handlers[spec.handler_id](runtime,action['args'],{**(state or {}),'action':action})
        if not self.validate_result(runtime,action,result):raise IntegrationError('TOOL_OUTPUT_INVALID','输出校验失败')
        return result
    def read(self,runtime,name,args,state):
        spec=self.get(name)
        if spec.layer!='read':raise IntegrationError('ACTION_DENIED','只读入口禁止观察')
        skills=[s['id'] for s in runtime.skills.registry if name in s['tools'] and s['id'] in spec.skill_ids]
        if not skills:raise IntegrationError('ACTION_DENIED','没有受信技能可用此工具')
        return self.invoke(runtime,{'tool':name,'args':args,'skill':skills[0]},state)
