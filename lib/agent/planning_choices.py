"""ReAct执行选择视图：单动作、实例目标和独立观察组使用同一主选择接口。

消费当前可见合法动作/KB/预算；只描述选择，不执行工具或授予证据资格。
react_graph最终仍逐项校验并晚绑定。model_view仅作短编号映射，供真实LLM和离线请求对拍；
不枚举组合、不强制并行，不产生未来工具序列。完整菜单审计不保存思维链或图像载荷。

主要定义：
- attach。给当前实际可见动作建立目标/问题入口；不把菜单外动作偷偷加进来。
- decision_fields / decision_instruction。按当轮菜单给出可用的决策格式，正常规划与格式修正共用。
- model_view。短编号只在一次请求内有效；返回送模副本及短编号反解，保持原上下文不变。
- decode。只展开模型明确选择的模式；不按优先级代选、不补第二个问题。
- diagnose。错误只留数量、已知命名空间与有限短编号；不保存无效自由值。
- audit。保存真实送模菜单而非全部合法集合；供下一次排查可见性。

关联：
- 调用方 「lib/agent/react_graph.py」、「tests/agent/test_planning_choices.py」、「tests/agent/test_reflection_strategy.py」、「tests/integration/test_fair_investigation.py」。
- 直接依赖 「lib/agent/semantic_execution.py」、「lib/decision/comparison_problems.py」、「lib/evidence/purpose_evidence.py」、「lib/runtime/storage.py」。
"""
import copy
import re

from TCM_Fusion.lib.decision.comparison_problems import _action_feature, action_question
from TCM_Fusion.lib.runtime.storage import IntegrationError, digest


def action_state(runtime, actions, evidence, targets):
    """投影实际菜单里可直接执行的步骤和同视野记录；不从观察文本推断状态。"""
    presence = {}
    for row in evidence.get('observation_states', []):
        key = (row['image_id'], row['feature_id'])
        presence.setdefault(key, set()).add(row['state'])
    rows = {}
    for action in actions:
        tool = action['tool']
        args = action.get('args') or {}
        target = targets.get(args.get('target_id'), {})
        image = args.get('image_id') or target.get('image_id')
        if image not in runtime.images or runtime.tools.get(tool).layer == 'read':
            continue
        fid = _action_feature(action, targets)
        row = {'image_id': image, 'entry_kind': 'discovery' if tool == 'detect_fov_initial'
               else 'current_step'}
        if fid:
            row.update(feature_id=fid, presence_states=sorted(presence.get((image, fid), ())))
        rows[action['action_id']] = row
    return rows


def add_crop_reuse_hints(rows, actions, targets):
    """给同视野同特征的补扫关联已有裁图入口；不重排、过滤或自动执行工具。"""
    reuse = {}
    for action in actions:
        tool = action['tool']
        row = rows.get(action['action_id'])
        if row is None or tool not in {'classify_target', 'classify_crops_batch',
                                       'segment_instance', 'measure_instances'}:
            continue
        target = targets.get(action['args'].get('target_id'), {})
        crops = (target.get('crops', []) if tool == 'classify_crops_batch' else
                 [targets.get(target.get('crop_id'), target)])
        for crop in crops:
            if (crop.get('kind') != 'crop' or crop.get('box_source') != 'yolo'
                    or crop.get('box_quality') not in {'usable', 'low_detail', 'tiny'}
                    or not crop.get('valid')):
                continue
            fid = _action_feature({'args': {'target_id': crop['id']}}, targets)
            if fid:
                entries = reuse.setdefault((crop['image_id'], fid), {})
                entries[action['action_id']] = None
    for action in actions:
        row = rows.get(action['action_id'])
        if row is None or action['tool'] != 'feature_scan':
            continue
        entries = reuse.get((row['image_id'], row.get('feature_id')))
        if entries:
            row['existing_crop_actions'] = list(entries)[:3]
            scan = targets.get(action['args']['target_id'], {})
            row['max_scan_tiles'] = scan.get('max_tiles')


def attach(runtime, context, *, can_pair, targets=None):
    """给当前实际可见动作建立目标/问题入口；不把菜单外动作偷偷加进来。"""
    out = dict(context)
    actions = context['actions']
    targets = runtime.workflow.targets() if targets is None else targets
    out['action_state'] = action_state(runtime, actions + context.get('parallel_observations', []),
                                       context.get('evidence') or {}, targets)
    add_crop_reuse_hints(out['action_state'], actions, targets)
    active = set(runtime.candidate_interest()['ids'])
    criteria = [dict(c, candidate_id=r['id']) for r in runtime.kb.candidates()
                if r['id'] in active for c in r['criteria']]
    questions = []
    if can_pair:
        for a in actions:
            if a['tool'] not in {'presence_check', 'scene_check'}:
                continue
            q = action_question(runtime, a, targets=targets)
            consumers = sorted({c['candidate_id'] for c in criteria if c['feature'] == q['feature_id']})
            questions.append({**q, 'action_id':a['action_id'], 'tool':a['tool'],
                'shared_consumers':consumers,
                'limits':'共同存在性不能区分这些消费者的形态/尺寸；未见仅限当前视野'
                    if consumers else '场景上下文不单独确认药材'})
    goals = []
    if runtime.workflow.config.get('semantic_execution', False):
        from TCM_Fusion.lib.agent.semantic_execution import options
        from TCM_Fusion.lib.evidence.purpose_evidence import dimension_use
        for row in options(runtime, actions):
            basis = [c for c in criteria if c['feature'] == row['feature_id'] and c.get('quant')]
            dims = {dimension_use(c['quant']['dim']) for c in basis if c['quant'].get('dim')}
            for entry in row['entry_actions']:
                tasks = (['verify_identity'] if entry['tool'] == 'classify_target' else [])
                tasks += ['measure_'+dim for dim in sorted(dims)]
                for task in tasks:
                    goal = {'semantic_task':task, 'instance_ref':row['instance_ref'],
                            'problem_id':'instance-question:'+digest([row['instance_ref'],task])[:20]}
                    goals.append({'goal_id':'goal:'+digest([goal,entry['action_id']])[:24],
                        'entry_action_id':entry['action_id'], 'feature_id':row['feature_id'],
                        'goal':goal,
                        'criterion_consumers':sorted({c['candidate_id'] for c in basis}),
                        'result':'仅身份，不生成mask/尺寸' if task == 'verify_identity' else
                                 '对应量纲合格数值；身份/QA/标定失败立即回交ReAct',
                        'planning_saved_upper_bound':0 if task == 'verify_identity' else
                            {'classify_target':2-int(entry['automatic_measurement']),
                             'segment_instance':1-int(entry['automatic_measurement']),
                             'measure_instances':0}[entry['tool']],
                        'cost_note':'上界已扣既有自动测量；不减少必需视觉调用，失败回交，不保证节省'})
    out['execution_options'] = {'goals':goals[:12], 'questions':questions,
        'max_questions':2 if can_pair else 1,
        'choice_note':'仅从本轮菜单选择；观察结果会改变下一问题时选单步；review仍可选'}
    return out


def decision_fields(context):
    """列出当前菜单可表达的主选择和字段，用于正常规划与格式修正。"""
    offers = context.get('execution_options') or {}
    fields = {'action': ['kind', 'action_id']}
    if offers.get('goals'):
        fields['instance_goal'] = ['kind', 'goal_id']
    if offers.get('questions') and offers.get('max_questions', 1):
        fields['observe_questions'] = ['kind', 'question_ids']
    if offers.get('physical_groups', offers.get('physical_pairs')):
        fields['physical_group'] = ['kind', 'action_ids']
    return fields


def decision_instruction(context):
    """展示实际有入口的决策格式。具体编号见同一份菜单。"""
    modes = decision_fields(context)
    offers = context.get('execution_options') or {}
    lines = [
        '只返回一个JSON对象，按证据选择本轮可用格式，顶层decision_summary说明为何值得做或停止。',
        '单步 {"decision":{"kind":"action","action_id":"A编号"}} 从 actions 选择一个动作，补扫和局部重检也从这里选。',
    ]
    if 'instance_goal' in modes:
        lines.append(
            '语义目标 {"decision":{"kind":"instance_goal","goal_id":"G编号"}} 只用 execution_options.goals 的 goal_id。'
            '说明希望获得的合格量纲及为何可连续执行。verify_identity只核验身份。'
            '执行器逐项绑定当前工件，各阶段照常消耗对应额度。量纲目标遇身份、质量或标定失败交回ReAct。'
        )
    if 'observe_questions' in modes:
        limit = min(offers.get('max_questions', 1), len(offers['questions']))
        count = '1项' if limit == 1 else f'1至{limit}项不同的观察'
        lines.append(
            '观察组 {"decision":{"kind":"observe_questions","question_ids":["Q编号"]}} '
            f'只用 execution_options.questions 的 question_id，选{count}。'
            'A/O编号用于动作，Q编号用于问题。若第二项依赖第一项结果，先做单步。'
            '独立执行后当轮汇总，同源观察不能视为统计独立。'
            'shared_consumers列出共用该问题的候选，存在性观察仍需结合各候选的形态和尺寸判据。'
        )
    if 'physical_group' in modes:
        group_rule = (
            '在 physical_groups 的同一工具组中从不同 views 各选一个，共2至limit项。'
            if offers.get('physical_groups') else '从 physical_pairs 选已有动作对。'
        )
        lines.append(
            '物理并发 {"decision":{"kind":"physical_group","action_ids":["A编号","另一A编号"]}} '
            + group_rule + '逐路判断：假设其余路都已返回理想结果，这一路仍需单独做才并发。'
            '在本轮decision_summary简述各路要解决的问题；同一待查项一次阳性就够、'
            '或第一路结果决定是否追加时，先选单步。limit只是上限。'
        )
    if context.get('parallel_observations') and any(
            action['tool'] == 'detect_fov_initial' for action in context.get('actions', [])):
        lines.append(
            '若有独立原图核验，单步初检可选填 parallel_observation_id，'
            '从 parallel_observations 选与初检同视野的O编号。YOLO负责出框。'
        )
    return ' '.join(lines)


def model_view(context):
    """短编号只在一次请求内有效；返回送模副本及短编号反解，保持原上下文不变。"""
    out = copy.deepcopy(context)
    actions = {f'A{i}': a for i, a in enumerate(out['actions'], 1)}
    observations = {f'O{i}': a for i, a in enumerate(out.get('parallel_observations', []), 1)}
    ids = {key: action['action_id'] for key, action in {**actions, **observations}.items()}
    aliases = {ids[key]: key for key in (*observations, *actions)}
    out['actions'] = [{**action, 'action_id': key} for key, action in actions.items()]
    out['parallel_observations'] = [{**action, 'action_id': key} for key, action in observations.items()]
    out['action_objectives'] = {aliases[k]:v for k,v in out.get('action_objectives',{}).items() if k in aliases}
    if 'action_state' in out:
        out['action_state'] = {aliases[k]: v for k,v in out['action_state'].items() if k in aliases}
        for row in out['action_state'].values():
            if 'existing_crop_actions' in row:
                row['existing_crop_actions'] = [aliases[a] for a in row['existing_crop_actions'] if a in aliases]
    for field in ('verification_opportunities', 'investigation_opportunities'):
        if field in out:
            out[field] = [{**r,'action_id':aliases[r['action_id']]}
                          for r in out[field] if r.get('action_id') in aliases]
    offers = out.get('execution_options') or {}
    if 'physical_pairs' in offers:
        offers['physical_pairs'] = [[aliases[a],aliases[b]] for a,b in offers['physical_pairs']]
    for group in offers.get('physical_groups',[]):
        group['views'] = [[aliases[a] for a in view] for view in group['views']]
    offers['goals'] = [{**g, 'goal_id':f'G{i}', 'entry_action_id':aliases[g['entry_action_id']]}
        for i,g in enumerate(offers.get('goals',[]),1) if g['entry_action_id'] in aliases]
    offers['questions'] = [{**q, 'question_id':f'Q{i}', 'canonical_question_id':q['question_id'],
                            'action_id':aliases[q['action_id']]}
        for i,q in enumerate(offers.get('questions',[]),1) if q['action_id'] in aliases]
    if 'execution_options' in out:
        out['execution_options'] = offers
        # 已在主选择表关联动作，不再重复两套无关联长ID菜单/契约。
        for field in ('semantic_options','semantic_goal_contract','parallel_questions'):
            out.pop(field,None)
    if out.get('format_feedback'):
        out['format_feedback']['allowed_decision_keys'] = decision_fields(out)
    workflow = (out.get('evidence') or {}).get('workflow') or {}
    comparison = workflow.get('comparison_problems') or {}
    if 'entry_path_details' in comparison:
        comparison['entry_path_details'] = [{**p, 'action_id':aliases[p['action_id']]}
            for p in comparison['entry_path_details'] if p['action_id'] in aliases]
    for query in comparison.get('shared_queries', []):
        query['legal_action_ids'] = [aliases[a] for a in query.get('legal_action_ids', []) if a in aliases]
    for problem in comparison.get('questions', []):
        problem['legal_action_ids'] = [aliases[a] for a in problem.get('legal_action_ids', []) if a in aliases]
        problem['entry_paths'] = [{**p, 'action_id': aliases[p['action_id']]}
            for p in problem.get('entry_paths', []) if p['action_id'] in aliases]
        problem['entry_visibility'] = 'current_menu' if problem['entry_paths'] else 'detail_or_unavailable'
    return out, ids


def physical_allowed(ids, offers):
    if not isinstance(ids,list) or not all(isinstance(a,str) for a in ids) or len(set(ids))!=len(ids):
        return False
    if 'physical_groups' not in offers:
        return len(ids)==2 and (ids in offers.get('physical_pairs',[]) or ids[::-1] in offers.get('physical_pairs',[]))
    return any(2 <= len(ids) <= group['limit'] and
               sum(bool(set(view).intersection(ids)) for view in group['views']) == len(ids)
               for group in offers.get('physical_groups',[]))


def decode(reply, context):
    """只展开模型明确选择的模式；不按优先级代选、不补第二个问题。"""
    if not isinstance(reply,dict) or 'decision' not in reply:
        return reply
    if 'action_id' in reply or 'proposal' in reply:
        raise IntegrationError('PLANNER_CHOICE_AMBIGUOUS','decision不能与旧主选择同时使用')
    decision = reply['decision']
    if not isinstance(decision,dict):
        raise IntegrationError('PLANNER_CHOICE_INVALID','decision须为对象')
    value = {k:v for k,v in reply.items() if k != 'decision'}
    # 只归一已知说明字段；未知字段、混合执行选择仍失败，绝不补选动作。
    notes = {'decision_summary', 'override_reason', 'reason', 'rationale', 'explanation'}
    moved = sorted(set(decision) & notes)
    if any(not isinstance(decision[k], str) for k in moved):
        raise IntegrationError('PLANNER_CHOICE_INVALID','选择中的说明字段须为字符串')
    if moved:
        if not value.get('decision_summary'):
            value['decision_summary'] = next((decision[k][:240] for k in
                ('decision_summary', 'reason', 'rationale', 'explanation') if decision.get(k)), '')
        if not value.get('override_reason') and decision.get('override_reason'):
            value['override_reason'] = decision['override_reason'][:240]
        value['choice_normalization'] = {'metadata_fields': moved, 'selected_action_changed': False,
                                          'new_planning_requests': 0}
        decision = {k:v for k,v in decision.items() if k not in notes}
    kind = decision.get('kind')
    audit_decision = decision
    offers = context.get('execution_options') or {}
    if kind == 'action' and set(decision) == {'kind','action_id'}:
        value['action_id'] = decision['action_id']
    elif kind in {'physical_group','physical_pair'} and set(decision) == {'kind','action_ids'}:
        pair = decision['action_ids']
        if (not physical_allowed(pair,offers) or kind=='physical_pair' and len(pair)!=2 or
                any(k in value for k in ('semantic_goal','parallel_question_id','parallel_observation_id','physical_partner_id','physical_partner_ids'))):
            raise IntegrationError('PLANNER_CHOICE_INVALID','须从同一工具组的不同视野中选取动作')
        value['action_id'], value['physical_partner_ids'] = pair[0],pair[1:]
    elif kind == 'instance_goal' and set(decision) == {'kind','goal_id'}:
        goal = next((g for g in offers.get('goals',[]) if g['goal_id']==decision['goal_id']),None)
        if not goal:
            raise IntegrationError('PLANNER_GOAL_UNAVAILABLE','本轮没有该语义目标')
        value.update(action_id=goal['entry_action_id'],semantic_goal=goal['goal'])
    elif kind == 'observe_questions' and set(decision) == {'kind','question_ids'}:
        requested = decision['question_ids']
        if (not isinstance(requested,list) or not 1 <= len(requested) <= offers.get('max_questions',1)
                or any(not isinstance(x,str) for x in requested) or len(set(requested))!=len(requested)):
            raise IntegrationError('PLANNER_QUESTIONS_INVALID','问题组须为本轮1–2个不同问题')
        selected=[];missing=0
        for identifier in requested:
            q=next((q for q in offers.get('questions',[]) if identifier in {
                q['question_id'],q.get('canonical_question_id'),q.get('choice_id'),q.get('action_id')}),None)
            if not q:
                missing+=1;continue
            selected.append(q)
        if not selected:
            raise IntegrationError('PLANNER_QUESTION_UNAVAILABLE','问题不在本轮可见集合')
        if len({q.get('canonical_question_id',q['question_id']) for q in selected}) != len(selected):
            raise IntegrationError('PLANNER_QUESTIONS_INVALID','不能重复同一物理问题')
        value['action_id']=selected[0]['action_id']
        if len(selected)>1:
            value['parallel_question_id']=selected[1].get('canonical_question_id',selected[1]['question_id'])
        if missing:
            value['partial_choice_rejection']={'reason':'PLANNER_QUESTION_UNAVAILABLE',
                'selected_current_questions':len(selected),'rejected_questions':missing,
                'replacement_questions_chosen':0,'extra_planning_requests':0}
            audit_decision={'kind':kind,'question_ids':[q['question_id'] for q in selected],
                            'rejected_question_count':missing}
    else:
        raise IntegrationError('PLANNER_CHOICE_INVALID','未知模式或多余选择字段')
    value['execution_choice']={'kind':kind,'requested':audit_decision}
    return value


def diagnose(reply, context):
    """错误只留数量、已知命名空间与有限短编号；不保存无效自由值。"""
    choice=reply.get('decision') if isinstance(reply,dict) else None
    choice=choice if isinstance(choice,dict) else {}
    requested=choice.get('question_ids');requested=requested if isinstance(requested,list) else []
    table=(context.get('execution_options')or{}).get('questions',[])
    aliases={q['question_id'] for q in table}
    canonical={q.get('canonical_question_id',q['question_id']) for q in table}
    expected = {'action': {'kind','action_id'}, 'instance_goal': {'kind','goal_id'},
                'observe_questions': {'kind','question_ids'}}.get(choice.get('kind'), {'kind'})
    return {'kind':choice.get('kind') if choice.get('kind') in {'action','instance_goal','observe_questions'}else 'unknown',
        'extra_field_names':[k for k in choice if isinstance(k,str) and k not in expected
                             and re.fullmatch(r'[a-z_]{1,40}',k)][:8],
        'requested_count':len(requested),'current_short_matches':sum(isinstance(x,str)and x in aliases for x in requested),
        'current_canonical_matches':sum(isinstance(x,str)and x in canonical for x in requested),
        'unmatched_short_ids':[x for x in requested if isinstance(x,str)and re.fullmatch(r'[AQGO][0-9]{1,3}',x)
                               and x not in aliases][:2], 'menu_sha256':digest(table)}


def audit(context):
    """保存真实送模菜单而非全部合法集合；供下一次排查可见性。"""
    return {'schema':'planner-offer/v1','actions':context['actions'],
        'execution_options':context.get('execution_options',{}),
        'parallel_observations':context.get('parallel_observations',[]),
        'action_state':context.get('action_state', {}),
        'remaining_budget':context.get('remaining_budget'),
        'investigation_opportunities':context.get('investigation_opportunities', []),
        'investigation': {k:((context.get('evidence')or{}).get('workflow')or{}).get(k)
                          for k in ('comparison_problems','investigation_focus','candidate_audit')},
        'reflection_offer': {k:v for k,v in ((context.get('task_reflection')or{}).get('pending')or{}).items()
            if k in {'issue_id','revision','reflection_id','evidence_refs'}},
        'context_sha256':digest(context)}
