"""ReAct 执行选择视图。单动作、实例或问题目标和独立观察组使用同一主选择接口。

消费当前可见合法动作/KB/预算；只描述选择，不执行工具或授予证据资格。
react_graph最终仍逐项校验并晚绑定。model_view仅作短编号映射，供真实LLM和离线请求对拍；
不枚举组合、不强制并行，不产生未来工具序列。完整菜单审计不保存思维链或图像载荷。

主要定义：
- attach。给当前实际可见动作建立目标/问题入口；不把菜单外动作偷偷加进来。
- with_criterion_impact。把可见动作关联到已有判据、量纲收益及同问题反馈。
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
import json
from collections import Counter, defaultdict

from TCM_Fusion.lib.decision.comparison_problems import _action_feature, action_question
from TCM_Fusion.lib.runtime.storage import IntegrationError, digest, read_json


def action_state(runtime, actions, evidence, targets):
    """投影实际菜单里可直接执行的步骤和同视野记录；不从观察文本推断状态。"""
    presence = {}
    for row in evidence.get('observation_states', []):
        key = (row['image_id'], row['feature_id'])
        presence.setdefault(key, set()).add(row['state'])
    rows = {}
    artifacts = runtime.workflow.store.current()['artifacts']
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
        if tool == 'feature_scan':
            from TCM_Fusion.lib.vision.image_tiling import scan_tiles
            limit = target.get('max_tiles') or runtime.workflow.config.get('feature_scan_max_tiles', 4)
            row['tile_limit'] = limit
            row['tile_policy'] = runtime.workflow.config.get('feature_scan_tile_policy', 'prescreen')
            size = artifacts.get('detected:' + image, {}).get('source_size_wh')
            if size and len(size) == 2:
                row['candidate_regions_px'] = scan_tiles(*size, limit)
        if tool == 'locate_feature' and target.get('roi'):
            row['roi_source_px'] = target['roi']
        if target.get('kind') == 'crop' and tool in {'inspect_crop', 'classify_target'}:
            row.update({k: target[k] for k in ('box_source','box_quality','det_score') if k in target})
            box = target.get('source_tight_bbox')
            if not box and target.get('path'):
                import contracts
                try:
                    box = contracts.CropMeta.from_json(target['path']).tight_box
                except (OSError, ValueError, AttributeError):
                    box = None
            if box:
                row['target_source_px'] = list(box)
            if args.get('observation_basis'):
                first = read_json(runtime.workflow.operations)['entries'][args['observation_basis']]
                row['previous_visual_observation'] = first['visual_observation']
            identity = artifacts.get('classification:' + target['id'], {})
            if identity.get('valid'):
                doc = read_json(identity['path'])
                visual = doc.get('visual_identity') or {}
                row['previous_reading'] = {
                    'identity_verified': bool(identity.get('identity_verified')),
                    'source': doc.get('confidence_source'), 'label': doc.get('label'),
                    'reason': str(visual.get('reported_reason') or doc.get('reason') or '')[:160],
                    **{k: visual[k] for k in ('target_part', 'visible_cue') if k in visual}}
        if action.get('scene_criteria'):
            row['scene_criteria'] = action['scene_criteria']
        rows[action['action_id']] = row
    return rows


def add_crop_reuse_hints(rows, actions, targets):
    """给同视野同特征的补扫关联已有裁图入口；不重排、过滤或自动执行工具。"""
    reuse = {}
    for action in actions:
        tool = action['tool']
        row = rows.get(action['action_id'])
        if row is None or tool not in {'classify_target', 'inspect_crop', 'classify_crops_batch',
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


def attach(runtime, context, *, can_pair, targets=None, measurement_actions=None,
           previous_problems=(), allow_problem=True):
    """给当前实际可见动作建立目标/问题入口；不把菜单外动作偷偷加进来。"""
    out = dict(context)
    actions = context['actions']
    targets = runtime.workflow.targets() if targets is None else targets
    out['action_state'] = action_state(runtime, actions + context.get('parallel_observations', []),
                                       context.get('evidence') or {}, targets)
    add_crop_reuse_hints(out['action_state'], actions, targets)
    scan_features = {out['action_state'][a['action_id']]['feature_id'] for a in actions
                     if a['tool'] in {'feature_scan', 'locate_feature'}
                     and out['action_state'].get(a['action_id'], {}).get('feature_id')}
    if scan_features:
        from TCM_Fusion.lib.vision.feature_profiles import build_profiles, hint_text
        profiles = build_profiles(runtime.workflow.config.get('feature_profiles'))
        out['scan_feature_hints'] = {fid: hint_text(profiles.get(fid)) for fid in sorted(scan_features)}
    for aid, state in out['action_state'].items():
        group_key = (measurement_actions or {}).get(aid)
        if group_key in (context.get('measurement_groups') or {}):
            state['measurement_group'] = group_key
    active = set(runtime.candidate_interest()['ids'])
    criteria = [dict(c, candidate_id=r['id']) for r in runtime.kb.candidates()
                if r['id'] in active for c in r['criteria']]
    questions = []
    if can_pair:
        for a in actions:
            if a['tool'] not in {'presence_check', 'scene_check'}:
                continue
            q = action_question(runtime, a, targets=targets)
            if a['tool'] == 'scene_check':
                consumers = sorted({c['candidate_id'] for c in a.get('scene_criteria', [])})
                limits = '只观察该视野中已有特征的组织语境，不能确认身份或尺寸'
            else:
                consumers = sorted({c['candidate_id'] for c in criteria if c['feature'] == q['feature_id']})
                limits = '共同存在性不能区分这些消费者的形态/尺寸；未见仅限当前视野'
            questions.append({**q, 'action_id':a['action_id'], 'tool':a['tool'],
                'shared_consumers':consumers, 'limits':limits})
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
                from TCM_Fusion.lib.agent.measurement_gain import goal_consumers
                group_key = (measurement_actions or {}).get(entry['action_id'])
                group = context.get('measurement_groups', {}).get(group_key)
                if group:
                    tasks += ['measure_' + dimension_use(dim) for dim, status in group['dimensions'].items()
                              if status['worth_measuring']]
                for task in dict.fromkeys(tasks):
                    consumers = goal_consumers(group, task)
                    if consumers == []:
                        continue
                    if consumers is None:
                        consumers = sorted({c['candidate_id'] for c in basis})
                    goal = {'semantic_task':task, 'instance_ref':row['instance_ref'],
                            'problem_id':'instance-question:'+digest([row['instance_ref'],task])[:20]}
                    goals.append({'goal_id':'goal:'+digest([goal,entry['action_id']])[:24],
                        'entry_action_id':entry['action_id'], 'feature_id':row['feature_id'],
                        'goal':goal,
                        'criterion_consumers':consumers,
                        **({'measurement_group': group_key} if group and task != 'verify_identity' else {}),
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
    if allow_problem and runtime.workflow.config.get('semantic_execution') and runtime.workflow.config.get('problem_goal_execution'):
        from TCM_Fusion.lib.agent.problem_goals import offers
        problems = offers(runtime, actions, targets, criteria, previous_problems, context.get('remaining_budget'),
                          measurement_actions=measurement_actions, measurement_groups=context.get('measurement_groups'))
        if problems:
            # 与实例目标共享原有十二个语义选项；仍能单步选择原动作及独立观察。
            problems = problems[:6] if goals else problems[:12]
            shown = out['execution_options']
            shown['problem_goals'] = problems
            shown['goals'] = shown['goals'][:12-len(problems)]
    return out


# 阶段标签描述可推进的步骤，实际证据资格由工具和判据检查。
_CRITERION_STEPS = {
    'detect_fov_initial': 'discover', 'detect_region': 'discover',
    'feature_scan': 'discover', 'locate_feature': 'discover', 'hybrid_scan': 'discover',
    'presence_check': 'presence', 'classify_target': 'identity', 'classify_crops_batch': 'identity',
    'inspect_crop': 'morphology_observation',
    'calibrate_fov': 'calibrate', 'segment_instance': 'mask', 'measure_instances': 'measure',
}


def _criterion_key(question, consumer):
    quant = consumer.get('quant') or {}
    dimension = question.get('dimension')
    bounds = (dimension, quant.get('min'), quant.get('max'), quant.get('extreme')) if dimension else None
    return (consumer['candidate_id'], consumer['role'], question['feature_id'], consumer.get('state'), bounds)


def with_criterion_impact(context, comparison, *, targets=None, feedback=()):
    """以当前菜单为准，用一次内共享的索引呈现各动作的判据用途。"""
    actions = {a['action_id']: a for a in (*context['actions'], *context.get('parallel_observations', []))}
    targets = targets or {}
    by_action = defaultdict(list)
    by_feature = defaultdict(list)
    for question in comparison.get('questions', []):
        by_feature[question['feature_id']].append(question)
        for aid in question.get('legal_action_ids', []):
            if aid in actions:
                by_action[aid].append(question)

    objectives = dict(context.get('action_objectives') or {})
    previous = {row['question_id']: row for row in feedback if row.get('question_id')}
    measurements = context.get('measurement_groups') or {}
    states = context.get('action_state') or {}
    rows = {}
    for aid, action in actions.items():
        tool = action['tool']
        step = _CRITERION_STEPS.get(tool)
        if step is None:
            continue
        questions = by_action[aid]
        target_id = (action.get('args') or {}).get('target_id')
        if tool == 'classify_crops_batch':
            features = dict.fromkeys(_action_feature({'args': {'target_id': crop['id']}}, targets)
                                     for crop in targets.get(target_id, {}).get('crops', []))
            questions = [q for fid in features for q in by_feature[fid]]
        elif tool == 'calibrate_fov':
            features = dict.fromkeys(_action_feature({'args': {'target_id': crop['id']}}, targets)
                                     for crop in targets.values() if crop.get('kind') == 'crop'
                                     and crop.get('image_id') == target_id)
            questions = [q for fid in features for q in by_feature[fid]]

        group = measurements.get(states.get(aid, {}).get('measurement_group'), {})
        references = {}
        for question in questions:
            dimension = question.get('dimension')
            if step in {'calibrate', 'mask', 'measure'} and not dimension:
                continue
            gain = (group.get('dimensions') or {}).get(dimension)
            beneficiaries = ({(row['candidate_id'], row.get('role')) for row in gain['consumers']
                              if row['reasons']} if gain is not None else None)
            for consumer in question.get('consumers', []):
                candidate, role = consumer['candidate_id'], consumer['role']
                if beneficiaries is not None:
                    if (candidate, role) not in beneficiaries:
                        continue
                elif consumer.get('criterion_assessed'):
                    continue
                key = _criterion_key(question, consumer)
                references[rows.setdefault(key, len(rows))] = None
        if references:
            base = objectives.get(aid) or {}
            row = {'criterion_refs': list(references), 'evidence_step': step}
            if 'recent_latency_seconds' in base:
                row['recent_latency_seconds'] = base['recent_latency_seconds']
            last = previous.get(base.get('question_id'))
            if last and not last.get('progress'):
                row['last_question_result'] = [last['tool'], 'no_gain' if last.get('accepted') else 'failed']
            objectives[aid] = row

    if not rows:
        return context
    workflow = (context.get('evidence') or {}).get('workflow') or {}
    visible_questions = (workflow.get('comparison_problems') or {}).get('questions', [])
    visible = {}
    for q, question in enumerate(visible_questions):
        for c, consumer in enumerate(question.get('consumers', [])):
            visible.setdefault(_criterion_key(question, consumer), [q, c])
    # 已送模的判据通过下标引用。比较窗口之外的必要判据各自只补一行。
    index = [visible[key] if key in visible else [*key[:4], list(key[4]) if key[4] else None]
             for key in rows]
    return {**context, 'criterion_index': index, 'action_objectives': objectives}


def decision_fields(context):
    """列出当前菜单可表达的主选择和字段，用于正常规划与格式修正。"""
    offers = context.get('execution_options') or {}
    fields = {'action': ['kind', 'action_id']}
    if offers.get('goals'):
        fields['instance_goal'] = ['kind', 'goal_id']
    if offers.get('problem_goals'):
        fields['problem_goal'] = ['kind', 'goal_id']
    if offers.get('questions') and offers.get('max_questions', 1):
        fields['observe_questions'] = ['kind', 'question_ids']
    if offers.get('identity_group'):
        fields['identity_group'] = ['kind', 'action_ids']
    if offers.get('physical_groups', offers.get('physical_pairs')):
        fields['physical_group'] = ['kind', 'action_ids']
    return fields


def decision_instruction(context):
    """展示实际有入口的决策格式。具体编号见同一份菜单。"""
    modes = decision_fields(context)
    offers = context.get('execution_options') or {}
    lines = [
        '只返回一个JSON对象。decision给出当轮选择和相关的视觉委托，理由写在顶层decision_summary。',
        '单步 {"decision":{"kind":"action","action_id":"A编号"}} 从 actions 选择一个动作，补扫和局部重检也从这里选。',
    ]
    from TCM_Fusion.lib.vision.visual_observation import VISUAL_TOOLS
    offered_tools = {a['tool'] for a in (*context.get('actions', []), *context.get('parallel_observations', []))}
    if offered_tools & VISUAL_TOOLS:
        lines.append('选视觉动作时，在同一个decision里为选中的入口表达你的视觉委托。'
                     '对照已有回执、当前取景和已有提示，决定本轮需要视觉模型帮助辨清什么。'
                     '用 visual_briefs 把能直接发给阅图者的话绑定到相应编号，如 '
                     '{"decision":{"kind":"action","action_id":"A编号","visual_briefs":{"A编号":"你决定让阅图者留意的部位或辨别细节"}}}。'
                     '如果现有提示已足够，该编号填 {"use":"preset"}。具体查什么和是否沿用预设由你决定。'
                     'scan_feature_hints 是补扫已有的特征提示，action_state 的 candidate_regions_px 是原图候选分块，'
                     '同一补扫委托会送到该动作实际执行的各块。不要假定局部已经出现目标。'
                     '组内可分别使用A，观察组也可用Q，伴随观察可用O，目标授权可用入口A。'
                     '委托聚焦下发图像中的可见细节，避免药材名或预期答案。'
                     'decision_summary给出供人阅读的选择依据，visual_observation记录视觉工具实际所见。')
    if 'classify_target' in offered_tools:
        lines.append('classify_target由CNN直接判读时不发GLM。')
    if 'inspect_crop' in offered_tools:
        lines.append('inspect_crop可单独查看当前裁图，包括已分类实例。开放观察后若有新疑点，可在关联'
                     'previous_visual_observation的入口附委托追问；省略则复用原观察。')
    if context.get('criterion_index'):
        lines.append(
            'criterion_refs从0起查criterion_index。双项行是comparison_problems.questions和consumers下标，'
            '五项行是候选、角色、特征、B规则状态、量程（量纲/下限/上限/极值）。调查进度见candidate_audit。evidence_step提示可能推进的步骤。'
            'last_question_result为同视野同问题反馈。量纲须合格实测。'
        )
    if 'instance_goal' in modes:
        lines.append(
            '语义目标 {"decision":{"kind":"instance_goal","goal_id":"G编号"}} 只用 execution_options.goals 的 goal_id。'
            '说明希望获得的合格量纲及为何可连续执行。verify_identity只核验身份。'
            '执行器逐项绑定当前工件，各阶段照常消耗对应额度。量纲目标遇身份、质量或标定失败交回ReAct。'
        )
    if 'problem_goal' in modes:
        lines.append('问题授权 {"decision":{"kind":"problem_goal","goal_id":"P编号"}}。只调查所列视野及量纲；'
                     '复用合格测量，或由所列入口取得唯一实例，最多一次定向补扫和一次身份核验，歧义即回交。')
    if 'observe_questions' in modes:
        limit = min(offers.get('max_questions', 1), len(offers['questions']))
        count = '1项' if limit == 1 else f'1至{limit}项不同的观察'
        width = offers.get('observation_wave_size')
        example_ids = '["Q编号"]'
        id_instruction = '只用'
        if width and limit > width:
            # 仅示意跨过一批的写法，避免用当轮真实编号替模型选项。
            example_ids = json.dumps(['Q甲', 'Q乙', 'Q丙'][:width+1], ensure_ascii=False, separators=(',', ':'))
            id_instruction = '示意编号须换为'
        lines.append(
            '观察组 {"decision":{"kind":"observe_questions","question_ids":' + example_ids + '}} '
            f'{id_instruction} execution_options.questions 的 question_id，选{count}。'
            'A/O编号用于动作，Q编号用于问题。各项必须独立；若后项依赖前项结果，先选前项。'
            '独立执行后当轮汇总，同源观察不能视为统计独立。'
            'shared_consumers列出共用该问题的候选，存在性观察仍需结合各候选的形态和尺寸判据。'
        )
        if width:
            lines.append(f'每批最多{width}项，各项单独扣费。新阳性、未知或调查方向改变即交回余项。')
            if limit > width:
                lines.append('列表按你的调查顺序填写。若首批顺利完成且方向稳定，已列的后项可直接执行。'
                             '若现在就值得查，可一并列入；若价值低可略过，若须等具体结果再决定可暂缓。'
                             '有意暂缓时，可在decision_summary开头简记取舍或所等的反馈。')
    if 'identity_group' in modes:
        lines.append('顺序身份核验 {"decision":{"kind":"identity_group","action_ids":["A编号","另一A编号"]}}。'
                     '从 identity_group.action_ids 按执行顺序选择2至limit张独立裁图，避开conflicts内的组合。'
                     '仅在前张读不清仍值得核验后张时选择；身份确认、冲突、调查依据改变或工具失败即回交。')
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
            '单步初检可搭配同视野的独立原图观察。格式 '
            '{"decision":{"kind":"action","action_id":"A编号"},"parallel_observation_id":"O编号"}。'
            'parallel_observation_id在顶层，从本轮parallel_observations选O编号，无合适项就省略。YOLO负责出框。'
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
    # 同一注册动作可同时有主动作 A 和伴随观察 O。两种选择均保留，只复用完全相同的参数。
    primary = {a['action_id']: (key, a) for key, a in actions.items()}
    for row, original in zip(out['parallel_observations'], observations.values()):
        match = primary.get(original['action_id'])
        if match and original == match[1] and original.get('args'):
            row.pop('args')
            row['args_from_action'] = match[0]
    if 'action_state' in out:
        for row in (*out['actions'], *out['parallel_observations']):
            row.pop('scene_criteria', None)
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
    if 'identity_group' in offers:
        group = offers['identity_group']
        group['action_ids'] = [aliases[a] for a in group['action_ids']]
        group['conflicts'] = [[aliases[a], aliases[b]] for a, b in group['conflicts']]
    if 'physical_pairs' in offers:
        offers['physical_pairs'] = [[aliases[a],aliases[b]] for a,b in offers['physical_pairs']]
    for group in offers.get('physical_groups',[]):
        group['views'] = [[aliases[a] for a in view] for view in group['views']]
    offers['goals'] = [{**g, 'goal_id':f'G{i}', 'entry_action_id':aliases[g['entry_action_id']]}
        for i,g in enumerate(offers.get('goals',[]),1) if g['entry_action_id'] in aliases]
    if offers.get('problem_goals'):
        rows = []
        for i, item in enumerate(offers['problem_goals'], 1):
            short = f'P{i}'
            ids[short] = item['choice_id']
            rows.append({'choice_id': short, 'problem_id': item['problem_id'], 'feature_id': item['feature_id'],
                         'image_id': item['image_id'], 'dimension': item['dimension'],
                         'morphology_question': item['morphology_question'],
                         'entry': aliases[item['entry_action_id']] if item['entry_action_id'] else 'reuse',
                         'max_tiles': item['max_tiles']})
        offers['problem_goals'] = rows
    offers['questions'] = [{**q, 'question_id':f'Q{i}', 'canonical_question_id':q['question_id'],
                            'action_id':aliases[q['action_id']]}
        for i,q in enumerate(offers.get('questions',[]),1) if q['action_id'] in aliases]
    limit = min(offers.get('max_questions', 1), len(offers['questions']))
    width = offers.get('observation_wave_size')
    if width and limit > width:
        offers['choice_note'] = (f'本轮可一次授权1至{limit}题，每批最多执行{width}题；'
                                 '可跨批选择、只选部分或直接复核。')
    if previous := out.get('last_observation_group'):
        remaining = set(previous['unissued'])
        out['last_observation_group'] = {
            'reason':previous['reason'], 'issued_count':len(previous['issued']), 'unissued_count':len(remaining),
            'still_offered':[q['question_id'] for q in offers['questions'] if ids[q['action_id']] in remaining]}
    if 'execution_options' in out:
        out['execution_options'] = offers
        # 已在主选择表关联动作，不再重复两套无关联长ID菜单/契约。
        for field in ('semantic_options','semantic_goal_contract','parallel_questions'):
            out.pop(field,None)
    if out.get('format_feedback'):
        out['format_feedback']['allowed_decision_keys'] = decision_fields(out)
    workflow = (out.get('evidence') or {}).get('workflow') or {}
    candidates = {row['candidate_id']: row for row in (workflow.get('candidate_audit') or {}).get('candidates', [])}
    for row in [*workflow.get('criterion_states', []), *workflow.get('supporting_states', [])]:
        if 'check_state' in row:
            row['presence_check_state'] = row.pop('check_state')
        if 'state' in row:
            role = row.get('role', 'supporting')
            investigation = candidates.get(row.get('candidate_id'), {}).get(role, {})
            if row.get('feature_id') in investigation:
                row['investigation_state'] = investigation[row['feature_id']]
    comparison = workflow.get('comparison_problems') or {}
    if 'entry_path_details' in comparison:
        comparison['entry_path_details'] = [{**p, 'action_id':aliases[p['action_id']]}
            for p in comparison['entry_path_details'] if p['action_id'] in aliases]
    for query in comparison.get('shared_queries', []):
        query['legal_action_ids'] = [aliases[a] for a in query.get('legal_action_ids', []) if a in aliases]
    for problem in comparison.get('questions', []):
        if 'current_states' in problem:
            problem['current_rule_states'] = problem.pop('current_states')
        for consumer in problem.get('consumers', []):
            if 'state' in consumer:
                consumer['rule_state'] = consumer.pop('state')
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
    # 只归一已知的附加字段，主选择仍按各模式的结构严格解码。
    normalization = {}
    if 'visual_briefs' in decision:
        decision = dict(decision)
        briefs = decision.pop('visual_briefs')
        if 'visual_briefs' in value and value['visual_briefs'] != briefs:
            value.pop('visual_briefs')
            normalization['discarded_optional_fields'] = {'visual_briefs': 'conflicting_values'}
        else:
            value['visual_briefs'] = briefs
            normalization['relocated_fields'] = ['visual_briefs']
    notes = {'decision_summary', 'override_reason', 'reason', 'rationale', 'explanation'}
    moved = sorted(set(decision) & notes)
    if any(not isinstance(decision[k], str) for k in moved):
        raise IntegrationError('PLANNER_CHOICE_INVALID','选择中的说明字段须为字符串')
    if moved:
        if not value.get('decision_summary'):
            value['decision_summary'] = next((decision[k] for k in
                ('decision_summary', 'reason', 'rationale', 'explanation') if decision.get(k)), '')
        if not value.get('override_reason') and decision.get('override_reason'):
            value['override_reason'] = decision['override_reason']
        normalization['metadata_fields'] = moved
        decision = {k:v for k,v in decision.items() if k not in notes}

    kind = decision.get('kind')
    observation_field = 'parallel_observation_id'
    if kind == 'action' and observation_field in decision:
        decision = dict(decision)
        observation_id = decision.pop(observation_field)
        if observation_field in value and value[observation_field] != observation_id:
            value.pop(observation_field)
            normalization.setdefault('discarded_optional_fields', {})[observation_field] = 'conflicting_values'
        else:
            value[observation_field] = observation_id
            normalization.setdefault('relocated_fields', []).append(observation_field)

    selectors = {'action_id', 'proposal', 'semantic_goal', 'problem_goal_id', 'parallel_question_id',
                 'parallel_observation_id', 'physical_partner_id', 'physical_partner_ids'}
    if (kind == 'problem_goal' and selectors.intersection(value)) or 'problem_goal_id' in value:
        raise IntegrationError('PLANNER_CHOICE_AMBIGUOUS', '问题授权只能作为本轮唯一主选择')
    if ('identity_tail' in value or 'observation_ids' in value
            or kind in {'identity_group','observe_questions'} and selectors.intersection(value)):
        raise IntegrationError('PLANNER_CHOICE_AMBIGUOUS', '分组核验只接受一种执行选择')
    audit_decision = decision
    offers = context.get('execution_options') or {}
    if kind == 'action' and set(decision) == {'kind','action_id'}:
        value['action_id'] = decision['action_id']
    elif kind == 'identity_group' and set(decision) == {'kind', 'action_ids'}:
        from TCM_Fusion.lib.agent.identity_group import allowed
        ids = decision['action_ids']
        if not allowed(ids, offers.get('identity_group', {})):
            raise IntegrationError('PLANNER_CHOICE_INVALID', '须选择本轮可组合的独立身份动作')
        value['action_id'], value['identity_tail'] = ids[0], ids[1:]
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
    elif kind == 'problem_goal' and set(decision) == {'kind', 'goal_id'}:
        identifier = decision['goal_id']
        if not any(item['choice_id'] == identifier for item in offers.get('problem_goals', [])):
            raise IntegrationError('PLANNER_GOAL_UNAVAILABLE', '本轮没有该视野的问题授权入口')
        value['problem_goal_id'] = identifier
    elif kind == 'observe_questions' and set(decision) == {'kind','question_ids'}:
        requested = decision['question_ids']
        if (not isinstance(requested,list) or not 1 <= len(requested) <= offers.get('max_questions',1)
                or any(not isinstance(x,str) for x in requested) or len(set(requested))!=len(requested)):
            raise IntegrationError('PLANNER_QUESTIONS_INVALID','须在本轮上限内选择不同问题')
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
        if len(selected)>2 or len(selected)==2 and offers.get('observation_wave_size') == 1:
            value['observation_ids']=[q['action_id'] for q in selected]
        elif len(selected)>1:
            value['parallel_question_id']=selected[1].get('canonical_question_id',selected[1]['question_id'])
        if missing:
            value['partial_choice_rejection']={'reason':'PLANNER_QUESTION_UNAVAILABLE',
                'selected_current_questions':len(selected),'rejected_questions':missing,
                'replacement_questions_chosen':0,'extra_planning_requests':0}
            audit_decision={'kind':kind,'question_ids':[q['question_id'] for q in selected],
                            'rejected_question_count':missing}
    else:
        raise IntegrationError('PLANNER_CHOICE_INVALID','未知模式或多余选择字段')
    if normalization:
        value['choice_normalization'] = {
            **normalization, 'selected_action_changed': False, 'new_planning_requests': 0,
        }
    value['execution_choice']={'kind':kind,'requested':audit_decision}
    return value


def diagnose(reply, context):
    """错误只留数量、已知命名空间与有限短编号；不保存无效自由值。"""
    choice=reply.get('decision') if isinstance(reply,dict) else None
    choice=choice if isinstance(choice,dict) else {}
    offers = context.get('execution_options') or {}
    if choice.get('kind') == 'identity_group':
        requested = choice.get('action_ids')
        table = offers.get('identity_group', {})
        aliases = set(table.get('action_ids', []))
        canonical = aliases
    else:
        requested = choice.get('question_ids')
        table = offers.get('questions', [])
        aliases = {q['question_id'] for q in table}
        canonical = {q.get('canonical_question_id', q['question_id']) for q in table}
    requested = requested if isinstance(requested, list) else []
    expected = {'action': {'kind','action_id'}, 'instance_goal': {'kind','goal_id'}, 'problem_goal': {'kind','goal_id'},
                'identity_group': {'kind', 'action_ids'}, 'observe_questions': {'kind','question_ids'}}.get(choice.get('kind'), {'kind'})
    return {'kind':choice.get('kind') if choice.get('kind') in {'action','instance_goal','problem_goal','identity_group','observe_questions'}else 'unknown',
        'extra_field_names':[k for k in choice if isinstance(k,str) and k not in expected
                             and re.fullmatch(r'[a-z_]{1,40}',k)][:8],
        'requested_count':len(requested),'current_short_matches':sum(isinstance(x,str)and x in aliases for x in requested),
        'current_canonical_matches':sum(isinstance(x,str)and x in canonical for x in requested),
        'unmatched_short_ids':[x for x in requested if isinstance(x,str)and re.fullmatch(r'[AQGO][0-9]{1,3}',x)
                               and x not in aliases][:2], 'menu_sha256':digest(table)}


def compact_view_references(context):
    """同一轮内可逆地压缩重复视野引用，包括复合工件 ID 和字典键。"""
    raw = json.dumps(context, ensure_ascii=False, separators=(',', ':'))
    if '@V' in raw or 'view_id_map' in context:
        return context, {}
    pattern = r'img_[0-9a-f]{64}(?![0-9a-f])'
    counts = Counter(re.findall(pattern, raw))
    names = {key: f'@V{i}@' for i, key in enumerate(sorted(counts), 1) if counts[key] > 1}
    if not names:
        return context, {}
    short = re.sub(pattern, lambda hit: names.get(hit[0], hit[0]), raw)
    reverse = {value: key for key, value in names.items()}
    return {**json.loads(short), 'view_id_map': reverse}, reverse


def restore_view_references(reply, reverse):
    if not reverse:
        return reply
    raw = json.dumps(reply, ensure_ascii=False)
    return json.loads(re.sub(r'@V[0-9]+@', lambda hit: reverse.get(hit[0], hit[0]), raw))


def audit(context):
    """保存真实送模菜单而非全部合法集合；供下一次排查可见性。"""
    readings = [row['visual_observation'] for row in context.get('recent_action_feedback', ())
                if row.get('accepted') and isinstance(row.get('visual_observation'), dict)]
    return {'schema':'planner-offer/v1','actions':context['actions'],
        **({'previous_visual_observations': readings} if readings else {}),
        **({'view_id_map':context['view_id_map']} if 'view_id_map' in context else {}),
        'execution_options':context.get('execution_options',{}),
        'parallel_observations':context.get('parallel_observations',[]),
        'action_state':context.get('action_state', {}),
        **({'scan_feature_hints': context['scan_feature_hints']} if context.get('scan_feature_hints') else {}),
        **({key: context[key] for key in ('criterion_index', 'action_objectives')}
           if context.get('criterion_index') else {}),
        'measurement_groups':context.get('measurement_groups', {}),
        'round_delta':context.get('round_delta', {}),
        'remaining_budget':context.get('remaining_budget'),
        'investigation_opportunities':context.get('investigation_opportunities', []),
        'investigation': {k:((context.get('evidence')or{}).get('workflow')or{}).get(k)
                          for k in ('comparison_problems','investigation_focus','candidate_audit')},
        'reflection_offer': {k:v for k,v in ((context.get('task_reflection')or{}).get('pending')or{}).items()
            if k in {'issue_id','revision','reflection_id','evidence_refs'}},
        'context_sha256':digest(context)}
