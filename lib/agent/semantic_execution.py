"""串行语义 DAG。接受 ReAct 的实例目标或单视野问题授权，逐节点绑定当前合法工具。

消费InstanceLedger/CropMeta/现有工具前置；不接收未来crop/seg ID或工具序列。
由react_graph通过已有MCP/LocalToolbox执行，身份、资格或输入变化即回交ReAct。
不实现D4并行调度/事件总线，不写EvidenceStore、不产生模型或尺寸事实。

主要定义：
- bind。精确实例内的等价别名可复用；像素/身份多解时不任取首个目标。
- options。
- compile_goal。
- compile_problem。绑定 ReAct 实际选择的问题和视野，首个节点只解析已提交的实例或可用补扫。
- next_action。每次执行前重读证据；返回当前动作/局部事实，永不构造未来工具target。
- finish。

关联：
- 调用方 「lib/agent/planning_choices.py」、「lib/agent/react_graph.py」、「tests/agent/test_planning_choices.py」、「tests/agent/test_reflection_strategy.py」、「tests/integration/test_agent_efficiency.py」、「tests/integration/test_d3.py」。
- 直接依赖 「lib/evidence/instance_ledger.py」、「lib/evidence/purpose_evidence.py」、「lib/runtime/storage.py」、「lib/tools/tool_registry.py」。
"""
import contracts

from TCM_Fusion.lib.evidence.instance_ledger import describe
from TCM_Fusion.lib.evidence.search_ledger import scan_coverage
from TCM_Fusion.lib.agent.problem_goals import candidates, qualified_facts
from TCM_Fusion.lib.evidence.purpose_evidence import dimension_use
from TCM_Fusion.lib.runtime.storage import digest, file_hash, IntegrationError, read_json, finite_number, reusable_reads
from TCM_Fusion.lib.tools.tool_registry import validate_arguments


GOAL_SCHEMA = {'type': 'object', 'additionalProperties': False,
    'required': ['semantic_task', 'instance_ref', 'problem_id'], 'properties': {
        'semantic_task': {'enum': ['verify_identity', 'measure_length', 'measure_width', 'measure_area']},
        'instance_ref': {'type': 'string'}, 'problem_id': {'type': 'string', 'minLength': 1, 'maxLength': 160}}}
TASKS = ('verify_identity', 'obtain_mask', 'check_use', 'measure', 'update_criteria')
GOAL_CONTRACT = {
    'verify_identity': {'result':'只核验身份；不承诺mask或尺寸', 'may_require':['classify_target']},
    'measure_length': {'result':'取得合格长度数值；身份、mask、标定或用途不合格即回交',
                       'may_require':['classify_target','segment_instance','measure_instances']},
    'measure_width': {'result':'取得合格宽度数值；不是仅确认身份',
                      'may_require':['classify_target','segment_instance','measure_instances']},
    'measure_area': {'result':'取得合格面积数值；不是仅确认身份',
                     'may_require':['classify_target','segment_instance','measure_instances']}}
REQUIRED_FACTS = {'verify_identity': ['instance_exists'],
    'obtain_mask': ['identity_eligible', 'calibration_trusted'],
    'check_use': ['identity_eligible', 'mask_valid', 'calibration_trusted', 'dimension_eligible'],
    'measure': ['identity_eligible', 'mask_valid', 'calibration_trusted', 'dimension_eligible'],
    'update_criteria': ['measurement_available']}


def bind(runtime, instance_ref):
    """精确实例内的等价别名可复用；像素/身份多解时不任取首个目标。"""
    snap = runtime.workflow.store.current()
    matches = []
    for crop in snap['artifacts'].values():
        if crop.get('valid') and crop.get('kind') == 'crop':
            try:
                bound = describe(crop, snap)
            except (OSError, ValueError, contracts.ContractError) as exc:
                raise IntegrationError('SEMANTIC_BINDING_INVALID', '实例工件不可读取或不满足契约') from exc
            if bound['instance_id'] == instance_ref:
                matches.append((crop, bound))
    if not matches:
        raise IntegrationError('SEMANTIC_INSTANCE_MISSING', '当前没有该实例的合法裁图')
    signatures = {digest([b['crop_sha256'], b['feature_id'], b['geometry'], b['px_per_um']]) for _, b in matches}
    if len(signatures) != 1:
        raise IntegrationError('SEMANTIC_BINDING_AMBIGUOUS', '同一实例存在非等价裁图或身份多解')
    verified = []
    for crop, bound in matches:
        identity = snap['artifacts'].get('classification:' + crop['id'], {})
        if identity.get('valid') and identity.get('identity_verified') and identity.get('feature_id') == bound['feature_id']:
            verified.append((crop, bound))
    crop, bound = min(verified or matches, key=lambda row: row[0]['id'])
    return crop, bound


def options(runtime, actions):
    rows = {}
    snap = runtime.workflow.store.current()
    for action in actions:
        if action['tool'] not in {'classify_target', 'segment_instance', 'measure_instances'}:
            continue
        target = runtime.workflow.targets().get(action.get('args', {}).get('target_id') or action.get('target_id'))
        if target:
            bound = describe(target, snap)
            if bound:
                row = rows.setdefault(bound['instance_id'], {'instance_ref': bound['instance_id'],
                    'feature_id': bound['feature_id'], 'semantic_tasks': list(GOAL_SCHEMA['properties']['semantic_task']['enum']),
                    'entry_actions':[]})
                row['entry_actions'].append({'action_id':action['action_id'],'tool':action['tool'],
                    'automatic_measurement':bool(target.get('measure_if') or
                        runtime.workflow.config.get('auto_measure_after_segment',False))})
    return list(rows.values())[:8]


def _bound_input(bound):
    return digest([bound['source_image_sha256'], bound['geometry'], bound['crop_sha256'], bound['feature_id']])


def _nodes(tasks, instance_ref, predecessor=None):
    result = []
    for task in tasks:
        result.append({'node_id': task, 'semantic_task': task, 'instance_ref': instance_ref,
                       'depends_on': [predecessor] if predecessor else [], 'status': 'pending',
                       'required_facts': REQUIRED_FACTS[task],
                       'invalid_if': ['bound_input_changed', 'qualification_revoked', 'task_closed']})
        predecessor = task
    return result


def compile_goal(runtime, goal, selected):
    validate_arguments(GOAL_SCHEMA, goal)
    if goal['semantic_task'] == 'verify_identity' and selected['tool'] != 'classify_target':
        raise IntegrationError('SEMANTIC_GOAL_SCOPE','仅身份目标不能授权分割或测量入口')
    crop, bound = bind(runtime, goal['instance_ref'])
    target = runtime.workflow.targets().get((selected.get('args') or {}).get('target_id'))
    selected_bound = describe(target, runtime.workflow.store.current()) if target else None
    if (selected['tool'] not in {'classify_target', 'segment_instance', 'measure_instances'}
            or not selected_bound or selected_bound['instance_id'] != goal['instance_ref']):
        raise IntegrationError('SEMANTIC_GOAL_SCOPE', '语义目标必须对应模型本轮选定实例')
    identity = runtime.workflow.store.current()['artifacts'].get('classification:'+crop['id'], {})
    if selected['tool'] != 'classify_target' and not (
            identity.get('valid') and identity.get('identity_verified') and identity.get('feature_id') == bound['feature_id']):
        raise IntegrationError('SEMANTIC_IDENTITY_REQUIRED', '语义测量不能跨过身份前置')
    dimension = {'measure_length':'length', 'measure_width':'width', 'measure_area':'area'}.get(goal['semantic_task'])
    tasks = TASKS if dimension else TASKS[:1]
    return {'schema': 'serial-semantic-plan/d3', 'plan_id': 'plan:'+digest(goal)[:24],
        'goal': goal, 'dimension': dimension, 'bound_input': _bound_input(bound),
        'bound_crop_id': selected_bound['crop_id'],
        'nodes': _nodes(tasks, goal['instance_ref']), 'status': 'running'}


def compile_problem(offer):
    """只接受主机从本轮菜单反解出的授权；首个节点暂时没有实例。"""
    dimension = dimension_use(offer['dimension'])
    return {'schema': 'serial-semantic-plan/problem-v1', 'plan_id': offer['choice_id'],
            'authorization': dict(offer), 'goal': {'problem_id': offer['problem_id'],
                'semantic_task': 'measure_' + dimension}, 'dimension': dimension, 'status': 'running',
            'issued': {}, 'nodes': [{'node_id': 'resolve_instance', 'semantic_task': 'resolve_instance',
                'depends_on': [], 'status': 'pending', 'required_facts': ['authorized_view_and_feature']}]}


def _resolve_problem(runtime, plan, state, snap):
    scope = plan['authorization']
    image, fid = scope['image_id'], scope['feature_id']
    groups, conflicts = candidates(snap, image)
    instances = groups.get(fid, {})
    if 'feature_scan' in plan['issued']:
        marker = snap['artifacts'].get('scanned:' + image + ':' + fid, {})
        operations = read_json(runtime.workflow.operations)['entries'].values()
        scan = next((r for r in operations if r.get('action_id') == plan['issued']['feature_scan']
                     and r.get('tool') == 'feature_scan' and r.get('target_id') == scope['entry_target']), None)
        if not marker.get('valid') or not scan:
            return None, 'scan_receipt_missing'
        status, coverage = scan_coverage(scan)
        plan['scan_receipt'] = {'artifact_id': marker['id'], 'sha256': marker['sha256'], 'coverage': status}
        if status != 'shareable':
            return None, {'partial': 'scan_partial', 'failed': 'scan_failed'}.get(status, 'scan_coverage_unknown')
        if not instances:
            return None, 'scan_empty' if coverage.get('crop_count') == 0 else 'no_reviewable_instance'
    facts = []
    for iid, versions in instances.items():
        if iid not in conflicts:
            for crop, bound in versions:
                facts.extend(qualified_facts(runtime, snap, crop, bound, scope['dimension']))
    if facts:
        runtime.workflow.store.verified_records()
        plan['facts'] = facts
        plan['nodes'][0]['status'] = 'succeeded'
        refs = {fact['instance_ref'] for fact in facts}
        if len(refs) == 1:
            plan['goal']['instance_ref'] = next(iter(refs))
        return None, 'local_goal_satisfied'
    if scope['entry_tool'] == 'reuse':
        return None, 'qualified_fact_unavailable'
    if not instances:
        if scope['entry_tool'] != 'feature_scan':
            return None, 'missing_legal_prerequisite'
        target = runtime.workflow.targets().get(scope['entry_target'], {})
        if target.get('image_id') != image or target.get('feature_id') != fid:
            return None, 'missing_legal_prerequisite'
        if target['max_tiles'] > scope['max_tiles']:
            return None, 'scan_scope_changed'
        action = next((a for a in runtime.legal_actions(state) if a['tool'] == 'feature_scan'
                       and a['action_id'] == scope['entry_action_id']), None)
        return (action, 'action_ready') if action else (None, 'missing_legal_prerequisite')
    if len(instances) != 1 or conflicts.intersection(instances):
        return None, 'ambiguous_instances'
    iid = next(iter(instances))
    crop, bound = bind(runtime, iid)
    if (bound['source_image_sha256'] != scope['source_sha256'] or crop['image_id'] != image
            or bound['feature_id'] != fid or bound['granularity'] != 'instance'):
        return None, 'bound_input_changed'
    plan['goal']['instance_ref'] = iid
    plan.update(bound_input=_bound_input(bound), bound_crop_id=crop['id'])
    plan['nodes'][0].update(status='succeeded', instance_ref=iid)
    plan['nodes'].extend(_nodes(TASKS, iid, 'resolve_instance'))
    return None, 'instance_bound'


def _note_identity_block(plan, snapshot, crop_id, expected_feature):
    """留下本实例的门禁结果，供 ReAct 区分身份冲突与图像不可判。"""
    identity = snapshot['artifacts'].get('classification:' + crop_id, {})
    if not identity.get('valid'):
        return
    observed = identity.get('feature_id') or None
    try:
        result = read_json(identity['path']) if file_hash(identity['path']) == identity['sha256'] else {}
    except (OSError, ValueError):
        result = {}
    if isinstance(result, dict) and result.get('reject') == 'unreadable':
        outcome = 'unreadable'
    elif identity.get('review_required'):
        outcome = 'review_pending'
    elif observed and observed != expected_feature:
        outcome = 'feature_mismatch'
    else:
        outcome = 'identity_unverified'
    plan['identity_failure'] = {
        'outcome': outcome, 'expected_feature': expected_feature, 'reported_feature': observed,
        'instance_ref': plan['goal']['instance_ref'], 'semantic_task': plan['goal']['semantic_task'],
        'crop_id': crop_id, 'classification_ref': identity['id'], 'classification_sha256': identity['sha256'],
        'evidence_version': snapshot['version'], 'bound_input': plan['bound_input'],
    }


@reusable_reads()
def next_action(runtime, plan, state):
    """每次执行前重读证据；返回当前动作/局部事实，永不构造未来工具target。"""
    workflow = runtime.workflow
    if read_json(workflow.budget_path).get('closed'):
        return None, 'task_closed'
    if workflow.cancel_event.is_set():
        return None, 'task_cancelled'
    scope = plan.get('authorization')
    if scope:
        snap = workflow.store.current()
        if not plan['issued'] and snap['version'] != scope['version']:
            return None, 'evidence_changed_before_dispatch'
        if not set(scope['consumers']).intersection(runtime.candidate_interest()['ids']):
            return None, 'problem_inactive'
        image = snap['artifacts'].get(scope['image_id'], {})
        scale = snap['artifacts'].get('scale:' + scope['image_id'], {})
        try:
            if (not image.get('valid') or image.get('sha256') != scope['source_sha256']
                    or file_hash(image['path']) != scope['source_sha256']):
                return None, 'source_changed'
            if (not scale.get('valid') or scale['sha256'] != scope['scale_sha256']
                    or file_hash(scale['path']) != scope['scale_sha256']):
                return None, 'calibration_changed'
        except OSError:
            return None, 'scope_artifact_unreadable'
        if 'instance_ref' not in plan['goal']:
            action, reason = _resolve_problem(runtime, plan, state, snap)
            if reason != 'instance_bound':
                return action, reason
    try:
        crop, bound = bind(runtime, plan['goal']['instance_ref'])
    except IntegrationError as exc:
        # 分类改判可能新增同像素异类侧车。只读原目标用于解释，保留绑定歧义的硬回交。
        if exc.code == 'SEMANTIC_BINDING_AMBIGUOUS' and plan.get('bound_crop_id'):
            snap = workflow.store.current()
            original = snap['artifacts'].get(plan['bound_crop_id'], {})
            if original.get('valid'):
                original_bound = describe(original, snap)
                identity = snap['artifacts'].get('classification:' + original['id'], {})
                if (original_bound and _bound_input(original_bound) == plan['bound_input'] and identity.get('feature_id')
                        and identity['feature_id'] != original_bound['feature_id']):
                    _note_identity_block(plan, snap, original['id'], original_bound['feature_id'])
        raise
    if _bound_input(bound) != plan['bound_input']:
        return None, 'bound_input_changed'
    snap = workflow.store.current(); arts = snap['artifacts']
    identity = arts.get('classification:'+crop['id'], {})
    meta = contracts.CropMeta.from_json(crop['path'])
    if scope and (meta.granularity != 'instance' or meta.px_per_um != scale.get('px_per_um')):
        return None, 'dimension_not_qualified'
    identity_ok = bool(identity.get('valid') and identity.get('identity_verified')
                       and identity.get('feature_id') == meta.feature_id)
    if (scope or plan.get('dimension')) and identity_ok and workflow.crop_granularity(crop, meta) != 'instance':
        return None, 'granularity_not_qualified'
    seg = arts.get('seg:'+crop['id'], {})
    measure = arts.get('measurement:seg:'+crop['id'], {})
    def qualified():
        scale = arts.get('scale:'+crop['image_id'], {})
        if (not identity_ok or not seg.get('valid') or seg.get('quality') != 'valid'
                or not scale.get('valid') or meta.px_per_um != scale.get('px_per_um')):
            return False
        if workflow.purpose_quality():
            qa = workflow.assessment(seg)
            return bool(qa and qa['integrity'] and qa['uses'][plan['dimension']]['allowed'])
        return True
    def requested_value_available():
        if scope:
            facts = qualified_facts(runtime, snap, crop, bound, scope['dimension'])
            if facts:
                plan['facts'] = facts
            return bool(facts)
        rows = [entry['data'] for entry in snap['records'] if entry['scope'] == seg.get('id')
                and measure.get('id') in entry['depends_on']]
        for row in rows:
            for field, value in row.items():
                if (field in {'feret_max_um', 'skeleton_len_um', 'area_um2'}
                        or 'width' in field and field.endswith('_um')):
                    if dimension_use(field) == plan['dimension'] and finite_number(value, minimum=0.000001):
                        return True
            primary = row.get('primary_dimension') or ''
            if primary and dimension_use(primary) == plan['dimension'] and finite_number(
                    row.get('primary_value_um'), minimum=0.000001):
                return True
        return False
    legal = runtime.legal_actions(state)
    for node in plan['nodes']:
        if node['status'] == 'succeeded':
            continue
        task = node['semantic_task']
        tool, target = None, crop['id']
        if task == 'verify_identity':
            if identity_ok:
                node['status'] = 'succeeded'; continue
            if identity.get('valid') and not identity.get('review_required'):
                _note_identity_block(plan, snap, crop['id'], meta.feature_id)
                return None, 'identity_rejected'
            tool = 'classify_target'
        elif task == 'obtain_mask':
            if not identity_ok:
                _note_identity_block(plan, snap, crop['id'], meta.feature_id)
                return None, 'identity_not_qualified'
            if seg.get('valid'):
                if seg.get('quality') != 'valid':
                    return None, 'mask_review'
                node['status'] = 'succeeded'; continue
            tool = 'segment_instance'
        elif task == 'check_use':
            if not qualified():
                return None, 'dimension_not_qualified'
            node['status'] = 'succeeded'; continue
        elif task == 'measure':
            if not qualified():
                return None, 'qualification_revoked'
            if measure.get('valid'):
                if not requested_value_available():
                    return None, 'requested_dimension_unavailable'
                node['status'] = 'succeeded'; continue
            tool, target = 'measure_instances', seg.get('id')
        elif task == 'update_criteria':
            runtime.score(state.get('observations', []))
            node['status'] = 'succeeded'; continue
        action = next((a for a in legal if a['tool'] == tool and a.get('args', {}).get('target_id') == target), None)
        if action is None:
            if task == 'verify_identity':
                _note_identity_block(plan, snap, crop['id'], meta.feature_id)
            return None, 'missing_legal_prerequisite'
        if tool in {'segment_instance', 'measure_instances'} and plan['dimension']:
            from TCM_Fusion.lib.agent.measurement_gain import measurement_menu
            from TCM_Fusion.lib.agent.problem_goals import measurement_useful
            scored = runtime.score(state.get('observations', []))
            choices, groups, bindings = measurement_menu(runtime, [action], scored, workflow.targets())
            group = groups.get(bindings.get(action['action_id']))
            consumers = scope['consumers'] if scope else None
            if not choices or not measurement_useful(group, plan['goal']['semantic_task'], consumers):
                return None, 'measurement_value_changed'
        return action, 'action_ready'
    plan['status'] = 'completed'
    return None, 'local_goal_satisfied'


def finish(plan, reason):
    plan['status'] = 'completed' if reason == 'local_goal_satisfied' else 'returned_to_react'
    plan['return_reason'] = reason
    for node in plan['nodes']:
        if node['status'] == 'pending':
            node['status'] = 'blocked'
    return plan
