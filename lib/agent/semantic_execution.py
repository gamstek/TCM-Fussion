"""D3实验串行语义图：模型选实例目标，程序逐节点晚绑定当前合法工具。

消费InstanceLedger/CropMeta/现有工具前置；不接收未来crop/seg ID或工具序列。
由react_graph通过已有MCP/LocalToolbox执行，身份、资格或输入变化即回交ReAct。
不实现D4并行调度/事件总线，不写EvidenceStore、不产生模型或尺寸事实。

主要定义：
- bind。精确实例内的等价别名可复用；像素/身份多解时不任取首个目标。
- options。
- compile_goal。
- next_action。每次执行前重读证据；返回当前动作/局部事实，永不构造未来工具target。
- finish。

关联：
- 调用方 「lib/agent/planning_choices.py」、「lib/agent/react_graph.py」、「tests/agent/test_planning_choices.py」、「tests/agent/test_reflection_strategy.py」、「tests/integration/test_agent_efficiency.py」、「tests/integration/test_d3.py」。
- 直接依赖 「lib/evidence/instance_ledger.py」、「lib/evidence/purpose_evidence.py」、「lib/runtime/storage.py」、「lib/tools/tool_registry.py」。
"""
import contracts

from TCM_Fusion.lib.evidence.instance_ledger import describe, project
from TCM_Fusion.lib.runtime.storage import digest, file_hash, IntegrationError, read_json, finite_number
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
        'nodes': [{'node_id': task, 'semantic_task': task, 'instance_ref': goal['instance_ref'],
                   'depends_on': [] if i == 0 else [tasks[i-1]], 'status': 'pending',
                   'required_facts': REQUIRED_FACTS[task],
                   'invalid_if': ['bound_input_changed', 'qualification_revoked', 'task_closed']}
                  for i, task in enumerate(tasks)], 'status': 'running'}


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


def next_action(runtime, plan, state):
    """每次执行前重读证据；返回当前动作/局部事实，永不构造未来工具target。"""
    workflow = runtime.workflow
    if read_json(workflow.budget_path).get('closed'):
        return None, 'task_closed'
    if workflow.cancel_event.is_set():
        return None, 'task_cancelled'
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
    identity_ok = bool(identity.get('valid') and identity.get('identity_verified')
                       and identity.get('feature_id') == meta.feature_id)
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
        from TCM_Fusion.lib.evidence.purpose_evidence import dimension_use
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
