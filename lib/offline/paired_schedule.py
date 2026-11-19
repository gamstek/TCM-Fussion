"""比较同图多视野真实任务的调查轨迹、判据覆盖和成本。只读运行记录，
校验像素、冷启动来源及 D4 开关，对旧报告保留当时可核实的字段。输出逐案数据，
不把缺失的 token 当作零，不把一次配对用于推断总体识别准确率。

主要定义：inspect_run 提取单案事实，包括无增益动作实际得到的诊断结果。
compare 将单步对照、并发组和可选历史案按输入顺序对齐，并校验配置一致性。

关联：cli.offline 的 schedule-compare；消费实际入口的 job、trace、证据包、
资源账本与 A 推理工件。不会发起模型请求或改变任何证据。
"""
import re
from collections import Counter, defaultdict, deque
from pathlib import Path

from TCM_Fusion.lib.evidence.search_ledger import scan_coverage
from TCM_Fusion.lib.runtime.storage import IntegrationError, digest, file_hash, finite_number, read_json


def require(condition, message):
    if not condition:
        raise IntegrationError('SCHEDULE_COMPARISON', message)


def token_counts(usages, expected):
    values = [(u or {}).get('total_tokens') for u in usages]
    known = [n for n in values if type(n) is int and n >= 0]
    return {'reported_tokens': sum(known), 'known_calls': len(known), 'expected_calls': expected,
            'complete': expected is not None and len(known) == expected}


def first_contrast_event(events):
    """从新旧轨迹中读取首个绑定证据的区分事件。"""
    return next((event for event in events if event['phase'] == 'first_qualified_contrast'), None)


def _recorded_object(run, path):
    """读取运行根内的可选诊断工件，容许旧记录缺件。"""
    if not path:
        return {}
    path = Path(path).resolve()
    if not path.is_relative_to(run):
        return {}
    try:
        value = read_json(path)
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _classification_outcome(run, operation):
    version = operation.get('evidence_version')
    if not isinstance(version, str) or not re.fullmatch(r'[0-9a-f]{64}', version):
        return 'classification_recorded'
    path = run / 'workflow' / 'evidence' / 'versions' / (version + '.json')
    arts = _recorded_object(run, path).get('artifacts', {})
    crop_id = operation.get('target_id', '')
    crop = arts.get(crop_id, {})
    classification = arts.get('classification:' + crop_id, {})
    if classification.get('valid') and classification.get('identity_verified'):
        return 'identity_verified'
    reported = _recorded_object(run, classification.get('path'))
    if reported.get('reject'):
        return 'identity_unresolved'
    expected = _recorded_object(run, crop.get('path')).get('feature_id')
    if expected and reported.get('feature_id') and reported['feature_id'] != expected:
        return 'classifier_reported_different_feature'
    return 'classification_recorded'


def _action_outcome(run, tool, event, observation):
    """按本次回执记录结果，供离线反思复核。"""
    detail = event.get('result') or {}
    if detail.get('cache_hit') or observation.get('cache_hit'):
        return 'previous_result_reused'
    if event.get('accepted') is False:
        return 'action_failed'
    if event.get('accepted') is not True:
        return 'outcome_unrecorded'

    if tool == 'scene_check':
        return 'scene_examined'
    if tool == 'presence_check':
        doc = observation.get('result') or {}
        if doc.get('ok') is True and finite_number(doc.get('conf'), 0.5, 1.0):
            if doc.get('present') is True:
                return 'presence_reported'
            if doc.get('present') is False:
                return 'not_seen_in_view'
        return 'presence_unresolved'
    if tool == 'classify_target':
        return _classification_outcome(run, detail)
    bundle = detail.get('bundle') or {}
    if tool == 'segment_instance' and bundle.get('quality') == 'review':
        return 'mask_review_required'
    if tool == 'image_exemplar_audit':
        return 'exemplar_audit_recorded'
    if tool in {'feature_scan', 'detect_region'}:
        relative = (bundle.get('paths') or {}).get('completion')
        completion = Path(bundle['root']) / relative if bundle.get('root') and relative else None
        doc = _recorded_object(run, completion)
        if tool == 'detect_region' and doc.get('count') == 0:
            return 'region_examined_without_replacement'
        if tool == 'feature_scan':
            status, info = scan_coverage(detail) if doc or detail.get('scan_coverage') else ('unavailable', {})
            if status == 'failed':
                return 'scan_failed'
            if status == 'partial':
                return 'scan_partial'
            if status == 'shareable' and info.get('crop_count') == 0:
                return 'scan_completed_without_crops'
    return 'action_completed'


def inspect_run(run):
    run = run.resolve()
    case = run.parent.parent
    proof = read_json(case / 'input_proof.json')
    views = [v['sha256'] for v in proof['views']]
    for v in proof['views']:
        image = (case / v['view']).resolve()
        require(image.is_relative_to(case) and file_hash(image) == v['sha256'], '原图摘要不符')
    job = read_json(run / 'job.json')
    result = {'case': case.name, 'run': run.name, 'input_shas': views, 'job': job['status'],
              'elapsed_seconds': job.get('elapsed_seconds'), 'config_sha256': job.get('config_sha256')}
    if job['status'] != 'completed':
        result['error_code'] = job.get('error_code')
        if (run/'code_signature_end.json').is_file():
            result['code'] = read_json(run/'code_signature_end.json').get('code_signature')
        if (run/'agent_trace.json').is_file():
            partial = read_json(run/'agent_trace.json')
            events = partial.get('events', [])
            result.update(planning_calls=partial.get('planning_requests'),
                          partial_choices=[e.get('execution_choice') for e in events if e['phase']=='thought'],
                          partial_planning_usage=token_counts([e.get('usage') for e in events if e['phase'] in
                              {'thought','planning_error','rejected_proposal','rejected_stop'}],
                              partial.get('planning_requests')))
        return result
    trace = read_json(run / 'agent_trace.json')
    require(trace.get('simulation') is False, '只接受真实任务')
    final = read_json(run / 'final_assessment.json')
    packet = read_json(run / 'final_evidence.json')
    events = trace['events']
    phases = Counter(e['phase'] for e in events)
    a = read_json(run / 'a_prediction.json')
    require([v.get('input_sha256') for v in a['view_predictions']] == views,
            'A 实际推理输入与 case 声明不一致')
    fresh = (proof.get('a_cache_reused') is False and proof.get('previous_visual_artifacts_reused') is False
             and len(a['view_predictions']) == len(views)
             and all(v.get('cache_hit') is False for v in a['view_predictions']))
    criteria = {}
    for fact in packet['facts']:
        if fact['kind'] == 'criterion':
            c = fact['data']
            cid = c['candidate_id']
            criteria.setdefault(cid, {}).setdefault(c['role'], {})[c['feature_id']] = {
                'state': c['state'], 'supported': cid in fact['supports_candidate_ids'],
                'counter': cid in fact['counters_candidate_ids'], 'fact_ref': fact['id']}
    observations = trace['observations']
    planning = [e for e in events if e['phase'] in
                {'thought','planning_error','rejected_proposal','rejected_stop'}]
    resources = read_json(run / 'workflow/resources.json')
    workers = [e for e in resources['entries'].values() if e['pool'] == 'network' and e.get('operation_key')]
    # 早期账本没有 worker 绑定键，故其 worker 调用分母未知。
    has_worker_tracking = 'candidate_audit' in final
    obs_calls = [o for o in observations if not o.get('cache_hit') and o.get('new_requests', 1)]
    tokens = {
        'planning': token_counts([e.get('usage') for e in planning], trace['planning_requests']),
        'observations': token_counts([o.get('usage') for o in obs_calls], trace.get('observation_requests')),
        'final': token_counts([final.get('usage')], 1 if final.get('usage') or final.get('method') == 'model'
                              else None if final.get('model_failure') else 0),
        'worker_http': token_counts([r.get('usage') for r in workers], len(workers) if has_worker_tracking else None)}
    actions = []
    observations_by_id = defaultdict(deque)
    for observation in observations:
        observations_by_id[observation['action']['action_id']].append(observation)
    for event in events:
        if event['phase'] not in {'recompute','action_observation'}:
            continue
        detail = event.get('result') or {}
        observed_rows = observations_by_id.get(event.get('action_id')) if event['phase'] == 'action_observation' else None
        observation = observed_rows.popleft() if observed_rows else {}
        observed = observation.get('action') or {}
        feedback = event.get('feedback') or {}
        tool = event.get('tool') or observed.get('tool')
        actions.append({'tool': tool, 'accepted': event.get('accepted'),
                        'action_id': event.get('action_id') or detail.get('action_id'),
                        'target': detail.get('target_id') or observed.get('args', {}).get('image_id'),
                        'feature': observed.get('args', {}).get('feature_id') or detail.get('feature_id'),
                        'useful': feedback.get('useful'), 'outcome': _action_outcome(run, tool, event, observation),
                        'error_code': event.get('error_code') or feedback.get('failure_code'),
                        'new_features': feedback.get('new_features', []),
                        'new_local_absence': feedback.get('new_local_absence', feedback.get('new_counterevidence', [])),
                        'qualified_counters': feedback.get('qualified_counter_changes', [])})
    thoughts = [e for e in events if e['phase'] == 'thought']
    actions_by_id = {a['action_id']: a for a in actions if a['action_id']}
    semantic_children = {}
    current_goal = None
    for event in events:
        if event['phase'] == 'thought':
            current_goal = event['action_id'] if (event.get('execution_choice') or {}).get('kind') == 'instance_goal' else None
            children = []
        elif current_goal and event['phase'] == 'execution_timing' and event['action_id'] != current_goal:
            children.append(event['action_id'])
        elif current_goal and event['phase'] == 'semantic_return':
            semantic_children[current_goal] = children
            current_goal = None
    graded, no_gain = 0, 0
    no_gain_outcomes = Counter()
    for thought in thoughts:
        ids = [thought['action_id'], *thought.get('physical_partner_ids', [])]
        if thought.get('parallel_observation_id'):
            ids.append(thought['parallel_observation_id'])
        if (thought.get('execution_choice') or {}).get('kind') == 'instance_goal':
            if thought['action_id'] not in semantic_children:
                continue
            ids.extend(semantic_children[thought['action_id']])
        feedback = [actions_by_id.get(aid, {}).get('useful') for aid in ids]
        if all(type(value) is bool for value in feedback):
            graded += 1
            if not any(feedback):
                no_gain += 1
                no_gain_outcomes.update(actions_by_id[aid]['outcome'] for aid in ids)
    audit = final.get('candidate_audit')
    ops = read_json(run / 'workflow/operations.json')['entries'].values()
    head = read_json(run / 'workflow/evidence/head.json')['version']
    require(head == trace['evidence_version'] == packet['evidence_version'], '终态证据头不一致')
    output = read_json(run / 'output.json')
    result.update(cold=fresh, code=(job.get('code_signature') or {}).get('current'),
                  a_folds_sha256=digest([v['fold_rankings'] for v in a['view_predictions']]),
                  head=head, delivery=output['status'], candidate=(output['top_candidate'] or {}).get('id'),
                  review=output['review_required'], final_method=final.get('method'), stop_reason=trace['stop_reason'],
                  planning_calls=trace['planning_requests'], phase_counts=dict(phases),
                  invalid_choices=phases['rejected_stop'] + phases['rejected_proposal'] +
                      sum(bool(e.get('format_repair_eligible')) for e in events if e['phase']=='planning_error'),
                  model_stop_requests=phases['rejected_stop'] + sum(e.get('action_id') in {'finish','review','request_review'}
                      for e in events if e['phase']=='thought'),
                  no_gain_actions=sum(a['useful'] is False for a in actions),
                  attributable_investigations=graded, thoughts_without_action_gain=no_gain,
                  no_gain_action_outcomes=dict(no_gain_outcomes),
                  first_contrast=first_contrast_event(events),
                  tokens=tokens, known_tokens=sum(v['reported_tokens'] for v in tokens.values()),
                  token_complete=all(v['complete'] for v in tokens.values()),
                  worker_http=dict(Counter(r['status'] for r in workers)), closed=resources['closed'],
                  pending_ops=sum(o['status']=='pending' for o in ops),
                  physical_stages=sum(bool(o.get('epoch')) for o in ops),
                  phases_by_choice=dict(Counter((e.get('execution_choice') or {}).get('kind','unrecorded')
                      for e in events if e['phase']=='thought')),
                  actions=actions, criteria=criteria, audit=audit and [
                      {'candidate':r['candidate_id'], **{role:{x['feature_id']:x['state'] for x in r[role]}
                       for role in ('required','supporting')}} for r in audit['candidates']])
    return result


def compare(serial, parallel, historical=None, *, semantic_serial=None, semantic_parallel=None):
    require((semantic_serial is None) == (semantic_parallel is None), '语义对照需要同时提供两个组')
    inputs = {'serial':serial, 'parallel':parallel, 'semantic_serial':semantic_serial,
              'semantic_parallel':semantic_parallel, 'historical':historical}
    paths = {k:Path(p).resolve() for k,p in inputs.items() if p is not None}
    require(len({p.parent.parent for p in paths.values()}) == len(paths), '各组必须使用独立 case')
    rows = {k:inspect_run(p) for k,p in paths.items()}
    require(all(r['input_shas'] == rows['serial']['input_shas'] for r in rows.values()), '同序输入摘要不匹配')
    configs = []
    for arm, path in paths.items():
        if arm == 'historical':
            continue
        row = rows[arm]
        expected = arm in {'parallel','semantic_parallel'}
        cfg = read_json(path.parent.parent / 'config.json')
        require(cfg['agent']['workflow'].pop('physical_parallel') is expected, '组别开关错误')
        if semantic_serial is not None:
            semantic_expected = arm.startswith('semantic_')
            require(cfg['agent']['workflow'].pop('semantic_execution') is semantic_expected,
                    '语义开关与组别不符')
        if row['job'] == 'completed':
            require(row['cold'] and row['closed'] and not row['pending_ops'] and not row['worker_http'].get('pending'),
                    '冷启动或终态账本校验失败')
            if not expected:
                require(row['physical_stages'] == 0, '关闭后仍派发物理并发')
        cfg.pop('sample_id')
        for stage, key in [('a','images'),('b','input'),('agent','image_roots')]:
            del cfg[stage][key]  # 已按每个视野的内容与顺序独立校验
        configs.append(digest(cfg))
    require(len(set(configs)) == 1, '各组除指定开关外配置不同')
    codes = [row.get('code') for arm,row in rows.items() if arm!='historical' and
             (row['job']=='completed' or row.get('code'))]
    require(not codes or (all(codes) and len(set(codes))==1), '当前各组代码签名不同')
    folds = {row['a_folds_sha256'] for arm,row in rows.items() if arm!='historical' and row['job']=='completed'}
    require(len(folds) <= 1, '当前各组实际五折候选不同')
    return {'schema':'schedule-comparison/v1','controlled_config':configs[0],
            'cases':rows}
