"""单智能体共享检索账本视图：只描述和汇总已有预留与结果，不改变证据、不调用模型。

候选只是图像问题的消费者，不拥有新的图像请求；账本在 resources.json/operations/
observations 之上保持只读，真正的原子执行控制仍由 probe_budget 的预留承担。

主要定义：
- summary: 汇总已结束的 presence_check/feature_scan 探测状态与结果。
- render_shared: 把账本摘要整理成规划器可直接消费的紧凑行。
- `_presence_result` / `_scan_result` / `_observations` 系列：解析观察与探测结果的内部工具。

关联：
- 调用方 「lib/agent/react_graph.py」、「tests/runtime/test_probe_budget.py」。
- 直接依赖 「lib/runtime/storage.py」。
"""
from pathlib import Path

from TCM_Fusion.lib.runtime.storage import read_json


def _observations(state):
    return [o for o in (state or {}).get('observations', []) if isinstance(o, dict)]


def _presence_result(state, probe):
    image, feature = probe.get('image_id'), probe.get('feature_id')
    for observation in reversed(_observations(state)):
        action = observation.get('action') or {}
        if action.get('tool') != 'presence_check':
            continue
        args = action.get('args') or {}
        if args.get('image_id') != image or args.get('feature_id') != feature:
            continue
        result = observation.get('result') or {}
        if observation.get('accepted') and result.get('ok') is True:
            if result.get('present') is True:
                return 'shareable', {'present': True,
                                      'confidence': result.get('conf'),
                                      'reason': result.get('reason')}
            return 'not_observed_local', {'present': False,
                                          'confidence': result.get('conf'),
                                          'reason': result.get('reason')}
        return 'failed', {'error': result.get('error_code') or 'OBSERVATION_FAILED'}
    return 'running', {}


def scan_coverage(operation):
    """报告成功、失败与未尝试的原图分块，兼容旧补扫回执。"""
    if operation.get('status') == 'failed':
        return 'failed', {'error': operation.get('error_code')}
    if operation.get('status') != 'done':
        return 'running', {}
    if operation.get('scan_coverage'):
        coverage = operation['scan_coverage']
        return coverage['status'], coverage

    bundle = operation.get('bundle') or {}
    relative = (bundle.get('paths') or {}).get('completion')
    if not relative or not bundle.get('root'):
        return 'unavailable', {'error': 'SCAN_COVERAGE_UNKNOWN'}
    try:
        doc = read_json(Path(bundle['root']) / relative)
    except (OSError, ValueError):
        return 'unavailable', {'error': 'SCAN_COVERAGE_UNKNOWN'}
    if not isinstance(doc, dict) or not isinstance(doc.get('tiles'), list):
        return 'unavailable', {'error': 'SCAN_COVERAGE_UNKNOWN'}

    tiles = doc['tiles']
    errors = doc.get('errors') or []
    failed = {row.get('tile') for row in errors}
    successful = [box for index, box in enumerate(tiles) if index not in failed]
    planned = doc.get('planned_tiles', None if doc.get('pilot') else tiles)
    unsearched = [box for box in planned if box not in tiles] if planned is not None else None
    incomplete = bool(errors or unsearched or doc.get('partial') or (planned is None and doc.get('pilot')))
    if not successful and not doc.get('count'):
        status = 'failed'
    else:
        status = 'partial' if incomplete else 'shareable'
    return status, {'status': status, 'crop_count': doc.get('count'), 'successful_tiles': successful,
                    'failed_tiles': [box for index, box in enumerate(tiles) if index in failed],
                    'unsearched_tiles': unsearched}


def _scan_result(runtime, probe):
    target_id = probe.get('target_id')
    if not target_id:
        return 'running', {}
    entries = read_json(runtime.workflow.operations).get('entries', {})
    rows = [row for row in entries.values()
            if row.get('tool') == 'feature_scan'
            and (row.get('target_id') == target_id
                 or row.get('feature_id') == probe.get('feature_id')
                 and row.get('image_id') == probe.get('image_id'))]
    if not rows:
        return 'running', {}
    return scan_coverage(rows[-1])


def summary(runtime, state, limit=8):
    """Return recent shared observations for model context only."""
    if runtime is None or getattr(runtime, 'workflow', None) is None:
        return []
    book = read_json(runtime.workflow.budget_path)
    probes = list((book.get('supporting_probes') or {}).values())
    rows = []
    for probe in probes:
        if probe.get('tool') == 'presence_check':
            status, result = _presence_result(state, probe)
        elif probe.get('tool') == 'feature_scan':
            status, result = _scan_result(runtime, probe)
        else:
            continue
        if status == 'running':
            continue
        rows.append({
            'request_key': probe.get('key'),
            'tool': probe.get('tool'),
            'question_group': probe.get('question_group'),
            'image_id': probe.get('image_id'),
            'feature_id': probe.get('feature_id'),
            'candidate_ids': probe.get('candidate_ids') or [],
            'status': status,
            'result': result,
        })
    return rows[-max(1, int(limit)):]


def render_shared(runtime, state, limit=8):
    """Compact text for the existing planner context; no extra model call."""
    rows = summary(runtime, state, limit=limit)
    if not rows:
        return []
    return [{'image_id': row['image_id'], 'feature_id': row['feature_id'],
             'tool': row['tool'], 'question_group': row['question_group'],
             'status': row['status'], 'candidate_ids': row['candidate_ids'],
             'result': row['result']} for row in rows]
