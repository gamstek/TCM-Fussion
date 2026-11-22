"""从当前菜单绑定独立的单裁图身份问题，串行授权只覆盖模型选定的实例。

接口：offer 产生可见选项及主机绑定；allowed 验证显式选择；return_reason 判读单张回执。
调用方：planning_choices 与 react_graph。计费、提交、分类资格仍由普通 classify_target 路径负责。
"""
from itertools import combinations
from pathlib import Path

from TCM_Fusion.lib.evidence.instance_ledger import describe
from TCM_Fusion.lib.runtime.storage import IntegrationError, digest, file_hash, read_json


def binding(workflow, target, snapshot):
    item = describe(target, snapshot)
    box = item['source_bbox']
    if (item['granularity'] != 'instance' or not box or box[2] <= box[0] or box[3] <= box[1]
            or not target.get('valid')):
        return None
    return {'target_id': target['id'], 'instance_id': item['instance_id'],
            'input': digest(item), 'operation_key': workflow.operation_key('classify_target', target),
            'source': item['source_image_sha256'], 'box': box, 'parents': target.get('parents', [])}


def independent(a, b):
    if (a['instance_id'] == b['instance_id'] or a['target_id'] in b['parents']
            or b['target_id'] in a['parents']):
        return False
    if a['source'] != b['source']:
        return True
    x, y = a['box'], b['box']
    return x[2] <= y[0] or y[2] <= x[0] or x[3] <= y[1] or y[3] <= x[1]


def offer(workflow, actions, targets, limit):
    actions = [a for a in actions if a['tool'] == 'classify_target']
    if limit < 2 or len(actions) < 2:
        return {}, {}
    snapshot = workflow.store.current()
    bindings = {}
    for a in actions:
        target = targets.get(a['args']['target_id'])
        if not target or a['args'].get('evidence_version') != snapshot['version']:
            continue
        try:
            row = binding(workflow, target, snapshot)
        except (OSError, ValueError, IntegrationError):
            continue
        if row:
            bindings[a['action_id']] = row
    conflicts = []
    usable = set()
    for left, right in combinations(bindings, 2):
        if independent(bindings[left], bindings[right]):
            usable.update((left, right))
        else:
            conflicts.append([left, right])
    if not usable:
        return {}, {}
    return {'action_ids': [a for a in bindings if a in usable], 'limit': min(limit, len(usable)),
            'conflicts': [p for p in conflicts if set(p) <= usable]}, bindings


def allowed(ids, menu):
    return (isinstance(ids, list) and all(isinstance(k, str) for k in ids)
            and 2 <= len(ids) <= menu.get('limit', 0) and len(set(ids)) == len(ids)
            and set(ids) <= set(menu.get('action_ids', []))
            and not any(set(pair) <= set(ids) for pair in menu.get('conflicts', [])))


def return_reason(event, snapshot, expected):
    """只有已绑定的新回执明确读不清且未改变调查依据时才可继续。"""
    result = event.get('result') or {}
    if (not result.get('accepted') or result.get('status') != 'done' or result.get('cache_hit')
            or result.get('target_id') != expected['target_id']
            or result.get('operation_key') != expected['operation_key']
            or result.get('instance_id') != expected['instance_id']
            or result.get('evidence_version') != snapshot['version']):
        return result.get('error_code') or 'identity_receipt_unavailable'
    identity = snapshot['artifacts'].get('classification:' + expected['target_id'], {})
    if not identity.get('valid') or expected['target_id'] not in identity.get('parents', []):
        return 'identity_receipt_unavailable'
    if identity.get('identity_verified'):
        return 'identity_verified'
    try:
        path = Path(identity['path'])
        if file_hash(path) != identity['sha256']:
            return 'identity_receipt_unavailable'
        doc = read_json(path)
    except (KeyError, OSError, ValueError):
        return 'identity_receipt_unavailable'
    if (doc.get('reject') != 'unreadable' or doc.get('feature_id') or doc.get('second')
            or identity.get('review_required')):
        return 'identity_result_changed'
    feedback = event.get('feedback') or {}
    if (feedback.get('criterion_or_ranking_changed') or feedback.get('candidate_support_changes')
            or feedback.get('qualified_counter_changes') or feedback.get('new_features')
            or feedback.get('capability_progress') or feedback.get('failure_code')):
        return 'investigation_changed'
    return None
