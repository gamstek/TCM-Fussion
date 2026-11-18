"""候选绑定的 supporting 探测到共享 resources.json 预留：只是控制元数据，不是显微证据。

候选名从不进入视觉提示词；预算按单个动作计，不存在 plan 级 feature。

主要定义：
- enabled: 判断当前运行时是否启用 supporting 探测预算。
- describe: 绑定可信知识库与物理查询身份，不信任调用方自报角色。
- check_reservation: 校验探测预留是否与已记账资源一致。
- available: 判断动作/目标是否仍有可用预留额度。

关联：
- 调用方 「lib/runtime/resource_budget.py」、「lib/tools/agent_tools.py」、「lib/workflow/evidence_runtime.py」、「lib/workflow/workflow_session.py」、「tests/integration/test_decision_view.py」、「tests/runtime/test_probe_budget.py」。
- 直接依赖 「lib/runtime/storage.py」。
"""
from TCM_Fusion.lib.runtime.storage import digest, read_json


def enabled(runtime):
    # Supporting budgets apply to individual actions; there is no plan feature.
    return getattr(runtime, 'workflow', None) is not None


def describe(runtime, action, target=None):
    """Trusted KB binding and physical-query identity; never trust caller roles."""
    if not enabled(runtime) or action['tool'] not in {'presence_check', 'feature_scan'}:
        return None
    workflow = runtime.workflow
    args = action['args']
    if action['tool'] == 'presence_check':
        image, fid = args.get('image_id'), args.get('feature_id')
        question = args.get('morph_hint', '')
        # Presence answers one physical question: is this feature visible in
        # this image? Candidate-specific morph wording is interpretation, not
        # a new image request. It is kept as consumer metadata.
        question_group = 'presence'
        dependency = runtime.images.get(image, {}).get('sha256')
    else:
        target = target or workflow.targets().get(args.get('target_id'), {})
        image, fid = target.get('image_id'), target.get('feature_id')
        question = {'max_tiles': target.get('max_tiles'), 'roi': target.get('roi'),
                    'px_per_um': target.get('px_per_um')}
        question_group = 'feature_scan'
        dependency = workflow.dependency_fingerprint(action['tool'], target) if target else None
    active = set(runtime.candidate_interest()['ids'])
    roles = []
    for row in runtime.kb.candidates():
        if row['id'] not in active:
            continue
        for criterion in row['criteria']:
            if criterion['feature'] != fid:
                continue
            roles.append((row['id'], criterion['role']))
    cids = sorted({cid for cid, role in roles if role == 'supporting'})
    # An active required criterion stays a required investigation, never relabelled
    # by a model to escape a supporting quota or by a supporting pair to block it.
    if not cids or any(role == 'required' for _, role in roles):
        return None
    return {'key': digest({'image': runtime.images.get(image, {}).get('sha256'),
                          'feature': fid, 'tool': action['tool'],
                          'tool_version': runtime.tools.get(action['tool']).version,
                          'question_group': question_group, 'dependency': dependency,
                          'model': runtime.config_identity, 'kb': runtime.kb_sha256}),
            'tool': action['tool'], 'question_group': question_group,
            'question': question, 'target_id': (target or {}).get('id') if action['tool'] != 'presence_check' else None,
            'image_id': image, 'feature_id': fid, 'candidate_ids': cids,
            'total_limit': int(workflow.config.get('supporting_probe_total', 6)),
            'candidate_limit': int(workflow.config.get('supporting_probe_per_candidate', 2))}


def check_reservation(book, probe):
    """Return an error code; called again under the resource reservation lock."""
    entries = book.get('supporting_probes', {})
    if probe['key'] in entries:
        return 'PROBE_ALREADY_ATTEMPTED'
    if len(entries) >= probe['total_limit']:
        return 'SUPPORTING_BUDGET'
    for cid in probe['candidate_ids']:
        used = sum(cid in row['candidate_ids'] for row in entries.values())
        if used >= probe['candidate_limit']:
            return 'SUPPORTING_CANDIDATE_BUDGET'
    return None


def available(runtime, action, target=None):
    probe = describe(runtime, action, target)
    return probe is None or check_reservation(read_json(runtime.workflow.budget_path), probe) is None
