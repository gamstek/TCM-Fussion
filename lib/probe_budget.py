"""Candidate-bound supporting probes -> shared resources.json reservations.

Consumed by proposal builders, LocalToolbox and WorkflowSession. This is control
metadata, not microscopic evidence. Candidate names never enter visual prompts.
"""
from .storage import digest, read_json


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
        dependency = runtime.images.get(image, {}).get('sha256')
    else:
        target = target or workflow.targets().get(args.get('target_id'), {})
        image, fid = target.get('image_id'), target.get('feature_id')
        question = {'max_tiles': target.get('max_tiles'), 'roi': target.get('roi'),
                    'px_per_um': target.get('px_per_um')}
        dependency = workflow.dependency_fingerprint(action['tool'], target) if target else None
    active = set(runtime.candidate_interest()['ids'])
    roles = []
    for row in runtime.kb.candidates():
        if row['id'] not in active:
            continue
        for criterion in row['criteria']:
            if criterion['feature'] != fid:
                continue
            if action['tool'] == 'presence_check' and (criterion.get('morph') or '') != question:
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
                          'question': question, 'dependency': dependency,
                          'model': runtime.config_identity, 'kb': runtime.kb_sha256}),
            'tool': action['tool'],
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
