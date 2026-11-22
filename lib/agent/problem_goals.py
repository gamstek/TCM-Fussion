"""把 D2 的定量鉴别问题绑定到一张原图，供现有语义 DAG 在实例出现前接受授权。

主要定义
- offers 从当轮可见动作及当前测量构造有界入口，记录问题、视野、初始证据和允许的第一步。
- candidates 从当前证据辨认精确实例和关联歧义。qualified_facts 只读同一实例已获准的主判量纲。

关联
- planning_choices 投影短编号，semantic_execution 消费授权。工具派发、预算、证据写入及药材裁决各归原组件。
"""
import contracts
from rule_engine import PRIMARY_DIMS, pool_evidence

from TCM_Fusion.lib.agent.measurement_gain import goal_consumers
from TCM_Fusion.lib.decision.comparison_problems import criterion_question
from TCM_Fusion.lib.evidence.instance_ledger import associations, describe
from TCM_Fusion.lib.evidence.purpose_evidence import dimension_use, project
from TCM_Fusion.lib.runtime.storage import digest, finite_number, read_json


def candidates(snapshot, image_id):
    """精确别名共用一个实例，重叠/包含或同坐标不同像素只标记歧义。"""
    groups = {}
    bounds = []
    for crop in snapshot['artifacts'].values():
        if crop.get('kind') != 'crop' or not crop.get('valid') or crop.get('image_id') != image_id:
            continue
        bound = describe(crop, snapshot)
        bounds.append({**bound, 'parents': crop.get('parents', [])})
        if crop.get('box_quality') == 'blank' or not bound['source_bbox'] or bound['granularity'] != 'instance':
            continue
        groups.setdefault(bound['feature_id'], {}).setdefault(bound['instance_id'], []).append((crop, bound))
    conflicts = {iid for pair in associations(bounds) for iid in pair['instance_ids']}
    signatures = {}
    for b in bounds:
        signatures.setdefault(b['instance_id'], set()).add(digest(
            [b['crop_sha256'], b['feature_id'], b['geometry'], b['px_per_um'], b['granularity']]))
    conflicts.update(iid for iid, variants in signatures.items() if len(variants) > 1)
    return groups, conflicts


def single_granularity(identity):
    """旧分类缺少粒度时延用单实例默认值，调用方另行核对裁图本身。"""
    grain = read_json(identity['path']).get('granularity')
    return contracts.effective_granularity('instance', grain) == 'instance'


def qualified_facts(runtime, snapshot, crop, bound, dimension):
    """按原 B 的主判量纲和用途门取可引用数值，其他宽度字段不能单独完成导管问题。"""
    arts = snapshot['artifacts']
    seg = arts.get('seg:' + crop['id'], {})
    measure = arts.get('measurement:' + seg.get('id', ''), {})
    if not measure.get('valid') or measure.get('diagnostic_review') or seg.get('quality') != 'valid':
        return []
    identity = arts.get('classification:' + crop['id'], {})
    scale = arts.get('scale:' + crop['image_id'], {})
    if (not seg.get('valid') or not identity.get('valid') or not identity.get('identity_verified')
            or identity.get('feature_id') != bound['feature_id'] or bound['granularity'] != 'instance'
            or not scale.get('valid') or not finite_number(bound['px_per_um'], minimum=1e-12)
            or bound['px_per_um'] != scale.get('px_per_um') or not single_granularity(identity)):
        return []
    if runtime.workflow.purpose_quality():
        rows = project(runtime, scope_id=seg['id'], measurement_id=measure['id'])
    else:
        rows = (entry['data'] for entry in snapshot['records']
                if entry['scope'] == seg['id'] and measure['id'] in entry['depends_on'])
    facts = []
    for row in rows:
        if (row.get('feature') != bound['feature_id'] or row.get('diagnostic_review')
                or row.get('primary_dimension') not in {'feret_max_um', 'mean_width_um', 'median_width_um'}
                or not finite_number(row.get('primary_value_um'), minimum=1e-12)):
            continue
        card = pool_evidence([row]).get(bound['feature_id'])
        if card and dimension in card.dims:
            facts.append({'measurement_id': measure['id'], 'sha256': measure['sha256'],
                          'instance_ref': bound['instance_id'], 'record_instance_id': row['instance_id'],
                          'primary_dimension': row.get('primary_dimension'), 'value_um': row['primary_value_um']})
    return facts


def _candidate_entry(snapshot, versions, entries):
    """与实例 DAG 同序，优先已核验的精确别名，按其当前前置条件匹配可见入口。"""
    arts = snapshot['artifacts']
    def verified(crop, bound):
        identity = arts.get('classification:' + crop['id'], {})
        return identity.get('valid') and identity.get('identity_verified') and identity.get('feature_id') == bound['feature_id']
    crop, bound = min(versions, key=lambda pair: (not verified(*pair), pair[0]['id']))
    seg = arts.get('seg:' + crop['id'], {})
    if not verified(crop, bound):
        tool, tid = 'classify_target', crop['id']
    elif seg.get('valid'):
        tool, tid = 'measure_instances', seg['id']
    else:
        tool, tid = 'segment_instance', crop['id']
    return next((a for a, t in entries if a['tool'] == tool and t['id'] == tid), None)


def scan_affordable(max_tiles, remaining):
    """一次逻辑补扫还需要各分块的请求额度，remaining 已扣除收尾预留。"""
    return (remaining.get('recompute', 0) > 0 and remaining.get('network', 0) >= max_tiles
            and remaining.get('total', 0) >= 1 + max_tiles)


def measurement_useful(group, task, consumers=None):
    """沿用测量收益表，问题授权只能为其绑定的判据消费者触发付费测量。"""
    beneficiaries = goal_consumers(group, task)
    if beneficiaries is None:
        return True
    return bool(beneficiaries) if consumers is None else not set(consumers).isdisjoint(beneficiaries)


def offers(runtime, actions, targets, criteria, previous=(), remaining=None, *,
           measurement_actions=None, measurement_groups=None):
    """从实际送模动作建立入口。有现成数值时只提供本地读取，同一版本已回交的问题不重复出售。"""
    workflow = runtime.workflow
    snapshot = workflow.store.current()
    by_scope = {}
    for action in actions:
        if action['tool'] not in {'feature_scan', 'classify_target', 'segment_instance', 'measure_instances'}:
            continue
        target = targets.get((action.get('args') or {}).get('target_id') or action.get('target_id'), {})
        image = target.get('image_id')
        if image not in runtime.images:
            continue
        if action['tool'] == 'feature_scan':
            feature = target['feature_id']
        else:
            crop = targets.get(target.get('crop_id'), target) if target.get('kind') == 'segmentation' else target
            if crop.get('kind') != 'crop':
                continue
            feature = contracts.CropMeta.from_json(crop['path']).feature_id
        by_scope.setdefault((image, feature), []).append((action, target))
    questions = {}
    for criterion in criteria:
        if (criterion.get('quant') or {}).get('dim') in PRIMARY_DIMS:
            q = criterion_question(criterion)
            question = questions.setdefault(q['problem_id'], {**q, 'consumers': []})
            if criterion['candidate_id'] not in question['consumers']:
                question['consumers'].append(criterion['candidate_id'])
    features = {q['feature_id'] for q in questions.values()}
    views = {image for image, fid in by_scope if fid in features}
    for record in snapshot['records']:
        if record['data'].get('feature') in features:
            views.add(snapshot['artifacts'][record['scope']]['image_id'])
    rows = []
    for image, source in runtime.images.items():
        scale = snapshot['artifacts'].get('scale:' + image, {})
        if image not in views or not scale.get('valid') or not finite_number(scale.get('px_per_um'), minimum=1e-12):
            continue
        groups, conflicts = candidates(snapshot, image)
        reusable = {}
        for q in questions.values():
            fid = q['feature_id']
            candidates_here = groups.get(fid, {})
            entries = by_scope.get((image, fid), ())
            if not entries and not candidates_here:
                continue
            if (q['problem_id'], image, snapshot['version']) in previous:
                continue
            key = (fid, q['dimension'])
            if key not in reusable:
                reusable[key] = any(qualified_facts(runtime, snapshot, crop, bound, q['dimension'])
                                    for iid, versions in candidates_here.items() if iid not in conflicts
                                    for crop, bound in versions)
            entry = None
            if not reusable[key]:
                if len(candidates_here) > 1 or conflicts.intersection(candidates_here):
                    continue
                if candidates_here:
                    entry = _candidate_entry(snapshot, next(iter(candidates_here.values())), entries)
                else:
                    entry = next((a for a, _ in entries if a['tool'] == 'feature_scan'), None)
                if not entry:
                    continue
                group_key = (measurement_actions or {}).get(entry['action_id'])
                group = (measurement_groups or {}).get(group_key)
                if group and not measurement_useful(group, 'measure_' + dimension_use(q['dimension']), q['consumers']):
                    continue
            target_id = ((entry.get('args') or {}).get('target_id') or entry.get('target_id')) if entry else None
            tiles = targets[target_id]['max_tiles'] if entry and entry['tool'] == 'feature_scan' else 0
            if tiles and remaining is not None and not scan_affordable(tiles, remaining):
                continue
            row = {**q, 'image_id': image, 'source_sha256': source['sha256'], 'scale_sha256': scale['sha256'],
                   'version': snapshot['version'], 'entry_action_id': entry['action_id'] if entry else None,
                   'entry_tool': entry['tool'] if entry else 'reuse', 'entry_target': target_id,
                   'max_tiles': tiles}
            row['choice_id'] = 'problem-goal:' + digest(row)[:24]
            rows.append(row)
    return rows
