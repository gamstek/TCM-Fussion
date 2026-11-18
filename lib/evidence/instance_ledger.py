"""任务内实例索引与语义尝试键（D1）。

消费 contracts.CropMeta 和 EvidenceStore 快照；只索引现有工件，不授予身份/量纲资格。
WorkflowSession 在 operations 锁内持久化索引并预约尝试，原图和证据版本只读。
精确原图坐标用于 D1 关联；高 IoU/包含关系的歧义合并留给 D2。

主要定义：
- describe。返回源图 SHA/原图框身份；文件名、候选名不参与身份。
- attempt。语义尝试键不含工件名称/路径；输入、方法或有效修复参数改变才允许重开。
- project。由证据和 operations 重建内部索引；attempt 状态直接引用唯一执行账本。
- associations。高重叠/包含/父crop只提出关联歧义；不自动合并、不赋予新资格。

关联：
- 调用方 「lib/agent/investigation_control.py」、「lib/agent/reflection_controller.py」、「lib/agent/semantic_execution.py」、「lib/workflow/workflow_runtime.py」、「lib/workflow/workflow_session.py」、「tests/evidence/test_instance_ledger.py」。
- 直接依赖 「lib/runtime/storage.py」。
"""
from pathlib import Path

import contracts
from TCM_Fusion.lib.runtime.storage import digest, file_hash


INSTANCE_TOOLS = frozenset({'classify_target', 'segment_instance', 'measure_instances'})


def describe(target, snapshot):
    """返回源图 SHA/原图框身份；文件名、候选名不参与身份。

    缺坐标时只绑定该 crop 的内容，避免把整幅图中的多个未知目标误合并。
    """
    arts = snapshot['artifacts']
    crop = target
    if target.get('kind') == 'segmentation':
        current = arts.get(target.get('id'), target)
        crop = arts.get(current.get('crop_id'), {})
    if crop.get('kind') != 'crop':
        return None
    meta = contracts.CropMeta.from_json(crop['path'])
    image = arts.get(crop.get('image_id'), {})
    source = image.get('sha256') or file_hash(meta.source_image_path)
    box = crop.get('source_tight_bbox') or meta.tight_box
    box = [float(x) for x in box] if box else None
    pixels = file_hash(meta.png)
    identity = {'source_image_sha256': source, 'source_bbox': box}
    if box is None:
        identity['unmapped_crop'] = crop['id']
    return {'instance_id': 'inst:' + digest(identity), **identity,
            'crop_id': crop['id'], 'crop_sha256': pixels,
            'geometry': {'tight': box, 'local': meta.tight_box_local,
                         'loose': meta.loose_box, 'pad': meta.raw.get('pad_info')},
            'feature_id': meta.feature_id, 'granularity': meta.granularity,
            'px_per_um': meta.px_per_um}


def attempt(tool, target, snapshot, method, parameters=None):
    """语义尝试键不含工件名称/路径；输入、方法或有效修复参数改变才允许重开。"""
    if tool not in INSTANCE_TOOLS:
        return None
    target = snapshot['artifacts'].get(target.get('id'), target)
    bound = describe(target, snapshot)
    if bound is None:
        return None
    inputs = {'pixels': bound['crop_sha256'], 'geometry': bound['geometry'],
              'feature': bound['feature_id'], 'granularity': bound['granularity']}
    arts = snapshot['artifacts']
    if tool != 'classify_target':
        inputs['scale'] = arts.get('scale:' + target['image_id'], {}).get('sha256')
        inputs['px_per_um'] = bound['px_per_um']
        identity = arts.get('classification:' + bound['crop_id'], {})
        # 名称不同但内容相同的身份工件不制造新尝试。
        inputs['identity'] = {k: identity.get(k) for k in (
            'sha256', 'valid', 'identity_verified', 'review_required')}
    if tool == 'measure_instances':
        inputs['mask_files'] = sorted(file_hash(p) for p in Path(target['masks']).glob('*.png'))
        inputs['segmentation'] = target.get('sha256')
        inputs['quality'] = target.get('quality')
        assessment = arts.get(target.get('assessment_id'), {})
        inputs['assessment'] = assessment.get('sha256')
    return {'instance_id': bound['instance_id'],
            'semantic_key': digest({'instance': bound['instance_id'], 'purpose': tool,
                                    'method': method, 'inputs': inputs,
                                    'parameters': parameters or {}})}


def project(snapshot, operations):
    """由证据和 operations 重建内部索引；attempt 状态直接引用唯一执行账本。

    不产生新的支持计数，不写入规划上下文，不替代 EvidenceStore 的 CAS。
    """
    instances = {}
    by_crop = {}
    bounds = []
    for key, item in snapshot['artifacts'].items():
        if item.get('kind') != 'crop' or not item.get('valid'):
            continue
        bound = describe(item, snapshot)
        bounds.append({**bound, 'parents': item.get('parents', [])})
        iid = bound['instance_id']; by_crop[key] = iid
        row = instances.setdefault(iid, {
            'instance_id': iid, 'source_image_sha256': bound['source_image_sha256'],
            'source_bbox': bound['source_bbox'],
            'association': 'exact_coordinates' if bound['source_bbox'] else 'unmapped',
            'crop_versions': [], 'classifications': [], 'mask_versions': [],
            'measurements': [], 'attempts': []})
        row['crop_versions'].append({'artifact_id': key, 'sha256': item['sha256'],
                                     'pixels_sha256': bound['crop_sha256']})
    for key, item in snapshot['artifacts'].items():
        kind = item.get('kind')
        if not item.get('valid') or kind not in {'classification', 'segmentation', 'measurement'}:
            continue
        stack = list(item.get('parents', [])); visited = set(); owners = set()
        while stack:
            parent = stack.pop()
            if parent in visited:
                continue
            visited.add(parent)
            if parent in by_crop:
                owners.add(by_crop[parent])
            stack.extend(snapshot['artifacts'].get(parent, {}).get('parents', []))
        field = {'classification': 'classifications', 'segmentation': 'mask_versions',
                 'measurement': 'measurements'}[kind]
        for iid in owners:
            instances[iid][field].append({'artifact_id': key, 'sha256': item['sha256']})
    for key, operation in operations.get('entries', {}).items():
        iid = operation.get('instance_id')
        if iid and iid not in instances:
            instances[iid] = {'instance_id': iid, 'association': 'historical_attempt',
                              'crop_versions': [], 'classifications': [], 'mask_versions': [],
                              'measurements': [], 'attempts': []}
        if iid:
            instances[iid]['attempts'].append({
                'operation_key': key, 'semantic_key': operation['semantic_key'],
                'tool': operation['tool'], 'status': operation['status'],
                'error_code': operation.get('error_code')})
    return {'schema': 'instance-ledger/d2', 'evidence_version': snapshot['version'],
            'instances': instances, 'associations': associations(bounds)}


def associations(bounds):
    """高重叠/包含/父crop只提出关联歧义；不自动合并、不赋予新资格。"""
    out = []
    if len(bounds) < 2:
        return out
    from geom import iou_box
    for index, left in enumerate(bounds):
        for right in bounds[index+1:]:
            if left['source_image_sha256'] != right['source_image_sha256']:
                continue
            a, b = left.get('source_bbox'), right.get('source_bbox')
            if not a or not b:
                continue
            overlap = iou_box(a, b)
            containment = (a[0]<=b[0] and a[1]<=b[1] and a[2]>=b[2] and a[3]>=b[3]) or (
                b[0]<=a[0] and b[1]<=a[1] and b[2]>=a[2] and b[3]>=a[3])
            lineage = left['crop_id'] in right.get('parents', []) or right['crop_id'] in left.get('parents', [])
            identity_conflict = left['feature_id'] != right['feature_id']
            same = left['instance_id'] == right['instance_id']
            if (same and not identity_conflict) or not (overlap >= .85 or containment or lineage):
                continue
            out.append({'state': 'ambiguous', 'crop_ids': [left['crop_id'],right['crop_id']],
                        'instance_ids': sorted({left['instance_id'],right['instance_id']}),
                        'iou': overlap, 'containment': containment, 'direct_lineage': lineage,
                        'identity_conflict': identity_conflict, 'auto_merged': False,
                        'qualification_effect': 'none',
                        'next_step': 'ReAct may inspect either crop, change investigation or request review'})
    return out
