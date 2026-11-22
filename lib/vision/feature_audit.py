"""从调查结束时的证据快照生成特征旁路审计。

按有效裁图 ID 关联分类与分割，读取并核对已登记工件的摘要。缺失或变更的工件登记在
报告的 issues 中。审计只写报告与供目检的连通域图，不联网或修改证据、资格与裁决。

主要定义：
- SCHEMA: 当前报告版本 `feature-audit/v2`。
- split_mask: 按连通域拆分 mask 为实例图，不虚构实例。
- build_feature_audit: 汇总有效裁图、分类和分割工件，可按裁图的特征 ID 过滤。

关联：
- 调用方 「lib/application/workflow_entry.py」、「tests/vision/test_feature_audit.py」。
- 直接依赖项目根 `contracts.CropMeta` 与 「lib/runtime/storage.py」。
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from contracts import CropMeta
from TCM_Fusion.lib.runtime.storage import file_hash, finite_number, read_json, write_json


SCHEMA = 'feature-audit/v2'


def _neighbors(binary):
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
    return cv2.filter2D(binary.astype(np.uint8), -1, kernel)


def _thin_zhang_suen(binary, max_iter=80):
    """纯 OpenCV/NumPy 的 Zhang-Suen 细化；不依赖 ximgproc/skimage。"""
    image = (binary > 0).astype(np.uint8)
    for _ in range(max_iter):
        changed = False
        for step in (0, 1):
            neighbors = _neighbors(image)
            p2 = np.roll(image, -1, 0)
            p3 = np.roll(np.roll(image, -1, 0), -1, 1)
            p4 = np.roll(image, -1, 1)
            p5 = np.roll(np.roll(image, 1, 0), -1, 1)
            p6 = np.roll(image, 1, 0)
            p7 = np.roll(np.roll(image, 1, 0), 1, 1)
            p8 = np.roll(image, 1, 1)
            p9 = np.roll(np.roll(image, -1, 0), 1, 1)
            b = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
            a = ((p2 == 0) & (p3 == 1)).astype(np.uint8)
            a += ((p3 == 0) & (p4 == 1)).astype(np.uint8)
            a += ((p4 == 0) & (p5 == 1)).astype(np.uint8)
            a += ((p5 == 0) & (p6 == 1)).astype(np.uint8)
            a += ((p6 == 0) & (p7 == 1)).astype(np.uint8)
            a += ((p7 == 0) & (p8 == 1)).astype(np.uint8)
            a += ((p8 == 0) & (p9 == 1)).astype(np.uint8)
            a += ((p9 == 0) & (p2 == 1)).astype(np.uint8)
            if step == 0:
                condition = ((p2 * p4 * p6) == 0) & ((p4 * p6 * p8) == 0)
            else:
                condition = ((p2 * p4 * p8) == 0) & ((p2 * p6 * p8) == 0)
            remove = (image == 1) & (neighbors >= 2) & (neighbors <= 6) & (a == 1) & condition
            if remove.any():
                image[remove] = 0
                changed = True
        if not changed:
            break
    return image.astype(bool)


def _instance_metrics(component, min_area=20):
    area = int(component.sum())
    if area < min_area:
        return None
    skeleton = _thin_zhang_suen(component)
    n_skeleton = int(skeleton.sum())
    if n_skeleton == 0:
        return None
    neighbors = _neighbors(skeleton)
    endpoints = int(((skeleton) & (neighbors == 1)).sum())
    branches = int(((skeleton) & (neighbors >= 3)).sum())
    distance = cv2.distanceTransform(component.astype(np.uint8), cv2.DIST_L2, 5)
    width = float(distance[skeleton].max() * 2.0) if n_skeleton else 0.0
    ys, xs = np.where(skeleton)
    if len(xs) >= 2:
        points = np.column_stack([xs, ys]).astype(np.float32)
        mean = points.mean(axis=0)
        cov = np.cov((points - mean).T)
        vals, vecs = np.linalg.eigh(cov)
        major = float(np.sqrt(max(vals[-1], 0.0)) * 4.0)
    else:
        major = 0.0
    return {
        'area_px': area,
        'skeleton_px': n_skeleton,
        'endpoints': endpoints,
        'branch_points': branches,
        'length_px': float(n_skeleton),
        'major_axis_px': round(major, 2),
        'width_px': round(width, 2),
        'merged': bool(endpoints > 2 or branches > 0),
    }


def split_mask(mask_path, out_dir=None, min_area=20):
    """按连通域拆分 mask；分叉/多端点组件标记 merged，不强行补边界。"""
    mask_path = Path(mask_path)
    try:
        buffer = np.fromfile(str(mask_path), dtype=np.uint8)
        image = cv2.imdecode(buffer, cv2.IMREAD_GRAYSCALE)
    except (OSError, ValueError):
        image = None
    if image is None:
        return {'mask': str(mask_path), 'error': 'MASK_UNREADABLE', 'instances': []}
    binary = (image > 0).astype(np.uint8)
    count, labels = cv2.connectedComponents(binary, connectivity=8)
    instances = []
    out_dir = Path(out_dir) if out_dir else None
    for index in range(1, count):
        component = (labels == index).astype(np.uint8)
        metrics = _instance_metrics(component, min_area=min_area)
        if metrics is None:
            continue
        item = {'instance_index': len(instances) + 1, **metrics}
        if out_dir is not None:
            out_dir.mkdir(parents=True, exist_ok=True)
            target = out_dir / f'{mask_path.stem}__inst{len(instances) + 1:02d}.png'
            encoded, png = cv2.imencode('.png', (component * 255).astype(np.uint8))
            if not encoded:
                raise OSError(f'实例图编码失败: {target.name}')
            target.write_bytes(png.tobytes())
            item['mask'] = str(target)
        instances.append(item)
    merged = any(row['merged'] for row in instances)
    return {
        'mask': str(mask_path),
        'n_components': len(instances),
        'n_instances': len(instances),
        'merged': merged,
        'status': 'merged' if merged else ('single' if len(instances) == 1 else 'split'),
        'instances': instances,
    }


def _verified_file(artifact, issues):
    path = Path(artifact['path'])
    try:
        if file_hash(path) == artifact['sha256']:
            return path
        error = 'ARTIFACT_CHANGED'
    except OSError:
        error = 'ARTIFACT_UNREADABLE'
    issues.append({'artifact_id': artifact['id'], 'error': error})
    return None


def _candidate(crop, meta, classification, issues):
    det_score = crop.get('det_score', meta.raw.get('det_score'))
    det_score = det_score if finite_number(det_score, maximum=1) else None
    status = 'high_score_unclassified' if det_score is not None and det_score >= 0.8 else 'low_score_unclassified'
    document = {}
    if classification:
        status = 'classification_unresolved'
        path = _verified_file(classification, issues)
        if path is not None:
            try:
                document = read_json(path)
                if not isinstance(document, dict):
                    raise ValueError('classification must be an object')
            except (OSError, ValueError):
                issues.append({'artifact_id': classification['id'], 'error': 'CLASSIFICATION_UNREADABLE'})
                document = {}

    visual_input = document.get('visual_input')
    visual_input = visual_input if isinstance(visual_input, dict) else {}
    feature = document.get('feature_id') or None
    score = document.get('confidence_score')
    target_score = negative_margin = None
    if feature and not document.get('reject') and not classification.get('review_required'):
        matches = feature == meta.feature_id
        status = 'classified_target' if matches else 'classified_non_target'
        if finite_number(score, maximum=1):
            target_score = score if matches else 0.0
            negative_margin = score if matches else -score
    return {
        'crop_id': crop['id'], 'crop_name': meta.crop_id,
        'view': meta.stem, 'source': meta.source_image_path,
        'feature_id': meta.feature_id, 'box_source': crop.get('box_source'),
        'det_score': det_score, 'audit_label': document.get('label') or 'unknown',
        'classified_feature_id': feature, 'confidence_source': document.get('confidence_source'),
        'classification_id': classification['id'] if classification else None,
        'target_score': target_score, 'negative_margin': negative_margin,
        'status': status, 'non_blocking': True, 'reject': document.get('reject'),
        'review_input_mode': visual_input.get('mode'),
        'review_fallback': visual_input.get('fallback'),
    }


def build_feature_audit(snapshot, *, out_dir, feature_ids=None):
    """Project one evidence version. A missing or empty filter audits all valid crops."""
    out = Path(out_dir)
    features = set(feature_ids or ())
    artifacts = {key: row for key, row in snapshot['artifacts'].items() if row.get('valid')}
    candidates = []
    issues = []
    for crop in artifacts.values():
        if crop['kind'] != 'crop':
            continue
        path = _verified_file(crop, issues)
        if path is None:
            continue
        try:
            meta = CropMeta.from_json(path)
        except (AttributeError, OSError, TypeError, ValueError):
            issues.append({'artifact_id': crop['id'], 'error': 'CROP_METADATA_UNREADABLE'})
            continue
        if features and meta.feature_id not in features:
            continue
        classification = artifacts.get('classification:' + crop['id'])
        if classification and (classification['kind'] != 'classification'
                               or crop['id'] not in classification.get('parents', [])):
            classification = None
        candidates.append(_candidate(crop, meta, classification, issues))

    instances = []
    selected = {row['crop_id']: row['feature_id'] for row in candidates}
    for segmentation in artifacts.values():
        if segmentation['kind'] != 'segmentation' or segmentation.get('crop_id') not in selected:
            continue
        if _verified_file(segmentation, issues) is None:
            continue
        crop_id = segmentation['crop_id']
        masks = Path(segmentation['masks']).resolve()
        for parent_id in segmentation.get('parents', []):
            mask = artifacts.get(parent_id)
            if not mask or mask['kind'] != 'segmentation_file':
                continue
            path = Path(mask['path']).resolve()
            # The IGM delivery keeps copied crop pixels under masks/crops.
            if path.parent != masks or path.suffix.lower() != '.png':
                continue
            if _verified_file(mask, issues) is None:
                continue
            result = split_mask(path, out_dir=out / 'instances' / f'{len(instances) + 1:04d}')
            if result.get('error'):
                issues.append({'artifact_id': mask['id'], 'error': result['error']})
                continue
            instances.append({**result, 'artifact_id': mask['id'], 'crop_id': crop_id,
                              'feature_id': selected[crop_id], 'segmentation_quality': segmentation.get('quality')})

    feature_ids = sorted(features or {row['feature_id'] for row in candidates if row['feature_id']})
    summary = {
        'feature_ids': feature_ids,
        'n_candidates': len(candidates),
        'n_input_views': sum(item['kind'] == 'image' for item in artifacts.values()),
        'views_with_candidates': len({row['view'] for row in candidates}),
        'n_target_only': sum(row['review_input_mode'] == 'target_only' for row in candidates),
        'n_with_examples': sum(row['review_input_mode'] == 'target_with_examples' for row in candidates),
        **{status: sum(row['status'] == status for row in candidates) for status in (
            'classified_target', 'classified_non_target', 'classification_unresolved',
            'high_score_unclassified', 'low_score_unclassified')},
        'n_masks': len(instances),
        'n_instances': sum(row['n_instances'] for row in instances),
        'merged_masks': sum(row['merged'] for row in instances),
    }
    report = {
        'schema': SCHEMA, 'evidence_version': snapshot['version'], 'non_blocking': True,
        'feature_ids': feature_ids, 'summary': summary, 'candidates': candidates,
        'instances': instances, 'issues': issues,
    }
    write_json(out / 'feature_audit.json', report)
    return report
