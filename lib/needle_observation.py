"""针晶旁路审计与 mask 实例拆分：只在 workflow 结束后读取已有工件，生成非阻断审计。

不参与 rule decide，不改变 det_score>=0.8 的 mask/measure 路径；不调用新模型、
不补造不可见实例；源图、crop、mask 缺失时记录 unknown，不抛出到主流程。

主要定义：
- SCHEMA: 审计产物版本标识 `needle-audit/v1`。
- split_mask: 按连通域拆分 mask 为实例图，不虚构实例。
- build_feature_audit: 主链使用的特征旁路审计入口。
- build_needle_audit: 旧针晶审计入口，写旁路 sidecar。
- `_thin_zhang_suen` / `_instance_metrics` / `_crop_records` 系列：细化、度量与 crop 收集的内部工具。

关联：
- TCM_Fusion.lib.storage: read_json/write_json/file_hash 读写审计产物。
- TCM_Fusion.lib.workflow_entry: feature_audit 开启时调用 build_feature_audit。
- cv2 / numpy: 形态学细化与连通域统计。
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from .storage import read_json, write_json, file_hash


SCHEMA = 'needle-audit/v1'


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
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
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
            cv2.imwrite(str(target), (component * 255).astype(np.uint8))
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


def _crop_records(directory, target_ids):
    target_ids = {str(x) for x in target_ids}
    directory = Path(directory)
    operations = read_json(directory / 'workflow' / 'operations.json').get('entries', {})
    classifications = {}
    masks = {}
    for operation in operations.values():
        bundle = operation.get('bundle') or {}
        root = Path(bundle.get('root') or '')
        if not bundle.get('root') or not root.is_dir():
            continue
        if operation.get('tool') == 'classify_target':
            for path in root.rglob('classification.json'):
                try:
                    value = read_json(path)
                except (OSError, ValueError):
                    continue
                if not isinstance(value, dict) or not value.get('crop_id'):
                    continue
                classifications[value['crop_id']] = value
        if operation.get('tool') == 'segment_instance':
            for path in root.rglob('mask.png'):
                if path.is_file():
                    masks[str(path)] = path
            for path in (root / 'segmentation' / 'masks').glob('*.png') if (root / 'segmentation' / 'masks').is_dir() else []:
                if path.is_file():
                    masks[str(path)] = path
    candidates = []
    for operation in operations.values():
        if operation.get('tool') != 'locate_feature':
            continue
        bundle = operation.get('bundle') or {}
        root = Path(bundle.get('root') or '')
        if not bundle.get('root') or not root.is_dir():
            continue
        for path in root.rglob('*.json'):
            try:
                value = read_json(path)
            except (OSError, ValueError):
                continue
            if not isinstance(value, dict):
                continue
            crop_id = value.get('crop_id')
            if not crop_id or not value.get('feature_id'):
                continue
            det_score = value.get('det_score')
            classification = classifications.get(crop_id) or {}
            label = classification.get('feature_id') or ''
            score = classification.get('confidence_score')
            if classification:
                target_score = float(score or 0) if label in target_ids else 0.0
                negative_margin = target_score - (float(score or 0) if label and label not in target_ids else 0.0)
                status = 'classified_target' if label in target_ids else 'classified_non_target'
                audit_label = classification.get('label')
            else:
                target_score = None
                negative_margin = None
                high = isinstance(det_score, (int, float)) and det_score >= 0.8
                status = 'high_score_unclassified' if high else 'low_score_unclassified'
                audit_label = 'unknown'
            try:
                numeric_det = float(det_score) if det_score is not None else None
            except (TypeError, ValueError):
                numeric_det = None
            row = {
                'crop_id': crop_id,
                'view': Path(str(value.get('source_image_path') or '')).stem,
                'source': value.get('source_image_path'),
                'feature_id': value.get('feature_id'),
                'det_score': numeric_det,
                'audit_label': audit_label,
                'target_score': target_score,
                'negative_margin': negative_margin,
                'target_features': sorted(target_ids),
                'status': status,
                'non_blocking': True,
            }
            # 兼容旧 needle_audit 消费方。
            row['needle_score'] = target_score if 'needle_crystal' in target_ids else None
            candidates.append(row)
    # hybrid_scan 的 GLM crop 侧车不在 bundle.files 的顶层，直接读固定目录。
    for operation in operations.values():
        if operation.get('tool') != 'hybrid_scan':
            continue
        bundle = operation.get('bundle') or {}
        root = Path(bundle.get('root') or '')
        folder = root / 'features' / 'crops'
        if not bundle.get('root') or not folder.is_dir():
            continue
        for path in sorted(folder.glob('*.json')):
            try:
                value = read_json(path)
            except (OSError, ValueError):
                continue
            if not isinstance(value, dict):
                continue
            crop_id = value.get('crop_id')
            if not crop_id or not value.get('feature_id'):
                continue
            det_score = value.get('det_score')
            classification = classifications.get(crop_id) or {}
            label = classification.get('feature_id') or ''
            score = classification.get('confidence_score')
            if classification:
                target_score = float(score or 0) if label in target_ids else 0.0
                negative_margin = target_score - (float(score or 0) if label and label not in target_ids else 0.0)
                status = 'classified_target' if label in target_ids else 'classified_non_target'
                audit_label = classification.get('label')
            else:
                target_score = None
                negative_margin = None
                high = isinstance(det_score, (int, float)) and det_score >= 0.8
                status = 'high_score_unclassified' if high else 'low_score_unclassified'
                audit_label = 'unknown'
            try:
                numeric_det = float(det_score) if det_score is not None else None
            except (TypeError, ValueError):
                numeric_det = None
            candidates.append({'crop_id': crop_id,
                'view': Path(str(value.get('source_image_path') or '')).stem,
                'source': value.get('source_image_path'), 'feature_id': value.get('feature_id'),
                'det_score': numeric_det, 'audit_label': audit_label,
                'target_score': target_score, 'negative_margin': negative_margin,
                'target_features': sorted(target_ids), 'status': status, 'non_blocking': True,
                'needle_score': target_score if 'needle_crystal' in target_ids else None})
    unique = {}
    for path in masks:
        try:
            key = file_hash(path)
        except (OSError, ValueError):
            key = str(path)
        unique.setdefault(key, masks[path])
    return candidates, list(unique.values())


def build_feature_audit(directory, *, feature_ids=None, out_dir=None):
    """通用 feature 旁路审计；feature_ids 为空时只审计已有 locate/crop 分类。"""
    directory = Path(directory)
    feature_ids = [str(x) for x in (feature_ids or ['needle_crystal'])]
    out = Path(out_dir) if out_dir else directory / 'feature_audit'
    out.mkdir(parents=True, exist_ok=True)
    candidates, masks = _crop_records(directory, feature_ids)
    instances = []
    for mask in masks:
        instances.append(split_mask(mask, out_dir=out / 'instances'))
    summary = {
        'feature_ids': feature_ids,
        'n_candidates': len(candidates),
        'classified_target': sum(row['status'] == 'classified_target' for row in candidates),
        'classified_non_target': sum(row['status'] == 'classified_non_target' for row in candidates),
        'high_score_unclassified': sum(row['status'] == 'high_score_unclassified' for row in candidates),
        'low_score_unclassified': sum(row['status'] == 'low_score_unclassified' for row in candidates),
        'n_masks': len(instances),
        'n_instances': sum(row.get('n_instances') or 0 for row in instances),
        'merged_masks': sum(bool(row.get('merged')) for row in instances),
    }
    result = {
        'schema': 'feature-audit/v1',
        'non_blocking': True,
        'feature_ids': feature_ids,
        'note': '旁路审计；不参与 rule decide，不阻断 det_score>=0.8 的 mask/measure',
        'summary': summary,
        'candidates': candidates,
        'instances': instances,
    }
    write_json(out / 'feature_audit.json', result)
    return result


def build_needle_audit(directory, *, out_dir=None):
    """兼容旧名：针晶专用 wrapper。"""
    result = build_feature_audit(directory, feature_ids=['needle_crystal'],
                                 out_dir=out_dir or (Path(directory) / 'needle_audit'))
    write_json(Path(directory) / 'needle_audit' / 'needle_audit.json', result)
    return result
