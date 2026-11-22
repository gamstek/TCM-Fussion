"""为已授权的单 crop 视觉身份复核选对照、拼图和校验可见部位。

主要定义：
- load_review_bank / review_plan：按易混组绑定人工确认的示例与当前 crop 的 CNN 前两类。
- prepare_review：将两例放入原请求图幅，保留目标原生像素，必要时退回目标图。
- finish_review：将模型报告归一为可见结构、非特征或身份待定。

关联：workflow_session 和 workflow_backend 选输入，classify_worker 在现有 HTTP
请求中复用 classify_crop。示例和报告部位均为身份复核信息，不提供测量坐标。
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from TCM_Fusion.lib.runtime.storage import finite_number, project_path


CUES = {
    'vessel': ('vessel_pattern', '管壁上可辨的环纹、网纹、螺纹或有序纹孔'),
    'fiber': ('fiber_wall_lumen', '连续纵向胞壁与狭长胞腔'),
    'stone_cell': ('stone_wall_lumen', '封闭厚壁围成的小胞腔，或壁上可辨层纹、孔沟'),
    'secretory_tissue': ('secretory_contents', '位于组织腔或乳管内的油滴、颗粒或分泌物'),
    'starch_grain': ('starch_hilum_layers', '颗粒内的脐点或层纹'),
    'cluster_crystal': ('radiating_facets', '相连晶棱自中心放射聚集'),
    'single_crystal': ('single_facets', '同一独立晶体上连续清晰的晶面与棱角'),
    'needle_crystal': ('needle_tips', '细长晶体具有可辨尖端，未见成形的胞壁与胞腔'),
    'non_glandular_hair': ('hair_structure', '可辨的毛体及尖端、基部或分叉形态'),
    'stoma': ('guard_cells', '一对保卫细胞及之间的气孔'),
    'epidermis_cell': ('epidermal_boundaries', '连续片中多个贴合的表皮细胞垂周壁'),
    'hypha': ('hypha_structure', '丝体上的分枝或明确横隔'),
    'pollen_grain': ('pollen_wall', '花粉外壁雕纹、萌发孔或孔沟'),
}
PARTS = frozenset(('左上', '上中', '右上', '左中', '中央', '右中', '左下', '下中', '右下'))
_HASH = re.compile(r'[0-9a-f]{64}')


def load_review_bank(path):
    """固定本次任务的示例目录，格式错误及不可用的对照资产可退回目标图。"""
    path = project_path(path)
    try:
        raw = path.read_bytes()
        doc = json.loads(raw)
    except (OSError, ValueError):
        return None
    if not isinstance(doc, dict) or doc.get('schema') not in ('identity-review-bank/v1', 'identity-review-bank/v2'):
        return None
    rows, pairs = doc.get('examples'), doc.get('pairs')
    if not isinstance(rows, list) or not isinstance(pairs, list):
        return None
    approved = []
    for row in rows:
        if (not isinstance(row, dict) or row.get('review_status') != 'human_reviewed'
                or not isinstance(row.get('reviewed_by'), str) or not row['reviewed_by'].strip()
                or not isinstance(row.get('feature_id'), str) or row['feature_id'] not in CUES
                or not all(isinstance(row.get(key), str) and row[key] for key in
                           ('file', 'id', 'sha256', 'source_sha256'))
                or not all(_HASH.fullmatch(row[key]) for key in ('sha256', 'source_sha256'))):
            continue
        try:
            asset = (path.parent / row['file']).resolve()
        except (OSError, RuntimeError):
            continue
        if not asset.is_relative_to(path.parent) or Path(row['file']).is_absolute():
            continue
        approved.append({key: row[key] for key in
                         ('id', 'feature_id', 'sha256', 'source_sha256', 'review_status', 'reviewed_by')}
                        | {'path': str(asset)})
    approved_by_id = {row['id']: row for row in approved}
    if len(approved_by_id) != len(approved):
        return None
    allowed_pairs = {}
    pair_ids = set()
    for pair in pairs:
        features = pair.get('features') if isinstance(pair, dict) else None
        if (not isinstance(features, list) or len(features) != 2
                or not all(isinstance(fid, str) and fid in CUES for fid in features)
                or features[0] == features[1]):
            continue
        key = frozenset(features)
        if doc['schema'] == 'identity-review-bank/v1':
            # 历史库仅按类别登记示例，保留原检索口径。
            allowed_pairs[key] = {'id': None, 'examples': [row for row in approved if row['feature_id'] in key]}
            continue
        ids = pair.get('example_ids')
        pair_id = pair.get('id')
        if (not isinstance(pair_id, str) or not pair_id.strip()
                or not isinstance(ids, list) or len(ids) != 2
                or not all(isinstance(i, str) and i in approved_by_id for i in ids)):
            continue
        examples = [approved_by_id[i] for i in ids]
        if frozenset(row['feature_id'] for row in examples) != key:
            continue
        if key in allowed_pairs or pair_id in pair_ids:
            return None
        pair_ids.add(pair_id)
        allowed_pairs[key] = {'id': pair_id, 'examples': examples}
    return {'sha256': hashlib.sha256(raw).hexdigest(), 'examples': approved, 'pairs': allowed_pairs}


def review_plan(bank, cnn, crop_feature, box_source, excluded_sources, *, margin_limit):
    """保留前两类及分差，首位冲突优先，其他时只对照接近的两类。"""
    top = []
    rows = (cnn or {}).get('topk') or []
    for row in rows[:2]:
        if (not isinstance(row, dict) or not isinstance(row.get('feature_id'), str)
                or not finite_number(row.get('prob'), minimum=0, maximum=1)):
            break
        top.append({'feature_id': row['feature_id'], 'prob': row['prob']})
    first, second = ([row['feature_id'] for row in top] + [None, None])[:2]
    margin = top[0]['prob'] - top[1]['prob'] if len(top) == 2 else None
    prior = crop_feature if crop_feature in CUES else None
    conflict = bool(prior and first in CUES and prior != first)
    close = margin is not None and 0 <= margin <= margin_limit + 1e-12
    positive = prior or next((fid for fid in (first, second) if fid in CUES), None)
    negative = first if conflict else second if close and second in CUES and second != positive else None
    basis = ('detector_conflict' if box_source == 'yolo' else 'crop_feature_conflict') if conflict else (
        'small_cnn_margin' if negative else 'no_close_alternative')
    doc = {'cnn_top2': top, 'cnn_margin': margin, 'margin_limit': margin_limit,
           'box_source': box_source, 'crop_feature': prior, 'detector_conflict': conflict if box_source == 'yolo' else None,
           'positive_feature': positive, 'negative_feature': negative, 'selection_basis': basis,
           'bank_sha256': bank['sha256'] if bank else None, 'references': []}
    if not positive or not negative:
        return {**doc, 'fallback': 'no_close_alternative'}
    pair = bank['pairs'].get(frozenset((positive, negative))) if bank else None
    if not pair:
        return {**doc, 'fallback': 'no_reviewed_pair'}
    if pair['id'] is not None:
        doc['pair_id'] = pair['id']
    pool = [row for row in pair['examples'] if row['source_sha256'] not in excluded_sources]
    for a in pool:
        if a['feature_id'] != positive:
            continue
        for b in pool:
            if (b['feature_id'] == negative and b['source_sha256'] != a['source_sha256']
                    and b['sha256'] != a['sha256']):
                return {**doc, 'references': [a, b]}
    return {**doc, 'fallback': 'no_reviewed_pair'}


def _reference_image(row):
    import cv2
    import numpy as np
    data = Path(row['path']).read_bytes()
    if hashlib.sha256(data).hexdigest() != row['sha256']:
        raise ValueError('示例已变化')
    try:
        image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    except cv2.error as exc:
        raise ValueError('示例无法解码') from exc
    if image is None:
        raise ValueError('示例无法解码')
    return image


def _fit(image, width, height, *, upscale):
    import cv2
    h, w = image.shape[:2]
    ratio = min(width / w, height / h)
    if not upscale:
        ratio = min(1.0, ratio)
    if ratio == 1.0:
        return image
    return cv2.resize(image, (max(1, round(w * ratio)), max(1, round(h * ratio))),
                      interpolation=cv2.INTER_CUBIC if ratio > 1 else cv2.INTER_AREA)


def _compose(crop, baseline, refs):
    import cv2
    import numpy as np
    h, w = baseline.shape[:2]
    # 用原请求长边约五分之一放示例，目标至少保留原生分辨率。
    if w >= h:
        target_w, target_h = w - w // 5 - 4, h
        sx, sy, sw, sh = target_w + 4, 0, w // 5, h
    else:
        target_w, target_h = w, h - h // 5 - 4
        sx, sy, sw, sh = 0, target_h + 4, w, h // 5
    if crop.shape[0] > target_h or crop.shape[1] > target_w:
        return baseline, None, 'target_detail'
    if min(sw, sh) < 36:
        return baseline, None, 'reference_detail'
    profiles = [(sw, sh // 2, 0, sh // 2), (sw // 2, sh, sw // 2, 0)]
    placements = []
    for cw, ch, dx, dy in profiles:
        thumbs = [_fit(image, cw - 6, ch - 17, upscale=False) for image in refs]
        if all(min(image.shape[:2]) >= 32 and max(image.shape[:2]) >= 64 for image in thumbs):
            placements = [(thumbs[0], sx, sy), (thumbs[1], sx + dx, sy + dy)]
            break
    if not placements:
        return baseline, None, 'reference_detail'

    target = _fit(crop, target_w, target_h, upscale=True)
    th, tw = target.shape[:2]
    x, y = (target_w - tw) // 2, (target_h - th) // 2
    canvas = np.full_like(baseline, 244)
    canvas[y:y + th, x:x + tw] = target
    for (image, x0, y0), caption in zip(placements, ('P', 'N')):
        ih, iw = image.shape[:2]
        cv2.putText(canvas, caption, (x0 + 3, y0 + 12), cv2.FONT_HERSHEY_SIMPLEX, .42, (28, 28, 28), 1)
        canvas[y0 + 15:y0 + 15 + ih, x0 + 3:x0 + 3 + iw] = image
    return canvas, [x, y, x + tw, y + th], None


def prepare_review(crop, baseline, plan, geometry):
    """同一请求一张图，复用原 H×W 和编码。提示只要求目标区内的可见部位。"""
    from detect_core import FEATURE_ID
    from prompts import NOT_FEATURE_LABEL, UNREADABLE_LABEL, VALID_LABELS

    sent, box = baseline, None
    fallback = plan.get('fallback')
    refs = plan['references']
    candidates = {fid: label for label, fid in FEATURE_ID.items() if label in VALID_LABELS and fid in CUES}
    focus = [fid for fid in (plan['positive_feature'], plan['negative_feature']) if fid in candidates]
    if refs and len(focus) == 2:
        try:
            images = [_reference_image(row) for row in refs]
            sent, box, fallback = _compose(crop, baseline, images)
        except (OSError, ValueError):
            fallback = 'reference_unavailable'
    elif refs:
        fallback = 'reference_class_unavailable'
    hint = '；'.join(f'{candidates[fid]}看{CUES[fid][1]}' for fid in focus)
    if not hint:
        hint = '看壁与腔、晶棱、脐点或可辨的纹理'
    # 按模型可选标签给出可校验的短代码，不让分类先验限制答案。
    labels = '、'.join(f'{label}({CUES[fid][0]})' for fid, label in candidates.items())
    scene = '整张图是待核验的目标裁图。'
    if box:
        side = '左侧' if baseline.shape[1] >= baseline.shape[0] else '上方'
        a, b = focus
        scene = (f'{side}最大区域是待核验目标。P 是{candidates[a]}正例，N 是{candidates[b]}对照。'
                 '示例只提示形态，判别结构和部位必须来自目标区。')
    prompt = (
        '复核显微目标的身份。' + scene + '\n'
        f'优先查看{hint}。可选类别和结构代码为{labels}。看清且为背景杂质时选{NOT_FEATURE_LABEL}。'
        f'仅有模糊长影、轮廓或无法找到鉴别结构时选{UNREADABLE_LABEL}，理由填身份待定。'
        '不得用细长、颜色或示例相似度单独确认。\n'
        '用目标区自己的九宫格指出可见结构的部位。target_part 取左上、上中、右上、左中、中央、右中、左下、下中、右下。'
        'visible_cue 必须是该处实见结构的代码。拒绝项的这两键填空。\n'
        'granularity 取 instance、bundle、sheet 或 clump。多根成束取 bundle。'
        'confidence 取 high、medium 或 low。reason 说明实见结构，限 24 字。'
        + (f'\n目标框几何仅作参考。{geometry}' if geometry else '') +
        '\n只输出 JSON。{"label":"","second":"","confidence":"","reason":"","granularity":"instance",'
        '"target_part":"","visible_cue":""}'
    )
    audit = {**plan, 'mode': 'target_with_examples' if box else 'target_only',
             'fallback': fallback, 'baseline_hw': list(baseline.shape[:2]),
             'native_hw': list(crop.shape[:2]), 'target_box_px': box or [0, 0, baseline.shape[1], baseline.shape[0]],
             'references': [{key: value for key, value in row.items() if key != 'path'} for row in refs] if box else []}
    return sent, prompt, audit


def finish_review(value):
    """模型必须报告与类别一致的可见结构和部位。拒绝、失败和身份待定分开记账。"""
    if value.get('error'):
        return value
    fid = value.get('feature_id')
    part, cue = value.pop('target_part', None), value.pop('visible_cue', None)
    reason = str(value.get('reason') or '')[:120]
    expected = CUES.get(fid)
    label = value.get('reject')
    if label == 'not_feature':
        return {**value, 'visual_identity': {'status': 'not_feature'}}
    # 明确的不可见描述不得同时给出正面身份，即使模型填了合法代码。
    vague = any(text in reason for text in ('身份待定', '模糊长影', '看不清', '无法分辨', '无法看清'))
    if (not label and expected and not vague
            and isinstance(part, str) and part in PARTS and cue == expected[0]):
        confidence = value.get('confidence')
        if confidence not in ('high', 'medium', 'low'):
            confidence = 'low'
        return {**value, 'confidence': confidence,
                'visual_identity': {'status': 'observed', 'target_part': part, 'visible_cue': cue}}
    return {**value, 'label': '身份待定', 'feature_id': '', 'reject': 'unreadable', 'confidence': 'low',
            'reason': '身份待定，目标图未报告可定位的鉴别结构',
            'visual_identity': {'status': 'pending', 'reported_reason': reason}}
