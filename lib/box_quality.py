"""框内容质量评估：只做保守的可用性判断，不替代 locate/classify。

目标是先挡住“空白/几乎无结构”的框；不因目标成簇、连续、相邻而拒绝。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def _load_gray(path, max_side=512):
    image = Image.open(path).convert('L')
    image.thumbnail((max_side, max_side), Image.Resampling.BILINEAR)
    return np.asarray(image, dtype=np.float32) / 255.0


def assess(path, *, box=None, source_size=None, touches_border=None):
    """返回 dict；status 只有 usable / low_detail / blank / tiny / invalid。

    - blank：几乎无纹理，阻止继续分割；
    - low_detail：细节弱，保留但标记 review；
    - tiny：尺寸过小，保留但标记；
    - invalid：像素不可读（旧工件/测试替身），不据此阻断；
    - 其余为 usable。
    """
    try:
        image = _load_gray(path)
    except Exception:
        return {'status': 'invalid', 'score': 0.0, 'flags': ['invalid_image'],
                'metrics': {}, 'block_segment': False}
    if image.size == 0:
        return {'status': 'blank', 'score': 0.0, 'flags': ['empty'],
                'metrics': {}, 'block_segment': True}
    height, width = image.shape
    std = float(image.std())
    gx = np.abs(np.diff(image, axis=1)) if width > 1 else np.zeros((height, 1), np.float32)
    gy = np.abs(np.diff(image, axis=0)) if height > 1 else np.zeros((1, width), np.float32)
    edge = float((float(gx.mean()) + float(gy.mean())) / 2.0)
    flags = []
    if min(width, height) < 16:
        flags.append('tiny')
    if std < 0.030 and edge < 0.006:
        flags.append('blank')
    elif std < 0.080 or edge < 0.008:
        flags.append('low_detail')
    if touches_border:
        flags.append('border')
    area_ratio = None
    if box and source_size:
        try:
            bw = max(0.0, float(box[2]) - float(box[0]))
            bh = max(0.0, float(box[3]) - float(box[1]))
            area_ratio = bw * bh / max(1.0, float(source_size[0]) * float(source_size[1]))
            if area_ratio > 0.50:
                flags.append('large')
        except (TypeError, ValueError, IndexError):
            pass
    if 'blank' in flags:
        status = 'blank'
    elif 'tiny' in flags:
        status = 'tiny'
    elif 'low_detail' in flags:
        status = 'low_detail'
    else:
        status = 'usable'
    # 分数只用于排序，不是概率。
    score = max(0.0, min(1.0, 0.55 * min(std / 0.20, 1.0) + 0.45 * min(edge / 0.018, 1.0)))
    return {'status': status, 'score': round(score, 4), 'flags': flags,
            'metrics': {'std': round(std, 4), 'edge': round(edge, 5),
                        'width': int(width), 'height': int(height),
                        'area_ratio': None if area_ratio is None else round(area_ratio, 4)},
            'block_segment': status == 'blank'}


def assess_meta(meta):
    source_size = None
    try:
        with Image.open(meta.source_image_path) as source:
            source_size = source.size
    except Exception:
        pass
    return assess(meta.png, box=meta.tight_box, source_size=source_size,
                  touches_border=bool(getattr(meta, 'touches_border', False)))
