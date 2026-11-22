"""为已有显微目标框选择原像素观察窗。

主要定义：feature_window 保留目标框，按轮廓或轴向为周围留出观察空间。
关联：融合检测 worker 写入视窗，几何测量按随行元数据核对坐标。
"""
import math

import cv2
import numpy as np

import contracts
from geom import LOOSE_MARGIN_MIN_PX, LOOSE_MARGIN_RATIO


_OUTLINE = frozenset({'cluster_crystal', 'single_crystal', 'stone_cell', 'starch_grain'})
_AXIAL = frozenset({'vessel', 'fiber', 'secretory_tissue'})


def _texture_axis(image, box):
    x1, y1, x2, y2 = map(int, box)
    roi = image[y1:y2, x1:x2]
    h, w = roi.shape[:2]
    if min(h, w) < 12:
        return None
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi
    scale = min(1.0, 192 / max(w, h))
    if scale < 1:
        gray = cv2.resize(gray, (max(1, round(w * scale)), max(1, round(h * scale))),
                          interpolation=cv2.INTER_AREA)
    if min(gray.shape) < 8:
        return None
    dx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)[2:-2, 2:-2]
    dy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)[2:-2, 2:-2]
    xx, yy, xy = float(np.mean(dx * dx)), float(np.mean(dy * dy)), float(np.mean(dx * dy))
    energy = xx + yy
    if energy < 25 or math.hypot(xx - yy, 2 * xy) < .55 * energy:
        return None
    return (math.degrees(.5 * math.atan2(2 * xy, xx - yy)) + 90) % 180


def feature_window(image, box, feature_id, *, context_box=None):
    """返回原图中的半开视窗和取景依据。方向提示只影响留边。"""
    if feature_id not in _OUTLINE | _AXIAL:
        return None
    h, w = image.shape[:2]
    left, top, right, bottom = box
    bw, bh = right - left, bottom - top
    short, long = min(bw, bh), max(bw, bh)
    margin = max(LOOSE_MARGIN_MIN_PX, LOOSE_MARGIN_RATIO * short, min(72, .18 * short))
    pad_x = pad_y = margin
    angle, basis = None, None
    if feature_id in _AXIAL:
        if long >= 2 * short:
            angle, basis = (0.0 if bw >= bh else 90.0), 'box_axis'
        else:
            angle = _texture_axis(image, box)
            if angle is not None:
                basis = 'texture_axis'
        if angle is not None:
            extension = max(0.0, min(96, .20 * long) - margin)
            pad_x += extension * abs(math.cos(math.radians(angle)))
            pad_y += extension * abs(math.sin(math.radians(angle)))
    x1, y1, x2, y2 = left - pad_x, top - pad_y, right + pad_x, bottom + pad_y
    if context_box is not None:
        a, b, c, d = context_box
        x1, y1, x2, y2 = min(x1, a), min(y1, b), max(x2, c), max(y2, d)
    window = [max(0, math.floor(x1)), max(0, math.floor(y1)),
              min(w, math.ceil(x2)), min(h, math.ceil(y2))]
    return window, {'schema': contracts.FEATURE_WINDOW_SCHEMA, 'source_size_wh': [w, h],
                    'profile': 'axial' if angle is not None else 'outline',
                    'axis_deg': round(angle, 2) if angle is not None else None, 'axis_basis': basis}
