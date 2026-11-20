# -*- coding: utf-8 -*-
"""检测框与分割掩码的共享几何原语。

接收像素坐标框或掩码，完成框归正、交并比、松框外扩和外轮廓矢量化。

主要定义：clamp_box、iou_box、to_polygon 和 loose_box。
关联：检测、分割及证据实例关联调用本模块。依赖 NumPy 和 OpenCV。
"""
import cv2
import numpy as np

# 松框外扩约定（与 docs/分割方案与设计.md §1.2 一致）：外扩量 = max(框短边 × 5%, 24px)
LOOSE_MARGIN_RATIO = 0.05
LOOSE_MARGIN_MIN_PX = 24


def clamp_box(box, width, height):
    """归正并夹紧像素框，夹后任一边不足 1px 时丢弃。"""
    x1, y1, x2, y2 = box
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    x1, x2 = min(max(x1, 0.0), float(width)), min(max(x2, 0.0), float(width))
    y1, y2 = min(max(y1, 0.0), float(height)), min(max(y2, 0.0), float(height))
    if x2 - x1 < 1.0 or y2 - y1 < 1.0:
        return None
    return x1, y1, x2, y2


def iou_box(a, b):
    """两个轴对齐框的 IoU。框为 (x1, y1, x2, y2)，单位不限（同侧一致即可）。

    与检测侧历史实现 `detect_core.iou_box` 逐值等价：重叠宽高先 clamp 到 0（不相交即 0），
    并集退化（面积为 0/负）时返回 0.0 而非抛错。
    """
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = ix * iy
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def to_polygon(mask, max_n=8, min_area=50):
    """取掩码的外轮廓，按面积选前 max_n 项，舍弃面积小于 min_area 的轮廓。"""
    cs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
    cs = sorted(cs, key=cv2.contourArea, reverse=True)[:max_n]
    polys = []
    for c in cs:
        if cv2.contourArea(c) < min_area:
            continue
        eps = 0.002 * cv2.arcLength(c, True)
        polys.append(np.round(cv2.approxPolyDP(c, eps, True).reshape(-1, 2), 1).tolist())
    return polys


def loose_box(tight, W, H, margin_ratio=LOOSE_MARGIN_RATIO, margin_min_px=LOOSE_MARGIN_MIN_PX):
    """检测紧框 (x1, y1, x2, y2) → 裁剪用松框（四边外扩后按 W×H 截断）。

    外扩量取 `max(框短边 × margin_ratio, margin_min_px)`：前者给阈值类算法留两侧背景平台，
    后者是标注噪声地板（紧框有 ±2% 容差 + 约 ±20px 噪声，直接按紧框裁会把残差变成不可逆切割）。
    这是**全流程约定**（`docs/分割方案与设计.md` §1.2），`ui/seg_stage.py` 与
    （已归档的）旧裁剪 CLI 共用同一实现（2026-09-29 P4-9 上收，值与旧实现逐项一致）。
    """
    x1, y1, x2, y2 = tight
    margin = max(margin_ratio * min(x2 - x1, y2 - y1), margin_min_px)
    return [max(0, int(x1 - margin)), max(0, int(y1 - margin)),
            min(W, int(x2 + margin)), min(H, int(y2 + margin))]
