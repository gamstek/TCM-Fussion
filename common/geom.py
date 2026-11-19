# -*- coding: utf-8 -*-
"""geom.py — 几何 / 掩码原语（纯函数，无 I/O、无 API）。

**为什么存在**（2026-09-29 P4-1a）：同一算法此前在检测/分割两侧各抄了若干份 ——
带 `iou` 名字的函数全仓有 5 个，且实测**分属两类**（框 IoU / 掩码 IoU），
`iou_mask` 还有一份在重叠区域为空时会 ZeroDivisionError 变体。原语层重复的直接后果是
"改一处漏三处"（项目记忆里已有多次此类教训），故上收到 `common/` 这一层。

2026-09-29 追加 `to_polygon`：三引擎分割管线（`seg_prelabel_group.py` 等）归档后，
该函数是唯一被生产链（`segmentation/lib/igm_segment.py`）继续使用的掩码原语，故一并上收。

2026-09-29 P4-9 追加 `loose_box`：它是"松框裁剪"的全流程约定（`docs/分割方案与设计.md` §1.2），
此前只存在于 `segmentation/build_seg_crops.py`，而 `ui/seg_stage.py` 只能 import 那个 CLI 脚本才拿到它
（该脚本已于 2026-09-29 归档，其 CLI 属旧管线）。

依赖方向：本模块只依赖 numpy / cv2（均为第三方库），**不得** import 任何业务模块。

命名分区（避免同名不同义）：
  iou_box     —— **框** IoU，入参 (x1, y1, x2, y2) 序列
  iou_mask    —— **掩码** IoU，入参前景掩码（布尔 / 0-255 / 0-1 均可）
  to_polygon  —— **掩码 → 多边形**（外轮廓矢量化，用于目检叠加与交付 JSON）
  loose_box   —— **紧框 → 松框**（按全流程约定外扩，供裁剪留背景上下文）
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


def iou_mask(a, b):
    """两个前景掩码的 IoU（按 `> 0` 取前景，故布尔 / 0-255 / 0-1 三种载体都成立）。

    契约（含一处为调用方便利而保留的语义，见下）：
      * 重叠为空（并集为 0）→ 返回 **0.0**。历史上有变体在此处 `0.0 / 0.0` 直接崩，
        调用方原本不可能依赖"崩溃"，故这里统一成 0.0（行为只可能变好）。
      * **任一入参为 None → 返回 `nan`**，表示"该侧无有效掩码"。这条是给调用方（尤其是
        已冻结的 `segmentation/experiments/`：框可能不存在）留的便利语义，故显式写在契约里，
        而不是让每个调用点各写一遍判空。需要区分"空掩码"与"无掩码"的调用方请自行判 None。
    """
    if a is None or b is None:
        return float("nan")
    fa = np.asarray(a) > 0
    fb = np.asarray(b) > 0
    union = np.logical_or(fa, fb).sum()
    if not union:
        return 0.0
    return float(np.logical_and(fa, fb).sum()) / float(union)


def to_polygon(mask, max_n=8, min_area=50):
    """掩码 → 多边形列表（RETR_EXTERNAL 外轮廓，按面积取前 `max_n` 个，面积 < `min_area` 丢弃）。

    实现 2026-09-29 从 `segmentation/seg_prelabel_group.py` 原样上收（三引擎管线归档时，
    该函数为其后唯一仍被生产链使用的原语）。逐值等价，未改行为。
    """
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
