# -*- coding: utf-8 -*-
"""box_metrics.py — gold 清单装载与框级匹配原语（`box_recall` 与 `e2e_eval` 共用）。

**为什么上收到 lib**：stage 1 的提框验收（`cli/box_recall.py`）与两阶段端到端验收
（`cli/e2e_eval.py`）必须用**同一套 IoU 匹配规则**，否则两个入口的数字不可比。
匹配语义与 P3 `det_eval` 一致：每个 GT 最多匹配一个预测（贪心取最大 IoU，IoU ≥ 阈值），
一个预测也只能被一个 GT 占用。

本模块只做纯计算与读盘，不 import 任何入口脚本。
"""
import json
import os

from geom import iou_box
from ls_common import canon_path, norm_path


def load_gold(gold_path, ann_path):
    """读 gold 清单 → (gold items, `{归一化路径: [{label, box}]}`)。

    `ann` 的键先过 `canon_path`（兼容历史绝对路径）再 `norm_path`（跨来源比较键），
    与 `box_recall` 历史口径一致。
    """
    with open(gold_path, encoding="utf-8") as f:
        gold = json.load(f)["items"]
    with open(ann_path, encoding="utf-8") as f:
        raw = json.load(f)["items"]
    ann = {}
    for k, boxes in raw.items():
        ann[norm_path(canon_path(k))] = [
            {"label": b["label"], "box": [float(v) for v in b["box"]]} for b in boxes]
    return gold, ann


def match_pairs(aboxes, preds, iou_thr):
    """单图贪心匹配 → ([(gt_idx, pred_idx, iou)], 未使用的预测下标列表)。

    两个方向都一对一：先按 GT 顺序为每个 GT 找当前未被占用的最大 IoU 预测。
    这个"顺序贪心"与 `box_recall` / `det_eval` 的历史实现逐字节等价。
    """
    used, pairs = set(), []
    for gi, a in enumerate(aboxes):
        best, bj = 0.0, None
        for j, pb in enumerate(preds):
            if j in used:
                continue
            v = iou_box(a["box"], pb["box"])
            if v > best:
                best, bj = v, j
        if best >= iou_thr and bj is not None:
            used.add(bj)
            pairs.append((gi, bj, best))
    return pairs, sorted(set(range(len(preds))) - used)


def match_image(aboxes, preds, iou_thr):
    """兼容 `box_recall` 的三返回值：(命中 GT 下标集, 未用预测下标集, 每 GT 最佳 IoU)。"""
    used, hits, best_ious = set(), set(), []
    for gi, a in enumerate(aboxes):
        best, bj = 0.0, None
        for j, pb in enumerate(preds):
            if j in used:
                continue
            v = iou_box(a["box"], pb["box"])
            if v > best:
                best, bj = v, j
        best_ious.append(best)
        if best >= iou_thr and bj is not None:
            used.add(bj)
            hits.add(gi)
    return hits, set(range(len(preds))) - used, best_ious
