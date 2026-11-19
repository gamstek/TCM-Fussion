# -*- coding: utf-8 -*-
"""ls_review.py — Label Studio「分割审核」域的通用件（建项辅助 + 标注读回 → mask）。

**为什么存在**（2026-09-29 P4-9）：这些函数原先分别住在
`segmentation/import_seg_review.py`（注册本地存储、构造 LS 图片 URL）与
`segmentation/export_annotations_from_ls.py`（分页拉任务、brushRLE/polygon → 二值 mask）里，
而它们**同时**被 `ui/seg_stage.py`、`ui/detect_stage.py` 复用 —— 于是 UI 只能
`importlib.import_module("import_seg_review")` 去 import 一个 CLI 脚本（P4-3 记过的同类问题）。
本次上收到 `segmentation/lib/`，CLI 脚本回归"只做入口"。

依赖方向：只依赖 `ls_common`（LS 鉴权 + 路径 + 配置）与 cv2/numpy，**不得** import 任何 CLI 入口。
"""
import json
import os
import urllib.parse
from collections import defaultdict

import cv2
import numpy as np

from ls_common import DATASET, SEG_CONFIG, ls_api


# ---------------------------------------------------------------- LS 建项/存储

def local_url(abs_path):
    """绝对路径 → LS 本地文件服务 URL（相对 `DATASET` 根）。"""
    rel = os.path.relpath(abs_path, DATASET).replace("\\", "/")
    return "/data/local-files/?d=" + urllib.parse.quote(rel)


def register_storage(cfg, pid, path, title):
    """为项目注册本地文件存储（已存在时静默跳过）。"""
    try:
        ls_api(cfg, "POST", "/api/storages/localfiles",
               json={"project": pid, "path": path, "title": title,
                     "use_blob_urls": True})
        print(f"已注册本地存储: {title}")
    except SystemExit:
        print(f"[提示] 存储 {title} 可能已注册，继续")


# ---------------------------------------------------------------- 读回任务与标注

def fetch_project_tasks(cfg, project_id, limit=500):
    """GET /api/tasks?project=N&fields=all（分页拉全）。

    注意：不同 LS 版本把任务列表放在 `results` 或 `tasks` 键下，这里两者都兼容。
    """
    tasks = []
    offset = 0
    while True:
        path = f"/api/tasks?project={project_id}&limit={limit}&offset={offset}&fields=all"
        resp = ls_api(cfg, "GET", path)
        data = resp.json()
        if not isinstance(data, dict):
            return data
        result = (
            data.get("results")
            or data.get("tasks")
            or (data if isinstance(data, list) else [])
        )
        if not result:
            break
        tasks.extend(result)
        total = data.get("total") or data.get("count") or len(result)
        if offset + len(result) >= int(total):
            break
        offset += len(result)
    return tasks


def rle_to_mask(rle_counts, shape):
    """COCO RLE counts (column-major) → 0/1 uint8 ndarray。"""
    h, w = shape
    flat = np.zeros(h * w, dtype=np.uint8)
    pos = val = 0
    for c in rle_counts:
        c = int(c)
        if val:
            flat[pos:pos + c] = 1
        pos += c
        val = 1 - val
    return flat.reshape((w, h)).T  # column-major → row-major


def _ls_brush_decode(rle, h, w):
    """用 label_studio_converter.brush.decode_rle 解本机 LS 的 brush RLE。

    value.rle 是 RGBA 的 RLE 编码；decode_rle 返回长度 h*w*4 的扁平 uint8，
    取第 4 通道(alpha)即二值掩码（brush 只存绘制的描边，或用 fill_hollow 填充）。
    若解码长度不符，回退旧式 [count,value] 配对解码。
    """
    try:
        from label_studio_converter.brush import decode_rle
        raw = np.frombuffer(decode_rle(rle), dtype=np.uint8)
        if len(raw) >= h * w * 4:
            alpha = np.asarray(raw[:h * w * 4]).reshape((h, w, 4))[:, :, 3]
            return (alpha > 0).astype(np.uint8) * 255
    except Exception:   # noqa: BLE001 — 回退
        pass
    # 回退：旧式 [count,value] 配对、列优先
    flat = np.zeros(h * w, np.uint8)
    pos = 0
    for i in range(0, len(rle) - 1, 2):
        c = int(rle[i]); v = int(rle[i + 1])
        if pos + c > h * w:
            c = max(0, h * w - pos)
        if v != 0:
            flat[pos:pos + c] = 255
        pos += c
        if pos >= h * w:
            break
    if pos < h * w:
        return None
    return np.ascontiguousarray(flat.reshape((h, w), order="F"))


def brush_annotation_to_mask(annotation):
    """解析 LS BrushLabels 标注结果 → 二值 mask（多结果取并集）。

    annotations[-1]["result"] 中每条 result 可能是：
      - brushlabels：value.format ∈ {rle, brushRLE}，
          * 新版 rle 为 [count,value] 扁平配对（尺寸在 result 层 original_width/height）
          * 旧版为 dict{"counts":[...]} 或逗号分隔字符串 → rle_to_mask
      - polygon：value.shape=="polygon"，points.path="x,y x,y ..."
      - 其它（choices 等）：忽略。
    """
    results = annotation.get("result", [])
    if not results:
        return None

    masks = []
    for res in results:
        val = res.get("value", {})
        seg_type = val.get("format", "") or val.get("type", "")

        # 尺寸优先取 result 层（新版本地文件任务）；旧版从 value.originalSize 兜底
        img_h = (res.get("original_height")
                 or (val.get("originalSize") or val.get("original_size") or {}).get("height", 0))
        img_w = (res.get("original_width")
                 or (val.get("originalSize") or val.get("original_size") or {}).get("width", 0))
        if not img_h or not img_w:
            continue

        is_poly = (res.get("type") == "polygonlabels") or (val.get("shape") == "polygon")
        if is_poly or seg_type == "polygon":
            pts = val.get("points")
            coords = []
            if isinstance(pts, dict):
                path = pts.get("path", [])
                if isinstance(path, str):   # 旧式："x,y x,y x,y"
                    coords = [[float(p) for p in pt.split(",")] for pt in path.split(" ") if "," in pt]
                elif isinstance(path, list):   # 列表式
                    coords = [[float(x), float(y)] for x, y in path]
            elif isinstance(pts, list):        # 新版：[[x,y],...]
                coords = [[float(x), float(y)] for x, y in pts]
            if len(coords) >= 3:
                mask = np.zeros((img_h, img_w), dtype=np.uint8)
                cv2.fillPoly(mask, [np.array(coords, np.int32)], 1)
                masks.append(mask)
        elif seg_type in ("rle", "brushRLE"):
            rle_data = val.get("rle", val.get("segmentation", {}))
            if isinstance(rle_data, list):
                mask = _ls_brush_decode(rle_data, img_h, img_w)
                if mask is not None:
                    masks.append(mask)
            elif isinstance(rle_data, dict) and "counts" in rle_data:
                masks.append(rle_to_mask(rle_data["counts"], (img_h, img_w)))
            elif isinstance(rle_data, str):
                counts = [int(x) for x in rle_data.split(",")]
                masks.append(rle_to_mask(counts, (img_h, img_w)))

    if not masks:
        return None

    # 合并多个连通域
    fused = np.zeros(masks[0].shape, dtype=np.uint8)
    for m in masks:
        fused = np.logical_or(fused, m > 0).astype(np.uint8) * 255
    return fused


def fill_holes(mask):
    """填封闭洞（先从 (0,0) 灌背景，再取补集并入前景）。"""
    ff = mask.copy()
    h, w = mask.shape
    ff2 = np.zeros((h + 2, w + 2), dtype=np.uint8)
    cv2.floodFill(ff, ff2, (0, 0), 255)
    holes = cv2.bitwise_not(ff)
    return cv2.bitwise_or(mask, holes)


# ---------------------------------------------------------------- 难度组规则文本

def load_rule_hints():
    """读 `config/seg_classes.json` → (表, 组→规则键, 默认组)。"""
    with open(os.path.join(SEG_CONFIG, "seg_classes.json"), encoding="utf-8") as f:
        sc = json.load(f)
    group_rule = {"G1": "R2_G1", "G2": "R3_G2_针晶束", "G3": "R4_G3G4_空腔填充",
                  "G4": "R4_G3G4_空腔填充", "G5": "R5_G5", "G6": "R6_G6", "G7": "R1"}
    default_group = sc["default"]["group"]
    return sc, group_rule, default_group


def seg_meta(herb, folder_feature, sc, group_rule, default_group):
    """(药材, 目录特征) → (难度组, 该组的 mask 规则提示文本)。"""
    key = f"{herb}|{folder_feature}"
    entry = sc["table"].get(key)
    group = entry["group"] if entry else default_group
    rule_key = next((rk for rk, txt in sc["mask_rules"].items() if rk.endswith(group)),
                    "R1")
    rule_text = sc["mask_rules"].get(rule_key, sc["mask_rules"]["R1"])
    hint = f"[{group}] {rule_text}（通用：紧贴可见实体边缘，禁切实体）"
    return group, hint
