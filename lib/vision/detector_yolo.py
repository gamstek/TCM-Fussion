# -*- coding: utf-8 -*-
"""读取模型和配置，结合权威类别表将 YOLO 结果转换为原图坐标检测框。整图与滑窗共用类别解析、框裁剪和合并，按权重路径与内容摘要复用已加载模型。检测结果由 worker 交给下游裁图。

主要定义：
- load_settings: 读 `detector.json` 推理键与命令行覆盖，产出含类别表/粒度的参数字典。
- resolve_names: 决定模型 class id → 中文类别名；只允许 `det_dataset.class_table` 一套类别事实源，数量或名称不符即报错。
- windows_of: 生成 tile/overlap 滑窗（原图坐标，含边缘对齐窗）。
- detect: 整图或滑窗推理，跨窗按 `nms_merge` 去重、按 `geom.clamp_box` 夹紧，输出 label/tight_bbox/confidence/score/morphology。
- SCORE_HIGH / SCORE_MEDIUM: confidence 粗档切点，只影响目检/复核排序，不参与判药。
- DETECTOR_DEFAULTS: 推理缺省参数（imgsz/conf/iou/device/max_det/tile/overlap）。

关联：
- 调用方 「cli/yolo_worker.py」。
- 直接依赖 「lib/runtime/storage.py」。
"""
import json
import os
import re

import numpy as np

import contracts
from 目标检测.lib.det_dataset import class_table
from TCM_Fusion.lib.runtime.storage import file_hash
from detect_core import nms_merge
from geom import clamp_box
from ls_common import DET_CONFIG, PROJECT_ROOT, load_config

# confidence 粗档切点：仅影响目检/复核排序，不参与判药。
SCORE_HIGH = 0.75
SCORE_MEDIUM = 0.40

# 推理配置缺省值：detector.json 只写差异键也能跑（P2 方案 §6.4 的键表）。
DETECTOR_DEFAULTS = {"imgsz": 1024, "conf": 0.25, "iou": 0.45, "device": "0",
                     "max_det": 300, "tile": 0, "overlap": 0.2}

_MODEL_CACHE = {}
_GENERIC_NAME = re.compile(r"^class_\d+$")


def _project_path(p):
    return p if os.path.isabs(p) else os.path.join(PROJECT_ROOT, p)


def load_settings(cfg, config_path=None, weights=None):
    """读 `detector.json` 推理键 + 命令行覆盖 → `detect()` 的参数字典（含类别表/粒度）。

    `detect_sample --engine yolo` 与 `det_eval --engine yolo` 共用这一份解析，
    避免两处各写一套默认值/相对路径规则。`weights` 非空时覆盖配置。
    """
    path = (config_path or os.path.join(DET_CONFIG, "detector.json"))
    path = _project_path(path)
    try:
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
    except (OSError, ValueError) as e:
        raise ValueError("读不到 YOLO 推理配置 {}：{}".format(path, e))
    w = str(weights or raw.get("weights") or "")
    if not w:
        raise ValueError("未指定 YOLO 权重：--weights 或 detector.json → weights")
    w = _project_path(w)
    if not os.path.isfile(w):
        raise FileNotFoundError("YOLO 权重不存在：{}".format(w))
    classes = class_table(cfg)
    granularity = {}
    for c in (cfg.get("yolo") or {}).get("classes") or []:
        g = str(c.get("granularity") or contracts.GRANULARITY_DEFAULT).strip().lower()
        granularity[str(c.get("zh") or "")] = (
            g if g in contracts.GRANULARITY_LEVELS else contracts.GRANULARITY_DEFAULT)
    out = {k: raw.get(k, v) for k, v in DETECTOR_DEFAULTS.items()}
    out.update({"weights": w, "classes": classes, "granularity": granularity})
    try:
        for key in ("imgsz", "max_det", "tile"):
            out[key] = int(out[key])
        for key in ("conf", "iou", "overlap"):
            out[key] = float(out[key])
    except (TypeError, ValueError) as e:
        raise ValueError("detector.json 推理参数类型非法：{}".format(e))
    out["device"] = None if out["device"] in (None, "") else str(out["device"])
    return out


def _model_for(weights):
    """按权重路径惰性加载并缓存 Ultralytics 模型（同进程逐图复用）。"""
    if not weights:
        raise ValueError("YOLO 推理缺少权重路径（--weights 或 detector.json → weights）")
    path = os.path.abspath(weights)
    if not os.path.isfile(path):
        raise FileNotFoundError("YOLO 权重不存在：{}".format(path))
    signature = file_hash(path)
    cached = _MODEL_CACHE.get(path)
    if cached is not None and cached[0] == signature:
        return cached[1]
    from ultralytics import YOLO              # 惰性导入：离线测试不需要 torch
    model = YOLO(path)
    _MODEL_CACHE[path] = (signature, model)
    return model


def _ordered_names(model):
    """模型自带的类别名（按 class id 升序）；读不到返回 []。"""
    names = getattr(model, "names", None)
    if isinstance(names, dict):
        try:
            keys = sorted(names, key=int)
        except (TypeError, ValueError):
            keys = sorted(names)
        return [str(names[k]) for k in keys]
    if names:
        return [str(x) for x in names]
    return []


def resolve_names(model, table):
    """决定模型 class id → 中文类别名的映射。

    - `model.names` 全是 `class_N`（本项目的训练口径）→ 用 `table`（vlm_config 派生）。
    - 权重自带中文名 → 直接用，但必须都在 `table` 里（只允许一套类别事实源）。
    - 数量不一致 → 报错，不猜测、不截断（错位会把框标成别的类且无告警）。
    """
    names = _ordered_names(model)
    if not names:
        return [c["zh"] for c in table]
    if all(_GENERIC_NAME.match(n) for n in names):
        if len(names) != len(table):
            raise ValueError(
                "权重类别数（{}）与 vlm_config 训练类别数（{}）不一致；"
                "请确认用的是 P0 数据集训出的权重".format(len(names), len(table)))
        return [c["zh"] for c in table]
    if len(names) != len(table):
        raise ValueError("权重自带类别数（{}）与 vlm_config（{}）不一致".format(
            len(names), len(table)))
    known = {c["zh"] for c in table}
    unknown = [n for n in names if n not in known]
    if unknown:
        raise ValueError("权重类别名不在 vlm_config 类别表里：{}".format(unknown))
    return names


def _to_numpy(value):
    """Ultralytics 的 tensor / ndarray 统一成 ndarray（`.cpu()` 只在 GPU tensor 上存在）。"""
    if value is None:
        return np.empty((0,))
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _predict(model, img, imgsz, conf, iou, device, max_det):
    """一次模型前向：返回 [(xyxy(相对当前窗口), score, class_id)]。"""
    results = model.predict(img, imgsz=imgsz, conf=conf, iou=iou, device=device,
                            max_det=max_det, verbose=False)
    r = results[0] if isinstance(results, (list, tuple)) else results
    boxes = getattr(r, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []
    xyxy = _to_numpy(boxes.xyxy).reshape(-1, 4)
    scores = _to_numpy(boxes.conf).reshape(-1)
    classes = _to_numpy(boxes.cls).reshape(-1)
    return [([float(v) for v in xyxy[i]], float(scores[i]), int(classes[i]))
            for i in range(len(xyxy))]


def windows_of(W, H, tile=0, overlap=0.2):
    """滑窗列表（原图坐标，含边缘对齐窗）；`tile<=0` 或图小于窗 → 整图一个窗。"""
    tile = int(tile or 0)
    if tile <= 0 or (W <= tile and H <= tile):
        return [(0, 0, W, H)]
    tw, th = min(tile, W), min(tile, H)
    ov = min(max(float(overlap), 0.0), 0.9)
    step_x = max(1, int(round(tw * (1.0 - ov))))
    step_y = max(1, int(round(th * (1.0 - ov))))

    xs = list(range(0, W - tw + 1, step_x)) or [0]
    if xs[-1] != W - tw:
        xs.append(W - tw)
    ys = list(range(0, H - th + 1, step_y)) or [0]
    if ys[-1] != H - th:
        ys.append(H - th)
    return [(x, y, x + tw, y + th) for y in ys for x in xs]


def _band(score):
    if score >= SCORE_HIGH:
        return "high"
    if score >= SCORE_MEDIUM:
        return "medium"
    return "low"


def detect(img, weights=None, imgsz=1024, conf=0.25, iou=0.45, device=None,
           max_det=300, tile=0, overlap=0.2, classes=None, model=None):
    """整图/滑窗 YOLO 检测 → 与 `detect_once.dets` 同形的框列表（原图像素坐标）。

    `classes` 为 `det_dataset.class_table` 的结果（按训练 class id 0..N-1 排列）；None 时
    从 `vlm_config.json` 现读。`model` 仅测试注入用；正常路径给 `weights`。
    """
    if img is None:
        raise ValueError("YOLO 推理收到空图像")
    H, W = img.shape[:2]
    model = model or _model_for(weights)
    table = classes if classes is not None else class_table(load_config())
    names = resolve_names(model, table)

    dets = []
    for (x1, y1, x2, y2) in windows_of(W, H, tile=tile, overlap=overlap):
        crop = img[y1:y2, x1:x2]
        for box, score, cid in _predict(model, crop, imgsz, conf, iou, device, max_det):
            if cid < 0 or cid >= len(names):
                raise ValueError("模型输出 class id {} 超出类别表（{} 类）："
                                 "权重与 vlm_config 不匹配".format(cid, len(names)))
            gbox = [box[0] + x1, box[1] + y1, box[2] + x1, box[3] + y1]
            cb = clamp_box(gbox, W, H)       # 与 VLM 路径同一份夹紧规则
            if cb is None:
                continue
            dets.append({
                "label": names[cid],
                "tight_bbox": [round(v, 1) for v in cb],
                "confidence": _band(score),
                "score": round(score, 4),
                "morphology": "",
            })

    if tile and len(dets) > 1:
        # 跨窗重复：同一实体在相邻窗各出一次。按模型自己的 iou 阈值聚簇，
        # 与 VLM 分块路径共用 detect_core 的唯一贪心实现。
        dets = nms_merge(dets, thr=iou)
    # 与 `detect_core.nms_merge` 同一排序口径：保证同图重跑时 crop_id 的序号稳定。
    dets.sort(key=lambda d: (d["label"], d["tight_bbox"][1], d["tight_bbox"][0]))
    return dets
