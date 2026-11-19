# -*- coding: utf-8 -*-
"""classify_crop.py — stage 2：把一张候选实体 crop 分类到显微特征类别（两阶段检测的第二步）。

**为什么需要它**（2026-10-08 论文分析 + 本项目实测）：一阶段 VLM 在**一次自回归生成**里同时做
三件事——定位、粒度、命名。论文明测命名几乎是确定性接续（标签 token 概率≈1.0，而坐标首位
数字才是真决策点 0.66 vs 0.50）；本项目 N=10 实测：同一位置**聚成一簇 10 票、标签却 6:4 摇摆**。
命名既然不需要推理，那类别间区分就只能靠 prompt 里堆定义硬挤 → 结构性互相干扰。

两阶段把这个复合判断拆开：stage 1（`prompts.PROMPT_S1`）只给框、不命名；stage 2（本模块）
在**放大后的 crop** 上做单实体分类。增量来自三处：① 有效分辨率提升一个量级（crop 放大到
`target_side`）；② 提示词不再需要 14 类定义共处一段（类别线索按类给，且只给易混对）；
③ 可归因（stage 1 的定位与 stage 2 的分类能分别评测）。

**可替换实现**：`impl="vlm"` 走 VLM；`impl="model"` 走自训分类器（`cls_net`，ResNet18 /
320 letterbox，见 `cli/train_cls.py`）。两者签名与返回值形状一致，故接入/替换/回退都不动
下游。**注意 `model` 没有拒绝类**：14 个训练类里「其他」是正类而非「都不是」，stage 1 的
假阳性会被强制归到某个特征类——补拒绝类需要先回流检测 FP crop。

**与"多次调用"路线的区别**（别被历史结论误伤）：分块兜底 / V1 / V2 / V3 都是**重复同一次
判断**再聚合（重复不增信息）；这里是**把一次复合判断拆成两次更简单的判断**。

读法限制（必须先记住）：本模块只做**特征归类**，不判"是不是药典真阳"、不做形态学判据推理——
判据归 `criteria_table.json`（所有权边界见 AGENTS.md）。
"""
import os
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

from contracts import GRANULARITY_DEFAULT, GRANULARITY_LEVELS   # 框粒度枚举的唯一事实源（契约层）
from detect_core import FEATURE_ID                      # 与检测侧同一份 zh→feature_id 映射
from prompts import (NOT_FEATURE_LABEL, REFUSE_LABELS, UNREADABLE_LABEL,  # 两个拒绝类必须分开
                     VALID_LABELS, build_classify_prompt)
from vlm_api import call_vlm, encode_jpeg_b64, parse_vlm_json

CLASSIFY_TARGET_SIDE = 768   # crop 一律归一到长边 768（**可放大**：整图上小目标只有几十像素）
CLASSIFY_QUALITY = 90

# ── 本地清晰度分（**不是门槛**，2026-10-08 实测校准后降级）─────────────────────
# 起因：档 1 实测发现 stage 1 会从**虚焦区域**框出东西。原打算加一道确定性硬门槛（算不出清晰度
# 就不调 API），**但用 360 张现存 crop 校准后否掉了**：拿"stage 2 自报虚焦/模糊"作代理标签（115 张
# 低质量 / 245 张其余），Tenengrad 均值（长边 512 归一）的两组分布重叠严重——
#   中位 241 vs 1088（有信号），但 阈=150 时误拦 5% 只拦下 35%；要拦下 83% 得把阈提到 400，
#   那时误拦 25%。根因：**Tenengrad 量的是纹理密度，不是清晰度**——虚焦但背景粉粒多 ⇒ 分高，
#   合焦但晶体透明平滑 ⇒ 分低。
# 结论：`focus` 只作**记录字段**（可复现、便于事后复盘），质量判断交给 stage 2 的
# `图像不可判` 类（模型有上下文，能判"看不清"）。`focus_gate` 默认**关**；真要开，MIN_FOCUS
# 取 150（已知性价比差，见上）。
FOCUS_NORM_SIDE = 512        # 归一化到长边 512 再算，使分数跨 crop 可比
MIN_FOCUS = 150.0            # 仅供 focus_gate=True 时使用；默认不启用（理由见上）


def _prep(crop, target_side=CLASSIFY_TARGET_SIDE):
    """长边归一到 target_side：小 crop 放大（INTER_CUBIC）、大 crop 缩小（INTER_AREA）。

    这是两阶段的核心增益之一——一阶段的 prompt 给不了"放大后的图"。
    """
    h, w = crop.shape[:2]
    if not target_side or max(h, w) == target_side:
        return crop
    s = target_side / float(max(h, w))
    interp = cv2.INTER_CUBIC if s > 1 else cv2.INTER_AREA
    return cv2.resize(crop, (max(1, int(round(w * s))), max(1, int(round(h * s)))),
                      interpolation=interp)


def focus_score(crop_bgr):
    """清晰度（Tenengrad 均值）：灰度 Sobel 梯度平方的均值，先归一到长边 512 保证可比。

    虚焦 crop 的梯度能量显著偏低。选它而非"方差 of Laplacian"是因为它对**纹理/边缘**更敏感，
    而显微图判特征靠的正是边缘与纹理。
    """
    g = cv2.cvtColor(_prep(crop_bgr, FOCUS_NORM_SIDE), cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    return float(np.mean(gx * gx + gy * gy))


def classify_crop(crop_bgr, cfg_vlm, api_key, ctx="", impl="vlm",
                  target_side=CLASSIFY_TARGET_SIDE, focus_gate=False,
                  weights=None, device="auto"):
    """单张 crop → dict；失败时返回 `{"error": ...}`（不抛，便于批量继续）。

    返回键：
      `label`       —— `VALID_LABELS` 之一，或两个拒绝类（`不是特征` / `图像不可判`）之一；
                       模型答了清单外的名字时为 `None`（**不静默映射到"最像"的类**）
      `reject`      —— `"not_feature"` / `"unreadable"` / `""`（**给代码用的稳定枚举**，
                       不必去 match 中文标签；两个拒绝类的区别见 `prompts.build_classify_prompt`）
      `raw_label`   —— 模型原始输出（便于审计非法答案，不参与下游）
      `feature_id`  —— 下游公共主键（`detect_core.FEATURE_ID` 同一份映射）；`label` 为
                       `None` 或拒绝类时为空串
      `second` / `confidence` / `reason` —— 次可能类别 / 置信 / 判别依据
      `source`      —— `"vlm"` / `"gate"`（本地清晰度门槛拦下，未调 API）/ `"model"`（预留）
      `focus`       —— 本地清晰度分（可复现；便于事后复盘门槛是否设错）
      `sent_hw`     —— 实际送去分类的图像尺寸（放大与否可由它看出）
      `granularity` —— 框粒度（`contracts.GRANULARITY_LEVELS` 之一，非法/缺失归 `instance`）：
                       量化可用性闸门据此决定该实例是否产 `primary_value_um`（见 contracts.py 常量注释）
    """
    focus = focus_score(crop_bgr)     # **始终计算并记录**（毫秒级）：它是事后复盘/校准的原始数据，
    if focus_gate and focus < MIN_FOCUS:   # 只有"拦截"这件事交给开关（默认关，理由见上方注释）
        return {"label": UNREADABLE_LABEL, "raw_label": "", "feature_id": "",
                "reject": "unreadable", "second": "", "confidence": "high",
                "reason": f"本地清晰度 {focus:.1f} < {MIN_FOCUS}（未调 API）",
                "source": "gate", "focus": round(focus, 1), "sent_hw": None,
                "granularity": GRANULARITY_DEFAULT}
    if impl == "model":
        # 自训分类网络（`cli/train_cls.py` 的产物）：预处理/类别表全在 `cls_net`，
        # 与训练同一份实现；本函数只把结果翻译成与 VLM 路径同形的 13 键。
        if not weights:
            raise ValueError("impl='model' 需要 weights（训练产物 best.pt 路径）")
        from cls_net import classify_arrays
        r = classify_arrays([crop_bgr], weights, device=device)[0]
        label = r["label"] if r["label"] in VALID_LABELS else None
        h, w = crop_bgr.shape[:2]
        return {
            "label": label,
            "reject": "",
            "raw_label": r["label"],
            "feature_id": "" if label is None else FEATURE_ID.get(label, ""),
            "second": r["second"],
            "confidence": r["prob"],
            "reason": "自训分类器 top-1 softmax",
            "source": "model",
            "focus": None if focus is None else round(focus, 1),
            "sent_hw": [h, w],
            "granularity": GRANULARITY_DEFAULT,
        }
    img = _prep(crop_bgr, target_side)
    b64, _ = encode_jpeg_b64(img, max_side=None, quality=CLASSIFY_QUALITY)
    text, err = call_vlm(cfg_vlm, api_key, b64, build_classify_prompt(ctx))
    if text is None:
        return {"error": err, "focus": round(focus, 1), "granularity": GRANULARITY_DEFAULT}
    try:
        parsed = parse_vlm_json(text)
    except Exception as e:
        return {"error": f"解析: {e}", "focus": round(focus, 1), "granularity": GRANULARITY_DEFAULT}
    raw = str(parsed.get("label") or "").strip()
    known = raw in VALID_LABELS or raw in REFUSE_LABELS
    label = raw if known else None
    reject = {"not_feature": NOT_FEATURE_LABEL, "unreadable": UNREADABLE_LABEL}
    # 粒度：只接受 GRANULARITY_LEVELS 内的取值；模型答了别的/漏答 → 归 instance（不静默映射到"束"）。
    gran = str(parsed.get("granularity") or "").strip().lower()
    if gran not in GRANULARITY_LEVELS:
        gran = GRANULARITY_DEFAULT
    return {
        "label": label,
        "reject": next((k for k, v in reject.items() if label == v), ""),
        "raw_label": raw,
        "feature_id": "" if (label is None or label in REFUSE_LABELS)
                      else FEATURE_ID.get(label, ""),
        "second": str(parsed.get("second") or ""),
        "confidence": parsed.get("confidence", "medium"),
        "reason": str(parsed.get("reason") or ""),
        "source": "vlm",
        "focus": None if focus is None else round(focus, 1),
        "sent_hw": [img.shape[0], img.shape[1]],
        "granularity": gran,
    }


def classify_crops(items, cfg_vlm, api_key, workers=8, target_side=CLASSIFY_TARGET_SIDE):
    """批量分类：`items = [(key, crop_bgr, ctx), ...]` → `{key: result}`（顺序同 items）。

    **并发收在本进程内**（跨进程各自计数会叠加突破账户并发上限，见 `detect_core.detect_consensus`）。
    `workers` 建议 ≤8：stage 2 的调用很轻（单实体小图），瓶颈在服务端而非本地。
    """
    def one(kc):
        k, crop, ctx = kc
        return k, classify_crop(crop, cfg_vlm, api_key, ctx=ctx, target_side=target_side)

    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as ex:
        return dict(ex.map(one, items))
