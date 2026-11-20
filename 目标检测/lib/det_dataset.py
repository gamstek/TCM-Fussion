# -*- coding: utf-8 -*-
"""提供 YOLO 类别顺序和检测交付分流。

从视觉配置读取类别并构造连续的模型索引。将「其他」框留给审计，交付只消费可进入下游的类别。

主要定义：DatasetError、class_table、split_delivery。
关联：融合 YOLO 适配、检测 worker 和原检测入口共同使用本模块。
"""
OTHER_ZH = "其他"
PENDING_IDS = ("pending",)
PENDING_ZH = ("疑似待定",)


class DatasetError(RuntimeError):
    """检测类别配置错误。"""


def class_table(cfg):
    """vlm_config.json → 训练类别表（排除 pending，按全局 id 升序，重编 train_id 0..N-1）。"""
    raw = (cfg.get("yolo") or {}).get("classes") or []
    out = []
    for c in raw:
        fid = str(c.get("feature_id", ""))
        zh = str(c.get("zh", ""))
        if fid in PENDING_IDS or zh in PENDING_ZH:
            continue
        out.append({"global_id": int(c["id"]), "feature_id": fid, "zh": zh})
    out.sort(key=lambda c: c["global_id"])
    for i, c in enumerate(out):
        c["train_id"] = i
    if not out:
        raise DatasetError("vlm_config.json 的 yolo.classes 为空，无法解析检测类别")
    return out


def split_delivery(dets, other_zh=OTHER_ZH):
    """把「其他」类检测框从交付集分流：返回 `(delivery, other)`，两侧都保持原顺序。

    **识别 ≠ 交付**（方案 §6.4）：`其他` 是第 14 个训练/评测正类，检测器照常输出；
    但 `feature_taxonomy.json` 没有 `other` 的 `igm_desc`，进了 crops/ 会让 IGM 在
    `load_feature_desc` 处退出。故调用方只让 `delivery` 走 `crop_sidecar`，`other`
    另落 `other_detections.json`（名字不叫 rejected —— 它不是负样本，只是不进交付）。
    """
    delivery, other = [], []
    for d in dets:
        (other if d.get("label") == other_zh else delivery).append(d)
    return delivery, other
