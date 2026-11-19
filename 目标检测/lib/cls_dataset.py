# -*- coding: utf-8 -*-
"""cls_dataset.py — stage 2 特征分类器的 crop 数据集构建（检测训练集 → 分类器训练集）。

**为什么存在**：两阶段检测把"定位"与"命名"拆开后，stage 2 需要一个**与推理同分布**的
crop 训练集。盘上现成的 `crops/` 是 VLM 实时检测产物（数量少、框有偏），不能当训练集；
直接用紧框切又会与推理口径（`detect_core.DET_LOOSE_MARGIN_RATIO = 30%` 松框）不一致。
故从 P0 多类训练集的**人工框**按同一口径重切。

**输入**：`dataset/检测训练集_v1/`（`images/{train,val,test}` + `labels/…` + `split_report.json`）。
**输出**：`dataset/特征分类器_v1/`
    crops/{train,val,test}/<train_id>_<feature_id>/<crop_id>.jpg   ← 可直接给 YOLO-cls / torchvision
    manifest.csv       每 crop 的来源、框、尺寸、药材/组（可追溯）
    classes.txt        `<train_id>\t<feature_id>\t<zh>`
    dataset_report.json  计数 / 断言 / 警告 / 指纹（可复跑的证据）

**划分口径（关键）**：**原样复用 P0 的 train/val/test 划分**，不重新划分。理由：
  ① P0 已按 `group_key` 整组划分、整味药材留出、60 张 gold 整帧预留；重划分必然引入
     "同一原图/同组同时出现在分类器 train 与 test"的泄漏；
  ② 两阶段评测要与 stage 1 同图同尺（`det_eval --gold-from-p0` 的对比才有意义）。
gold_eval 的 60 图**不进入**任何 split（本模块只处理 split ∈ train/val/test 的记录）。

**类别口径**：14 类全保留（含「其他」）。「其他」是第 14 个正类而非背景；分类器照常预测它，
交付侧继续按 `det_dataset.split_delivery` 分流。**本数据集没有"都不是/非特征"拒绝类**——
GT 框里不存在该样本；stage 1 的假阳性要等"检测 FP crop"回流后才能补（见 report 的 warnings）。

**crop 尺寸**：保留原长宽比与外扩背景（不预先拉成方形，避免细长纤维被压扁）；长边只降不升，
上限 `DEFAULT_MAX_SIDE=768`（与 stage 2 VLM 的 `CLASSIFY_TARGET_SIDE` 同一上限），JPEG q92。
训练侧的 letterbox/增强由训练脚本决定，本数据集不烘焙。
"""
import csv
import hashlib
import json
import math
import os
import shutil
import statistics
from collections import Counter

import cv2

import paths
from detect_core import DET_LOOSE_MARGIN_RATIO
from geom import loose_box
from ls_common import imread_cn

SPLITS = ("train", "val", "test")
GOLD_SPLIT = "gold_eval"
SPLIT_POLICIES = ("p0", "rebalance")
DEFAULT_VAL_RATIO = 0.15
DEFAULT_SEED = 20261030
CROPS_DIR = "crops"
MANIFEST_NAME = "manifest.csv"
CLASSES_NAME = "classes.txt"
REPORT_NAME = "dataset_report.json"

DEFAULT_MAX_SIDE = 768          # 只降不升：与 stage 2 送 VLM 的 CLASSIFY_TARGET_SIDE 同上限
DEFAULT_JPEG_QUALITY = 92
MIN_CROP_SIDE = 4               # 松框短边 < 4px 视为退化框（源标注里 0.3% 的极小框）
RARE_TRAIN_WARN = 100           # train crop 数低于此值 → 警告（不足时靠重采样/损失加权）
RARE_TRAIN_CRITICAL = 30        # 低于此值 → 严重不足（分类器几乎学不到）

MANIFEST_FIELDS = [
    "crop_id", "split", "source_split", "class_id", "feature_id", "zh",
    "source_image", "source_md5", "herb", "group_key", "source_dirs",
    "img_w", "img_h", "tight_box", "loose_box", "crop_w", "crop_h",
    "saved_w", "saved_h", "scale", "box_area_frac", "edge_touch", "source_repeat",
]


class ClsDatasetError(ValueError):
    """源数据不完整/口径不一致/输出目录冲突（调用方应直接报错退出）。"""


# ── 源数据读取 ─────────────────────────────────────────────────────────

def _md5_file(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_source(src_dir):
    """读 `split_report.json` → `(src_dir_abs, report, class_map)`。

    类别表以 P0 报告的 `class_map` 为准（它由 `vlm_config` 的 yolo.classes 派生并已与
    `classes.txt`/训练标签对齐）；要求 train_id 连续、feature_id/zh 唯一。
    """
    src_dir = os.path.abspath(src_dir)
    rp = os.path.join(src_dir, "split_report.json")
    if not os.path.isfile(rp):
        raise ClsDatasetError(f"源数据集缺 split_report.json：{src_dir}")
    with open(rp, encoding="utf-8") as f:
        report = json.load(f)
    cmap = sorted(report.get("class_map") or [], key=lambda c: int(c["train_id"]))
    if not cmap:
        raise ClsDatasetError("split_report 的 class_map 为空")
    ids = [int(c["train_id"]) for c in cmap]
    if ids != list(range(len(cmap))):
        raise ClsDatasetError(f"class_map 的 train_id 不连续：{ids}")
    for key in ("feature_id", "zh"):
        vals = [str(c.get(key, "")) for c in cmap]
        if any(not v for v in vals) or len(set(vals)) != len(vals):
            raise ClsDatasetError(f"class_map 的 {key} 缺失或重复")
    return src_dir, report, cmap


def parse_label(path, n_classes):
    """多类 YOLO 标签 → `[{"class_id", "cx", "cy", "w", "h"}]`（归一化）；坏行跳过并计数。"""
    boxes, bad = [], 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 5:
                bad += 1
                continue
            try:
                cid = int(parts[0])
                cx, cy, w, h = [float(v) for v in parts[1:5]]
            except ValueError:
                bad += 1
                continue
            if not 0 <= cid < n_classes:
                raise ClsDatasetError(f"{path}：class_id {cid} 超出类别表 0..{n_classes - 1}")
            if w <= 0 or h <= 0:
                bad += 1
                continue
            boxes.append({"class_id": cid, "cx": cx, "cy": cy, "w": w, "h": h})
    return boxes, bad


def _pixel_box(b, W, H):
    """归一化框 → 原图像素框（夹紧）；退化框返回 None。与 `geom.clamp_box` 同口径。"""
    x1 = min(max((b["cx"] - b["w"] / 2) * W, 0.0), float(W))
    x2 = min(max((b["cx"] + b["w"] / 2) * W, 0.0), float(W))
    y1 = min(max((b["cy"] - b["h"] / 2) * H, 0.0), float(H))
    y2 = min(max((b["cy"] + b["h"] / 2) * H, 0.0), float(H))
    if x2 - x1 < 1 or y2 - y1 < 1:
        return None
    return [x1, y1, x2, y2]


def _sha1_int(text):
    return int(hashlib.sha1(text.encode("utf-8")).hexdigest()[:12], 16)


def rebalance_split(plan, cmap, seed=DEFAULT_SEED, val_ratio=DEFAULT_VAL_RATIO, log=print):
    """把 P0 train+val 的**整组**按框数配额重划为 train/val（test 不参与）。

    **为什么不能沿用 P0 的 val**：P0 按图像配额划分，稀有类在少数组里、单组可带大量框，
    val 会反超 train（淀粉粒 val 259 vs train 80；导管 val 20 vs train 15）。
    分类器以 crop 为样本，配额应按**框数**算。

    规则：每类 val 目标 = val_ratio × 该类 pool 框数（至少 1，除非只有 1 框），
    且 val ≤ 该类 pool 的 50%（保证 train 至少一半）；组不可拆，允许少量超额。
    整组分配 + 稀有类优先 + 新药材加分（同分时按 seed 哈希定序，可复现）。
    """
    groups = {}
    total = Counter()
    for p in plan:
        key = p["rec"].get("group_key") or p["rec"].get("md5")
        g = groups.setdefault(key, {"herb": p["rec"].get("herb", ""),
                                    "counts": Counter(), "records": []})
        g["records"].append(p)
        for b in p["boxes"]:
            g["counts"][b["class_id"]] += 1
            total[b["class_id"]] += 1

    targets, caps = {}, {}
    for cid in range(len(cmap)):
        n = total.get(cid, 0)
        target = max(1, int(round(val_ratio * n))) if n > 1 else 0
        caps[cid] = max(0, n - max(1, int(math.ceil(0.5 * n))))
        targets[cid] = min(target, caps[cid])

    assigned = Counter()
    remaining = set(groups)
    val_groups, val_herbs = [], set()
    while remaining:
        best, best_key = None, None
        for key in remaining:
            g = groups[key]
            gain = sum(min(n, max(0, targets[c] - assigned[c]))
                       for c, n in g["counts"].items())
            if gain <= 0:
                continue
            # 超过该类 50% 上限的组不能进 val（即使是唯一候选）
            if any(assigned[c] + n > caps[c] for c, n in g["counts"].items() if n > 0):
                continue
            # 单组可能一次带来远超目标的框（淀粉粒一组 122 框）；给超额罚分，
            # 优先选“刚好填满剩余目标”的组，避免 val 被一个超大组撑爆。
            overshoot = sum(max(0, assigned[c] + n - targets[c])
                            for c, n in g["counts"].items() if n > 0)
            score = (gain - 0.5 * overshoot) * (1.15 if g["herb"] not in val_herbs else 1.0)
            if score <= 0:
                continue
            rarity = min((total[c] for c in g["counts"] if g["counts"][c] > 0),
                         default=10 ** 9)
            key_order = (score, -rarity, _sha1_int(f"{seed}:{key}"))
            if best_key is None or key_order > best_key:
                best, best_key = key, key_order
        if best is None:
            break
        g = groups[best]
        assigned.update(g["counts"])
        val_herbs.add(g["herb"])
        val_groups.append(best)
        remaining.remove(best)

    if not val_groups and groups:
        raise ClsDatasetError("rebalance 未能分配任何 val 组（每类 pool 框数过少）")

    split_by_md5 = {}
    for key in val_groups:
        for p in groups[key]["records"]:
            split_by_md5[p["rec"]["md5"]] = "val"
    for key in remaining:
        for p in groups[key]["records"]:
            split_by_md5[p["rec"]["md5"]] = "train"

    detail = {
        "seed": seed,
        "val_ratio": val_ratio,
        "pool_boxes": int(sum(total.values())),
        "val_groups": len(val_groups),
        "train_groups": len(remaining),
        "val_herbs": sorted(h for h in val_herbs if h),
        "targets": {
            cmap[cid]["zh"]: {"target": int(targets[cid]), "actual": int(assigned[cid]),
                              "pool_boxes": int(total.get(cid, 0)), "cap": int(caps[cid])}
            for cid in range(len(cmap))
        },
        "target_shortfall": [cmap[cid]["zh"] for cid in range(len(cmap))
                             if targets[cid] > 0 and assigned[cid] < targets[cid]],
        "train_floor_ok": all(assigned[cid] <= caps[cid] for cid in range(len(cmap))),
    }
    log(f"  [rebalance] val 组 {detail['val_groups']} / train 组 {detail['train_groups']}；"
        f"pool {detail['pool_boxes']} 框；val 药材 {detail['val_herbs']}；"
        f"目标未达 {detail['target_shortfall']}")
    return split_by_md5, detail


def _prepare(src_dir, report, cmap, split_policy="p0",
             val_ratio=DEFAULT_VAL_RATIO, seed=DEFAULT_SEED, log=print):
    """预检、按策略定 split、计数对账。任何源侧不一致直接报错（不产出半套）。"""
    n_classes = len(cmap)
    records = [r for r in report.get("images") or [] if r.get("split") in SPLITS]
    records.sort(key=lambda r: (SPLITS.index(r["split"]), r.get("md5", ""), r.get("name", "")))

    plan = []
    src_images = {s: 0 for s in SPLITS}
    src_boxes = {s: 0 for s in SPLITS}
    src_class_counts = {c["zh"]: {s: 0 for s in SPLITS} for c in cmap}
    n_negative_images = 0
    n_bad_lines = 0

    for r in records:
        src_split, name = r["split"], r["name"]
        stem = os.path.splitext(name)[0]
        # 源图/标签按 **P0 split** 目录取；rebalance 只改输出 split，不改源盘位置。
        img_path = os.path.join(src_dir, "images", src_split, name)
        lbl_path = os.path.join(src_dir, "labels", src_split, stem + ".txt")
        for p in (img_path, lbl_path):
            if not os.path.isfile(p):
                raise ClsDatasetError(f"源数据集缺文件：{p}")
        boxes, bad = parse_label(lbl_path, n_classes)
        n_bad_lines += bad
        src_images[src_split] += 1
        src_boxes[src_split] += len(boxes)
        if not boxes:
            n_negative_images += 1
        for b in boxes:
            src_class_counts[cmap[b["class_id"]]["zh"]][src_split] += 1
        plan.append({"rec": r, "src_split": src_split, "split": src_split,
                     "image": img_path, "label": lbl_path, "boxes": boxes})

    # 与 P0 报告逐项对账（图像数 / 框数 / 逐类框数）；不对账就说明源被改过，停止。
    for split in SPLITS:
        exp = (report.get("split_counts") or {}).get(split) or {}
        if exp and int(exp.get("images", -1)) != src_images[split]:
            raise ClsDatasetError(
                f"{split} 图像数 {src_images[split]} ≠ split_report {exp.get('images')}")
        if exp and int(exp.get("boxes", -1)) != src_boxes[split]:
            raise ClsDatasetError(
                f"{split} 框数 {src_boxes[split]} ≠ split_report {exp.get('boxes')}")
    for zh, per_split in (report.get("class_counts") or {}).items():
        if zh not in src_class_counts:
            raise ClsDatasetError(f"split_report 的 class_counts 含未知类：{zh}")
        for split in SPLITS:
            exp = int((per_split.get(split) or {}).get("boxes", 0))
            if exp != src_class_counts[zh][split]:
                raise ClsDatasetError(
                    f"{zh}/{split} 框数 {src_class_counts[zh][split]} ≠ split_report {exp}")

    rebalance = None
    if split_policy == "p0":
        split_by_md5 = {p["rec"]["md5"]: p["src_split"] for p in plan}
    else:
        # test 固定为 P0 test（分类器 test 必须与 stage 1 同图且未被分类器见过）；
        # train/val 只在 P0 train+val 的组池里重划。
        test_plan = [p for p in plan if p["src_split"] == "test"]
        pool = [p for p in plan if p["src_split"] in ("train", "val")]
        for cid, c in enumerate(cmap):
            exp = int(((report.get("class_counts") or {}).get(c["zh"], {})
                       .get("test", {}) or {}).get("boxes", 0))
            got = sum(1 for p in test_plan for b in p["boxes"] if b["class_id"] == cid)
            if got != exp:
                raise ClsDatasetError(f"test/{c['zh']} 框数 {got} ≠ split_report {exp}")
        rb, rebalance = rebalance_split(pool, cmap, seed=seed, val_ratio=val_ratio, log=log)
        if not rebalance["train_floor_ok"]:
            raise ClsDatasetError(f"rebalance 违反 train 下限：{rebalance['targets']}")
        split_by_md5 = {p["rec"]["md5"]: "test" for p in test_plan}
        split_by_md5.update(rb)
        for p in plan:
            p["split"] = split_by_md5[p["rec"]["md5"]]

        # 总量守恒 + test 不变 + 每类 train 至少一半
        for cid, c in enumerate(cmap):
            total_src = sum(src_class_counts[c["zh"]].values())
            total_new = sum(1 for p in plan for b in p["boxes"] if b["class_id"] == cid)
            if total_src != total_new:
                raise ClsDatasetError(f"rebalance 前后 {c['zh']} 框数不守恒")
            test_new = sum(1 for p in plan if p["split"] == "test"
                           for b in p["boxes"] if b["class_id"] == cid)
            exp = int(((report.get("class_counts") or {}).get(c["zh"], {})
                       .get("test", {}) or {}).get("boxes", 0))
            if test_new != exp:
                raise ClsDatasetError(f"rebalance 改了 test/{c['zh']}：{test_new} ≠ {exp}")
            pool_total = total_src - exp
            train_new = sum(1 for p in plan if p["split"] == "train"
                            for b in p["boxes"] if b["class_id"] == cid)
            if pool_total and train_new < max(1, int(math.ceil(0.5 * pool_total))):
                raise ClsDatasetError(f"rebalance 后 {c['zh']} train 低于 50%："
                                      f"{train_new}/{pool_total}")

    # 组必须整体在一个 split（P0 本身如此；rebalance 也按组分配）
    group_splits = {}
    for p in plan:
        group_splits.setdefault(p["rec"].get("group_key", ""), set()).add(p["split"])
    mixed = [g for g, ss in group_splits.items() if len(ss) > 1]
    if mixed:
        raise ClsDatasetError(f"组跨 split：{mixed[:5]}（共 {len(mixed)}）")

    # 最终 split 的计数（报告/输出用）
    split_images = {s: 0 for s in SPLITS}
    split_boxes = {s: 0 for s in SPLITS}
    class_counts = {c["zh"]: {s: 0 for s in SPLITS} for c in cmap}
    for p in plan:
        split = p["split"]
        split_images[split] += 1
        split_boxes[split] += len(p["boxes"])
        for b in p["boxes"]:
            class_counts[cmap[b["class_id"]]["zh"]][split] += 1

    stats = {
        "images": split_images,
        "boxes": split_boxes,
        "class_counts": class_counts,
        "source_split_images": src_images,
        "source_split_boxes": src_boxes,
        "source_class_counts": src_class_counts,
        "policy": split_policy,
        "val_ratio": val_ratio,
        "seed": seed,
        "rebalance": rebalance,
        "n_negative_images_skipped": n_negative_images,
        "n_malformed_label_lines": n_bad_lines,
    }
    return plan, stats


# ── crop 写出 ─────────────────────────────────────────────────────────

def _resize_long_side(crop, max_side):
    """长边只降不升（小 crop 保留原分辨率，训练侧再放大）。"""
    h, w = crop.shape[:2]
    if not max_side or max(h, w) <= max_side:
        return crop, 1.0
    scale = max_side / float(max(h, w))
    return cv2.resize(crop, (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
                      interpolation=cv2.INTER_AREA), scale


def _write_jpeg(path, crop, quality):
    ok, buf = cv2.imencode(".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise ClsDatasetError(f"JPEG 编码失败：{path}")
    with open(path, "wb") as f:      # Windows 文件 IO：不用 ndarray.tofile(str)
        f.write(buf.tobytes())
    return hashlib.md5(buf.tobytes()).hexdigest()


def build_cls_dataset(src_dir=paths.DET_TRAIN_V1, out_dir=paths.CLS_TRAIN_V1, *,
                      margin_ratio=DET_LOOSE_MARGIN_RATIO,
                      max_side=DEFAULT_MAX_SIDE, jpeg_quality=DEFAULT_JPEG_QUALITY,
                      split_policy="rebalance", val_ratio=DEFAULT_VAL_RATIO, seed=DEFAULT_SEED,
                      limit=0, overwrite=False, log=print):
    """构建 crop 数据集并返回 report dict。

    `split_policy="rebalance"`（默认）：test 固定 P0 test；P0 train+val 的组按框数配额重划
    train/val（分类器以 crop 为样本，避免稀有类 val 反超 train）。
    `split_policy="p0"`：原样复用 P0 train/val/test（用于与 stage 1 严格同划分的对照）。
    `limit > 0` 只处理前 N 张**源图**（冒烟/估算用，报告标 `partial=true` 并跳过总量断言）。
    """
    src_dir, report, cmap = load_source(src_dir)
    out_dir = os.path.abspath(out_dir)
    if os.path.abspath(src_dir) == out_dir:
        raise ClsDatasetError("输出目录不能等于源数据集目录（会删掉输入）")
    if os.path.isdir(out_dir) and os.listdir(out_dir) and not overwrite:
        raise ClsDatasetError(f"输出目录已存在且非空：{out_dir}（确认重建请加 --overwrite）")

    plan, src_stats = _prepare(src_dir, report, cmap, split_policy=split_policy,
                               val_ratio=val_ratio, seed=seed, log=log)
    n_plan = len(plan)
    if limit:
        plan = plan[:int(limit)]
    partial = bool(limit) and int(limit) < n_plan
    log(f"  [plan] split_policy={split_policy}，源图 {len(plan)} 张 / 框 "
        f"{sum(len(p['boxes']) for p in plan)} 个"
        f"{'（partial，仅前 %d 张）' % limit if limit else ''}")

    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    for split in SPLITS:
        for c in cmap:
            os.makedirs(os.path.join(out_dir, CROPS_DIR, split,
                                     f"{int(c['train_id']):02d}_{c['feature_id']}"),
                        exist_ok=True)

    rows = []
    crop_digest = hashlib.md5()
    saved = {s: 0 for s in SPLITS}
    skipped_degenerate = 0
    skipped_empty = 0
    edge_touch = 0
    giant = 0
    crop_w_list, crop_h_list, saved_long_list = [], [], []

    for i, p in enumerate(plan, 1):
        rec, boxes = p["rec"], p["boxes"]
        split, src_split = p["split"], p["src_split"]
        name, md5 = rec["name"], rec["md5"]
        img = imread_cn(p["image"])
        if img is None:
            raise ClsDatasetError(f"图像读取失败：{p['image']}")
        H, W = img.shape[:2]
        prepared = []
        for b in boxes:
            tight = _pixel_box(b, W, H)
            if tight is None:
                skipped_degenerate += 1
                continue
            prepared.append((b, tight))
        # 稳定顺序：类别 → y → x（与标签行顺序无关，重切两次产物逐字节一致）
        prepared.sort(key=lambda t: (t[0]["class_id"], t[1][1], t[1][0]))
        source_dirs = ";".join(
            f"{s.get('dir', '')}/{s.get('name', '')}" for s in (rec.get("sources") or []))

        for idx, (b, tight) in enumerate(prepared):
            loose = loose_box(tight, W, H, margin_ratio=margin_ratio)
            lx1, ly1, lx2, ly2 = loose
            if lx2 - lx1 < MIN_CROP_SIDE or ly2 - ly1 < MIN_CROP_SIDE:
                skipped_degenerate += 1
                continue
            crop = img[ly1:ly2, lx1:lx2]
            if crop.size == 0:
                skipped_empty += 1
                continue
            ch, cw = crop.shape[:2]
            crop, resize_scale = _resize_long_side(crop, max_side)
            sh, sw = crop.shape[:2]
            crop_id = f"{md5}_b{idx:02d}"
            feature_id = cmap[b["class_id"]]["feature_id"]
            folder = f"{int(b['class_id']):02d}_{feature_id}"
            dst = os.path.join(out_dir, CROPS_DIR, split, folder, crop_id + ".jpg")
            file_md5 = _write_jpeg(dst, crop, jpeg_quality)
            crop_digest.update(f"{crop_id}:{file_md5}\n".encode("utf-8"))
            saved[split] += 1
            crop_w_list.append(cw)
            crop_h_list.append(ch)
            saved_long_list.append(max(sh, sw))
            afrac = (tight[2] - tight[0]) * (tight[3] - tight[1]) / float(W * H)
            if afrac > 0.8:
                giant += 1
            touches = (tight[0] <= 1 or tight[1] <= 1 or tight[2] >= W - 1
                       or tight[3] >= H - 1)
            if touches:
                edge_touch += 1
            rows.append({
                "crop_id": crop_id, "split": split, "source_split": src_split,
                "class_id": b["class_id"], "feature_id": feature_id,
                "zh": cmap[b["class_id"]]["zh"],
                "source_image": f"images/{src_split}/{name}", "source_md5": md5,
                "herb": rec.get("herb", ""), "group_key": rec.get("group_key", ""),
                "source_dirs": source_dirs,
                "img_w": W, "img_h": H,
                "tight_box": ",".join(f"{v:.1f}" for v in tight),
                "loose_box": f"{lx1},{ly1},{lx2},{ly2}",
                "crop_w": cw, "crop_h": ch, "saved_w": sw, "saved_h": sh,
                "scale": round(sw / float(cw), 4),
                "box_area_frac": round(afrac, 4),
                "edge_touch": int(touches),
                "source_repeat": rec.get("repeat", 1),
            })
        if i % 200 == 0:
            log(f"    …已处理 {i}/{len(plan)} 图，crop {sum(saved.values())}")

    manifest_path = os.path.join(out_dir, MANIFEST_NAME)
    with open(manifest_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        w.writeheader()
        w.writerows(rows)
    manifest_md5 = _md5_file(manifest_path)

    classes_path = os.path.join(out_dir, CLASSES_NAME)
    with open(classes_path, "w", encoding="utf-8", newline="\n") as f:
        for c in cmap:
            f.write(f"{int(c['train_id']):02d}\t{c['feature_id']}\t{c['zh']}\n")
    classes_md5 = _md5_file(classes_path)

    out = _report(out_dir, src_dir, report, cmap, rows, src_stats, partial,
                  margin_ratio, max_side, jpeg_quality, skipped_degenerate,
                  skipped_empty, edge_touch, giant, crop_w_list, crop_h_list,
                  saved_long_list, manifest_md5, classes_md5, crop_digest.hexdigest())
    with open(os.path.join(out_dir, REPORT_NAME), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    log(f"  [done] {out_dir}：crop {len(rows)}（train {saved['train']} / "
        f"val {saved['val']} / test {saved['test']}），报告 {REPORT_NAME}")
    return out


# ── 报告 / 断言 ────────────────────────────────────────────────────────

def _report(out_dir, src_dir, src_report, cmap, rows, src_stats, partial,
            margin_ratio, max_side, jpeg_quality, skipped_degenerate, skipped_empty,
            edge_touch, giant, crop_w, crop_h, saved_long, manifest_md5,
            classes_md5, crops_digest):
    n = len(rows)
    counts = {c["zh"]: {s: 0 for s in SPLITS} for c in cmap}
    imgs = {s: set() for s in SPLITS}
    for r in rows:
        counts[r["zh"]][r["split"]] += 1
        imgs[r["split"]].add(r["source_image"])

    assertions = []

    def check(name, ok, detail):
        assertions.append({"name": name, "ok": bool(ok), "detail": detail})

    md5_by_split = {s: {r["source_md5"] for r in rows if r["split"] == s} for s in SPLITS}
    cross = [(a, b, len(md5_by_split[a] & md5_by_split[b]))
             for a, b in (("train", "val"), ("train", "test"), ("val", "test"))]
    check("cross_split_md5_disjoint", all(n == 0 for _, _, n in cross),
          f"交集 {[(a, b, k) for a, b, k in cross if k]}")
    group_map = {}
    for r in rows:
        group_map.setdefault(r["group_key"], set()).add(r["split"])
    mixed = sorted(g for g, s in group_map.items() if len(s) > 1)
    check("group_split_pure", not mixed, f"跨 split 组 {mixed[:5]}（共 {len(mixed)}）")
    check("gold_eval_excluded", all(r["split"] in SPLITS for r in rows),
          f"gold_eval 记录数 {sum(1 for r in rows if r['split'] == GOLD_SPLIT)}")
    ids = [r["crop_id"] for r in rows]
    check("crop_ids_unique", len(set(ids)) == len(ids),
          f"{len(ids)} 行 / {len(set(ids))} 唯一")
    check("manifest_rows", len(rows) == n, f"{len(rows)} 行")
    dirs_ok = all(
        os.path.isdir(os.path.join(out_dir, CROPS_DIR, s, f"{int(c['train_id']):02d}_{c['feature_id']}"))
        for s in SPLITS for c in cmap)
    check("class_dirs_complete", dirs_ok,
          f"{len(SPLITS)}×{len(cmap)} 个类目录")
    if not partial:
        check("source_box_total", sum(src_stats["boxes"].values()) == n,
              f"源框 {sum(src_stats['boxes'].values())} / crop {n}")
        policy = src_stats.get("policy", "p0")
        if policy == "p0":
            per_class_ok = all(
                counts[c["zh"]][s] == (src_report.get("class_counts", {})
                                       .get(c["zh"], {}).get(s, {}).get("boxes", 0))
                for c in cmap for s in SPLITS)
            check("class_counts_match_source", per_class_ok,
                  "逐类逐 split 与 split_report 一致" if per_class_ok else "存在不一致项")
        else:
            total_ok = all(
                sum(counts[c["zh"]].values())
                == sum((src_report.get("class_counts", {}).get(c["zh"], {})
                        .get(s, {}).get("boxes", 0)) for s in SPLITS)
                for c in cmap)
            check("class_totals_match_source", total_ok,
                  "逐类总数与 split_report 一致" if total_ok else "存在不一致项")
            test_ok = all(
                counts[c["zh"]]["test"]
                == (src_report.get("class_counts", {}).get(c["zh"], {})
                    .get("test", {}).get("boxes", 0))
                for c in cmap)
            check("test_split_unchanged", test_ok,
                  "test 与 P0 test 逐类一致" if test_ok else "test 被改动")
            rb = src_stats.get("rebalance") or {}
            check("rebalance_train_floor", bool(rb.get("train_floor_ok")),
                  "每类 train ≥ pool 的 50%" if rb.get("train_floor_ok") else "违反 train 下限")

    warnings = []
    rare = []
    for c in cmap:
        k = counts[c["zh"]]["train"]
        if k < RARE_TRAIN_CRITICAL:
            rare.append({"zh": c["zh"], "train_crops": k, "severity": "critical"})
        elif k < RARE_TRAIN_WARN:
            rare.append({"zh": c["zh"], "train_crops": k, "severity": "warn"})
    if rare:
        warnings.append({
            "code": "rare_train_class",
            "detail": "train crop 不足：分类器对这些类的召回不可信，训练需重采样/类权重；"
                      "不要用 val/test 的数当训练量",
            "classes": rare,
        })
    imbalanced = [{"zh": c["zh"], "train": counts[c["zh"]]["train"],
                   "val": counts[c["zh"]]["val"], "test": counts[c["zh"]]["test"]}
                  for c in cmap
                  if counts[c["zh"]]["train"] < max(counts[c["zh"]]["val"], counts[c["zh"]]["test"])]
    if imbalanced:
        cause = ("P0 稀有类图像配额所致" if src_stats.get("policy", "p0") == "p0"
                 else "rebalance 后仍存在（组不可拆，单组框数过大）")
        warnings.append({
            "code": "val_test_exceeds_train",
            "detail": f"val/test 的 crop 数不少于 train（{cause}）；这是**评测集**，"
                      "不能并入训练",
            "classes": imbalanced,
        })
    rb = src_stats.get("rebalance") or {}
    if rb.get("target_shortfall"):
        warnings.append({
            "code": "val_target_shortfall",
            "detail": "rebalance 未能达到这些类的 val 框数目标（剩余组超过 50% 上限或组数不足）",
            "classes": rb["target_shortfall"],
        })
    warnings.append({
        "code": "no_reject_class",
        "detail": "本数据集只有 14 个 GT 类，**没有**『都不是/非特征』类；stage 1 假阳性的"
                  "crop 会被强制归到某类。补齐需回流检测 FP crop（stage 2 方案 §3 第 5 条）",
    })
    if n and edge_touch / n > 0.2:
        warnings.append({"code": "edge_touch",
                         "detail": f"{edge_touch}/{n}（{edge_touch / n:.1%}）框贴原图边缘，"
                                   "松框被画面截断；与推理侧同一限制"})
    if n and giant / n > 0.02:
        warnings.append({"code": "giant_box",
                         "detail": f"{giant}/{n}（{giant / n:.1%}）框占原图 >80%，"
                                   "crop 近似整图，类别上下文弱"})
    if src_stats["n_negative_images_skipped"]:
        warnings.append({"code": "negative_images_skipped",
                         "detail": f"{src_stats['n_negative_images_skipped']} 张无框图未产 crop"
                                   "（本数据集不产负样本）"})

    def _q(vals, p):
        if not vals:
            return None
        vals = sorted(vals)
        return int(round(vals[int(p * (len(vals) - 1))]))

    return {
        "version": os.path.basename(os.path.normpath(out_dir)),
        "source": {
            "dataset": os.path.relpath(src_dir, paths.PROJECT_ROOT).replace(os.sep, "/"),
            "split_report_md5": _md5_file(os.path.join(src_dir, "split_report.json")),
            "source_version": src_report.get("version", ""),
            "images": len({r["source_image"] for r in rows}),
            "boxes": n,
            "gold_eval_excluded": True,
            "gold_eval_count": src_report.get("gold_eval_count"),
            "source_split_counts": {
                s: {"images": src_stats["source_split_images"][s],
                    "boxes": src_stats["source_split_boxes"][s]}
                for s in SPLITS
            },
            "split_remapped": src_stats.get("policy", "p0") == "rebalance",
            "n_negative_images_skipped": src_stats["n_negative_images_skipped"],
            "n_malformed_label_lines": src_stats["n_malformed_label_lines"],
        },
        "params": {
            "split_policy": (
                "复用 P0 检测训练集 train/val/test（按 group_key 整组、整味留出）"
                if src_stats.get("policy", "p0") == "p0" else
                "rebalance：test 固定 P0 test；P0 train+val 的组按框数配额 + 药材分层"
                "重划 train/val（每类 val ≈ val_ratio × pool 框数，且 ≤ pool 的 50%）"),
            "split_policy_code": src_stats.get("policy", "p0"),
            "val_ratio": src_stats.get("val_ratio"),
            "seed": src_stats.get("seed"),
            "crop_rule": "geom.loose_box(短边 × margin_ratio, 下限 24px)",
            "margin_ratio": margin_ratio,
            "max_side": max_side,
            "jpeg_quality": jpeg_quality,
            "class_folders": "<train_id:02d>_<feature_id>",
            "class_rule": "14 类含『其他』；无拒绝类",
        },
        "rebalance": src_stats.get("rebalance"),
        "split_counts": {
            s: {"source_images": len(imgs[s]), "crops": sum(counts[c["zh"]][s] for c in cmap)}
            for s in SPLITS
        },
        "class_counts": {c["zh"]: counts[c["zh"]] for c in cmap},
        "crop_size": {
            "crop_w_p50": _q(crop_w, 0.5), "crop_h_p50": _q(crop_h, 0.5),
            "crop_w_p90": _q(crop_w, 0.9), "crop_h_p90": _q(crop_h, 0.9),
            "saved_long_side_p50": _q(saved_long, 0.5),
            "saved_long_side_p90": _q(saved_long, 0.9),
            "skipped_degenerate_boxes": skipped_degenerate,
            "skipped_empty_crops": skipped_empty,
        },
        "class_map": [
            {"train_id": int(c["train_id"]), "feature_id": c["feature_id"],
             "zh": c["zh"], "delivery": bool(c.get("delivery", True))}
            for c in cmap
        ],
        "warnings": warnings,
        "assertions": assertions,
        "partial": partial,
        "fingerprint": {
            "n_crops": n,
            "manifest_md5": manifest_md5,
            "classes_md5": classes_md5,
            "crops_digest": crops_digest,
        },
    }
