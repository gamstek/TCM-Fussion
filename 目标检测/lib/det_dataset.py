# -*- coding: utf-8 -*-
"""det_dataset.py — P0 数据工程：把 14 个单类 YOLO 导出集合并成一个可信的多类训练集

**为什么存在**：`cli/export_det_dataset.py` 是按类逐目录导出的，每个目录只有本类标签，
且 753 个文件名跨目录重复（同名不同图）。直接拼目录会把不同图的标签混到一张图上
（见 docs/目标检测_文献调研与实现方案_20261030.md §4.2 / §7）。本模块做四件事：

1. **以内容 md5 为键合并**（不是文件名），重复内容标签取并集；
2. **按 group_key 组感知划分** train/val/test，留整味药材做跨药材 test；
3. **train-only 稀有类重采样**（写进 train_list.txt 的重复行，不在磁盘复制图片）；
4. 出可复跑的审计/断言/指纹（`split_report.json` + `dataset_fingerprint.json`）。

**分层**：本文件是具名库（无 `__main__`、不 import 入口脚本）；入口在
`目标检测/cli/build_det_dataset.py`，离线断言在 `目标检测/tests/test_det_dataset.py`。

**不变量**（与方案文档 §6.2 对齐）：
- 输出 `images/<split>/` 里同一内容只出现一次（重复靠 train_list.txt 表达）；
- 同一 `group_key` 只落一个 split；hold-out 药材不进 train/val；
- `其他` 是第 14 个**正类**，正常参与 train/val/test；但它没有 IGM 特征描述，
  所以 `delivery=False`，不进入交付类集合；
- 类别 id 在输出里**连续重编**（`vlm_config.json` 的全局 id 有 pending 空洞，
  YOLO 要求 0..N-1）。
"""
import csv
import hashlib
import json
import math
import os
import random
import shutil
from collections import Counter, defaultdict

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp")
SPLITS = ("train", "val", "test")
OTHER_ZH = "其他"
PENDING_IDS = ("pending",)
PENDING_ZH = ("疑似待定",)

# 划分默认值：方案文档 §6.2.1
DEFAULT_SEED = 20261030
DEFAULT_VAL_RATIO = 0.15
DEFAULT_TEST_RATIO = 0.15
MIN_SPLIT_IMAGES = 10          # 每类 val / test 至少 10 张图
MIN_SPLIT_BOXES = 20           # 每类 val / test 至少 20 个框
MAX_SPLIT_FRAC = 0.35          # 单类单 split 上限（防止稀有类被切空 train）
MIN_TRAIN_FRAC = 0.20          # 组粒度会放大配额：任何类至少留 20% 给 train（下限 1 张）
DEFAULT_RARE_THRESHOLD = 100   # train 图数 < 该值的类视为稀有类
DEFAULT_MAX_REPEAT = 3         # 稀有类 train-only 重采样上限
DEFAULT_HOLDOUT_MAX = 2
DEFAULT_HOLDOUT_MIN_IMAGES = 20
DEFAULT_HOLDOUT_MIN_BOXES = 20
EDGE_EPS = 0.005               # 归一化坐标下距边框 0.5% 内算“贴边”
GIANT_AREA = 0.50              # 单框面积 > 画面 50% 视为巨型框
TINY_AREA = 0.005              # 单框面积 < 画面 0.5% 视为小框


class DatasetError(RuntimeError):
    """数据工程断言失败；消息里带可回查的路径/键。"""


# ── 基础工具 ────────────────────────────────────────────────────────────

def md5_file(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha1_text(text):
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


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
        raise DatasetError("vlm_config.json 的 yolo.classes 为空，无法建训练集")
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


def _class_index(table):
    by_zh = {}
    for c in table:
        if c["zh"] in by_zh:
            raise DatasetError(f"类别中文名重复：{c['zh']}")
        by_zh[c["zh"]] = c
    return by_zh


def _iter_class_dirs(src_dir):
    if not os.path.isdir(src_dir):
        raise DatasetError(f"源目录不存在：{src_dir}")
    for name in sorted(os.listdir(src_dir)):
        p = os.path.join(src_dir, name)
        if os.path.isdir(os.path.join(p, "images")) and os.path.isdir(os.path.join(p, "labels")):
            yield name, p


def _read_classes_txt(path):
    if not os.path.isfile(path):
        raise DatasetError(f"缺 classes.txt：{path}")
    with open(path, encoding="utf-8-sig") as f:
        names = [ln.strip() for ln in f.read().splitlines() if ln.strip()]
    if not names:
        raise DatasetError(f"classes.txt 为空：{path}")
    return names


def _read_attributes(path):
    """attributes.csv → {image: {herb, group_key}}；同图多框必须元数据一致。"""
    if not os.path.isfile(path):
        return {}
    out, bad = {}, []
    with open(path, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            img = (r.get("image") or "").strip()
            if not img:
                continue
            meta = {"herb": (r.get("herb") or "").strip(),
                    "group_key": (r.get("group_key") or "").strip()}
            old = out.get(img)
            if old and (old["herb"] != meta["herb"] or old["group_key"] != meta["group_key"]):
                bad.append(img)
            out[img] = meta
    if bad:
        raise DatasetError(f"同一张图的 herb/group_key 不一致：{sorted(set(bad))[:5]}（{path}）")
    return out


def _parse_label(path, local_names, zh_by_name):
    """读 YOLO txt（本目录的局部 class id）→ 全局类别盒子列表。空文件 = 负样本。"""
    boxes = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) != 5:
                raise DatasetError(f"{path}:{lineno} 不是 5 列 YOLO 行：{line!r}")
            try:
                cid = int(parts[0])
                cx, cy, w, h = (float(x) for x in parts[1:])
            except ValueError as e:
                raise DatasetError(f"{path}:{lineno} 数值解析失败：{line!r}") from e
            if cid < 0 or cid >= len(local_names):
                raise DatasetError(f"{path}:{lineno} class id {cid} 超出 classes.txt 范围")
            zh = local_names[cid]
            c = zh_by_name.get(zh)
            if not c:
                raise DatasetError(f"{path}:{lineno} 类别「{zh}」不在 vlm_config.json 的 yolo.classes")
            if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < w <= 1 and 0 < h <= 1):
                raise DatasetError(f"{path}:{lineno} 坐标越界：{cx} {cy} {w} {h}")
            boxes.append({"zh": zh, "feature_id": c["feature_id"], "train_id": c["train_id"],
                          "cx": cx, "cy": cy, "w": w, "h": h})
    return boxes


def _verify_export(dir_path, actual_images, actual_boxes, actual_negatives):
    """与 export_report.json 对账；源被手工改动过就停下，避免在错误源上建集。"""
    rep_path = os.path.join(dir_path, "export_report.json")
    if not os.path.isfile(rep_path):
        return None
    with open(rep_path, encoding="utf-8") as f:
        rep = json.load(f)
    for key, actual in (("images", actual_images), ("boxes", actual_boxes),
                        ("negatives", actual_negatives)):
        if key in rep and int(rep[key]) != actual:
            raise DatasetError(
                f"{dir_path} 的 export_report.json.{key}={rep[key]}，磁盘实际 {actual}"
                f"（源目录被改动过；先回到导出一致状态，或删掉该报告再建集）")
    return rep


# ── 1. 扫描单类导出集 ──────────────────────────────────────────────────

def scan_source(src_dir, table, log=print):
    """扫描 14 个单类目录 → 每文件一条记录 + 每类源统计。"""
    zh_by_name = _class_index(table)
    records, src_stats = [], {}
    for dir_name, dir_path in _iter_class_dirs(src_dir):
        if dir_name not in zh_by_name:
            raise DatasetError(
                f"目录「{dir_name}」不是 vlm_config.json 的 yolo.classes 类别（pending 不入训练）")
        local_names = _read_classes_txt(os.path.join(dir_path, "classes.txt"))
        for zh in local_names:
            if zh not in zh_by_name:
                raise DatasetError(f"{dir_path}/classes.txt 的「{zh}」不在全局类别表")
        attrs = _read_attributes(os.path.join(dir_path, "attributes.csv"))
        img_dir = os.path.join(dir_path, "images")
        lbl_dir = os.path.join(dir_path, "labels")

        images = sorted(n for n in os.listdir(img_dir)
                        if os.path.splitext(n)[1].lower() in IMAGE_EXTS)
        stems = {os.path.splitext(n)[0] for n in images}
        label_files = {os.path.splitext(n)[0] for n in os.listdir(lbl_dir) if n.endswith(".txt")}
        if stems - label_files:
            raise DatasetError(f"{dir_path} 有图无标签：{sorted(stems - label_files)[:5]}")
        if label_files - stems:
            raise DatasetError(f"{dir_path} 有标签无图：{sorted(label_files - stems)[:5]}")

        n_boxes, n_neg = 0, 0
        for name in images:
            stem, ext = os.path.splitext(name)
            label_path = os.path.join(lbl_dir, stem + ".txt")
            boxes = _parse_label(label_path, local_names, zh_by_name)
            if boxes:
                n_boxes += len(boxes)
            else:
                n_neg += 1
            meta = attrs.get(name)
            if boxes and not meta:
                raise DatasetError(
                    f"{dir_path}/images/{name} 有框但 attributes.csv 没有该图元数据，"
                    f"无法按 group_key 划分")
            if meta and (not meta["herb"] or not meta["group_key"]):
                raise DatasetError(
                    f"{dir_path}/images/{name} 的 attributes.csv 缺 herb/group_key，无法组感知划分")
            records.append({
                "md5": md5_file(os.path.join(img_dir, name)),
                "ext": ext.lower(), "stem": stem, "name": name,
                "dir": dir_name, "herb": (meta or {}).get("herb", ""),
                "group_key": (meta or {}).get("group_key", ""),
                "boxes": boxes, "meta_source": "attributes" if meta else "missing",
                "path": os.path.join(img_dir, name),
                "src_box_lines": len(boxes),
            })

        rep = _verify_export(dir_path, len(images), n_boxes, n_neg)
        src_stats[dir_name] = {"files": len(images), "boxes": n_boxes, "negatives": n_neg,
                               "unique_md5": len({r["md5"] for r in records if r["dir"] == dir_name}),
                               "export_report": rep}
        log(f"  [src] {dir_name}: {len(images)} 图 / {n_boxes} 框 / {n_neg} 空图")

    if not records:
        raise DatasetError(f"{src_dir} 下没有可用的单类导出集")
    return records, src_stats


def resolve_negative_meta(records, manifest_path="", log=print):
    """空图（无框）在 attributes.csv 里没有行 → 用 manifest.csv 按**内容 md5** 回查 herb/group_key。

    文件名在多个药材目录里重复，所以只按 basename 匹配会张冠李戴；这里必须内容匹配。
    找不到就退化成 `neg_<md5>` 独立组（内容唯一，不会泄漏）。
    """
    negatives = [r for r in records if not r["boxes"] and not r["herb"]]
    stats = {"n_negatives": len(negatives), "resolved": 0, "fallback": 0, "ambiguous": 0}
    if not negatives:
        return stats
    if not manifest_path or not os.path.isfile(manifest_path):
        log("  [warn] 没有 manifest.csv，空图按独立组处理（herb 未知）")
        for r in negatives:
            r["group_key"] = f"neg_{r['md5'][:10]}"
            r["meta_source"] = "fallback"
        stats["fallback"] = len(negatives)
        return stats

    by_name = defaultdict(list)
    with open(manifest_path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            p = (row.get("image_path") or "").strip()
            if p:
                by_name[os.path.basename(p).lower()].append(row)
    md5_cache = {}

    def _md5(p):
        if p not in md5_cache:
            md5_cache[p] = md5_file(p) if os.path.isfile(p) else ""
        return md5_cache[p]

    for r in negatives:
        cands = by_name.get((r["stem"] + r["ext"]).lower(), [])
        hits = [row for row in cands if _md5(row["image_path"].strip()) == r["md5"]]
        if len(hits) > 1:
            stats["ambiguous"] += 1
        if hits:
            row = sorted(hits, key=lambda x: x["image_path"])[0]
            r["herb"] = (row.get("herb") or "").strip()
            r["group_key"] = (row.get("group_key") or "").strip() or f"{r['herb']}-{r['stem']}"
            r["meta_source"] = "manifest_md5"
            stats["resolved"] += 1
        else:
            r["group_key"] = f"neg_{r['md5'][:10]}"
            r["meta_source"] = "fallback"
            stats["fallback"] += 1
    log(f"  [neg] 空图 {len(negatives)} 张：manifest 按内容匹配 {stats['resolved']}，"
        f"退化独立组 {stats['fallback']}，多候选 {stats['ambiguous']}")
    return stats


# ── 2. 按内容 md5 合并 + 组键并查集 ───────────────────────────────────

class _UnionFind:
    def __init__(self):
        self.parent = {}

    def add(self, x):
        self.parent.setdefault(x, x)

    def find(self, x):
        self.add(x)
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            if ra > rb:
                ra, rb = rb, ra
            self.parent[rb] = ra

    def members(self):
        out = defaultdict(set)
        for x in list(self.parent):
            out[self.find(x)].add(x)
        return out


def _box_key(b):
    return (b["zh"], round(b["cx"], 6), round(b["cy"], 6), round(b["w"], 6), round(b["h"], 6))


def merge_by_md5(records, log=print):
    """同内容只留一条；组键用并查集合并（同内容出现在不同 group 时必须锁成一个组）。"""
    by_md5 = defaultdict(list)
    for r in records:
        by_md5[r["md5"]].append(r)

    uf = _UnionFind()
    for md5, rs in by_md5.items():
        keys = [r["group_key"] or f"neg_{md5[:10]}" for r in rs]
        for k in keys:
            uf.add(k)
        for k in keys[1:]:
            uf.union(keys[0], k)
    # 并查集只算一次：4,944 条记录 × 1,074 个组键，逐条重算 members() 会变 O(n²)
    canon = {k: min(members) for members in uf.members().values() for k in members}

    merged, collapsed_by_class, duplicate_groups = [], Counter(), []
    for md5, rs in sorted(by_md5.items()):
        rs = sorted(rs, key=lambda r: (r["dir"], r["name"]))
        group_key = canon[rs[0]["group_key"] or f"neg_{md5[:10]}"]
        boxes, seen_boxes = [], set()
        for r in rs:
            for b in r["boxes"]:
                k = _box_key(b)
                if k not in seen_boxes:
                    seen_boxes.add(k)
                    boxes.append(b)
        boxes.sort(key=lambda b: (b["train_id"], b["cx"], b["cy"], b["w"], b["h"]))

        # 对账：每个源目录的框行数 - 同目录内重复框 = 合并后该类框数
        per_dir_lines = Counter()
        per_dir_union = defaultdict(set)
        for r in rs:
            for b in r["boxes"]:
                per_dir_lines[b["zh"]] += 1
                per_dir_union[b["zh"]].add(_box_key(b))
        local_collapsed = 0
        for zh, n in per_dir_lines.items():
            c = n - len(per_dir_union[zh])
            collapsed_by_class[zh] += c
            local_collapsed += c

        herbs = sorted({r["herb"] for r in rs if r["herb"]})
        merged.append({
            "md5": md5, "ext": rs[0]["ext"], "stem": rs[0]["stem"],
            "herb": "|".join(herbs), "herbs": herbs,
            "group_key": group_key,
            "boxes": boxes,
            "srcs": [{"dir": r["dir"], "name": r["name"]} for r in rs],
            "src_dirs": sorted({r["dir"] for r in rs}),
            "paths": [r["path"] for r in rs],
            "n_srcs": len(rs),
            "src_box_lines": sum(r["src_box_lines"] for r in rs),
            "meta_source": ("attributes" if any(r["src_box_lines"] for r in rs)
                            else (rs[0]["meta_source"] if len(rs) == 1 else "merged_negative")),
        })
        if len(rs) > 1:
            duplicate_groups.append({
                "md5": md5,
                "files": [{"dir": r["dir"], "name": r["name"]} for r in rs],
                "boxes_collapsed": local_collapsed,
            })

    stats = {
        "n_input_files": len(records), "n_unique": len(merged),
        "n_duplicate_groups": len(duplicate_groups),
        "n_duplicate_files": len(records) - len(merged),
        "n_cross_dir_duplicate_groups": sum(
            1 for g in duplicate_groups if len({x["dir"] for x in g["files"]}) > 1),
        "collapsed_boxes_by_class": dict(collapsed_by_class),
        "duplicate_groups": duplicate_groups,
        "mixed_herb_groups": [{"md5": r["md5"], "herbs": r["herbs"]}
                              for r in merged if len(r["herbs"]) > 1],
    }
    log(f"  [merge] {len(records)} 文件 → {len(merged)} 唯一内容；"
        f"重复组 {stats['n_duplicate_groups']}（跨目录 {stats['n_cross_dir_duplicate_groups']}）；"
        f"折叠重复框 {sum(collapsed_by_class.values())}")
    return merged, stats


# ── 3. 整味药材 hold-out 与组感知划分 ─────────────────────────────────

def _class_counts(records):
    """按类统计图片/框（多类图在每类各计一次；只用于配额，不用于总数）。"""
    out = defaultdict(lambda: {"images": 0, "boxes": 0})
    for r in records:
        zhs = sorted({b["zh"] for b in r["boxes"]})
        for zh in zhs:
            out[zh]["images"] += 1
        for b in r["boxes"]:
            out[b["zh"]]["boxes"] += 1
    return out


def select_holdout_herbs(records, holdout_max=DEFAULT_HOLDOUT_MAX,
                         min_train_images=DEFAULT_HOLDOUT_MIN_IMAGES,
                         min_train_boxes=DEFAULT_HOLDOUT_MIN_BOXES,
                         explicit=None, log=print):
    """选整味药材做跨药材 test：优先覆盖类别多、样本多的药材，且不得掏空任何类。"""
    herb_classes = defaultdict(Counter)      # herb -> Counter(zh -> images)
    herb_boxes = defaultdict(Counter)        # herb -> Counter(zh -> boxes)
    herb_images = Counter()
    for r in records:
        for h in r["herbs"]:
            herb_images[h] += 1
            for zh in {b["zh"] for b in r["boxes"]}:
                herb_classes[h][zh] += 1
            for b in r["boxes"]:
                herb_boxes[h][b["zh"]] += 1

    total = _class_counts(records)

    if explicit:
        picked = [h.strip() for h in explicit if h.strip()]
        missing = [h for h in picked if h not in herb_images]
        if missing:
            raise DatasetError(f"--holdout-herbs 指定的药材不在数据里：{missing}")
    else:
        picked = []
        remaining = {zh: {"images": v["images"], "boxes": v["boxes"]} for zh, v in total.items()}
        candidates = sorted(
            (h for h in herb_images
             if sum(1 for c, n in herb_classes[h].items() if n > 0) >= 2),
            key=lambda h: (-sum(1 for c, n in herb_classes[h].items() if n > 0),
                           -herb_images[h], h))
        for h in candidates:
            if len(picked) >= holdout_max:
                break
            ok = True
            for zh, n in herb_classes[h].items():
                if remaining[zh]["images"] - n < min_train_images:
                    ok = False
                    break
            for zh, n in herb_boxes[h].items():
                if remaining[zh]["boxes"] - n < min_train_boxes:
                    ok = False
                    break
            if not ok:
                log(f"  [holdout] 跳过 {h}（会让某类 train 低于配额）")
                continue
            picked.append(h)
            for zh, n in herb_classes[h].items():
                remaining[zh]["images"] -= n
            for zh, n in herb_boxes[h].items():
                remaining[zh]["boxes"] -= n

    # 显式指定也要过同一道约束（方案文档：不得让某类 train 归零/掏空）
    if picked:
        remain = {zh: {"images": v["images"], "boxes": v["boxes"]} for zh, v in total.items()}
        for h in picked:
            for zh, n in herb_classes[h].items():
                remain[zh]["images"] -= n
            for zh, n in herb_boxes[h].items():
                remain[zh]["boxes"] -= n
        bad = [zh for zh, v in remain.items()
               if v["images"] < min_train_images or v["boxes"] < min_train_boxes]
        if bad:
            raise DatasetError(
                f"hold-out 药材 {picked} 会让这些类的 train 低于配额"
                f"（<{min_train_images} 图或 <{min_train_boxes} 框）：{bad}")
    log(f"  [holdout] 跨药材 test 药材：{picked or '（无）'}")
    return {"herbs": picked,
            "herb_classes": {h: dict(herb_classes[h]) for h in picked},
            "herb_images": {h: herb_images[h] for h in picked}}


def _group_table(records, holdout_herbs, gold_md5):
    """只把**非 gold_eval**记录纳入划分；gold 内容整体预留评测，不参与任何 split。"""
    groups = defaultdict(lambda: {"records": [], "class_images": Counter(),
                                  "class_boxes": Counter(), "herbs": set()})
    for r in records:
        if r["md5"] in gold_md5:
            continue
        g = groups[r["group_key"]]
        g["records"].append(r)
        g["herbs"].update(r["herbs"])
        for zh in {b["zh"] for b in r["boxes"]}:
            g["class_images"][zh] += 1
        for b in r["boxes"]:
            g["class_boxes"][b["zh"]] += 1
    for g in groups.values():
        g["is_holdout"] = bool(g["herbs"] & set(holdout_herbs))
    return groups


def _quota(n_images, n_boxes, ratio, max_frac=MAX_SPLIT_FRAC):
    """单类单 split 配额：比例、最小图数、最小框数三者取大，再用上限封顶。"""
    if n_images <= 0:
        return 0
    q = max(math.ceil(ratio * n_images), min(MIN_SPLIT_IMAGES, n_images))
    if n_boxes > 0:
        per_img = n_boxes / n_images
        q = max(q, min(math.ceil(MIN_SPLIT_BOXES / per_img), n_images))
    return int(min(q, max(1, math.floor(max_frac * n_images))))


def plan_split(records, holdout_herbs, seed=DEFAULT_SEED,
               val_ratio=DEFAULT_VAL_RATIO, test_ratio=DEFAULT_TEST_RATIO,
               gold_md5=(), log=print):
    """按 group_key 整组划分；gold 内容整体预留（split=gold_eval，不参与 train/val/test）。"""
    holdout_herbs = list(holdout_herbs or [])
    gold_md5 = set(gold_md5 or ())

    # gold_eval：det_eval 的 60 张人工盲标图必须整帧退出训练；同组其它图仍可正常划分
    # （组内近似图的风险记入报告；若把整组都推出，淀粉粒/导管这类小类会被掏空 —— 实测过）。
    gold_records = [r for r in records if r["md5"] in gold_md5]
    for r in gold_records:
        r["split"] = "gold_eval"
        r["cross_herb"] = False
        r["repeat"] = 1

    groups = _group_table(records, holdout_herbs, gold_md5)

    # 同域 = 不含 hold-out 药材、且不含 gold_eval 的组
    domain_groups = {k: g for k, g in groups.items() if not g["is_holdout"]}
    domain_records = [r for g in domain_groups.values() for r in g["records"]]
    totals = _class_counts(domain_records)

    quotas = {}
    for zh, v in totals.items():
        quotas[zh] = {"val": _quota(v["images"], v["boxes"], val_ratio),
                      "test": _quota(v["images"], v["boxes"], test_ratio)}

    assigned = {"val": Counter(), "test": Counter()}
    for key, g in sorted(groups.items()):
        if g["is_holdout"]:
            for r in g["records"]:
                r["split"] = "test"
                r["cross_herb"] = True

    # 剩余可进 train 的同域图数（hold-out 不在 totals 里）
    remaining_after = Counter({zh: v["images"] for zh, v in totals.items()})
    train_floor = {zh: max(1, int(math.floor(MIN_TRAIN_FRAC * v["images"])))
                   for zh, v in totals.items()}

    remaining = [k for k, g in groups.items() if not g["is_holdout"]]

    def _allowed(split, imgs):
        for zh, n in imgs.items():
            if remaining_after[zh] - n < train_floor[zh]:
                return False
        return True

    def _order_key(key):
        g = groups[key]
        classes = g["class_images"]
        rarity = min(totals[c]["images"] for c in classes) if classes else 10 ** 9
        return (rarity, _sha1_text(f"{seed}:{key}"))

    remaining.sort(key=_order_key)
    total_quota = {s: sum(quotas[zh][s] for zh in quotas) for s in ("val", "test")}

    for key in remaining:
        g = groups[key]
        imgs = g["class_images"]
        if not imgs:
            for r in g["records"]:
                r["split"] = "train"
                r["cross_herb"] = False
            continue
        gains = {}
        for split in ("val", "test"):
            if not _allowed(split, imgs):
                gains[split] = 0
                continue
            need = sum(min(n, max(0, quotas[zh][split] - assigned[split][zh]))
                       for zh, n in imgs.items())
            denom = sum(quotas[zh][split] for zh in imgs) or 1
            gains[split] = need / denom
        if gains["val"] <= 0 and gains["test"] <= 0:
            split = "train"
        elif gains["val"] != gains["test"]:
            split = "val" if gains["val"] > gains["test"] else "test"
        else:
            ratio = {s: sum(assigned[s].values()) / (total_quota[s] or 1) for s in ("val", "test")}
            split = "val" if ratio["val"] <= ratio["test"] else "test"
        for r in g["records"]:
            r["split"] = split
            r["cross_herb"] = False
        if split in assigned:
            for zh, n in imgs.items():
                assigned[split][zh] += n
                remaining_after[zh] -= n

    # 未达标项（分组粒度决定无法精确命中；记录到报告，不当作失败）
    unmet, box_unmet = {}, {}
    actual = {zh: {s: {"images": 0, "boxes": 0} for s in ("val", "test")} for zh in quotas}
    for r in records:
        if r["split"] not in ("val", "test"):
            continue
        for zh in {b["zh"] for b in r["boxes"]}:
            actual[zh][r["split"]]["images"] += 1
        for b in r["boxes"]:
            actual[b["zh"]][r["split"]]["boxes"] += 1
    for zh, q in quotas.items():
        for split in ("val", "test"):
            if actual[zh][split]["images"] < q[split]:
                unmet.setdefault(zh, {})[split] = {"target": q[split],
                                                   "actual": actual[zh][split]["images"]}
            if actual[zh][split]["boxes"] < MIN_SPLIT_BOXES:
                box_unmet.setdefault(zh, {})[split] = {
                    "target": MIN_SPLIT_BOXES, "actual": actual[zh][split]["boxes"]}
    gold_groups = {r["group_key"] for r in gold_records}
    same_group_active = sum(1 for r in records
                            if r["split"] != "gold_eval" and r["group_key"] in gold_groups)
    gold_overlap = {"n_gold_md5": len(gold_md5),
                    "matched_records": len(gold_records),
                    "excluded_records": len(gold_records),
                    "groups_touched": sorted(gold_groups),
                    "same_group_active_records": same_group_active,
                    "policy": "gold_eval（不参与 train/val/test；同组其它图正常划分）"}
    for r in records:
        r.setdefault("split", "train")
        r.setdefault("cross_herb", False)

    policy = {"seed": seed, "val_ratio": val_ratio, "test_ratio": test_ratio,
              "min_split_images": MIN_SPLIT_IMAGES, "min_split_boxes": MIN_SPLIT_BOXES,
              "max_split_frac": MAX_SPLIT_FRAC, "min_train_frac": MIN_TRAIN_FRAC,
              "quotas": quotas, "unmet": unmet, "box_unmet": box_unmet,
              "holdout_groups": sorted(k for k, g in groups.items() if g["is_holdout"]),
              "train_floor": train_floor}
    log(f"  [split] 组 {len(groups)}；hold-out 组 {len(policy['holdout_groups'])}；"
        f"gold 预留 {len(gold_records)} 张；未达配额类 {len(unmet)}")
    return policy, gold_overlap


# ── 4. 稀有类 train-only 重采样 ───────────────────────────────────────

def oversample_train(records, rare_threshold=DEFAULT_RARE_THRESHOLD,
                     max_repeat=DEFAULT_MAX_REPEAT, log=print):
    train = [r for r in records if r["split"] == "train"]
    counts = _class_counts(train)
    repeat_of = {}
    for zh, v in counts.items():
        n = v["images"]
        if n and n < rare_threshold:
            repeat_of[zh] = min(max_repeat, max(1, math.ceil(rare_threshold / n)))
    for r in records:
        if r["split"] != "train":
            r["repeat"] = 1
            continue
        reps = [repeat_of.get(zh, 1) for zh in {b["zh"] for b in r["boxes"]}]
        r["repeat"] = max(reps) if reps else 1
    table = {zh: {"train_images": counts[zh]["images"], "repeat": repeat_of.get(zh, 1)}
             for zh in repeat_of}
    log("  [balance] train-only 重采样：" +
        (", ".join(f"{zh}×{v['repeat']}" for zh, v in table.items()) or "（无稀有类）"))
    return table


# ── 5. 标签口径审计 ───────────────────────────────────────────────────

def _quantile(sorted_vals, p):
    if not sorted_vals:
        return None
    i = max(0, min(len(sorted_vals) - 1, int(round((len(sorted_vals) - 1) * p))))
    return sorted_vals[i]


def audit_records(records, log=print):
    """按类统计框面积/长宽比/贴边/巨型框，供人工抽检“整束 vs 单根”等口径。"""
    by_cls = defaultdict(list)
    for r in records:
        for b in r["boxes"]:
            by_cls[b["zh"]].append(b)
    out = {}
    for zh, boxes in sorted(by_cls.items()):
        areas = sorted(b["w"] * b["h"] for b in boxes)
        aspects = sorted(b["w"] / b["h"] for b in boxes if b["h"] > 0)
        edge = sum(1 for b in boxes
                   if b["cx"] - b["w"] / 2 <= EDGE_EPS or b["cy"] - b["h"] / 2 <= EDGE_EPS
                   or b["cx"] + b["w"] / 2 >= 1 - EDGE_EPS or b["cy"] + b["h"] / 2 >= 1 - EDGE_EPS)
        n_img = sum(1 for r in records if any(b["zh"] == zh for b in r["boxes"]))
        out[zh] = {
            "images": n_img, "boxes": len(boxes),
            "boxes_per_image": round(len(boxes) / n_img, 2) if n_img else 0,
            "area_median": round(_quantile(areas, 0.5), 4) if areas else None,
            "area_p10": round(_quantile(areas, 0.1), 4) if areas else None,
            "area_p90": round(_quantile(areas, 0.9), 4) if areas else None,
            "aspect_median": round(_quantile(aspects, 0.5), 3) if aspects else None,
            "edge_touch_rate": round(edge / len(boxes), 3) if boxes else 0,
            "giant_rate": round(sum(1 for a in areas if a > GIANT_AREA) / len(boxes), 3) if boxes else 0,
            "tiny_rate": round(sum(1 for a in areas if a < TINY_AREA) / len(boxes), 3) if boxes else 0,
        }
    log("  [audit] " + "; ".join(f"{z}: 框/图 {v['boxes_per_image']}、面积中位 {v['area_median']}"
                                 for z, v in list(out.items())[:4]) + " …")
    return out


# ── 6. 落盘 ────────────────────────────────────────────────────────────

def _safe(s):
    for ch in '\\/:*?"<>|\r\n\t':
        s = s.replace(ch, "_")
    return s.strip() or "unknown"


def _out_basename(rec, used):
    """`<md5前10位>_<herb>_<stem>.<ext>`；同名冲突时加 md5 后缀（必须全局唯一）。"""
    herb = _safe(rec["herb"].split("|")[0]) if rec["herb"] else "unknown"
    stem = _safe(rec["stem"])
    base = f"{rec['md5'][:10]}_{herb}_{stem}"
    name = base + rec["ext"]
    if name in used and used[name] != rec["md5"]:
        name = f"{base}_{rec['md5'][10:16]}{rec['ext']}"
    if name in used and used[name] != rec["md5"]:
        raise DatasetError(f"输出文件名仍冲突：{name}")
    used[name] = rec["md5"]
    return name


def _write_yaml(path, out_dir, table):
    lines = [f"path: {out_dir.replace(os.sep, '/')}", "train: train_list.txt",
             "val: images/val", "test: images/test", "names:"]
    lines += [f"  {c['train_id']}: {c['zh']}" for c in table]
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")


def _fingerprint(files):
    lines = [f"{rel}\t{md5}" for rel, md5 in sorted(files.items())]
    return hashlib.md5("\n".join(lines).encode("utf-8")).hexdigest()


def write_dataset(records, out_dir, table, delivery_ids, report, overwrite=False, log=print):
    if os.path.isdir(out_dir) and os.listdir(out_dir):
        if not overwrite:
            raise DatasetError(f"输出目录已存在且非空：{out_dir}（确认要覆盖请加 --overwrite）")
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    used_names = {}
    for r in sorted(records, key=lambda x: x["md5"]):
        r["out_name"] = _out_basename(r, used_names)

    files = {}
    written = [r for r in records if r["split"] in SPLITS]
    for split in SPLITS:
        os.makedirs(os.path.join(out_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(out_dir, "labels", split), exist_ok=True)
    for i, r in enumerate(written, 1):
        split = r["split"]
        rel_img = f"images/{split}/{r['out_name']}"
        rel_lbl = f"labels/{split}/{os.path.splitext(r['out_name'])[0]}.txt"
        shutil.copyfile(r["paths"][0], os.path.join(out_dir, rel_img.replace("/", os.sep)))
        lines = [f"{b['train_id']} {b['cx']:.6f} {b['cy']:.6f} {b['w']:.6f} {b['h']:.6f}"
                 for b in r["boxes"]]
        with open(os.path.join(out_dir, rel_lbl.replace("/", os.sep)), "w",
                  encoding="utf-8", newline="\n") as f:
            f.write("\n".join(lines) + ("\n" if lines else ""))
        files[rel_img] = r["md5"]
        files[rel_lbl] = md5_file(os.path.join(out_dir, rel_lbl.replace("/", os.sep)))
        if i % 500 == 0:
            log(f"    …已复制 {i}/{len(written)} 图")

    # train 重复清单：重复行只在 txt 里，磁盘不复制图片（保证 images/ 内容唯一）
    train_lines = []
    for r in sorted(records, key=lambda x: x["out_name"]):
        if r["split"] == "train":
            train_lines += [f"./images/train/{r['out_name']}"] * int(r.get("repeat", 1))
    with open(os.path.join(out_dir, "train_list.txt"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(train_lines) + "\n")
    files["train_list.txt"] = md5_file(os.path.join(out_dir, "train_list.txt"))

    with open(os.path.join(out_dir, "classes.txt"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(c["zh"] for c in table) + "\n")
    files["classes.txt"] = md5_file(os.path.join(out_dir, "classes.txt"))

    _write_yaml(os.path.join(out_dir, "data.yaml"), out_dir, table)
    files["data.yaml"] = md5_file(os.path.join(out_dir, "data.yaml"))

    with open(os.path.join(out_dir, "attributes.csv"), "w", newline="",
              encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=[
            "image", "md5", "split", "cross_herb", "repeat", "feature_id", "zh", "class_id",
            "group_key", "herb", "n_boxes", "is_negative", "source"])
        w.writeheader()
        for r in sorted(written, key=lambda x: x["out_name"]):
            base = {"image": r["out_name"], "md5": r["md5"], "split": r["split"],
                    "cross_herb": "1" if r["cross_herb"] else "0", "repeat": r["repeat"],
                    "group_key": r["group_key"], "herb": r["herb"],
                    "n_boxes": len(r["boxes"]), "is_negative": "0" if r["boxes"] else "1",
                    "source": ";".join(f"{s['dir']}/{s['name']}" for s in r["srcs"])}
            if not r["boxes"]:
                w.writerow(dict(base, feature_id="", zh="", class_id=""))
            for b in r["boxes"]:
                w.writerow(dict(base, feature_id=b["feature_id"], zh=b["zh"],
                                class_id=b["train_id"]))
    files["attributes.csv"] = md5_file(os.path.join(out_dir, "attributes.csv"))

    fp = {"version": os.path.basename(os.path.normpath(out_dir)),
          "n_files": len(files), "root_hash": _fingerprint(files),
          "class_map": [{"train_id": c["train_id"], "global_id": c["global_id"],
                         "feature_id": c["feature_id"], "zh": c["zh"],
                         "delivery": c["feature_id"] in delivery_ids} for c in table],
          "files": files}
    with open(os.path.join(out_dir, "dataset_fingerprint.json"), "w", encoding="utf-8") as f:
        json.dump(fp, f, ensure_ascii=False, indent=2)

    report["fingerprint"] = {"root_hash": fp["root_hash"], "n_files": fp["n_files"]}
    with open(os.path.join(out_dir, "split_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    log(f"  [write] {out_dir}：{len(written)} 图 / "
        f"{sum(len(r['boxes']) for r in written)} 框"
        f"（gold_eval 预留 {len(records) - len(written)} 图未写）；"
        f"fingerprint {fp['root_hash'][:12]}")
    return fp


# ── 7. 断言与报告 ──────────────────────────────────────────────────────

def _check(checks, name, cond, detail=""):
    checks.append({"name": name, "ok": bool(cond), "detail": detail})
    return bool(cond)


def validate_plan(records, table, delivery_ids, src_stats, merge_stats, holdout,
                  policy, gold_overlap, repeat_table, max_repeat, gold_md5=(), checks=None):
    checks = checks if checks is not None else []
    md5s = [r["md5"] for r in records]
    _check(checks, "content_md5_unique", len(set(md5s)) == len(md5s),
           f"{len(md5s)} 条记录 / {len(set(md5s))} 唯一内容")
    names = [r["out_name"] for r in records]
    _check(checks, "filename_unique", len(set(names)) == len(names),
           f"{len(names)} 文件名 / {len(set(names))} 唯一")

    by_group = defaultdict(set)
    for r in records:
        if r["split"] not in SPLITS:
            continue
        by_group[r["group_key"]].add(r["split"])
    bad = {g: sorted(s) for g, s in by_group.items() if len(s) > 1}
    _check(checks, "group_single_split", not bad,
           f"跨 split 组：{list(bad)[:5]}（gold_eval 不参与划分，见 gold_overlap.policy）")

    by_herb_group = defaultdict(set)
    for r in records:
        if r["split"] not in SPLITS:
            continue
        for h in r["herbs"]:
            by_herb_group[(h, r["group_key"])].add(r["split"])
    bad = {k: sorted(s) for k, s in by_herb_group.items() if len(s) > 1}
    _check(checks, "herb_group_single_split", not bad,
           f"跨 split 的 (herb, group_key)：{list(bad)[:5]}（tif/jpg 代理同组，故一并覆盖）")

    bad = sorted({r["herb"] for r in records
                  if r["split"] in ("train", "val")
                  and (set(r["herbs"]) & set(holdout["herbs"]))})
    _check(checks, "holdout_only_test", not bad, f"hold-out 药材出现在 train/val：{bad}")

    bad = sorted({r["md5"] for r in records
                  if r["split"] in ("train", "val") and r["md5"] in set(gold_md5)})
    _check(checks, "gold_not_train_val", not bad, f"gold 内容出现在 train/val：{bad[:5]}")
    _check(checks, "gold_eval_excluded",
           all(r["split"] == "gold_eval" for r in records if r["md5"] in set(gold_md5))
           and gold_overlap.get("excluded_records") == sum(1 for r in records
                                                          if r["md5"] in set(gold_md5)),
           f"gold 预留 {gold_overlap.get('excluded_records')} 张")

    other = next((c for c in table if c["zh"] == OTHER_ZH), None)
    _check(checks, "other_is_trained_class", bool(other), "「其他」必须是非 pending 的类别")
    if other:
        _check(checks, "other_not_delivery", other["feature_id"] not in delivery_ids,
               f"other feature_id={other['feature_id']} delivery=False")
        cnt = Counter(r["split"] for r in records
                      if any(b["zh"] == OTHER_ZH for b in r["boxes"]))
        _check(checks, "other_in_all_splits", all(cnt[s] > 0 for s in SPLITS), dict(cnt))
        if delivery_ids:
            covered = all(c["feature_id"] in delivery_ids for c in table if c["zh"] != OTHER_ZH)
            _check(checks, "delivery_covers_target_classes", covered,
                   "13 个目标类都应有 igm_desc（taxonomy 与 yolo.classes 对齐）")

    ids = [c["train_id"] for c in table]
    _check(checks, "class_ids_contiguous", ids == list(range(len(table))), f"ids={ids}")
    _check(checks, "all_classes_in_train",
           all(any(b["train_id"] == c["train_id"] for r in records if r["split"] == "train"
                   for b in r["boxes"]) for c in table),
           "每个训练类都要在 train 里出现")

    # 与单类导出报告对账
    out_boxes = Counter()
    for r in records:
        for b in r["boxes"]:
            out_boxes[b["zh"]] += 1
    ok = True
    details = []
    for zh, st in src_stats.items():
        expected = st["boxes"] - merge_stats["collapsed_boxes_by_class"].get(zh, 0)
        if out_boxes[zh] != expected:
            ok = False
            details.append(f"{zh}: 源 {st['boxes']} - 折叠 "
                           f"{merge_stats['collapsed_boxes_by_class'].get(zh, 0)} = {expected}，"
                           f"输出 {out_boxes[zh]}")
        n_records_from_dir = sum(1 for r in records if zh in r["src_dirs"])
        if n_records_from_dir != st["unique_md5"]:
            ok = False
            details.append(f"{zh}: 唯一内容 {st['unique_md5']} vs 记录 {n_records_from_dir}")
    _check(checks, "source_reconciliation", ok, "; ".join(details[:5]))

    _check(checks, "negatives_have_empty_labels",
           all(not r["boxes"] for r in records if r.get("src_box_lines", 0) == 0),
           "源标签为空的图，合并后标签也必须为空")
    _check(checks, "repeat_only_train_and_capped",
           all(r["repeat"] == 1 for r in records if r["split"] != "train")
           and all(1 <= r["repeat"] <= max_repeat for r in records),
           f"max_repeat={max_repeat}")
    _check(checks, "splits_nonempty", all(any(r["split"] == s for r in records) for s in SPLITS),
           {s: sum(1 for r in records if r["split"] == s) for s in SPLITS})

    failed = [c for c in checks if not c["ok"]]
    if failed:
        raise DatasetError("数据工程断言失败：\n  " +
                           "\n  ".join(f"[{c['name']}] {c['detail']}" for c in failed))
    return checks


def validate_output(out_dir, records, table, checks=None, log=print):
    checks = checks if checks is not None else []
    counts = {s: 0 for s in SPLITS}
    ok_dirs = True
    for s in SPLITS:
        img_dir = os.path.join(out_dir, "images", s)
        lbl_dir = os.path.join(out_dir, "labels", s)
        if not (os.path.isdir(img_dir) and os.path.isdir(lbl_dir)):
            ok_dirs = False
            continue
        imgs = {os.path.splitext(n)[0] for n in os.listdir(img_dir)}
        lbls = {os.path.splitext(n)[0] for n in os.listdir(lbl_dir)}
        counts[s] = len(imgs)
        _check(checks, f"{s}_image_label_pairs", imgs == lbls,
               f"images={len(imgs)} labels={len(lbls)}")
    _check(checks, "split_dirs_exist", ok_dirs, f"counts={counts}")
    _check(checks, "output_counts_match_plan",
           all(counts[s] == sum(1 for r in records if r["split"] == s) for s in SPLITS),
           f"dir={counts} plan={ {s: sum(1 for r in records if r['split'] == s) for s in SPLITS} }")

    with open(os.path.join(out_dir, "classes.txt"), encoding="utf-8") as f:
        names = [x.strip() for x in f.read().splitlines() if x.strip()]
    _check(checks, "classes_txt_matches_table", names == [c["zh"] for c in table], f"{names}")

    with open(os.path.join(out_dir, "data.yaml"), encoding="utf-8") as f:
        yaml_txt = f.read()
    _check(checks, "data_yaml_lists_all_names",
           all(f"  {c['train_id']}: {c['zh']}" in yaml_txt for c in table) and "train: train_list.txt" in yaml_txt,
           "data.yaml 缺少 names 或 train_list")

    with open(os.path.join(out_dir, "train_list.txt"), encoding="utf-8") as f:
        lines = [x.strip() for x in f.read().splitlines() if x.strip()]
    expected = sum(r["repeat"] for r in records if r["split"] == "train")
    _check(checks, "train_list_repeats", len(lines) == expected,
           f"train_list 行 {len(lines)} / 期望 {expected}")
    _check(checks, "train_list_paths_present",
           all(os.path.isfile(os.path.join(out_dir, ln.replace("./", "", 1).replace("/", os.sep)))
               for ln in lines[:10]), "抽样前 10 行")

    with open(os.path.join(out_dir, "dataset_fingerprint.json"), encoding="utf-8") as f:
        fp = json.load(f)
    _check(checks, "fingerprint_root_recomputes", _fingerprint(fp["files"]) == fp["root_hash"],
           "dataset_fingerprint 自校验")

    failed = [c for c in checks if not c["ok"]]
    if failed:
        raise DatasetError("输出落盘校验失败：\n  " +
                           "\n  ".join(f"[{c['name']}] {c['detail']}" for c in failed))
    log(f"  [check] 落盘断言全部通过（{len(checks)} 项）")
    return checks


# ── 8. 总入口（供 CLI 调用；不做 argparse）──────────────────────────────

def build(src_dir, out_dir, cfg, taxonomy_path="", *, seed=DEFAULT_SEED,
          val_ratio=DEFAULT_VAL_RATIO, test_ratio=DEFAULT_TEST_RATIO,
          holdout_herbs=None, holdout_max=DEFAULT_HOLDOUT_MAX,
          holdout_min_images=DEFAULT_HOLDOUT_MIN_IMAGES,
          holdout_min_boxes=DEFAULT_HOLDOUT_MIN_BOXES,
          rare_threshold=DEFAULT_RARE_THRESHOLD, max_repeat=DEFAULT_MAX_REPEAT,
          manifest_path="", gold_set_path="", overwrite=False, dry_run=False, log=print):
    table = class_table(cfg)
    delivery_ids = set()
    if taxonomy_path and os.path.isfile(taxonomy_path):
        with open(taxonomy_path, encoding="utf-8") as f:
            tax = json.load(f)
        delivery_ids = {x.get("feature_id") for x in tax.get("features", [])
                        if x.get("feature_id") and x.get("igm_desc")}
    else:
        log("  [warn] 没有可读的 feature_taxonomy.json → delivery 集合为空（仅影响报告标记）")

    log(f"[1/6] 扫描 {src_dir}")
    raw, src_stats = scan_source(src_dir, table, log=log)
    neg_stats = resolve_negative_meta(raw, manifest_path, log=log)
    log("[2/6] 按内容 md5 合并")
    records, merge_stats = merge_by_md5(raw, log=log)
    log("[3/6] 选 hold-out 药材")
    holdout = select_holdout_herbs(
        records, holdout_max=holdout_max, min_train_images=holdout_min_images,
        min_train_boxes=holdout_min_boxes, explicit=holdout_herbs, log=log)

    gold_md5 = set()
    gold_stats = {"path": gold_set_path, "n_items": 0, "n_existing": 0, "n_md5": 0}
    if gold_set_path and os.path.isfile(gold_set_path):
        with open(gold_set_path, encoding="utf-8") as f:
            gold = json.load(f)
        items = gold.get("items") or []
        gold_stats["n_items"] = len(items)
        for it in items:
            p = (it.get("image_path") or "").strip()
            if p and os.path.isfile(p):
                gold_stats["n_existing"] += 1
                gold_md5.add(md5_file(p))
        gold_stats["n_md5"] = len(gold_md5)
        log(f"  [gold] {len(items)} 项 / 文件存在 {gold_stats['n_existing']} / 唯一内容 {len(gold_md5)}")
    elif gold_set_path:
        log(f"  [warn] gold set 不存在：{gold_set_path}（跳过防泄漏剔除）")

    log("[4/6] 组感知划分")
    policy, gold_overlap = plan_split(
        records, holdout["herbs"], seed=seed, val_ratio=val_ratio,
        test_ratio=test_ratio, gold_md5=gold_md5, log=log)
    gold_overlap.update({"path": gold_set_path, "stats": gold_stats})
    log("[5/6] 稀有类重采样")
    repeat_table = oversample_train(records, rare_threshold=rare_threshold,
                                    max_repeat=max_repeat, log=log)
    audit = audit_records(records, log=log)

    # 报告：按 split / 类计数
    split_counts = {}
    class_counts = {}
    for s in SPLITS:
        rs = [r for r in records if r["split"] == s]
        split_counts[s] = {"images": len(rs), "boxes": sum(len(r["boxes"]) for r in rs),
                           "cross_herb_images": sum(1 for r in rs if r["cross_herb"])}
    gold_eval = [r for r in records if r["split"] == "gold_eval"]
    gold_eval_count = {"images": len(gold_eval),
                       "boxes": sum(len(r["boxes"]) for r in gold_eval)}
    for c in table:
        per_split = {}
        for s in SPLITS:
            imgs = sum(1 for r in records if r["split"] == s
                       and any(b["zh"] == c["zh"] for b in r["boxes"]))
            boxes = sum(1 for r in records if r["split"] == s
                        for b in r["boxes"] if b["zh"] == c["zh"])
            per_split[s] = {"images": imgs, "boxes": boxes}
        class_counts[c["zh"]] = per_split

    herb_coverage = {}
    for c in table:
        per_split = {}
        for s in SPLITS:
            herbs = sorted({h for r in records if r["split"] == s
                            and any(b["zh"] == c["zh"] for b in r["boxes"])
                            for h in r["herbs"]})
            per_split[s] = herbs
        herb_coverage[c["zh"]] = {
            "n_herbs": {s: len(v) for s, v in per_split.items()},
            "herbs": per_split}

    # 训练图数过少的类：固定 split 下只能靠 3× 重采样；训练时必须同时看 5 折 CV 或强增强
    rare_warnings = {
        c["zh"]: {"train_images": class_counts[c["zh"]]["train"]["images"],
                  "train_boxes": class_counts[c["zh"]]["train"]["boxes"],
                  "note": "train 图数 < 20，固定 split 证据薄；建议对该类做组感知 5 折 CV 或强增强"}
        for c in table if class_counts[c["zh"]]["train"]["images"] < 20}

    report = {
        "version": os.path.basename(os.path.normpath(out_dir)),
        "source": {"src_dir": src_dir, "n_src_dirs": len(src_stats),
                   "src_stats": src_stats, "negatives": neg_stats},
        "merge": merge_stats,
        "class_map": [{"train_id": c["train_id"], "global_id": c["global_id"],
                       "feature_id": c["feature_id"], "zh": c["zh"],
                       "delivery": c["feature_id"] in delivery_ids} for c in table],
        "split_policy": policy,
        "holdout": holdout,
        "gold_overlap": gold_overlap,
        "gold_eval_count": gold_eval_count,
        "oversample": repeat_table,
        "split_counts": split_counts,
        "class_counts": class_counts,
        "herb_coverage": herb_coverage,
        "rare_class_warnings": rare_warnings,
        "audit": audit,
        "images": [],
    }

    checks = []
    # out_name 在 dry-run 也要算（断言文件名唯一）
    used = {}
    for r in sorted(records, key=lambda x: x["md5"]):
        r["out_name"] = _out_basename(r, used)
    validate_plan(records, table, delivery_ids, src_stats, merge_stats, holdout,
                  policy, gold_overlap, repeat_table, max_repeat, gold_md5=gold_md5,
                  checks=checks)
    report["assertions_plan"] = checks

    report["images"] = [{
        "name": r["out_name"], "md5": r["md5"], "split": r["split"],
        "cross_herb": r["cross_herb"], "herb": r["herb"], "group_key": r["group_key"],
        "n_boxes": len(r["boxes"]), "repeat": r["repeat"],
        "sources": r["srcs"]} for r in sorted(records, key=lambda x: x["out_name"])]

    if dry_run:
        log("[dry-run] 不落盘")
        return report

    log("[6/6] 落盘 + 指纹 + 输出断言")
    write_dataset(records, out_dir, table, delivery_ids, report,
                  overwrite=overwrite, log=log)
    # 写盘后再校验（split_report 已写；finance 断言只读其它文件）
    checks_out = []
    validate_output(out_dir, records, table, checks=checks_out, log=log)
    # 把落盘断言并回 split_report
    report["assertions_output"] = checks_out
    with open(os.path.join(out_dir, "split_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return report
