# -*- coding: utf-8 -*-
"""
measure.py — 几何测量环节批量入口（第 4 步 · 几何测量模块）

输入
  --masks        实例分割 Mask：目录 / 实例编号PNG / 每实例PNG / COCO json
                 （契约见 geomlib/mask_io.py；mask 是尺寸测量的唯一真值）
  --images       原图目录（可选，仅用于叠加图绘制；支持子目录递归）
                 crop 局部 mask 按 `<masks>/crops/*.json` 的 loose_box 回贴源图对应区域，
                 源图找不到时退回 crops/ 的彩色小图，都没有才画灰底
  --manifest     manifest.csv 路径（可选；用于把图关联到 药材/特征/划分/批号）
  --out-prefix   输出前缀，如 out/demo → out/demo_measurements.csv / .jsonl
  --overlay-dir  叠加图输出目录（可选）
  --split-policy all（默认）| test_full_train20（test/val 全量 + train 确定性抽 20%，
                 落实"量化属性只对 test 集全标、训练集抽 20%"，需 --manifest）

输出
  <prefix>_measurements.csv  平面表（供判据核验 / Agent 规则推理 / attributes 合并）
  <prefix>_measurements.jsonl 全量记录（含 flags、标定来源等）
  契约细节（Agent 侧按此解析）：
    * instance_id = 交付 mask 文件名中的 inst<k>（稳定身份，可回溯到 mask 文件）；
    * flags：CSV 内为分号连接串（如 "uncalibrated;multi_component"），JSONL 内为数组；
    * 标定（2026-09-20 契约）：px_per_um 只认分割交付清单 `<stem>*_measure_manifest.csv`
      里的 px_per_um（该值由检测阶段的双引擎 `common/scalebar_duo.py` 测定并随
      mask 一起交付）。全项目已停用 scale_table.csv，也不做比例尺自动探测；
      **交付清单缺 px_per_um 即报错退出**（不再降级为 uncalibrated —— 那会产出无 μm 的测量行）。
    * 元数据（herb/feature）：交付清单的 herb 优先；group_key/split 由 --manifest 关联，
      关联用完整路径 → 交付清单 source_image_path（末两级）→ 唯一 stem，歧义即放弃，
      绝不猜——全库同名图极多，猜错会把记录挂到别的药材上。
    * bbox_x/bbox_y、feret_p1_px/feret_p2_px 与叠加图和 mask 同一坐标框架
      （crop 级交付即 crop 局部坐标）；feret_p1_px/feret_p2_px 是长径两端点，
      仅写 JSONL（CSV 平面表契约不变）。

用法示例
  python measure.py --masks <分割交付目录> --images "D:/dataset/显微镜像" ^
      --manifest "../manifest.csv" --out-prefix out/demo --overlay-dir out/overlay
"""
import os
import sys
import csv
import json
import hashlib
import argparse

import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from geomlib import load_instances, measure_instance
from geomlib.calibration import Calibration
from geomlib.imutil import imread, imwrite

import contracts  # 交付清单契约（项目根 contracts.py；装载与 px_per_um 校验的唯一事实源）

import cv2

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

CSV_COLUMNS = [
    "image_stem", "instance_id", "feature", "herb", "group_key", "split",
    "calibration_px_per_um", "calibration_source", "calibration_status",
    "primary_dimension", "primary_value_um",
    "feret_max_um", "feret_min_um", "eq_diameter_um", "max_inscribed_dia_um",
    "perimeter_um", "area_px", "area_um2",
    "skeleton_len_um", "mean_width_um", "median_width_um", "width_p25_um", "width_p75_um",
    "tortuosity", "aspect_ratio", "eccentricity", "solidity", "extent",
    "convexity", "circularity", "roundness", "rectangularity",
    "defect_count", "mean_defect_depth_ratio", "dt_peak_count",
    "n_components", "bbox_x", "bbox_y", "bbox_w_px", "bbox_h_px",
    "feret_angle_deg", "rect_w_um", "rect_h_um", "flags",
]


def find_image(images_dir, stem):
    """在 --images 目录（含子目录）按 stem 找原图。"""
    if not images_dir:
        return None
    for root, _dirs, files in os.walk(images_dir):
        for fn in files:
            b, ext = os.path.splitext(fn)
            if b == stem and ext.lower() in IMG_EXTS:
                return os.path.join(root, fn)
    return None


def _norm_path(p):
    """统一路径写法：反斜杠→正斜杠、小写、去首尾空白（manifest 与交付清单写法常不一致）。"""
    return str(p or "").replace("\\", "/").strip().lower()


def _path_tail(p, n=2):
    """路径末 n 级（如 "虎杖/1.jpg"）。全库同名图极多，末两级才是可用的消歧键。"""
    parts = [s for s in _norm_path(p).split("/") if s]
    return "/".join(parts[-n:])


def load_manifest_index(manifest_path):
    """manifest.csv → (by_path, by_tail, by_stem)。

    不同药材文件夹下同名文件极多（实测 1885 个 stem 中 798 个重名、覆盖 79% 的行），
    故按完整路径与末两级路径建索引，裸 stem 只作最后兜底。
    """
    by_path, by_tail, by_stem = {}, {}, {}
    if not manifest_path or not os.path.isfile(manifest_path):
        return by_path, by_tail, by_stem
    with open(manifest_path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            p = _norm_path(row["image_path"])
            by_path.setdefault(p, row)
            by_tail.setdefault(_path_tail(p), []).append(row)
            by_stem.setdefault(os.path.splitext(os.path.basename(p))[0], []).append(row)
    return by_path, by_tail, by_stem


_WARNED_KEYS = set()


def _warn_once(key, msg):
    if key not in _WARNED_KEYS:
        _WARNED_KEYS.add(key)
        print(f"[warn] {msg}")


def manifest_lookup(manifest_idx, image_path, stem, hint_path=None):
    """关联 manifest 行：完整路径 → 末两级路径 → 唯一 stem；有歧义就放弃。

    hint_path 取分割交付清单的 source_image_path：它与 manifest 的路径前缀常不一致
    （如 "D:/dataset/数据量太少合集/虎杖/1.jpg" vs "<repo>/dataset/数据量太少合集/虎杖/1.jpg"），
    靠末两级兜住。返回 {} 表示关联不上——缺元数据好过错元数据（herb 串到别的药材上）。
    """
    by_path, by_tail, by_stem = manifest_idx
    for p in (image_path, hint_path):
        if p:
            row = by_path.get(_norm_path(p))
            if row:
                return row
    for p in (hint_path, image_path):
        if not p:
            continue
        tail, cands = _path_tail(p), by_tail.get(_path_tail(p)) or []
        if len(cands) == 1:
            return cands[0]
        if len(cands) > 1:
            _warn_once(f"tail:{tail}",
                       f"路径末两级 '{tail}' 在 manifest 中命中 {len(cands)} 条记录"
                       f"（如 {cands[0].get('image_path')}）——放弃关联")
            return {}
    cands = by_stem.get(stem) or []
    if len(cands) == 1:
        return cands[0]
    if len(cands) > 1:
        _warn_once(f"stem:{stem}",
                   f"stem '{stem}' 在 manifest 中有 {len(cands)} 条同名记录"
                   f"（如 {cands[0].get('image_path')}、herb={cands[0].get('herb')}）"
                   f"——放弃关联，请用交付清单 source_image_path 或完整路径消歧")
    return {}


def load_crop_index(masks_dir):
    """读分割交付的 crop 侧车 json（`<masks>/crops/*.json`）→ {源图stem: [row, ...]}。

    侧车 json 与 seg_manifest 同契约：loose_box = 该 crop 在源图中的位置（源图坐标，
    小图不经缩放），crop_size = crop 尺寸、与 mask 同尺寸 → mask 即 loose_box 局部坐标。
    叠加图靠它把 crop 局部 mask 回贴到原图（否则只能画在灰底上）。
    """
    idx, cdir = {}, os.path.join(masks_dir or "", "crops")
    if not os.path.isdir(cdir):
        return idx
    for fn in sorted(os.listdir(cdir)):
        if not fn.lower().endswith(".json"):
            continue
        try:
            with open(os.path.join(cdir, fn), encoding="utf-8") as f:
                row = json.load(f)
        except (OSError, ValueError):
            continue
        stem = str(row.get("stem") or "").strip()
        if stem:
            row["_png"] = os.path.join(cdir, str(row.get("png") or os.path.splitext(fn)[0] + ".png"))
            idx.setdefault(stem, []).append(row)
    return idx


def resolve_overlay_base(images_path, inst, crop_idx):
    """给叠加图找与 mask 同尺寸的底图；找不到返回 None（调用方回退灰底）。

    * 原图级 mask：--images 里的源图本身同尺寸，直接用；
    * crop 局部 mask（分割 §1.3）：按 crop 侧车 json 的 loose_box 从源图裁出对应区域；
      源图不可用（未给 --images / 找不到该 stem）时，退回 `crops/` 里同尺寸的彩色小图。
    """
    h, w = inst.mask.shape[:2]
    ipath = find_image(images_path, inst.image_stem)
    img = imread(ipath) if ipath else None
    if img is not None and img.shape[:2] == (h, w):
        return img
    for row in crop_idx.get(inst.image_stem) or []:
        lb = [int(round(v)) for v in (row.get("loose_box") or [])]
        if img is not None and len(lb) == 4:
            x0, y0, x1, y1 = lb
            if (y1 - y0, x1 - x0) == (h, w) and 0 <= x0 < x1 <= img.shape[1] \
                    and 0 <= y0 < y1 <= img.shape[0]:
                return img[y0:y1, x0:x1]
        crop = imread(row.get("_png") or "")
        if crop is not None and crop.shape[:2] == (h, w):
            return crop
    return None


def keep_by_split_policy(rec_meta, policy, rng_seed=0):
    """test/val 全量，train 按 image+instance 确定性抽 20%。"""
    if policy == "all":
        return True
    split = rec_meta.get("split") or ""
    if split in ("test", "val", ""):
        return True
    if split != "train":
        return True
    key = f"{rec_meta.get('image_stem')}#{rec_meta.get('instance_id')}".encode()
    return int(hashlib.md5(key).hexdigest(), 16) % 100 < 20


FONT = cv2.FONT_HERSHEY_SIMPLEX
TEXT_MAX_SCALE = 0.9      # 图上文字最大字号（原固定值）
TEXT_MIN_SCALE = 0.35     # 字号下限：再小就不可辨，宁可略溢出
TEXT_SPLIT_BELOW = 0.55   # 单行字号低于此值即拆两行——窄图上换成两行能显著放大字
TEXT_PAD_X = 12           # 左侧留白
TEXT_THICK = 2            # 文字笔画宽（描边另加 2）


def _text_metrics(s, thickness):
    """scale=1.0 下的 (宽, 高, 基线) —— 按描边外沿量，保证描边也不出界。"""
    (tw, th), tb = cv2.getTextSize(s, FONT, 1.0, thickness)
    return float(tw), float(th), float(tb)


def draw_caption(vis, parts, thickness=TEXT_THICK):
    """左上角标注：**按图宽自适应字号**，单行放不下就拆两行，避免文字被裁掉。

    parts 为段落列表（如 ["cluster_crystal #1", "feret_max=23.2um"]）：
    图够宽时拼成一行；单行字号会低于 TEXT_SPLIT_BELOW 时改为逐段一行（字更大）。
    原实现固定 0.9 号，在 crop 级小图（实测宽 255~386px）上文字超出右边界被截断。
    """
    h, w = vis.shape[:2]
    avail = max(60.0, w - 2 * TEXT_PAD_X)
    parts = [p for p in parts if p]
    if not parts:
        return

    def scale_for(lines):
        widest = max(_text_metrics(ln, thickness + 2)[0] for ln in lines)
        return min(TEXT_MAX_SCALE, avail / widest) if widest > 0 else TEXT_MAX_SCALE

    single = parts[0] + (": " + " ".join(parts[1:]) if len(parts) > 1 else "")
    lines = [single]
    scale = max(TEXT_MIN_SCALE, scale_for(lines))
    if scale < TEXT_SPLIT_BELOW and len(parts) > 1:
        lines = parts                       # 单行太挤 → 拆行（行更短，字号更大）
        scale = max(TEXT_MIN_SCALE, scale_for(lines))

    _tw, th, _tb = _text_metrics(lines[0], thickness + 2)
    line_h = int(round(th * scale)) + 8
    y = int(round(th * scale)) + 8
    for i, ln in enumerate(lines):
        org = (TEXT_PAD_X, y + i * line_h)
        cv2.putText(vis, ln, org, FONT, scale, (0, 0, 0), thickness + 2)
        cv2.putText(vis, ln, org, FONT, scale, (0, 255, 255), thickness)


def draw_overlay(img, inst, rec):
    """叠加图：绿=轮廓，红=最大长径（凸包最大距离点对的连线），左上角写 主判尺寸。

    img 须与 mask 同尺寸（由 resolve_overlay_base 保证：原图级 mask 用源图，
    crop 局部 mask 按 crop 侧车 json 回贴源图对应区域）；img=None 时退回灰底。
    """
    if img is None:
        img = np_full_gray(inst.mask)
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img.copy()
    cnts, _ = cv2.findContours(inst.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, cnts, -1, (0, 200, 0), 3)
    title = f"{rec['feature']} #{rec['instance_id']}"
    detail = ""
    if rec.get("primary_value_um") is not None:
        detail = f"{rec['primary_dimension'].replace('_um','')}={rec['primary_value_um']:.1f}um"
    elif rec.get("feret_max_um") is None:
        detail = "[uncalibrated]"
    if rec.get("feret_p1_px") and rec.get("feret_p2_px"):
        p1 = tuple(int(round(v)) for v in rec["feret_p1_px"])
        p2 = tuple(int(round(v)) for v in rec["feret_p2_px"])
        cv2.line(vis, p1, p2, (0, 0, 255), 2)
    draw_caption(vis, [title, detail])
    return vis


def np_full_gray(mask):
    g = np.zeros((mask.shape[0], mask.shape[1], 3), np.uint8)
    g[:] = (235, 235, 235)
    g[mask > 0] = (200, 220, 220)
    return g


def main(argv=None):
    ap = argparse.ArgumentParser(description="中药显微 · 几何测量（mask → 几何特征 μm）")
    ap.add_argument("--masks", required=True)
    ap.add_argument("--images", default=None)
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--out-prefix", required=True)
    ap.add_argument("--overlay-dir", default=None)
    ap.add_argument("--split-policy", choices=["all", "test_full_train20"], default="all")
    args = ap.parse_args(argv)

    os.makedirs(os.path.dirname(os.path.abspath(args.out_prefix)) or ".", exist_ok=True)
    instances = load_instances(args.masks)
    if not instances:
        print(f"[warn] {args.masks} 未解析到任何实例 mask")
        return 1
    print(f"[info] 解析到 {len(instances)} 个实例 mask")

    manifest_idx = load_manifest_index(args.manifest)
    calib_cache = {}
    # 交付清单（<stem>*_measure_manifest.csv）：px_per_um 的唯一来源，同时携带
    # herb/object_label/source_image_path → 兼作 manifest 关联的消歧线索。
    # 2026-09-29（P2-1）改为契约装载器：字段解析与"px_per_um 是否有效"的判定统一在 contracts.py。
    delivery_meta = contracts.load_manifest_rows(args.masks)
    # crop 侧车 json：crop 局部 mask 回贴源图坐标系的依据（叠加图底图）
    crop_idx = load_crop_index(args.masks)
    base_cache = {}

    # 标定硬门槛（2026-09-20）：px_per_um 唯一来源 = 分割交付清单，缺失即报错退出。
    # 一次列出全部缺项（而非逐图中断），便于批量补测后重跑。
    def _ppu_of(stem):
        row = delivery_meta.get(stem)
        return row.px_per_um if (row and row.px_per_um) else 0.0

    missing = sorted({i.image_stem for i in instances if not _ppu_of(i.image_stem)})
    if missing:
        print(f"[FAIL] {len(missing)} 个源图在分割交付清单里没有有效 px_per_um，"
              f"无法换算 μm（共 {len(instances)} 个实例）：", file=sys.stderr)
        for s in missing[:10]:
            print(f"         - {s}", file=sys.stderr)
        if len(missing) > 10:
            print(f"         ...（另有 {len(missing) - 10} 个）", file=sys.stderr)
        print(f"       交付目录: {args.masks}\n"
              f"       期望清单: <源图stem>*_measure_manifest.csv（须含 px_per_um 列）\n"
              f"       px_per_um 由检测阶段双引擎（common/scalebar_duo.py）测定，"
              f"随 mask 一起交付；请先补测再跑测量。", file=sys.stderr)
        return 2

    n_csv = n_json = 0
    csv_path = args.out_prefix + "_measurements.csv"
    jsonl_path = args.out_prefix + "_measurements.jsonl"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as fc, \
         open(jsonl_path, "w", encoding="utf-8") as fj:
        w = csv.DictWriter(fc, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for inst in instances:
            drow = delivery_meta.get(inst.image_stem)      # MeasureManifestRow | None（契约对象）
            if inst.image_stem not in calib_cache:
                ipath = find_image(args.images, inst.image_stem)
                if ipath is None:
                    ipath = os.path.join(args.images or ".", inst.image_stem + ".jpg")
                ppu = _ppu_of(inst.image_stem)
                # 已在循环前统一校验，此处必有有效标定值
                calib_cache[inst.image_stem] = (
                    ipath, Calibration(ppu, None, None, "delivery_manifest", "ok"))
            ipath, calib = calib_cache[inst.image_stem]

            # manifest 消歧：完整路径 → 交付清单 source_image_path（末两级）→ 唯一 stem，
            # 歧义即放弃关联（宁缺勿错，避免把记录挂到别的药材上）
            meta = manifest_lookup(manifest_idx, ipath if os.path.isfile(ipath) else None,
                                   inst.image_stem, drow.source_image_path if drow else "")
            mrec = {"image_stem": inst.image_stem, "instance_id": inst.instance_id,
                    "split": meta.get("suggested_split", "")}
            if not keep_by_split_policy(mrec, args.split_policy):
                continue

            # herb 优先取交付清单（随 mask 一起人工交付），manifest 只作补充
            herb = (drow.herb if drow else "") or meta.get("herb") or None
            rec = measure_instance(inst, calib, herb=herb)
            rec["group_key"] = meta.get("group_key", "")
            rec["split"] = meta.get("suggested_split", "")
            rec["feature"] = inst.label or rec["feature"] or meta.get("feature_label", "") or "generic"
            # IGM 交付清单里的 mask 状态：非 ok（review/empty/full）说明该实例的 mask 极性或
            # 框先验异常 → 打 igm_review 旗标随记录下传。判药侧据此**整条丢弃该实例**（不进
            # 证据池、不计入打分，见 rule_engine.pool_evidence），并在报告局限声明里点名。
            igm_status = drow.igm_status if drow else ""
            if igm_status and igm_status != "ok":
                rec.setdefault("flags", []).append("igm_review")
                rec["igm_status"] = igm_status
            # 量化可用性闸门（2026-10-09）：框粒度非"单个实例"（束/片/团）→ 该实例的尺寸不可
            # 对照药典单值量纲（束宽被当单根直径、整片尺寸被当单个细胞直径），打 coarse_quant。
            # 与 igm_review 同构，区别是**不丢弃**：判药侧降为"只作存在性"证据（rule_engine.pool_evidence）。
            gran = (drow.granularity if drow else "") or contracts.GRANULARITY_DEFAULT
            if gran != contracts.GRANULARITY_DEFAULT:
                rec.setdefault("flags", []).append("coarse_quant")
                rec["granularity"] = gran

            # CSV 契约：flags 用分号连接（"uncalibrated;multi_component"），
            # JSONL 保留原生数组；下游 Agent.evidence_pool 按此解析。
            csv_rec = dict(rec)
            csv_rec["flags"] = ";".join(rec.get("flags") or [])
            w.writerow(csv_rec)
            fj.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_csv += 1

            if args.overlay_dir:
                bkey = (inst.image_stem, inst.mask.shape)
                if bkey not in base_cache:
                    base_cache[bkey] = resolve_overlay_base(args.images, inst, crop_idx)
                vis = draw_overlay(base_cache[bkey], inst, rec)
                imwrite(os.path.join(args.overlay_dir,
                                     f"{inst.image_stem}__inst{inst.instance_id}_测量.jpg"), vis)

    print(f"[done] 写出 {n_csv} 条测量 → {csv_path}")
    print(f"       jsonl → {jsonl_path}")
    if args.overlay_dir:
        print(f"       叠加图 → {args.overlay_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
