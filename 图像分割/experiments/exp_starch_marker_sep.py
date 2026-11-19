# -*- coding: utf-8 -*-
"""exp_starch_marker_sep.py — 实验 D2：VLM 粒框当 marker 先验的团内分离（方案 C′ 离线验证）

实验 D（exp_starch_cluster_sep.py）的结论：IGM 团 mask 保留了粒间边界（好），但纯"距离变换
峰值"自适应 marker 不可靠——同团换提示词实例数 51↔118↔4，α 曲线无平台，目检有合并/切碎，
且自适应粒半径在 6~48px 间乱跳（marker 全靠猜）。

方案 C′ 的假设：**让 VLM 在团 crop 上逐粒框，用这些框当 marker 种子**，
而不是把粒框各自裁成 crop（那会炸 IGM 成本）。理由：
  - 落点是"语义级"的（模型认出那是颗粒），比距离变换峰值可靠，直接压掉合并/切碎；
  - VLM 调用比 IGM 便宜一个数量级，且**不产生新 crop** → IGM 调用数不变（1 团 1 次）。

本脚本只做离线验证：1 次 VLM 调用取粒框 → 在**已落盘的 gen_raw** 上跑两种 marker 的
分水岭对照（零 IGM 开销）。

判据（为什么用"粒径离散度"而不是"实例数对不对"）：
  同一视野的淀粉粒粒径本应相近。分离正确 ⇒ 每区域一颗粒 ⇒ 直径分布集中；
  合并 ⇒ 直径偏大；切碎 ⇒ 直径偏小。**实例数可能因合并与切碎相互抵消而恰好接近真值**
  （实验 D 中 51 vs 目检约 50~60 就是这种假象），故必须看分布集中度（IQR/中位、CV），
  而不是只看个数。

用法:
  python exp_starch_marker_sep.py                # 复用缓存粒框（首次会发起 1 次 VLM 调用）
  python exp_starch_marker_sep.py --dry-run      # 只打印将发的请求
  python exp_starch_marker_sep.py --force        # 强制重发 VLM
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SEG_DIR = os.path.dirname(HERE)
DET_DIR = os.path.join(os.path.dirname(SEG_DIR), "detection")
COMMON_DIR = os.path.join(os.path.dirname(SEG_DIR), "common")
for p in (HERE, SEG_DIR, DET_DIR, COMMON_DIR):
    sys.path.insert(0, p)

import igm_segment as ig                                    # noqa: E402
import igm_exp_common as ec                                # noqa: E402
from importlib import import_module                        # noqa: E402
from ls_common import imread_cn, load_config, resolve_key   # noqa: E402
from prompts import PROMPT, PROMPT_VERSION                  # noqa: E402 提示词唯一事实源
from vlm_api import call_vlm, encode_jpeg_b64, parse_vlm_json  # noqa: E402

RESULTS_SEG = os.path.join(os.path.dirname(SEG_DIR), "results", "segmentation")
EXP_D = os.path.join(RESULTS_SEG, "igm_seg_expD")
SEP_DIR = os.path.join(EXP_D, "sep")
CROP_ID = "starch_grain_1"
VARIANT = "prod"                    # 用生产提示词的产物（生产口径）
CACHE = os.path.join(EXP_D, CROP_ID, "vlm_grain_boxes.json")


def vlm_grain_boxes(crop_path, dry_run=False, force=False):
    """VLM 在团 crop 上逐粒框 → crop 像素坐标框列表。结果缓存，避免重复扣费。"""
    if os.path.exists(CACHE) and not force and not dry_run:
        d = json.load(open(CACHE, encoding="utf-8"))
        print(f"[缓存] 复用粒框 {len(d['boxes'])} 个 ← {CACHE}")
        return d
    img = imread_cn(crop_path)
    if img is None:
        sys.exit(f"[FAIL] 读不到 crop: {crop_path}")
    H, W = img.shape[:2]
    cfg = load_config()
    key = resolve_key(cfg)
    b64, _ = encode_jpeg_b64(img, cfg["vlm"].get("max_side", 1536))
    prompt = PROMPT               # 生产提示词原文（含"淀粉粒密集时尽量逐个框出"）
    if dry_run:
        print(f"[dry-run] 将对 {os.path.basename(crop_path)} 发起 1 次 VLM 调用"
              f"（model={cfg['vlm']['model']}，载荷 {len(b64)/1024:.0f}KB base64）")
        return None
    if not key:
        sys.exit("[FAIL] 未配置 VLM API key")
    print(f"[调用] VLM 逐粒框：{os.path.basename(crop_path)} {W}x{H} …（约 20~60s）")
    text, err = call_vlm(cfg["vlm"], key, b64, prompt=prompt)
    if text is None:
        sys.exit(f"[FAIL] VLM 调用失败: {err}")
    parsed = parse_vlm_json(text)
    boxes, dropped = [], 0
    for o in parsed.get("objects", []):
        if o.get("label") != "淀粉粒":
            dropped += 1
            continue
        bb = o.get("bbox_2d")
        if not bb or len(bb) != 4:
            dropped += 1
            continue
        x1, y1, x2, y2 = [float(v) for v in bb]
        if x2 <= x1 or y2 <= y1:
            dropped += 1
            continue
        boxes.append([x1 / 1000.0 * W, y1 / 1000.0 * H, x2 / 1000.0 * W, y2 / 1000.0 * H])
    out = {"crop": crop_path, "crop_hw": [H, W], "model": cfg["vlm"]["model"],
           "prompt_version": PROMPT_VERSION,
           "n_boxes": len(boxes), "n_dropped_other_label": dropped,
           "boxes": [[round(v, 1) for v in b] for b in boxes],
           "raw_excerpt": text[:800]}
    with open(CACHE, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"[结果] VLM 给出 {len(boxes)} 个淀粉粒框（丢弃非本类 {dropped}）→ {CACHE}")
    return out


def nms_points(pts, min_dist):
    """按 distance 值降序的贪心 NMS（用于自适应 marker）。"""
    acc = []
    for y, x, v in pts:
        if all((y - ay) ** 2 + (x - ax) ** 2 >= min_dist ** 2 for ay, ax, _ in acc):
            acc.append((y, x, v))
    return acc


def adaptive_markers(mask, min_dist):
    """基线：距离变换局部极大 + NMS（无任何语义信息）。"""
    dist = cv2.distanceTransform((mask > 0).astype(np.uint8), cv2.DIST_L2, 5)
    lm = dist >= cv2.dilate(dist, np.ones((5, 5), np.uint8)) - 1e-6
    ys, xs = np.nonzero(lm & (dist > 2))
    order = np.argsort(-dist[ys, xs])
    pts = [(int(ys[i]), int(xs[i]), float(dist[ys[i], xs[i]])) for i in order]
    return nms_points(pts, min_dist), dist


def vlm_markers(mask, boxes):
    """C′：每个 VLM 粒框内取距离变换峰值作为该粒的种子（框中心可能落在缝上，故取域内峰值）。"""
    dist = cv2.distanceTransform((mask > 0).astype(np.uint8), cv2.DIST_L2, 5)
    pts = []
    for x1, y1, x2, y2 in boxes:
        xa, ya = int(max(0, x1)), int(max(0, y1))
        xb, yb = int(min(mask.shape[1], x2)), int(min(mask.shape[0], y2))
        if xb <= xa or yb <= ya:
            continue
        sub = dist[ya:yb, xa:xb]
        if not sub.size or sub.max() <= 0:
            continue
        j, i = np.unravel_index(int(np.argmax(sub)), sub.shape)
        pts.append((ya + int(j), xa + int(i), float(sub[j, i])))
    # 同一像素/过近的种子去重（框重叠时会出现）
    pts.sort(key=lambda t: -t[2])
    return nms_points(pts, 3.0), dist


def vlm_markers_capped(mask, boxes, k=1.15):
    """C″：每个 VLM 粒框按其**自身内切半径**封顶，只保留 marker 邻域，邻粒切出去。

    动机（2026-09-24 实测）：VLM 在团上只召回约 1/3 的粒（18/约 60），未标记的邻粒会被
    就近 marker 的分水岭区域吞并 → 中位直径 110→160px（+45%）、区域轮廓呈多瓣团块。
    封顶半径取该框内 mask 距离变换峰值（= 该粒在 mask 里的内切半径）× k，
    超出部分**不计入任何实例**（记为未分配），从而把"测错"降级为"测少"。
    返回 (markers, 每 marker 的封顶半径 px)。
    """
    dist = cv2.distanceTransform((mask > 0).astype(np.uint8), cv2.DIST_L2, 5)
    pts = []
    for x1, y1, x2, y2 in boxes:
        xa, ya = int(max(0, x1)), int(max(0, y1))
        xb, yb = int(min(mask.shape[1], x2)), int(min(mask.shape[0], y2))
        if xb <= xa or yb <= ya:
            continue
        sub = dist[ya:yb, xa:xb]
        if not sub.size or sub.max() <= 0:
            continue
        j, i = np.unravel_index(int(np.argmax(sub)), sub.shape)
        pts.append((ya + int(j), xa + int(i), float(sub[j, i]) * k))
    pts.sort(key=lambda t: -t[2])
    kept = []
    for y, x, r in pts:
        if all((y - ay) ** 2 + (x - ax) ** 2 >= 3.0 ** 2 for ay, ax, _ in kept):
            kept.append((y, x, r))
    return kept


def capped_areas(mask, ws, kept):
    """分水岭区域 ∩ 各 marker 的封顶圆盘 → (各实例面积, 未分配前景面积)。"""
    m8 = (mask > 0).astype(np.uint8)
    H, W = m8.shape
    yy, xx = np.mgrid[0:H, 0:W]
    areas, assigned = [], np.zeros((H, W), bool)
    for i, (y, x, r) in enumerate(kept, 1):
        disk = (yy - y) ** 2 + (xx - x) ** 2 <= r * r
        sel = (ws == i) & disk
        areas.append(int(sel.sum()))
        assigned |= sel
    unassigned = int(((m8 > 0) & (~assigned)).sum())
    return areas, unassigned, assigned


def split_stats(areas):
    """面积 → 直径统计（px），返回集中度指标。"""
    if not areas:
        return {}
    a = np.array(sorted(areas), dtype=np.float64)
    med = float(np.median(a))
    iqr = float(np.percentile(a, 75) - np.percentile(a, 25))
    d = 2 * np.sqrt(a / np.pi)
    return {"n": int(a.size), "area_med": round(med, 1),
            "diam_med_px": round(2 * np.sqrt(med / np.pi), 1),
            "iqr_over_med": round(iqr / med, 3) if med else None,
            "cv": round(float(d.std() / d.mean()), 3) if d.mean() else None,
            "max_over_min_diam": round(float(d.max() / max(d.min(), 1e-6)), 2)}


def run_watershed(mask, pts):
    """marker → 分水岭 → 各区域面积（过滤极小碎片）与可视化 mask。"""
    m8 = (mask > 0).astype(np.uint8)
    if not pts:
        return [], None
    markers = np.zeros(m8.shape, np.int32)
    for i, (y, x, _) in enumerate(pts, 1):
        markers[y, x] = i
    markers[m8 == 0] = len(pts) + 1
    ws = cv2.watershed(cv2.cvtColor(m8 * 255, cv2.COLOR_GRAY2BGR), markers.copy())
    areas = [int((ws == v).sum()) for v in range(1, len(pts) + 1)]
    base = np.median([a for a in areas if a > 0]) if any(a > 0 for a in areas) else 0
    keep = [a for a in areas if a >= 0.15 * base] if base else []
    return keep, ws


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crop-id", default=CROP_ID)
    ap.add_argument("--variant", default=VARIANT)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true", help="强制重发 VLM")
    args = ap.parse_args()

    crop_path = os.path.join(EXP_D, "crops", args.crop_id + ".png")
    run_dir = os.path.join(EXP_D, args.crop_id, args.variant)
    crop = imread_cn(crop_path)
    if crop is None:
        sys.exit(f"[FAIL] 读不到 crop {crop_path}（先跑 exp_starch_cluster_sep.py）")
    h, w = crop.shape[:2]

    boxes_info = vlm_grain_boxes(crop_path, args.dry_run, args.force)
    if args.dry_run:
        return
    boxes = boxes_info["boxes"]

    # 团 mask：不填洞（保留粒间缝隙）—— 与实验 D 的 nofill 档同口径
    gen = ig.imread_any(os.path.join(run_dir, "gen_raw.png"))
    meta = json.load(open(os.path.join(run_dir, "meta.json"), encoding="utf-8"))
    g = cv2.cvtColor(gen, cv2.COLOR_BGR2GRAY) if gen.ndim == 3 else gen
    roi = ig._crop_content(g, meta["letterbox"])
    small = cv2.resize(roi, (w, h), interpolation=cv2.INTER_AREA)
    t, _ = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    flag = cv2.THRESH_BINARY_INV if meta.get("invert") else cv2.THRESH_BINARY
    mask = cv2.threshold(small, int(t), 255, flag)[1]

    # 基线：自适应 marker（粒半径由距离图估计）
    dist = cv2.distanceTransform((mask > 0).astype(np.uint8), cv2.DIST_L2, 5)
    lm = dist >= cv2.dilate(dist, np.ones((5, 5), np.uint8)) - 1e-6
    vals = dist[lm & (dist > 2)]
    d_est = max(4.0, 2.0 * (float(np.median(vals)) if vals.size else 0.0))
    ad_pts, _ = adaptive_markers(mask, d_est)
    base_areas, base_ws = run_watershed(mask, ad_pts)

    # C′：VLM 粒框当 marker（不封顶，看裸分水岭）
    vp_pts, _ = vlm_markers(mask, boxes)
    vlm_areas, vlm_ws = run_watershed(mask, vp_pts)

    # C″：VLM 粒框 + 每框内切半径封顶（邻粒切出去标未分配）
    cap_pts = vlm_markers_capped(mask, boxes)
    keep_areas, unassigned, assigned_mask = capped_areas(mask, vlm_ws, cap_pts)

    bs, vs, v2 = split_stats(base_areas), split_stats(vlm_areas), split_stats(keep_areas)
    v2["unassigned_foreground_px"] = unassigned
    print("\n" + "=" * 82)
    print(f"团 crop {args.crop_id}（{w}x{h}）| 产物 {args.variant} | "
          f"VLM 粒框 {len(boxes)} 个 → 可用 marker {len(vp_pts)} 个")
    print(f"{'':<22}{'实例数':>7}{'中位直径px':>11}{'IQR/中位':>10}{'直径CV':>8}{'max/min':>9}")
    for name, s in (("基线(自适应)", bs), ("C′(VLM框,不封顶)", vs), ("C″(VLM框+半径封顶)", v2)):
        print(f"{name:<22}{s.get('n', 0):>7}{s.get('diam_med_px', 0):>11}"
              f"{s.get('iqr_over_med', 0):>10}{s.get('cv', 0):>8}{s.get('max_over_min_diam', 0):>9}")
    print(f"\n[C″] 未分配前景（被切出、不计入任何实例）: {unassigned}px"
          f"（占前景 {unassigned/max(1,int((mask>0).sum()))*100:.1f}%）")
    print(f"      → 语义：'漏测'而不是'测错'；粒径分布由已标记粒给出（系统无偏）")

    # 渲染对照（含 C″ 的封顶区域 + 未分配前景标记）
    os.makedirs(SEP_DIR, exist_ok=True)
    for tag, ws, pts, do_cap in (("base", base_ws, ad_pts, False),
                                 ("vlm_marker", vlm_ws, vp_pts, False),
                                 ("vlm_capped", vlm_ws, cap_pts, True)):
        if ws is None:
            continue
        vis = crop.copy()
        if do_cap:
            vis[assigned_mask] = (vis[assigned_mask] * 0.45).astype(np.uint8)  # 已分配区压暗
            vis[~assigned_mask & (mask > 0)] = (0, 0, 255)                     # 未分配=红底
        med = np.median([int((ws == v).sum()) for v in range(1, int(ws.max()))] or [1])
        for i, (y, x, r) in enumerate(pts, 1):
            mm = np.zeros(ws.shape, np.uint8)
            mm[ws == i] = 255
            if do_cap:
                yy, xx = np.mgrid[0:ws.shape[0], 0:ws.shape[1]]
                mm[(yy - y) ** 2 + (xx - x) ** 2 > r * r] = 0                # 封顶外切掉
            if int(mm.sum()) // 255 < 0.15 * med:
                continue
            cs, _ = cv2.findContours(mm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, cs, -1, (0, 200, 255), 2)
            cv2.circle(vis, (x, y), max(2, int(r)), (0, 0, 255), 1)
        s = min(1.0, 1100 / max(vis.shape[:2]))
        if s < 1.0:
            vis = cv2.resize(vis, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        p = os.path.join(SEP_DIR, f"{args.crop_id}_cmp_{tag}.jpg")
        cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 88])[1].tofile(p)
        print(f"  [图] {p}")

    out = {"crop_id": args.crop_id, "variant": args.variant, "crop_hw": [w, h],
           "vlm_boxes": len(boxes), "vlm_markers": len(vp_pts),
           "adaptive_min_dist_px": round(d_est, 1),
           "baseline_adaptive": bs, "c_prime_vlm_marker": vs, "c_double": v2,
           "note": "离线验证：同一 gen_raw 上只换 marker 来源；C″ 用框内切半径封顶把邻粒切出"
                   "（'漏测'优于'测错'）"}
    p = os.path.join(EXP_D, f"cmp_marker_{args.crop_id}.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"  [json] {p}")
    print("\n判读：实例数接近不算通过（合并与切碎可相互抵消）；口径是"
          " 中位直径是否回到单粒量级 + 直径CV/IQR中位 是否收窄")


if __name__ == "__main__":
    main()