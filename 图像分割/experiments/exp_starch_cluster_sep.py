# -*- coding: utf-8 -*-
"""exp_starch_cluster_sep.py — 实验 D：淀粉粒「团框 → 1 次 IGM → 团内实例分离」可行性

背景（2026-09-24）：gold 评测暴露 淀粉粒 标注单元口径冲突（人工整团一框 14 vs 模型逐粒 79，
5.64×）。方案 C 想让「检测框 = 裁剪单元（团）」与「测量单元 = 单粒」解耦：
  团框 1 个 → 1 次 IGM → 团 mask → 拆成 N 个实例 → N 次独立测径
几何契约已支持（mask_io._instances_from_id_png：一张图多实例），故架构上近乎免费。
**但前提是团 mask 能可靠拆开** —— 本实验就是测这个前提。

本实验不改生产代码，只回答两个问题：
  Q1 提示词要不要显式要求"保留粒间缝隙"？——生产模板说"只有一个前景"，可能把整团刷成
     一整块实心，颗粒边界丢失 → 分离无从谈起。故同 crop 跑两版提示词对照：
       prod  = igm_segment.PROMPT_TEMPLATE + 淀粉粒 igm_desc（生产口径）
       gap   = prod + 一句"保留颗粒之间的缝隙，不要把整团填成一整块"
  Q2 团 mask 拆实例的稳定性 —— 对 gen_raw 做离线后处理扫描：
       ①生产后处理（填洞=全填）→ 连通域数
       ②不填洞（保留缝隙）→ 连通域数
       ③距离变换 + 非极大抑制 marker + 分水岭，按自适应 min_dist 扫描 → 实例数曲线
       曲线出现平台 ⇒ 分离稳定可用；随参数单调变化 ⇒ 不可用。

产物：results/segmentation/igm_seg_expD/
  crops/<crop_id>.png + .json      团 crop（人工团框 + margin，来源可回溯）
  <crop_id>/<variant>/             逐次 IGM 产物（沿用 igm_segment 落盘结构）
  sep/<crop_id>_<variant>.jpg      分离结果叠加图（目检）
  summary.json                     实测表（供报告引用）

用法（默认幂等，已有 meta.json 即复用，不扣额度）:
  python exp_starch_cluster_sep.py --dry-run        # 只列将发起的调用
  python exp_starch_cluster_sep.py                 # 真实跑（3 crop × 2 提示词 = 6 次调用）
  python exp_starch_cluster_sep.py --offline       # 只做离线分离分析，绝不调用 API
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SEG_DIR = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, SEG_DIR)
sys.path.insert(0, os.path.join(os.path.dirname(SEG_DIR), "common"))
# detection/ 也须注入：本实验用 ls_tasks（LS 任务读取）。此前它靠 import_module("05_/13_") 取同类函数，
# 而那两个也在 detection/ —— 即"用任意解释器独立运行"本来就不成立，这里一并补齐（P4-3）。
sys.path.insert(0, os.path.join(os.path.dirname(SEG_DIR), "detection"))

import igm_segment as ig              # noqa: E402
import igm_exp_common as ec           # noqa: E402
from ls_common import canon_path, imread_cn  # noqa: E402

RESULTS_SEG = os.path.join(os.path.dirname(SEG_DIR), "results", "segmentation")
EXP_D = os.path.join(RESULTS_SEG, "igm_seg_expD")
CROPS_DIR = os.path.join(EXP_D, "crops")
SEP_DIR = os.path.join(EXP_D, "sep")
GOLD = os.path.join(os.path.dirname(SEG_DIR), "results", "detection", "det_eval", "gold_set.json")
MARGIN = 0.10                      # 团框外扩比例（团框已含整团，10% 留邻域上下文即可）

# 对照提示词：prod = 生产模板原文；gap = 追加"保留粒间缝隙"
GAP_RULE = ("注意：图中前景可能由许多相互紧挨的颗粒组成。请逐个颗粒分别勾勒，"
            "**保留颗粒之间的缝隙与暗边界**（缝隙属于背景，画成黑色），"
            "不要把多颗颗粒连成一整块实心色斑，也不要凭空合并或增删颗粒。")


def build_crops(gold_path=GOLD, n=3):
    """从 gold 里取淀粉粒的「人工团框（最大框）」→ 团 crop + 随行 json。"""
    gold = json.load(open(gold_path, encoding="utf-8"))
    pre_dir = os.path.join(os.path.dirname(SEG_DIR), "results", "detection", "prelabels")
    items = []
    for it in gold["items"]:
        if it["feature_label"] != "淀粉粒":
            continue
        stem = it["stem"]
        # 找该图的 prelabel（按 canon_path 后的路径比对），取其 object 数作参考
        items.append({"image_path": canon_path(it["image_path"]), "stem": stem,
                      "herb": it.get("herb", ""), "feature_id": "starch_grain"})
    items = items[:n]
    if not items:
        sys.exit("[FAIL] gold_set.json 里没有淀粉粒项（先跑 det_eval --build-gold）")
    _ = pre_dir
    ann = _load_gold_annotations()
    os.makedirs(CROPS_DIR, exist_ok=True)
    out = []
    for it in items:
        img = imread_cn(it["image_path"])
        if img is None:
            print(f"[skip] 读不到 {it['image_path']}")
            continue
        H, W = img.shape[:2]
        boxes = ann.get(os.path.normpath(it["image_path"]).replace("\\", "/").lower())
        if not boxes:
            print(f"[skip] 无人工标注 {it['stem']}")
            continue
        # 人工最大框 = 团框（整团一框那一个）
        box = max((b["box"] for b in boxes), key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
        x1, y1, x2, y2 = [float(v) for v in box]
        pw, ph = (x2 - x1) * MARGIN, (y2 - y1) * MARGIN
        lx1, ly1 = int(max(0, x1 - pw)), int(max(0, y1 - ph))
        lx2, ly2 = int(min(W, x2 + pw)), int(min(H, y2 + ph))
        crop = img[ly1:ly2, lx1:lx2]
        crop_id = f"{it['feature_id']}_{it['stem']}"
        png = os.path.join(CROPS_DIR, crop_id + ".png")
        cv2.imencode(".png", crop)[1].tofile(png)      # 中文名安全：imencode + tofile
        cj = {"crop_id": crop_id, "feature_id": it["feature_id"], "stem": it["stem"],
              "herb": it["herb"], "source_image_path": it["image_path"],
              "tight_box_local": [round(x1 - lx1, 1), round(y1 - ly1, 1),
                                  round(x2 - lx1, 1), round(y2 - ly1, 1)],
              "loose_box": [lx1, ly1, lx2, ly2], "margin": MARGIN,
              "n_human_boxes": len(boxes), "note": "实验D 团 crop：由人工团框外扩 margin 得到"}
        with open(os.path.join(CROPS_DIR, crop_id + ".json"), "w", encoding="utf-8") as f:
            json.dump(cj, f, ensure_ascii=False, indent=2)
        out.append({"crop": png, "crop_id": crop_id, **cj,
                    "crop_hw": list(crop.shape[:2])})
        print(f"[crop] {crop_id}: {crop.shape[1]}x{crop.shape[0]} "
              f"（人工框 {len(boxes)} 个，取最大者作团框）")
    return out


def _load_gold_annotations():
    """从 LS 项目 31 读人工框 → {norm_path: [{'label','box'}]}。"""
    from ls_common import load_config, load_prelabel_index
    from ls_tasks import ann_from_tasks, fetch_tasks
    gold = json.load(open(GOLD, encoding="utf-8"))
    gold["items"] = [dict(x, image_path=canon_path(x["image_path"])) for x in gold["items"]]
    pre_index = load_prelabel_index()
    ann, _, _ = ann_from_tasks(fetch_tasks(load_config(), 31), gold, pre_index)
    return ann


# ---------------------------------------------------------------- 离线分离分析
def peaks_nms(dist, min_dist):
    """距离图上的非极大抑制取峰：峰值按距离降序贪心接受，间距 ≥ min_dist。"""
    k = max(3, int(round(min_dist)) | 1)
    local_max = dist >= cv2.dilate(dist, np.ones((k, k), np.uint8)) - 1e-6
    ys, xs = np.nonzero(local_max & (dist > 1.0))
    order = np.argsort(-dist[ys, xs])
    acc = []
    for i in order:
        y, x = int(ys[i]), int(xs[i])
        if all((y - ay) ** 2 + (x - ax) ** 2 >= min_dist ** 2 for ay, ax in acc):
            acc.append((y, x))
    return acc


def watershed_instances(mask, min_dist):
    """距离变换 + marker 分水岭 → (实例 mask 列表(面积 px), marker 图)。"""
    m8 = (mask > 0).astype(np.uint8)
    if not m8.any():
        return [], None, None
    dist = cv2.distanceTransform(m8, cv2.DIST_L2, 5)
    pts = peaks_nms(dist, min_dist)
    if not pts:
        return [], None, None
    markers = np.zeros(m8.shape, np.int32)
    for i, (y, x) in enumerate(pts, 1):
        markers[y, x] = i
    markers[m8 == 0] = len(pts) + 1
    bgr = cv2.cvtColor(m8 * 255, cv2.COLOR_GRAY2BGR)
    ws = cv2.watershed(bgr, markers.copy())
    areas = []
    for v in range(1, len(pts) + 1):
        a = int((ws == v).sum())
        if a > 0:
            areas.append(a)
    return areas, ws, dist


def analyze_run(run_dir, crop_bgr, tag):
    """对一次 IGM 产物做分离分析，返回指标 dict + 渲染叠加图。"""
    gen = ig.imread_any(os.path.join(run_dir, "gen_raw.png"))
    meta = json.load(open(os.path.join(run_dir, "meta.json"), encoding="utf-8"))
    h, w = crop_bgr.shape[:2]
    g = cv2.cvtColor(gen, cv2.COLOR_BGR2GRAY) if gen.ndim == 3 else gen
    roi = ig._crop_content(g, meta["letterbox"])
    small = cv2.resize(roi, (w, h), interpolation=cv2.INTER_AREA)
    inv = meta.get("invert")
    t = meta.get("threshold")
    if t is None:
        t, _ = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    flag = cv2.THRESH_BINARY_INV if inv else cv2.THRESH_BINARY
    raw = cv2.threshold(small, int(t), 255, flag)[1]          # 未填洞（保留缝隙）

    masks = {
        "prod(填洞=全填+形态学)": (raw.copy(), True),
        "nofill(保留缝隙)": (raw.copy(), False),
    }
    res = {"tag": tag, "invert": inv, "threshold": float(t),
           "gen_hw": list(gen.shape[:2]), "mask_hw": [h, w]}
    overlays = []
    for name, (mk, do_prod) in masks.items():
        m = mk
        if do_prod:
            cap = None                                            # 生产默认：全填
            m = ig.fill_holes_limited(m, cap)
            m = ig.clean_mask(m, ig.MIN_AREA_DEFAULT, False)
        n_cc = max(0, int(cv2.connectedComponents((m > 0).astype(np.uint8), 8)[0]) - 1)
        area_ratio = float((m > 0).mean())
        entry = {"n_components": n_cc, "area_ratio": round(area_ratio, 4)}
        if not do_prod:
            # 自适应粒径直估计：距离图局部极大的距离中位数 → 粒半径
            dist = cv2.distanceTransform((m > 0).astype(np.uint8), cv2.DIST_L2, 5)
            lm = dist >= cv2.dilate(dist, np.ones((5, 5), np.uint8)) - 1e-6
            vals = dist[lm & (dist > 2)]
            r_est = float(np.median(vals)) if vals.size else 0.0
            d_est = max(4.0, 2.0 * r_est)
            entry["r_est_px"] = round(r_est, 1)
            for alpha in (0.6, 0.8, 1.0, 1.3, 1.6):
                md = max(3.0, alpha * d_est)
                areas, ws, _ = watershed_instances(m, md)
                med = float(np.median(areas)) if areas else 0.0
                keep = [a for a in areas if a >= 0.15 * med] if med else []
                entry[f"ws_alpha{alpha}"] = len(keep)
                if abs(alpha - 1.0) < 1e-9 and ws is not None:
                    entry["ws_diam_px_med"] = round(2 * np.sqrt(med / np.pi), 1) if med else 0
                    vis = crop_bgr.copy()
                    for v in range(1, int(ws.max())):
                        mm = np.zeros(ws.shape, np.uint8)
                        mm[ws == v] = 255
                        if int(mm.sum()) // 255 < 0.15 * med:
                            continue
                        cs, _ = cv2.findContours(mm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        cv2.drawContours(vis, cs, -1, (0, 0, 255), 2)
                    overlays.append((f"{tag}", vis))
        res[name] = entry
    return res, overlays


def render(overlays, prefix):
    os.makedirs(SEP_DIR, exist_ok=True)
    paths = []
    for tag, vis in overlays:
        s = min(1.0, 1100 / max(vis.shape[:2]))
        if s < 1.0:
            vis = cv2.resize(vis, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        p = os.path.join(SEP_DIR, f"{prefix}_{tag}.jpg")
        cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 88])[1].tofile(p)
        paths.append(p)
    return paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3, help="用几张团 crop（默认 3）")
    ap.add_argument("--dry-run", action="store_true", help="只列将发起的调用，不执行")
    ap.add_argument("--offline", action="store_true", help="只做离线分析，缺产物即报错")
    ap.add_argument("--force", action="store_true", help="重跑 IGM（默认复用已有 meta.json）")
    args = ap.parse_args()

    os.makedirs(EXP_D, exist_ok=True)
    crops = build_crops(n=args.n)
    variants = {"prod": ig.build_prompt(ig.load_feature_desc("starch_grain")),
                "gap": ig.build_prompt(ig.load_feature_desc("starch_grain")) + GAP_RULE}
    calls = len(crops) * len(variants)
    ec.print_usage_warning(calls)
    if args.dry_run:
        for c in crops:
            for v in variants:
                print(f"  [dry-run] {c['crop_id']} / {v} → "
                      f"{os.path.join(EXP_D, c['crop_id'], v)}")
        print(f"\n提示词对照：prod = 生产模板原文；gap = prod + 「保留粒间缝隙」一句")
        return

    summary = []
    for c in crops:
        crop_bgr = imread_cn(c["crop"])
        for vname, prompt in variants.items():
            out_dir = os.path.join(EXP_D, c["crop_id"], vname)
            if args.offline and not os.path.exists(os.path.join(out_dir, "meta.json")):
                print(f"[offline] 缺产物，跳过 {c['crop_id']}/{vname}")
                continue
            ok, meta = ec.run_igm(c["crop"], prompt, out_dir, reuse=not args.force)
            if not ok:
                print(f"[FAIL] {c['crop_id']}/{vname} IGM 调用失败")
                continue
            res, ovl = analyze_run(out_dir, crop_bgr, vname)
            res.update({"crop_id": c["crop_id"], "crop_hw": c["crop_hw"],
                        "n_human_boxes": c["n_human_boxes"],
                        "igm_status": meta.get("status"),
                        "igm_n_components": meta.get("n_components")})
            render(ovl, f"{c['crop_id']}_{vname}")
            summary.append(res)
            e = res["nofill(保留缝隙)"]
            print(f"[结果] {c['crop_id']}/{vname}: 生成图{res['gen_hw']} | "
                  f"生产后处理连通域 {res['prod(填洞=全填+形态学)']['n_components']} | "
                  f"不填洞连通域 {e['n_components']} | r_est={e.get('r_est_px')}px | "
                  f"分水岭实例 α=1.0 → {e.get('ws_alpha1.0')} "
                  f"（α0.6/0.8/1.0/1.3/1.6 = {[e.get(f'ws_alpha{a}') for a in (0.6,0.8,1.0,1.3,1.6)]}）")

    if summary:
        with open(os.path.join(EXP_D, "summary.json"), "w", encoding="utf-8") as f:
            json.dump({"margin": MARGIN, "gap_rule": GAP_RULE, "runs": summary},
                      f, ensure_ascii=False, indent=2)
        print(f"\n汇总 → {os.path.join(EXP_D, 'summary.json')}")
        print(f"叠加图 → {SEP_DIR}")


if __name__ == "__main__":
    main()