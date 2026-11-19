# -*- coding: utf-8 -*-
"""box_recall.py — 检测器的**类别无关框级指标**（stage 1「无类别提框」的验收口径）。

**为什么需要它**：`det_eval` 的召回是**类别匹配**后的（预测类 == 人工类），衡量的是
"检测+命名"的合成能力；两阶段方案要把 stage 1 拆成只提框，就必须单独看"框提得对不对"。
本入口在同一批 gold 图上算类别无关的召回/精确，并按置信度阈值与每图提案数（top-K）扫描，
用来判断「换分辨率 / 换单类模型 / 换滑窗」是否真的提高了 proposal 覆盖。

**口径**（与 det_eval 的 gold 一致）：
  · 匹配：每个 GT 框最多匹配一个预测框（贪心取最大 IoU，IoU ≥ `--iou-thr`，默认 0.5）；
  · 预测来源：`detector_yolo.detect_agnostic`（不依赖 14 类表，single_cls 权重同样可用）；
  · 推理阈值固定 `--infer-conf 0.001` 一次跑完，再按 `--conf` 扫描与 top-K 截断——同一批
    预测只推理一次，扫描之间严格可比；
  · 只报**定位**指标，不报类别；类别正确率由 `det_eval` 负责。

用法（项目根）：
    python -X utf8 目标检测/cli/box_recall.py --weights <best.pt> --imgsz 1280 \
        --out results/detection/box_recall/v10s_img1280.json

退出码：0 = 成功；2 = 输入缺失/权重不存在。
"""
import argparse
import json
import os
import sys

import detector_yolo
import paths
from box_metrics import load_gold, match_image
from ls_common import canon_path, imread_cn, norm_path

GOLD_SET = os.path.join(paths.DET_RESULTS, "det_eval", "p3", "gold_set.json")
GOLD_ANN = os.path.join(paths.DET_RESULTS, "det_eval", "p3", "gold_ann.json")
OUT_DIR = os.path.join(paths.DET_RESULTS, "box_recall")

INFER_CONF = 0.001          # 一次推理到地板值；扫描只做过滤，不重复推理
CONF_GRID = (0.25, 0.10, 0.05, 0.01, 0.001)
TOPK_GRID = (1, 3, 5, 10, 20, 50, 100)


def predict_all(gold, weights, imgsz, tile, overlap, iou, device, max_det, log=print):
    preds = {}
    for i, it in enumerate(gold, 1):
        p = canon_path(it["image_path"])
        img = imread_cn(p)
        if img is None:
            raise FileNotFoundError("读图失败：{}".format(p))
        preds[norm_path(p)] = detector_yolo.detect_agnostic(
            img, weights=weights, imgsz=imgsz, conf=INFER_CONF, iou=iou, device=device,
            max_det=max_det, tile=tile, overlap=overlap)
        if i % 10 == 0 or i == len(gold):
            log("  [{}/{}] {} → {} 框".format(i, len(gold), os.path.basename(p),
                                             len(preds[norm_path(p)])))
    return preds


def evaluate(ann, preds, conf=0.25, topk=None, iou_thr=0.5):
    """按阈值/每图提案数过滤后聚合（纯函数）。"""
    n_gt = n_pred = tp = 0
    per_class = {}
    missed = {}
    per_image = []
    for key, aboxes in ann.items():
        P = [p for p in preds.get(key, []) if p["score"] >= conf]
        P.sort(key=lambda p: -p["score"])
        if topk:
            P = P[:topk]
        hits, _unused, best_ious = match_image(aboxes, P, iou_thr)
        n_gt += len(aboxes)
        n_pred += len(P)
        tp += len(hits)
        per_image.append({"image": os.path.basename(key), "n_gt": len(aboxes),
                          "n_pred": len(P), "tp": len(hits)})
        for gi, a in enumerate(aboxes):
            st = per_class.setdefault(a["label"], {"n_gt": 0, "tp": 0})
            st["n_gt"] += 1
            if gi in hits:
                st["tp"] += 1
            else:
                missed[a["label"]] = missed.get(a["label"], 0) + 1
    for st in per_class.values():
        st["recall"] = round(st["tp"] / st["n_gt"], 4) if st["n_gt"] else None
    return {
        "conf": conf, "topk_per_image": topk, "iou_thr": iou_thr,
        "n_images": len(ann), "n_gt": n_gt, "n_pred": n_pred,
        "n_pred_per_image": round(n_pred / max(1, len(ann)), 2),
        "tp_loc": tp,
        "recall_loc": round(tp / n_gt, 4) if n_gt else None,
        "precision_loc": round(tp / n_pred, 4) if n_pred else None,
        "per_class_recall": per_class,
        "missed_by_class": missed,
        "per_image": per_image,
    }


def build_report(ann, preds, weights, imgsz, tile, overlap, iou_thr, include_per_image):
    out = {
        "weights": weights if os.path.isabs(weights) else os.path.abspath(weights),
        "imgsz": imgsz, "tile": tile, "overlap": overlap,
        "infer_conf": INFER_CONF, "match_iou_thr": iou_thr,
        "n_images": len(ann),
        "n_gt": sum(len(v) for v in ann.values()),
        "conf_sweep": [evaluate(ann, preds, conf=c, iou_thr=iou_thr) for c in CONF_GRID],
        "topk_sweep": [evaluate(ann, preds, conf=INFER_CONF, topk=k, iou_thr=iou_thr)
                       for k in TOPK_GRID],
        "strict_iou75": evaluate(ann, preds, conf=0.25, iou_thr=0.75),
    }
    out["weights"] = os.path.relpath(os.path.abspath(weights), paths.PROJECT_ROOT).replace(os.sep, "/")
    if not include_per_image:
        for block in out["conf_sweep"] + out["topk_sweep"] + [out["strict_iou75"]]:
            block.pop("per_image", None)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="类别无关框级指标（stage 1 提框验收）")
    ap.add_argument("--weights", required=True, help="YOLO 权重（多类或 single_cls 均可）")
    ap.add_argument("--gold", default=GOLD_SET)
    ap.add_argument("--ann", default=GOLD_ANN)
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--tile", type=int, default=0, help="滑窗边长；0=整图一次")
    ap.add_argument("--overlap", type=float, default=0.2)
    ap.add_argument("--iou-thr", type=float, default=0.5, help="匹配 IoU 阈值")
    ap.add_argument("--nms-iou", type=float, default=0.45, help="推理内部 NMS/跨窗去重阈值")
    ap.add_argument("--device", default="0")
    ap.add_argument("--max-det", type=int, default=300)
    ap.add_argument("--out", default="", help="输出 JSON（默认 results/detection/box_recall/<权重名>_img<尺寸>.json）")
    ap.add_argument("--per-image", action="store_true", help="报告里保留逐图明细")
    args = ap.parse_args(argv)

    w = args.weights if os.path.isabs(args.weights) else os.path.join(paths.PROJECT_ROOT, args.weights)
    if not os.path.isfile(w):
        print("[FAIL] 权重不存在：{}".format(w), file=sys.stderr)
        return 2
    if not (os.path.isfile(args.gold) and os.path.isfile(args.ann)):
        print("[FAIL] 缺 gold 清单：{} / {}".format(args.gold, args.ann), file=sys.stderr)
        return 2

    gold, ann = load_gold(args.gold, args.ann)
    print("[info] gold {} 图 / {} 框；imgsz={} tile={} overlap={}".format(
        len(gold), sum(len(v) for v in ann.values()), args.imgsz, args.tile, args.overlap))
    preds = predict_all(gold, w, args.imgsz, args.tile, args.overlap, args.nms_iou,
                        args.device, args.max_det)
    rep = build_report(ann, preds, w, args.imgsz, args.tile, args.overlap,
                       args.iou_thr, args.per_image)

    out = args.out or os.path.join(OUT_DIR, "{}_img{}{}.json".format(
        os.path.splitext(os.path.basename(w))[0], args.imgsz,
        "_tile{}".format(args.tile) if args.tile else ""))
    out = out if os.path.isabs(out) else os.path.join(paths.PROJECT_ROOT, out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)

    print("\n[conf 扫描 | 类别无关 IoU≥{}]".format(args.iou_thr))
    for b in rep["conf_sweep"]:
        print("  conf {:>5}: 框/图 {:>5}  召回 {}  精确 {}".format(
            b["conf"], b["n_pred_per_image"], b["recall_loc"], b["precision_loc"]))
    print("[top-K/图 | conf={}]".format(INFER_CONF))
    for b in rep["topk_sweep"]:
        print("  K {:>3}: 框/图 {:>5}  召回 {}  精确 {}".format(
            b["topk_per_image"], b["n_pred_per_image"], b["recall_loc"], b["precision_loc"]))
    print("[严格 IoU≥0.75 | conf=0.25] 召回 {} 精确 {}".format(
        rep["strict_iou75"]["recall_loc"], rep["strict_iou75"]["precision_loc"]))
    print("[逐类召回 | conf=0.25]")
    for zh, st in sorted(rep["conf_sweep"][0]["per_class_recall"].items(),
                         key=lambda kv: kv[1]["recall"] or 0):
        print("  {:<8} {}/{} = {}".format(zh, st["tp"], st["n_gt"], st["recall"]))
    print("\n[done] {}".format(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
