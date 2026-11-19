# -*- coding: utf-8 -*-
"""e2e_eval.py — 两阶段检测的**端到端验收**：stage 1 提框 + stage 2 分类，在同一批 gold 上算
「框对且类对」。

**为什么需要它**：stage 1 的框级召回（`box_recall`）与 stage 2 的 crop 分类准确率
（`train_cls report.json`）分开看都还行，但两者是串联的——只有端到端口径能回答
"一张 original 图上，最终能不能给出正确的特征与位置"。本入口把两个模型接起来，
并保留三段可归因的数字：定位召回、命中框上的分类准确率、端到端召回/精确。

**口径**：
  · gold：与 `box_recall` / `det_eval` 同一批（60 图 / 239 框，`det_eval/p3/`）；
  · stage 1：`detector_yolo.detect_agnostic`（类别无关，conf 固定 0.01 跑一次）；
  · crop：`geom.loose_box(短边×DET_LOOSE_MARGIN_RATIO)`——与训练集切法逐像素同口径；
  · stage 2：`cls_net.classify_arrays`（与训练共用 320 letterbox 变换）；
  · 匹配：`box_metrics.match_pairs`，IoU≥0.5 贪心一对一；框匹配后**再**比类别，
    类别不等记为"框对类错"（单独统计，不混入定位召回）。

用法（项目根）：
    python -X utf8 目标检测/cli/e2e_eval.py \
        --stage1-weights results/detection/yolo_runs/v1_yolov10s_1024/weights/best.pt \
        --cls-weights results/detection/cls_runs/abl_320_lb_e40_ls01_mix01_s20261030/best.pt \
        --stage1-imgsz 1024

退出码：0 = 成功；2 = 权重/清单缺失。
"""
import argparse
import json
import os
import sys
from collections import Counter

import detector_yolo
import paths
from box_metrics import load_gold, match_pairs
from cls_net import classify_arrays, describe
from detect_core import DET_LOOSE_MARGIN_RATIO
from geom import loose_box
from ls_common import canon_path, imread_cn, norm_path

GOLD_SET = os.path.join(paths.DET_RESULTS, "det_eval", "p3", "gold_set.json")
GOLD_ANN = os.path.join(paths.DET_RESULTS, "det_eval", "p3", "gold_ann.json")
OUT_DIR = os.path.join(paths.DET_RESULTS, "e2e")
DEFAULT_STAGE1 = os.path.join(paths.DET_RESULTS, "yolo_runs", "v1_yolov10s_1024",
                              "weights", "best.pt")
DEFAULT_STAGE2 = os.path.join(paths.DET_RESULTS, "cls_runs",
                              "abl_320_lb_e40_ls01_mix01_s20261030", "best.pt")

INFER_CONF = 0.01                 # 一次推理到 0.01（再低只会让分类白跑），扫描只做过滤
CONF_GRID = (0.25, 0.10, 0.05, 0.01)


def _abs(p):
    return p if os.path.isabs(p) else os.path.join(paths.PROJECT_ROOT, p)


def predict_all(gold, stage1_weights, cls_weights, imgsz, iou, device, max_det,
                margin, log=print):
    """逐图：stage 1 提框 → 松框裁 crop → stage 2 分类 → `{key: [box,score,label,prob]}`。"""
    preds = {}
    for i, it in enumerate(gold, 1):
        p = canon_path(it["image_path"])
        img = imread_cn(p)
        if img is None:
            raise FileNotFoundError("读图失败：{}".format(p))
        H, W = img.shape[:2]
        dets = detector_yolo.detect_agnostic(
            img, weights=stage1_weights, imgsz=imgsz, conf=INFER_CONF, iou=iou,
            device=device, max_det=max_det)
        crops, boxes = [], []
        for d in dets:
            ox, oy, cx2, cy2 = loose_box(d["box"], W, H, margin_ratio=margin)
            crop = img[oy:cy2, ox:cx2]
            if crop.size == 0:
                continue
            crops.append(crop)
            boxes.append(d)
        labels = classify_arrays(crops, cls_weights, device=device) if crops else []
        preds[norm_path(p)] = [
            {"box": b["box"], "score": b["score"], "label": r["label"], "prob": r["prob"]}
            for b, r in zip(boxes, labels)]
        if i % 10 == 0 or i == len(gold):
            log("  [{}/{}] {} → 检测 {} 框".format(i, len(gold), os.path.basename(p), len(boxes)))
    return preds


def evaluate(ann, preds, conf=0.25, iou_thr=0.5):
    """按 conf 过滤后聚合：定位召回 / 命中框分类准确率 / 端到端召回与精确（纯函数）。"""
    n_gt = n_pred = loc_tp = cls_tp = 0
    per_class, confusions, per_image = {}, Counter(), []
    for key, aboxes in ann.items():
        P = [p for p in preds.get(key, []) if p["score"] >= conf]
        P.sort(key=lambda p: -p["score"])
        pairs, _unused = match_pairs(aboxes, P, iou_thr)
        matched = {gi: P[j] for gi, j, _ in pairs}
        correct = 0
        for gi, a in enumerate(aboxes):
            st = per_class.setdefault(a["label"], {"n_gt": 0, "loc_tp": 0, "cls_tp": 0})
            st["n_gt"] += 1
            if gi in matched:
                st["loc_tp"] += 1
                if matched[gi]["label"] == a["label"]:
                    st["cls_tp"] += 1
                    correct += 1
                else:
                    confusions["{}→{}".format(a["label"], matched[gi]["label"])] += 1
        n_gt += len(aboxes)
        n_pred += len(P)
        loc_tp += len(pairs)
        cls_tp += correct
        per_image.append({"image": os.path.basename(key), "n_gt": len(aboxes),
                          "n_pred": len(P), "loc_tp": len(pairs), "cls_tp": correct})
    for st in per_class.values():
        st["loc_recall"] = round(st["loc_tp"] / st["n_gt"], 4) if st["n_gt"] else None
        st["cls_recall"] = round(st["cls_tp"] / st["n_gt"], 4) if st["n_gt"] else None
        st["cls_acc_on_matched"] = (round(st["cls_tp"] / st["loc_tp"], 4)
                                    if st["loc_tp"] else None)
    return {
        "conf": conf, "iou_thr": iou_thr,
        "n_images": len(ann), "n_gt": n_gt, "n_pred": n_pred,
        "n_pred_per_image": round(n_pred / max(1, len(ann)), 2),
        "loc_tp": loc_tp,
        "recall_loc": round(loc_tp / n_gt, 4) if n_gt else None,
        "precision_loc": round(loc_tp / n_pred, 4) if n_pred else None,
        "cls_tp": cls_tp,
        "cls_acc_on_matched": round(cls_tp / loc_tp, 4) if loc_tp else None,
        "recall_e2e": round(cls_tp / n_gt, 4) if n_gt else None,
        "precision_e2e": round(cls_tp / n_pred, 4) if n_pred else None,
        "per_class": per_class,
        "confusions": dict(confusions.most_common()),
        "per_image": per_image,
    }


def build_report(ann, preds, args, include_per_image=False):
    rep = {
        "stage1": {"weights": os.path.relpath(_abs(args.stage1_weights), paths.PROJECT_ROOT)
                   .replace(os.sep, "/"),
                   "imgsz": args.stage1_imgsz, "infer_conf": INFER_CONF,
                   "nms_iou": args.nms_iou, "margin_ratio": args.margin},
        "stage2": describe(_abs(args.cls_weights)),
        "match_iou_thr": args.iou_thr,
        "conf_sweep": [evaluate(ann, preds, conf=c, iou_thr=args.iou_thr) for c in CONF_GRID],
    }
    rep["stage2"]["path"] = os.path.relpath(_abs(args.cls_weights), paths.PROJECT_ROOT)\
        .replace(os.sep, "/")
    if not include_per_image:
        for block in rep["conf_sweep"]:
            block.pop("per_image", None)
    return rep


def _print_report(rep):
    print("\n[端到端 | IoU≥{} | stage1={}@{} | stage2={}]".format(
        rep["match_iou_thr"], os.path.basename(rep["stage1"]["weights"]),
        rep["stage1"]["imgsz"], rep["stage2"]["arch"]))
    print("  {:>6} {:>7} {:>8} {:>8} {:>9} {:>9} {:>9}".format(
        "conf", "框/图", "定位R", "定位P", "命中框分类", "端到端R", "端到端P"))
    for b in rep["conf_sweep"]:
        print("  {:>6} {:>7} {:>8} {:>8} {:>9} {:>9} {:>9}".format(
            b["conf"], b["n_pred_per_image"], b["recall_loc"], b["precision_loc"],
            b["cls_acc_on_matched"], b["recall_e2e"], b["precision_e2e"]))
    best = rep["conf_sweep"][0]
    print("\n[逐类 | conf={}]".format(best["conf"]))
    for zh, st in sorted(best["per_class"].items(), key=lambda kv: kv[1]["cls_recall"] or 0):
        print("  {:<8} n={:<4} 定位R {:<6} 类召回 {:<6} 命中框分类acc {}".format(
            zh, st["n_gt"], st["loc_recall"], st["cls_recall"], st["cls_acc_on_matched"]))
    if best["confusions"]:
        print("[框对类错 top]")
        for k, v in list(best["confusions"].items())[:8]:
            print("  {} × {}".format(k, v))


def main(argv=None):
    ap = argparse.ArgumentParser(description="两阶段端到端验收（stage1 提框 + stage2 分类）")
    ap.add_argument("--stage1-weights", default=DEFAULT_STAGE1)
    ap.add_argument("--stage1-imgsz", type=int, default=1024)
    ap.add_argument("--cls-weights", default=DEFAULT_STAGE2)
    ap.add_argument("--gold", default=GOLD_SET)
    ap.add_argument("--ann", default=GOLD_ANN)
    ap.add_argument("--iou-thr", type=float, default=0.5)
    ap.add_argument("--nms-iou", type=float, default=0.45)
    ap.add_argument("--margin", type=float, default=DET_LOOSE_MARGIN_RATIO)
    ap.add_argument("--device", default="0")
    ap.add_argument("--max-det", type=int, default=300)
    ap.add_argument("--out", default="", help="输出 JSON（默认 results/detection/e2e/e2e_<时间>.json）")
    ap.add_argument("--per-image", action="store_true")
    args = ap.parse_args(argv)

    for label, p in (("stage1 权重", args.stage1_weights), ("stage2 权重", args.cls_weights),
                     ("gold 清单", args.gold), ("gold 标注", args.ann)):
        if not os.path.isfile(_abs(p)):
            print("[FAIL] {}不存在：{}".format(label, _abs(p)), file=sys.stderr)
            return 2

    gold, ann = load_gold(_abs(args.gold), _abs(args.ann))
    print("[info] gold {} 图 / {} 框 | stage1 imgsz={} conf≥{} | margin={:.2f}".format(
        len(gold), sum(len(v) for v in ann.values()), args.stage1_imgsz,
        INFER_CONF, args.margin))
    preds = predict_all(gold, _abs(args.stage1_weights), _abs(args.cls_weights),
                        args.stage1_imgsz, args.nms_iou, args.device, args.max_det,
                        args.margin)
    rep = build_report(ann, preds, args, args.per_image)

    out = _abs(args.out) if args.out else os.path.join(
        OUT_DIR, "e2e_{}".format(os.path.splitext(os.path.basename(args.cls_weights))[0]))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)
    _print_report(rep)
    print("\n[done] {}".format(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
