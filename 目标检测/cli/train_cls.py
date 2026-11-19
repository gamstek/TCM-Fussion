# -*- coding: utf-8 -*-
"""train_cls.py — stage 2 特征分类器训练入口（crop 图 → 14 类）。

**为什么存在**：两阶段检测把「定位」与「命名」拆开；stage 1 只给框，本入口训 stage 2 的
crop 分类器。训练数据必须与推理**同口径**——即 `build_cls_dataset.py` 用
`detect_core.DET_LOOSE_MARGIN_RATIO`（30% 松框）从 P0 人工框重切的
`dataset/特征分类器_v1/crops/{train,val,test}/<train_id>_<feature_id>/*.jpg`，
不是盘上由 VLM 实时检测产出的 `crops/`。

**类别口径**：14 类含「其他」（`13_other` 是第 14 个正类，不是背景）。**本数据集没有
「都不是/非特征」拒绝类**——stage 1 的假阳性要等检测 FP crop 回流后才能补，训练侧
不伪造该样本。评估报告按类给 P/R/F1，避免整体准确率被大类掩盖。

**模型**：torchvision ResNet18（ImageNet 预训练）+ 替换 fc；输入 224、ImageNet 归一化。
只做分类器本体，**不接线** `classify_crop(impl="model")`——推理接入是单独一步。

**输出**：`results/detection/cls_runs/<name>/`
    best.pt          最高 val macro-F1 的权重
    model_meta.json  arch / imgsz / 类别表 / 归一化参数（推理侧重建用）
    history.csv      每轮 train/val 损失与指标
    report.json      test 指标 + 每类 P/R/F1 + 混淆矩阵 + 数据统计

用法（项目根执行）:
  python 目标检测/cli/train_cls.py --dry-run          # 只校验数据与参数
  python 目标检测/cli/train_cls.py                    # 默认 20 轮
  python 目标检测/cli/train_cls.py --epochs 30 --batch 64 --workers 8
"""
import argparse
import csv
import json
import os
import random
import sys

import paths
from cls_net import build_transforms

DEFAULT_DATA = os.path.join(paths.CLS_TRAIN_V1, "crops")
DEFAULT_OUT = os.path.join(paths.DET_RESULTS, "cls_runs")
DEFAULT_SEED = 20261030
SPLITS = ("train", "val", "test")
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
RARE_TRAIN_WARN = 100
RARE_TRAIN_CRITICAL = 30
CLASS_WEIGHT_MODES = ("sqrt", "inv", "none")
INPUT_MODES = ("center", "squash", "letterbox")


def scan_splits(data_dir, limit=0):
    """扫 `crops/<split>/<class_dir>/*.jpg` → (每 split 每类计数, 类别目录名列表)。

    三个 split 的类别目录必须**完全一致**：不一致说明数据集是半成品（重建中断），
    分类器会静默少类，故直接报错而不是取交集。
    """
    counts, names = {}, None
    for split in SPLITS:
        root = os.path.join(data_dir, split)
        if not os.path.isdir(root):
            raise FileNotFoundError("缺少 split 目录：{}".format(root))
        dirs = sorted(d for d in os.listdir(root)
                      if os.path.isdir(os.path.join(root, d)))
        if not dirs:
            raise FileNotFoundError("{} 下没有任何类别目录".format(root))
        if names is None:
            names = dirs
        elif dirs != names:
            raise ValueError("{} 的类别目录与 train 不一致：{} vs {}".format(split, dirs, names))
        per = {}
        for d in dirs:
            files = [f for f in os.listdir(os.path.join(root, d))
                     if f.lower().endswith(IMG_EXTS)]
            per[d] = len(files[:limit]) if limit else len(files)
            if per[d] == 0:
                raise ValueError("{}/{} 没有图像".format(split, d))
        counts[split] = per
    return counts, names


def class_weights(counts, mode="sqrt"):
    """类别权重：`sqrt` = 平方根逆频率（压大类但不至于把稀有类放大到过拟合）。

    `inv` 为完全逆频率，`none` 为等权。权重按均值归一，使总损失量级不随模式漂移。
    """
    import math
    n = sum(counts)
    k = len(counts)
    if mode == "none":
        return [1.0] * k
    expo = 0.5 if mode == "sqrt" else 1.0
    raw = [(n / (k * max(c, 1))) ** expo for c in counts]
    mean = sum(raw) / len(raw)
    return [w / mean for w in raw]


def macro_scores(conf, names):
    """混淆矩阵 → 整体准确率 + 每类 P/R/F1 + macro-F1（只统计 support>0 的类）。

    `conf[i][j]` = 真实类 i 被判成 j 的数量。macro-F1 作为选模指标：它不会被
    「其他」或淀粉粒这类大类单独拉高，稀有类判错会直接反映出来。
    """
    n = len(names)
    per = []
    macro_f1 = 0.0
    macro_p = 0.0
    macro_r = 0.0
    used = 0
    total = sum(sum(row) for row in conf)
    correct = sum(conf[i][i] for i in range(n))
    for i in range(n):
        tp = conf[i][i]
        support = sum(conf[i])
        pred = sum(conf[j][i] for j in range(n))
        p = tp / pred if pred else 0.0
        r = tp / support if support else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        per.append({"class": names[i], "support": support, "pred": pred,
                    "precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4)})
        if support:
            used += 1
            macro_p += p
            macro_r += r
            macro_f1 += f1
    return {
        "n": total,
        "accuracy": round(correct / total, 4) if total else 0.0,
        "macro_precision": round(macro_p / used, 4) if used else 0.0,
        "macro_recall": round(macro_r / used, 4) if used else 0.0,
        "macro_f1": round(macro_f1 / used, 4) if used else 0.0,
        "per_class": per,
    }


def _make_loader(root, tf, batch, workers, shuffle, seed=0):
    from torch.utils.data import DataLoader
    from torchvision.datasets import ImageFolder

    ds = ImageFolder(root, transform=tf)
    gen = None
    if shuffle:
        import torch
        gen = torch.Generator()
        gen.manual_seed(seed)
    return ds, DataLoader(ds, batch_size=batch, shuffle=shuffle, num_workers=workers,
                          pin_memory=True, persistent_workers=workers > 0,
                          drop_last=False, generator=gen)


def mixup_batch(x, y, alpha):
    """Mixup：返回 (x_mix, y_a, y_b, lam)；`alpha<=0` 时原样返回（lam=1）。

    显微 crop 的类别边界本身有模糊（石细胞↔纤维、分泌组织↔石细胞），mixup 让模型
    在两张图的线性插值上做软约束，比单纯加深网络更对症；`alpha=0.1` 时 lam 强烈双峰，
    等价于"约一半 batch 做接近 50/50 的混合"。
    """
    import torch

    if alpha <= 0:
        return x, y, y, 1.0
    lam = float(torch.distributions.Beta(alpha, alpha).sample())
    perm = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1.0 - lam) * x[perm], y, y[perm], lam


def _run_epoch(model, loader, criterion, device, optimizer=None, scaler=None,
               mixup_alpha=0.0):
    """跑一轮；给 optimizer 就是训练，否则只评估。返回 (loss, 混淆矩阵)。"""
    import torch

    training = optimizer is not None
    model.train(training)
    total_loss, n = 0.0, 0
    conf = None
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        ya = yb = y
        lam = 1.0
        if training and mixup_alpha > 0:
            x, ya, yb, lam = mixup_batch(x, y, mixup_alpha)
        with torch.set_grad_enabled(training):
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                out = model(x)
                loss = criterion(out, ya)
                if lam < 1.0:
                    loss = lam * loss + (1.0 - lam) * criterion(out, yb)
        if training:
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
        if conf is None:
            conf = torch.zeros((out.shape[1], out.shape[1]), dtype=torch.long)
        pred = out.argmax(1)
        idx = y.view(-1, 1) * out.shape[1] + pred.view(-1, 1)
        conf += torch.bincount(idx.view(-1), minlength=conf.numel()).view_as(conf).cpu()
        total_loss += float(loss.detach()) * len(y)
        n += len(y)
    return total_loss / max(n, 1), conf.tolist()


def _train(args):
    import time

    import torch
    import torch.nn as nn
    from torchvision import models

    t0 = time.time()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.benchmark = True
    device = torch.device(args.device if args.device != "auto"
                          else ("cuda" if torch.cuda.is_available() else "cpu"))
    counts, class_dirs = scan_splits(args.data, limit=args.limit)
    n_cls = len(class_dirs)
    train_counts = [counts["train"][d] for d in class_dirs]
    weights = class_weights(train_counts, args.class_weight)

    out_dir = os.path.join(args.out, args.name)
    os.makedirs(out_dir, exist_ok=True)
    train_tf, eval_tf, mean, std = build_transforms(args.imgsz, args.input_mode)
    tr_ds, tr_loader = _make_loader(os.path.join(args.data, "train"), train_tf,
                                    args.batch, args.workers, True, args.seed)
    va_ds, va_loader = _make_loader(os.path.join(args.data, "val"), eval_tf,
                                    args.batch, args.workers, False)
    te_ds, te_loader = _make_loader(os.path.join(args.data, "test"), eval_tf,
                                    args.batch, args.workers, False)
    for ds, split in ((tr_ds, "train"), (va_ds, "val"), (te_ds, "test")):
        if list(ds.classes) != class_dirs:
            raise ValueError("{} 的 ImageFolder 类别顺序与扫描结果不一致：{}".format(
                split, ds.classes))

    model = getattr(models, args.arch)(weights="DEFAULT" if args.pretrained else None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, n_cls)
    model.to(device)

    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32,
                                                        device=device),
                                    label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    print("[info] {} | 类别 {} | train {} / val {} / test {} | device={} | imgsz={} | mode={} | ls={} mixup={}".format(
        args.arch, n_cls, sum(train_counts),
        sum(counts["val"].values()), sum(counts["test"].values()), device, args.imgsz,
        args.input_mode, args.label_smoothing, args.mixup))
    print("[info] 类别权重（{}）：{}".format(
        args.class_weight,
        " ".join("{}={:.2f}".format(d.split("_", 1)[0], w) for d, w in zip(class_dirs, weights))))

    history, best = [], {"epoch": 0, "macro_f1": -1.0}
    for epoch in range(1, args.epochs + 1):
        tr_loss, _ = _run_epoch(model, tr_loader, criterion, device,
                                optimizer=optimizer, scaler=scaler,
                                mixup_alpha=args.mixup)
        va_loss, va_conf = _run_epoch(model, va_loader, criterion, device)
        m = macro_scores(va_conf, class_dirs)
        scheduler.step()
        history.append({"epoch": epoch, "train_loss": round(tr_loss, 4),
                        "val_loss": round(va_loss, 4), "val_acc": m["accuracy"],
                        "val_macro_f1": m["macro_f1"]})
        print("[{:03d}/{}] train_loss {:.4f} val_loss {:.4f} val_acc {:.4f} val_macroF1 {:.4f}"
              .format(epoch, args.epochs, tr_loss, va_loss, m["accuracy"], m["macro_f1"]))
        if m["macro_f1"] > best["macro_f1"]:
            best = {"epoch": epoch, "macro_f1": m["macro_f1"], "val_acc": m["accuracy"],
                    "val_loss": round(va_loss, 4)}
            torch.save({"state_dict": model.state_dict(), "arch": args.arch,
                        "imgsz": args.imgsz, "input_mode": args.input_mode,
                        "classes": class_dirs},
                       os.path.join(out_dir, "best.pt"))

    with open(os.path.join(out_dir, "history.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(history[0]))
        w.writeheader()
        w.writerows(history)

    ckpt = torch.load(os.path.join(out_dir, "best.pt"), map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    te_loss, te_conf = _run_epoch(model, te_loader, criterion, device)
    test_m = macro_scores(te_conf, class_dirs)

    report = {
        "version": 1,
        "name": args.name,
        "arch": args.arch,
        "pretrained": bool(args.pretrained),
        "imgsz": args.imgsz,
        "input_mode": args.input_mode,
        "epochs": args.epochs,
        "batch": args.batch,
        "lr": args.lr,
        "seed": args.seed,
        "class_weight": args.class_weight,
        "label_smoothing": args.label_smoothing,
        "mixup": args.mixup,
        "data": args.data,
        "class_dirs": class_dirs,
        "counts": counts,
        "class_weights": {d: round(w, 4) for d, w in zip(class_dirs, weights)},
        "best": best,
        "val": macro_scores(_run_epoch(model, va_loader, criterion, device)[1], class_dirs),
        "test": {"loss": round(te_loss, 4), **test_m},
        "confusion_matrix": {"labels": class_dirs, "matrix": te_conf},
        "elapsed_sec": round(time.time() - t0, 1),
        "warnings": [],
    }
    for d, c in zip(class_dirs, train_counts):
        if c < RARE_TRAIN_CRITICAL:
            report["warnings"].append("{} train 仅 {} crop（严重不足）".format(d, c))
        elif c < RARE_TRAIN_WARN:
            report["warnings"].append("{} train 仅 {} crop（偏少）".format(d, c))
    with open(os.path.join(out_dir, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    with open(os.path.join(out_dir, "model_meta.json"), "w", encoding="utf-8") as f:
        json.dump({"arch": args.arch, "imgsz": args.imgsz,
                   "input_mode": args.input_mode, "classes": class_dirs,
                   "mean": mean, "std": std, "checkpoint": "best.pt"}, f,
                  ensure_ascii=False, indent=2)

    print("[done] best epoch {} | val macroF1 {:.4f} / acc {:.4f}".format(
        best["epoch"], best["macro_f1"], best["val_acc"]))
    print("[done] test loss {:.4f} | acc {:.4f} | macro F1 {:.4f} | macro P {:.4f} / R {:.4f}"
          .format(te_loss, test_m["accuracy"], test_m["macro_f1"],
                  test_m["macro_precision"], test_m["macro_recall"]))
    print("[done] 产物目录：{}".format(out_dir))
    return 0


def _dry_run(args):
    counts, class_dirs = scan_splits(args.data, limit=args.limit)
    n_cls = len(class_dirs)
    print("[dry-run] 数据：{}（{} 类）".format(args.data, n_cls))
    print("  {:<24}{:>8}{:>8}{:>8}".format("class", "train", "val", "test"))
    for d in class_dirs:
        print("  {:<24}{:>8}{:>8}{:>8}".format(
            d, counts["train"][d], counts["val"][d], counts["test"][d]))
    print("  {:<24}{:>8}{:>8}{:>8}".format(
        "合计", sum(counts["train"].values()), sum(counts["val"].values()),
        sum(counts["test"].values())))
    print("[dry-run] 计划：arch={} imgsz={} input_mode={} epochs={} batch={} workers={} device={} "
          "label_smoothing={} mixup={}".format(
              args.arch, args.imgsz, args.input_mode, args.epochs, args.batch,
              args.workers, args.device, args.label_smoothing, args.mixup))
    print("[dry-run] 产物：{}".format(os.path.join(args.out, args.name)))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description="stage 2 特征分类器训练（crop → 14 类）")
    ap.add_argument("--data", default=DEFAULT_DATA, help="crop 数据集根（含 train/val/test）")
    ap.add_argument("--out", default=DEFAULT_OUT, help="运行产物根")
    ap.add_argument("--name", default=None,
                    help="运行名（默认 v1_<arch>_<imgsz>_<input_mode>）")
    ap.add_argument("--arch", default="resnet18", help="torchvision 分类网络名")
    ap.add_argument("--no-pretrained", dest="pretrained", action="store_false",
                    help="不加载 ImageNet 预训练（离线/对照用）")
    ap.add_argument("--imgsz", type=int, default=224)
    ap.add_argument("--input-mode", choices=INPUT_MODES, default="center",
                    help="center=短边缩放后中心裁（旧口径，会截断长条）；squash=整图压成方形；"
                         "letterbox=等比缩放 + 填充（保留全物体与长宽比）")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--device", default="auto", help="auto / cuda / cpu / 0")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--class-weight", choices=CLASS_WEIGHT_MODES, default="sqrt")
    ap.add_argument("--label-smoothing", type=float, default=0.0,
                    help="CE 标签平滑（噪声标签/类间边界模糊时用 0.1）")
    ap.add_argument("--mixup", type=float, default=0.0,
                    help="Mixup 的 Beta(α,α) 参数；0=关闭，0.1=轻度混合")
    ap.add_argument("--limit", type=int, default=0, help="每类最多取 N 张（冒烟用，0=全部）")
    ap.add_argument("--dry-run", action="store_true", help="只校验数据与参数，不训练")
    args = ap.parse_args(argv)
    if not args.name:
        args.name = "v1_{}_{}_{}".format(args.arch, args.imgsz, args.input_mode)

    args.data = args.data if os.path.isabs(args.data) else os.path.join(
        paths.PROJECT_ROOT, args.data)
    args.out = args.out if os.path.isabs(args.out) else os.path.join(
        paths.PROJECT_ROOT, args.out)
    if args.device.isdigit():
        args.device = "cuda:" + args.device
    try:
        return _dry_run(args) if args.dry_run else _train(args)
    except (FileNotFoundError, ValueError) as e:
        print("[error] {}".format(e), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
