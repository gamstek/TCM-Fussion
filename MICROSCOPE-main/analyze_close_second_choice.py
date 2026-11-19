# -*- coding: utf-8 -*-
"""analyze_close_second_choice.py — 找出「正确答案掉到第 2/3 名、且与第一名分数接近」的药材。

数据与标签
    用原始 68 类数据集 dataset/中药分类数据/显微镜像/<药材名>/<图片>，
    真实类别 = 目录名（与 v3_resnet50_full/fold*_meta.json 的 class_names 一一对应，
    各类张数与 cv_summary.json 的 support 一致）。这里不再用检测标注的代理标签。

预测口径
    整图确定性预处理（与训练验证完全一致，无随机增强）→ 5 折模型 softmax 平均
    → 68 类概率降序 → 记录真实类别排名与其概率。

本脚本关心的问题
    不是「错了多少」，而是「哪些药材属于**边界型混淆**」：
        真实类别排第 2 或第 3，且 p(第一名) - p(真实) 很小（默认 ≤0.10），
        并且同一药材有多张图如此 → 说明该药材的特征本身与另一类高度重合，不是偶发噪声。

用法
    python analyze_close_second_choice.py                 # 全量 4672 张
    python analyze_close_second_choice.py --limit 200     # 自检
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from predictv3_resnet50 import (  # noqa: E402
    build_inference_transform,
    load_checkpoint,
    make_resnet50,
)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


class ImageDataset(Dataset):
    def __init__(self, items, transform):
        self.items = items
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        try:
            with Image.open(self.items[i][0]) as im:
                return self.transform(im), i
        except Exception:
            return None, i


def _collate(batch):
    return [b[0] for b in batch], [b[1] for b in batch]


def collect(root: Path, class_names: list[str]):
    items = []
    extra = []
    for cls in class_names:
        d = root / cls
        if not d.is_dir():
            extra.append((cls, 0))
            continue
        files = sorted(
            (p for p in d.iterdir() if p.suffix.lower() in IMAGE_EXTS), key=lambda p: p.name
        )
        if not files:
            extra.append((cls, 0))
        for p in files:
            items.append((p, cls))
    return items, extra


def main() -> int:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path,
                    default=here.parent / "dataset" / "中药分类数据" / "显微镜像")
    ap.add_argument("--checkpoint-dir", type=Path, default=here / "v3_resnet50_full")
    ap.add_argument("--out-dir", type=Path,
                    default=here.parent / "results" / "resnet50_class68_analysis")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--report-every", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--margin", type=float, default=0.10, help="与第一名的概率差阈值")
    ap.add_argument(
        "--oof",
        action="store_true",
        help="折外模式：按原训练脚本的分层 5 折（seed=42），每张图只由未训练它的那折模型预测；不加则为 5 折集成",
    )
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = load_checkpoint(args.checkpoint_dir / "fold1_best.pt")
    meta = ck["meta"]
    class_names = meta["class_names"]
    num_classes = len(class_names)
    img_size = int(meta.get("img_size", 320))
    auto_crop = bool(meta.get("auto_crop", True))
    k = int(meta.get("k_folds", 5))
    print(f"device={device} 类别={num_classes} img={img_size} auto_crop={auto_crop} folds={k}", flush=True)

    items, missing = collect(args.data_dir, class_names)
    print(f"数据集={args.data_dir}")
    print(f"图片数={len(items)}  空类别={[c for c, n in missing if n == 0]}", flush=True)
    if args.limit > 0:
        items = items[: args.limit]
        print(f"[--limit] 只跑 {len(items)} 张", flush=True)

    models = {}
    for i in range(1, k + 1):
        c = load_checkpoint(args.checkpoint_dir / f"fold{i}_best.pt")
        m = make_resnet50(num_classes)
        m.load_state_dict(c["model_state"], strict=True)
        m.to(device).eval()
        models[i] = m

    tf = build_inference_transform(img_size=img_size, auto_crop=auto_crop)

    if args.oof:
        # 复刻训练脚本的划分：ImageFolder 顺序（类名有序 + 类内文件名有序）→ 分层 k 折
        from trainv3_resnet50_full import stratified_k_fold_val_indices

        ci_map = {c: i for i, c in enumerate(class_names)}
        targets = [ci_map[cls] for _, cls in items]
        val_folds = stratified_k_fold_val_indices(targets, k, args.seed)
        pred_sets = [(fi + 1, sorted(ids)) for fi, ids in enumerate(val_folds)]
        print(f"[OOF] seed={args.seed} 各折验证数: {[len(ids) for _, ids in pred_sets]}", flush=True)
    else:
        pred_sets = [(1, list(range(len(items))))]

    probs: dict[int, torch.Tensor] = {}
    t0 = time.perf_counter()
    for fold_no, idxs in pred_sets:
        model = models[fold_no]
        sub = [items[i] for i in idxs]
        loader = DataLoader(
            ImageDataset(sub, tf), batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, collate_fn=_collate,
        )
        nb = len(loader)
        done = 0
        with torch.inference_mode():
            for bi, (xs, ridx) in enumerate(loader, 1):
                keep = [j for j, v in enumerate(xs) if v is not None]
                if keep:
                    xb = torch.stack([xs[j] for j in keep]).to(device)
                    pr = torch.softmax(model(xb), dim=1).cpu()
                    for row, j in enumerate(keep):
                        probs[idxs[int(ridx[j])]] = pr[row]
                done += len(xs)
                if bi % args.report_every == 0 or bi == nb:
                    print(f"  fold{fold_no} {done}/{len(idxs)} | 累计 {len(probs)}/{len(items)} "
                          f"| {time.perf_counter() - t0:.0f}s", flush=True)
        # worker 偶发失败 → 单进程重试
        miss = [i for i in idxs if i not in probs]
        if miss:
            print(f"  fold{fold_no} 单进程重试 {len(miss)} 张", flush=True)
            with torch.inference_mode():
                for i in miss:
                    try:
                        with Image.open(items[i][0]) as im:
                            xb = tf(im).unsqueeze(0).to(device)
                        probs[i] = torch.softmax(model(xb), dim=1).cpu()[0]
                    except Exception as e:
                        print(f"    失败 {items[i][0]}: {type(e).__name__}", flush=True)

    ci = {c: i for i, c in enumerate(class_names)}
    rows = []
    for i, (p, cls) in enumerate(items):
        pr = probs.get(i)
        if pr is None:
            continue
        gi = ci[cls]
        v, ix = torch.topk(pr, 5)
        rank = int((pr > pr[gi]).sum().item()) + 1
        rows.append({
            "image": p.name, "true_class": cls,
            "top1": class_names[ix[0].item()], "p1": float(v[0]),
            "top2": class_names[ix[1].item()], "p2": float(v[1]),
            "top3": class_names[ix[2].item()], "p3": float(v[2]),
            "top4": class_names[ix[3].item()], "p4": float(v[3]),
            "top5": class_names[ix[4].item()], "p5": float(v[4]),
            "true_rank": rank,
            "true_prob": float(pr[gi]),
            "margin": float(v[0]) - float(pr[gi]),
            "path": str(p),
        })

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cols = ["image", "true_class", "top1", "p1", "top2", "p2", "top3", "p3", "top4", "p4",
            "top5", "p5", "true_rank", "true_prob", "margin", "path"]
    with open(args.out_dir / "predictions_68class.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: (round(r[c], 6) if isinstance(r[c], float) else r[c]) for c in cols})

    n = len(rows)
    n1 = sum(1 for r in rows if r["true_rank"] == 1)
    print(f"\nTotal={n}  Top1={n1} ({n1 / max(n, 1):.4f})", flush=True)

    # ── 边界型混淆：真实类别第 2/3 名，且与第一名接近 ──────────────────────
    close = [r for r in rows if r["true_rank"] in (2, 3) and r["margin"] <= args.margin]
    by_cls = defaultdict(list)
    for r in close:
        by_cls[r["true_class"]].append(r)

    ranked = sorted(by_cls.items(), key=lambda kv: (-len(kv[1]), min(x["margin"] for x in kv[1])))

    for thr in (0.03, 0.05, 0.10, 0.20):
        sel = [r for r in rows if r["true_rank"] in (2, 3) and r["margin"] <= thr]
        c = Counter(r["true_class"] for r in sel)
        multi = {k: v for k, v in c.items() if v >= 2}
        print(f"\n[阈值 margin<={thr}] 边界型(第2/3名)共 {len(sel)} 张，涉及 {len(c)} 个药材，"
              f"其中≥2张的药材 {len(multi)} 个: {sorted(multi.items(), key=lambda x: -x[1])}", flush=True)

    # 最容易与之混淆的对手类
    rival = Counter()
    for r in close:
        for t in ("top1", "top2", "top3"):
            if r[t] != r["true_class"] and r[t] in (
                r["top1"], r["top2"], r["top3"]
            ):
                rival[(r["true_class"], r[t])] += 1
                break

    summary = {
        "data_dir": str(args.data_dir),
        "n_images": n,
        "top1_correct": n1,
        "top1_accuracy": n1 / max(n, 1),
        "margin_threshold": args.margin,
        "close_rank23": len(close),
        "by_class": [
            {
                "class": cls,
                "count": len(v),
                "images": [
                    {"image": x["image"], "true_rank": x["true_rank"],
                     "top1": x["top1"], "p1": round(x["p1"], 6),
                     "true_prob": round(x["true_prob"], 6), "margin": round(x["margin"], 6),
                     "path": x["path"]}
                    for x in sorted(v, key=lambda y: y["margin"])
                ],
            }
            for cls, v in ranked
        ],
        "rival_pairs": [{"true": a, "rival": b, "count": c} for (a, b), c in rival.most_common()],
    }
    (args.out_dir / "close_margin_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== 药材（真实类）| 边界型张数 | 最接近的一次 (margin) ===")
    for cls, v in ranked:
        best = min(v, key=lambda y: y["margin"])
        print(f"{cls:<12} {len(v):>3}  margin={best['margin']:.4f} "
              f"({best['p1']:.3f} vs {best['true_prob']:.3f}, 对 {best['top1']})", flush=True)
    print(f"\n输出: {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
