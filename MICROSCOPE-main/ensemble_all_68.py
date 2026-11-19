# -*- coding: utf-8 -*-
"""全量 68 类 · 5 折集成：找出"集成层面仍然分不开"的药材对。

与 OOF 脚本的区别
    OOF 版（predictions_68class.csv）每张图只取"未训练它的那一折"，得到 82.94% 的诚实准确率。
    本脚本做的是**5 折 softmax 平均**：训练集内图片会被 4 个"见过它"的模型抬高分数，
    因此绝大多数样本集成后极度自信；这里要找的是**连 5 折平均都拉不开**的少数样本，
    并同时记录"有几折把正确答案判为第一"，用来区分：
        - 真·多折摇摆：多折之间互相矛盾（≥2 折判错，或胜折数接近半数）
        - 单折捷径：只有 1 折例外，其余 4 折都很确信（集成后通常会拉开）

用法
    python ensemble_all_68.py                 # 全量
    python ensemble_all_68.py --limit 200      # 自检
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
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = load_checkpoint(args.checkpoint_dir / "fold1_best.pt")
    meta = ck["meta"]
    class_names = meta["class_names"]
    num_classes = len(class_names)
    img_size = int(meta.get("img_size", 320))
    auto_crop = bool(meta.get("auto_crop", True))
    k = int(meta.get("k_folds", 5))
    print(f"device={device} 类别={num_classes} img={img_size} folds={k}", flush=True)

    items = []
    for cls in class_names:
        d = args.data_dir / cls
        files = sorted((p for p in d.iterdir() if p.suffix.lower() in IMAGE_EXTS),
                       key=lambda p: p.name)
        for p in files:
            items.append((p, cls))
    if args.limit > 0:
        items = items[: args.limit]
    print(f"图片数={len(items)}", flush=True)

    models = []
    for i in range(1, k + 1):
        c = load_checkpoint(args.checkpoint_dir / f"fold{i}_best.pt")
        m = make_resnet50(num_classes)
        m.load_state_dict(c["model_state"], strict=True)
        m.to(device).eval()
        models.append(m)

    tf = build_inference_transform(img_size=img_size, auto_crop=auto_crop)
    loader = DataLoader(ImageDataset(items, tf), batch_size=args.batch_size,
                        shuffle=False, num_workers=args.num_workers, collate_fn=_collate)

    ci = {c: i for i, c in enumerate(class_names)}
    rows = []
    t0 = time.perf_counter()
    nb = len(loader)
    with torch.inference_mode():
        for bi, (xs, idxs) in enumerate(loader, 1):
            keep = [j for j, v in enumerate(xs) if v is not None]
            if keep:
                xb = torch.stack([xs[j] for j in keep]).to(device)
                per = torch.stack([torch.softmax(m(xb), dim=1).cpu() for m in models], 0)
                ens = per.mean(0)
                for row, j in enumerate(keep):
                    i = int(idxs[j])
                    path, cls = items[i]
                    gi = ci[cls]
                    prob = ens[row]
                    v, ix = torch.topk(prob, 5)
                    fold_arg = [int(per[f][row].argmax()) for f in range(k)]
                    rows.append({
                        "true_class": cls,
                        "image": path.name,
                        "top1": class_names[ix[0].item()],
                        "p1": float(v[0]),
                        "top2": class_names[ix[1].item()],
                        "p2": float(v[1]),
                        "top3": class_names[ix[2].item()],
                        "p3": float(v[2]),
                        "true_prob": float(prob[gi]),
                        "true_rank": int((prob > prob[gi]).sum()) + 1,
                        "ens_gap": float(v[0]) - float(prob[gi]),
                        "folds_true_first": sum(1 for a in fold_arg if a == gi),
                        "fold_argmax": "|".join(class_names[a] for a in fold_arg),
                        "path": str(path),
                    })
            if bi % args.report_every == 0 or bi == nb:
                print(f"  {len(rows)}/{len(items)} | {time.perf_counter()-t0:.0f}s", flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cols = ["image", "true_class", "top1", "p1", "top2", "p2", "top3", "p3",
            "true_prob", "true_rank", "ens_gap", "folds_true_first", "fold_argmax", "path"]
    with open(args.out_dir / "ensemble_68class.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: (round(r[c], 6) if isinstance(r[c], float) else r[c]) for c in cols})

    n = len(rows)
    n1 = sum(1 for r in rows if r["true_rank"] == 1)
    print(f"\n集成 Top1 正确 = {n1}/{n} ({n1/max(n,1):.4f})", flush=True)
    print("集成差分布（true_rank>=2 的样本）:")
    r23 = [r for r in rows if r["true_rank"] >= 2]
    print(f"  rank>=2 共 {len(r23)} 张")
    for lo, hi in [(0, .05), (.05, .1), (.1, .2), (.2, .3), (.3, .5), (.5, .8), (.8, 1.01)]:
        print(f"    [{lo:.2f},{hi:.2f}) : {sum(1 for r in r23 if lo <= r['ens_gap'] < hi)}")

    for thr in (0.10, 0.20, 0.30):
        sel = [r for r in r23 if r["ens_gap"] <= thr]
        c = Counter(f"{r['true_class']} -> {r['top1']}" for r in sel)
        multi = sorted(((a, b) for a, b in c.items() if b >= 2), key=lambda x: -x[1])
        print(f"\n[集成差<= {thr}] {len(sel)} 张 / {len(c)} 对 / >=2 张的对 {len(multi)} 个")
        for a, b in multi:
            print(f"    {a}: {b}")

    summary = {
        "n_images": n, "ensemble_top1_correct": n1,
        "ensemble_top1_accuracy": n1 / max(n, 1),
        "n_rank_ge2": len(r23),
        "buckets": {f"{lo}-{hi}": sum(1 for r in r23 if lo <= r["ens_gap"] < hi)
                    for lo, hi in [(0, .05), (.05, .1), (.1, .2), (.2, .3), (.3, .5), (.5, .8), (.8, 1.01)]},
    }
    (args.out_dir / "ensemble_68class_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n输出: {args.out_dir / 'ensemble_68class.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
