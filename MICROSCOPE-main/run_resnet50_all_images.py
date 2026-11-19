# -*- coding: utf-8 -*-
"""run_resnet50_all_images.py — 用现有 5 折 ResNet50 权重，把 dataset/目标检测标注 下的**全部图片**跑一遍。

定位（三条流程互不替代，本脚本是第三条）
    1) 复现原训练 82.94% 的 OOF —— 需要原图像库 dataset/显微镜像 及原始折分配，本机不具备，做不了。
    2) 抽样复测重放 —— 见 replay_sample_topk.py（只跑冻结的 189 张）。
    3) 本脚本：**把所有图片都跑一遍**。对每张整图做与训练一致的确定性预处理（关闭增强），
       用 5 折模型做 softmax 平均（与 predictv3_resnet50.py 的集成口径一致），
       输出逐图 Top-5 与概率、期望类别与排名、Top-K 命中率、混淆矩阵、错误清单。

期望类别（expected_class）的来源
    dataset/目标检测标注/<特征>/attributes.csv 给出每张图的 herb 与 feature_id：
        - 直接匹配：herb 与模型类别同名（如 甘草 → 甘草）；
        - 组合匹配：药材 + 特征对到复合类别（如下表），属**代理标签**，不等于人工复核真值。
    对不上的（药材不在模型 68 类内，或特征为「其他」等）记为 unmapped：
    仍然会跑预测，但不计入准确率/混淆矩阵。

用法（.venv 无 torch，用带 torch 的解释器；见 README）
    python run_resnet50_all_images.py --batch-size 32 --num-workers 4
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
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

# 「药材 + 特征(feature_id)」→ 复合模型类别（与抽样复测所用映射表一致）
COMBO_MAP: dict[str, tuple[str, str]] = {
    "党参石细胞": ("党参", "stone_cell"),
    "党参联结乳管": ("党参", "secretory_tissue"),
    "大黄导管": ("大黄", "vessel"),
    "大黄淀粉粒": ("大黄", "starch_grain"),
    "大黄草酸钙簇晶": ("大黄", "cluster_crystal"),
    "木香木纤维": ("木香", "fiber"),
    "板蓝根木纤维": ("板蓝根", "fiber"),
    "板蓝根石细胞": ("板蓝根", "stone_cell"),
    "黄柏晶纤维": ("黄柏", "fiber"),
}


def load_attributes(feature_dir: Path) -> dict[str, dict]:
    f = feature_dir / "attributes.csv"
    out: dict[str, dict] = {}
    if not f.exists():
        return out
    with open(f, encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            name = (row.get("image") or "").strip()
            if name:
                out.setdefault(name, row)
    return out


def expected_class(herb: str, feature_id: str, class_set: set[str]) -> str | None:
    if herb in class_set:
        return herb
    for cls, (h, fid) in COMBO_MAP.items():
        if herb == h and feature_id == fid:
            return cls
    return None


def collect_all_images(data_dir: Path, class_names: list[str]):
    class_set = set(class_names)
    rows: list[dict] = []
    unmapped: Counter = Counter()
    no_attr = 0
    for feat_dir in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        img_dir = feat_dir / "images"
        if not img_dir.is_dir():
            continue
        attrs = load_attributes(feat_dir)
        for img in sorted(
            (p for p in img_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS),
            key=lambda p: p.name,
        ):
            row = attrs.get(img.name)
            if row is None:
                no_attr += 1
                rows.append({"path": img, "feature": feat_dir.name, "herb": "", "feature_id": "",
                             "zh": "", "group_key": "", "expected": None})
                continue
            herb = (row.get("herb") or "").strip()
            fid = (row.get("feature_id") or "").strip()
            exp = expected_class(herb, fid, class_set)
            if exp is None:
                unmapped[(herb, (row.get("zh") or "").strip())] += 1
            rows.append({
                "path": img,
                "feature": feat_dir.name,
                "herb": herb,
                "feature_id": fid,
                "zh": (row.get("zh") or "").strip(),
                "group_key": (row.get("group_key") or "").strip(),
                "expected": exp,
            })
    return rows, unmapped, no_attr


class ImageDataset(Dataset):
    def __init__(self, paths: list[Path], transform):
        self.paths = paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        try:
            with Image.open(self.paths[i]) as im:
                return self.transform(im), i
        except Exception:
            return None, i


def _collate(batch):
    """容忍个别图片读取失败（返回 None）：不做张量堆叠，交回主进程处理。"""
    return [b[0] for b in batch], [b[1] for b in batch]


def main() -> int:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description="把 目标检测标注 下全部图片跑一遍")
    ap.add_argument("--data-dir", type=Path, default=here.parent / "dataset" / "目标检测标注")
    ap.add_argument("--checkpoint-dir", type=Path, default=here / "v3_resnet50_full")
    ap.add_argument("--out-dir", type=Path,
                    default=here.parent / "results" / f"resnet50_all_{time.strftime('%Y%m%d_%H%M%S')}")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--report-every", type=int, default=10, help="每 N 个 batch 打印一次进度")
    ap.add_argument("--copy-errors", action="store_true", help="把错误图片拷到输出目录便于目检")
    ap.add_argument("--limit", type=int, default=0, help=">0 时只跑前 N 张（自检用）")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}", flush=True)

    ck = load_checkpoint(args.checkpoint_dir / "fold1_best.pt")
    meta = ck["meta"]
    class_names: list[str] = meta["class_names"]
    num_classes = len(class_names)
    img_size = int(meta.get("img_size", 320))
    auto_crop = bool(meta.get("auto_crop", True))
    k = int(meta.get("k_folds", 5))
    print(f"类别数={num_classes} img_size={img_size} auto_crop={auto_crop} 折数={k}", flush=True)

    rows, unmapped, no_attr = collect_all_images(args.data_dir, class_names)
    if args.limit > 0:
        rows = rows[: args.limit]
        print(f"[--limit] 只跑前 {len(rows)} 张", flush=True)
    mapped = sum(1 for r in rows if r["expected"])
    print(f"全部图片={len(rows)}  有期望类别={mapped}  未映射={sum(unmapped.values())}  缺attributes={no_attr}",
          flush=True)

    models = []
    for i in range(1, k + 1):
        c = load_checkpoint(args.checkpoint_dir / f"fold{i}_best.pt")
        m = make_resnet50(num_classes)
        m.load_state_dict(c["model_state"], strict=True)
        m.to(device).eval()
        models.append(m)
    print(f"已载入 {len(models)} 个折模型（集成=softmax 平均）", flush=True)

    transform = build_inference_transform(img_size=img_size, auto_crop=auto_crop)
    loader = DataLoader(
        ImageDataset([r["path"] for r in rows], transform),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=False, collate_fn=_collate,
    )

    top_k = max(1, min(args.top_k, num_classes))
    probs_by_idx: dict[int, torch.Tensor] = {}
    t0 = time.perf_counter()
    n_batches = len(loader)
    with torch.inference_mode():
        for bi, (xs, idxs) in enumerate(loader, 1):
            keep = [j for j, v in enumerate(xs) if v is not None]
            if keep:
                xb = torch.stack([xs[j] for j in keep]).to(device)
                prob = torch.stack(
                    [torch.softmax(m(xb), dim=1).cpu() for m in models], dim=0
                ).mean(dim=0)
                for row, j in enumerate(keep):
                    probs_by_idx[int(idxs[j])] = prob[row]
            if bi % args.report_every == 0 or bi == n_batches:
                done = len(probs_by_idx)
                print(f"  进度 {done}/{len(rows)} | 用时 {time.perf_counter() - t0:.0f}s", flush=True)

    # worker 里偶发失败（多为 3840x2160 大图的进程内存压力）→ 退回单进程重试
    missing = [i for i in range(len(rows)) if i not in probs_by_idx]
    if missing:
        print(f"  重试 {len(missing)} 张（单进程，规避 worker 内存压力）", flush=True)
        with torch.inference_mode():
            for n, i in enumerate(missing, 1):
                try:
                    with Image.open(rows[i]["path"]) as im:
                        xb = transform(im).unsqueeze(0).to(device)
                    probs_by_idx[i] = torch.stack(
                        [torch.softmax(m(xb), dim=1).cpu() for m in models], dim=0
                    ).mean(dim=0)[0]
                except Exception as e:
                    print(f"    仍失败 {rows[i]['path']}: {type(e).__name__} {e}", flush=True)
                if n % 5 == 0 or n == len(missing):
                    print(f"    重试 {n}/{len(missing)}", flush=True)

    out_rows: list[dict] = []
    fail = 0
    failed_paths: list[str] = []
    for i, r in enumerate(rows):
        prob = probs_by_idx.get(i)
        if prob is None:
            fail += 1
            failed_paths.append(str(r["path"]))
            continue
        vals, idx = torch.topk(prob, k=top_k)
        preds = [class_names[idx[t].item()] for t in range(top_k)]
        pvals = [float(vals[t].item()) for t in range(top_k)]
        exp = r["expected"]
        rank = ""
        if exp is not None and exp in preds:
            rank = preds.index(exp) + 1
        out_rows.append({
            **{kk: r[kk] for kk in ("feature", "herb", "feature_id", "zh", "group_key")},
            "image": r["path"].name,
            "expected_class": exp if exp is not None else "",
            "mapped": int(exp is not None),
            **{f"top{t + 1}": preds[t] for t in range(top_k)},
            **{f"p{t + 1}": round(pvals[t], 6) for t in range(top_k)},
            "expected_rank": rank,
            "top1_hit": int(rank == 1) if exp is not None else "",
            "top3_hit": int(isinstance(rank, int) and rank <= 3) if exp is not None else "",
            "top5_hit": int(isinstance(rank, int)) if exp is not None else "",
            "path": str(r["path"]),
        })

    m_rows = [r for r in out_rows if r["mapped"]]
    n_m = len(m_rows)
    n_top1 = sum(r["top1_hit"] for r in m_rows)
    n_top3 = sum(r["top3_hit"] for r in m_rows)
    n_top5 = sum(r["top5_hit"] for r in m_rows)

    cm = [[0] * num_classes for _ in range(num_classes)]
    ci = {c: i for i, c in enumerate(class_names)}
    for r in m_rows:
        cm[ci[r["expected_class"]]][ci[r["top1"]]] += 1

    per_class = []
    for i, c in enumerate(class_names):
        sup = sum(cm[i])
        if sup == 0:
            continue
        cr = [r for r in m_rows if r["expected_class"] == c]
        per_class.append({
            "class": c, "support": sup,
            "top1_hit": sum(r["top1_hit"] for r in cr), "top1_acc": sum(r["top1_hit"] for r in cr) / sup,
            "top3_hit": sum(r["top3_hit"] for r in cr), "top3_cov": sum(r["top3_hit"] for r in cr) / sup,
            "top5_hit": sum(r["top5_hit"] for r in cr),
        })
    top_conf = sorted(
        ((cm[i][j], class_names[i], class_names[j], sum(cm[i]))
         for i in range(num_classes) for j in range(num_classes) if i != j and cm[i][j] > 0),
        key=lambda x: -x[0],
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cols = ["image", "feature", "herb", "feature_id", "zh", "group_key", "expected_class", "mapped",
            "top1", "p1", "top2", "p2", "top3", "p3", "top4", "p4", "top5", "p5",
            "expected_rank", "top1_hit", "top3_hit", "top5_hit", "path"]
    with open(args.out_dir / "predictions_all.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in out_rows:
            w.writerow({c: r.get(c, "") for c in cols})

    with open(args.out_dir / "errors_top1.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in m_rows:
            if r["top1_hit"] == 0:
                w.writerow({c: r.get(c, "") for c in cols})

    with open(args.out_dir / "uncovered_top3.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in m_rows:
            if r["top3_hit"] == 0:
                w.writerow({c: r.get(c, "") for c in cols})

    with open(args.out_dir / "confusion_matrix.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["true\\pred"] + class_names)
        for i, c in enumerate(class_names):
            w.writerow([c] + cm[i])

    with open(args.out_dir / "per_class_topk.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_class[0].keys()) if per_class else ["class"])
        w.writeheader()
        for r in per_class:
            w.writerow({k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()})

    with open(args.out_dir / "unmapped_groups.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["herb", "feature_zh", "count"])
        for (h, z), n in unmapped.most_common():
            w.writerow([h, z, n])

    errors = [r for r in m_rows if r["top1_hit"] == 0]
    uncovered = [r for r in m_rows if r["top3_hit"] == 0]
    summary = {
        "flow": "第三条：把全部图片跑一遍（5 折集成）；既非 82.94% 复现，也非 189 张抽样复测",
        "data_dir": str(args.data_dir),
        "checkpoint_dir": str(args.checkpoint_dir),
        "n_images_total": len(rows),
        "n_predict_failed": fail,
        "failed_paths": failed_paths[:20],
        "n_mapped": n_m,
        "n_unmapped": sum(unmapped.values()),
        "n_no_attributes": no_attr,
        "top1_hit": n_top1, "top1_accuracy": n_top1 / max(n_m, 1),
        "top3_hit": n_top3, "top3_coverage": n_top3 / max(n_m, 1),
        "top5_hit": n_top5, "top5_coverage": n_top5 / max(n_m, 1),
        "n_top1_errors": len(errors),
        "n_top3_uncovered": len(uncovered),
        "top_confusions": [
            {"true": a, "pred": b, "count": c, "true_support": s} for c, a, b, s in top_conf[:30]
        ],
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.copy_errors:
        for r in errors:
            d = args.out_dir / "错误图片" / f"{r['expected_class']}__误判为__{r['top1']}"
            d.mkdir(parents=True, exist_ok=True)
            try:
                shutil.copy2(r["path"], d / f"{r['feature']}_{r['image']}")
            except Exception as e:
                print(f"  复制失败 {r['path']}: {e}", flush=True)

    print("\n========== 全部图片跑一遍（5 折集成）==========", flush=True)
    print(f"总图片     = {len(rows)}（预测失败 {fail}）")
    print(f"有期望类别 = {n_m}    未映射 = {sum(unmapped.values())}    缺 attributes = {no_attr}")
    print(f"Top-1 命中 = {n_top1}/{n_m} ({n_top1 / max(n_m, 1):.4f})")
    print(f"Top-3 命中 = {n_top3}/{n_m} ({n_top3 / max(n_m, 1):.4f})")
    print(f"Top-5 命中 = {n_top5}/{n_m} ({n_top5 / max(n_m, 1):.4f})")
    print(f"Top-1 错误 = {len(errors)}   Top-3 未覆盖 = {len(uncovered)}")
    print("最严重误判 Top-10:")
    for c, a, b, s in top_conf[:10]:
        print(f"  {a} -> {b}: {c}  (真实类共 {s})")
    print(f"\n输出目录: {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
