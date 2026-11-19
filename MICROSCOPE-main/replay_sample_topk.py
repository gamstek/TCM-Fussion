# -*- coding: utf-8 -*-
"""replay_sample_topk.py — 按**冻结的抽样清单**重放 ResNet50 五折集成 Top-K（不重新抽样）。

背景（两条流程必须分清，本脚本只做第一条）
    流程一（可完整追溯）：2026-11-03 的「目标检测图片抽样复测」——
        dataset/目标检测标注/<特征>/attributes.csv 给出每张图的 herb(药材) 与 feature_id，
        按 herb→模型类别（直接匹配）或（herb+特征）→复合类别（组合匹配）得到 expected_class，
        每类按 seed=20261103 抽 3 张（类内按 sha256 去重），共 63 类 189 张，
        对每张整图做**五折 softmax 平均**后取 Top-3。
    流程二（缺前提、做不了）：复现原训练 OOF 的 82.94%——
        需要原分类图像库 dataset/显微镜像（68 类 4672 张）及其原始折分配，本机不具备，
        且当前图像池与训练池高度重叠，任何重划分都会泄漏。故本脚本不涉及流程二。

本脚本刻意**不重新抽样**：直接读 samples.json，重放时逐张校验 sha256，确保用的是同一批图片。
输出可与冻结的 topk_per_image.csv 逐图对拍。

用法（.venv 无 torch，用带 torch 的解释器；见 README）
    python replay_sample_topk.py ^
        --samples  ..\\results\\resnet50_speed_20261103_110526\\samples.json ^
        --out-dir  ..\\results\\resnet50_speed_20261103_110526\\replay_check
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from predictv3_resnet50 import (  # noqa: E402
    build_inference_transform,
    load_checkpoint,
    make_resnet50,
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description="按冻结抽样清单重放五折集成 Top-K")
    ap.add_argument("--samples", type=Path, required=True, help="冻结的 samples.json")
    ap.add_argument(
        "--checkpoint-dir", type=Path, default=here / "v3_resnet50_full"
    )
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--batch-report", type=int, default=20, help="每 N 张打印一次进度")
    ap.add_argument(
        "--skip-hash",
        action="store_true",
        help="跳过 sha256 校验（默认必须校验）",
    )
    args = ap.parse_args()

    frozen = json.loads(args.samples.read_text(encoding="utf-8"))
    samples = frozen["samples"]
    print(f"冻结清单: {args.samples}")
    print(f"seed={frozen.get('seed')}  样本数={len(samples)}  声明缺失类={frozen.get('missing')}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}")

    # ── 1. 逐张校验：文件存在 + sha256 与冻结记录一致 ─────────────────────
    hash_ok = hash_bad = file_missing = 0
    hash_report: list[dict] = []
    t0 = time.perf_counter()
    for i, s in enumerate(samples, 1):
        p = Path(s["path"])
        if not p.is_file():
            file_missing += 1
            hash_report.append({"path": s["path"], "status": "missing"})
            continue
        if args.skip_hash:
            continue
        got = sha256_file(p)
        if got == s.get("sha256"):
            hash_ok += 1
        else:
            hash_bad += 1
            hash_report.append(
                {"path": s["path"], "status": "sha256_mismatch", "expect": s.get("sha256"), "got": got}
            )
        if i % 50 == 0:
            print(f"  校验 {i}/{len(samples)} ...", flush=True)
    print(
        f"[校验] sha256 一致={hash_ok} 不一致={hash_bad} 文件缺失={file_missing} "
        f"({time.perf_counter() - t0:.1f}s)",
        flush=True,
    )
    if hash_bad or file_missing:
        print("!! 存在不一致/缺失，仍继续推理，但结果不可视为同一批图片 !!", flush=True)

    # ── 2. 载入五折模型与推理预处理（与训练/推理脚本同一实现）─────────────
    first = load_checkpoint(args.checkpoint_dir / "fold1_best.pt")
    meta = first["meta"]
    class_names: list[str] = meta["class_names"]
    num_classes = len(class_names)
    img_size = int(meta.get("img_size", 320))
    auto_crop = bool(meta.get("auto_crop", True))
    k = int(meta.get("k_folds", 5))
    print(f"类别数={num_classes} img_size={img_size} auto_crop={auto_crop} 折数={k}")

    models = []
    for i in range(1, k + 1):
        ck = load_checkpoint(args.checkpoint_dir / f"fold{i}_best.pt")
        m = make_resnet50(num_classes)
        m.load_state_dict(ck["model_state"], strict=True)
        m.to(device).eval()
        models.append(m)
    print(f"已载入 {len(models)} 个折模型", flush=True)

    transform = build_inference_transform(img_size=img_size, auto_crop=auto_crop)

    # ── 3. 逐图（小批）推理，边跑边报 ────────────────────────────────────
    rows: list[dict] = []
    t0 = time.perf_counter()
    with torch.inference_mode():
        for i, s in enumerate(samples, 1):
            p = Path(s["path"])
            if not p.is_file():
                continue
            with Image.open(p) as im:
                x = transform(im).unsqueeze(0).to(device)
            probs = torch.stack(
                [torch.softmax(m(x), dim=1)[0].cpu() for m in models], dim=0
            ).mean(dim=0)
            vals, idx = torch.topk(probs, k=max(1, min(args.top_k, num_classes)))
            preds = [class_names[idx[r].item()] for r in range(len(idx))]
            expected = s["class"]
            correct_rank = preds.index(expected) + 1 if expected in preds else 0
            rows.append(
                {
                    "expected": expected,
                    "path": s["path"],
                    "herb": s.get("herb", ""),
                    "label_basis": s.get("label_basis", ""),
                    **{f"top{r + 1}": preds[r] for r in range(len(preds))},
                    **{f"p{r + 1}": float(vals[r].item()) for r in range(len(vals))},
                    "correct_rank": correct_rank,
                }
            )
            if i % args.batch_report == 0:
                done = sum(1 for r in rows if r["correct_rank"] == 1)
                print(
                    f"  {i}/{len(samples)} | 用时 {time.perf_counter() - t0:.0f}s | "
                    f"Top-1 累计 {done}/{len(rows)}",
                    flush=True,
                )

    n = len(rows)
    n_top1 = sum(1 for r in rows if r["correct_rank"] == 1)
    n_topk = sum(1 for r in rows if r["correct_rank"] > 0)

    # ── 4. 与冻结的 topk_per_image.csv 逐图对拍（若存在）──────────────────
    frozen_csv = args.samples.parent / "topk_per_image.csv"
    compare = {"available": False}
    if frozen_csv.is_file():
        with open(frozen_csv, encoding="utf-8-sig", newline="") as f:
            old = {r["path"]: r for r in csv.DictReader(f)}
        same = diff = 0
        diffs: list[dict] = []
        for r in rows:
            o = old.get(r["path"])
            if o is None:
                continue
            if o["top1"] == r["top1"] and o["top2"] == r["top2"] and o["top3"] == r["top3"]:
                same += 1
            else:
                diff += 1
                diffs.append(
                    {"path": r["path"], "frozen": [o["top1"], o["top2"], o["top3"]],
                     "replay": [r["top1"], r["top2"], r["top3"]]}
                )
        compare = {
            "available": True,
            "frozen_csv": str(frozen_csv),
            "compared": same + diff,
            "top3_identical": same,
            "top3_different": diff,
            "examples": diffs[:10],
        }
        print(f"[对拍] 与冻结 topk_per_image.csv: 相同={same} 不同={diff}", flush=True)

    # ── 5. 落盘 ─────────────────────────────────────────────────────────
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cols = ["expected", "path", "herb", "label_basis",
            "top1", "top2", "top3", "p1", "p2", "p3", "correct_rank"]
    with open(args.out_dir / "replay_topk_per_image.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: (round(r[c], 6) if isinstance(r.get(c), float) else r.get(c, "")) for c in cols})

    summary = {
        "flow": "流程一：抽样复测重放（非流程二 82.94% 复现）",
        "samples_json": str(args.samples),
        "seed": frozen.get("seed"),
        "n_images": n,
        "n_classes_covered": len({r["expected"] for r in rows}),
        "top1_correct": n_top1,
        "top1_accuracy": n_top1 / max(n, 1),
        "topk_covered": n_topk,
        f"top{args.top_k}_coverage": n_topk / max(n, 1),
        "top1_errors": [
            {"path": r["path"], "expected": r["expected"], "predicted": r["top1"],
             "top2": r["top2"], "top3": r["top3"], "correct_rank": r["correct_rank"]}
            for r in rows if r["correct_rank"] != 1
        ],
        "hash_check": {"ok": hash_ok, "mismatch": hash_bad, "missing": file_missing,
                       "details": hash_report[:20]},
        "compare_with_frozen": compare,
    }
    (args.out_dir / "replay_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n========== 抽样复测重放结果（流程一，非 82.94% 复现）==========")
    print(f"图片数     = {n}")
    print(f"Top-1 正确 = {n_top1}  ({n_top1 / max(n, 1):.4f})")
    print(f"Top-{args.top_k} 命中 = {n_topk}  ({n_topk / max(n, 1):.4f})")
    print(f"sha256 一致 = {hash_ok} / 不一致 = {hash_bad} / 缺失 = {file_missing}")
    print(f"输出目录: {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
