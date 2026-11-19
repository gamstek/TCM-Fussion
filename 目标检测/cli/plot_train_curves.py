# -*- coding: utf-8 -*-
"""plot_train_curves.py — 把 YOLO 训练的 results.csv 画成 loss / 指标随 epoch 曲线（2026-11-05）

**为什么存在**：`cli/train_det.py` 只负责训练与评测，Ultralytics 自带的 `results.png`
把全部曲线挤在一张固定布局里，多个 run 无法叠加对比。本入口按"同一坐标轴叠多个 run"的
方式重绘：loss 六宫格（train/val × box/cls/dfl）+ 加权总损失两宫格 + 指标四宫格
（P/R/mAP50/mAP50-95），产物落 `results/detection/yolo_runs/curves/`，便于直接放进汇报或论文。

**加权总损失为什么直接相加**：Ultralytics 的 `v8DetectionLoss.get_assigned_targets_and_loss`
先执行 `loss[0] *= hyp.box`、`loss[1] *= hyp.cls`、`loss[2] *= hyp.dfl`，再把这组**已加权**的
分量写进 `results.csv`；反向传播的总损失就是三分量之和。故 `total = box_loss + cls_loss + dfl_loss`，
**不能再乘一遍权重**（否则是 7.5²/0.5²/1.5² 的平方加权）。YOLOv10 走 `E2EDetectLoss`，
CSV 分量是 one-to-many 与 one-to-one 两分支之和，求和口径仍是优化目标本身。权重只从
`args.yaml` 读出用于图注，不参与计算。

**数据兼容**：`v1_yolov10s_1024/results.csv` 第 72 行缺少换行，上一 epoch 的 lr 与下一 epoch
的行号粘在一起（`0.00017069272,3295.19,...`）。本入口按"上一行 lr 值回切"做容错解析，
不改写原 CSV；遇到修复会打印提示，方便上游补数据。

**分层**：入口脚本（`cli/` 不进 import 注入点）；只读 `results.csv` / `args.yaml`，不碰训练与
评测逻辑；路径全部取 `paths.DET_RESULTS`，不硬编码结果目录。

用法（项目根执行）:
  python 目标检测/cli/plot_train_curves.py
  python 目标检测/cli/plot_train_curves.py --runs v1_yolov8s_1024 v1_yolov10s_1024
  python 目标检测/cli/plot_train_curves.py --runs-dir <别的 run 根> --out-dir <输出目录>
"""
import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")                    # 无显示环境（容器/服务器）必须置于 pyplot 之前
import matplotlib.pyplot as plt
import numpy as np

import paths

NCOLS = 15                               # Ultralytics results.csv 的列数

# (run 目录名, 图例名, 颜色)；自定义 --runs 时图例名自动取目录名并循环配色
DEFAULT_RUNS = (
    ("v1_yolov8s_1024", "YOLOv8s 1024", "#1f77b4"),
    ("v1_yolov10s_1024", "YOLOv10s 1024", "#ff7f0e"),
    ("v1_yolov10s_1024_rareaug", "YOLOv10s 1024 + rare aug", "#2ca02c"),
)
DEFAULT_RUNS_DIR = os.path.join(paths.DET_RESULTS, "yolo_runs")

LOSS_PANELS = (
    ("train/box_loss", "Train box loss"),
    ("train/cls_loss", "Train cls loss"),
    ("train/dfl_loss", "Train dfl loss"),
    ("val/box_loss", "Val box loss"),
    ("val/cls_loss", "Val cls loss"),
    ("val/dfl_loss", "Val dfl loss"),
)
TOTAL_LOSS_PANELS = (
    ("train/total_loss", "Train weighted total loss"),
    ("val/total_loss", "Val weighted total loss"),
)
METRIC_PANELS = (
    ("metrics/precision(B)", "Precision"),
    ("metrics/recall(B)", "Recall"),
    ("metrics/mAP50(B)", "mAP@0.5"),
    ("metrics/mAP50-95(B)", "mAP@0.5:0.95"),
)


def _load_results(csv_path):
    """容错读取 Ultralytics results.csv，返回 {列名: ndarray}。

    正常行 15 列；若上一行末列（lr/pg2）与下一行 epoch 粘连，整行会变成 29 列，
    按"粘连前缀 == 上一行 lr/pg0"回切成两行。
    """
    with open(csv_path, encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    header = [c.strip() for c in lines[0].split(",")]
    rows, repaired = [], 0
    for ln in lines[1:]:
        parts = ln.split(",")
        if len(parts) == NCOLS:
            rows.append(parts)
        elif len(parts) == 2 * NCOLS - 1:
            head, tail = parts[:NCOLS], parts[NCOLS:]
            prev_lr, glued = head[12], head[-1]
            if not glued.startswith(prev_lr):
                raise ValueError("{}: 无法回切粘连行: {}".format(csv_path, glued))
            head[-1] = prev_lr
            rows.append(head)
            rows.append([glued[len(prev_lr):]] + tail)
            repaired += 1
        else:
            raise ValueError("{}: 字段数 {}，无法解析".format(csv_path, len(parts)))
    if repaired:
        print("  [修复] {}: 回切 {} 处缺换行的粘连行（原文件未改动）".format(
            os.path.basename(os.path.dirname(csv_path)), repaired))
    data = {k: [] for k in header}
    for r in rows:
        for k, v in zip(header, r):
            data[k].append(float(v))
    return {k: np.asarray(v) for k, v in data.items()}


def _smooth(y, weight=0.6):
    """YOLO 官方 results.png 同款指数滑动平均，用于在原始折线上叠一条趋势线。"""
    out = np.empty_like(y)
    last = y[0]
    for i, v in enumerate(y):
        last = last * weight + (1 - weight) * v
        out[i] = last
    return out


def _style_axis(ax, title, ylabel):
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.6)
    ax.tick_params(labelsize=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def _draw(panels, nrows, ncols, data, runs, figsize, fig_title, ylabel, out_path, dpi,
          mark_best=False):
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    for ax, (col, panel_title) in zip(axes.flat, panels):
        for name, label, color in runs:
            df = data[name]
            epoch, y = df["epoch"], df[col]
            ax.plot(epoch, y, color=color, lw=1.0, alpha=0.45)
            ax.plot(epoch, _smooth(y), color=color, lw=2.0, label=label)
            if mark_best:
                i = int(np.nanargmax(y))
                ax.scatter(epoch[i], y[i], s=22, color=color, zorder=5,
                           edgecolor="white", linewidth=0.5)
        _style_axis(ax, panel_title, ylabel)
        if mark_best:
            ax.set_ylim(bottom=0)

    handles, labels = axes.flat[0].get_legend_handles_labels()
    if mark_best:
        handles.append(plt.Line2D([], [], marker="o", ls="", color="#555555",
                                  markersize=6, markeredgecolor="white"))
        labels.append("best epoch")
    fig.legend(handles, labels, loc="upper center", ncol=len(handles), frameon=False,
               fontsize=10, bbox_to_anchor=(0.5, 0.965))
    fig.suptitle(fig_title, fontsize=14, fontweight="bold", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print("  [写入] {}".format(out_path))


def _read_gains(run_dir):
    """从 args.yaml 读 box/cls/dfl 增益，仅用于图注（CSV 分量已含增益，计算不参与）。"""
    gains = {"box": 7.5, "cls": 0.5, "dfl": 1.5}       # Ultralytics default.yaml
    args_path = os.path.join(run_dir, "args.yaml")
    if not os.path.exists(args_path):
        return gains
    import yaml
    with open(args_path, encoding="utf-8") as f:
        train_args = yaml.safe_load(f) or {}
    for key in gains:
        value = train_args.get(key)
        if isinstance(value, (int, float)):
            gains[key] = float(value)
    return gains


def main(argv=None):
    ap = argparse.ArgumentParser(description="绘制 YOLO 训练 loss / 指标曲线（多 run 叠加）")
    ap.add_argument("--runs", nargs="+", metavar="NAME",
                    help="run 目录名（默认三个基线 run）")
    ap.add_argument("--runs-dir", default=DEFAULT_RUNS_DIR,
                    help="run 根目录（默认 %(default)s）")
    ap.add_argument("--out-dir", default=None,
                    help="图片输出目录（默认 <runs-dir>/curves）")
    ap.add_argument("--dpi", type=int, default=160, help="输出分辨率（默认 160）")
    args = ap.parse_args(argv)

    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    if args.runs:
        runs = [(n, n, palette[i % len(palette)]) for i, n in enumerate(args.runs)]
    else:
        runs = list(DEFAULT_RUNS)
    out_dir = args.out_dir or os.path.join(args.runs_dir, "curves")

    missing = [os.path.join(args.runs_dir, n, "results.csv")
               for n, _, _ in runs
               if not os.path.exists(os.path.join(args.runs_dir, n, "results.csv"))]
    if missing:
        for p in missing:
            print("  [缺失] {}".format(p))
        return 1

    print("读取 results.csv：")
    data, gains = {}, {}
    for name, _, _ in runs:
        csv_path = os.path.join(args.runs_dir, name, "results.csv")
        df = _load_results(csv_path)
        epoch = df["epoch"]
        if not np.all(np.diff(epoch) == 1):
            print("  [错误] {}: epoch 不连续".format(name))
            return 1
        # CSV 三分量已乘过 hyp 增益，总损失直接相加（不可再乘权重）
        df["train/total_loss"] = (df["train/box_loss"] + df["train/cls_loss"]
                                  + df["train/dfl_loss"])
        df["val/total_loss"] = (df["val/box_loss"] + df["val/cls_loss"]
                                + df["val/dfl_loss"])
        gains[name] = _read_gains(os.path.join(args.runs_dir, name))
        data[name] = df
        print("  [ok] {}: {} epochs (1~{})，增益 box/cls/dfl = {:g}/{:g}/{:g}".format(
            name, len(epoch), int(epoch[-1]),
            gains[name]["box"], gains[name]["cls"], gains[name]["dfl"]))

    uniform = len({tuple(g.values()) for g in gains.values()}) == 1
    gain_note = ("YOLO Weighted Total Loss Curves ({:g}x box + {:g}x cls + {:g}x dfl)".format(
        gains[runs[0][0]]["box"], gains[runs[0][0]]["cls"], gains[runs[0][0]]["dfl"])
        if uniform else "YOLO Weighted Total Loss Curves (per-run gains)")

    os.makedirs(out_dir, exist_ok=True)
    _draw(LOSS_PANELS, 2, 3, data, runs, (15, 8.5),
          "YOLO Training Loss Curves", "Loss",
          os.path.join(out_dir, "yolo_training_loss_curves.png"), args.dpi)
    _draw(TOTAL_LOSS_PANELS, 1, 2, data, runs, (12, 4.8),
          gain_note, "Weighted loss",
          os.path.join(out_dir, "yolo_training_total_loss_curves.png"), args.dpi)
    _draw(METRIC_PANELS, 2, 2, data, runs, (11, 8.5),
          "YOLO Training Metric Curves", "Score",
          os.path.join(out_dir, "yolo_training_metric_curves.png"), args.dpi,
          mark_best=True)

    print("\n末轮 / 最优：")
    for name, label, _ in runs:
        df = data[name]
        i50 = int(np.nanargmax(df["metrics/mAP50(B)"]))
        i5095 = int(np.nanargmax(df["metrics/mAP50-95(B)"]))
        print("  {:<26s} last ep {:3d} | best mAP50 {:.4f}@{:<3d} | "
              "best mAP50-95 {:.4f}@{:<3d} | last P/R {:.4f}/{:.4f}".format(
                  label, int(df["epoch"][-1]),
                  df["metrics/mAP50(B)"][i50], int(df["epoch"][i50]),
                  df["metrics/mAP50-95(B)"][i5095], int(df["epoch"][i5095]),
                  df["metrics/precision(B)"][-1], df["metrics/recall(B)"][-1]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
