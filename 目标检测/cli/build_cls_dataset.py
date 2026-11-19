# -*- coding: utf-8 -*-
"""build_cls_dataset.py — 从 P0 检测训练集重切 stage 2 特征分类器的 crop 数据集。

**为什么存在**：stage 2 分类器的训练 crop 必须与推理同一口径（检测松框 30% + 长边上限），
盘上现成的 VLM crop 是实时产物、数量与框分布都不可控。本入口从人工框重切。

两种划分：
  `--split-policy rebalance`（默认）：test 固定 P0 test；P0 train+val 的组按**框数配额 + 药材分层**
  重划 train/val（稀有类 val 不再反超 train）。分类器训练用这一版。
  `--split-policy p0`：原样复用 P0 的组感知 train/val/test，用于与 stage 1 严格同划分的对照。

用法（项目根）：
    python -X utf8 目标检测/cli/build_cls_dataset.py --overwrite
    python -X utf8 目标检测/cli/build_cls_dataset.py --limit 40      # 冒烟：只切前 40 张源图
    python -X utf8 目标检测/cli/build_cls_dataset.py --split-policy p0 --overwrite  # 严格同划分对照

产物：`dataset/特征分类器_v1/`（crops/{train,val,test}/<id>_<feature_id>/ + manifest.csv +
classes.txt + dataset_report.json）。图像目录 gitignored；manifest/report 入库。

退出码：0 = 成功；2 = 源数据不一致 / 输出目录冲突（不产出半套）。
"""
import argparse
import os
import sys

import cls_dataset as cd
import paths


def main(argv=None):
    ap = argparse.ArgumentParser(description="stage 2 特征分类器 crop 数据集构建（复用 P0 划分）")
    ap.add_argument("--src", default=paths.DET_TRAIN_V1,
                    help="P0 检测训练集目录（须含 images/labels/split_report.json）")
    ap.add_argument("--out", default=paths.CLS_TRAIN_V1, help="输出目录")
    ap.add_argument("--margin-ratio", type=float, default=cd.DET_LOOSE_MARGIN_RATIO,
                    help="松框外扩（短边比例），默认与检测推理同一常量 0.30")
    ap.add_argument("--max-side", type=int, default=cd.DEFAULT_MAX_SIDE,
                    help="crop 长边上限（只降不升），默认 768")
    ap.add_argument("--jpeg-quality", type=int, default=cd.DEFAULT_JPEG_QUALITY, help="JPEG 质量")
    ap.add_argument("--split-policy", choices=cd.SPLIT_POLICIES, default="rebalance",
                    help="rebalance=默认，test 固定 P0 test、其余组按框数配额重划；p0=原样复用 P0 划分")
    ap.add_argument("--val-ratio", type=float, default=cd.DEFAULT_VAL_RATIO,
                    help="rebalance 时每类 val 框数占 pool 的比例（默认 0.15，且 val ≤ 50%%）")
    ap.add_argument("--seed", type=int, default=cd.DEFAULT_SEED, help="rebalance 的定序种子")
    ap.add_argument("--limit", type=int, default=0,
                    help="只处理前 N 张源图（冒烟/估算；产物标 partial，不能当训练集）")
    ap.add_argument("--overwrite", action="store_true", help="输出目录非空时整目录重建")
    args = ap.parse_args(argv)

    try:
        rep = cd.build_cls_dataset(args.src, args.out,
                                   margin_ratio=args.margin_ratio,
                                   max_side=args.max_side,
                                   jpeg_quality=args.jpeg_quality,
                                   split_policy=args.split_policy,
                                   val_ratio=args.val_ratio, seed=args.seed,
                                   limit=args.limit, overwrite=args.overwrite)
    except (OSError, ValueError) as e:
        print(f"[FAIL] {e}", file=sys.stderr)
        return 2

    sc = rep["split_counts"]
    print(f"[OK] split_policy={rep['params']['split_policy_code']}；crop "
          f"{rep['fingerprint']['n_crops']} 个："
          f"train {sc['train']['crops']} / val {sc['val']['crops']} / test {sc['test']['crops']}")
    rb = rep.get("rebalance")
    if rb:
        print(f"     rebalance：val {rb['val_groups']} 组 / train {rb['train_groups']} 组；"
              f"val 药材 {rb['val_herbs']}；目标未达 {rb['target_shortfall']}")
    bad = [a for a in rep["assertions"] if not a["ok"]]
    print(f"     断言 {len(rep['assertions']) - len(bad)}/{len(rep['assertions'])} 通过"
          + ("" if not bad else "；未过：" + ", ".join(a["name"] for a in bad)))
    for w in rep["warnings"]:
        print(f"     [warn:{w['code']}] {w['detail']}")
    print(f"     报告 {os.path.join(os.path.abspath(args.out), cd.REPORT_NAME)}")
    if rep.get("partial"):
        print("     [partial] 仅冒烟产物，缺源图未处理；不要直接拿去训练")
    return 0 if not bad else 3


if __name__ == "__main__":
    sys.exit(main())
