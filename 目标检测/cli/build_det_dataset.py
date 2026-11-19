# -*- coding: utf-8 -*-
"""build_det_dataset.py — 入口：把 14 个单类 YOLO 导出集合并成多类训练集（P0）

**为什么存在**：`export_det_dataset.py` 只按类逐目录导出；本入口调用 `lib/det_dataset.py`
做内容 md5 合并、group_key 组感知划分、整味药材 hold-out、稀有类 train-only 重采样、
标签口径审计与数据指纹，产物落 `dataset/检测训练集_v1/`（gitignored 运行产物）。

用法（项目根执行）:
  python 目标检测\\cli\\build_det_dataset.py --dry-run          # 只算不落盘，先看划分数字
  python 目标检测\\cli\\build_det_dataset.py                    # 落盘
  python 目标检测\\cli\\build_det_dataset.py --overwrite        # 覆盖重建（冻结后不要随便跑）
  python 目标检测\\cli\\build_det_dataset.py --holdout-herbs 常山,黄柏
  python 目标检测\\cli\\build_det_dataset.py --val-ratio 0.15 --test-ratio 0.15

产物:
  images/{train,val,test}/     唯一内容图（同一 md5 只出现一次）
  labels/{train,val,test}/     YOLO 标签（类别已连续重编 0..13）
  train_list.txt               训练清单；稀有类按重复行实现 train-only 重采样
  classes.txt / data.yaml      14 个训练类（13 目标 + 其他）
  attributes.csv               每框一行（含 md5/split/herb/group_key 溯源）
  split_report.json            划分、配额、审计、断言
  dataset_fingerprint.json     全文件 md5 与 root_hash
"""
import argparse
import json
import os
import sys

import det_dataset
import ls_common


def main(argv=None):
    ap = argparse.ArgumentParser(description="合并单类导出集为多类 YOLO 训练集（P0）")
    ap.add_argument("--src-dir", default=os.path.join(ls_common.PROJECT_ROOT, "dataset", "目标检测标注"))
    ap.add_argument("--out-dir", default=os.path.join(ls_common.PROJECT_ROOT, "dataset", "检测训练集_v1"))
    ap.add_argument("--config", default=ls_common.CONFIG_PATH)
    ap.add_argument("--taxonomy", default=os.path.join(ls_common.PROJECT_ROOT, "feature_taxonomy.json"))
    ap.add_argument("--manifest", default=ls_common.MANIFEST_PATH)
    ap.add_argument("--gold-set", default=os.path.join(ls_common.DET_RESULTS, "det_eval", "gold_set.json"))
    ap.add_argument("--seed", type=int, default=det_dataset.DEFAULT_SEED)
    ap.add_argument("--val-ratio", type=float, default=det_dataset.DEFAULT_VAL_RATIO)
    ap.add_argument("--test-ratio", type=float, default=det_dataset.DEFAULT_TEST_RATIO)
    ap.add_argument("--holdout-herbs", default="", help="逗号分隔；默认自动选 1~2 味")
    ap.add_argument("--holdout-max", type=int, default=det_dataset.DEFAULT_HOLDOUT_MAX)
    ap.add_argument("--holdout-min-images", type=int,
                    default=det_dataset.DEFAULT_HOLDOUT_MIN_IMAGES)
    ap.add_argument("--holdout-min-boxes", type=int, default=det_dataset.DEFAULT_HOLDOUT_MIN_BOXES)
    ap.add_argument("--rare-threshold", type=int, default=det_dataset.DEFAULT_RARE_THRESHOLD)
    ap.add_argument("--repeat-max", type=int, default=det_dataset.DEFAULT_MAX_REPEAT)
    ap.add_argument("--overwrite", action="store_true", help="覆盖已存在的输出目录")
    ap.add_argument("--dry-run", action="store_true", help="只计算与断言，不落盘")
    args = ap.parse_args(argv)

    with open(args.config, encoding="utf-8") as f:
        cfg = json.load(f)
    holdout = [x.strip() for x in args.holdout_herbs.split(",") if x.strip()]

    try:
        report = det_dataset.build(
            args.src_dir, args.out_dir, cfg, taxonomy_path=args.taxonomy,
            seed=args.seed, val_ratio=args.val_ratio, test_ratio=args.test_ratio,
            holdout_herbs=holdout or None, holdout_max=args.holdout_max,
            holdout_min_images=args.holdout_min_images,
            holdout_min_boxes=args.holdout_min_boxes,
            rare_threshold=args.rare_threshold, max_repeat=args.repeat_max,
            manifest_path=args.manifest, gold_set_path=args.gold_set,
            overwrite=args.overwrite, dry_run=args.dry_run)
    except det_dataset.DatasetError as e:
        print(f"[FAIL] {e}", file=sys.stderr)
        return 2

    merge = report["merge"]
    print("\n" + "=" * 72)
    print(f"唯一内容 {merge['n_unique']}（输入文件 {merge['n_input_files']}，"
          f"重复文件 {merge['n_duplicate_files']}，跨目录重复组 "
          f"{merge['n_cross_dir_duplicate_groups']}）")
    for s in det_dataset.SPLITS:
        c = report["split_counts"][s]
        print(f"  {s:<5} 图 {c['images']:>5} / 框 {c['boxes']:>5}"
              f"（跨药材 {c['cross_herb_images']}）")
    ge = report["gold_eval_count"]
    print(f"  gold_eval 预留（不进任何 split）：图 {ge['images']} / 框 {ge['boxes']}")
    print(f"  与 gold 同组但仍参与划分的图：{report['gold_overlap']['same_group_active_records']}"
          "（近似泄漏风险，见 split_report.json）")
    print(f"  hold-out 药材：{report['holdout']['herbs'] or '（无）'}")
    print(f"  重采样：{report['oversample'] or '（无稀有类）'}")
    unmet = report["split_policy"]["unmet"]
    box_unmet = report["split_policy"].get("box_unmet") or {}
    print(f"  未达配额类：{len(unmet)}" + (f" → {sorted(unmet)}" if unmet else "")
          + (f"；框数未达 20 的类-分集：{sorted(box_unmet)}" if box_unmet else ""))
    warn = report.get("rare_class_warnings") or {}
    if warn:
        print("  train 图数 <20（建议 5 折 CV/强增强）："
              + ", ".join(f"{zh}({v['train_images']})" for zh, v in warn.items()))
    print("  逐类图数（train/val/test）与框数：")
    for c in report["class_map"]:
        cc = report["class_counts"][c["zh"]]
        print(f"    {c['zh']:<6} 图 {cc['train']['images']:>4}/{cc['val']['images']:>4}/"
              f"{cc['test']['images']:>4}  框 {cc['train']['boxes']:>4}/"
              f"{cc['val']['boxes']:>4}/{cc['test']['boxes']:>4}"
              f"{'  [交付]' if c['delivery'] else '  [不交付]'}")
    print(f"  断言：plan {len(report['assertions_plan'])} 项全过"
          + (f"，output {len(report.get('assertions_output', []))} 项全过"
             if report.get("assertions_output") else ""))
    if report.get("fingerprint"):
        print(f"  指纹 root_hash：{report['fingerprint']['root_hash']}")
    print(f"  输出：{args.out_dir}" + ("（dry-run 未落盘）" if args.dry_run else ""))
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
