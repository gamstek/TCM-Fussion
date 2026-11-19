# -*- coding: utf-8 -*-
"""export_det_dataset.py — 把 Label Studio 的人工核验结果导出为 YOLO 目标检测数据集

定位：训练集构建闭环的最后一步——人工在 LS 核验完后，把**人工 annotation**导成
      `images/` + `labels/` 的 YOLO 格式数据集（AI predictions 一律忽略）。
      按类逐目录导出（`dataset/目标检测标注/<特征>/`），每个目录是一份独立可训数据集。

与归档的 `archive/detection_scripts/04_export_yolo.py` 的差异：
  - 输出目录由 `--out-dir` 指定（04 硬编码 YOLO_OUT_DIR）；
  - 不写 04 那种 train/val/test 全指 images 的占位划分——本脚本只出全量，
    划分另行处理（防泄漏要按 group_key 组感知，见 attributes.csv）；
  - LS 读取复用 `lib/ls_tasks.py` 的 `fetch_tasks` / `rects_of`，不重写分页与坐标转换。

用法:
  python cli\\export_det_dataset.py --project-id 34 --out-dir "dataset\\目标检测标注\\草酸钙簇晶"
  python cli\\export_det_dataset.py --project-id 34 --out-dir <dir> --labels "草酸钙簇晶"
  python cli\\export_det_dataset.py --project-id 34 --out-dir <dir> --dry-run   # 只统计，不落盘

产物（<out-dir>/）:
  images/<名>.<ext>   原图**原样复制**（tif/jpg 保留扩展名，不重编码）
  labels/<名>.txt     YOLO 归一化 "<cls> <cx> <cy> <w> <h>"；无目标图写成空文件（负样本）
                      框先归正（LS 反向拖拽会给负 width）再夹到图像范围内，退化框丢弃
  classes.txt         本数据集实际出现的类别（按 yolo.classes 全局 id 升序，重编为 0..N-1）
  data.yaml           YOLO 训练配置（path/names；train/val/test 同指 images/，划分另行处理）
  attributes.csv      每框一行：image/feature_id/zh/judge/note/group_key
  export_report.json  统计 + 未核验 / 缺原图 / 空框图清单

tif 说明：LS 里 tif 任务的 `image` 指向 `ls_view` 的 jpg 代理，本脚本按该约定反推回
         `.tif` 原图再复制（代理只服务于浏览器显示，数据集要的是原图）。
"""
import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
from urllib.parse import parse_qs, unquote, urlparse

from ls_common import DATASET, PROJECT_ROOT, VIEW_ROOT, load_config  # noqa: E402
from ls_tasks import fetch_tasks  # noqa: E402 LS 读取原语（分页拉取任务+标注）


def source_abs_path(image_url):
    """任务 image URL → 原图绝对路径。

    URL 形如 `/data/local-files/?d=<相对 DATASET 的路径>`。tif 任务显示的是 ls_view 的
    jpg 代理，按 view_file_url() 的约定反推回 .tif/.tiff 原图（代理目录外的路径原样返回）。
    """
    q = parse_qs(urlparse(image_url or "").query)
    if not q.get("d"):
        return ""
    p = os.path.join(DATASET, unquote(q["d"][0]).replace("/", os.sep))
    view_root = os.path.join(DATASET, os.path.relpath(VIEW_ROOT, DATASET))
    if os.path.normcase(p).startswith(os.path.normcase(view_root + os.sep)):
        # 代理在 ls_view 下，原图在 DATASET 下**同构**的相对路径（去掉 ls_view 前缀），
        # 代理恒为 .jpg、原图恒为 .tif/.tiff（view_file_url 只对 tif 出代理）。
        orig_stem = os.path.join(DATASET, os.path.relpath(os.path.splitext(p)[0], view_root))
        for ext in (".tif", ".tiff"):
            if os.path.exists(orig_stem + ext):
                return orig_stem + ext
        print(f"  [warn] 找不到 tif 原图，退回 jpg 代理: {p}", file=sys.stderr)
    return p


def _rects_choices(ann_results):
    """一条 annotation 的 result → (矩形结果列表, choices 按 parent_id 索引, 备注串)。"""
    rects, choices, notes = [], {}, []
    for res in ann_results or []:
        t, v = res.get("type"), res.get("value", {})
        if t == "rectanglelabels":
            rects.append(res)
        elif t == "choices":
            choices[res.get("parent_id") or res.get("id")] = (v.get("choices") or [""])[0]
        elif t == "textarea":
            notes.append(" ".join(v.get("textarea", [])))
    return rects, choices, " | ".join(notes)


def collect(cfg, pid, labels=None):
    """拉取项目人工标注 → 记录列表 + 跳过后清单。

    只认**最新一条非取消**的人工 annotation；无标注的任务计入 skipped（不算负样本）。
    labels 非空时只保留其中的框（目录命中但无此类框 → 负样本，写空 label）。
    """
    keep = set(labels or [])
    records, skipped, dropped = [], [], []
    for t in fetch_tasks(cfg, pid):
        data = t.get("data") or {}
        anns = [a for a in (t.get("annotations") or []) if not a.get("was_cancelled")]
        if not anns:
            skipped.append({"task_id": t.get("id"), "image": data.get("original_filename", ""),
                            "reason": "无人工作业"})
            continue
        rects, choices, note = _rects_choices(anns[-1].get("result", []))
        src = source_abs_path(data.get("image", ""))
        if not src or not os.path.exists(src):
            skipped.append({"task_id": t.get("id"), "image": data.get("original_filename", ""),
                            "reason": f"原图缺失: {src or data.get('image','')[:80]}"})
            continue
        boxes = []
        for r in rects:
            zh = (r["value"].get("rectanglelabels") or [""])[0]
            if keep and zh not in keep:
                continue
            try:
                x, y, w, h = (float(r["value"][k]) for k in ("x", "y", "width", "height"))
            except (TypeError, KeyError, ValueError):
                # LS 偶发把矩形几何存成 null（2026-10-14 实测 石细胞项目 task 1890 / B20250257.jpg
                # 一条记录 x/y/width/height 全 None，疑似画框中途保存或记录损坏）→ 此前会让整个导出
                # 在 float(None) 处抛 TypeError 直接崩掉。改为丢弃 + 上报，与下面「越界夹取后面积归零」
                # 同一口径，靠 dropped 清单回查，不静默吞掉。
                dropped.append({"image": os.path.basename(src), "zh": zh,
                                "raw_xywh_pct": [r["value"].get(k) for k in ("x", "y", "width", "height")],
                                "reason": "矩形几何为 null（LS 记录损坏）"})
                continue
            # LS 允许框越界、也允许反向拖拽（此时 width/height 为负）→ 先归正成 x1<x2/y1<y2，
            # 再夹到图像范围内 [0,100]%；夹后面积退化为 0 的框丢弃（训练用不了）。
            x1, x2 = sorted((x, x + w))
            y1, y2 = sorted((y, y + h))
            x1, x2 = max(0.0, x1), min(100.0, x2)
            y1, y2 = max(0.0, y1), min(100.0, y2)
            if x2 - x1 <= 0 or y2 - y1 <= 0:
                # 夹到图像范围后面积归零（框整体落在画幅外/退化）→ 丢弃并上报，不静默吞掉
                dropped.append({"image": os.path.basename(src), "zh": zh,
                                "raw_xywh_pct": [x, y, w, h], "reason": "越界夹取后面积归零"})
                continue
            boxes.append({"zh": zh, "xyxy_pct": (x1, y1, x2, y2),
                          "judge": choices.get(r.get("id"), ""), "note": note})
        records.append({"task_id": t.get("id"), "src": src,
                        "stem": os.path.splitext(os.path.basename(src))[0],
                        "ext": os.path.splitext(src)[1].lower(),
                        "boxes": boxes,
                        "group_key": data.get("group_key", ""),
                        "herb": data.get("herb", "")})
    return records, skipped, dropped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-id", type=int, required=True)
    ap.add_argument("--out-dir", required=True, help="数据集输出目录（相对项目根或绝对路径）")
    ap.add_argument("--labels", default="", help="逗号分隔类别白名单（默认空=全部）")
    ap.add_argument("--dry-run", action="store_true", help="只统计，不落盘")
    args = ap.parse_args()

    cfg = load_config()
    out_dir = args.out_dir if os.path.isabs(args.out_dir) else os.path.join(PROJECT_ROOT, args.out_dir)
    labels = [x.strip() for x in args.labels.split(",") if x.strip()]

    records, skipped, dropped = collect(cfg, args.project_id, labels)
    zh2gid = {c["zh"]: c["id"] for c in cfg["yolo"]["classes"]}
    zh2fid = {c["zh"]: c["feature_id"] for c in cfg["yolo"]["classes"]}
    # 本数据集实际出现的类别 → 按全局 id 升序重编为 0..N-1（单类导出即 class 0）
    used = sorted({b["zh"] for r in records for b in r["boxes"]},
                  key=lambda zh: zh2gid.get(zh, 999))
    cls_of = {zh: i for i, zh in enumerate(used)}
    n_box = sum(len(r["boxes"]) for r in records)
    n_neg = sum(1 for r in records if not r["boxes"])

    print(f"人工标注任务: {len(records)} 张 | 框: {n_box} | 空框(负样本): {n_neg}")
    print(f"类别: {used or '（无）'}  → {[cls_of[z] for z in used]}")
    if skipped:
        print(f"跳过 {len(skipped)} 张：")
        for s in skipped[:10]:
            print(f"  [{s['reason']}] {s['image']} (task {s['task_id']})")
        if len(skipped) > 10:
            print(f"  … 其余 {len(skipped) - 10} 条见 export_report.json")
    if dropped:
        # 原因从各条记录里取（可能是「越界夹取后面积归零」也可能是「矩形几何为 null」），
        # 不写死标题：否则新增的丢弃原因在控制台上会冒充成越界问题，误导复核方向。
        print(f"丢弃框 {len(dropped)} 个（需回 LS 复核）：")
        for d in dropped:
            print(f"  {d['image']} {d['zh']} raw(x,y,w,h)%={d['raw_xywh_pct']} ← {d['reason']}")
    if args.dry_run:
        print("[dry-run] 不落盘。")
        return

    img_dir = os.path.join(out_dir, "images")
    lbl_dir = os.path.join(out_dir, "labels")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    # 同名消歧必须按 **stem** 而不是完整文件名：YOLO 的 label 名 = 图像名去扩展名 + .txt，
    # 若两张不同源图只差扩展名（如 x.tif 与 x.jpg），完整名不同但 label 名相同 → 会互相覆盖。
    used_stems, attr_rows = {}, []
    for r in records:
        stem = r["stem"]
        if used_stems.get(stem, r["src"]) != r["src"]:
            stem = f'{stem}_{hashlib.md5(r["src"].encode("utf-8")).hexdigest()[:6]}'
        used_stems[stem] = r["src"]
        name = stem + r["ext"]
        shutil.copyfile(r["src"], os.path.join(img_dir, name))
        lines = []
        for b in r["boxes"]:
            x1, y1, x2, y2 = b["xyxy_pct"]
            cx, cy = (x1 + x2) / 200, (y1 + y2) / 200
            lines.append(f'{cls_of[b["zh"]]} {cx:.6f} {cy:.6f} '
                         f'{(x2 - x1) / 100:.6f} {(y2 - y1) / 100:.6f}')
            attr_rows.append({
                "image": name, "feature_id": zh2fid.get(b["zh"], ""), "zh": b["zh"],
                "judge": b["judge"], "note": b["note"],
                "group_key": r["group_key"], "herb": r["herb"],
            })
        # 无目标图写空文件（YOLO 负样本约定），不跳过
        with open(os.path.join(lbl_dir, os.path.splitext(name)[0] + ".txt"),
                  "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + ("\n" if lines else ""))

    with open(os.path.join(out_dir, "classes.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(used) + ("\n" if used else ""))
    with open(os.path.join(out_dir, "data.yaml"), "w", encoding="utf-8") as f:
        f.write(f"path: {out_dir.replace(os.sep, '/')}\n")
        f.write("train: images\nval: images\ntest: images\n")
        f.write("names:\n" + "\n".join(f"  {cls_of[z]}: {z}" for z in used) + "\n")
    with open(os.path.join(out_dir, "attributes.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["image", "feature_id", "zh", "judge", "note",
                                          "group_key", "herb"])
        w.writeheader()
        w.writerows(attr_rows)
    with open(os.path.join(out_dir, "export_report.json"), "w", encoding="utf-8") as f:
        json.dump({"project_id": args.project_id, "images": len(records), "boxes": n_box,
                   "negatives": n_neg, "classes": used, "skipped": skipped,
                   "dropped_boxes": dropped},
                  f, ensure_ascii=False, indent=2)

    print(f"\n导出完成 → {out_dir}")
    print(f"  images/ {len(records)} 张 | labels/ {len(records)} 个 | classes.txt {len(used)} 类")
    print(f"  data.yaml 的 train/val/test 同指 images/（未划分；划分请按 attributes.csv 的 group_key 组感知进行）")


if __name__ == "__main__":
    main()
