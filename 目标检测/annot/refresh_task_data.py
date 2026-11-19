# -*- coding: utf-8 -*-
"""
refresh_task_data.py — 从当前 prelabels（v7.1）重建 LS 任务静态字段 data
背景：LS 的任务级 PATCH /api/tasks/{id}/ 对 data 是【整体替换】而非合并——
      只提交部分字段会抹掉其余字段（曾导致 216 张旧任务 image/herb 等列变空）。
动作：按 data.image 缺失与否分两种情况，从 prelabels JSON 重建【完整 data 字典】：
      image / original_filename / herb / folder_feature / group_key / needs_review /
      px_per_um / criteria_hint / prelabel_summary
      → PATCH 完整字典（整体替换语义下也安全）。
用法:
  python annot\refresh_task_data.py            # 执行
  python annot\refresh_task_data.py --dry-run  # 只统计将要修改的任务
"""
import argparse
import json
import os


from contracts import PrelabelRecord  # noqa: E402 第 ⑤ 段契约（prelabel 记录）
from ls_common import (PRELABEL_DIR, canon_path, legacy_file_url, load_config, local_file_url,  # noqa: E402
                       ls_api, criteria_hint_for, prelabel_scale, view_file_url)


def build_full_data(pre):
    """从 prelabel JSON 重建完整任务 data（与 import_to_labelstudio 的字段集一致）。

    px_per_um / 比例尺状态取 prelabel JSON 的 `scale` 字段（检测阶段双引擎标定）。
    """
    ip = canon_path(pre.image_path)   # 兼容迁移前旧绝对路径
    ppu, status = prelabel_scale(pre)
    return {
        "image": view_file_url(ip),
        "original_filename": os.path.basename(ip),
        "herb": pre.herb,
        "folder_feature": pre.feature_label_folder,
        "group_key": pre.group_key,
        "needs_review": pre.needs_review,
        "px_per_um": ppu,
        "criteria_hint": criteria_hint_for(pre.herb, pre.feature_label_folder, status),
        "prelabel_summary": json.dumps(
            [{"label": b.label, "conf": b.confidence,
              "morphology": b.morphology} for b in pre.objects],
            ensure_ascii=False),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--project-id", type=int, default=1)
    args = ap.parse_args()
    cfg = load_config()
    pid = args.project_id

    pre_by_url = {}
    for fn in sorted(os.listdir(PRELABEL_DIR)):
        if not fn.endswith(".json"):
            continue
        with open(os.path.join(PRELABEL_DIR, fn), encoding="utf-8") as f:
            pre = PrelabelRecord.from_dict(json.load(f))
        raw = pre.image_path                      # prelabel 里仍是迁移前的旧绝对路径
        # 库中存量的 data.image 是**按旧根**算的 URL → 用 legacy_file_url 才能匹配上；
        # 同时注册新 URL，兼容已刷新过的任务。
        pre_by_url[legacy_file_url(raw)] = pre
        pre_by_url[local_file_url(canon_path(raw))] = pre
        pre_by_url[view_file_url(canon_path(raw))] = pre

    tasks, page = [], 1
    while True:
        r = ls_api(cfg, "GET", "/api/tasks", params={"project": pid, "page": page,
                   "page_size": 100, "fields": "all"}).json()
        batch = r.get("tasks", [])
        if not batch:
            break
        tasks.extend(batch)
        if len(batch) < 100:
            break
        page += 1
    print(f"任务 {len(tasks)}，prelabels {len(pre_by_url)}")

    # image 缺失的任务无法用 URL 匹配 prelabel，退回 original_filename+herb 模糊匹配
    def match_pre(t):
        d = t.get("data") or {}
        if d.get("image"):
            return pre_by_url.get(d["image"])
        fn = d.get("original_filename")
        if fn:
            hits = [p for p in pre_by_url.values()
                    if os.path.basename(p.image_path) == fn
                    and p.herb == d.get("herb")]
            if len(hits) == 1:
                return hits[0]
        return None

    patched = same = no_pre = 0
    for t in tasks:
        pre = match_pre(t)
        if pre is None:
            no_pre += 1
            print(f"  [无匹配] task {t['id']} data={ {k: (v[:30] if isinstance(v, str) else v) for k, v in (t.get('data') or {}).items()} }")
            continue
        full = build_full_data(pre)
        if (t.get("data") or {}) == full:
            same += 1
            continue
        if args.dry_run:
            print(f"  [将重建] task {t['id']} {full['original_filename']}"
                  f"（现缺字段: {sorted(set(full) - set(t.get('data') or {}))}）")
        else:
            ls_api(cfg, "PATCH", f"/api/tasks/{t['id']}/", json={"data": full})
        patched += 1

    print(f"\n{'将' if args.dry_run else '已'}重建/更新 {patched}，已一致 {same}，无匹配 {no_pre}")


if __name__ == "__main__":
    main()
