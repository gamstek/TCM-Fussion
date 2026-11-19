# -*- coding: utf-8 -*-
"""
sync_prelabels_to_ls.py — 把 prelabels 里的 v7.x 全量预标注同步到 Label Studio（替换旧结果）
- 已有任务（按 data.image URL 匹配）：删除旧 predictions → 写入新预测（不动 annotations 层）
- 没有任务的图（数据量太少合集等 201 张新图）：组装任务（含 criteria_hint）分块导入
- 过滤条件是 prompt_version 以 "v7" 开头（故不限于 v7.1；旧名 sync_v71_to_ls 有误导，2026-09-29 更名）
用法:
  python annot\sync_prelabels_to_ls.py            # 执行
  python annot\sync_prelabels_to_ls.py --dry-run  # 预览
"""
import argparse
import json
import os


from contracts import PrelabelRecord  # noqa: E402 第 ⑤ 段契约（prelabel 记录）
from ls_common import (PRELABEL_DIR, ensure_dirs, load_config, local_file_url, ls_api,  # noqa: E402
                       criteria_hint_for, prelabel_scale, view_file_url)
from ls_tasks import ensure_local_storage  # noqa: E402 LS 存储注册（P4-3 上收）


def build_results_local(pre):
    """与 03/09 的 predictions result 结构一致（03 无此函数，本地实现）。"""
    W0, H0 = pre.original_size
    results = []
    for j, b in enumerate(pre.objects):
        x1, y1, x2, y2 = b.tight_bbox
        results.append({
            "id": f"pred_{j}",
            "type": "rectanglelabels",
            "from_name": "feature", "to_name": "image",
            "original_width": W0, "original_height": H0,
            "image_rotation": 0,
            "value": {
                "x": round(x1 / W0 * 100, 3), "y": round(y1 / H0 * 100, 3),
                "width": round((x2 - x1) / W0 * 100, 3),
                "height": round((y2 - y1) / H0 * 100, 3),
                "rotation": 0,
                "rectanglelabels": [b.label],
            },
            "score": (b.score if b.score is not None else 0.6),
        })
    return results


def build_task_data(pre):
    """任务 data（字段集与 import_to_labelstudio 一致）；px_per_um 取 prelabel JSON 的 `scale` 字段。"""
    ppu, scale_status = prelabel_scale(pre)
    return {
        "image": view_file_url(pre.image_path),
        "original_filename": os.path.basename(pre.image_path),
        "herb": pre.herb,
        "folder_feature": pre.feature_label_folder,
        "group_key": pre.group_key,
        "needs_review": pre.needs_review,
        "px_per_um": ppu,
        "criteria_hint": criteria_hint_for(
            pre.herb, pre.feature_label_folder, scale_status),
        "prelabel_summary": json.dumps(
            [{"label": b.label, "conf": b.confidence,
              "morphology": b.morphology} for b in pre.objects],
            ensure_ascii=False),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--project-id", type=int, default=1)
    ap.add_argument("--purge-annotations", action="store_true",
                    help="同时删除这些任务上的人工 annotations（默认保留）")
    args = ap.parse_args()
    ensure_dirs()
    cfg = load_config()
    pid = args.project_id
    build_results = build_results_local

    # 1. 读取全部 v7.1 预标注
    pres = []
    for fn in sorted(os.listdir(PRELABEL_DIR)):
        if not fn.endswith(".json"):
            continue
        with open(os.path.join(PRELABEL_DIR, fn), encoding="utf-8") as f:
            pre = PrelabelRecord.from_dict(json.load(f))
        if pre.prompt_version.startswith("v7"):
            pres.append(pre)
    print(f"v7.1 预标注 JSON: {len(pres)}")

    # 2. 拉取项目现有任务
    url2task, page = {}, 1
    while True:
        r = ls_api(cfg, "GET", "/api/tasks",
                   params={"project": pid, "page": page, "page_size": 100,
                           "fields": "all"}).json()
        batch = r.get("tasks", [])
        if not batch:
            break
        for t in batch:
            url2task[t["data"]["image"]] = t
        if len(batch) < 100:
            break
        page += 1
    n_ann = sum(len(t.get("annotations") or []) for t in url2task.values())
    print(f"LS 现有任务: {len(url2task)}，其中人工 annotations 共 {n_ann} 条")

    # 3. 分流：更新已有 / 导入新任务
    new_tasks, upd = [], 0
    for pre in pres:
        url = local_file_url(pre.image_path)
        t = url2task.get(url)
        if t is not None:
            if args.dry_run:
                print(f"  [更新] task {t['id']} ← {len(pre.objects)}框 "
                      f"(旧预测 {len(t.get('predictions') or [])}条, 人工标注 {len(t.get('annotations') or [])}条)")
            upd += 1
            continue
        new_tasks.append({"data": build_task_data(pre),
                          "predictions": [{"model_version":
                                           f'{pre.model or "vlm"} {pre.prompt_version or "v7.1"}',
                                           "score": 0.8,
                                           "result": build_results(pre)}]})
    print(f"将更新已有任务: {upd}，新导入任务: {len(new_tasks)}")
    if args.dry_run:
        return

    # 4. 替换已有任务的 predictions
    ok = 0
    for pre in pres:
        url = local_file_url(pre.image_path)
        t = url2task.get(url)
        if t is None:
            continue
        for p in (t.get("predictions") or []):
            ls_api(cfg, "DELETE", f"/api/predictions/{p['id']}/")
        if args.purge_annotations:
            for a in (t.get("annotations") or []):
                ls_api(cfg, "DELETE", f"/api/annotations/{a['id']}/")
        if pre.objects:
            ls_api(cfg, "POST", "/api/predictions", json={
                "task": t["id"],
                "model_version": f'{pre.model or "vlm"} {pre.prompt_version or "v7.1"}',
                "score": 0.8,
                "result": build_results(pre),
            })
        ok += 1
        if ok % 50 == 0:
            print(f"  已更新 {ok}")
    print(f"predictions 替换完成: {ok}")

    # 5. 分块导入新任务
    if new_tasks:
        ensure_local_storage(cfg, pid)
        total = 0
        for i in range(0, len(new_tasks), 50):
            part = new_tasks[i:i + 50]
            r = ls_api(cfg, "POST", f"/api/projects/{pid}/import", json=part)
            total += r.json().get("task_count", len(part))
            print(f"  导入 {i + len(part)}/{len(new_tasks)}（累计 {total}）")
        print(f"新任务导入完成: {total}")

    # 6. 校验
    proj = ls_api(cfg, "GET", f"/api/projects/{pid}").json()
    print(f"\n项目 {pid} 现状: 任务 {proj.get('task_number')}，"
          f"predictions {proj.get('total_predictions')}，annotations {proj.get('total_annotations')}")


if __name__ == "__main__":
    main()
