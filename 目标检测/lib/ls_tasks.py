# -*- coding: utf-8 -*-
"""ls_tasks.py — Label Studio 项目/任务侧的读写通用件（编号脚本与 UI 共用）。

**为什么存在**（2026-09-29 P4-3）：这些函数原先分别住在
`import_to_labelstudio.py`（注册本地存储 / 分块导入）、`selfcheck_kappa.py`
（分页拉任务 / 标注转框）、`det_eval.py`（按 gold 清单回链人工框）里，被别的脚本用
`import_module("03_…")` 反向取用。编号脚本既当 CLI 又当库有两重代价：
  ① "编号文件名可被 import"只在 sys.path 魔法下成立 —— `03_…` 不是合法标识符（PEP 3131），
     换个解释器/换条路径就崩，且这种失败没有静态信号；
  ② 入口脚本的任何改动都会波及借用方，而借用关系只写在注释里。
本次上收到具名模块，编号脚本回归**纯 CLI**（P1-3 的原则到此彻底兑现）。

依赖方向：只依赖 `ls_common`（LS 鉴权 + 路径 + 归一化），**不得** import 任何编号脚本。
"""
import os
from collections import defaultdict

from ls_common import DATASET, IMAGE_SUBDIRS, canon_path, ls_api, norm_path


def fetch_tasks(cfg, pid):
    """分页拉取项目全部任务（`fields=all`，含 predictions 与 annotations）。"""
    tasks, page = [], 1
    while True:
        r = ls_api(cfg, "GET", "/api/tasks",
                   params={"project": pid, "page": page, "page_size": 100, "fields": "all"},
                   ok_statuses=(404,))
        # 任务数恰为 page_size 整数倍时，还会多请求一页；本 LS 版本对越界页返回
        # 404 {"detail":"Invalid page"}（而非空列表），故必须显式按状态码收尾——
        # 否则分页在此中断、返回值残缺（项目 43 的 500 任务正好命中）。
        if r.status_code != 200:
            break
        j = r.json()
        batch = j["tasks"] if isinstance(j, dict) and "tasks" in j else j
        if not isinstance(batch, list) or not batch:
            break
        tasks.extend(batch)
        if len(batch) < 100:
            break
        page += 1
    return tasks


def rects_of(ann_results, W, H):
    """LS annotation 的 `result` 列表 → `[{"label", "box"}]`（百分比坐标回转像素）。"""
    out = []
    for res in ann_results:
        if res.get("type") != "rectanglelabels":
            continue
        v = res["value"]
        out.append({
            "label": v["rectanglelabels"][0],
            # 转回像素坐标
            "box": [v["x"] / 100 * W, v["y"] / 100 * H,
                    (v["x"] + v["width"]) / 100 * W, (v["y"] + v["height"]) / 100 * H],
        })
    return out


def ann_from_tasks(tasks, gold, pre_index):
    """LS 任务列表（活实例或导出 JSON 同构）→ `({norm_path: [人工框]}, 未匹配, 空白标注数)`。

    回链优先级：`data.image_path` > `original_filename` 唯一命中；两者都失败则计入未匹配。
    无 annotation 的任务 → 视为未标注（返回 None，不计入指标分母）。
    """
    by_name = defaultdict(list)
    for it in gold["items"]:
        by_name[os.path.basename(it["image_path"])].append(it["image_path"])
    out, unmatched, blank = {}, [], 0
    for t in tasks:
        d = t.get("data") or {}
        p = d.get("image_path") or ""
        if p:
            p = canon_path(p)
            if norm_path(p) not in pre_index:
                p = ""
        if not p:
            cands = by_name.get(d.get("original_filename", ""), [])
            p = cands[0] if len(cands) == 1 else ""
        if not p:
            unmatched.append(d.get("original_filename") or d.get("image", "")[:60])
            continue
        anns = [a for a in (t.get("annotations") or []) if not a.get("was_cancelled")]
        if not anns:
            continue
        W, H = pre_index[norm_path(p)].original_size      # C2a 起 pre_index 值是 PrelabelRecord
        res = anns[-1].get("result", [])
        boxes = rects_of(res, W, H)
        if not boxes:
            blank += 1
        out[norm_path(p)] = boxes
    return out, unmatched, blank


def ensure_local_storage(cfg, pid):
    """为项目注册本地文件存储（已存在时静默跳过）。"""
    for sub in IMAGE_SUBDIRS:
        try:
            ls_api(cfg, "POST", "/api/storages/localfiles",
                   json={"project": pid, "path": os.path.join(DATASET, sub),
                         "title": os.path.basename(sub), "use_blob_urls": True})
            print(f"已注册本地存储: {sub}")
        except SystemExit:
            print(f"[提示] 存储 {sub} 可能已注册，继续")


def import_in_chunks(cfg, pid, tasks, chunk=50):
    """把任务分块 POST 到项目，返回实际导入条数。"""
    total = 0
    for i in range(0, len(tasks), chunk):
        part = tasks[i:i + chunk]
        r = ls_api(cfg, "POST", f"/api/projects/{pid}/import", json=part)
        total += r.json().get("task_count", len(part))
        print(f"  导入 {i + len(part)}/{len(tasks)}（累计 {total}）")
    return total
