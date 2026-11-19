# -*- coding: utf-8 -*-
"""
sync_criteria.py — 判据速查修复：页头动态化 + 任务级 criteria_hint 回填
问题: labeling_config 页头只写死"大黄：簇晶 60~140μm"，审核板蓝根/党参/麦冬石细胞
     与导管时会张冠李戴，且大黄石细胞排他判据（掺伪信号）缺失。
修复:
  1) labeling_config.xml 判据页头改为 Header value="$criteria_hint"（按任务动态）
  2) PATCH 项目标注配置（LS 会校验合法性）
  3) 逐任务 PATCH data.criteria_hint（criteria_table.json 按 药材|特征 生成；
     prelabel JSON 的 `scale.status` 非 OK（双引擎未取得可信标定）时附加"μm读数仅供参考"提示）
  4) import_to_labelstudio 的 build_tasks 已同步引用同一逻辑，未来导入自动带上
用法:
  python annot\sync_criteria.py --dry-run   # 预览前10条的 hint
  python annot\sync_criteria.py             # 执行（PATCH 配置 + 216 个任务）
"""
import argparse
import csv
import json
import os


from contracts import PrelabelRecord  # noqa: E402 第 ⑤ 段契约（prelabel 记录）
from ls_common import (DET_CONFIG, DATASET, DET_RESULTS, PRELABEL_DIR, canon_path, load_config,  # noqa: E402
                       ls_api, criteria_hint_for, load_criteria, prelabel_scale)

CONFIG_XML = os.path.join(DET_CONFIG, "labeling_config.xml")
MANIFEST = os.path.join(DET_RESULTS, "pilot_manifest.csv")


def load_scale_status():
    """image_path(规范路径) → 双引擎标定 status（取自 prelabel JSON 的 `scale` 字段）。"""
    out = {}
    for fn in sorted(os.listdir(PRELABEL_DIR)):
        if not fn.endswith(".json"):
            continue
        with open(os.path.join(PRELABEL_DIR, fn), encoding="utf-8") as f:
            pre = PrelabelRecord.from_dict(json.load(f))
        _, st = prelabel_scale(pre)
        out[os.path.normcase(canon_path(pre.image_path))] = st
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--project-id", type=int, default=1)
    args = ap.parse_args()
    cfg = load_config()
    pid = args.project_id
    scale = load_scale_status()

    # 1. 组装 image_path(规范路径) → hint
    hints = {}
    with open(MANIFEST, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            p = os.path.normcase(canon_path(r["image_path"]))
            hints[p] = criteria_hint_for(
                r["herb"], r["feature_label"], scale.get(p, ""))

    if args.dry_run:
        crit = load_criteria()
        print(f"判据表条目: {len(crit['table'])}")
        for p, h in list(hints.items())[:10]:
            print(f"  {os.path.basename(p)}: {h[:80]}...")
        return

    # 2. PATCH 项目标注配置（非法配置会被 LS 400 拒绝，ls_api 会退出）
    with open(CONFIG_XML, encoding="utf-8") as f:
        label_config = f.read()
    ls_api(cfg, "PATCH", f"/api/projects/{pid}", json={"label_config": label_config})
    print(f"项目 {pid} 标注配置已更新（页头动态判据）")

    # 3. 拉取任务 image_url → task id
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

    # 4. 逐任务回填 criteria_hint（保留原 data 其他字段）
    n = 0
    for url, t in url2task.items():
        orig = os.path.join("", "")  # 从 url 反解不了 path，直接用任务数据
        # 任务数据里有 image 的原相对路径不可逆 URL-decode 后即相对路径；
        # 这里改用 manifest hints 按 original 匹配：data 中无 path 字段，
        # 用 urllib.unquote 反解 local-files url 得相对路径再拼 DATASET 根
        import urllib.parse
        rel = urllib.parse.unquote(url.split("?d=")[-1])
        path = os.path.normcase(os.path.normpath(
            os.path.join(DATASET, rel.replace("/", os.sep))))
        hint = hints.get(path)
        if hint is None:
            print(f"  [跳过] task {t['id']} 无 manifest 匹配: {rel}")
            continue
        data = dict(t["data"])
        data["criteria_hint"] = hint
        ls_api(cfg, "PATCH", f"/api/tasks/{t['id']}/", json={"data": data})
        n += 1
        if n % 50 == 0:
            print(f"  已回填 {n}/{len(url2task)}")
    print(f"完成: 回填 {n}/{len(url2task)} 个任务的 criteria_hint")


if __name__ == "__main__":
    main()
