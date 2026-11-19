# -*- coding: utf-8 -*-
"""cleanup_e2e.py — 清理 e2e 测试残留：自检项目、冒标任务、导出目录"""
import json
import os
import shutil

# 本脚本位于 detection/tests/：ls_common 在 标注体系/common/
from ls_common import DET_RESULTS, load_config, ls_api  # noqa: E402

cfg = load_config()

# 1) 删自检项目（从 state 读，state 不在则扫项目列表找"自检"标题）
def safe_delete_project(pid):
    try:
        ls_api(cfg, "DELETE", "/api/projects/%s/" % pid)
        print("已删自检项目", pid)
    except SystemExit:
        print("自检项目", pid, "已不存在，跳过")

try:
    state = json.load(open(os.path.join(DET_RESULTS, "selfcheck_state.json"), encoding="utf-8"))
    safe_delete_project(state["selfcheck_project"])
except FileNotFoundError:
    pass

r = ls_api(cfg, "GET", "/api/projects").json()
items = r["results"] if isinstance(r, dict) else r
for p in items:
    if "自检" in p["title"]:
        safe_delete_project(p["id"])

# 2) 删项目1里所有任务（e2e 冒标）
j = ls_api(cfg, "GET", "/api/tasks", params={"project": 1, "page_size": 100}).json()
ts = j["tasks"] if isinstance(j, dict) and "tasks" in j else j
for t in ts:
    ls_api(cfg, "DELETE", "/api/tasks/%s/" % t["id"])
    print("已删任务", t["id"])

# 3) 清导出目录与 state
shutil.rmtree(os.path.join(DET_RESULTS, "yolo_export"), ignore_errors=True)
for f in ("selfcheck_state.json",):
    fp = os.path.join(DET_RESULTS, f)
    if os.path.exists(fp):
        os.remove(fp)

j = ls_api(cfg, "GET", "/api/tasks", params={"project": 1, "page_size": 100}).json()
ts = j["tasks"] if isinstance(j, dict) and "tasks" in j else j
print("项目1剩余任务数:", len(ts))
r = ls_api(cfg, "GET", "/api/projects").json()
items = r["results"] if isinstance(r, dict) else r
print("现存项目:", [(p["id"], p["title"]) for p in items])
print("CLEANUP DONE")
