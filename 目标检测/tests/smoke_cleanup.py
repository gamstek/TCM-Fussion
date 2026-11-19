# -*- coding: utf-8 -*-
"""smoke_cleanup.py — 冒烟任务清点与清理"""
import sys, os
# 本脚本位于 detection/tests/：ls_common 在 标注体系/common/
from ls_common import load_config, ls_api  # noqa: E402

cfg = load_config()
r = ls_api(cfg, "GET", "/api/tasks", params={"project": 1, "page_size": 10})
j = r.json()
t = j["tasks"] if isinstance(j, dict) and "tasks" in j else j
print("/api/tasks?project=1 →", len(t), "个任务")
if t:
    tid = t[0]["id"]
    print("task id:", tid, "| image:", t[0]["data"].get("image"))
    r2 = ls_api(cfg, "DELETE", "/api/tasks/%s/" % tid)
    print("DELETE 状态:", r2.status_code)
    r3 = ls_api(cfg, "GET", "/api/tasks", params={"project": 1, "page_size": 10})
    j3 = r3.json()
    t3 = j3["tasks"] if isinstance(j3, dict) and "tasks" in j3 else j3
    print("清理后任务数:", len(t3))
