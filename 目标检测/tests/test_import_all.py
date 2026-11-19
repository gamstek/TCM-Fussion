# -*- coding: utf-8 -*-
"""test_import_all.py — 入口导入冒烟：每个活动模块在独立子进程里 importlib.import_module 一次。

护栏目的（2026-09-29 共享层上收配套）：模块不是包、跨模块导入靠手拼 sys.path —— 共享层从
detection/ 上收到 common/ 后，若漏改某文件的路径注入，该文件在“只有它自己所在目录”的 sys.path
下就会 import 失败（ls_common / scalebar_duo / textvis 解析不到）。本测试对每个 .py 单独起子进程，
把子进程 sys.path 的项目目录收窄为该文件所在目录，验证其路径注入自洽。

枚举范围：目标检测/ 图像分割/ common/ ui/ 下全部 .py，跳过 archive/ results/ .venv/ experiments/ 与 _tmp_*.py；
并跳过 tests/ —— 该目录下是“执行型”测试脚本（import 即联网/建删产物，不是可导入模块），
它们由 目标检测/tests/ 下的回归命令显式运行，不属于生产入口。

产出：全部可导入 → 打印 "IMPORT-ALL TEST: ALL ASSERTIONS PASS"；否则列出失败文件与异常并 exit(1)。
用 tests/ 目录现有测试的脚本式风格（断言、结尾打印大写 PASS）。
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))           # <项目>/目标检测/tests
PROJ = os.path.dirname(os.path.dirname(HERE))               # 项目根
# 2026-10-09 目录重构：原枚举根「标注体系/」已解散 —— 检测/分割两阶段与共享层 common、
# 编排层 ui 均上提到项目根平级，故枚举根改为这四个代码目录。
ROOTS = [os.path.join(PROJ, d) for d in ("目标检测", "图像分割", "common", "ui")]
PY = sys.executable

SKIP_DIRS = {"archive", "results", ".venv", "experiments", "tests", "__pycache__"}


def iter_modules():
    """枚举四个代码目录下的活动 .py（跳过 SKIP_DIRS 与 _tmp_*.py），返回绝对路径列表。"""
    out = []
    for root in ROOTS:
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
            for fn in filenames:
                if not fn.endswith(".py") or fn.startswith("_tmp_"):
                    continue
                out.append(os.path.join(dirpath, fn))
    return sorted(out)


def _check(path):
    """在子进程里导入该模块：sys.path 只保留 stdlib/site-packages + 该文件所在目录。"""
    d, fn = os.path.split(path)
    name = fn[:-3]
    code = (
        "import os, sys\n"
        "_d = os.path.abspath(%r)\n"
        "sys.path = [_d] + [p for p in sys.path if p and os.path.abspath(p) != _d]\n"
        "import importlib\n"
        "importlib.import_module(%r)\n" % (d, name)
    )
    cp = subprocess.run([PY, "-X", "utf8", "-c", code], capture_output=True,
                        text=True, encoding="utf-8", errors="replace", timeout=300)
    return cp.returncode, (cp.stderr or "").strip()


def main():
    mods = iter_modules()
    fails = []
    for p in mods:
        rc, err = _check(p)
        rel = os.path.relpath(p, PROJ)
        if rc == 0:
            print("  [ok  ] %s" % rel)
        else:
            fails.append((rel, err))
            print("  [FAIL] %s" % rel)
    print("\n枚举 %d 个活动 .py（跳过 archive/ results/ .venv/ experiments/ tests/ 与 _tmp_*.py）"
          % len(mods))
    if fails:
        print("\n导入失败 %d 个：" % len(fails))
        for rel, err in fails:
            print("---- %s ----" % rel)
            print(err)
        sys.exit(1)
    print("IMPORT-ALL TEST: ALL ASSERTIONS PASS")


if __name__ == "__main__":
    main()