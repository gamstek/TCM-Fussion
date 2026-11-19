# -*- coding: utf-8 -*-
"""test_scalebar_consensus.py — 离线回归：OK-consensus 放宽判据（纯本地，不联网、不耗 API）

覆盖 2026-09-28 新增的"第二独立证据"：
  · cross_validate 的 px_ok_set 分支（B 未证实但有旁证 → OK-consensus）
  · 默认行为不变（不传 px_ok_set 时与旧版逐字段一致）
  · confirmed_bar_px 的**防自我放大**（只由精确 OK 构成，OK-consensus/OK-medium 不回灌）
  · criteria_hint_for 的 caveat 分流 + 空 status 的旧行为保持
  · relax_consensus 离线提升的一致性/幂等/不改 px_per_um 与 objects
"""
import importlib.util
import json
import os
import shutil
import sys
import tempfile
from io import StringIO

DET_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import ls_common as L                                              # noqa: E402
import scalebar_duo as SD                                          # noqa: E402

_fail = []


def ck(name, cond, extra=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  {extra}" if not cond else ""))
    if not cond:
        _fail.append(name)


def A(px, um=20.0):
    return {"bar_px": px, "bar_um": um, "scale_type": "black"}


def B(px, um=20.0):
    return {"bar_px": px, "bar_um": um, "color": "black"}


def cv(a, b, **kw):
    return SD.cross_validate("t.jpg", "", a, b, 0.10, **kw)


def write_pre(root, fn, status, bar_px, px_per_um=11.65):
    p = os.path.join(root, fn)
    with open(p, "w", encoding="utf-8") as f:
        json.dump({"feature_label_folder": "淀粉粒", "herb": "大黄",
                   "image_path": f"C:/fake/{fn}.jpg",
                   "objects": [{"label": "淀粉粒", "tight_bbox": [1, 2, 3, 4]}],
                   "scale": {"px_per_um": px_per_um, "status": status, "grade": "g",
                             "bar_px": bar_px, "bar_um": 20.0, "note": "pxΔ=0.0% um_agree=False"}},
                  f, ensure_ascii=False)
    return p


# ---------------------------------------------------------------- ① cross_validate
print("\n① cross_validate 判定（A=233/20μm → 11.65 px/μm）")
r = cv(A(233), B(233))
ck("双读一致 → OK / grade=high", r["status"] == "OK" and r["grade"] == "high", str(r["status"]))
ck("px_per_um = bar_px/bar_um", abs(r["px_per_um"] - 11.65) < 1e-9, str(r["px_per_um"]))

r = cv(A(233), B(0))
ck("B 未证实 + 不传 px_ok_set → OK-medium（默认行为不变）",
   r["status"] == "OK-medium" and r["grade"] == "medium", str(r["status"]))

r = cv(A(233), B(0), px_ok_set={233})
ck("B 未证实 + bar_px 有旁证 → OK-consensus / grade=consensus",
   r["status"] == "OK-consensus" and r["grade"] == "consensus", str(r))
ck("OK-consensus 仍给出同值 px_per_um", abs(r["px_per_um"] - 11.65) < 1e-9)

r = cv(A(233), B(0), px_ok_set={311, 579})
ck("旁证集不含本图 bar_px → 保持 OK-medium", r["status"] == "OK-medium", str(r["status"]))

r = cv(A(331), B(0), px_ok_set={233, 311})
ck("旁证集不含 331（实测那 1 例）→ 保持 OK-medium", r["status"] == "OK-medium", str(r["status"]))

r = cv(A(233), B(233), px_ok_set={233})
ck("B 已证实优先于旁证 → 仍是 OK（不被降/改档）", r["status"] == "OK", str(r["status"]))

r = cv(A(0), B(0), px_fam=233)
ck("A 未量到尺 + 族先验 → OK-low（原分支不变）",
   r["status"] == "OK-low" and abs(r["px_per_um"] - 11.65) < 1e-9, str(r["status"]))

r = cv(A(0), B(233))
ck("A 未量到尺 + B 过窗 → OK-low（原分支不变）", r["status"] == "OK-low", str(r["status"]))

r = cv(A(233, um=20.0), B(233, um=40.0))
ck("A/B μm 冲突 → manual（不被旁证救回）",
   r["status"] == "manual" and r["px_per_um"] == "", str(r["status"]))

r = cv(A(233), B(0), px_ok_set={233}, img_w=400)          # 尺横过 0.5 幅 → px_A_ok 假
ck("bar_px 超 0.5 幅宽 → 不走 OK-consensus", r["status"] != "OK-consensus", str(r["status"]))

r = cv(A(10), B(0), px_ok_set={10})
ck("px_A 低于 MIN_BAR_PX → 不走 OK-consensus", r["status"] != "OK-consensus", str(r["status"]))

# ---------------------------------------------------------------- ② 状态判定入口
print("\n② is_trusted_status / TRUSTED_STATUSES")
ck("TRUSTED_STATUSES = OK + OK-consensus", L.TRUSTED_STATUSES == ("OK", "OK-consensus"))
ck("OK / OK-consensus 可信（含大小写与空格容错）",
   all(L.is_trusted_status(x) for x in ("OK", "ok", " OK ", "OK-consensus", "OK-CONSENSUS ")))
ck("OK-medium / OK-low / manual / n/a / 空 不可信",
   not any(L.is_trusted_status(x) for x in ("OK-medium", "OK-low", "manual", "n/a", "", None)))

# ---------------------------------------------------------------- ③ 旁证集构建
print("\n③ confirmed_bar_px（防自我放大 + 门槛）")
tmp1 = tempfile.mkdtemp(prefix="scalebar_consensus_")
try:
    for i in range(3):
        write_pre(tmp1, f"ok233_{i}.json", "OK", 233)
    for i in range(2):
        write_pre(tmp1, f"ok311_{i}.json", "OK", 311)          # 只 2 张 → 未达 min_n=3
    for i in range(5):
        write_pre(tmp1, f"cons420_{i}.json", "OK-consensus", 420)   # midpoint 不得回灌
    for i in range(3):
        write_pre(tmp1, f"med500_{i}.json", "OK-medium", 500)
    s = L.confirmed_bar_px(prelabel_dir=tmp1, use_cache=False)
    ck("≥3 张精确 OK 的 bar_px 入集（233）", 233 in s)
    ck("仅 2 张精确 OK 的 311 不入集（门槛 min_n=3）", 311 not in s, str(s))
    ck("OK-consensus 的 420 不回灌（防自我放大）", 420 not in s, str(s))
    ck("OK-medium 的 500 不入集", 500 not in s, str(s))
    s2 = L.confirmed_bar_px(prelabel_dir=tmp1, min_n=2, use_cache=False)
    ck("min_n=2 时 311 入集", 311 in s2 and 233 in s2)
finally:
    shutil.rmtree(tmp1, ignore_errors=True)

# ---------------------------------------------------------------- ④ caveat 分流
print("\n④ criteria_hint_for 的 caveat 分流")
caveat = L.load_criteria()["scale_caveat"]
h_ok = L.criteria_hint_for("大黄", "石细胞", "OK")
h_cs = L.criteria_hint_for("大黄", "石细胞", "OK-consensus")
h_med = L.criteria_hint_for("大黄", "石细胞", "OK-medium")
h_low = L.criteria_hint_for("大黄", "石细胞", "OK-low")
h_empty = L.criteria_hint_for("大黄", "石细胞", "")
ck("OK → 不追加 caveat", caveat not in h_ok)
ck("OK-consensus → 不追加 caveat（新可信档）", caveat not in h_cs)
ck("OK-medium / OK-low → 追加 caveat", caveat in h_med and caveat in h_low)
ck("空 status → 保持旧行为（不追加）", caveat not in h_empty)
ck("OK 与 OK-consensus 的判据文本完全一致", h_ok == h_cs)

# ---------------------------------------------------------------- ⑤ 离线提升
print("\n⑤ relax_consensus（离线提升，不改 px_per_um/objects）")
m02 = None
spec = importlib.util.spec_from_file_location(
    "m02_scale", os.path.join(DET_DIR, "cli", "vlm_prelabel.py"))
m02 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m02)

tmp2 = tempfile.mkdtemp(prefix="relax_consensus_")
try:
    for i in range(3):
        write_pre(tmp2, f"ok233_{i}.json", "OK", 233)
    up_a = write_pre(tmp2, "medium233_a.json", "OK-medium", 233)
    up_b = write_pre(tmp2, "medium233_b.json", "OK-medium", 233, px_per_um=15.55)
    keep = write_pre(tmp2, "medium999.json", "OK-medium", 999)
    low = write_pre(tmp2, "low0.json", "OK-low", 0)

    n = m02.relax_consensus(prelabel_dir=tmp2, dry_run=True)
    ck("dry-run 报出 2 份待提升", n == 2, str(n))
    ck("dry-run 不写盘",
       json.load(open(up_a, encoding="utf-8"))["scale"]["status"] == "OK-medium")

    n = m02.relax_consensus(prelabel_dir=tmp2)
    ck("真跑提升 2 份", n == 2, str(n))
    d = json.load(open(up_a, encoding="utf-8"))
    ck("status → OK-consensus / grade → consensus",
       d["scale"]["status"] == "OK-consensus" and d["scale"]["grade"] == "consensus", str(d["scale"]))
    ck("note 留痕（写明旁证来源）", "旁证" in d["scale"]["note"], d["scale"]["note"])
    ck("px_per_um 未被改动", abs(d["scale"]["px_per_um"] - 11.65) < 1e-9)
    ck("objects 未被改动", d["objects"][0]["tight_bbox"] == [1, 2, 3, 4])
    ck("异值样本（15.55）保留自身 px_per_um",
       abs(json.load(open(up_b, encoding="utf-8"))["scale"]["px_per_um"] - 15.55) < 1e-9)
    ck("bar_px 无旁证者保持 OK-medium",
       json.load(open(keep, encoding="utf-8"))["scale"]["status"] == "OK-medium")
    ck("OK-low 不动", json.load(open(low, encoding="utf-8"))["scale"]["status"] == "OK-low")
    ck("幂等：再跑一次提升 0 份", m02.relax_consensus(prelabel_dir=tmp2) == 0)
finally:
    shutil.rmtree(tmp2, ignore_errors=True)

# ---------------------------------------------------------------- ⑥ 路径迁移
print("\n⑥ migrate_paths（旧根 → 规范路径，幂等，图缺失不伪造路径）")
REAL = os.path.join(L.PROJECT_ROOT, "dataset", "显微镜像", "大黄导管", "4.jpg")
LEGACY = r"D:\dataset\显微镜像\大黄导管\4.jpg"          # canon_path 能映射到 REAL
GONE = r"D:\dataset\显微镜像\不存在的目录\nope.jpg"      # 两处都不存在 → missing
tmp3 = tempfile.mkdtemp(prefix="migrate_paths_")
try:
    if not os.path.isfile(REAL):
        ck("前置：需要一个真实图做迁移样本", False, REAL)
    else:
        def w(fn, p):
            fp = os.path.join(tmp3, fn)
            with open(fp, "w", encoding="utf-8") as f:
                json.dump({"image_path": p, "objects": [{"label": "导管"}],
                           "scale": {"px_per_um": 11.65}}, f, ensure_ascii=False)
            return fp
        f_leg = w("legacy.json", LEGACY)
        f_gone = w("gone.json", GONE)
        f_cur = w("current.json", L.canon_path(LEGACY))
        n = m02.migrate_paths(prelabel_dir=tmp3, dry_run=True)
        ck("dry-run 报出 1 份待改写", n == 1, str(n))
        ck("dry-run 不写盘",
           json.load(open(f_leg, encoding="utf-8"))["image_path"] == LEGACY)
        n = m02.migrate_paths(prelabel_dir=tmp3)
        ck("真跑改写 1 份", n == 1, str(n))
        d = json.load(open(f_leg, encoding="utf-8"))
        ck("image_path → canon_path 结果", d["image_path"] == L.canon_path(LEGACY), d["image_path"])
        ck("原值留痕 image_path_legacy", d.get("image_path_legacy") == LEGACY)
        ck("image_status = ok", d.get("image_status") == "ok")
        ck("objects/scale 未被改动", d["objects"][0]["label"] == "导管"
           and d["scale"]["px_per_um"] == 11.65)
        dg = json.load(open(f_gone, encoding="utf-8"))
        ck("图缺失者：路径不动 + 标 missing",
           dg["image_path"] == GONE and dg.get("image_status") == "missing", str(dg))
        dc = json.load(open(f_cur, encoding="utf-8"))
        ck("已是规范路径者不动", dc["image_path"] == L.canon_path(LEGACY)
           and "image_path_legacy" not in dc)
        ck("幂等：再跑一次改写 0 份", m02.migrate_paths(prelabel_dir=tmp3) == 0)
finally:
    shutil.rmtree(tmp3, ignore_errors=True)

# ---------------------------------------------------------------- ⑦ CLI 接线（只读）
print("\n⑦ CLI：--relax-consensus --dry-run 走通且只读")
_buf, _old = StringIO(), sys.stdout
try:
    sys.stdout = _buf
    sys.argv = ["vlm_prelabel.py", "--relax-consensus", "--dry-run"]
    m02.main()
    ok = True
except SystemExit as e:
    ok = e.code in (0, None)
except Exception as e:                                     # noqa: BLE001
    ok = False
    print("  CLI 异常:", e)
finally:
    sys.stdout = _old
out = _buf.getvalue()
ck("CLI 无异常退出", ok)
ck("CLI 输出含旁证集与待提升计数", "旁证集" in out and "待提升" in out, out[:200])

print()
if _fail:
    print(f"SCALEBAR-CONSENSUS TEST: FAIL {len(_fail)} 项")
    for f in _fail:
        print("  -", f)
    sys.exit(1)
print("SCALEBAR-CONSENSUS TEST: ALL ASSERTIONS PASS")