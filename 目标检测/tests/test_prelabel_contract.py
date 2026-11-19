# -*- coding: utf-8 -*-
"""test_prelabel_contract.py — 第 ⑤ 段契约（prelabel）离线回归。不联网、不调 LS、不耗额度。

**为什么这一段单独测**：prelabel 是仓内被读得最广的自由格式（10 个模块），
其中 `objects[].*` 与 `scale.*` 的字段名此前**只靠"记得叫什么"**在 writer 与各 reader 之间传递。
本测试锁三件事：
  ① **键序**：writer 的落盘字节依赖 `to_dict()` 的键序（改序 = 静默改产物字节）；
  ② **两套构造不互相污染**：`from_info`（写侧）**刻意不兜底**、`from_dict`（读侧）兜底 ——
     若有人把 `or ""` 加到写侧，产物字节会静默变化，此处即暴露；
  ③ **装载必需字段**：真实语料 422 份现状为 0 缺口，本测试把它变成常驻护栏
     （将来新增 prelabel 若缺 `prompt_version`/`original_size`，在此报出而不是静默 None）。

用法：python test_prelabel_contract.py   （退出码 0 = 全过）
"""
import json
import os
import sys
import tempfile

import contracts as K
import ls_common as L

n_ok = 0
n_fail = 0


def check(name, cond, extra=""):
    global n_ok, n_fail
    if cond:
        n_ok += 1
        print(f"  [OK] {name}")
    else:
        n_fail += 1
        print(f"  [FAIL] {name} {extra}")


# ─────────────────────────────────────────── ① 键序（writer 字节依赖）
print("\n① to_dict 键序（writer 落盘字节依赖此序）")
check("DetBox 键序", list(K.DetBox(label="x").to_dict()) ==
      ["label", "tight_bbox", "confidence", "score", "morphology", "note"],
      list(K.DetBox(label="x").to_dict()))
check("ScaleReading 键序", list(K.ScaleReading().to_dict()) ==
      ["px_per_um", "status", "grade", "source", "bar_px", "bar_um", "note"],
      list(K.ScaleReading().to_dict()))

# ─────────────────────────────────────────── ② 写侧不兜底 / 读侧兜底
print("\n② 写侧 from_info 不兜底 vs 读侧 from_dict 兜底")
info = {"px_per_um": "11.7", "status": "OK", "grade": "high", "source": "A+B",
        "bar_px_cv": 234, "bar_um": 20.0, "note": "n"}
d = K.ScaleReading.from_info(info).to_dict()
check("from_info 映射 bar_px_cv → bar_px", d["bar_px"] == 234 and d["px_per_um"] == "11.7", d)
# 写侧遇 None 必须原样带出（若加了 `or ""` 会静默改产物字节）
info2 = dict(info, px_per_um=None, note=None, grade=None)
d2 = K.ScaleReading.from_info(info2).to_dict()
check("from_info 对 None 原样带出（不兜底）",
      d2["px_per_um"] is None and d2["note"] is None and d2["grade"] is None, d2)
check("from_info 缺 grade 时取空串（与旧 writer 逐字一致）",
      K.ScaleReading.from_info({"px_per_um": "1", "status": "OK", "source": "A",
                                "bar_px_cv": 1, "bar_um": 1.0, "note": ""}).grade == "")
# 读侧兜底
check("from_dict 无 scale → 全空", K.ScaleReading.from_dict(None).to_dict() ==
      {"px_per_um": "", "status": "", "grade": "", "source": "", "bar_px": None,
       "bar_um": None, "note": ""})
check("from_dict 非 dict 不崩", K.ScaleReading.from_dict("garbage").status == "")
check("from_dict px_per_um=null → 空串", K.ScaleReading.from_dict({"px_per_um": None}).px_per_um == "")
check("px_per_um_value 容错", K.ScaleReading.from_dict({"px_per_um": "11.7"}).px_per_um_value == 11.7
      and K.ScaleReading.from_dict({"px_per_um": 0}).px_per_um_value is None)

# ─────────────────────────────────────────── ③ 框层
print("\n③ DetBox 两套构造")
det = {"label": "草酸钙簇晶", "tight_bbox": [1.0, 2.0, 3.0, 4.0],
       "confidence": "high", "score": 0.92, "morphology": "星状"}
raw = {"label": "草酸钙簇晶", "bbox_2d": [10, 20, 30, 40], "note": "n1"}
b = K.DetBox.from_det(det, raw)
check("from_det 取 note 自原始 object", b.note == "n1" and b.label == "草酸钙簇晶")
check("from_det 保留 raw", b.raw.get("bbox_2d") == [10, 20, 30, 40])
check("from_det 缺 raw 不崩", K.DetBox.from_det(det).note == "")
check("from_dict 非 dict 不崩", K.DetBox.from_dict(None).label == "")

# ─────────────────────────────────────────── ④ 顶层装载（宽容）
print("\n④ PrelabelRecord 宽容装载")
full = {"image_path": "a/b.tif", "prompt_version": "v7.4_starch_single_grain",
        "original_size": [2048, 1536], "objects": [{"label": "淀粉粒", "tight_bbox": [1, 2, 3, 4]}]}
rec = K.PrelabelRecord.from_dict(full)
check("齐备者无缺口", K.check_prelabel(rec) == [], K.check_prelabel(rec))
check("objects 装载成 DetBox", len(rec.objects) == 1 and rec.objects[0].label == "淀粉粒")
for bad_in in (None, {}, "garbage", {"objects": None}):
    check(f"病态输入不崩 {bad_in!r}", isinstance(K.PrelabelRecord.from_dict(bad_in), K.PrelabelRecord))

# ─────────────────────────────────────────── ⑤ check_prelabel 判据
print("\n⑤ check_prelabel 判据（一次列全）")
g = K.check_prelabel({"objects": []})
check("空 objects 允许（v7.4 明确可输出空）", any("objects" not in x for x in g) and
      not any("为空 objects" in x for x in g), g)
g = K.check_prelabel({"image_path": "a", "objects": [], "original_size": [1, 1]})
check("缺 prompt_version 报到", any("prompt_version" in x for x in g), g)
g = K.check_prelabel({"image_path": "", "prompt_version": "v", "objects": [], "original_size": [1, 1]})
check("字段为空也报到", any("为空 image_path" in x for x in g), g)
g = K.check_prelabel({"image_path": "a", "prompt_version": "v", "objects": [
    {"label": "", "tight_bbox": [1, 1, 1, 5]}, {"label": "x", "tight_bbox": None}], "original_size": [1, 1]})
check("退化框报到", any("退化框" in x for x in g), g)
check("缺 label 报到", any("缺 label" in x for x in g), g)
check("tight_bbox 非法报到", any("tight_bbox 非法" in x for x in g), g)
check("旧代无 scale 不报（避免淹没真问题）", K.check_prelabel(full) == [])

# ─────────────────────────────────────────── ⑥ load_prelabel 往返 + 真实语料护栏
print("\n⑥ 文件装载往返")
tmp = tempfile.mkdtemp(prefix="prelabel_contract_")
p = os.path.join(tmp, "x.json")
with open(p, "w", encoding="utf-8") as f:
    json.dump({**full, "scale": K.ScaleReading.from_info(info).to_dict()}, f, ensure_ascii=False)
loaded = K.load_prelabel(p)
check("往返：框与标定都在", loaded.objects[0].label == "淀粉粒"
      and loaded.scale.bar_px == 234 and loaded.scale.status == "OK")
check("往返后仍无缺口", K.check_prelabel(loaded) == [], K.check_prelabel(loaded))

# ─────────────────────────────────────────── ⑦ resolve_out_key 幂等（历史哈希名必须认出来）
# 2026-10-12 修：原实现只比对**无后缀**的 `<base>.json`。同一张图若因图像库换根 /
# `--migrate-paths`（只改 image_path、不改文件名）落在 `<base>_<旧哈希>.json` 下，重跑就认不出来
# → 再写一份（实测麦冬针晶/龙胆针晶 7 对），下游还会给同一张图建重复 LS 任务。
print("\n⑦ resolve_out_key 幂等与消歧")
import importlib.util                                    # noqa: E402
_vspec = importlib.util.spec_from_file_location(
    "vlm_pre", os.path.join(L.PROJECT_ROOT, "目标检测", "cli", "vlm_prelabel.py"))
vp = importlib.util.module_from_spec(_vspec)
_vspec.loader.exec_module(vp)
_tmp2 = tempfile.mkdtemp(prefix="resolve_out_key_")
vp.PRELABEL_DIR = _tmp2
_img_a = os.path.join(_tmp2, "src_a", "1.jpg")           # 与既有文件不同图、但都叫 1.jpg
_img_c = os.path.join(_tmp2, "src_c", "2.jpg")           # 历史哈希名对应的那张图
with open(os.path.join(_tmp2, "草酸钙簇晶_1.json"), "w", encoding="utf-8") as f:
    json.dump({"image_path": os.path.join(_tmp2, "src_a", "9.jpg")}, f)
with open(os.path.join(_tmp2, "草酸钙簇晶_2_deadbe.json"), "w", encoding="utf-8") as f:
    json.dump({"image_path": _img_c}, f)
vp._PRELABEL_INDEX = None
check("历史哈希名被认出来（沿用旧名，不另起）",
      vp.resolve_out_key({"feature_label": "草酸钙簇晶", "image_path": _img_c}, "2")[0]
      == "草酸钙簇晶_2_deadbe")
check("无后缀名被别的图占用时才追加哈希",
      vp.resolve_out_key({"feature_label": "草酸钙簇晶", "image_path": _img_a}, "1")[0]
      .startswith("草酸钙簇晶_1_"))

print(f"\n⑧ 真实语料护栏（{L.PRELABEL_DIR}）")
n_file, n_unparsed, bad = 0, 0, []
for fn in sorted(os.listdir(L.PRELABEL_DIR)):
    if not fn.endswith(".json"):
        continue
    n_file += 1
    try:
        gaps = K.check_prelabel(K.load_prelabel(os.path.join(L.PRELABEL_DIR, fn)))
    except Exception as e:                       # 半写状态（批量跑检测时可能读到截断文件）
        n_unparsed += 1
        print(f"  [warn] 无法解析 {fn}: {e}")
        continue
    if gaps:
        bad.append((fn, gaps))
check(f"装载必需字段全齐（{n_file} 份，无法解析 {n_unparsed}）", not bad,
      [f"{fn}: {g[:2]}" for fn, g in bad[:3]])

print(f"\nprelabel 契约回归：{n_ok} 通过 / {n_fail} 失败")
sys.exit(1 if n_fail else 0)
