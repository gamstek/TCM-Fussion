# -*- coding: utf-8 -*-
"""
selftest_synthetic.py — 几何测量模块自测（合成已知真值的 mask）

覆盖三类验证：
  A. 描述子精度：圆/椭圆/星形/针状/弯纤维/导管/复粒等合成 mask，
     已知像素真值 → 检查 Feret 径、等效径、骨架宽、骨架长、实心度、凹角数、分粒数；
  B. μm 换算：给定 px_per_um 标定，检查换算精度与标定缺失时的降级行为；
  C. 标定链路：分割交付清单读取（`load_delivery_manifest`）+ 比例尺探测对照工具 +
     **measure.py 缺 px_per_um 时报错退出的硬门槛**。

同时在 demo_masks/ 落盘全部合成 mask（文件名带 feature 标签，
可直接喂给 measure.py 演示端到端），并生成 demo_manifest.csv。

运行： python selftest_synthetic.py     （CPU 数秒，退出码 0 = 全部 PASS）
"""
import os
import sys
import math
import shutil
import subprocess

import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from geomlib import describe_mask, measure_instance, load_instances, \
    detect_scale_bar_px, load_delivery_manifest
from geomlib.calibration import Calibration

BASE = os.path.dirname(os.path.abspath(__file__))
DEMO_MASK_DIR = os.path.join(BASE, "demo_masks")

RESULTS = []

# 控制台 GBK 环境防崩（输出内容仍以 UTF-8 语义为准）
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def check(name, value, expect, tol):
    ok = expect is None or abs(value - expect) <= tol
    RESULTS.append(ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:38s} 实测={value:9.3f} 期望={expect}{f' ±{tol}' if expect is not None else ''}")
    return ok


def check_bool(name, cond):
    RESULTS.append(bool(cond))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    return cond


# ------------------------------------------------------------------ 合成场景
def scene_circle():
    m = np.zeros((160, 160), np.uint8)
    cv2.circle(m, (80, 80), 25, 1, -1)          # 直径 50px
    return m, "cluster_crystal", {"feret_max": 50, "eq_diameter": 50, "max_inscribed_dia": 50}


def scene_ellipse():
    m = np.zeros((200, 220), np.uint8)
    cv2.ellipse(m, (110, 100), (50, 20), 0, 0, 360, 1, -1)   # 100×40px
    return m, "starch_grain", {"feret_max": 100, "feret_min": 40, "aspect_ratio": 2.5,
                               "eq_diameter": 2 * math.sqrt(math.pi * 50 * 20) / math.pi}


def scene_star():
    pts = []
    for i in range(10):
        r = 60 if i % 2 == 0 else 30
        a = math.pi * i / 5
        pts.append([80 + r * math.cos(a), 80 + r * math.sin(a)])
    m = np.zeros((160, 160), np.uint8)
    cv2.fillPoly(m, [np.array(pts, np.int32)], 1)
    # 5 外凸点星：最远点对=相邻隔一个的外顶点，弦长 2·r·sin(72°)≈114.1
    return m, "cluster_crystal", {"feret_max": 2 * 60 * math.sin(math.radians(72))}


def scene_needle():
    m = np.zeros((80, 260), np.uint8)
    cv2.fillConvexPoly(m, np.array([[10, 40], [250, 37], [250, 43], [10, 40]], np.int32), 1)
    return m, "needle_crystal", {"feret_max": 240, "mean_width": 5.5}


def scene_curved_fiber():
    m = np.zeros((260, 320), np.uint8)
    for t in np.linspace(0, 1, 400):
        x = 20 + t * 280
        y = 130 + 40 * math.sin(t * math.pi * 2)
        cv2.circle(m, (int(x), int(y)), 4, 1, -1)      # 宽 8px 正弦管
    return m, "fiber", {"mean_width": 8, "tortuosity_low": 1.02}


def scene_vessel():
    m = np.zeros((220, 460), np.uint8)
    for t in np.linspace(0, 1, 700):
        x = 20 + t * 420
        y = 110 + 30 * math.sin(t * math.pi * 1.5)
        cv2.circle(m, (int(x), int(y)), 12, 1, -1)     # 宽 24px 管道
    return m, "vessel", {"mean_width": 24}


def scene_compound_starch():
    m = np.zeros((180, 180), np.uint8)
    centers = [(90, 55), (60, 110), (120, 112)]        # 三分粒复粒，r=20
    for cx, cy in centers:
        cv2.circle(m, (cx, cy), 20, 1, -1)
    dmax = max(math.dist(a, b) for i, a in enumerate(centers)
               for b in centers[i + 1:])
    return m, "starch_grain", {"feret_max": dmax + 40, "dt_peak_count": 3}


# ------------------------------------------------------------------ A: 描述子
def run_descriptors_tests():
    print("=" * 72)
    print("A. 描述子精度（合成 mask，像素真值）")
    m, label, gt = scene_circle()
    d = describe_mask(m)
    check("圆 feret_max", d["feret_max_px"], gt["feret_max"], 1.5)
    check("圆 eq_diameter", d["eq_diameter_px"], gt["eq_diameter"], 1.5)
    check("圆 max_inscribed_dia", d["max_inscribed_dia_px"], gt["max_inscribed_dia"], 2.0)
    check_bool("圆 circularity>0.9", d["circularity"] > 0.9)

    m, label, gt = scene_ellipse()
    d = describe_mask(m)
    check("椭圆 feret_max", d["feret_max_px"], gt["feret_max"], 1.5)
    check("椭圆 feret_min", d["feret_min_px"], gt["feret_min"], 2.0)
    check("椭圆 aspect_ratio", d["aspect_ratio"], gt["aspect_ratio"], 0.1)

    m, _, gt = scene_star()
    d = describe_mask(m)
    check("星形(簇晶) feret_max", d["feret_max_px"], gt["feret_max"], 2.0)
    check_bool("星形 solidity<0.9 (多瓣)", 0.4 < d["solidity"] < 0.9)
    check_bool("星形 凹角数>=5", d["defect_count"] >= 5)

    m, _, gt = scene_needle()
    d = describe_mask(m)
    check("针晶 feret_max", d["feret_max_px"], gt["feret_max"], 2.5)
    check("针晶 沿骨架宽", d.get("mean_width_px", 0), gt["mean_width"], 1.5)
    check_bool("针晶 长细比>20", d["aspect_ratio"] > 20)

    m, _, gt = scene_curved_fiber()
    d = describe_mask(m)
    check("弯纤维 沿骨架宽", d.get("mean_width_px", 0), gt["mean_width"], 1.6)
    check_bool("弯纤维 tortuosity>1.02", d["tortuosity"] > 1.02)
    check_bool("弯纤维 骨架长>feret(曲率)", d["skeleton_len_px"] > d["feret_max_px"])

    m, _, gt = scene_vessel()
    d = describe_mask(m)
    check("导管 沿骨架宽(管径)", d.get("mean_width_px", 0), gt["mean_width"], 2.0)
    check_bool("导管 median宽≈mean", abs(d["median_width_px"] - d["mean_width_px"]) < 3)

    m, _, gt = scene_compound_starch()
    d = describe_mask(m)
    check("复粒 feret_max", d["feret_max_px"], gt["feret_max"], 3.0)
    check_bool("复粒 分粒数==3", d["dt_peak_count"] == 3)


# ------------------------------------------------------------------ B: μm 换算
def run_conversion_tests():
    print("=" * 72)
    print("B. μm 换算（px_per_um=10 标定）")
    m, _, _ = scene_needle()
    inst = type("I", (), {"image_stem": "t_needle", "instance_id": 1,
                          "mask": m, "label": "needle_crystal", "meta": {}})()
    calib = Calibration(10.0, 100.0, 20.0, "delivery_manifest", "ok")
    rec = measure_instance(inst, calib, herb="半夏")
    check("针晶 feret_max μm", rec["feret_max_um"], 24.0, 0.3)
    check("针晶 主判尺寸=feret_max", 0 if rec["primary_dimension"] == "feret_max_um" else 1, 0, 0)
    check("针晶 area um^2", rec["area_um2"], rec["area_px"] / 100.0, 1e-6)

    calib_bad = Calibration(None, None, None, "uncalibrated", "failed")
    rec2 = measure_instance(inst, calib_bad)
    check_bool("未标定 → μm=None 且打 flag",
               rec2["feret_max_um"] is None and "uncalibrated" in rec2["flags"])

    # B2. 量化可用性闸门 L1（2026-10-09）：整片/无量纲单值的特征（表皮细胞/气孔）恒不产
    #     primary_value_um 并打 coarse_quant；对照：细长/单粒特征（针晶）不受闸门影响。
    check_bool("量化闸门: 针晶不受影响（产量化值、无 coarse_quant）",
               rec["primary_value_um"] is not None and "coarse_quant" not in rec["flags"])
    inst_ep = type("I", (), {"image_stem": "t_ep", "instance_id": 1,
                             "mask": m, "label": "epidermis_cell", "meta": {}})()
    rec_ep = measure_instance(inst_ep, calib)
    check_bool("量化闸门: 表皮细胞不产 primary_value_um 且打 coarse_quant",
               rec_ep["primary_value_um"] is None and rec_ep["primary_dimension"] is None
               and "coarse_quant" in rec_ep["flags"])


# ------------------------------------------------------------------ C: 标定链路
def run_calibration_tests():
    print("=" * 72)
    print("C. 标定链路（分割交付清单读取 + 比例尺探测对照 + 缺标定报错门槛）")
    tmp = os.path.join(BASE, "tmp_selftest_calib")
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)

    # C1: 交付清单按源图 stem 命中（清单名 == <stem>__<feature>__measure_manifest.csv）
    with open(os.path.join(tmp, "W20161479__cluster_crystal__measure_manifest.csv"),
              "w", encoding="utf-8-sig", newline="") as f:
        f.write("instance_id,inst_mask,px_per_um\n")
        f.write("1,W20161479__cluster_crystal__inst1.png,11.700\n")
    meta = load_delivery_manifest(tmp)
    row = meta.get("W20161479") or {}
    check_bool("交付清单按 stem 命中且读到 px_per_um",
               abs(float(row.get("px_per_um", 0)) - 11.7) < 1e-9)

    # C2: 比例尺探测对照工具（合成烧录比例尺：180px 暗线于右下角）
    img = np.full((600, 800), 230, np.uint8)
    cv2.rectangle(img, (590, 560), (590 + 180, 564), 30, -1)
    check("比例尺线段探测(px)", float(detect_scale_bar_px(img)), 180.0, 2.0)

    # C3: 缺 px_per_um → measure.py 报错退出（不再降级为 uncalibrated）
    img2 = np.zeros((120, 120), np.uint8)
    cv2.circle(img2, (60, 60), 40, 255, -1)
    cv2.imencode(".png", img2)[1].tofile(os.path.join(tmp, "nomiss__vessel__inst1.png"))
    p = subprocess.run(
        [sys.executable, os.path.join(BASE, "measure.py"),
         "--masks", tmp, "--out-prefix", os.path.join(tmp, "out")],
        capture_output=True, text=True, encoding="utf-8", errors="replace")
    check_bool("缺 px_per_um → 退出码 2", p.returncode == 2)
    check_bool("报错点名缺项源图 stem", "nomiss" in (p.stderr or ""))
    check_bool("未产出测量行", not os.path.exists(os.path.join(tmp, "out_measurements.csv")))

    # C4: mask 加载器（实例编号 PNG / 每实例 PNG / 中文路径）
    if os.path.isdir(DEMO_MASK_DIR):
        insts = load_instances(DEMO_MASK_DIR)
        check_bool(f"demo_masks 目录解析（{len(insts)} 实例）", len(insts) >= 7)
        labels = {i.label for i in insts}
        check_bool("文件名 feature 标签解析", "needle_crystal" in labels and "vessel" in labels)

    shutil.rmtree(tmp, ignore_errors=True)


# ------------------------------------------------------------------ 落盘 demo
def write_demo_assets():
    """落盘 demo_masks/ + demo_manifest.csv（端到端演示物料）。

    px_per_um 由 mask 目录同名的 `*_measure_manifest.csv` 携带（本函数一并落盘），
    供 measure.py 直接跑通端到端（scale_table.csv 已停用）。
    """
    if os.path.isdir(DEMO_MASK_DIR):
        shutil.rmtree(DEMO_MASK_DIR)
    os.makedirs(DEMO_MASK_DIR)

    scenes = [
        ("demo01_circle", scene_circle, 10.0, "大黄", "cluster_crystal", "test"),
        ("demo02_ellipse", scene_ellipse, 10.0, "大黄", "starch_grain", "test"),
        ("demo03_star", scene_star, 2.0, "大黄", "cluster_crystal", "test"),
        ("demo04_needle", scene_needle, 10.0, "半夏", "needle_crystal", "test"),
        ("demo05_fiber", scene_curved_fiber, 2.0, "板蓝根", "fiber", "train"),
        ("demo06_vessel", scene_vessel, 10.0, "大黄", "vessel", "test"),
        ("demo07_compound", scene_compound_starch, 2.0, "大黄", "starch_grain", "val"),
    ]
    with open(os.path.join(DEMO_MASK_DIR, "demo__measure_manifest.csv"),
              "w", encoding="utf-8-sig", newline="") as fs, \
         open(os.path.join(BASE, "demo_manifest.csv"), "w", encoding="utf-8-sig", newline="") as fm:
        fs.write("instance_id,inst_mask,px_per_um,herb,object_label\n")
        fm.write("image_path,root,folder_name,granule,herb,feature_label,ext,width,height,"
                 "group_key,suggested_split,needs_review\n")
        for stem, fn, s, herb, feat, split in scenes:
            m, label, _gt = fn()
            cv2.imencode(".png", (m * 255).astype(np.uint8))[1].tofile(
                os.path.join(DEMO_MASK_DIR, f"{stem}__{label}__inst1.png"))
            fs.write(f"1,{stem}__{label}__inst1.png,{s},{herb},{feat}\n")
            fm.write(f"D:\\demo\\{stem}.jpg,demo,{stem},feature,{herb},{feat},jpg,1,1,"
                     f"{herb}-seq,{split},\n")
    print(f"\n[info] demo 物料已落盘: {DEMO_MASK_DIR}\\（含 demo__measure_manifest.csv）+ demo_manifest.csv")


def main():
    run_descriptors_tests()
    run_conversion_tests()
    run_calibration_tests()
    write_demo_assets()          # 放最后：避免测试期间覆盖 demo 物料
    print("=" * 72)
    n_ok, n_all = sum(RESULTS), len(RESULTS)
    print(f"自测结果: {n_ok}/{n_all} PASS → {'✅ 全部通过' if n_ok == n_all else '❌ 存在失败项'}")
    return 0 if n_ok == n_all else 1


if __name__ == "__main__":
    sys.exit(main())
