# -*- coding: utf-8 -*-
"""实验A：把 prompt 从"纯白/纯黑二值"改为"允许灰度概率过渡"，验证机制是否成立。

问题背景：低对比小目标（目标本体仅 91~296px、边界处仅 9~12 灰阶差）上，二值 prompt 的
生成图退化为"74~90% 纯黑 + 10~26% 纯白、中间灰阶 <0.2%"，mask 明显偏小。

本实验只改一处：PROMPT 的最后一条输出格式（生产版禁止灰度渐变，实验版允许）。
判定标准（硬指标）：中间灰阶(40~210)占比是否显著上升；辅以 corr(生成图, 原图)、
两种极性下的阈值扫描（看"降低操作点"是否把过渡带收回来）。

用法：
  python exp_gray_prompt.py                 # 复用已有产物，只做分析
  python exp_gray_prompt.py --force         # 重新调用 API（消耗额度）
  python exp_gray_prompt.py --stem 草酸钙簇晶_马3_b3
"""
import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import igm_exp_common as C  # noqa: E402
import igm_segment as ig   # noqa: E402

ap = argparse.ArgumentParser(description="实验A：灰度概率提示词机制验证")
ap.add_argument("--stem", default="草酸钙簇晶_马2_b4", help="样本 stem（默认最差样本）")
ap.add_argument("--force", action="store_true", help="重新调用 API（否则复用已有产物）")
args = ap.parse_args()

stem = args.stem
pm = C.prod_meta(stem)
img = pm["image"]
h, w = pm["image_hw"]
cj = C.load_crop_json(img)
box = C.box_from_crop(cj, h, w)
out_dir = os.path.join(C.EXP_A, stem)

C.print_usage_warning(1)
ok, meta = C.run_igm(img, C.gray_prompt(), out_dir, reuse=not args.force)
if not ok:
    sys.exit("[FAIL] 生成失败")

# ---- 分析 ----
soft, m = C.soft_map(out_dir, h, w)
gen = ig.imread_any(os.path.join(out_dir, "gen_raw.png"))
g = cv2.cvtColor(gen, cv2.COLOR_BGR2GRAY) if gen.ndim == 3 else gen
roi = ig._crop_content(g, m["letterbox"])
hist = np.histogram(roi, bins=[0, 40, 90, 160, 210, 256])[0] / roi.size
orig = ig.imread_any(img)
oc = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY) if orig.ndim == 3 else orig
corr = float(np.corrcoef(soft.ravel(), cv2.resize(oc, (w, h),
                                                 interpolation=cv2.INTER_AREA).ravel())[0, 1])

print(f"\n=== 实验A：{stem} {w}x{h} ===")
print(f"生成图灰阶分布[0-40,40-90,90-160,160-210,210-255] = "
      f"{[round(float(x), 3) for x in hist]}")
print(f"**中间灰阶(40~210)占比 = {C.mid_tone_ratio(roi):.3f}**"
      f"（对照：二值 prompt 三例均 <0.002）")
print(f"corr(生成图, 原图) = {corr:+.3f}（对照：二值 prompt 为 +0.11/−0.17/−0.28）")
if box is not None:
    print(f"检测框占比 = {box.mean():.3f}")
for inv in (False, True):
    cells = []
    for th in (32, 64, 96, 128, 160):
        mm, info = ig.generated_to_mask(gen, None, (h, w), m["letterbox"], threshold=th,
                                        invert=inv, min_area=ig.MIN_AREA_DEFAULT)
        cells.append(f"t={th}:{info['area_ratio']:.3f}/IoU{C.iou(mm > 0, box):.2f}")
    print(f"  invert={inv!s:5s} " + "  ".join(cells))
print(f"  生产策略自动判定: invert={m['invert']} status={m['status']} area={m['area_ratio']}")
print(f"  产物: {out_dir}")