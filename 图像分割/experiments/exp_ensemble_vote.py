# -*- coding: utf-8 -*-
"""实验B：同一原图多次采样 → 对比两种聚合方式。

方式1「先二值化再投票」：把各次的 0/255 mask 求和，阈值 255*x/n（x=1 任一命中 … x=n 一致同意）。
方式2「软概率图平均」（思路3 结合）：各次软图回缩到原图尺寸后取平均 → 扫操作点。

结论（见 docs/IGM分割_灰度提示与集成采样实验报告.md）：
  方式1 基本无效且档位方向与直觉相反——模型是**系统性画小（偏差）**，取交集只会更小，
  x 越大越差；只有 x=1（并集）≈ 单次最好。
  方式2 明显更优：低操作点能把被 Otsu 切掉的过渡带收回，且单轮噪声被平均掉。

用法：
  python exp_ensemble_vote.py                  # 复用已有产物（含 A 的那次）+ 离线分析
  python exp_ensemble_vote.py --n 4 --force    # 重新调用 4 次
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import igm_exp_common as C  # noqa: E402
import igm_segment as ig   # noqa: E402

ap = argparse.ArgumentParser(description="实验B：多采样聚合方式对比")
ap.add_argument("--stem", default="草酸钙簇晶_马2_b4")
ap.add_argument("--n", type=int, default=4, help="新增采样次数（实验B 用 4）")
ap.add_argument("--force", action="store_true", help="重新调用 API")
args = ap.parse_args()

stem = args.stem
pm = C.prod_meta(stem)
img = pm["image"]
h, w = pm["image_hw"]
box = C.box_from_crop(C.load_crop_json(img), h, w)

C.print_usage_warning(args.n)
# 参与聚合的样本：实验A 的那次（同提示词、同图，合法样本）+ 本次 run1..n
# —— 报告 §3.2 的表即按 n=5（A 的 1 次 + B 的 4 次）给出
dirs = []
a_dir = os.path.join(C.EXP_A, stem)
if os.path.exists(os.path.join(a_dir, "meta.json")):
    dirs.append(a_dir)
for k in range(1, args.n + 1):
    out = os.path.join(C.EXP_B, f"run{k}")
    ok, _ = C.run_igm(img, C.gray_prompt(), out, reuse=not args.force)
    if ok:
        dirs.append(out)
    else:
        print(f"[warn] run{k} 失败，跳过")
if not dirs:
    sys.exit("[FAIL] 无有效 run")

hard, soft = [], []
print(f"\n=== 实验B：{stem} {w}x{h} 检测框占比="
      f"{'n/a' if box is None else format(box.mean(), '.3f')} ===")
for d in dirs:
    hh = C.hard_mask(d)
    s, m = C.soft_map(d, h, w)
    hard.append(hh)
    soft.append(s)
    print(f"  {os.path.basename(d)}: 阈值={m['threshold']:.0f} invert={m['invert']!s:5s} "
          f"占比={m['area_ratio']:.3f} IoU={C.iou(hh, box):.3f}")
n = len(hard)
areas = [float(hh.mean()) for hh in hard]
print(f"  单次占比区间 [{min(areas):.3f}, {max(areas):.3f}] → "
      f"{'provider 非确定（有方差可平均）' if max(areas) - min(areas) > 1e-6 else '各次相同'}")

print(f"\n[方式1] 先二值化再投票（阈值 255*x/{n}）")
votes = np.zeros((h, w), np.int32)
for hh in hard:
    votes += hh.astype(np.int32)
for x in range(1, n + 1):
    e = votes >= x
    tag = "任一命中" if x == 1 else ("一致同意" if x == n else ("过半" if x * 2 > n else "少数即入"))
    print(f"  x={x}/{n} ({tag}): 占比={float(e.mean()):.3f} IoU={C.iou(e, box):.3f}")

print(f"\n[方式2] 软概率图平均 → 扫操作点")
avg = np.mean(soft, axis=0)
for thr in (0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60):
    e = avg >= thr
    print(f"  操作点 {thr:.2f}: 占比={float(e.mean()):.3f} IoU={C.iou(e, box):.3f} "
          f"覆盖框={C.cover(e, box):.3f}")
try:                                         # 概率图只是便利产物，被占用也不该中断实验
    ig.write_img(os.path.join(C.EXP_B, "soft_avg_prob.png"), (avg * 255).astype(np.uint8), ".png")
    print(f"\n留档: {os.path.join(C.EXP_B, 'soft_avg_prob.png')}（软图平均=经验概率图）")
except RuntimeError as e:
    print(f"\n[warn] 概率图写盘失败（不影响指标）: {e}")