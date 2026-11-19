# -*- coding: utf-8 -*-
"""实验C：多样本扩大验证 —— 4 个样本 × n=3（灰度 prompt）→ 软图平均 → 扫操作点。

目的：实验B 只在单样本上做，无法判断"低操作点"能否泛化。本实验：
  * 3 个已知失败样本（马2_b4 / 马3_b3 / 796_b0，目标本体仅 91~296px、低对比）
  * 1 个较清晰对照样本（1_b0，生产单次 IoU 已达 0.73）
    —— 用来反向验证"低操作点会不会把本来就画得对的样本过度扩张"
  * 基线：生产产物（二值 prompt + 生产策略）的 mask 一并度量

关键结论（见报告）：
  * 失败样本 3/3 一致改善：IoU 0.23/0.33/0.36 → 0.66/0.52/0.54（操作点 0.05）
  * 对照样本无过度扩张：0.05 下 IoU 0.728→0.742、占比仅 +5.6%
  * 中间灰阶在 12 次运行中全部出现 → 灰度提示词机制普适

用法：
  python exp_multisample_validate.py                  # 复用已有产物（不花额度）
  python exp_multisample_validate.py --force          # 重新调用 12 次（消耗额度）
  python exp_multisample_validate.py --no-api         # 只允许复用；缺产物即报错（防误扣费）
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import igm_exp_common as C  # noqa: E402
import igm_segment as ig   # noqa: E402

SAMPLES = ["草酸钙簇晶_马2_b4", "草酸钙簇晶_马3_b3", "草酸钙簇晶_796_b0", "草酸钙簇晶_1_b0"]
N = 3
LEVELS = (0.05, 0.10, 0.20, 0.30, 0.50)

ap = argparse.ArgumentParser(description="实验C：多样本 × n 次扩样验证")
ap.add_argument("--n", type=int, default=N)
ap.add_argument("--force", action="store_true", help="重新调用 API")
ap.add_argument("--no-api", action="store_true", help="只复用已有产物，缺产物即报错")
args = ap.parse_args()

C.print_usage_warning(len(SAMPLES) * args.n)
rows = []
for stem in SAMPLES:
    pm = C.prod_meta(stem)
    img = pm["image"]
    h, w = pm["image_hw"]
    box = C.box_from_crop(C.load_crop_json(img), h, w)
    base = C.hard_mask(os.path.join(C.PROD_DIR, stem))          # 生产基线（二值 prompt）
    soft, per = [], []
    for k in range(1, args.n + 1):
        out = os.path.join(C.EXP_C, stem, f"run{k}")
        exists = os.path.exists(os.path.join(out, "meta.json"))
        if args.no_api and not exists:
            sys.exit(f"[FAIL] --no-api 模式下缺产物: {out}")
        ok, m = C.run_igm(img, C.gray_prompt(), out, reuse=not args.force)
        if not ok:
            print(f"[warn] {stem} run{k} 失败，跳过")
            continue
        s, mm = C.soft_map(out, h, w)
        soft.append(s)
        per.append((mm["area_ratio"], C.iou(C.hard_mask(out), box)))
    if not soft:
        continue
    avg = np.mean(soft, axis=0)
    try:                                     # 概率图只是便利产物，被看图软件占用也不该中断实验
        ig.write_img(os.path.join(C.EXP_C, stem, "soft_avg_prob.png"),
                     (avg * 255).astype(np.uint8), ".png")
    except RuntimeError as e:
        print(f"[warn] 概率图写盘失败（不影响指标）: {e}")
    print(f"\n=== {stem} {w}x{h} 检测框占比="
          f"{'n/a' if box is None else format(box.mean(), '.3f')} ===")
    print(f"  生产单次(二值prompt): 占比={float((base > 0).mean()):.3f} "
          f"IoU={C.iou(base, box):.3f}")
    for k, (a, i_) in enumerate(per, 1):
        print(f"  灰度prompt run{k}: 占比={a:.3f} IoU={i_:.3f}")
    ious = []
    for thr in LEVELS:
        e = avg >= thr
        i_ = C.iou(e, box)
        ious.append(i_)
        print(f"    软图平均 操作点 {thr:.2f}: 占比={float(e.mean()):.3f} IoU={i_:.3f} "
              f"覆盖框={C.cover(e, box):.3f}")
    rows.append((stem, box, C.iou(base, box), ious))

print("\n\n===== 汇总 IoU（生产单次 | " + " | ".join(f"{t:.2f}" for t in LEVELS) + "）=====")
for stem, box, b, ious in rows:
    bx = "n/a" if box is None else f"{box.mean():.3f}"
    print(f"  {stem:22s} 框={bx}  {b:.3f} | " + " | ".join(f"{v:.3f}" for v in ious))