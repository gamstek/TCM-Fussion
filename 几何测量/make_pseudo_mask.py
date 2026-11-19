# -*- coding: utf-8 -*-
"""
make_pseudo_mask.py — 伪 mask 生成器（几何测量环节的冒烟测试专用）

⚠️ 这不是分割模块：只在实例分割（第 3 步）上线前，用传统阈值法从原图造
   粗糙伪 mask，用于冒烟验证几何测量链路（标定 → 测量 → μm 换算）。
   正式测量必须等分割 mask（mask 是尺寸测量的唯一真值）。

内置两条项目已踩坑记录：
  * 中文路径 imread 封装（np.fromfile + imdecode）；
  * 明视野亮背景下 Otsu 会把整片背景连成"假溢出轮廓" → 剔除贴图像四缘的组件。

用法:
  python make_pseudo_mask.py --images dataset/显微镜像/大黄草酸钙簇晶 \
      --out-dir pseudo_masks --max-images 10
输出: <out-dir>/<图像stem>__<feature>__inst<k>.png（二值，每实例一张）
"""
import os
import sys
import re
import glob
import argparse

import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from geomlib.imutil import imread, imwrite

# 文件夹名 → feature_id 关键词映射（feature_taxonomy 命名习惯）
KEYWORDS = [
    ("草酸钙簇晶", "cluster_crystal"), ("簇晶", "cluster_crystal"),
    ("方晶", "single_crystal"), ("针晶", "needle_crystal"),
    ("导管", "vessel"), ("石细胞", "stone_cell"), ("淀粉粒", "starch_grain"),
    ("菌丝", "hypha"), ("纤维", "fiber"), ("菊糖", "inulin"),
    ("乳管", "secretory_tissue"), ("油管", "secretory_tissue"),
]


def guess_feature(name):
    for kw, fid in KEYWORDS:
        if kw in name:
            return fid
    return None


def border_touch(mask):
    """组件是否贴图像四缘（Otsu 假溢出轮廓的判别特征）。"""
    h, w = mask.shape
    return bool(mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any())


def pseudo_instances(img_bgr, min_area=150):
    """Otsu（暗目标）→ 去贴缘组件 → 取面积前 N 的组件，返回 0/1 mask 列表。"""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    # 明视野: 目标偏暗 → 对灰度做 Otsu 取暗侧
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(bw)
    comps = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < min_area:
            continue
        comp = (lab == i).astype(np.uint8)
        if border_touch(comp):
            continue                      # 剔除假溢出轮廓（项目坑 #5）
        comps.append((area, comp))
    comps.sort(key=lambda t: -t[0])
    return [c for _, c in comps[:8]]


def main():
    ap = argparse.ArgumentParser(description="伪 mask 生成（冒烟测试专用，非正式分割）")
    ap.add_argument("--images", required=True, help="原图目录（如 显镜像/大黄草酸钙簇晶）")
    ap.add_argument("--out-dir", default="pseudo_masks")
    ap.add_argument("--max-images", type=int, default=10)
    ap.add_argument("--min-area", type=int, default=150)
    ap.add_argument("--filter", default=None,
                    help="只处理文件名匹配该正则的图（建议挑 stem 全库唯一的图，如 ^W2016）")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    feat = guess_feature(os.path.basename(os.path.normpath(args.images)))
    rx = re.compile(args.filter) if args.filter else None
    files = sorted(glob.glob(os.path.join(args.images, "*.jpg")) +
                   glob.glob(os.path.join(args.images, "*.tif")))
    if rx:
        files = [f for f in files if rx.search(os.path.basename(f))]
    n_inst = n_img = 0
    for fp in files[:args.max_images]:
        img = imread(fp)
        if img is None:
            print(f"[skip] 读不到 {fp}")
            continue
        stem = os.path.splitext(os.path.basename(fp))[0]
        comps = pseudo_instances(img, args.min_area)
        if not comps:
            continue
        for k, comp in enumerate(comps, 1):
            name = f"{stem}__{feat or 'generic'}__inst{k}.png"
            imwrite(os.path.join(args.out_dir, name), comp * 255)
            n_inst += 1
        n_img += 1
    print(f"[done] {n_img} 张图 → {n_inst} 个伪实例 mask → {args.out_dir}/"
          f"（feature={feat or 'generic'}）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
