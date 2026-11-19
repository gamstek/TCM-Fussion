# -*- coding: utf-8 -*-
"""mask_io — 实例分割 Mask 的统一读取层（输入契约）。

支持三种输入（measure.py 的 --masks 参数按类型自动识别）：

1) 实例编号 PNG：一张灰度 PNG，0=背景，1..255=实例编号。
   文件名 `<图像stem>.png` 或 `<图像stem>__<feature_id>.png`。
2) 目录：混装上面两种 PNG，外加每实例一张的二值 PNG：
   `<图像stem>__<feature_id>__inst<k>.png`（0 背景 / >0 前景）。
3) COCO JSON：images + annotations（segmentation 支持 polygon 与
   未压缩 RLE {"counts":[...]}），categories 提供特征名。

约定：
  * mask 逻辑值 0/1，uint8，与 crop 同尺寸（同宽高即可，不做对齐插值）；
    上游按《分割方案》§1.3 交付 **crop 局部尺寸** mask——测量量在 crop 像素上算，
    bbox_x/bbox_y 即 crop 坐标系；
    亦接受放回原图全 0 背景的整图尺寸 mask：两者像素网格分辨率一致（crop 未重采样），
    px_per_um 相同、物理测量（μm）结果不变，仅 bbox_x/bbox_y 落入原图坐标系；
  * **instance_id 是稳定身份**：每实例 PNG 取文件名中的 `<k>`（`__inst<k>`），
    实例编号 PNG 取像素编号，COCO 取 annotation id——与交付文件名一一对应，
    换目录/换批次重测稳定不漂移（Agent 的 `instance_refs`、conf join 键都靠它）；
    同一源图内 `<k>` 必须唯一（分割交付契约：同一源图的多 crop 跨 feature 连续编号）；
  * 一个实例允许多个连通域（针晶束、成束纤维、复粒等）；
  * feature 标签只从文件名 / COCO category 提取，供 per-feature 测量剖面与
    范围核查使用，缺失时回退到通用（generic）测量剖面。
"""
import os
import re
import json
from dataclasses import dataclass, field

import numpy as np
import cv2

from .imutil import imread


@dataclass
class InstanceMask:
    image_stem: str                 # 对应原图 stem（不含扩展名）
    instance_id: int
    mask: np.ndarray                # uint8 0/1
    label: str | None = None        # feature_id（如 cluster_crystal），可为空
    meta: dict = field(default_factory=dict)


def _decode_png_mask(path):
    m = imread(path, cv2.IMREAD_UNCHANGED)
    if m is None:
        return None
    if m.ndim == 3:                                # 彩色存的 mask → 取灰度
        m = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)
    return m


def _instances_from_id_png(arr, stem, label_hint=None):
    ids = [int(v) for v in np.unique(arr) if v > 0]
    out = []
    for k, v in enumerate(ids, 1):
        out.append(InstanceMask(stem, v, (arr == v).astype(np.uint8),
                                label_hint, {"format": "id_png"}))
    return out


def _parse_stem_parts(stem):
    """`img01__cluster_crystal__inst2` → (img01, cluster_crystal)"""
    parts = stem.split("__")
    base = parts[0]
    label = None
    for p in parts[1:]:
        if p.startswith("inst") and p[4:].isdigit():
            continue
        label = p
        break
    return base, label


def _inst_k(stem):
    """文件名尾部的 `__inst<k>` → int k（稳定身份）；无则 None。"""
    m = re.search(r"__inst(\d+)$", stem)
    return int(m.group(1)) if m else None


def load_instances(masks_path):
    """按输入类型自动识别并展开为 InstanceMask 列表（附 (stem,id) 唯一性护栏）。"""
    out = _load_instances_raw(masks_path)
    # 每实例 PNG 契约要求同一源图跨 feature 连续编号；若各 feature 都从 inst1 重启，
    # 测量表主键 (image_stem, instance_id) 与叠加图文件名都会冲突（join 歧义、互相覆盖）。
    seen, clashes = {}, []
    for i in out:
        key = (i.image_stem, i.instance_id)
        if key in seen and seen[key] != i.label:
            clashes.append(key)
        seen.setdefault(key, i.label)
    if clashes:
        uniq = sorted({f"{s}#inst{i}" for s, i in clashes})
        print(f"[warn] (image_stem, instance_id) 跨 feature 冲突 {len(clashes)} 处——"
              f"请令导出侧对同一源图跨 feature 连续编号：{uniq[:5]}"
              f"{' …' if len(uniq) > 5 else ''}")
    return out


def _load_instances_raw(masks_path):
    if os.path.isdir(masks_path):
        files = []
        for fn in sorted(os.listdir(masks_path)):
            if fn.lower().endswith((".png", ".bmp", ".tif", ".tiff")):
                files.append(os.path.join(masks_path, fn))
        out = []
        for fp in files:
            stem = os.path.splitext(os.path.basename(fp))[0]
            arr = _decode_png_mask(fp)
            if arr is None:
                continue
            base, label = _parse_stem_parts(stem)
            if re.search(r"__inst\d+$", stem):              # 每实例二值图（支持 inst1 / inst01 补零）
                # instance_id = 文件名 inst<k>（交付契约的稳定身份，不用遍历序号）
                out.append(InstanceMask(base, _inst_k(stem), (arr > 0).astype(np.uint8),
                                        label, {"format": "per_instance", "file": fp}))
            elif arr.max() > 1:                            # 实例编号图
                out += _instances_from_id_png(arr, base, label)
            else:                                          # 单实例二值图
                out.append(InstanceMask(base, 1, (arr > 0).astype(np.uint8),
                                        label, {"format": "binary_png", "file": fp}))
        return out

    if os.path.isfile(masks_path):
        ext = os.path.splitext(masks_path)[1].lower()
        if ext == ".json":
            return _load_coco(masks_path)
        if ext in (".png", ".bmp", ".tif", ".tiff"):
            stem = os.path.splitext(os.path.basename(masks_path))[0]
            arr = _decode_png_mask(masks_path)
            if arr is None:
                return []
            base, label = _parse_stem_parts(stem)
            if arr.max() > 1:
                return _instances_from_id_png(arr, base, label)
            k = _inst_k(stem)
            return [InstanceMask(base, k if k is not None else 1,
                                 (arr > 0).astype(np.uint8), label,
                                 {"format": "binary_png"})]
    raise ValueError(f"--masks 不支持的输入: {masks_path}（须为 mask PNG / 目录 / COCO json）")


def _rle_decode(rle, h, w):
    """未压缩 COCO RLE（column-major counts）→ 0/1 uint8。"""
    flat = np.zeros(h * w, np.uint8)
    pos, val = 0, 0
    for c in rle["counts"]:
        c = int(c)
        if val:
            flat[pos:pos + c] = 1
        pos += c
        val = 1 - val
    return flat.reshape((w, h)).T          # column-major → row-major


def _poly_to_mask(polys, h, w):
    m = np.zeros((h, w), np.uint8)
    for poly in polys:
        pts = np.asarray(poly, np.float64).reshape(-1, 2)
        cv2.fillPoly(m, [np.round(pts).astype(np.int32)], 1)
    return m


def _load_coco(path):
    with open(path, encoding="utf-8") as f:
        coco = json.load(f)
    cat = {c["id"]: c.get("name") for c in coco.get("categories", [])}
    imgs = {im["id"]: im for im in coco.get("images", [])}
    out = []
    for ann in coco.get("annotations", []):
        im = imgs.get(ann["image_id"])
        if im is None:
            continue
        h, w = im["height"], im["width"]
        stem = os.path.splitext(os.path.basename(im.get("file_name", "")))[0]
        seg = ann.get("segmentation")
        if isinstance(seg, dict):                          # RLE
            m = _rle_decode(seg, h, w)
        elif isinstance(seg, list) and seg:                # polygon 列表
            m = _poly_to_mask(seg, h, w)
        else:
            continue
        out.append(InstanceMask(stem, int(ann.get("id", len(out) + 1)), m,
                                cat.get(ann.get("category_id")), {"format": "coco"}))
    return out
