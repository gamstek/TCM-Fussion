# -*- coding: utf-8 -*-
"""
render_prelabels.py — 渲染预标注叠加图供人工验收
绿框=high(可采信) 橙框=medium(需复核) 灰框=疑似待定；标注类别/置信度/μm(用标定表换算)
输出: results/detection/prelabel_review/<stem>.jpg
"""
import json
import os

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


from contracts import PrelabelRecord  # noqa: E402 第 ⑤ 段契约（prelabel 记录）
from ls_common import DET_RESULTS, PRELABEL_DIR, imread_cn, load_config, prelabel_scale  # noqa: E402

OUT_DIR = os.path.join(DET_RESULTS, "prelabel_review")
os.makedirs(OUT_DIR, exist_ok=True)

COLOR = {"high": (0, 200, 0), "medium": (0, 165, 255), "疑似待定": (160, 160, 160)}


def main():
    cfg = load_config()
    font = ImageFont.truetype(r"C:\Windows\Fonts\msyh.ttc", 30)
    n_img, n_box = 0, 0
    used_names = set()
    for fn in sorted(os.listdir(PRELABEL_DIR)):
        if not fn.endswith(".json"):
            continue
        with open(os.path.join(PRELABEL_DIR, fn), encoding="utf-8") as f:
            pre = PrelabelRecord.from_dict(json.load(f))
        img = imread_cn(pre.image_path)
        if img is None:
            continue
        # px/μm 取自 prelabel JSON 的 `scale` 字段（检测阶段双引擎标定；离线标定表已停用）
        ppu, _st = prelabel_scale(pre)
        tags = []
        for b in pre.objects:
            x1, y1, x2, y2 = [int(v) for v in b.tight_bbox]
            color = COLOR.get(b.label, COLOR.get(b.confidence, (0, 165, 255)))
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
            size_um = None  # VLM 尺寸自估已弃用，一律用框径×标定换算
            box_um = ""
            if ppu:
                long_px = max(x2 - x1, y2 - y1)
                box_um = f"框径{long_px / ppu:.0f}μm"
            conf = "高" if b.confidence == "high" else "复核"
            tags.append((x1, y1, y2, color,
                         f'{b.label} [{conf}] {box_um}'))
        pim = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pim)
        for x1, y1, y2, color, text in tags:
            ty = y1 - 38 if y1 > 50 else y2 + 8
            bb = draw.textbbox((x1, ty), text, font=font)
            draw.rectangle([bb[0] - 4, bb[1] - 2, bb[2] + 4, bb[3] + 2], fill=(255, 255, 255))
            draw.text((x1, ty), text, font=font, fill=tuple(reversed(color)))
        out = cv2.cvtColor(np.array(pim), cv2.COLOR_RGB2BGR)
        stem = os.path.splitext(os.path.basename(pre.image_path))[0]
        feat = pre.feature_label_folder
        out_name = f"{feat}_{stem}_预标注.jpg" if feat else f"{stem}_预标注.jpg"
        if out_name in used_names:  # 不同目录同名图：路径短哈希消歧
            import hashlib
            h = hashlib.md5(str(pre.image_path).encode("utf-8")).hexdigest()[:6]
            out_name = out_name.replace("_预标注.jpg", f"_{h}_预标注.jpg")
        used_names.add(out_name)
        cv2.imencode(".jpg", out)[1].tofile(os.path.join(OUT_DIR, out_name))
        n_img += 1
        n_box += len(pre.objects)
    print(f"渲染 {n_img} 图 / {n_box} 框 → {OUT_DIR}")


if __name__ == "__main__":
    main()
