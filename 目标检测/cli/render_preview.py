# -*- coding: utf-8 -*-
"""render_preview.py — 把 YOLO 检测数据集的标注框画回原图，生成压缩预览图（仅供目检）

**为什么需要**：数据集入库后要快速翻看「框画得对不对」，但 `images/` 是 GB 级原图（含 tif）。
本脚本按目录批量渲染 `preview/<同名>.jpg`：先缩到长边 `--max-side`，再按缩放后的坐标画框，
JPEG 压缩到约原图的 1/50~1/100。预览图**只用于目检，不属于训练数据、不入库**（.gitignore 已排除）。

输入：`<dataset-dir>/images/*` 与 `<dataset-dir>/labels/*.txt`（YOLO 归一化 `<cls> <cx> <cy> <w> <h>`）
输出：`<dataset-dir>/preview/<stem>.jpg`

用法:
  python cli\\render_preview.py --dataset-dir "dataset\\目标检测标注\\草酸钙簇晶"
  python cli\\render_preview.py --dataset-dir <dir> --max-side 512 --quality 75
"""
import argparse
import os

import cv2

from ls_common import PROJECT_ROOT, imread_cn  # noqa: E402 中文路径安全读图（唯一实现）

IMG_EXT = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp")
BOX_COLOR = (60, 60, 235)   # BGR 红
BOX_THICK = 3


def render(img, label_path, max_side):
    """缩到长边 max_side（仅缩不放）后按**缩放后坐标**画框（先缩再画，线宽才稳定）。"""
    h, w = img.shape[:2]
    s = min(1.0, max_side / max(h, w))
    if s < 1:
        img = cv2.resize(img, (int(w * s), int(h * s)))
    hh, ww = img.shape[:2]
    boxes = 0
    if os.path.exists(label_path):
        with open(label_path, encoding="utf-8") as f:
            for line in f:
                parts = line.split()
                if len(parts) != 5:
                    continue                       # 空 label（负样本）自然跳过
                _, cx, cy, bw, bh = (float(v) for v in parts)
                x1, y1 = int((cx - bw / 2) * ww), int((cy - bh / 2) * hh)
                x2, y2 = int((cx + bw / 2) * ww), int((cy + bh / 2) * hh)
                cv2.rectangle(img, (x1, y1), (x2, y2), BOX_COLOR, BOX_THICK)
                boxes += 1
    return img, boxes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", required=True, help="数据集目录（含 images/ 与 labels/）")
    ap.add_argument("--max-side", type=int, default=640, help="预览图长边上限（默认 640）")
    ap.add_argument("--quality", type=int, default=80, help="JPEG 质量（默认 80）")
    args = ap.parse_args()

    root = (args.dataset_dir if os.path.isabs(args.dataset_dir)
            else os.path.join(PROJECT_ROOT, args.dataset_dir))
    img_dir, lbl_dir = os.path.join(root, "images"), os.path.join(root, "labels")
    out_dir = os.path.join(root, "preview")
    if not os.path.isdir(img_dir):
        raise SystemExit(f"找不到 images/：{img_dir}")
    os.makedirs(out_dir, exist_ok=True)

    names = [f for f in sorted(os.listdir(img_dir)) if f.lower().endswith(IMG_EXT)]
    done = fail = n_box = 0
    total_bytes = 0
    for fn in names:
        img = imread_cn(os.path.join(img_dir, fn))
        if img is None:
            print(f"  [FAIL 读取] {fn}")
            fail += 1
            continue
        img, boxes = render(img, os.path.join(lbl_dir, os.path.splitext(fn)[0] + ".txt"),
                            args.max_side)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, args.quality])
        if not ok:
            print(f"  [FAIL 编码] {fn}")
            fail += 1
            continue
        # ndarray.tofile 在 Windows 上对中文路径可能 OSError，统一走文件对象写
        with open(os.path.join(out_dir, os.path.splitext(fn)[0] + ".jpg"), "wb") as f:
            f.write(buf.tobytes())
        done += 1
        n_box += boxes
        total_bytes += len(buf)

    print(f"渲染 {done} 张预览（失败 {fail}，画框 {n_box} 个）→ {out_dir}")
    print(f"总体积 {total_bytes / 1048576:.1f} MB（长边 {args.max_side} / JPEG q{args.quality}）")


if __name__ == "__main__":
    main()
