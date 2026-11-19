# -*- coding: utf-8 -*-
"""
build_tif_proxies.py — 为 417 任务中的 tif 图生成浏览器可渲染的 jpg 代理
- 浏览器原生不渲染 TIFF：LS 表格/标注页打不开 .tif 任务图（服务端 200 image/tiff 正常）
- 代理输出: results/detection/ls_view/<root>/<folder>/<stem>.jpg（镜像原目录结构，q=90）
- 幂等：已存在的代理跳过；--force 重建
- 结束后自动在项目上注册 ls_view 本地存储
用法:
  python build_tif_proxies.py            # 只处理 prelabels 里引用到的 tif
  python build_tif_proxies.py --force    # 强制重建全部代理
"""
import argparse
import json
import os


from contracts import PrelabelRecord  # noqa: E402 第 ⑤ 段契约（prelabel 记录）
from ls_common import (DET_RESULTS, DATASET, LS_VIEW_SUBDIR, PRELABEL_DIR, canon_path,  # noqa: E402
                       imread_cn, load_config, ls_api)

VIEW_ROOT = os.path.join(DET_RESULTS, LS_VIEW_SUBDIR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    cfg = load_config()

    tif_paths, seen = [], set()
    for fn in sorted(os.listdir(PRELABEL_DIR)):
        if not fn.endswith(".json"):
            continue
        with open(os.path.join(PRELABEL_DIR, fn), encoding="utf-8") as f:
            p = PrelabelRecord.from_dict(json.load(f))
        # prelabel 里的 image_path 可能仍是迁移前旧根（D:\dataset\...）——必须 canon_path
        # 改写到当前 <项目>\dataset\ 下，否则 103 张 tif 全部读取失败（2026-09-24 实测）
        ip = canon_path(p.image_path)
        if ip.lower().endswith((".tif", ".tiff")) and ip not in seen:
            seen.add(ip)
            tif_paths.append(ip)
    print(f"任务引用的 tif: {len(tif_paths)}")

    import cv2
    done = skip = fail = 0
    for ip in tif_paths:
        rel = os.path.relpath(ip, DATASET).replace("\\", "/")
        parts = os.path.splitext(rel)[0].split("/")
        out = os.path.join(VIEW_ROOT, *parts) + ".jpg"
        if os.path.exists(out) and not args.force:
            skip += 1
            continue
        img = imread_cn(ip)
        if img is None:
            print(f"  [FAIL 读取] {ip}")
            fail += 1
            continue
        os.makedirs(os.path.dirname(out), exist_ok=True)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            print(f"  [FAIL 编码] {ip}")
            fail += 1
            continue
        buf.tofile(out)
        done += 1
        if done % 20 == 0:
            print(f"  已转换 {done}")
    print(f"转换 {done}，已存在跳过 {skip}，失败 {fail}")

    # 注册 ls_view 本地存储（重复注册会被 LS 拒绝，静默跳过）。
    # 2026-09-24：project=1 已退役，写死会 400；仅当配置了有效 project_id 时注册。
    pid = cfg["labelstudio"].get("project_id")
    if pid:
        try:
            ls_api(cfg, "GET", f"/api/projects/{pid}")
            ls_api(cfg, "POST", "/api/storages/localfiles",
                   json={"project": pid, "path": VIEW_ROOT, "title": "ls_view(tif代理)",
                         "use_blob_urls": True})
            print(f"已注册 ls_view 本地存储（项目 {pid}）")
        except SystemExit:
            print("ls_view 本地存储可能已注册，继续")
    else:
        print("跳过 ls_view 存储注册：vlm_config.json 未配置 labelstudio.project_id"
              "（导入 tif 任务时由 UI/03 脚本按项目注册即可）")


if __name__ == "__main__":
    main()
