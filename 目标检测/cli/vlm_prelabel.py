# -*- coding: utf-8 -*-
"""
vlm_prelabel.py — 阶段一：调用多模态视觉大模型 API 批量预标注
输入: pilot_manifest.csv
输出: results/detection/prelabels/<stem>.json
      （VLM 原始预标注：tight_bbox/confidence/形态；外加 `scale` 字段 =
        比例尺双引擎标定结果，见下）

约定:
- 图像先缩放到最长边 max_side（默认1536），VLM 在该坐标系下给框，脚本按比例映射回原图 2048x1536；
- confidence: high=形态典型可直接采信, medium=必须人工复核（导入 Label Studio 后以 score 区分排序）;
- 接口为 OpenAI 兼容 /chat/completions（智谱/通义兼容模式/OpenAI/LM Studio 均可）;
- API Key 优先读环境变量 VLM_API_KEY。
- **比例尺标定（2026-09-20 起）**：检测阶段即调用 `scalebar_duo.measure_img_px_per_um()`
  双引擎（A 本地形态学+OCR / B VLM，交叉验证）逐图测出 px_per_um，写进 JSON 的 `scale` 字段；
  离线 `scale_table.csv` 已全项目停用。下游（LS 任务 data、分割交付清单、几何测量）都从这份值取。

用法:
  python vlm_prelabel.py            # 全量跑（pilot_manifest.csv）
  python vlm_prelabel.py --limit 5  # 先跑5张验证 prompt 效果
  python vlm_prelabel.py --dry-run  # 只打印请求不做调用
  python vlm_prelabel.py --dirs 大黄导管,常山针晶 --force
      # 按目录名从根 manifest.csv 取行，启用【目录先验 prompt】（v7_dir_prior）：
      # 目录名指明的特征必然存在，必须逐一框出；同时图内其他类别特征也要框。
      # --force 覆盖已存在的输出 JSON（默认跳过）。
  python vlm_prelabel.py --only-scale            # 只给已有 JSON 补测比例尺（不重跑检测）
  python vlm_prelabel.py --only-scale --dry-run  # 只列出待补测清单
  python vlm_prelabel.py --relax-consensus       # 离线把 OK-medium 升为 OK-consensus（不调 API）
      # 依据：A 的 bar_px 是离散可重复读数，若已被 ≥3 张精确 OK 样本旁证，则"B 未证实"
      # 只是 B 的噪声 → 判 OK-consensus（可信档）。实测 119 份 OK-medium 中 118 份可升级。
  python vlm_prelabel.py --migrate-paths         # 离线把旧根 image_path 改写为当前规范路径
      # 417/422 份仍是迁移前旧根 D:\dataset\...；原值存 image_path_legacy，幂等；配 --dry-run 预览

目录先验（FOLDER_PRIOR）:
  目录名中含 簇晶/针晶/方晶/导管/石细胞/淀粉粒 关键词 → 该特征为"必框"先验，
  注入 prompt 提示 VLM（herb 取目录名去掉关键词与"草酸钙"后的剩余部分）。
"""
import argparse
import concurrent.futures as futures
import csv
import hashlib
import json
import os
import sys
import time

import cv2

from contracts import DetBox, ScaleReading  # noqa: E402 第 ⑤ 段契约（prelabel 的框层/标定层）
from detect_core import detect_once  # noqa: E402 单次检测的唯一实现（B1，2026-09-29）
from ls_common import (DATASET, DET_RESULTS, MANIFEST_PATH, PRELABEL_DIR, canon_path, imread_cn,  # noqa: E402
                       load_config, resolve_key)

# 类别白名单 / 目录先验 / 提示词已下沉到 prompts.py（2026-09-29 P1-3）
# —— 编号脚本按语言约束不能作为包成员被 import（'02_vlm_prelabel'.isidentifier() 为 False），
#    提示词属库内容而非入口逻辑，故由 prompts.py 承载；本文件只做 CLI 入口。
from prompts import (LABEL_ALIASES, PRIOR_KEYWORDS, PROMPT, PROMPT_VERSION,  # noqa: E402
                     VALID_LABELS, build_prompt, prior_from_dir)


def _norm_path(p):
    return os.path.normpath(str(p)).replace("\\", "/").lower()


_PRELABEL_INDEX = None      # norm(canon image_path) → 已存在的输出键（文件名去掉 .json）


def _prelabel_index():
    """PRELABEL_DIR 现有 prelabel 的「图像 → 输出键」索引，首次调用时构建、本进程内复用。

    **为什么需要**（2026-10-12 修）：原 `resolve_out_key` 只比对**无后缀**的 `<base>.json`，
    于是同一张图若历史上因图像库换根 / `--migrate-paths`（只改 image_path、不改文件名）而
    落在 `<base>_<旧哈希>.json` 下，重跑就认不出来 → 再写一份，同一张图出现两份 prelabel
    （实测麦冬针晶/龙胆针晶 7 对，且会让下游给同一张图建重复 LS 任务）。
    这里改成按**图像身份**找既有结果——无论文件名长什么样都复用，resolve_out_key 才真正幂等。
    key 走 canon_path，以兼容仍是旧根的存量条目。
    """
    global _PRELABEL_INDEX
    if _PRELABEL_INDEX is None:
        idx = {}
        for fn in os.listdir(PRELABEL_DIR):
            if not fn.endswith(".json"):
                continue
            try:
                with open(os.path.join(PRELABEL_DIR, fn), encoding="utf-8") as f:
                    ip = json.load(f).get("image_path", "")
            except Exception:
                continue
            if ip:
                idx[_norm_path(canon_path(ip))] = fn[:-5]
        _PRELABEL_INDEX = idx
    return _PRELABEL_INDEX


def resolve_out_key(row, stem):
    """输出键 = feature_stem；同名文件属于另一张图（跨目录同名）时追加短哈希消歧。

    优先**按图像身份复用**既有结果（含历史哈希名），见 `_prelabel_index`；只有该图从未出过
    结果、且 `<base>.json` 已被别的图占用时，才新算哈希。
    """
    base = f'{row["feature_label"]}_{stem}'
    hit = _prelabel_index().get(_norm_path(row["image_path"]))
    if hit:
        return hit, os.path.join(PRELABEL_DIR, hit + ".json")
    out_path = os.path.join(PRELABEL_DIR, base + ".json")
    if not os.path.exists(out_path):
        return base, out_path
    h = hashlib.md5(str(row["image_path"]).encode("utf-8")).hexdigest()[:6]
    alt = f"{base}_{h}"
    return alt, os.path.join(PRELABEL_DIR, alt + ".json")


def process_one(row, cfg_vlm, api_key, dry_run=False, force=False):
    stem = os.path.splitext(os.path.basename(row["image_path"]))[0]
    key, out_path = resolve_out_key(row, stem)
    if os.path.exists(out_path) and not force:
        return key, "skip(已存在)"

    img = imread_cn(row["image_path"])
    if img is None:
        return key, "FAIL(读取失败)"
    H0, W0 = img.shape[:2]

    dir_prior = row.get("dir_prior") or None
    prompt = build_prompt(dir_prior)
    max_side = cfg_vlm.get("max_side", 1536)

    if dry_run:
        s = min(1.0, max_side / max(H0, W0))        # 与 detect_once 同一"只缩不放"公式
        return key, (f"DRY-RUN scale={s:.4f} ({W0}x{H0}→{int(W0 * s)}x{int(H0 * s)}) "
                     f"prior={dir_prior[2] if dir_prior else '无'}")

    # 单次检测走唯一实现（2026-09-29 B1）：编码/调用/解析/白名单/退化框/坐标映射都在 detect_once。
    # 口径与旧实现逐项一致：max_side 取配置值（默认 1536）、quality=90、clamp="full"（全图写法）。
    res, err = detect_once(img, cfg_vlm, api_key, prompt, VALID_LABELS,
                           max_side=max_side, quality=90, clamp="full")
    if res is None:
        return key, f"FAIL({err})"
    W1, H1 = int(W0 * res["scale"]), int(H0 * res["scale"])

    # 框层收口到契约（2026-09-29 C1a）：note 只在原始 object 里，故 from_det 需一并传
    objs = [DetBox.from_det(d, raw).to_dict() for d, raw in zip(res["dets"], res["accepted"])]
    dropped = res["dropped"]

    result = {
        "source": "vlm_api",
        "model": cfg_vlm["model"],
        "prompt_version": PROMPT_VERSION,
        "image_path": row["image_path"],
        "feature_label_folder": row["feature_label"],
        "herb": row["herb"],
        "group_key": row["group_key"],
        "needs_review": row.get("needs_review", ""),
        "dir_prior": {"dir": dir_prior[0], "herb": dir_prior[1], "feature": dir_prior[2]} if dir_prior else None,
        "original_size": [W0, H0],
        "inference_size": [W1, H1],
        "objects": objs,
        "dropped_objects": dropped,
        "raw_response_excerpt": res["text"][:500] if dropped else "",
        "scale": measure_scale(img, row["image_path"]).to_dict(),
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return key, f"OK {len(objs)}框(弃{len(dropped)})"


def measure_scale(img, image_path):
    """比例尺双引擎标定（检测阶段职责，2026-09-20 起取代离线 scale_table.csv）。

    结果随 prelabel JSON 流转：LS 任务 data.px_per_um ← 本字段，分割交付清单 px_per_um 亦源于此，
    几何测量只认交付清单里的这份值。

    **编排只有一份**（2026-09-29 A2）：实际调用 `detect_core.lookup_px_per_um(full=True)`
    —— 双引擎现测与两个语料级证据源（`family_bar_px` 同目录族先验、`confirmed_bar_px` 已旁证
    bar_px 离散集）都在那里注入；本函数只做契约转换（2026-09-29 C1a：`ScaleReading.from_info`，
    字段名单点，含引擎的 `bar_px_cv` → 契约的 `bar_px`）。调用方落盘时 `.to_dict()`。
    延迟导入：`--relax-consensus` / `--migrate-paths` 等纯离线动作不经过本函数，
    无须为此拉起检测原语链。
    """
    from detect_core import lookup_px_per_um
    return ScaleReading.from_info(lookup_px_per_um(image_path, img=img, full=True))


def backfill_scale(cfg, api_key, workers=4, dry_run=False, force=False):
    """--only-scale：只给已存在的 prelabel JSON 补测 px_per_um（不重跑目标检测）。

    用于存量 prelabel（422 份，写在比例尺改由双引擎测之前）的回填；
    随后跑 refresh_task_data.py 即可把这些值同步进 LS 任务 data。
    """
    import concurrent.futures as futures
    files = sorted(f for f in os.listdir(PRELABEL_DIR) if f.endswith(".json"))
    todo = []
    for fn in files:
        with open(os.path.join(PRELABEL_DIR, fn), encoding="utf-8") as f:
            pre = json.load(f)
        if ScaleReading.from_dict(pre.get("scale")).px_per_um and not force:
            continue
        todo.append((fn, canon_path(pre.get("image_path", ""))))
    print(f"待补测 {len(todo)}/{len(files)} 份 prelabel（已有值的跳过{'，--force 覆盖' if force else ''}）")
    if dry_run:
        for fn, ip in todo:
            print(f"  [DRY-RUN] {fn} ← {ip}")
        return

    def work(item):
        fn, ip = item
        if not os.path.isfile(ip):
            return fn, None, f"FAIL(图不存在 {ip})"
        img = imread_cn(ip)
        if img is None:
            return fn, None, "FAIL(读取失败)"
        return fn, measure_scale(img, ip), ""

    stats, done = {}, 0
    with futures.ThreadPoolExecutor(max_workers=workers) as ex:
        for i, (fn, info, err) in enumerate(ex.map(work, todo), 1):
            p = os.path.join(PRELABEL_DIR, fn)
            if err:
                print(f"[{i}/{len(todo)}] {fn}: {err}")
                stats["FAIL"] = stats.get("FAIL", 0) + 1
                continue
            with open(p, encoding="utf-8") as f:
                pre = json.load(f)
            pre["scale"] = info.to_dict()
            with open(p, "w", encoding="utf-8") as f:
                json.dump(pre, f, ensure_ascii=False, indent=2)
            tag = info.status or "?"
            stats[tag] = stats.get(tag, 0) + 1
            done += 1
            print(f"[{i}/{len(todo)}] {fn}: {tag} px_per_um={info.px_per_um} "
                  f"bar_px={info.bar_px} um={info.bar_um}")
    print(f"\n回填完成：{done} 份，统计 {stats} → {PRELABEL_DIR}")


def relax_consensus(prelabel_dir=None, dry_run=False):
    """离线把存量 prelabel 的 OK-medium 提升为 OK-consensus —— **不调 API、不改 px_per_um**。

    为什么能离线做：prelabel 的 `scale.bar_px` 就是 A 引擎的读数，而"是否已旁证"只取决于
    `ls_common.confirmed_bar_px()` 这个由**精确 OK 样本**构建的离散集 → 无需重跑标定。
    实测（2026-09-28，422 份）：119 份 OK-medium 里 118 份的 bar_px 已被旁证（233/311 两档），
    仅 1 份 bar_px=331 无旁证保持 medium。

    只重写 status/grade/note 三个字段，px_per_um 与标注 objects 一律不动（可回溯：note 里留痕）。
    """
    from ls_common import STATUS_OK_CONSENSUS, confirmed_bar_px
    root = prelabel_dir or PRELABEL_DIR
    ok_set = confirmed_bar_px(prelabel_dir=prelabel_dir, use_cache=prelabel_dir is None)
    files = sorted(f for f in os.listdir(root) if f.endswith(".json"))
    up, skip_nofam, skip_other = [], [], []
    for fn in files:
        p = os.path.join(root, fn)
        try:
            with open(p, encoding="utf-8") as f:
                pre = json.load(f)
        except Exception:
            continue
        sc = pre.get("scale") or {}
        if sc.get("status") != "OK-medium":
            skip_other.append(fn)
            continue
        if not sc.get("bar_px") or int(sc["bar_px"]) not in ok_set:
            skip_nofam.append((fn, sc.get("bar_px")))
            continue
        up.append((fn, p, pre))
    print(f"旁证集（≥3 张精确 OK 的 bar_px）: {sorted(ok_set)}")
    print(f"待提升 OK-medium → OK-consensus: {len(up)} 份"
          f" | bar_px 无旁证保持 medium: {len(skip_nofam)} | 其它状态不动: {len(skip_other)}")
    for fn, bpx in skip_nofam:
        print(f"  [保持] {fn}: bar_px={bpx} 无旁证")
    if dry_run:
        for fn, _, pre in up[:10]:
            sc = pre["scale"]
            print(f"  [DRY-RUN] {fn}: OK-medium → OK-consensus (bar_px={sc['bar_px']})")
        if len(up) > 10:
            print(f"  [DRY-RUN] …另 {len(up) - 10} 份")
        return len(up)
    for fn, p, pre in up:
        sc = pre["scale"]
        sc["status"] = STATUS_OK_CONSENSUS
        sc["grade"] = "consensus"
        sc["note"] = ((sc.get("note") or "").split("；bar_px=")[0]
                      + f"；bar_px={sc['bar_px']} 已由同批精确 OK 样本旁证（离线提升，未重跑）")
        _dump_prelabel(p, pre)
    print(f"已提升 {len(up)} 份 → {root}（px_per_um 与 objects 未改动）")
    return len(up)


def _dump_prelabel(path, pre):
    """prelabel JSON 统一落盘格式（供 relax_consensus / migrate_paths 复用）。"""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(pre, f, ensure_ascii=False, indent=2)


def fix_out_of_frame(prelabel_dir=None, dry_run=False):
    """离线把存量 prelabel 里「越界 / 反向」的框归正并夹到画幅内（离线、幂等、可 --dry-run）。

    背景（2026-10-12）：VLM 偶尔返回**超出 0~1000 归一化系**的坐标；旧的 `clamp="full"` 只对
    min 角保下界 → 映射回原图后 gx1 越过画幅右缘并与 gx2 反向（如 `[3870.7, 838.1, 3840.0, 1105.9]`），
    产物里留成退化框（`contracts.check_prelabel` 报「退化框」，导出侧只能整框丢弃）。写侧已修
    （`detect_core.detect_once` 统一加"归正 + 双向夹取"）；本函数把**修之前落盘**的存量按同一口径
    就地修一遍，不读图、不调 API。

    规则（与 `export_det_dataset` 的夹取口径一致）：
      · 先归正（x1<x2 / y1<y2），再双向夹到 [0,W]×[0,H]（W/H 取 `original_size`）
      · 夹后宽或高 < 1px 的框删除
      · 只动各框的 `tight_bbox`；其余键、键序、落盘格式一律不变（统一走 `_dump_prelabel`）
    返回受影响文件数（0 = 无需修复，幂等）。
    """
    root = prelabel_dir or PRELABEL_DIR
    n_file = n_fix = n_drop = 0
    for fn in sorted(f for f in os.listdir(root) if f.endswith(".json")):
        p = os.path.join(root, fn)
        try:
            with open(p, encoding="utf-8") as f:
                pre = json.load(f)
        except Exception:
            continue
        size = pre.get("original_size") or []
        if len(size) != 2 or not size[0] or not size[1]:
            continue
        W, H = float(size[0]), float(size[1])
        kept, fixed, dropped = [], 0, 0
        for o in (pre.get("objects") or []):
            bb = o.get("tight_bbox")
            if not isinstance(bb, (list, tuple)) or len(bb) != 4:
                kept.append(o)
                continue
            x1, y1, x2, y2 = [float(v) for v in bb]
            nx1, nx2 = sorted((x1, x2))
            ny1, ny2 = sorted((y1, y2))
            nx1, nx2 = min(max(nx1, 0.0), W), min(max(nx2, 0.0), W)
            ny1, ny2 = min(max(ny1, 0.0), H), min(max(ny2, 0.0), H)
            if nx2 - nx1 < 1.0 or ny2 - ny1 < 1.0:
                dropped += 1
                continue
            new = [round(nx1, 1), round(ny1, 1), round(nx2, 1), round(ny2, 1)]
            if new != [round(v, 1) for v in (x1, y1, x2, y2)]:
                o["tight_bbox"] = new
                fixed += 1
            kept.append(o)
        if not (fixed or dropped):
            continue
        n_file += 1
        n_fix += fixed
        n_drop += dropped
        print(f"  {fn}: 归正 {fixed} 框，删除 {dropped} 框")
        if not dry_run:
            pre["objects"] = kept
            _dump_prelabel(p, pre)
    print(f"\n{'[dry-run] ' if dry_run else ''}越界框修复：{n_file} 份受影响，"
          f"归正 {n_fix} 框，删除 {n_drop} 框")
    return n_file


def migrate_paths(prelabel_dir=None, dry_run=False):
    """把存量 prelabel 的「迁移前旧根 image_path」改写为当前规范路径（离线、幂等、可 --dry-run）。

    背景（2026-09-28 实测）：417/422 份 prelabel 的 image_path 仍是迁移前旧根 `D:\\dataset\\...`
    （只有 v7.2 那 5 份是当前根）。读取侧 ls_common.canon_path() 能正确改写，但**依赖每个
    调用方自觉**，已有一次实锤事故（build_tif_proxies 裸用旧路径 → 103 张 tif 读取失败），
    且仍有误报失败（detect_sample --group 拿旧根键与 abspath 比对 → 明明有图却报未命中）。

    规则：
      · canon_path 能解析到真实文件 → 改写为规范路径；原值存入 image_path_legacy（留痕，只写一次）
      · 无法解析（图确实不在）      → image_path 不动，标 image_status="missing"（不伪造路径）
      · 已是规范路径                 → 不动（幂等）；若曾标 missing 而图已恢复则清掉该标记
    只动 image_path / image_path_legacy / image_status 三个键，objects / scale 等一律不动。
    """
    from ls_common import canon_path
    root = prelabel_dir or PRELABEL_DIR
    moved, missing, same = [], [], 0
    for fn in sorted(f for f in os.listdir(root) if f.endswith(".json")):
        p = os.path.join(root, fn)
        try:
            with open(p, encoding="utf-8") as f:
                pre = json.load(f)
        except Exception:
            continue
        raw = pre.get("image_path") or ""
        can = canon_path(raw)
        if not os.path.isfile(can):
            if pre.get("image_status") != "missing":
                pre["image_status"] = "missing"
                if not dry_run:
                    _dump_prelabel(p, pre)
            missing.append((fn, raw))
            continue
        if os.path.normcase(os.path.normpath(raw)) == os.path.normcase(os.path.normpath(can)):
            if pre.get("image_status") == "missing":      # 曾缺图、现已被找回
                pre.pop("image_status", None)
                if not dry_run:
                    _dump_prelabel(p, pre)
            same += 1
            continue
        pre["image_path"] = can
        pre.setdefault("image_path_legacy", raw)          # 原值留痕（二次运行不覆盖）
        pre["image_status"] = "ok"
        if not dry_run:
            _dump_prelabel(p, pre)
        moved.append((fn, raw, can))
    print(f"prelabel {len(moved) + same + len(missing)} 份 | "
          f"待改写旧根路径 {len(moved)} | 已是规范路径 {same} | 图缺失 {len(missing)}")
    for fn, raw, can in moved[:5]:
        print(f"  [改写] {fn}\n         {raw}\n      →  {can}")
    if len(moved) > 5:
        print(f"  [改写] …另 {len(moved) - 5} 份")
    for fn, raw in missing:
        print(f"  [缺图] {fn}: {raw}（image_path 不动，标 image_status=missing）")
    if dry_run:
        print("[DRY-RUN] 未写盘")
    else:
        print(f"已迁移 → {root}")
    return len(moved)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="只处理前N张(0=全量)")
    ap.add_argument("--sample", type=str, default="",
                    help='按类抽样 "类名:张数,类名:张数"，如 "石细胞:5,导管:5"')
    ap.add_argument("--dirs", type=str, default="",
                    help='逗号分隔的目录名（从根 manifest.csv 取行并启用目录先验），如 "大黄导管,常山针晶"；'
                         '也支持目录名包含的关键词匹配，如 "针晶" 命中 麦冬针晶/龙胆针晶/常山草酸钙针晶')
    ap.add_argument("--only", type=str, default="",
                    help="逗号分隔的文件名 stem（不含扩展名），只处理这些图，如 \"2,10,B20130506\"")
    ap.add_argument("--force", action="store_true", help="覆盖已存在的输出 JSON（默认跳过）")
    ap.add_argument("--only-scale", action="store_true",
                    help="只给已有 prelabel JSON 补测比例尺 px_per_um（不重跑目标检测；配合 --force 覆盖已有值）")
    ap.add_argument("--relax-consensus", action="store_true",
                    help="离线把存量 prelabel 的 OK-medium 提升为 OK-consensus：bar_px 已被 ≥3 张精确 OK "
                         "样本旁证者升级（不读图、不调 API、px_per_um 与 objects 不动）；配 --dry-run 预览")
    ap.add_argument("--migrate-paths", action="store_true",
                    help="离线把存量 prelabel 的旧根 image_path 改写为当前规范路径（原值存 image_path_legacy，"
                         "幂等；图缺失者标 image_status=missing）；配 --dry-run 预览")
    ap.add_argument("--fix-oob", action="store_true",
                    help="离线把存量 prelabel 里越界/反向的框归正并夹到画幅内（夹后 <1px 的删除）；"
                         "不读图、不调 API；配 --dry-run 预览")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--workers", type=int, default=0, help="覆盖配置中的并发数")
    args = ap.parse_args()

    cfg = load_config()
    ensure_dirs = __import__("ls_common").ensure_dirs
    ensure_dirs()

    # --relax-consensus / --migrate-paths 纯离线（不读图、不调 API），故放在 api_key 校验之前
    if args.relax_consensus:
        relax_consensus(dry_run=args.dry_run)
        return
    if args.migrate_paths:
        migrate_paths(dry_run=args.dry_run)
        return
    if args.fix_oob:
        fix_out_of_frame(dry_run=args.dry_run)
        return

    api_key = resolve_key(cfg)
    if not api_key and not args.dry_run:
        print("错误: 未设置 VLM_API_KEY 环境变量，且 vlm_config.json 中 api_key 为空。", file=sys.stderr)
        sys.exit(2)

    if args.only_scale:
        backfill_scale(cfg, api_key,
                       workers=args.workers or cfg["vlm"].get("max_workers", 4),
                       dry_run=args.dry_run, force=args.force)
        return

    if args.dirs:
        with open(MANIFEST_PATH, "r", encoding="utf-8-sig") as f:
            all_rows = list(csv.DictReader(f))
        wants = [w.strip() for w in args.dirs.split(",") if w.strip()]
        rows = [r for r in all_rows
                if any(w == r["folder_name"] or w in r["folder_name"] for w in wants)]
        if not rows:
            print(f"错误: 根 manifest.csv 中未命中目录 {wants}", file=sys.stderr)
            sys.exit(2)
        print(f"--dirs 命中 {len(rows)} 张（启用目录先验 {PROMPT_VERSION}）")
    else:
        with open(os.path.join(DET_RESULTS, "pilot_manifest.csv"), "r", encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
    # 兼容迁移前旧绝对路径（manifest 已重建，但 pilot_manifest.csv 等历史产物仍是旧值）
    for r in rows:
        r["image_path"] = canon_path(r["image_path"])
    if args.limit:
        rows = rows[:args.limit]

    # 统一 schema + 标签归一化 + 目录先验（必须在 --sample 过滤之前，否则归一化后的类名匹配不到）
    norm_rows = []
    for r in rows:
        fl = LABEL_ALIASES.get(r["feature_label"], r["feature_label"])
        _dir, _herb, _feat = prior_from_dir(r["image_path"])
        if fl not in VALID_LABELS:
            # manifest 标签为空/不在白名单 → 回退目录先验（支持药材级目录，见 folder_feature.json）
            if _feat:
                if fl:
                    print(f"警告: 目录先验({_feat})与 manifest 标签({fl})不一致，以目录为准: {r['image_path']}",
                          file=sys.stderr)
                fl = _feat
            else:
                print(f"警告: 跳过未知 feature_label={r['feature_label']} ({r['image_path']})", file=sys.stderr)
                continue
        elif _feat and _feat != fl:
            print(f"警告: 目录先验({_feat})与 manifest 标签({fl})不一致，以目录为准: {r['image_path']}",
                  file=sys.stderr)
            fl = _feat
        norm_rows.append({
            "image_path": r["image_path"], "feature_label": fl,
            "herb": r.get("herb", ""), "group_key": r.get("group_key", ""),
            "needs_review": r.get("needs_review", ""),
            "dir_prior": (_dir, _herb, _feat) if _feat else None,
        })
    rows = norm_rows

    if args.sample:
        want = {}
        for part in args.sample.split(","):
            k, v = part.rsplit(":", 1)
            want[k.strip()] = int(v)
        picked = []
        for fl, n in want.items():
            pool = [r for r in rows if r["feature_label"] == fl]
            picked.extend(pool[:n])
        rows = picked

    if args.only:
        stems = {s.strip() for s in args.only.split(",") if s.strip()}
        rows = [r for r in rows
                if os.path.splitext(os.path.basename(r["image_path"]))[0] in stems]
        if not rows:
            print(f"错误: --only 未命中任何图 {sorted(stems)}", file=sys.stderr)
            sys.exit(2)
        print(f"--only 命中 {len(rows)} 张")

    print(f"待预标注 {len(rows)} 张，模型 {cfg['vlm']['model']}")

    workers = args.workers or cfg["vlm"].get("max_workers", 4)
    stats = {}
    t0 = time.time()
    with futures.ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(process_one, r, cfg["vlm"], api_key, args.dry_run, args.force): r for r in rows}
        for i, fut in enumerate(futures.as_completed(futs), 1):
            key, status = fut.result()
            tag = status.split("(")[0]
            stats[tag] = stats.get(tag, 0) + 1
            print(f"[{i}/{len(rows)}] {key}: {status}")

    print(f"\n完成于 {time.time()-t0:.0f}s，统计: {stats}")
    n_ok = stats.get("OK", 0)
    if n_ok and not args.dry_run:
        n_box = 0
        for fn in os.listdir(PRELABEL_DIR):
            with open(os.path.join(PRELABEL_DIR, fn), encoding="utf-8") as f:
                n_box += len(json.load(f).get("objects", []))
        print(f"共产出 {n_box} 个预标注框 → {PRELABEL_DIR}")


if __name__ == "__main__":
    main()
