# -*- coding: utf-8 -*-
"""
detect_sample.py — 检测管线（样品级重构版）：同一样品的多幅显微图像 → 目标子图 + 比例尺
输入: 同一样品（同一批粉末）的多幅显微图像（--dir 整目录 / --images 逗号列表）
流程: 每幅图 → 检测引擎（`--engine vlm|yolo`，默认 vlm）：
      · vlm  —— **整图 VLM 检测 1 次**（模型与思考强度见 `config/vlm_config.json`）；
                **空检分块兜底已于 2026-09-30 关闭**
                （detect_core.detect_adaptive：整图空就是空，不再追加 2×2 放大重检）
      · yolo —— P1 训练的专用域检测器（`lib/detector_yolo.py`），类别顺序取自
                `vlm_config.json → yolo.classes`，权重/阈值取 `config/detector.json`
      → 每个检测框按**共享松框原语**（`common/geom.loose_box`）裁剪目标子图，外扩量取短边的
      30%（见 `DET_LOOSE_MARGIN_RATIO`：极性探针要四角是背景、VLM 框可能欠覆盖，故不能收小）
      → 比例尺双引擎标定该幅图的 px/μm（scalebar_duo）
      「其他」类**识别≠交付**：检测照常输出，但不进 crops/（IGM 无该特征），另落
      `other_detections.json`；crops.json 记 `n_other`（方案 §6.4）
关于目录先验（2026-09-30）：**默认不注入**。本入口是送检样品路径，产出的框是判药证据，而
  目录名对送检样品是**待验证身份**——注进提示词等于把答案喂给取证环节（且送检目录多为药材名，
  feature 推导恒为空，注入只会塞进 'None' 字面量）。标注语料（目录=标签）口径走 `--dir-prior`。
输出（默认 results/detection/sample_detect/{sample}/，或用 --out-dir 指定；run_pipeline 指向
runs/<run_id>/detection/ 以保持「一个 run = 一个工件集」）:
  crops/{crop_id}.png      目标子图（外扩 crop，BGR）
  crops/{crop_id}.json     每 crop 随行元数据（**「检测 → 分割」输入契约**）：feature_id / stem /
                           herb / px_per_um / tight_box / tight_box_local / loose_box / crop_size …
                           —— 故 `crops/` 可直接作 `igm_segment_batch --crops` 的输入
  crops.json               聚合清单（crop_id / confidence / morphology / scale_status …，供复核；
                           含 engine / detector / n_other 溯源字段）
  other_detections.json    仅当检出「其他」类时生成：识别结果留档，不进交付 crops/
  review/detect_{stem}.jpg  每幅视野图的**交付类**检测框目检图（中文标签；「其他」见 json）

关于"是否还要记录子图属于哪个原图"：
  下游（分割/测量）契约不再依赖原图身份——聚合主键是样品，每张子图自带 px_per_um；
  但模块内部仍保留 source_image 字段，原因：①比例尺按原图双引擎现测（不同视野
  标定可不同）②跨目录同名 stem 消歧（如 20230311.tif 同时存在于 4 个药材目录）
  ③人工复核时需要回到原图定位。

用法:
  python detect_sample.py --dir 大黄草酸钙簇晶 --limit 2
  python detect_sample.py --images "p1.tif,p2.tif" --sample 我的样品
  python detect_sample.py --dir 大黄草酸钙簇晶 --engine yolo --limit 2   # P2：专用域检测器
  python detect_sample.py --dir 大黄草酸钙簇晶 --skip-crop   # 只出检测框不裁剪
  python detect_sample.py --dir <原图目录> --out-dir <run>/detection   # 供 run_pipeline 编排
  python detect_sample.py --dir <原图目录> --rollouts 5 --out-dir <实验目录>
      # 多轮自洽采样（V1 只加票不过滤）：每框多 votes/rollout_n，目检图标签带票数。
      # 默认 1 = 关；N>1 是实验口径，产物不能直接当交付（见 docs/检测多轮采样投票_方案与实验设计.md）
"""
import argparse
import json
import os
import sys
import time

import cv2

import contracts  # noqa: E402 crop 随行 json 的写侧唯一实现（项目根 contracts.py）
from classify_crop import classify_crop  # noqa: E402 stage 2 分类（两阶段检测第二步）
from det_dataset import split_delivery  # noqa: E402 识别≠交付分流（类别顺序由适配器统一解析）
from detect_core import (DET_LOOSE_MARGIN_RATIO, FEATURE_ID, detect_adaptive,  # noqa: E402
                         detect_consensus, lookup_px_per_um)
import detector_yolo  # noqa: E402 P2 推理适配器（YOLO 权重 → 与 VLM 同形的 dets）
from geom import loose_box  # noqa: E402 松框裁剪原语（全流程约定，与 UI 链同一份）
from ls_common import (DATASET, DET_RESULTS, IMAGE_SUBDIRS, PROJECT_ROOT,  # noqa: E402
                       imread_cn, load_config, resolve_key)
from prompts import (CLASSIFY_VERSION, ENTITY_LABEL, PRIOR_KEYWORDS, PROMPT_S1,  # noqa: E402
                     PROMPT_VERSION, PROMPT_VERSION_S1)
import textvis  # noqa: E402 中文文字渲染（cv2.putText 不支持中文）

OUT_ROOT = os.path.join(DET_RESULTS, "sample_detect")
IMG_EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")


def resolve_images(args):
    """解析指定目录或图像列表，返回路径及其来源目录名。"""
    items = []
    if args.images:
        for p in [s.strip() for s in args.images.split(",") if s.strip()]:
            if not os.path.exists(p):
                sys.exit(f"[FAIL] 图像不存在: {p}")
            items.append((os.path.abspath(p),
                          os.path.basename(os.path.normpath(os.path.dirname(p)))))
        return items
    d = args.dir
    if not os.path.isdir(d):
        # 允许传目录名，在图像库两个子根下模糊查找（2026-09-20 起图像库迁入 <项目>\dataset\，
        # 用 ls_common 的 IMAGE_SUBDIRS 单一事实源，不再指向已不存在的 <项目>\显微镜像）
        for sub in IMAGE_SUBDIRS:
            cand = os.path.join(DATASET, sub, d)
            if os.path.isdir(cand):
                d = cand
                break
        else:
            sys.exit(f"[FAIL] 目录不存在: {args.dir}")
    files = sorted(f for f in os.listdir(d) if f.lower().endswith(IMG_EXTS))
    items = [(os.path.abspath(os.path.join(d, f)), os.path.basename(d)) for f in files]
    if args.limit:
        items = items[:args.limit]
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="", help="样品图像目录（也可传显微镜像库或小样本库中的目录名）")
    ap.add_argument("--images", default="", help="逗号分隔的图像路径列表（与 --dir 二选一）")
    ap.add_argument("--sample", default="", help="样品名（默认=目录名）")
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 幅")
    ap.add_argument("--engine", choices=["vlm", "yolo"], default="vlm",
                    help="检测引擎：vlm=默认现行为；yolo=P1 专用域检测器（P2）。"
                         "yolo 与 --two-stage/--rollouts/--dir-prior 互斥")
    ap.add_argument("--weights", default="",
                    help="YOLO 权重（仅 --engine yolo；默认取 detector.json → weights，"
                         "相对路径按项目根解析）")
    ap.add_argument("--detector-config", default="",
                    help="YOLO 推理配置（默认 目标检测/config/detector.json）")
    ap.add_argument("--herb", default="", help="药材名（默认从目录名先验推导）")
    ap.add_argument("--dir-prior", action="store_true",
                    help="把目录名当特征先验注入检测提示词（**标注语料口径**，默认关）。本入口是"
                         "**送检样品**路径：目录名是待验证的样品身份，注入=把答案喂给取证环节，"
                         "检测框就不再是无偏证据；且送检目录名多为药材名，feature 推导恒为空，"
                         "打开只会往提示词里塞 'None' 字面量。需要复现标注口径时才开")
    ap.add_argument("--skip-crop", action="store_true",
                    help="只检测不裁剪（此时不产随行 json，产物不能喂分割）")
    ap.add_argument("--rollouts", type=int, default=1,
                    help="多轮自洽采样轮数（默认 1 = 关，行为与 detect_adaptive 完全一致）。"
                         ">1 时改走 detect_consensus：同图同提示词采样 N 轮、按出现轮数聚簇，"
                         "**V1 只加票不过滤**，每框多出 votes/rollout_n 两个字段。"
                         "方案见 docs/检测多轮采样投票_方案与实验设计.md")
    ap.add_argument("--rollout-temp", type=float, default=0.7,
                    help="多轮采样的温度（仅 --rollouts>1 生效；默认 0.7 与所借鉴论文一致）")
    ap.add_argument("--two-stage", action="store_true",
                    help="两阶段检测：stage 1 无类别提框（PROMPT_S1，只给位置）→ stage 2 在"
                         "**放大后的 crop** 上分类（classify_crop）。**默认关**。与 --rollouts、"
                         "--skip-crop 互斥。被 stage 2 拒的框不进交付集（缺 feature_id 喂不进 IGM），"
                         "但落 stage2_rejected.json 供复盘。见 docs/检测两阶段_…")
    ap.add_argument("--classifier", choices=["vlm", "model"], default="vlm",
                    help="stage 2 实现：vlm=提示词分类；model=自训分类器（需同时给 --cls-weights，"
                         "可配合 --engine yolo 走「YOLO 提框 + CNN 分类」）")
    ap.add_argument("--cls-weights", default="",
                    help="--classifier model 的权重（原工程训练产出的 best.pt）")
    ap.add_argument("--out-dir", default="",
                    help="产物根（默认 results/detection/sample_detect/<sample>）；run_pipeline 指向"
                         " runs/<run_id>/detection 以保持「一个 run = 一个工件集」")
    args = ap.parse_args()
    # 显式拒绝不兼容组合（不静默取其一）：两阶段需要 crop 图做 stage 2，且多轮采样是另一套机制
    if args.two_stage and args.skip_crop:
        sys.exit("[FAIL] --two-stage 与 --skip-crop 互斥：stage 2 必须在 crop 图上判类别")
    if args.two_stage and args.rollouts > 1:
        sys.exit("[FAIL] --two-stage 与 --rollouts 互斥：两者是不同机制（拆分 vs 重采样），"
                 "同时开会让误差无法归因")
    if args.classifier == "model":
        if not args.two_stage:
            sys.exit("[FAIL] --classifier model 必须配 --two-stage：分类器只判 stage 1 给出的框")
        if not args.cls_weights:
            sys.exit("[FAIL] --classifier model 需要 --cls-weights（原工程训练产出的 best.pt）")
        if not os.path.isfile(args.cls_weights if os.path.isabs(args.cls_weights)
                              else os.path.join(PROJECT_ROOT, args.cls_weights)):
            sys.exit("[FAIL] 分类器权重不存在：{}".format(args.cls_weights))
    elif args.cls_weights:
        sys.exit("[FAIL] --cls-weights 只在 --classifier model 下生效，请勿静默传入")
    if args.engine != "vlm":
        # YOLO 路径没有提示词/多轮采样；stage 2 只在 model 分类器下成立（YOLO 提框 + CNN 分类）。
        if args.two_stage and args.classifier != "model":
            sys.exit("[FAIL] --engine yolo 与 VLM 两阶段互斥；要两阶段请加 "
                     "--two-stage --classifier model --cls-weights <best.pt>")
        if args.rollouts > 1:
            sys.exit("[FAIL] --engine yolo 与 --rollouts 互斥：多轮采样是 VLM 口径")
        if args.dir_prior:
            sys.exit("[FAIL] --engine yolo 与 --dir-prior 互斥：YOLO 不使用提示词先验")

    items = resolve_images(args)
    dir_name = prior_dir_name(items[0][0], items[0][1])   # 样品身份（不是 images/ 这类容器名）
    sample = args.sample or dir_name
    out_dir = os.path.abspath(args.out_dir) if args.out_dir else os.path.join(OUT_ROOT, sample)
    crops_dir = os.path.join(out_dir, "crops")
    review_dir = os.path.join(out_dir, "review")
    for d in (crops_dir, review_dir):
        os.makedirs(d, exist_ok=True)
    herb = args.herb or _full_dir_herb(dir_name)
    feature = next((v for k, v in PRIOR_KEYWORDS if k in dir_name and v), None)
    # 目录先验**默认不注入**（2026-09-30）：本入口产出的框是判药证据，而目录名对送检样品而言
    # 是**待验证身份**，注进提示词即标签泄露（"检测器被告知答案"→ 结论不再是独立证据）。
    # 药材名仍从目录名推导并写进 crop 随行 json——那是溯源字段，只是不进提示词。
    # 显式 opt-in：--dir-prior 且确有特征词可注入时才构造三元组。
    dir_prior = (dir_name, herb, feature) if (args.dir_prior and feature) else None
    print(f"样品: {sample} | 图像 {len(items)} 幅 | 药材(仅溯源字段): {herb}")
    print(f"目录先验: {dir_prior[2] if dir_prior else '不注入（送检口径；标注语料用 --dir-prior）'}")
    print(f"输出: {out_dir}")

    # vlm 引擎必须能解析 API key；yolo 引擎不调 VLM 检测（比例尺 B 引擎失败会回退 A，
    # 见 scalebar_duo），故不因缺 key 直接退出。
    cfg, api_key = load_config_checked(need_key=(args.engine == "vlm"))
    vlm_cfg = cfg["vlm"]
    yolo_cfg = yolo_settings(args, cfg) if args.engine == "yolo" else None
    if yolo_cfg:
        print(f"引擎: yolo | 权重 {os.path.relpath(yolo_cfg['weights'], PROJECT_ROOT)}"
              f" | imgsz {yolo_cfg['imgsz']} conf {yolo_cfg['conf']} iou {yolo_cfg['iou']}"
              f" tile {yolo_cfg['tile']} overlap {yolo_cfg['overlap']}")
    else:
        print(f"引擎: vlm | 模型 {vlm_cfg.get('model', '')}")

    all_crops = []
    seq_by_feat = {}
    other_rows = []            # 「其他」类：识别照常、交付过滤（落 other_detections.json）
    raw_by_image = []          # 仅 --rollouts>1 填充：逐图逐轮原始框（落 rollouts.json）
    rejected = []              # 仅 --two-stage 填充：被 stage 2 拒的框（落 stage2_rejected.json）
    t0 = time.time()
    for idx, (img_path, _) in enumerate(items, 1):
        stem = os.path.splitext(os.path.basename(img_path))[0]
        img = imread_cn(img_path)
        if img is None:
            print(f"[{idx}/{len(items)}] FAIL 读取: {img_path}")
            continue
        H, W = img.shape[:2]
        px_per_um = lookup_px_per_um(img_path, img)   # 检测阶段双引擎标定（离线标定表已停用）
        print(f"[{idx}/{len(items)}] {stem} {W}x{H} px_per_um={px_per_um or '未标定'}")

        # 目录先验默认不注入（见 main 里 dir_prior 的构造）；检测器只看「图 + 类别定义」。
        if args.engine == "yolo":
            if args.two_stage:
                # 「YOLO 提框 + CNN 分类」：stage 1 只回框与分数，不依赖 14 类表。
                _preds = detector_yolo.detect_agnostic(
                    img, weights=yolo_cfg["weights"], imgsz=yolo_cfg["imgsz"],
                    conf=yolo_cfg["conf"], iou=yolo_cfg["iou"], device=yolo_cfg["device"],
                    max_det=yolo_cfg["max_det"], tile=yolo_cfg["tile"],
                    overlap=yolo_cfg["overlap"])
                dets = [{"label": ENTITY_LABEL, "tight_bbox": p["box"], "morphology": "",
                         "confidence": ("high" if p["score"] >= 0.75 else
                                        "medium" if p["score"] >= 0.40 else "low"),
                         "score": p["score"]} for p in _preds]
            else:
                # P2 适配器：输出与 detect_adaptive 同形的框（原图像素坐标、同一夹紧规则）。
                dets = detector_yolo.detect(
                    img, weights=yolo_cfg["weights"], imgsz=yolo_cfg["imgsz"],
                    conf=yolo_cfg["conf"], iou=yolo_cfg["iou"], device=yolo_cfg["device"],
                    max_det=yolo_cfg["max_det"], tile=yolo_cfg["tile"],
                    overlap=yolo_cfg["overlap"], classes=yolo_cfg["classes"])
        elif args.rollouts > 1:
            # 多轮自洽采样（V1）：同图同提示词采样 N 轮、按出现轮数聚簇，**只加票不过滤**
            # 第二个返回值是**逐轮原始框**，连同聚簇结果一起落盘（`rollouts.json`）——
            # 它是事后做"忽略标签"再聚簇（V3）或换 thr 复盘的唯一数据源。
            dets, raw = detect_consensus(img, vlm_cfg, api_key, n=args.rollouts,
                                         temp=args.rollout_temp, dir_prior=dir_prior)
            raw_by_image.append({"stem": stem, "source_image": img_path,
                                 "image_size": [W, H], "rollouts": raw})
        else:
            if args.two_stage:
                # ── stage 1：无类别提框（只给位置与边界，**不命名**）────────────────
                # 提示词换成 PROMPT_S1、白名单只留「实体」、缺 label 的对象按兜底标签收下。
                # 与单阶段的差别只有这三处，编码/调用/解析/坐标映射仍是同一份 detect_once。
                dets = detect_adaptive(img, vlm_cfg, api_key, dir_prior, prompt=PROMPT_S1,
                                       whitelist={ENTITY_LABEL}, default_label=ENTITY_LABEL)
            else:
                dets = detect_adaptive(img, vlm_cfg, api_key, dir_prior)   # 内含跨块 NMS
        # 「其他」是第 14 个训练/评测正类（识别照常），但没有 IGM 特征描述（交付过滤）：
        # 只让 delivery 进 crops/，other 另落 other_detections.json（方案 §6.4）。
        if args.two_stage:
            # stage 1 的框没有类别，「交付/其他」分流留到 stage 2 之后；先全部送分类。
            delivery, other = dets, []
        else:
            delivery, other = split_delivery(dets)
        for d in other:
            other_rows.append({"stem": stem, "source_image": img_path,
                               "label": d["label"], "tight_bbox": d["tight_bbox"],
                               "confidence": d["confidence"], "score": d["score"],
                               "engine": args.engine})
        dets = delivery
        print(f"    检测 {len(delivery) + len(other)} 框（交付 {len(delivery)} / 其他 {len(other)}）")

        texts = []
        s = min(1.0, 1536 / max(H, W))
        ov = cv2.resize(img, (int(W * s), int(H * s)),
                        interpolation=cv2.INTER_AREA) if s < 1.0 else img.copy()
        for d in dets:
            x1, y1, x2, y2 = d["tight_bbox"]
            crop = None
            if not args.skip_crop:
                # 松框走**共享原语**（全流程约定），外扩量见 DET_LOOSE_MARGIN_RATIO 的注释：
                # 不能收小 —— 极性探针要四角是背景，且 VLM 框可能欠覆盖。
                ox, oy, cx2, cy2 = loose_box(d["tight_bbox"], W, H,
                                             margin_ratio=DET_LOOSE_MARGIN_RATIO)
                crop = img[oy:cy2, ox:cx2]
                tight_crop = [round(x1 - ox, 1), round(y1 - oy, 1),
                              round(x2 - ox, 1), round(y2 - oy, 1)]
            else:
                tight_crop = None
            # ── 类别来源 ────────────────────────────────────────────────────────────
            # 默认（单阶段）：stage 1 直接给出类别。`--two-stage`：stage 1 只给位置，
            # 类别由 stage 2 在**放大后的 crop** 上判（见 docs/检测两阶段_…）。
            stage2 = None
            if args.two_stage:
                stage2 = classify_crop(crop, vlm_cfg, api_key,
                                       ctx=_geom_ctx(d["tight_bbox"], W, H),
                                       impl=args.classifier,
                                       weights=args.cls_weights or None)
                if not stage2.get("feature_id"):
                    # 无类别 ⇒ **不进交付集**：crop 缺 feature_id 会被 check_crops 拦下（喂不进 IGM）。
                    # 但信息不丢——落 stage2_rejected.json，供复盘"拒了多少、为什么拒"。
                    rejected.append({"stem": stem, "tight_box": d["tight_bbox"],
                                     "appearance": d["morphology"], **stage2})
                    continue
                label, fid = stage2["label"], stage2["feature_id"]
            else:
                label, fid = d["label"], FEATURE_ID[d["label"]]
            seq_by_feat[fid] = seq_by_feat.get(fid, 0) + 1
            crop_id = f"{fid}_{stem}_{seq_by_feat[fid]:02d}"
            if not args.skip_crop:
                cv2.imencode(".png", crop)[1].tofile(os.path.join(crops_dir, crop_id + ".png"))
                # 每 crop 随行 json —— 「检测 → 分割」输入契约（<crop_id>.json + .png 成对）。
                # 键表走契约层 crop_sidecar()，不在本文件另抄一份；缺它 crops/ 就喂不进 --crops。
                side = contracts.crop_sidecar(
                    crop_id, crop_id + ".png", img_path, stem, fid, label, herb,
                    "" if px_per_um is None else px_per_um,
                    [round(v, 1) for v in d["tight_bbox"]], tight_crop, [ox, oy, cx2, cy2],
                    f"{crop.shape[1]}x{crop.shape[0]}",
                    # 框粒度（量化可用性闸门）：两阶段由 stage 2 给；YOLO 不输出粒度，
                    # 按 vlm_config → yolo.classes 的类级默认回填（方案 §6.4）；
                    # VLM 单阶段仍沿用 instance（不改变既有产物语义）。
                    granularity=((stage2 or {}).get("granularity")
                                 or (yolo_cfg or {}).get("granularity", {}).get(label)
                                 or contracts.GRANULARITY_DEFAULT))
                with open(os.path.join(crops_dir, crop_id + ".json"), "w", encoding="utf-8") as sf:
                    json.dump(side, sf, ensure_ascii=False, indent=2)
            row = {
                "crop_id": crop_id, "feature_id": fid, "object_label": label,
                "tight_box_crop": tight_crop,
                "tight_box_source": d["tight_bbox"],
                "crop_size": None if args.skip_crop else [crop.shape[1], crop.shape[0]],
                "px_per_um": px_per_um,
                "scale_status": "OK" if px_per_um else "未标定(双引擎未取得双读一致)",
                "confidence": d["confidence"], "score": d["score"],
                "morphology": d["morphology"],
                "source_image": img_path,          # 内部溯源；下游契约不依赖
            }
            if args.rollouts > 1:
                # 多轮采样的诊断字段**只在 N>1 时落盘**：默认路径的产物形状保持逐字段不变
                row["votes"] = d.get("votes")
                row["rollout_n"] = d.get("rollout_n")
            if stage2 is not None:
                # stage 2 的判据与质量信号（reject/focus）——**只在两阶段时落盘**
                row["stage2"] = {k: stage2.get(k) for k in
                                 ("label", "reject", "confidence", "second", "reason",
                                  "source", "focus", "raw_label", "granularity")}
            all_crops.append(row)
            bx = [int(v * s) for v in d["tight_bbox"]]
            cv2.rectangle(ov, (bx[0], bx[1]), (bx[2], bx[3]), (0, 0, 255), 2)
            tag = f'{label}({d["confidence"]})'
            if args.rollouts > 1:
                tag += f'·{d.get("votes", 1)}票'      # 目检图直接标票数，便于人工判"几轮都看到了"
            texts.append((tag, (bx[0], max(2, bx[1] - 22)), (0, 0, 255), 18))
        ov = textvis.draw_texts_cn(ov, texts)
        cv2.imencode(".jpg", ov, [cv2.IMWRITE_JPEG_QUALITY, 88])[1].tofile(
            os.path.join(review_dir, f"detect_{stem}.jpg"))

    crops_json = os.path.join(out_dir, "crops.json")
    manifest = {"sample": sample, "n_images": len(items), "n_crops": len(all_crops),
                "herb": herb, "engine": args.engine, "crops": all_crops}
    # 溯源字段（additive）：同一份 crops.json 必须能自证是哪个引擎/权重/阈值产出的。
    if yolo_cfg:
        manifest["detector"] = {
            "engine": "yolo",
            "weights": os.path.relpath(yolo_cfg["weights"], PROJECT_ROOT),
            "imgsz": yolo_cfg["imgsz"], "conf": yolo_cfg["conf"], "iou": yolo_cfg["iou"],
            "device": yolo_cfg["device"], "max_det": yolo_cfg["max_det"],
            "tile": yolo_cfg["tile"], "overlap": yolo_cfg["overlap"],
            "class_names": [c["zh"] for c in yolo_cfg["classes"]],
        }
    else:
        manifest["detector"] = {"engine": "vlm", "model": vlm_cfg.get("model", ""),
                                "prompt_version": PROMPT_VERSION}
    manifest["n_other"] = len(other_rows)      # 检出但未进交付的「其他」类框数
    if args.rollouts > 1:
        # 产物自描述：N 轮采样的结果必须能自证是哪套参数产出的（默认路径不加这两个键）
        manifest["rollouts"] = args.rollouts
        manifest["rollout_temp"] = args.rollout_temp
    if args.two_stage:
        # 同上：两阶段的产物必须能自证是哪两版提示词产出的
        manifest["two_stage"] = True
        manifest["stage1_prompt_version"] = PROMPT_VERSION_S1
        manifest["stage2_version"] = CLASSIFY_VERSION
        manifest["classifier"] = args.classifier
        if args.classifier == "model":
            manifest["cls_weights"] = args.cls_weights
        manifest["n_rejected"] = len(rejected)
    with open(crops_json, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    if other_rows:
        # 识别 ≠ 交付：这些框没有 IGM 特征描述，不能进 crops/；但它们是检测器真实输出，
        # 留档用于复盘误报/后续把 other 接入 taxonomy 时回填（方案 §6.4）。
        oj = os.path.join(out_dir, "other_detections.json")
        with open(oj, "w", encoding="utf-8") as f:
            json.dump({"sample": sample, "herb": herb, "engine": args.engine,
                       "n_other": len(other_rows),
                       "note": "「其他」类检测框：识别照常、但不进交付 crops/（IGM 无该特征）。"
                               "不是负样本，勿称 rejected",
                       "items": other_rows}, f, ensure_ascii=False, indent=2)
    if args.rollouts > 1:
        # 逐轮原始框单独成文件（不进 crops.json：那里是供复核的聚合清单，塞原始框会撑爆它）。
        # 契约归属：这是**实验口径的本地诊断件**，不进五段交付链，形状在本处与方案文档定义；
        # 生产路径（--rollouts 1）永不产出本文件。
        rj = os.path.join(out_dir, "rollouts.json")
        with open(rj, "w", encoding="utf-8") as f:
            json.dump({"sample": sample, "herb": herb, "rollouts": args.rollouts,
                       "rollout_temp": args.rollout_temp, "cluster_thr": 0.5,
                       "prompt_version": PROMPT_VERSION,
                       "note": "逐轮原始框：事后做忽略标签的再聚簇（V3）或换 thr 复盘的唯一数据源；"
                               "只含成功轮，每图成功轮数 = len(rollouts)",
                       "images": raw_by_image}, f, ensure_ascii=False, indent=2)
        print(f"  逐轮原始框 → {rj}")
    if args.two_stage and rejected:
        rj = os.path.join(out_dir, "stage2_rejected.json")
        with open(rj, "w", encoding="utf-8") as f:
            json.dump({"sample": sample, "herb": herb, "stage2_version": CLASSIFY_VERSION,
                       "note": "被 stage 2 拒（不是特征 / 图像不可判 / 非法答案）而**未进交付集**的框。"
                               "缺 feature_id 的 crop 喂不进 IGM，故不入 crops/；留档供复盘"
                               "『拒了多少、为什么拒』",
                       "items": rejected}, f, ensure_ascii=False, indent=2)
        from collections import Counter as _C
        why = _C(str(r.get("label") or "非法答案") for r in rejected)
        print(f"  被 stage 2 拒 {len(rejected)} 个框 {dict(why)} → {rj}")
        if any(r.get("reject") == "unreadable" for r in rejected):
            print("    （其中「图像不可判」占比高 = 上游采集/切片质量问题，见 docs §4.4）")
    from collections import Counter
    stat = Counter(c["object_label"] for c in all_crops)
    print(f"\n完成 {time.time()-t0:.0f}s：{len(items)} 幅图 → {len(all_crops)} 个目标子图")
    print("  按类:", dict(stat))
    print(f"  子图 + 随行 json → {crops_dir}")
    print(f"  聚合清单 → {crops_json}")
    if other_rows:
        print(f"  「其他」类 {len(other_rows)} 框（识别留档、不进交付） → "
              f"{os.path.join(out_dir, 'other_detections.json')}")
    print(f"  (分割阶段) python 图像分割\\cli\\igm_segment_batch.py "
          f"--crops \"{crops_dir}\" --out-dir <run>\\segmentation")
    print(f"  (整链) python run_pipeline.py --images \"{os.path.dirname(items[0][0])}\" "
          f"--run <run_id> --sample \"{sample}\"")


# 容器目录名：`--dir <样品>/images` 这类给法的先验不能用容器名（见 prior_dir_name）
CONTAINER_DIRS = ("images", "imgs")


# 松框外扩量 `DET_LOOSE_MARGIN_RATIO` 已上收到 `detect_core`（2026-11-05）：stage 2 分类器
# 训练集必须用**同一 crop 口径**重切，常量若留在这里，再过一层就会漂成两份。


def _geom_ctx(tight_bbox, W, H):
    """给 stage 2 的**纯几何先验**（来自原图坐标，不需要再读原图）。

    这是分解白拿的增量：一阶段的 prompt 只能给"整图"，两阶段可以同时给"放大后的 crop +
    局部几何"（长宽比/尺寸/占画面比）。长宽比对"细长 vs 近方"这类易混对最有用。
    """
    x1, y1, x2, y2 = [float(v) for v in tight_bbox]
    w, h = max(1e-6, x2 - x1), max(1e-6, y2 - y1)
    ar = w / h
    shape = "细长条形" if (ar >= 3 or ar <= 1 / 3) else ("近方形" if 0.8 <= ar <= 1.25 else "长方形")
    return (f"- 候选框长宽比 {ar:.2f}（{shape}），原图内尺寸约 {w:.0f}×{h:.0f} 像素"
            f"（占画面 {100.0 * w * h / max(1.0, W * H):.1f}%）")


def prior_dir_name(img_path, dir_name):
    """先验用的"样品名"：给的是 `<样品>/images` 这类**容器目录**时，退回上一级目录名。

    药材/特征先验来自**样品身份**，不是容器目录名——否则 `--dir <样品>/images` 会把药材推成
    "images"、特征先验取不到，随行 json 的 `herb` 也跟着错，一路污染到几何清单与判药。
    图像就在容器目录**里面**，故从图像路径往上退两级才是样品目录。
    """
    if dir_name in CONTAINER_DIRS:
        up = os.path.basename(os.path.dirname(os.path.dirname(os.path.abspath(img_path))))
        if up and up not in CONTAINER_DIRS:
            return up
    return dir_name


def _full_dir_herb(dir_name):
    """从目录名推导药材名（去特征关键词）。"""
    herb = dir_name
    for kw, lab in PRIOR_KEYWORDS:
        if kw in herb:
            herb = herb.replace(kw, "").replace("草酸钙", "")
            return herb or dir_name
    return dir_name


def yolo_settings(args, cfg):
    """读 `detector.json` 推理键（解析实现与 det_eval 共用 `detector_yolo.load_settings`）。"""
    try:
        return detector_yolo.load_settings(cfg, args.detector_config or None,
                                           args.weights or None)
    except (OSError, ValueError, RuntimeError) as e:
        sys.exit(f"[FAIL] YOLO 推理配置不可用：{e}")


def load_config_checked(need_key=True):
    cfg = load_config()
    if not need_key:
        return cfg, ""
    api_key = resolve_key(cfg)
    if not api_key:
        sys.exit("[FAIL] 未配置 VLM API key（VLM_API_KEY 环境变量或 vlm_config.json）")
    return cfg, api_key


if __name__ == "__main__":
    main()
