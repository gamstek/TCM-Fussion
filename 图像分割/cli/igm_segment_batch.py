# -*- coding: utf-8 -*-
"""igm_segment_batch.py — 样品级 IGM 分割：crop 小图目录 → 几何测量交付（mask + 交付清单）

把 IGM（图像生成大模型）分割接到几何测量之前：同一样品由检测阶段裁出的多个 crop 小图，
逐图跑 `igm_segment.py`（按随行 json 的 feature_id 取该特征形态描述），再把产物对齐
**几何测量输入契约**（`几何测量/geomlib/mask_io.py` + `*_measure_manifest.csv` 标定），
几何与判药因此零改动。

输入: --crops 样品 crop 目录（也可直接给含 crops/ 的样品目录）
      `<crop_id>.png` 松框小图 + `<crop_id>.json` 同名随行元数据（检测阶段产物：
      feature_id / herb / px_per_um / stem / tight_box_local / loose_box …）。
      px_per_um 是几何标定的唯一来源（2026-09-20 契约），任一 crop 缺失即报错退出
      （不静默降级——那会产出无 μm 的测量行）。

输出（--out-dir，run_pipeline 用 runs/<run_id>/segmentation/）:
  masks/{stem}__{feature_id}__inst{k}.png   几何契约每实例 mask（crop 局部坐标；同一源图
                                            跨 feature 连续编号，避免 (stem,inst) 主键冲突）
  masks/{stem}__measure_manifest.csv        交付清单（px_per_um / herb / source_image_path…）
  masks/crops/{crop_id}.json + .png         侧车副本：几何叠加图按 loose_box 回贴源图
  igm/{crop_id}/                            逐 crop IGM 原始产物（mask / gen_raw / overlay / meta）
  instances.json                            实例清单（crop_id → mask 文件、IGM 状态）

用法:
  python igm_segment_batch.py --crops "..\\..\\流水线数据\\input\\虎杖草酸钙簇晶\\crops" ^
      --out-dir "..\\..\\流水线数据\\runs\\<id>\\segmentation" [--force] [--dry-run] [--selftest]
"""
import argparse
import csv
import json
import os
import shutil
import sys

SEG_DIR = os.path.dirname(os.path.abspath(__file__))

import contracts  # noqa: E402 数据契约唯一事实源（项目根 contracts.py）
import igm_segment as ig  # noqa: E402

MANIFEST_SUFFIX = contracts.MANIFEST_SUFFIX
CSV_COLUMNS = list(contracts.MANIFEST_COLUMNS)


def resolve_crops_dir(path):
    """允许直接传样品目录（内含 crops/）或 crop 目录本身。"""
    sub = os.path.join(path, "crops")
    return sub if os.path.isdir(sub) else path


def load_crops(crops_dir):
    """读 `<crop_id>.json` 随行元数据 + 同名 png → 列表（按 crop_id 升序）。

    侧车 json 是单图自包含契约（分割方案 §1.4）：crop 身份、特征、标定、源图出处都在里面，
    故本函数不做任何目录级猜测——缺字段即在此处报错，绝不静默补默认值。
    """
    items = []
    try:
        metas = contracts.load_crop_metas(crops_dir)
    except contracts.ContractError as e:      # 子图缺失/不可读 → 直接给出契约级错误
        sys.exit(str(e))
    for m in metas:
        items.append({
            "crop_id": m.crop_id, "png": m.png, "json_path": m.json_path, "cj": m.raw,
            "feature_id": m.feature_id,
            "object_label": m.object_label,
            "herb": m.herb,
            "px_per_um": m.px_per_um_text,     # 原样带出（清单 CSV 契约要求逐字一致）
            "stem": m.stem,
            "source_image_path": m.source_image_path,
            "meta": m,                          # 契约对象（校验/取值为类型化访问）
        })
    if not items:
        sys.exit(f"[FAIL] {crops_dir} 下没有 crop 随行 json"
                 f"（期望 <crop_id>.json + <crop_id>.png，检测阶段产物）")
    return items


def _pos_float(s):
    try:
        v = float(s)
    except (TypeError, ValueError):
        return 0.0
    return v if v > 0 else 0.0


def check_items(items):
    """入口校验：feature_id / stem / px_per_um 三样缺一不可（一次列全，便于批量补）。"""
    try:
        contracts.check_crops([it["meta"] for it in items])   # 校验唯一发生在契约层
    except contracts.ContractError as e:
        sys.exit(str(e))


def box_local(cj, w, h):
    """随行 json → crop 局部紧框 [x1,y1,x2,y2]。

    tight_box_local 优先；缺失时由 tight_box（源图坐标）− loose_box 原点换算；
    两者都缺则退回全图（仅清单信息用，几何不读此列）。
    """
    if cj.get("tight_box_local"):
        box = cj["tight_box_local"]
    elif cj.get("tight_box") and cj.get("loose_box"):
        tb, lb = cj["tight_box"], cj["loose_box"]
        box = [tb[0] - lb[0], tb[1] - lb[1], tb[2] - lb[0], tb[3] - lb[1]]
    else:
        box = [0, 0, w, h]
    return [int(round(float(v))) for v in box[:4]]


def run_igm(it, igm_dir, force=False, dry_run=False, box_association="center_point", size=""):
    """调 igm_segment.main() 分割单个 crop（in-process，同 experiments/igm_exp_common 的做法）。

    带上随行 json 的 tight_box_local 作 `--box`：启用目标框先验（丢框外邻粒 + 裁到框内），
    否则画面边缘露出的同类别实体会并入同一实例，几何把全部连通域点合起来求凸包 → 尺寸虚高
    （2026-09-29 实测：淀粉粒_img3_b2 多算 3 块邻粒，Feret 16.42 μm vs 7.61 μm）。
    框字段缺失时不传 --box（退回旧行为，不做框先验）。

    幂等：igm_segment 见该目录已有 mask.png 即跳过（不耗 API），--force 才重跑。
    返回 (ok, meta|None)——meta 缺失不算失败（离线复用已有 mask 时没有 meta 也照常组装）。
    """
    granularity = getattr(it.get("meta"), "granularity", "") or contracts.GRANULARITY_DEFAULT
    argv = ["igm_segment.py", "--image", it["png"], "--feature", it["feature_id"],
            "--granularity", granularity,
            "--out-dir", igm_dir, "--quiet", "--box-association", box_association]
    cj = it.get("cj") or {}
    if cj.get("tight_box_local") or (cj.get("tight_box") and cj.get("loose_box")):
        img = ig.imread_any(it["png"])
        if img is not None:
            box = box_local(cj, img.shape[1], img.shape[0])
            argv += ["--box", ",".join(str(v) for v in box)]
    if size:
        argv += ["--size", size]
    if force:
        argv.append("--force")
    if dry_run:
        argv.append("--dry-run")
    saved = sys.argv
    sys.argv = argv
    try:
        rc = ig.main()
    except SystemExit as e:                       # igm_segment 用 sys.exit 报错
        rc = e.code if isinstance(e.code, int) else 1
    finally:
        sys.argv = saved
    if rc != 0 or not os.path.exists(os.path.join(igm_dir, "mask.png")):
        return False, None
    meta = None
    meta_path = os.path.join(igm_dir, "meta.json")
    if os.path.exists(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
    return True, meta


def _clean_delivery(masks_dir):
    """清掉上一次组装遗留的交付文件，避免"新编号 + 旧残留"造成实例串号。

    重跑时若 crop 集合变化（新增/删除 crop），同一源图的实例编号会整体位移：旧文件不会被
    同名覆盖，会继续留在目录里 → (源图stem, instance_id) 出现重复或凭空多出实例
    （实测 2026-09-28：img2 同时残留旧 vessel__inst01 与新写的 cluster__inst01/vessel__inst02，
    几何测量因此凭空多出 1 条记录，且触发 mask_io 的跨 feature 冲突告警）。
    只清本模块自己产出的两类文件（`*__inst*.png` 与 `*_measure_manifest.csv`）与 `crops/` 侧车副本。
    """
    if not os.path.isdir(masks_dir):
        return
    for fn in os.listdir(masks_dir):
        p = os.path.join(masks_dir, fn)
        if not os.path.isfile(p):
            continue
        if fn.lower().endswith("_measure_manifest.csv") or (
                "__inst" in fn and fn.lower().endswith(".png")):
            os.remove(p)
    shutil.rmtree(os.path.join(masks_dir, "crops"), ignore_errors=True)


def assemble(items, results, out_dir):
    """按几何契约落交付：每实例 mask + 每源图清单 + crop 侧车副本 + 实例清单。

    实例编号 k 在**同一源图内跨 feature 连续**（mask_io 契约：否则 (image_stem, instance_id)
    主键跨 feature 冲突，测量表与叠加图文件名互相覆盖）。
    写盘前先清旧交付（`_clean_delivery`）——否则重跑换了 crop 集合时，旧编号的文件会留下。
    """
    masks_dir = os.path.join(out_dir, "masks")
    sidecar_dir = os.path.join(masks_dir, "crops")
    _clean_delivery(masks_dir)
    os.makedirs(sidecar_dir, exist_ok=True)
    rows_by_stem, inst_rows = {}, []
    for stem in sorted({it["stem"] for it in items}):
        group = sorted((it for it in items if it["stem"] == stem),
                       key=lambda x: (x["feature_id"], x["crop_id"]))
        rows = []
        for k, it in enumerate((g for g in group if results[g["crop_id"]]["ok"]), 1):
            res = results[it["crop_id"]]
            window = it["cj"].get("feature_window") or {}
            igm_meta = res["meta"] or {}
            name = f"{stem}__{it['feature_id']}__inst{k:02d}.png"
            shutil.copyfile(os.path.join(out_dir, "igm", it["crop_id"], "mask.png"),
                            os.path.join(masks_dir, name))
            img = ig.imread_any(it["png"])
            mask = ig.imread_any(os.path.join(masks_dir, name))
            if img is not None and mask is not None and mask.shape[:2] != img.shape[:2]:
                print(f"[warn] {it['crop_id']}: mask 尺寸 {mask.shape[:2]} ≠ crop 尺寸 "
                      f"{img.shape[:2]}（不应发生，请检查 IGM 产物）")
            rows.append({
                "instance_id": k, "inst_mask": name, "crop_id": it["crop_id"],
                "source_image_path": it["source_image_path"], "feature_id": it["feature_id"],
                "object_label": it["object_label"], "herb": it["herb"],
                "tight_bbox": json.dumps(box_local(it["cj"], *(img.shape[:2][::-1] if img is not None else (0, 0)))),
                "px_per_um": it["px_per_um"],
                "igm_status": (res["meta"] or {}).get("status", ""),
                "igm_area_ratio": (res["meta"] or {}).get("area_ratio", ""),
                "feature_window_schema": window.get("schema", ""),
                "igm_associated_clipped_px": igm_meta.get("associated_clipped_px", igm_meta.get("clipped_px", "")),
                # 框粒度（量化可用性闸门）：由检测侧 stage 2 逐 crop 给出（crop 随行 json），
                # 缺省 instance。几何侧据此对 bundle/sheet/clump 打 coarse_quant（只作存在性）。
                "granularity": getattr(it["meta"], "granularity", "") or contracts.GRANULARITY_DEFAULT,
            })
            # 侧车副本：几何叠加图按 loose_box 从源图裁回该 crop（源图不可用时退回这张彩色小图）
            shutil.copyfile(it["json_path"], os.path.join(sidecar_dir, it["crop_id"] + ".json"))
            shutil.copyfile(it["png"], os.path.join(sidecar_dir, os.path.basename(it["png"])))
            inst_rows.append({**{k2: it[k2] for k2 in
                                 ("crop_id", "stem", "feature_id", "object_label", "herb", "px_per_um")},
                              "mask_file": f"masks/{name}",
                              "igm_dir": f"igm/{it['crop_id']}",
                              "igm_status": (res["meta"] or {}).get("status", ""),
                              "igm_area_ratio": (res["meta"] or {}).get("area_ratio", ""),
                              "granularity": getattr(it["meta"], "granularity", "")
                                             or contracts.GRANULARITY_DEFAULT,
                              "n_components": (res["meta"] or {}).get("n_components", ""),
                              # 审计字段：极性是否与提示词契约冲突、目标框先验丢/裁了多少
                              # （status=review 时这些数字就是人工复核的依据）
                              "igm_polarity_conflict": (res["meta"] or {}).get("polarity_conflict", ""),
                              "igm_box": (res["meta"] or {}).get("box", ""),
                              "igm_dropped_components": (res["meta"] or {}).get("dropped_components", ""),
                              "igm_dropped_px": (res["meta"] or {}).get("dropped_px", ""),
                              "igm_clipped_px": (res["meta"] or {}).get("clipped_px", ""),
                              "igm_fg_outside_box_ratio": (res["meta"] or {}).get("fg_outside_box_ratio", ""),
                              "igm_touches_border": (res["meta"] or {}).get("touches_border", ""),
                              "igm_border_contact_ratio": (res["meta"] or {}).get("border_contact_ratio", ""),
                              "igm_center_comp_ratio": (res["meta"] or {}).get("center_comp_ratio", "")})
        if not rows:
            continue
        rows_by_stem[stem] = rows
        with open(os.path.join(masks_dir, stem + MANIFEST_SUFFIX), "w",
                  newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            w.writeheader()
            w.writerows(rows)
    with open(os.path.join(out_dir, "instances.json"), "w", encoding="utf-8") as f:
        json.dump({"n_crops": len(items), "n_instances": len(inst_rows),
                   "instances": inst_rows}, f, ensure_ascii=False, indent=2)
    return rows_by_stem


# ---------------------------------------------------------------- 自检（纯离线，不耗 API）
def selftest():
    """离线自检交付组装：编号连续性、清单字段、侧车副本、实例清单（IGM 走"已存在即跳过"）。"""
    import tempfile

    passed, failed = 0, []

    def check(name, cond, extra=""):
        nonlocal passed
        if cond:
            passed += 1
            print(f"  [PASS] {name}")
        else:
            failed.append(f"{name} {extra}")
            print(f"  [FAIL] {name} {extra}")

    tmp = tempfile.mkdtemp(prefix="igm_batch_selftest_")
    try:
        crops_dir = os.path.join(tmp, "crops")
        out_dir = os.path.join(tmp, "out")
        os.makedirs(crops_dir)
        # 两个 crop 同源图、不同 feature → 编号必须跨 feature 连续（inst01/inst02）
        # 第 4 列 granularity：a 显式 bundle（测透传）、b 缺省 None（测 instance 兜底）
        specs = [("a_1_b0", "cluster_crystal", [24, 24, 80, 60], "bundle"),
                 ("b_1_b1", "vessel", [10, 10, 80, 50], None)]
        for cid, fid, local, gran in specs:
            ig.write_img(os.path.join(crops_dir, cid + ".png"),
                         ig.np.full((60, 80, 3), 200, ig.np.uint8), ".png")
            cj = {"crop_id": cid, "png": cid + ".png", "stem": "1",
                  "feature_id": fid, "object_label": fid, "herb": "X",
                  "px_per_um": "10.0", "source_image_path": "D:/x/1.jpg",
                  "tight_box_local": local, "crop_size": "80x60"}
            if gran is not None:
                cj["granularity"] = gran
            with open(os.path.join(crops_dir, cid + ".json"), "w", encoding="utf-8") as f:
                json.dump(cj, f)
            # 预置 IGM 产物 → igm_segment 见 mask.png 已存在即跳过，全程离线
            idir = os.path.join(out_dir, "igm", cid)
            os.makedirs(idir)
            ig.write_img(os.path.join(idir, "mask.png"),
                         ig.np.full((60, 80), 255, ig.np.uint8), ".png")
            with open(os.path.join(idir, "meta.json"), "w", encoding="utf-8") as f:
                json.dump({"status": "ok", "area_ratio": 0.5, "n_components": 1}, f)

        items = load_crops(crops_dir)
        check("装载: 读到 2 个 crop", len(items) == 2, f"got {len(items)}")
        check_items(items)
        # 预置"上一次组装"的残留（旧编号 mask / 旧清单 / 旧侧车录像）→ 必须被清掉，
        # 否则新编号与原残留共存会造出 (stem, inst) 重复与凭空多出的实例
        stale = os.path.join(out_dir, "masks")
        os.makedirs(os.path.join(stale, "crops"), exist_ok=True)
        for n in ("1__vessel__inst09.png", "1__stale_crystal__inst01.png"):
            ig.write_img(os.path.join(stale, n), ig.np.full((10, 10), 255, ig.np.uint8), ".png")
        with open(os.path.join(stale, "1__measure_manifest.csv"), "w", encoding="utf-8") as f:
            f.write("instance_id\n9\n")
        with open(os.path.join(stale, "crops", "老crop.json"), "w", encoding="utf-8") as f:
            f.write("{}")
        results = {}
        for it in items:
            ok, meta = run_igm(it, os.path.join(out_dir, "igm", it["crop_id"]))
            results[it["crop_id"]] = {"ok": ok, "meta": meta}
        check("IGM: 两图均 ok", all(r["ok"] for r in results.values()), f"{results}")
        rows_by_stem = assemble(items, results, out_dir)

        masks_dir = os.path.join(out_dir, "masks")
        m1 = os.path.join(masks_dir, "1__cluster_crystal__inst01.png")
        m2 = os.path.join(masks_dir, "1__vessel__inst02.png")
        check("交付: 同源图跨 feature 连续编号 inst01/inst02",
              os.path.exists(m1) and os.path.exists(m2))
        man = os.path.join(masks_dir, "1" + MANIFEST_SUFFIX)
        check("交付: 清单按源图落盘", os.path.exists(man))
        with open(man, encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        check("清单: 两行且列齐全", len(rows) == 2 and set(rows[0]) == set(CSV_COLUMNS),
              f"rows={len(rows)} cols={sorted(rows[0]) if rows else []}")
        check("清单: px_per_um 原样带出（几何标定唯一来源）",
              all(r["px_per_um"] == "10.0" for r in rows))
        check("清单: 实例编号唯一", sorted(int(r["instance_id"]) for r in rows) == [1, 2])
        # 量化可用性闸门：框粒度随 crop 侧车 → 交付清单透传；缺列/未写 → instance 兜底
        gran_by_crop = {r["crop_id"]: r.get("granularity") for r in rows}
        check("清单: granularity 透传（bundle）/ 缺省兜底（instance）",
              gran_by_crop.get("a_1_b0") == "bundle" and gran_by_crop.get("b_1_b1") == "instance",
              f"got {gran_by_crop}")
        check("侧车: crops/ 副本齐备（json + png）",
              all(os.path.exists(os.path.join(masks_dir, "crops", n))
                  for n in ("a_1_b0.json", "a_1_b0.png", "b_1_b1.json", "b_1_b1.png")))
        check("清理: 上次残留的旧编号 mask 已删",
              not os.path.exists(os.path.join(masks_dir, "1__vessel__inst09.png"))
              and not os.path.exists(os.path.join(masks_dir, "1__stale_crystal__inst01.png")))
        check("清理: 上次残留的旧侧车副本已删",
              not os.path.exists(os.path.join(masks_dir, "crops", "老crop.json")))
        check("实例清单: instances.json 落盘", os.path.exists(os.path.join(out_dir, "instances.json")))
        check("实例清单: 与清单同序", len(rows_by_stem.get("1", [])) == 2)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    total = passed + len(failed)
    if failed:
        print(f"\nSELFTEST FAIL {passed}/{total}")
        for f in failed:
            print(f"  - {f}")
        return 1
    print(f"\nSELFTEST PASS {passed}/{total}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="样品级 IGM 分割：crop 小图目录 → 几何测量交付（mask + 交付清单）")
    ap.add_argument("--crops", default="",
                    help="样品 crop 目录（<crop_id>.png + <crop_id>.json）；"
                         "也可给含 crops/ 的样品目录")
    ap.add_argument("--out-dir", default="",
                    help="交付与 IGM 产物输出目录（run_pipeline 传 runs/<id>/segmentation）")
    ap.add_argument("--force", action="store_true", help="覆盖已有 IGM mask（默认幂等跳过）")
    ap.add_argument("--dry-run", action="store_true", help="只打印 IGM 请求，不发 API、不组装")
    ap.add_argument("--selftest", action="store_true", help="跑离线自检（不耗 API）")
    ap.add_argument("--size", default="", help="显式 IGM 生成尺寸，如 2048x2048；空则使用配置")
    ap.add_argument("--box-association", choices=["center_point", "region_component"], default="center_point",
                    help="目标关联质量策略；原B基线保持center_point")
    args = ap.parse_args(argv)

    if args.selftest:
        return selftest()
    if not args.crops or not args.out_dir:
        sys.exit("[FAIL] 必须指定 --crops 与 --out-dir（或 --selftest）")

    crops_dir = resolve_crops_dir(args.crops)
    if not os.path.isdir(crops_dir):
        sys.exit(f"[FAIL] crop 目录不存在: {args.crops}")
    items = load_crops(crops_dir)
    check_items(items)
    stems = sorted({it["stem"] for it in items})
    print(f"[info] crop {len(items)} 个 | 源图 {len(stems)} 张 {stems[:8]}"
          f"{' …' if len(stems) > 8 else ''} | "
          f"药材 {sorted({it['herb'] for it in items})} | 输出 {args.out_dir}")

    igm_root = os.path.join(args.out_dir, "igm")
    os.makedirs(igm_root, exist_ok=True)
    results = {}
    for i, it in enumerate(items, 1):
        print(f"\n[{i}/{len(items)}] {it['crop_id']} | feature={it['feature_id']} | "
              f"px_per_um={it['px_per_um']}", flush=True)
        ok, meta = run_igm(it, os.path.join(igm_root, it["crop_id"]),
                           force=args.force, dry_run=args.dry_run, box_association=args.box_association,
                           size=args.size)
        results[it["crop_id"]] = {"ok": ok, "meta": meta}
        if not ok and not args.dry_run:
            print(f"  ⚠ IGM 失败/无产物（详见上方 [FAIL] 行）", flush=True)
        elif (meta or {}).get("status") not in (None, "", "ok"):
            print(f"  ⚠ IGM 状态 {meta['status']} → 需人工复核该 mask"
                  f"（极性冲突={meta.get('polarity_conflict')}、"
                  f"中心块/框={meta.get('center_comp_ratio')}、"
                  f"框外占比={meta.get('fg_outside_box_ratio')}）", flush=True)

    if args.dry_run:
        print("\n[dry-run] 只打印 IGM 请求，未落盘、未组装交付")
        return 0

    failed = sorted(k for k, v in results.items() if not v["ok"])
    rows_by_stem = assemble(items, results, args.out_dir)
    n_inst = sum(len(v) for v in rows_by_stem.values())
    print(f"\n[done] 实例 {n_inst} 个 → {os.path.join(args.out_dir, 'masks')}")
    print(f"       IGM 原始产物 → {igm_root}")
    print(f"       实例清单 → {os.path.join(args.out_dir, 'instances.json')}")
    if failed:
        print(f"[FAIL] {len(failed)}/{len(items)} 个 crop 未产出 mask，交付不完整："
              f"{failed}", file=sys.stderr)
        return 2
    print(f"       （几何测量）--masks \"{os.path.join(args.out_dir, 'masks')}\"")
    review = sorted(k for k, v in results.items()
                    if (v["meta"] or {}).get("status") not in (None, "", "ok"))
    if review:
        print(f"[warn] {len(review)} 个 crop 的 mask 状态非 ok：这些实例在判药侧会被整条丢弃（不计证据），需人工复核：{review}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
