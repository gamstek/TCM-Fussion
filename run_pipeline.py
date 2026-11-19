# -*- coding: utf-8 -*-
"""run_pipeline — 流水线编排入口（VLM 检测 → IGM 分割 → 几何测量 → Agent判药）。

把 目标检测 / 图像分割 的 mask 一路跑到 Agent判药 的报告，产物按 run_id 版本化落到
`流水线数据/runs/<run_id>/`（medallion 分层：segmentation=bronze, measurements=silver,
verdicts=gold），三个下游模块代码零改动，仅由本脚本注入路径。

**输入约定**（三选一，互斥）：
  --images 原始显微图目录 —— **从 VLM 目标检测开始**：本脚本先跑检测
           （`detection/cli/detect_sample.py`：逐图 VLM 出框 + 30% 外扩裁 crop + 双引擎标定
           px_per_um，产出 `crops/<crop_id>.json`+`.png`），再进 IGM 分割。
           ⚠ 会同时耗 VLM（检测）与 IGM（分割）两份额度。
  --crops  现成的 crop 小图目录（`crops/<crop_id>.png` + 同名随行 json）——跳过检测，
           先逐图跑 IGM 分割（图像分割/cli/igm_segment_batch.py）组装成几何交付，
           再进几何测量。
  --masks  既有分割交付（mask + `*_measure_manifest.csv`），跳过检测与 IGM，直连几何。
三者都可传 `流水线数据/input/` 下的相对名或绝对路径。
`--images` 是**双重语义**：单独给 = 上述输入（从原始图开始）；与 `--crops`/`--masks` 同给 =
只作「原图根目录」（叠加图 + VLM 证据图）。缺省时自动取样品目录下的 `images/`。
输入按引用不复制（见 run_manifest.json）。

目录结构（run_id 自动生成年月日时分秒，或用 --run 指定）：
  流水线数据/
    ├─ input/<batch>/                    ← 样品输入（原图 / crops/ / 分割交付 + 交付清单）
    └─ runs/<run_id>/
        ├─ detection/                                      ← [仅 --images] VLM 检测交付：crops/ + crops.json + review/
        ├─ segmentation/masks/                             ← IGM 交付：每实例 mask + 交付清单
        ├─ segmentation/igm/<crop_id>/                     ← 逐 crop IGM 原始产物（人眼核对）
        ├─ measurements/<run_id>_measurements.jsonl/.csv   ← 几何测量交付（silver）
        ├─ measurements/overlay/                            ← 叠加图（判药引用）
        ├─ verdicts/                                        ← 判药报告（gold）
        └─ run_manifest.json                                ← provenance：输入/配置/产出快照

用法
  python run_pipeline.py --images 流水线数据\\input\\虎杖草酸钙簇晶\\images ^
      --sample 送检样品A                    # 原始显微图 → VLM 检测 → IGM → 几何 → 判药
  python run_pipeline.py --crops 虎杖草酸钙簇晶 ^
      --run 20260923_igm                    # 样品 crop 小图 → IGM → 几何 → 判药
  python run_pipeline.py --masks manual_masks_measure ^
      --run 20260917_batch01                # 既有分割交付直连几何

可选透传
  --split-policy   all（默认）｜ test_full_train20
  --enable-vlm     启用 Agent(VLM) 证据补全
  --limit N        仅处理前 N 个样本（--images 输入时即前 N 幅图）
  --sample NAME    把所有测量归为一个样本（同一份送检粉末的多视野）
  --force          重跑 IGM（默认见 mask.png 已存在即跳过，不耗 API）；
                   --images 输入时亦重跑 VLM 检测（会再耗 VLM 额度）
  --dry-run        只打印 IGM 请求，不发 API、不落盘（冒烟自检用；仅 --crops 输入可用）
  --python         默认取 .venv\\Scripts\\python.exe，
                   缺失则用解释 run_pipeline 的当前解释器

标定约定（2026-09-20）：几何只认交付清单的 px_per_um（缺失即报错）；
`scale_table.csv` 已全项目停用，px_per_um 由检测阶段的双引擎
（`common/scalebar_duo.py`）测定并随 crop 随行 json / 交付清单流转。
"""
import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import paths   # 路径唯一权威（项目根 paths.py；直启时脚本目录即项目根，venv 下有 .pth 注入点）

HERE = Path(paths.PROJECT_ROOT)
VENV_PY = Path(paths.VENV_PY)
DEFAULT_MANIFEST = HERE / "manifest.csv"           # 2026-10-09：随知识资产上提到项目根
RUNS_ROOT = Path(paths.RUNS_ROOT)
INPUT_ROOT = Path(paths.INPUT_ROOT)


def resolve_input(arg):
    """把 --masks/--images 参数解析为实际路径：绝对路径直接用；相对名在 input 根下解析。"""
    if not arg:
        return None
    p = Path(arg)
    if p.exists():
        return p
    cand = INPUT_ROOT / arg
    return cand if cand.exists() else None


def resolve_python(user_py):
    if user_py:
        return user_py
    if VENV_PY.exists():
        return str(VENV_PY)
    return sys.executable


def run(cmd, cwd):
    print("$ " + " ".join(str(c) for c in cmd))
    r = subprocess.run(cmd, cwd=cwd)
    if r.returncode != 0:
        sys.exit(r.returncode)


def _load_stage_module(module_name, script_path):
    """取阶段模块对象：优先按模块名 import；不在 sys.path 上则按**文件路径**加载。

    P4-4 起 `cli/` 不进 import 注入点（"只做入口"的目录不该被 import），于是
    `--inproc` 对这些阶段（`detect_sample` / `igm_segment_batch`）会 ImportError。
    这里退回按文件路径 exec_module —— 与 tests 加载 CLI 的做法一致，
    使 `--inproc` 与文档承诺一致（几何 `measure` / 判药 `judge` 仍在注入点，走 import）。
    """
    import importlib
    import importlib.util
    try:
        return importlib.import_module(module_name)
    except ImportError:
        spec = importlib.util.spec_from_file_location(module_name, str(script_path))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod


def run_stage(module_name, script_path, argv, cwd, inproc=False, python=None):
    """跑一个阶段脚本。

    默认起子进程：阶段脚本的 stdout/stderr 各自成流、状态不串，崩溃也不带走编排进程。
    `--inproc`（2026-09-29 P2-2）改为进程内 `import <module>; main(argv)`：
    便于调试与单进程跟踪，代价是各阶段共享解释器状态（故默认不开）。
    两种方式都要求阶段脚本暴露 `main(argv=None) -> int` 的可编程入口。
    """
    argv = [str(a) for a in argv]
    if inproc:
        print("[inproc] {} {}".format(module_name, " ".join(argv)))
        rc = _load_stage_module(module_name, script_path).main(argv)
        if rc:
            sys.exit(rc)
        return
    run([resolve_python(python), "-X", "utf8", str(script_path)] + argv, cwd)


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def snapshot_input(path):
    """对单个输入路径做轻量指纹：单文件→内容 sha256；目录→递归列出文件名+逐文件 sha256。

    只记录、不复制任何文件（输入资产按引用，保持不可变语义）。
    """
    p = Path(path)
    if p.is_file():
        return {
            "path": str(p), "type": "file", "size": p.stat().st_size,
            "sha256": _sha256(p),
        }
    if p.is_dir():
        files = sorted(f for f in p.rglob("*") if f.is_file())
        return {
            "path": str(p), "type": "dir", "file_count": len(files),
            "files": [{"name": str(f), "sha256": _sha256(f)} for f in files[:1000]],
            "truncated": len(files) > 1000,
        }
    return {"path": str(p), "type": "missing"}


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="中药显微 · 流水线编排（VLM 检测 → IGM 分割 → 几何测量 → Agent判药）")
    ap.add_argument("--run", default=None, help="run_id（缺省取时间戳）")
    ap.add_argument("--crops", default=None,
                    help="样品 crop 小图目录（含 <crop_id>.png + 随行 json）：先跑 IGM 分割组装"
                         "几何交付，再进几何测量。可为 流水线数据/input/ 下相对名或绝对路径。"
                         "与 --masks 互斥")
    ap.add_argument("--masks", default=None,
                    help="既有实例分割交付（文件/目录/COCO json）：跳过 IGM 直连几何。"
                         "可为 流水线数据/input/ 下相对名或绝对路径。与 --crops 互斥")
    ap.add_argument("--images", default=None,
                    help="原始显微图目录（单独给 = 从 VLM 检测开始的输入，会耗 VLM+IGM 额度）；"
                         "或与 --crops/--masks 同给，只作原图根目录（叠加图 + VLM 证据图）。"
                         "可为 input 下相对名或绝对路径；后者缺省自动取样品目录下的 images/")
    ap.add_argument("--manifest", default=None, help="manifest.csv（缺省用 项目根/manifest.csv）")
    ap.add_argument("--split-policy", choices=["all", "test_full_train20"], default="all")
    ap.add_argument("--enable-vlm", action="store_true", help="启用 Agent(VLM) 证据补全")
    ap.add_argument("--limit", type=int, default=0, help="仅处理前 N 个样本（调试）")
    ap.add_argument("--sample", default=None,
                    help="把所有测量归为一个样本（同一份送检粉末的多视野）；缺省按 manifest 归并")
    ap.add_argument("--force", action="store_true",
                    help="重跑 IGM（默认幂等：见该 crop 已有 mask.png 即跳过，不耗 API）")
    ap.add_argument("--dry-run", action="store_true",
                    help="只打印 IGM 请求，不发 API、不落盘（冒烟自检用）")
    ap.add_argument("--python", default=None, help="运行子模块的解释器")
    ap.add_argument("--inproc", action="store_true",
                    help="进程内调用各阶段（按模块名 import，失败则按文件路径加载）而非起子进程；"
                         "便于单进程调试与跟踪，各阶段因此共享解释器状态（默认关）")
    args = ap.parse_args(argv)

    run_id = args.run or datetime.now().strftime("%Y%m%d_%H%M%S")
    manifest = (Path(args.manifest) if args.manifest
                else (DEFAULT_MANIFEST if DEFAULT_MANIFEST.exists() else None))

    run_dir = RUNS_ROOT / run_id
    meas_dir = run_dir / "measurements"
    overlay_dir = meas_dir / "overlay"
    verdict_dir = run_dir / "verdicts"
    for d in (meas_dir, overlay_dir, verdict_dir):
        d.mkdir(parents=True, exist_ok=True)

    out_prefix = str(meas_dir / run_id)
    meas_jsonl = out_prefix + "_measurements.jsonl"

    # 解析输入：三选一 —— ①--images 原始显微图（先跑 VLM 检测）②--crops 现成 crop 小图（先跑 IGM）
    # ③--masks 既有分割交付（直连几何）。--images 与 ②③ 同给时只作"原图根"（证据图），不作输入。
    n_core = sum(bool(x) for x in (args.crops, args.masks))
    from_images = bool(args.images) and n_core == 0        # 只给 --images = 从原始图开始
    if n_core > 1:
        print("[error] --crops 与 --masks 互斥，只能给一个")
        return 2
    if n_core == 0 and not from_images:
        avail = [p.name for p in sorted(INPUT_ROOT.iterdir())] if INPUT_ROOT.is_dir() else []
        print(f"[error] 必须指定一个输入：--images（原始显微图，先跑 VLM 检测）"
              f"｜--crops（样品 crop 小图，先跑 IGM 分割）｜--masks（既有分割交付）。"
              f"该流水线输入统一放 {INPUT_ROOT}，其中可用：{avail or '（空）'}")
        return 2
    if args.dry_run and not args.crops:
        print("[error] --dry-run 只预演 IGM 阶段（--crops 输入）：打印 IGM 请求但不发 API。"
              "--images 输入要先跑 VLM 检测、--masks 输入没有 IGM，两者都无法预演")
        return 2

    crops_dir = None
    masks_path = None
    sample_dir = None
    detect_src = None                                    # --images 输入时的"原图目录"
    if from_images:
        ip = resolve_input(args.images)
        if ip is None or not ip.is_dir():
            print(f"[error] --images 不存在或不是目录：{args.images}（已在 {INPUT_ROOT} 下也找不到）")
            return 2
        sub = ip / "images"                              # 允许给样品目录（内含 images/）
        detect_src = sub if sub.is_dir() else ip
        sample_dir = ip
    elif args.crops:
        cp = resolve_input(args.crops)
        if cp is None or not cp.exists():
            print(f"[error] --crops 不存在：{args.crops}（已在 {INPUT_ROOT} 下也找不到）")
            return 2
        is_sample_dir = (cp / "crops").is_dir()      # 允许直接给样品目录
        crops_dir = (cp / "crops") if is_sample_dir else cp
        sample_dir = cp if is_sample_dir else cp.parent
    else:
        masks_path = resolve_input(args.masks)
        if masks_path is None or not masks_path.exists():
            print(f"[error] --masks 不存在：{args.masks}（已在 {INPUT_ROOT} 下也找不到）")
            return 2
        sample_dir = masks_path

    if from_images:
        images_path = detect_src                          # 输入目录本身即证据图根
    else:
        images_path = resolve_input(args.images)
        if images_path is None and args.images is not None:
            print(f"[error] --images 不存在：{args.images}")
            return 2
        if images_path is None:                          # 缺省取样品目录下的 images/
            cand = sample_dir / "images"
            images_path = cand if cand.is_dir() else None

    # 子进程 cwd 已切到各模块目录，所有路径必须用绝对路径（相对路径会因 cwd 不同而失效）
    images_abs = str(images_path.resolve()) if images_path else None
    seg_dir = run_dir / "segmentation"
    seg_masks_dir = seg_dir / "masks"

    detect_dir = run_dir / "detection"

    # ⓪a VLM 目标检测（仅 --images 输入）：原始视野图 → 检测框 → 30% 外扩裁 crop + 随行 json。
    # 产物 `crops/<crop_id>.json`+`.png` 正是「检测 → 分割」输入契约，故下一步可直接喂 --crops。
    # 幂等：该 run 的 detection/crops 已有随行 json 即跳过（--force 才重跑，重跑会再耗 VLM 额度）。
    if from_images:
        existing = (sorted((detect_dir / "crops").glob("*.json"))
                    if (detect_dir / "crops").is_dir() else [])
        if existing and not args.force:
            print(f"[skip] 检测产物已存在（{len(existing)} 个 crop）——默认不重跑（重跑耗 VLM 额度），"
                  f"要重跑加 --force")
        else:
            d_argv = ["--dir", str(detect_src.resolve()), "--out-dir", str(detect_dir)]
            if args.sample:
                d_argv += ["--sample", args.sample]
            if args.limit:
                d_argv += ["--limit", str(args.limit)]
            run_stage("detect_sample", Path(paths.DETECTION_CLI_DIR) / "detect_sample.py",
                      d_argv, paths.DETECTION_DIR, args.inproc, args.python)
        crops_dir = detect_dir / "crops"

    # ⓪b IGM 分割（--crops / --images 输入）：逐 crop 生成 mask 并组装成几何测量交付
    # 对齐点：IGM 产物（crop 同尺寸 mask）→ 几何契约（<stem>__<feature>__inst<k>.png +
    # 交付清单 px_per_um + crops/ 侧车），几何/判药因此零改动。
    if crops_dir is not None:
        seg_argv = ["--crops", str(crops_dir.resolve()), "--out-dir", str(seg_dir)]
        if args.force:
            seg_argv.append("--force")
        if args.dry_run:
            seg_argv.append("--dry-run")
        run_stage("igm_segment_batch", Path(paths.SEGMENTATION_CLI_DIR) / "igm_segment_batch.py",
                  seg_argv, paths.SEGMENTATION_DIR, args.inproc, args.python)
        if args.dry_run:
            print("\n[dry-run] 已打印 IGM 请求；几何测量与判药未执行")
            return 0
        masks_path = seg_masks_dir
    masks_abs = str(masks_path.resolve())

    geometry = Path(paths.GEOM_DIR) / "measure.py"
    judge = Path(paths.JUDGE_SCRIPTS_DIR) / "judge.py"

    # ① 几何测量：mask → measurements（silver）
    # --images 用已解析的 images_abs（含缺省推导的 <masks>/images）：measure.py 靠它把
    # 叠加图画在原始视野上，否则轮廓只能画在灰底上（证据图失去视野语境）。
    g = ["--masks", masks_abs, "--out-prefix", out_prefix,
         "--overlay-dir", str(overlay_dir), "--split-policy", args.split_policy]
    if images_abs:
        g += ["--images", images_abs]
    if manifest:
        g += ["--manifest", str(manifest)]
    run_stage("measure", geometry, g, paths.GEOM_DIR, args.inproc, args.python)

    # ② Agent判药：读取几何输出 → verdicts（gold）
    j = ["--measurements", meas_jsonl, "--out", str(verdict_dir), "--overlays", str(overlay_dir)]
    if manifest:
        j += ["--manifest", str(manifest)]
    if images_abs:
        j += ["--images", images_abs]
    if args.enable_vlm:
        j += ["--enable-vlm"]
    if args.limit:
        j += ["--limit", str(args.limit)]
    if args.sample:
        j += ["--sample", args.sample]
    run_stage("judge", judge, j, paths.JUDGE_DIR, args.inproc, args.python)

    # ③ provenance：快照本次输入/配置/产出到 run_manifest.json（只记录，不复制）
    manifest_snap = {
        "run_id": run_id,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "config": {
            "split_policy": args.split_policy,
            "enabled_vlm": bool(args.enable_vlm),
            "limit": args.limit,
            "sample": args.sample,
            "igm_segmentation": bool(crops_dir),        # True = 本轮先跑 IGM 再几何
            "igm_force": bool(args.force),
            "detection": bool(from_images),             # True = 本轮从原始显微图先跑 VLM 检测
        },
        "inputs": {
            "crops": snapshot_input(str(crops_dir.resolve())) if crops_dir else None,
            "masks": None if crops_dir else snapshot_input(masks_abs),
            # --images 输入时检测的输入 = images 本身，故不再重复快照（避免对同一目录算两遍 sha256）
            "images": snapshot_input(images_abs) if images_abs else None,
            "manifest": snapshot_input(str(manifest)) if manifest else None,
        },
        "outputs": {
            "detection": str(detect_dir) if from_images else None,
            "detection_crops": (len(sorted((detect_dir / "crops").glob("*.png")))
                                if from_images else None),
            "segmentation": str(seg_masks_dir) if crops_dir else None,
            "segmentation_instances": (str(seg_dir / "instances.json") if crops_dir else None),
            "measurements": str(meas_jsonl),
            "measurement_records": sum(1 for _ in open(meas_jsonl, encoding="utf-8"))
                                   if Path(meas_jsonl).is_file() else 0,
            "overlay_dir": str(overlay_dir),
            "verdicts": str(verdict_dir),
        },
    }
    manifest_path = run_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest_snap, ensure_ascii=False, indent=2),
                             encoding="utf-8")

    print(f"\n[done] run_id={run_id}")
    if from_images:
        print(f"  detection    → {detect_dir}")
    print(f"  measurements → {meas_jsonl}")
    print(f"  overlays     → {overlay_dir}")
    print(f"  verdicts     → {verdict_dir}")
    print(f"  provenance   → {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())