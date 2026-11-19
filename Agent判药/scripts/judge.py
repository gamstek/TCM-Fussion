# -*- coding: utf-8 -*-
"""judge.py — Agent 判药批量入口（第 5 步 · Agent 最终判断模块）。

输入
  --measurements  measure.py 产出的 *_measurements.jsonl/.csv（文件/目录，可多个）
  --manifest      <项目根>\manifest.csv（可选；用于样本归并与评测 ground truth，只读）
  --kb            判据知识库（默认 ../data/herb_criteria.json）
  --images        原图根目录（可选；启用 VLM 时用于取证据图）
  --enable-vlm    启用 Agent（VLM）证据补全环；默认关闭（纯规则引擎）
  --vlm-config    VLM 配置（默认 目标检测/config/vlm_config.json，只读）

输出
  <out>/reports/<sample>/verdict.json + report.md
  <out>/run_summary.jsonl

用法示例
  python judge.py --measurements ../几何测量/out/pseudo_measurements.jsonl ^
      --manifest <项目根>\manifest.csv --out reports/demo ^
      --overlays ../几何测量/out/pseudo_overlay

安全纪律（设计方案 §1.3）：推理链路不读取 manifest 的 herb 列；测量记录中的
herb 字段在 evidence_pool.load_measurements 中被强制丢弃。
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

_HERE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE_DIR))
sys.path.insert(0, str(_HERE_DIR.parent.parent))   # 项目根：contracts.py

import contracts  # noqa: E402 判决契约（写 verdict.json 前校验）
import evidence_pool  # noqa: E402
import report  # noqa: E402
import scene_attributes  # noqa: E402
from agent_loop import VLMClient, augment_and_rescore, build_image_index  # noqa: E402
from rule_engine import KnowledgeBase, decide  # noqa: E402
from adjudicator import AdjudicatorClient, run_shadow_adjudication  # noqa: E402

HERE = Path(__file__).resolve().parent


def collect_evidence_files(overlay_dirs, sample_records):
    """按测量契约文件名 <image_stem>__inst<instance_id>_测量.jpg 关联叠加图路径。

    只引用 几何测量/out 产物路径，不复制（设计方案 §2.4：零写入上游）。
    """
    files, seen = [], set()
    for rec in sample_records:
        stem, iid = rec.get("image_stem"), rec.get("instance_id")
        if not stem or iid is None:
            continue
        fname = f"{stem}__inst{iid}_测量.jpg"
        if fname in seen:
            continue
        for od in overlay_dirs:
            p = Path(od) / fname
            if p.exists():
                seen.add(fname)
                files.append(str(p))
                break
    return files


def collect_source_images(images_dirs, sample_records):
    """按 image_stem 在 --images 根目录里解析原始视野图（去重），供报告内嵌。

    复用 build_image_index 的 stem→path 索引；返回该样本涉及的源图绝对路径列表。
    """
    if not images_dirs:
        return []
    idx = build_image_index(images_dirs)
    seen, files = set(), []
    for rec in sample_records:
        stem = rec.get("image_stem")
        if not stem or stem in seen:
            continue
        ip = idx.get(stem)
        if ip:
            seen.add(stem)
            files.append(str(ip))
    return files


def main(argv=None):
    ap = argparse.ArgumentParser(description="中药显微 · Agent 判药（测量证据 → 药典判据规则推理）")
    ap.add_argument("--measurements", nargs="+", required=True,
                    help="measurements jsonl/csv 文件或目录（可多个）")
    ap.add_argument("--manifest", default=None, help="manifest.csv（样本归并；herb 列不会进入推理）")
    ap.add_argument("--kb", default=str(HERE.parent / "data" / "herb_criteria.json"))
    ap.add_argument("--out", required=True, help="输出目录（verdict.json/report.md/run_summary.jsonl）")
    ap.add_argument("--images", nargs="*", default=[], help="原图根目录（VLM 证据图用）")
    ap.add_argument("--overlays", nargs="*", default=[],
                    help="measure.py --overlay-dir 产出的叠加图目录（可选；按 <stem>__inst<id>_测量.jpg "
                         "关联为 verdict.evidence_files，只引用路径不复制）")
    ap.add_argument("--enable-vlm", action="store_true", help="启用 Agent（VLM）证据补全环")
    ap.add_argument("--vlm-config",
                    default=str(HERE.parent.parent / "目标检测" / "config" / "vlm_config.json"))
    ap.add_argument("--vlm-max-calls", type=int, default=6, help="单样本 VLM 调用上限")
    ap.add_argument("--shadow-adjudicat", action="store_true",
                    help="启用 LLM 有界仲裁者（方案D，影子模式）：仅对带内模糊案例产出仲裁建议，不改变 verdict")
    ap.add_argument("--shadow-model", default=None, help="影子仲裁 LLM 模型名（默认取 vlm_config）")
    ap.add_argument("--limit", type=int, default=0, help="仅处理前 N 个样本（调试用）")
    ap.add_argument("--sample", default=None,
                    help="把所有测量记录归为这一个样本（同一份送检粉末的多视野，如多张 crop）；"
                         "缺省按 manifest 的文件夹/图 stem 归并")
    args = ap.parse_args(argv)

    kb = KnowledgeBase(args.kb)
    # 特征中文名映射（知识库 feature_vocab：tier → {feature_id: 中文}），供报告表头标注
    feature_zh = {fid: zh
                  for tier, feats in kb.raw.get("feature_vocab", {}).items()
                  for fid, zh in feats.items()}
    records = evidence_pool.load_measurements(args.measurements)
    if not records:
        print("[warn] 未解析到任何测量记录")
        return 1
    manifest_idx = evidence_pool.load_manifest_index(args.manifest)
    if args.sample:
        samples = {args.sample: records}
        print(f"[info] --sample={args.sample}：测量记录 {len(records)} 条 → 归并为 1 个样本")
    else:
        samples = evidence_pool.group_samples(records, manifest_idx)
        print(f"[info] 测量记录 {len(records)} 条 → 样本 {len(samples)} 个")

    vlm, image_index = None, {}
    if args.enable_vlm:
        try:
            vlm = VLMClient(args.vlm_config)
            image_index = build_image_index(args.images)
            print(f"[info] VLM 已启用：{vlm.model}（图像索引 {len(image_index)} 张）")
        except Exception as e:  # noqa: BLE001
            print(f"[warn] VLM 初始化失败，降级为 rule_only：{e}")
    else:
        print("[info] VLM 未启用（--enable-vlm），纯规则引擎模式")

    adjudicator = None
    if args.shadow_adjudicat:
        try:
            adjudicator = AdjudicatorClient(args.vlm_config, model=args.shadow_model)
            print(f"[info] 影子仲裁已启用：{adjudicator.model}（方案D，不改 verdict）")
        except Exception as e:  # noqa: BLE001
            print(f"[warn] 影子仲裁初始化失败，跳过（不影响判药）：{e}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "run_summary.jsonl"
    n_done = 0

    with open(summary_path, "w", encoding="utf-8") as fs:
        for sid, srecs in sorted(samples.items()):
            cards = evidence_pool.build_cards(srecs)
            # 场景证据（三层模型第 2/3 层）：纯几何提取，仅影响平分排序与报告，不改裁决
            scene = scene_attributes.build_scene(srecs)
            decision = decide(kb, cards, scene=scene)
            vlm_notes = None
            extra = {"agent": {"mode": "rule_only", "vlm_calls": 0, "degraded": False},
                     "evidence_files": collect_evidence_files(args.overlays, srecs),
                     "source_images": collect_source_images(args.images, srecs),
                     "scene_attributes": scene}
            # 方案D 影子仲裁：在 VLM 证据环之前，对纯引擎 top-k 做带内仲裁（不改 verdict）
            shadow = run_shadow_adjudication(decision, adjudicator) if adjudicator else None
            if shadow is not None:
                extra["shadow"] = shadow
            if vlm is not None:
                decision, vlm_notes, extra["agent"], scene = augment_and_rescore(
                    kb, cards, decision, vlm, image_index, srecs,
                    max_calls=args.vlm_max_calls, scene=scene)
                extra["scene_attributes"] = scene
            rdir = out_dir / "reports" / sid
            # 交付前过一遍判决契约（contracts.Verdict）：gold 工件一旦落盘就会被下游引用，
            # 缺 `verdict` 之类的结构性缺陷必须在这里拦住，而不是等读的人发现。
            vd = contracts.Verdict.from_obj({"sample_id": sid, **decision})
            vpath = report.write_verdict(rdir, sid, decision, extra)
            n_imgs = len({r.get("image_stem") for r in srecs})
            mpath = report.write_report(rdir, sid, decision, n_images=n_imgs,
                                vlm_notes=vlm_notes, evidence_files=extra["evidence_files"],
                                source_images=extra["source_images"], feature_zh=feature_zh,
                                shadow=extra.get("shadow"),
                                scene=extra.get("scene_attributes"),
                                review_refs=report.igm_review_refs(srecs))
            fs.write(json.dumps({
                "sample_id": sid, "verdict": decision["verdict"],
                "conclusion": (decision.get("conclusion") or {}).get("display"),
                "confidence": (decision.get("conclusion") or {}).get("confidence"),
                "needs_review": decision.get("needs_review", False),
                "vlm_calls": extra["agent"].get("vlm_calls", 0),
                "shadow_applied": bool(extra.get("shadow")),
                "verdict_json": str(vpath), "report_md": str(mpath),
                "generated_at": datetime.now().isoformat(timespec="seconds"),
            }, ensure_ascii=False) + "\n")
            n_done += 1
            print(f"[judge] {sid}: {vd.verdict} → {vd.conclusion_display or '-'}")
            if args.limit and n_done >= args.limit:
                break

    print(f"[done] 判药完成 {n_done} 样本 → {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
