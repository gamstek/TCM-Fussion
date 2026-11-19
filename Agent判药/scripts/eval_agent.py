# -*- coding: utf-8 -*-
"""eval_agent — Agent 判药评测 harness（设计方案 §8）。

输入：measurements（同 judge.py）+ manifest.csv（ground truth：文件夹药材名，
      仅在评测阶段使用）。
指标：
  - Top-1 / Top-3 准确率（组级结论命中任一成员即得分）；
  - 组级正确归组率；
  - 弃权（insufficient_evidence）率 与 弃权样本规模；
  - 分层：量化判据层（该药在 KB 中任一判据带量化范围）vs 仅定性层；
  - 混淆对清单（gt → 预测）：预测 = conclusion.display（无结论记"(弃权)"）。

用法：
  python eval_agent.py --measurements ../几何测量/out/pseudo_measurements.jsonl ^
      --manifest <项目根>\manifest.csv --out ../reports/eval_summary.json
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent))

import evidence_pool  # noqa: E402
from rule_engine import KnowledgeBase, decide  # noqa: E402

HERE = Path(__file__).resolve().parent


def kb_has_quant(kb, herb):
    h = kb.herbs.get(herb)
    if h is None:
        gid = kb.group_of.get(herb)
        if gid:
            return any(c.get("quant") and c["quant"].get("min") is not None
                       for c in kb.groups[gid]["criteria"])
        return False
    return any(c.get("quant") and c["quant"].get("min") is not None for c in h["criteria"])


def expand_members(kb, cid):
    cand = kb.groups.get(cid)
    if cand:
        return set(cand["members"])
    return {cid}


def main():
    ap = argparse.ArgumentParser(description="Agent 判药评测")
    ap.add_argument("--measurements", nargs="+", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--kb", default=str(HERE.parent / "data" / "herb_criteria.json"))
    ap.add_argument("--out", default=None, help="评测摘要 JSON 输出路径")
    args = ap.parse_args()

    kb = KnowledgeBase(args.kb)
    records = evidence_pool.load_measurements(args.measurements)
    manifest_idx = evidence_pool.load_manifest_index(args.manifest)
    samples = evidence_pool.group_samples(records, manifest_idx)

    n = hit1 = hit3 = n_group = hit_group = n_insuff = 0
    strat = {"quant": [0, 0], "qual": [0, 0]}   # [n, hit1]
    confusion = Counter()
    per_sample = []

    for sid, srecs in sorted(samples.items()):
        gt = None
        for r in srecs:
            mrow = manifest_idx.get(r.get("image_stem", ""))
            if mrow and mrow.get("herb"):
                gt = mrow["herb"]
                break
        if gt is None:
            continue
        if kb.herbs.get(gt, {}).get("no_criteria"):
            continue  # 玄明粉：不适用桶
        cards = evidence_pool.build_cards(srecs)
        d = decide(kb, cards)
        n += 1
        top_ids = [r["id"] for r in d.get("ranking", [])[:3]]
        top3_members = set()
        for cid in top_ids:
            top3_members |= expand_members(kb, cid)
        top1_members = expand_members(kb, top_ids[0]) if top_ids else set()

        ok1 = gt in top1_members
        ok3 = gt in top3_members
        hit1 += ok1
        hit3 += ok3
        key = "quant" if kb_has_quant(kb, gt) else "qual"
        strat[key][0] += 1
        strat[key][1] += ok1
        if d["verdict"] == "group_level":
            n_group += 1
            if d.get("conclusion") and gt in expand_members(kb, d["conclusion"]["id"]):
                hit_group += 1
        if d["verdict"] == "insufficient_evidence":
            n_insuff += 1
        pred = (d.get("conclusion") or {}).get("display") or "(弃权)"
        if not ok1:
            confusion[f"{gt} → {pred}"] += 1
        per_sample.append({"sample": sid, "gt": gt, "verdict": d["verdict"],
                           "pred": pred,
                           "confidence": (d.get("conclusion") or {}).get("confidence"),
                           "top1_ok": ok1})

    def pct(a, b):
        return round(a / b, 4) if b else None

    summary = {
        "n_samples": n,
        "top1_acc": pct(hit1, n),
        "top3_acc": pct(hit3, n),
        "insufficient_rate": pct(n_insuff, n),
        "group_level": {"n": n_group, "correct": hit_group,
                        "acc": pct(hit_group, n_group) if n_group else None},
        "stratified": {k: {"n": v[0], "top1_acc": pct(v[1], v[0])} for k, v in strat.items()},
        "confusion_pairs": confusion.most_common(30),
        "per_sample": per_sample,
    }
    print(json.dumps({k: v for k, v in summary.items() if k != "per_sample"},
                     ensure_ascii=False, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[done] 评测摘要 → {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
