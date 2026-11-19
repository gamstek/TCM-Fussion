# -*- coding: utf-8 -*-
"""run_single — 对单个测量文件执行完整判药流水线并输出标准报告。

**一次性脚本**（2026-09-29 从 `scripts/` 迁入 `scripts/experiments/`）：输入是硬编码的
历史产物 `几何测量/out/measure_草酸钙簇晶_20250378.csv`，用于当时的手工核对。
正式批量判药走 `scripts/judge.py`（编排层由 `run_pipeline.py` 调用）。
"""
import sys
import json
from pathlib import Path
from datetime import datetime

sys.stdout.reconfigure(encoding="utf-8")

# 本文件位于 scripts/experiments/：scripts/ 才是 rule_engine / evidence_pool 所在
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rule_engine import KnowledgeBase, decide
from evidence_pool import load_measurements, group_samples, build_cards


_SCRIPTS = Path(__file__).resolve().parent.parent          # Agent判药/scripts
_JUDGE = _SCRIPTS.parent                                   # Agent判药
KB_PATH = _JUDGE / "data" / "herb_criteria.json"
CSV_PATH = _JUDGE.parent / "几何测量" / "out" / "measure_草酸钙簇晶_20250378.csv"
MANIFEST_PATH = _JUDGE.parent / "manifest.csv"
OUT_DIR = _JUDGE / "reports" / "single_20250378"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load measurements
    records = load_measurements([str(CSV_PATH)])
    print(f"[✓] 加载 {len(records)} 条测量记录")

    # 2. Group samples & build cards
    manifest_idx = None
    if MANIFEST_PATH.exists():
        from evidence_pool import load_manifest_index
        manifest_idx = load_manifest_index(str(MANIFEST_PATH))
    
    samples = group_samples(records, manifest_idx)
    print(f"[✓] 分为 {len(samples)} 个样本")

    for sample_id, sample_records in samples.items():
        cards = build_cards(sample_records)
        
        evidential_features = {fid for fid, c in cards.items() if c.present}
        print(f"   样本 '{sample_id}': {evidential_features}")

        kb = KnowledgeBase(KB_PATH)
        
        # Filter candidates by evidence
        all_cands = kb.candidates()
        evidential_feats_map = set(evidential_features)
        evidential_cands = [
            c for c in all_cands 
            if any(fc in evidential_feats_map for fc in {cr["feature"] for cr in c["criteria"]})
        ]
        
        result = decide(kb, cards, evidential_cands)

        # Write verdict.json
        verdict_out = {
            "schema": "agent-judgement/v1",
            "input_files": [str(CSV_PATH)],
            "n_measurements": len(sample_records),
            "n_feature_types": len(cards),
            "evaluated_candidates": len(evidential_cands),
        }
        verdict_out.update(result)
        
        with open(OUT_DIR / "verdict.json", "w", encoding="utf-8") as f:
            json.dump(verdict_out, f, ensure_ascii=False, indent=2)
        print(f"[✓] verdict.json → {OUT_DIR / 'verdict.json'}")

        # Write markdown report
        report_lines = []
        report_lines.append(f"# Agent 判药报告 — 样本 {sample_id}")
        report_lines.append("")
        report_lines.append(f"> **时间**: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        report_lines.append(f"> **输入文件**: {CSV_PATH.name}")
        report_lines.append("")
        
        # Conclusion card
        report_lines.append("## 鉴别结论")
        report_lines.append("")
        report_lines.append(f"**裁决等级**: {result['verdict']}")
        if result.get("conclusion"):
            report_lines.append(f"**候选药材**: {result['conclusion']['display']}")
            report_lines.append(f"**置信度**: {result['conclusion']['confidence']:.4f}")
        else:
            report_lines.append("**原因**: 证据不足以支持任何候选")
        if result.get("margin") is not None:
            report_lines.append(f"**分差 (margin)**: {result['margin']:.4f}")
        report_lines.append("")

        # Ranking
        report_lines.append("## 候选排名（Top 5）")
        report_lines.append("")
        report_lines.append("| 排名 | 候选 | 类型 | 分数 | 置信度 |")
        report_lines.append("|------|------|------|------|--------|")
        ranking = result.get("ranking", [])
        for i, r in enumerate(ranking[:5], 1):
            kind_str = "herb" if r["kind"] == "herb" else r["kind"]
            report_lines.append(f"| {i} | {r['display']} | {kind_str} | {r['score']:.4f} | {r['confidence']:.4f} |")
        report_lines.append("")

        # Criteria check
        report_lines.append("## 判据核验表")
        report_lines.append("")
        top_checks = ranking[0]["checks"] if ranking else []
        if top_checks:
            report_lines.append("| 特征 | 角色 | 检出 | 量化状态 | 形态吻合 | 贡献分 |")
            report_lines.append("|------|------|------|----------|----------|--------|")
            for chk in top_checks:
                present_str = "✓" if chk["present"] else "✗"
                state_str = chk["state"]
                morph_str = f"{chk['morph_plausibility']:.2f}" if chk.get("morph_plausibility") is not None else "N/A"
                report_lines.append(
                    f"| {chk['feature_id']} | {chk['role']} | {present_str} | {state_str} | {morph_str} | {chk['score_contribution']:.4f} |"
                )
        report_lines.append("")

        # Limitations
        report_lines.append("## 局限性声明")
        report_lines.append("")
        report_lines.append(f"- 仅分析 {len(sample_records)} 条测量记录，覆盖 {len(cards)} 个特征类型")
        report_lines.append(f"- 未检出的所有特征视为 NOT_DETECTED（受限于上游目标检测与分割模块的覆盖率）")
        report_lines.append("- 本系统为辅助工具，最终判别需由具备资质的人员结合宏观性状、理化检验等综合判断")
        report_lines.append("")
        
        # Notes
        if result.get("notes"):
            report_lines.append("## 备注")
            report_lines.append("")
            for note in result["notes"]:
                report_lines.append(f"- {note}")
            report_lines.append("")

        report_text = "\n".join(report_lines)
        with open(OUT_DIR / "report.md", "w", encoding="utf-8") as f:
            f.write(report_text)
        print(f"[✓] report.md → {OUT_DIR / 'report.md'}")

    print(f"\n[✓] 完成！输出目录: {OUT_DIR}")


if __name__ == "__main__":
    main()
