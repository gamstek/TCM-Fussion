# -*- coding: utf-8 -*-
"""diagnose_single — 对单个测量文件进行判药诊断（只读，无副作用）。

**一次性脚本**（2026-09-29 从 `scripts/` 迁入 `scripts/experiments/`）：输入是硬编码的
历史产物 `几何测量/out/measure_草酸钙簇晶_20250378.csv`，用于当时的手工诊断。
"""
import sys
import json
from pathlib import Path
from pprint import pprint

# Ensure scripts dir is in path（本文件在 scripts/experiments/，上两级才是 scripts/）
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

from rule_engine import KnowledgeBase, EvidenceCard, pool_evidence, check_quant_state, morphology_plausibility, reliability, evaluate_candidate, decide
from evidence_pool import load_measurements

_JUDGE = Path(__file__).resolve().parent.parent.parent      # Agent判药
KB_PATH = _JUDGE / "data" / "herb_criteria.json"
# 历史一次性产物（非标准命名，列为 crop_id/source_image、flags 为 Python repr）：
# evidence_pool.load_measurements 的兼容层负责规整；正式批量输入仍须是 measure.py
# 按 <prefix>_measurements.csv/jsonl 契约产出（见 几何测量 README §一/§三）。
CSV_PATH = _JUDGE.parent / "几何测量" / "out" / "measure_草酸钙簇晶_20250378.csv"


def print_separator(title=""):
    print("\n" + "=" * 80)
    if title:
        print(f"  【{title}】")
        print("=" * 80)


def main():
    # 1. Load measurements
    print("=== 输入数据 ===")
    records = load_measurements([str(CSV_PATH)])
    print(f"读取到 {len(records)} 条测量记录\n")
    
    for rec in records:
        print(f"  image_stem : {rec.get('image_stem')}")
        print(f"  instance_id: {rec.get('instance_id')}")
        print(f"  feature    : {rec.get('feature')}")
        print(f"  calib_ok   : {rec.get('calibration_status')} ({rec.get('calibration_px_per_um')} px/um)")
        print(f"  flags      : {rec.get('flags', [])}")
        
        primary_dim = rec.get('primary_dimension', 'N/A')
        primary_val = rec.get('primary_value_um', 'N/A')
        print(f"  primary_dim: {primary_dim}")
        print(f"  primary_val: {primary_val} μm")
        
        # Key geometric features
        for key in ['feret_max_um', 'area_um2', 'defect_count', 'aspect_ratio', 
                     'solidity', 'rectangularity', 'roundness', 'eccentricity',
                     'mean_width_um', 'median_width_um', 'skeleton_len_um']:
            val = rec.get(key)
            if val is not None:
                print(f"  {key}: {val}")
        print()

    # 2. Build evidence cards
    print_separator("证据卡构建")
    cards = pool_evidence(records)
    print(f"构建了 {len(cards)} 个特征的证据卡")
    
    for fid, card in cards.items():
        print(f"\n  --- {fid} ---")
        print(f"  present: {card.present}")
        print(f"  n_instances (quantifiable): {card.n_instances}")
        print(f"  n_uncal (existence only): {len(card.uncal_refs)}")
        print(f"  median_conf: {card.median_conf}")
        print(f"  multi_component count: {card.n_components_multi}")
        
        for dim in ["feret_max", "mean_width"]:
            stats = card.dim_stats(dim)
            if stats:
                print(f"  {dim} stats:")
                print(f"    median={stats['median']:.2f} μm, p25={stats['p25']:.2f}, p75={stats['p75']:.2f}")
                print(f"    min={stats['min']:.2f}, max={stats['max']:.2f}, n={stats['n_measured']}")
        
        print(f"  shape descriptors: {card.shape}")
    
    # 3. Evaluate all candidates
    print_separator("候选评分与裁决")
    kb = KnowledgeBase(KB_PATH)
    
    # Only evaluate candidates that have evidence
    evidential_features = {fid for fid, card in cards.items() if card.present}
    print(f"有正证据的特征: {evidential_features}")
    
    # Filter candidates
    all_cands = kb.candidates()
    evidential_cands = []
    for cand in all_cands:
        cand_feats = {c["feature"] for c in cand["criteria"]}
        if cand_feats & evidential_features:
            evidential_cands.append(cand)
    
    print(f"进入候选池的药材/组: {len(evidential_cands)}")
    for c in evidential_cands:
        print(f"  - {c['display']} ({c['kind']})")
    
    # Evaluate each
    results = []
    for cand in evidential_cands:
        res = evaluate_candidate(cand, cards, kb)
        results.append(res)
        print(f"\n  【{res['display']}】 score={res['score']:.4f}, conf={res['confidence']:.4f}")
        for chk in res["checks"]:
            if chk["present"] or chk["state"] != "NOT_DETECTED":
                marker = "✓" if chk["present"] else "✗"
                print(f"    [{marker}] {chk['feature_id']} ({chk['role']}, {chk['tier']})")
                print(f"         state={chk['state']}, contrib={chk['score_contribution']:.4f}, rel={chk['reliability']:.4f}")
                if chk["quant_note"]:
                    print(f"         quant_note: {chk['quant_note']}")
        if res["wrong_present"]:
            for wp in res["wrong_present"]:
                print(f"    ⚠ WRONG: {wp['note']} (-{wp['penalty']})")
        if res["absent_required"]:
            for ar in res["absent_required"]:
                print(f"    ✗ ABSENT: {ar['note']} (-{ar['penalty']})")
    
    # Full decide
    result = decide(kb, cards, evidential_cands)
    print_separator("最终裁决")
    print(f"  verdict  : {result['verdict']}")
    if result.get("conclusion"):
        print(f"  conclusion: {result['conclusion']['display']} (conf={result['conclusion']['confidence']})")
    if result.get("alternatives"):
        for alt in result["alternatives"][:3]:
            print(f"  alternative: {alt['display']} (conf={alt['confidence']})")
    if result.get("margin"):
        print(f"  margin   : {result['margin']}")
    if result.get("notes"):
        print(f"  备注:")
        for note in result["notes"]:
            print(f"    - {note}")
    
    return result


if __name__ == "__main__":
    result = main()
    print_separator("结束")
