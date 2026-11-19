# -*- coding: utf-8 -*-
"""test_rule_engine — 规则引擎单测（数据驱动）。

用例数据存放于 tests/cases/rule_engine.json；本程序读取该文件，
对每条用例的 records 展开为测量记录 → pool_evidence → decide()，
再按 checks[].kind 的断言语义逐项校验。
直接运行： python tests/test_rule_engine.py
"""
import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "scripts"))

from rule_engine import KnowledgeBase, decide, pool_evidence  # noqa: E402

KB = KnowledgeBase(HERE.parent / "data" / "herb_criteria.json")
CASES = HERE / "cases" / "rule_engine.json"
PASS = 0


def expand_record(item):
    """紧凑 record（{feature, values, conf, stem, calib, dim, shape}）→ 测量记录列表。"""
    fid = item["feature"]
    vals = item["values"]
    conf = item.get("conf", 0.9)
    stem = item.get("stem", "T1")
    calib = item.get("calib", "ok")
    dim = item.get("dim", "feret_max_um")
    shape = item.get("shape", {})
    ok = calib == "ok"
    px = 11.7 if ok else None
    out = []
    for i, v in enumerate(vals, 1):
        r = {
            "image_stem": stem, "instance_id": i, "feature": fid,
            "calibration_status": calib, "calibration_px_per_um": px,
            "primary_dimension": dim, "primary_value_um": v if ok else None,
            "conf": conf, "flags": [],
        }
        r.update(shape)
        out.append(r)
    return out


def _cand(d, idx):
    return d["ranking"][idx]


def kind_check(ch, d):
    k = ch["kind"]
    if k == "verdict_eq":
        return d["verdict"] == ch["val"]
    if k == "conclusion_id":
        return bool(d["conclusion"]) and d["conclusion"]["id"] == ch["val"]
    if k == "conclusion_conf_ge":
        return bool(d["conclusion"]) and d["conclusion"]["confidence"] >= ch["val"]
    if k == "not_confirmed_as":
        return not (d["verdict"] == "confirmed"
                    and bool(d["conclusion"]) and d["conclusion"]["id"] == ch["id"])
    if k == "insufficient_evidence_no_conclusion":
        return d["verdict"] == "insufficient_evidence" and d["conclusion"] is None
    if k == "alternatives_keep":
        return (bool(d["alternatives"])
                and d["alternatives"][0]["confidence"] >= ch["first_conf_ge"]
                and d.get("needs_review") == ch.get("needs_review"))
    if k == "support_feature_count":
        return d.get("support_feature_count") == ch["val"]
    if k == "ranking_top_ids_contains":
        return ch["val"] in [r["id"] for r in d["ranking"][: ch["top_n"]]]
    if k == "ranking_ids_contains":
        return ch["val"] in [r["id"] for r in d["ranking"]]
    if k == "candidate_conf_ge":
        return any(r["id"] == ch["id"] and r["confidence"] >= ch["val"] for r in d["ranking"])
    if k == "candidate_conf_le":
        return any(r["id"] == ch["id"] and r["confidence"] <= ch["val"] for r in d["ranking"])
    if k == "verdict_needs_review":
        return d["verdict"] == ch["val"] and d.get("needs_review")
    if k == "verdict_in":
        return d["verdict"] in ch["vals"]
    if k == "ranking_len_ge":
        return len(d["ranking"]) >= ch["val"]
    if k == "insufficient_multi":
        return (d["verdict"] == "insufficient_evidence"
                and len(d["alternatives"]) >= ch["alternatives_ge"]
                and d.get("needs_review"))
    if k == "conclusion_is_null":
        return d["conclusion"] is None
    if k == "rank0_check_state":
        return any(c["feature_id"] == ch["feature"] and c["state"] == ch["state"]
                   for c in d["ranking"][0]["checks"])
    if k == "candidate_check_state":
        # 指定候选的判据状态（rank0_check_state 依赖排序，只适合"排序稳定"的用例）
        return any(r["id"] == ch["id"]
                   and any(c["feature_id"] == ch["feature"] and c["state"] == ch["state"]
                           for c in r["checks"])
                   for r in d["ranking"])
    if k == "verdict_ne":
        return d["verdict"] != ch["val"]
    if k == "entry_checks_len":
        return any(r["id"] == ch["id"] and len(r["checks"]) == ch["len"]
                   for r in d["ranking"])
    if k == "positive_hit":
        return (d["verdict"] in ("probable", "confirmed")
                and bool(d["conclusion"]) and d["conclusion"]["id"] == ch["id"])
    if k == "rank_id_at":
        return (len(d["ranking"]) > ch["index"]
                and d["ranking"][ch["index"]]["id"] == ch["val"])
    if k == "notes_contains":
        return any(ch["val"] in n for n in d.get("notes", []))
    if k == "scene_fit_eq":
        return any(r["id"] == ch["id"] and r.get("scene_fit") == ch["val"]
                   for r in d["ranking"])
    raise ValueError(f"未知断言语义 kind: {k}")


def check(name, cond):
    global PASS
    if not cond:
        raise AssertionError(f"用例失败: {name}")
    PASS += 1
    print(f"  [PASS] {name}")


def main():
    data = json.loads(CASES.read_text(encoding="utf-8"))
    for case in data["cases"]:
        recs = []
        for item in case["records"]:
            recs += expand_record(item)
        # scene（可选）：三层模型场景证据，仅影响平分排序与 notes（rule_engine.decide）
        d = decide(KB, pool_evidence(recs), scene=case.get("scene"))
        for ch in case["checks"]:
            check(ch["name"], kind_check(ch, d))

    # ── 量化可用性闸门（coarse_quant，2026-10-09）：粗粒度实例（束/片/团）只作存在性 ──
    # 几何对 granularity != instance 的实例打 coarse_quant；pool_evidence 须让其走 uncal_refs
    # （与"未标定"同构），不得进 dims 参与尺寸核验——否则束宽会被当单根直径去对药典区间。
    base = {"image_stem": "T1", "instance_id": 1, "feature": "fiber",
            "calibration_status": "ok", "calibration_px_per_um": 11.7, "conf": 0.9,
            "primary_dimension": "mean_width_um", "primary_value_um": 20.0}
    ctrl = pool_evidence([dict(base, flags=[])])["fiber"]
    check("量化闸门: 对照——无 coarse_quant 时产量化值",
          ctrl.dims.get("mean_width", {}).get("values") == [20.0] and not ctrl.uncal_refs)
    coarse = pool_evidence([dict(base, flags=["coarse_quant"])])["fiber"]
    check("量化闸门: 有 coarse_quant → 降为存在性（不产量化值）",
          not coarse.dims and len(coarse.uncal_refs) == 1)

    print(f"\n== 规则引擎单测全部通过：{PASS} 项断言 (数据源 {CASES.name}) ==")


if __name__ == "__main__":
    main()