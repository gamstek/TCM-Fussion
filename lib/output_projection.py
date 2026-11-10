# -*- coding: utf-8 -*-
"""把规则裁决投影成“永远有输出”的交付视图（方案 A）。

**为什么存在**：规则引擎在证据不足时会返回 `insufficient_evidence` + `needs_review`，
但交付层仍需要给出 top 候选、备选、强观察和缺口。本模块只做投影，不修改
`decide`/阈值/证据卡；交付结果同时保留真实复核状态。

设计边界：
  * `status` 只取 confirmed/probable/group_level/tentative，不把 tentative 写成 confirmed；
  * 强观察（GLM 存在性/场景，conf≥0.6）全部保留，并标注它支持了哪些候选；
  * 分数完全并列时，用 A 的逐折 Top-1 多数做 tie-break（只影响排序，不改变证据）；
  * 不联网、不写证据、不读取原图。
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

from .storage import read_json

STRONG_CONF = 0.6
TIE_EPS = 1e-9


def _view_map(a):
    rows = {}
    for view in a.get("view_predictions") or []:
        stem = Path(view.get("image", "")).stem
        if view.get("input_sha256"):
            rows["img_" + view["input_sha256"]] = stem
        if stem:
            rows[stem] = stem
    return rows


def _a_majority(a):
    counts = Counter()
    for view in a.get("view_predictions") or []:
        for fold in view.get("fold_rankings") or []:
            if fold:
                counts[fold[0].get("label")] += 1
    if not counts:
        for fold in a.get('fold_rankings') or []:
            if fold:
                counts[fold[0].get('label')] += 1
    if not counts:
        ranking = a.get("ranking") or []
        if ranking:
            counts[ranking[0].get("label")] += 1
    total = sum(counts.values())
    if not counts:
        return None, 0, 0
    label, count = counts.most_common(1)[0]
    return label, count, total


def _detections(operations):
    rows = []
    for op in (operations or {}).values():
        if op.get("tool") not in {"detect_fov_initial", "detect_region", "locate_feature"}:
            continue
        bundle = op.get("bundle") or {}
        root = Path(bundle.get("root") or "")
        doc_path = None
        for name in ("detection/crops.json", "detection/region_result.json"):
            candidate = root / name
            if candidate.is_file():
                doc_path = candidate
                break
        if doc_path is None:
            continue
        try:
            doc = read_json(doc_path)
        except (OSError, ValueError):
            continue
        for det in doc.get("dets") or []:
            rows.append({"tool": op.get("tool"), "label": det.get("label"),
                         "score": det.get("score"), "box": det.get("tight_bbox"),
                         "source_sha256": doc.get("source_sha256")})
    return rows


def _classifications(operations):
    rows = []
    for op in (operations or {}).values():
        if op.get("tool") != "classify_target":
            continue
        bundle = op.get("bundle") or {}
        root = Path(bundle.get("root") or "")
        doc_path = root / "classification" / "classification.json"
        if not doc_path.is_file():
            continue
        try:
            doc = read_json(doc_path)
        except (OSError, ValueError):
            continue
        rows.append({"crop_id": doc.get("crop_id"), "label": doc.get("label"),
                     "reject": doc.get("reject"), "confidence": doc.get("confidence"),
                     "feature_id": doc.get("feature_id"), "reason": doc.get("reason"),
                     "glm_confidence": doc.get("glm_confidence"),
                     "yolo_confidence": doc.get("yolo_confidence"),
                     "confidence_source": doc.get("confidence_source")})
    return rows


def _image_audits(operations):
    rows=[]
    for op in (operations or {}).values():
        if op.get('tool')!='image_exemplar_audit':
            continue
        bundle=op.get('bundle') or {}; root=Path(bundle.get('root') or '')
        path=root/'image_audit/audit.json'
        if not path.is_file(): continue
        try: doc=read_json(path)
        except (OSError,ValueError): continue
        rows.append({'target_id':doc.get('target_id'),'feature_id':doc.get('feature_id'),
                     'target_score':doc.get('target_score'),'needle_score':doc.get('needle_score'),
                     'negative_similarity':doc.get('negative_similarity'),
                     'negative_margin':doc.get('negative_margin'),'audit_label':doc.get('audit_label'),
                     'confidence':doc.get('confidence'),'reason':doc.get('reason'),
                     'review_status':doc.get('review_status'),'non_blocking':doc.get('non_blocking')})
    return rows


def _observations(result, views):
    rows = []
    for obs in result.get("observations") or []:
        if not obs.get("accepted"):
            continue
        action = obs.get("action") or {}
        payload = obs.get("result") or {}
        conf = payload.get("conf")
        if not isinstance(conf, (int, float)) or conf < STRONG_CONF:
            continue
        args = action.get("args") or {}
        image_id = args.get("image_id", "")
        rows.append({
            "view": views.get(image_id, Path(image_id).stem or None),
            "tool": action.get("tool"),
            "feature_id": args.get("feature_id") or ("scene_check" if action.get("tool") == "scene_check" else None),
            "present": payload.get("present"),
            "confidence": conf,
            "source": "glm_presence" if action.get("tool") == "presence_check" else "glm_scene",
        })
    return rows


def _supporting_candidates(ranking, feature_id):
    names = []
    for row in ranking:
        for check in row.get("checks") or []:
            if (check.get("feature_id") == feature_id and check.get("present")
                    and (check.get("score_contribution") or 0) > 0):
                names.append(row.get("id"))
                break
    return names[:5]


def project_output(sample_id, verdict, result, a):
    ranking = list(verdict.get("ranking") or [])
    a_label, a_count, a_total = _a_majority(a)
    tie_break = {"applied": False, "a_majority": a_label,
                 "a_fold_top1": a_count, "total_folds": a_total}
    if ranking:
        top_score = ranking[0].get("score", 0)
        ties = [r for r in ranking if abs((r.get("score") or 0) - (top_score or 0)) <= TIE_EPS]
        if len(ties) > 1 and a_label and any(r.get("id") == a_label for r in ties):
            tied = sorted(ties, key=lambda r: (r.get("id") == a_label,
                                               r.get("confidence") or 0), reverse=True)
            ranking = tied + [r for r in ranking if r not in ties]
            tie_break["applied"] = bool(ranking and ranking[0].get("id") == a_label)
    top = ranking[0] if ranking else None
    ambiguous = [r for r in ranking[1:5] if top and abs((r.get("score") or 0) - (top.get("score") or 0)) <= TIE_EPS]
    status = {"confirmed": "confirmed", "probable": "probable",
              "group_level": "group_level"}.get(verdict.get("verdict"), "tentative")
    status_note = None
    needs_review = bool(verdict.get('needs_review') or result.get('needs_review')
                        or status == 'tentative')
    conclusion = verdict.get('conclusion')
    conclusion_id = conclusion.get('id') if isinstance(conclusion, dict) else conclusion
    if conclusion_id and top and conclusion_id != top.get('id'):
        needs_review = True
    if status in {'confirmed','probable','group_level'} and needs_review:
        status = "tentative"
        status_note = "identity/policy review required; not promoted to confirmed"
    admission = verdict.get("candidate_admission") or {}
    if status in {"confirmed", "probable"} and admission.get("decision") in {"verify", "suppress"}:
        needs_review = True
        status = "tentative"
        status_note = "candidate admission not passed; not promoted"
    views = _view_map(a)
    observations = _observations(result, views)
    for row in observations:
        row["supporting_candidates"] = _supporting_candidates(ranking, row.get("feature_id"))
    detections = _detections(result.get("operations"))
    classifications = _classifications(result.get("operations"))
    missing = []
    if top:
        for check in top.get("checks") or []:
            if check.get("role") != "required":
                continue
            if not check.get("present"):
                missing.append({"feature_id": check.get("feature_id"), "state": "NOT_DETECTED"})
            elif check.get("state") not in {"IN_RANGE", "EXTREME", "NO_REF"}:
                missing.append({"feature_id": check.get("feature_id"), "state": check.get("state")})
    return {
        "schema": "diagnosis-output/v1",
        "sample_id": sample_id,
        "status": status,
        "top_candidate": None if top is None else {
            "id": top.get("id"), "display": top.get("display"), "kind": top.get("kind"),
            "score": top.get("score"), "confidence": top.get("confidence"),
            "support_count": top.get("support_count"),
            "a_probability": top.get("a_probability"),
            "b_confidence": top.get("b_confidence"),
            "fused_score": top.get("fused_score"),
            "admission": verdict.get("candidate_admission", {}).get("decision"),
        },
        "candidate_admission": verdict.get("candidate_admission"),
        "alternatives": [{"id": r.get("id"), "display": r.get("display"),
                          "score": r.get("score"), "confidence": r.get("confidence")}
                         for r in ranking[1:5]],
        "ambiguous_with": [r.get("id") for r in ambiguous],
        "tie_break": tie_break,
        "strong_observations": observations,
        "detections": detections,
        "classifications": classifications,
        "missing_required": missing,
        "review_required": needs_review,
        "final_assessment": result.get('final_assessment'),
        "judgement": (result.get('final_assessment') or {}).get('judgement'),
        "needle_audit": result.get('needle_audit'),
        "feature_audit": result.get('feature_audit'),
        "image_exemplar_audits": _image_audits(result.get('operations')),
        "audit": {
            "rule_verdict": verdict.get("verdict"),
            "internal_needs_review": bool(verdict.get("needs_review")),
            "status_note": status_note,
            "candidate_scope": verdict.get("candidate_scope"),
            "fusion_weights": verdict.get("fusion_weights"),
            "fusion_policy": verdict.get("fusion_policy"),
            "policy": "always_output_tentative",
        },
        "note": "tentative 是未确认候选；review_required 如实表示复核需要，不阻止报告交付。",
    }
