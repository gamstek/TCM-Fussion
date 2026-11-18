# -*- coding: utf-8 -*-
"""把规则裁决投影成“永远有输出”的交付视图，不修改 decide/阈值/证据卡，并保留真实复核状态。

规则引擎在证据不足时会返回 `insufficient_evidence` + `needs_review`，交付层仍须给出
top 候选、备选、强观察与缺口；本模块只做投影，不联网、不写证据、不读取原图。

主要定义：
- STRONG_CONF: 强观察置信度阈值（0.6）。
- TIE_EPS: 判定分数并列的容差。
- project_output: 生成交付视图；status 只取 confirmed/probable/group_level/tentative，不把 tentative 写成 confirmed。
- `_a_majority` / `_view_map` / `_detections` 等系列：A 多数票、视野与操作摘要的内部辅助。

关联：
- 调用方 「lib/application/workflow_entry.py」、「tests/decision/test_candidate_audit.py」、「tests/decision/test_output_projection.py」、「tests/integration/test_fair_investigation.py」。
- 直接依赖 「lib/evidence/provisional_evidence.py」、「lib/runtime/storage.py」。
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import contracts

from TCM_Fusion.lib.evidence.evidence_projection import supports_criterion
from TCM_Fusion.lib.evidence.search_ledger import scan_coverage
from TCM_Fusion.lib.runtime.storage import finite_number, read_json
from TCM_Fusion.lib.evidence.provisional_evidence import build as build_provisional

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


def _detections(operations, artifacts):
    if artifacts is not None:
        rows = []
        for item in artifacts.values():
            if not item['valid'] or item['kind'] != 'crop' or not item.get('detection_tool'):
                continue
            if item.get('reclassified_from'):
                continue
            meta = contracts.CropMeta.from_json(item['path'])
            image_id = item.get('image_id')
            rows.append({'tool': item['detection_tool'], 'label': meta.object_label,
                         'feature_id': meta.feature_id, 'crop_id': meta.crop_id,
                         'artifact_id': item['id'], 'image_id': image_id,
                         'score': item.get('det_score'), 'box': meta.tight_box,
                         'source_sha256': artifacts.get(image_id, {}).get('sha256'),
                         'box_quality': item.get('box_quality')})
        return rows

    rows = []
    for op in (operations or {}).values():
        if op.get('tool') == 'feature_scan':
            bundle = op.get('bundle') or {}
            folder = (bundle.get('paths') or {}).get('crops_dir')
            if folder and bundle.get('root'):
                for meta in contracts.load_crop_metas(str(Path(bundle['root']) / folder)):
                    rows.append({'tool': 'feature_scan', 'label': meta.object_label, 'feature_id': meta.feature_id,
                                 'crop_id': meta.crop_id, 'image_id': op.get('image_id'),
                                 'score': meta.raw.get('det_score'), 'box': meta.tight_box})
            continue
        if op.get("tool") == "hybrid_scan":
            root = Path((op.get("bundle") or {}).get("root") or "")
            doc_path = root / "yolo" / "detection" / "crops.json"
            if doc_path.is_file():
                try:
                    doc = read_json(doc_path)
                except (OSError, ValueError):
                    doc = {}
                for det in doc.get("dets") or []:
                    rows.append({"tool": "hybrid_scan", "label": det.get("label"),
                                 "score": det.get("score"), "box": det.get("tight_bbox"),
                                 "source_sha256": doc.get("source_sha256")})
            feature_root = root / "features" / "crops"
            if feature_root.is_dir():
                for path in sorted(feature_root.glob("*.json")):
                    try:
                        doc = read_json(path)
                    except (OSError, ValueError):
                        continue
                    rows.append({"tool": "hybrid_scan", "label": doc.get("object_label") or doc.get("feature_id"),
                                 "score": doc.get("det_score"), "box": doc.get("tight_box"),
                                 "source_sha256": doc.get("source_sha256")})
            continue
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
        tool = op.get('tool')
        bundle = op.get('bundle') or {}
        if tool == 'classify_crops_batch':
            folders = [row['classification'] for row in bundle.get('classifications', [])
                       if row.get('status') == 'done']
        elif tool == 'classify_target':
            folders = [(bundle.get('paths') or {}).get('classification', 'classification')]
        else:
            continue
        if not bundle.get('root'):
            continue
        root = Path(bundle['root'])
        fields = ('crop_id', 'label', 'reject', 'confidence', 'feature_id', 'reason',
                  'glm_confidence', 'yolo_confidence', 'confidence_source')
        for folder in folders:
            try:
                doc = read_json(root / folder / 'classification.json')
            except (OSError, ValueError):
                continue
            rows.append({'tool': tool, **{key: doc.get(key) for key in fields}})
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
        target_score = doc.get('target_score')
        if target_score is None and doc.get('feature_id') == 'needle_crystal':
            target_score = doc.get('needle_score')
        rows.append({'target_id':doc.get('target_id'),'feature_id':doc.get('feature_id'),
                     'target_score':target_score,
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
        if payload.get('ok') is False or not finite_number(conf, minimum=STRONG_CONF, maximum=1):
            continue
        args = action.get("args") or {}
        image_id = args.get("image_id", "")
        is_presence = action.get('tool') == 'presence_check'
        present = payload.get('present')
        if is_presence and type(present) is not bool:
            continue
        rows.append({
            "view": views.get(image_id, Path(image_id).stem or None),
            "tool": action.get("tool"),
            "feature_id": args.get("feature_id") or ("scene_check" if action.get("tool") == "scene_check" else None),
            "present": present,
            "scope": "observed_fov_only",
            "qualification": "presence_only" if is_presence else "scene_context",
            "confidence": conf,
            "source": "glm_presence" if action.get("tool") == "presence_check" else "glm_scene",
        })
    return rows


def _supporting_candidates(ranking, feature_id):
    names = []
    for row in ranking:
        for check in row.get("checks") or []:
            if check.get("feature_id") == feature_id and supports_criterion(check):
                names.append(row.get("id"))
                break
    return names[:5]


def project_output(sample_id, verdict, result, a, *, artifacts=None):
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
    try:
        provisional = build_provisional(verdict, result)
    except Exception as exc:
        # 暂定排序是旁路；失败不能改变正式交付。
        provisional = {"schema": "provisional-ranking/v1", "error": f"{type(exc).__name__}: {exc}"}
    formal_top = ranking[0] if ranking else None
    top = formal_top
    selection = 'formal'
    if provisional.get('ranking') and formal_top:
        prov_top = provisional['ranking'][0]
        prov_id = prov_top.get('candidate_id')
        has_partial = any(s.get('state') in {'DETECTED_PENDING', 'PARTIAL_MATCH'}
                          for s in prov_top.get('criterion_states') or [])
        # 正式未确认时，允许暂定排序成为交付层的 tentative 首选；
        # 正式 verdict/support_count 不变，仍单独保留。
        if (verdict.get('verdict') not in {'confirmed', 'probable', 'group_level'}
                and prov_id and prov_id != formal_top.get('id') and has_partial):
            candidate = next((r for r in ranking if r.get('id') == prov_id), None)
            if candidate:
                top = candidate
                selection = 'provisional'
    assessment = result.get('final_assessment') or {}
    assessment_bound = assessment.get('schema') == 'final-assessment/v1'
    projection_error = None
    if assessment_bound:
        # finalize已完成来源/引用校验；交付不再独立重选另一个药材。
        cid = assessment.get('candidate_id')
        top = next((r for r in ranking if r.get('id') == cid), None) if cid else None
        selection = 'final_assessment'
        if cid and top is None:
            projection_error = 'ASSESSMENT_CANDIDATE_NOT_IN_RANKING'
    ambiguous = [r for r in ranking if r is not top and top and
                 abs((r.get("score") or 0) - (top.get("score") or 0)) <= TIE_EPS][:4]
    status = {"confirmed": "confirmed", "probable": "probable",
              "group_level": "group_level"}.get(verdict.get("verdict"), "tentative")
    status_note = None
    needs_review = bool(verdict.get('needs_review') or result.get('needs_review')
                        or (result.get('final_assessment')or{}).get('review_required')
                        or status == 'tentative' or selection == 'provisional')
    if assessment_bound and assessment.get('status') not in {'confirmed','probable','group_level'}:
        status = 'tentative' if top else 'undetermined'
        needs_review = True
    if projection_error:
        status = 'undetermined'
        needs_review = True
        status_note = projection_error
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
        row["supporting_candidates"] = (_supporting_candidates(ranking, row.get("feature_id"))
                                        if row["present"] is True else [])
    operations = result.get('operations') or {}
    detections = _detections(operations, artifacts)
    classifications = result.get('valid_classifications')
    if classifications is None:
        classifications = _classifications(operations)
    feature_scans = []
    for operation in operations.values():
        if operation.get('tool') == 'feature_scan':
            status, summary = scan_coverage(operation)
            feature_scans.append({'target_id': operation.get('target_id'), 'image_id': operation.get('image_id'),
                                  'feature_id': operation.get('feature_id'), **summary, 'status': status})
    missing = []
    if top:
        for check in top.get("checks") or []:
            if check.get("role") != "required":
                continue
            if not check.get("present"):
                missing.append({"feature_id": check.get("feature_id"), "state": "NOT_DETECTED"})
            elif check.get("state") not in {"IN_RANGE", "EXTREME", "NO_REF"}:
                missing.append({"feature_id": check.get("feature_id"), "state": check.get("state")})
    provisional['selection'] = selection
    provisional['selected_candidate'] = top.get('id') if top else None
    alternative_rows = []
    if formal_top is not None and formal_top is not top:
        alternative_rows.append(formal_top)
    for row in ranking:
        if row is not top and row is not formal_top:
            alternative_rows.append(row)
    return {
        "schema": "diagnosis-output/v1",
        "sample_id": sample_id,
        "status": status,
        "top_candidate": None if top is None else {
            "id": top.get("id"), "display": top.get("display"), "kind": top.get("kind"),
            "score": top.get("score"), "confidence": top.get("confidence"),
            "support_count": top.get("support_count"),
            "verified_feature_count": (assessment.get('confidence_basis') or {}).get('verified_feature_count'),
            "a_probability": top.get("a_probability"),
            "b_confidence": top.get("b_confidence"),
            "rule_score": top.get("b_score", top.get("score")),
            "fused_score": top.get("fused_score"),
            "selection": selection,
            "formal_score": top.get("score"),
            "provisional_score": next((r.get('provisional_score') for r in provisional.get('ranking', [])
                                       if r.get('candidate_id') == top.get('id')), None),
            "admission": verdict.get("candidate_admission", {}).get("decision"),
        },
        "candidate_admission": verdict.get("candidate_admission"),
        "provisional": provisional,
        "alternatives": [{"id": r.get("id"), "display": r.get("display"),
                          "score": r.get("score"), "confidence": r.get("confidence")}
                         for r in alternative_rows[:4]],
        "ambiguous_with": [r.get("id") for r in ambiguous],
        "tie_break": tie_break,
        "strong_observations": observations,
        "detections": detections,
        "classifications": classifications,
        "feature_scans": feature_scans,
        "missing_required": missing,
        "review_required": needs_review,
        "final_assessment": result.get('final_assessment'),
        "candidate_audit": (result.get('final_assessment')or{}).get('candidate_audit'),
        "judgement": None if projection_error else assessment.get('judgement'),
        "fusion_ranking": [{"id":r.get('id'), "score":r.get('score')} for r in verdict.get('ranking',[])],
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
            "assessment_projection_error": projection_error,
        },
        "note": "tentative 是未确认候选；review_required 如实表示复核需要，不阻止报告交付。",
    }
