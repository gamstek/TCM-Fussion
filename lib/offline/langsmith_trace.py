"""将一个已落盘或正在更新的融合任务投影为 LangSmith 调用树。

主要定义：project_trace 只读运行工件，返回样品、规划、工具与 D4 分支的时间和结构化摘要。
LangSmithPublisher 汇总实时反馈后提交一次结束，也可从完整工件生成归档回放。

关联：cli.langsmith_trace。读取 application、agent 和 workflow 已写出的工件，
上传时才使用 LangSmith SDK。不导入识别、模型或工作流执行入口。
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
import re
from uuid import NAMESPACE_URL, uuid5

from TCM_Fusion.lib.runtime.storage import IntegrationError, digest, finite_number, read_json


_RUN_ID = re.compile(r"fusion_[0-9a-f]{32}")
_IMAGE_ID = re.compile(r"img_[0-9a-f]{64}")
_ACTION_CODE = re.compile(r"[AGOPQ][0-9]{1,3}")
_CODE = re.compile(r"[A-Z][A-Z0-9_]{0,63}")
_SENSITIVE_TEXT = re.compile(
    r"\b(?:https?|file|data):|\\|[a-z]:/|(?:^|[\s(（：])(?:\.{1,2}/|/)"
    r"|\b(?:lsv2_|ls__|sk-|ghp_|github_pat_|bearer\s+|api[_ -]?key\b|authorization\b)"
    r"|[a-f0-9]{32,}|[a-z0-9_+=/-]{48,}|[<>]", re.IGNORECASE,
)
_TERMINAL = {"done", "failed", "cancelled", "canceled", "completed"}
_DECISIONS = {"confirmed": "确认", "probable": "较可能", "group_level": "候选组",
              "tentative": "暂定", "undetermined": "未判定"}
_DIMENSIONS = {"feret_max", "mean_width", "skeleton_len", "eq_diameter", "area", "length", "width"}
_PURPOSES = {"observe", "identity", "measure", "presence", "morphology"}
_AUDIT_STATES = {
    "supported", "quantitative_counter", "local_absent", "local_absent_partial",
    "present_pending", "failed_or_unknown", "investigated_unresolved", "unchecked",
}
_TOOLS = frozenset("""
    detect_fov_initial calibrate_fov detect_region locate_feature feature_scan hybrid_scan
    classify_target inspect_crop classify_crops_batch segment_instance measure_instances image_exemplar_audit
    presence_check scene_check read_evidence read_evidence_detail read_scale list_fovs
    list_features list_artifacts read_artifact_status read_quality read_action_catalog
    query_criteria criteria_diff request_review finish
""".split())


@dataclass
class Span:
    key: str
    name: str
    start: float
    end: float | None = None
    kind: str = "chain"
    parent: str | None = "case"
    inputs: dict = field(default_factory=dict)
    outputs: dict = field(default_factory=dict)
    error: str | None = None
    awaiting_feedback: bool = False


def optional_json(path):
    try:
        return read_json(path)
    except FileNotFoundError:
        return {}


def find_run(directory):
    """定位显式提供的 run 或样品目录。运行目录尚未创建时可等待。"""
    if (directory / "job.json").is_file():
        return directory
    runs = [path for path in (directory / "runs").glob("fusion_*")
            if _RUN_ID.fullmatch(path.name) and (path / "job.json").is_file()]
    if len(runs) > 1:
        raise IntegrationError("TRACE_RUN_AMBIGUOUS", "此目录含多个任务，请指定 runs 下的一个正式运行目录。")
    return runs[0] if runs else None


def load_vocabulary(root):
    kb = read_json(root / "Agent判药" / "data" / "herb_criteria.json")
    candidates = set(kb["groups"]) | {row["herb"] for row in kb["herbs"]}
    entries = [*kb["groups"].values(), *kb["herbs"]]
    features = {check["feature"] for row in entries for check in row.get("criteria", [])}
    labels = {code: label for tier in kb["feature_vocab"].values() for code, label in tier.items()}
    return candidates, {code: labels.get(code, code) for code in features}


def _stamp(value):
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value)
            return parsed.timestamp() if parsed.tzinfo else None
        except ValueError:
            return None
    return float(value) if finite_number(value, minimum=1, maximum=253402300799) else None


def _end(start, value):
    stamp = _stamp(value)
    return stamp if stamp is not None and stamp >= start else None


def _code(value, fallback=None):
    return value if isinstance(value, str) and _CODE.fullmatch(value) else fallback


def _view(fields, image_views):
    for name in ("image_id", "target_id"):
        value = fields.get(name)
        match = (_IMAGE_ID.search(value) or re.search(r'@V[0-9]+@', value)) if isinstance(value, str) else None
        if match and match[0] in image_views:
            return image_views[match[0]]
    return None


def _feature_name(feature, features):
    return features.get(feature, feature) if isinstance(features, dict) else feature


def _can_show_visual_text(value, limit=None):
    return (isinstance(value, str) and bool(value) and (limit is None or len(value) <= limit)
            and value.isprintable() and not _SENSITIVE_TEXT.search(value))


def _visual_reading(value, include_text):
    if not isinstance(value, dict) or value.get('status') not in ('reported', 'unreported', 'cnn_only'):
        return {}
    def fields(report):
        result = {'status': report['status']}
        if report.get('readability') in ('clear', 'partial', 'unreadable'):
            result['readability'] = report['readability']
        if include_text:
            for key, limit in (('brief', None), ('finding', 320), ('location', 100)):
                text = report.get(key)
                if _can_show_visual_text(text, limit):
                    result[key] = text
        return result
    reading = {**fields(value), 'guided': bool(value.get('brief'))}
    if isinstance(value.get('tiles'), list):
        reading['tiles'] = [{'tile': row['tile'], **fields(row)} for row in value['tiles']
                            if isinstance(row, dict) and type(row.get('tile')) is int
                            and row.get('status') in ('reported', 'unreported')]
    return {'visual_observation': reading}


def _observation_receipts(rows, image_views, features, *, include_text=False):
    """按动作关联真实回执。存在性观察的范围是实际下发的单个视野。"""
    receipts = {}
    for row in rows:
        action = row.get("action") or {}
        key, tool = action.get("action_id"), action.get("tool")
        if not isinstance(key, str) or tool not in {"presence_check", "scene_check"}:
            continue
        args, result = action.get("args") or {}, row.get("result") or {}
        accepted = row.get("accepted") is True and result.get("ok") is True
        receipt = {"tool": tool, "accepted": accepted, "scope": "current_view"}
        if row.get('cache_hit'):
            receipt['cache_hit'] = True
        if accepted:
            receipt.update(_visual_reading(row.get('visual_observation'), include_text))
        if view := _view(args, image_views):
            receipt["view"] = view
        confidence = result.get("conf")
        if finite_number(confidence, maximum=1):
            receipt["model_confidence"] = confidence
        if code := _code(result.get("error_code")):
            receipt["error_code"] = code
        feature = args.get("feature_id")
        if tool == "presence_check":
            if feature in features:
                receipt.update(feature=feature, feature_name=_feature_name(feature, features))
            receipt["state"] = "unknown"
            if (accepted and view and feature in features and finite_number(confidence, minimum=.5, maximum=1)
                    and type(result.get("present")) is bool):
                receipt["state"] = "present" if result["present"] else "not_seen"
        elif accepted:
            for field, allowed in (("crystal_location", {"free", "intracellular"}),
                                   ("arrangement", {"rows", "scattered", "unknown"}),
                                   ("abundance", {"few", "many", "very_many", "unknown"})):
                if result.get(field) in allowed:
                    receipt[field] = result[field]
            if type(result.get("cell_wall_visible")) is bool:
                receipt["cell_wall_visible"] = result["cell_wall_visible"]
        receipts[key] = receipt
    return receipts


def _add_observations(outputs, rows):
    if rows:
        outputs["view_observations"] = rows
        reading = {"present": "可见", "not_seen": "本视野未见", "unknown": "未能判读"}
        scenes = {"free": "游离晶体", "intracellular": "细胞内晶体"}
        outputs["summary"] = "；".join(
            f"{row.get('view', '视野')} · "
            + (f"{row.get('feature_name', '目标特征')} {reading[row['state']]}" if "state" in row
               else scenes.get(row.get("crystal_location"), "场景未能判读"))
            for row in rows)
    return outputs


def _feedback(event, vocabulary):
    candidates, features = vocabulary
    feedback = event.get("feedback") or {}
    outputs = {}
    for key in ("accepted", "evidence_changed", "useful", "capability_progress", "candidate_order_changed"):
        value = feedback.get(key, event.get(key))
        if type(value) is bool:
            outputs[key] = value
    if new_features := [feature for feature in (feedback.get("new_features") or []) if feature in features]:
        outputs["new_features"] = new_features
    if code := _code(feedback.get("failure_code") or event.get("error_code")):
        outputs["failure_code"] = code
    measurement = event.get("automatic_measurement") or {}
    summary = {}
    if type(measurement.get("accepted")) is bool:
        summary["accepted"] = measurement["accepted"]
    if finite_number(measurement.get("latency_ms")):
        summary["latency_ms"] = measurement["latency_ms"]
    if code := _code(measurement.get("failure_code")):
        summary["error_code"] = code
    if summary:
        outputs["automatic_measurement"] = summary
    support = {cid: [fid for fid in items if fid in features]
               for cid, items in feedback.get("candidate_support_changes", {}).items() if cid in candidates}
    counters = []
    for item in feedback.get("qualified_counter_changes", []):
        candidate, _, feature = item.partition(":")
        if candidate in candidates and feature in features:
            counters.append({"candidate": candidate, "feature": feature})
    transitions = []
    for row in feedback.get("criterion_transitions", []):
        before, after = _code(row.get("before")), _code(row.get("after"))
        if row.get("candidate_id") in candidates and row.get("feature_id") in features:
            change = {"candidate": row["candidate_id"], "feature": row["feature_id"],
                      "before": before, "after": after}
            if row.get("role") in {"required", "supporting"}:
                change["role"] = row["role"]
            transitions.append(change)
    for key, values in (("new_support", support), ("new_counterevidence", counters),
                        ("criterion_transitions", transitions[:24])):
        if values:
            outputs[key] = values
    return outputs


def _usage(row):
    usage = row.get("usage") or {}
    keys = {"input_tokens": "prompt_tokens", "output_tokens": "completion_tokens",
            "total_tokens": "total_tokens"}
    counts = {key: usage[source] for key, source in keys.items()
              if type(usage.get(source)) is int and usage[source] >= 0}
    details = usage.get('completion_tokens_details')
    reasoning = details.get('reasoning_tokens') if isinstance(details, dict) else None
    if type(reasoning) is int and reasoning >= 0:
        counts['output_token_details'] = {'reasoning': reasoning}
    return counts


def _events(events):
    plans, results, execution, choices, parallel = {}, {}, [], {}, {}
    current = None
    last_recompute = None
    for event in events:
        phase = event.get("phase")
        number = event.get("request_number")
        if phase == "planning_started" and type(number) is int and number > 0:
            current = number
            last_recompute = None
        if phase in {"planning_started", "planner_offer", "planning_timing"} and type(number) is int and number > 0:
            plans.setdefault(number, {})[phase] = event
        elif phase in {"thought", "planning_error"} and current is not None:
            plans.setdefault(current, {})[phase] = event
            if phase == "thought" and isinstance(event.get("action_id"), str):
                choices[event["action_id"]] = (current, event)
                for partner in event.get("physical_partner_ids", []):
                    choices[partner] = (current, event)
        elif phase == "observation_group_wave":
            thought = plans.get(current, {}).get("thought", {})
            if (thought.get("execution_choice") or {}).get("kind") == "observe_questions":
                for aid in event.get("action_ids", []):
                    choices[aid] = (current, thought)
        elif phase == "identity_group_item" and isinstance(event.get("action_id"), str):
            thought = plans.get(current, {}).get("thought", {})
            if (thought.get("execution_choice") or {}).get("kind") == "identity_group":
                choices[event["action_id"]] = (current, thought)
        elif phase in {"recompute", "action_observation"}:
            key = event.get("action_id") or (event.get("result") or {}).get("action_id")
            if isinstance(key, str):
                results[key] = event
                if phase == "recompute":
                    last_recompute = key
        elif phase == "automatic_measurement_feedback" and last_recompute is not None:
            results[last_recompute] = {**results[last_recompute], "automatic_measurement": event.get("feedback")}
        elif phase == "parallel_join":
            actions = [value for value in event.get("actions", []) if isinstance(value, str)]
            if actions:
                parallel[actions[0]] = actions
        elif phase == "execution_timing":
            execution.append(event)
            last_recompute = None
    return plans, choices, results, execution, parallel


def _short_id(value):
    return value if isinstance(value, str) and _ACTION_CODE.fullmatch(value) else None


def _menu_target(fields, views, features):
    target = {}
    if view := _view(fields, views):
        target["view"] = view
    if fields.get("feature_id") in features:
        target["feature"] = fields["feature_id"]
    return target


def _source_box(value):
    return (isinstance(value, list) and len(value) == 4 and all(type(n) is int for n in value)
            and 0 <= value[0] < value[2] and 0 <= value[1] < value[3])


def _menu_options(menu, views, vocabulary):
    """展开本轮短编号，参数只保留可读的视野、特征和目标。"""
    candidates, features = vocabulary
    mapping = menu.get('view_id_map')
    if isinstance(mapping, dict):
        views = {**views, **{alias: views[original] for alias, original in mapping.items()
                             if re.fullmatch(r'@V[0-9]+@', alias) and isinstance(original, str)
                             and original in views}}
    options = {}
    action_states = menu.get("action_state") or {}
    for rows in (menu.get("actions", []), menu.get("parallel_observations", [])):
        for row in rows:
            alias = _short_id(row.get("action_id"))
            tool = row.get("tool")
            if alias is None or tool not in _TOOLS:
                continue
            linked = options.get(row.get("args_from_action"), {})
            target = {key: linked[key] for key in ("view", "feature") if key in linked}
            state = action_states.get(alias, {})
            target.update(_menu_target({**(row.get("args") or {}), **state},
                                       views, features))
            for field in ('target_source_px', 'roi_source_px'):
                if _source_box(state.get(field)):
                    target[field] = state[field]
            regions = state.get('candidate_regions_px')
            if isinstance(regions, list) and regions and all(_source_box(box) for box in regions):
                target['candidate_regions_px'] = regions
            options[alias] = {"id": alias, "tool": tool, **target}

    choices = menu.get("execution_options") or {}
    for field, id_field, entry_field in (
        ("goals", "goal_id", "entry_action_id"),
        ("questions", "question_id", "action_id"),
        ("problem_goals", "choice_id", "entry"),
    ):
        for row in choices.get(field, []):
            alias = _short_id(row.get(id_field))
            if alias is None:
                continue
            entry = row.get(entry_field)
            option = {**options.get(entry, {}), **_menu_target(row, views, features), "id": alias}
            if entry in options or entry == "reuse":
                option["entry"] = entry
            task = (row.get("goal") or {}).get("semantic_task")
            if task in {"verify_identity", "measure_length", "measure_width", "measure_area"}:
                option["goal"] = task
            if row.get("purpose") in _PURPOSES:
                option["purpose"] = row["purpose"]
            if row.get("dimension") in _DIMENSIONS:
                option["dimension"] = row["dimension"]
            consumers = row.get("criterion_consumers") or row.get("shared_consumers") or []
            if consumer_ids := [candidate for candidate in consumers if candidate in candidates]:
                option["candidates"] = consumer_ids
            options[alias] = option
    return options


def _planning_inputs(number, menu, options, vocabulary, *, include_visual_observations=False):
    inputs = {"planning_request": number}
    if not menu:
        return inputs
    candidates, features = vocabulary
    investigation = menu.get("investigation") or {}
    comparison = investigation.get("comparison_problems") or {}
    focus = investigation.get("investigation_focus") or {}
    inputs["menu_source"] = "planner_offer"
    if include_visual_observations:
        hints = {fid: hint for fid, hint in (menu.get('scan_feature_hints') or {}).items()
                 if fid in features and _can_show_visual_text(hint)}
        if hints:
            inputs['scan_feature_hints'] = hints
    readings = [_visual_reading(v, include_visual_observations) for v in menu.get('previous_visual_observations', ())]
    if readings:
        inputs['previous_visual_observations'] = [r['visual_observation'] for r in readings if r]
    budget = menu.get("remaining_budget") or {}
    quota = {key: budget[key] for key in ("planning", "observation", "network", "recompute", "total", "finalization")
             if type(budget.get(key)) is int and budget[key] >= 0}
    if quota:
        inputs["remaining_budget"] = quota
    for key, values in (("active_candidates", comparison.get("active_candidates")),
                        ("plan_a_leads", focus.get("a_leads"))):
        if known := [candidate for candidate in values or [] if candidate in candidates]:
            inputs[key] = known
    required = {}
    audit = investigation.get("candidate_audit") or {}
    for row in audit.get("candidates", []):
        if row.get("candidate_id") in candidates and isinstance(row.get("required"), dict):
            checks = {feature: state for feature, state in row["required"].items()
                      if feature in features and state in _AUDIT_STATES}
            if checks:
                required[row["candidate_id"]] = checks
    if required:
        inputs["required_criteria_at_request"] = required
    if 'receipts_by_feature' in audit:
        inputs['feature_receipt_counts'] = {fid: len(rows) for fid, rows in audit['receipts_by_feature'].items()
                                          if fid in features and isinstance(rows, list)}

    questions = []
    for row in comparison.get("questions", []):
        if row.get("feature_id") not in features or row.get("purpose") not in _PURPOSES:
            continue
        question = {"feature": row["feature_id"], "purpose": row["purpose"]}
        if row.get("dimension") in _DIMENSIONS:
            question["dimension"] = row["dimension"]
        consumers = [{"candidate": item["candidate_id"], "role": item["role"]}
                     for item in row.get("consumers", []) if item.get("candidate_id") in candidates
                     and item.get("role") in {"required", "supporting"}]
        if consumers:
            question["criteria"] = consumers
        for kind, key in (("feature_step", "direct_actions"), ("discovery_prerequisite", "discovery_actions")):
            ids = [item["action_id"] for item in row.get("entry_paths", [])
                   if item.get("kind") == kind and item.get("action_id") in options]
            if ids:
                question[key] = list(dict.fromkeys(ids))
        questions.append(question)
    if questions:
        inputs["comparison_questions"] = questions
    if options:
        inputs["available_choices"] = list(options.values())
    return inputs


def _plan_choice(rows, options, *, include_decision_summary, include_visual_observations=False, view_names=None):
    thought = rows.get("thought", {})
    choice = thought.get("execution_choice") or {}
    selected = choice.get("requested") or {}
    ids = [selected.get("action_id"), selected.get("goal_id"),
           *(selected.get("action_ids") or []), *(selected.get("question_ids") or [])]
    aliases = [alias for value in ids if (alias := _short_id(value))]
    requested = [options[alias] for alias in aliases if alias in options]
    tool = thought.get("tool") or (requested[0].get("tool") if requested else None)
    outputs = {"selected_tool": tool} if tool in _TOOLS else {}
    for alias, item in (thought.get('visual_guidance_choices') or {}).items():
        if alias not in options or not isinstance(item, dict):
            continue
        mode = item.get('mode')
        if mode not in {'agent_brief', 'explicit_preset', 'unspecified', 'invalid'}:
            continue
        entry = {'mode': mode}
        brief = (thought.get('visual_briefs') or {}).get(item.get('action_id'))
        if include_visual_observations and mode == 'agent_brief' and _can_show_visual_text(brief):
            entry['question'] = brief
        outputs.setdefault('visual_guidance_choices', {})[alias] = entry
    if isinstance(thought.get('visual_briefs'), dict):
        outputs['requested_visual_briefs'] = len(thought['visual_briefs'])
        if include_visual_observations:
            outputs['planned_visual_questions'] = [text for text in thought['visual_briefs'].values()
                                                  if _can_show_visual_text(text)]
    if aliases:
        outputs["menu_ids"] = aliases
    if requested:
        outputs["requested_choices"] = requested
    if choice.get("kind") in {"action", "instance_goal", "problem_goal", "observe_questions",
                              "physical_pair", "physical_group", "identity_group"}:
        outputs["decision_kind"] = choice["kind"]
    if include_decision_summary:
        notes = {}
        if thought.get('decision_summary_source') == 'candidate_audit':
            notes['required_investigation_summary'] = thought.get('decision_summary')
        if thought.get('decision_source') == 'model':
            notes['model_decision_summary'] = (thought.get('proposed_decision_summary')
                if 'required_investigation_summary' in notes else thought.get('decision_summary'))
        for name, note in notes.items():
            if isinstance(note, str):
                if name == 'model_decision_summary':
                    if thought.get('decision_summary_capture') == 'complete':
                        outputs['model_summary_capture'] = 'complete'
                    elif len(note) == 240:
                        outputs['model_summary_capture'] = 'legacy_240_char_boundary'
                if view_names:
                    note = _IMAGE_ID.sub(lambda m: view_names.get(m[0], m[0]), note)
                note = ' '.join(note.split())
                if note and not _SENSITIVE_TEXT.search(note):
                    outputs[name] = note
                elif note and name == 'model_decision_summary':
                    outputs['model_summary_visibility'] = 'withheld_by_text_filter'
    usage = _usage(thought or rows.get("planning_error", {}))
    if usage:
        outputs["usage_metadata"] = usage
    return outputs


def _scan_requests(run, operations, features, include_text):
    """按操作目录核对实际准备的局部参数和对应的定位回执。未返回的图块只登记已准备。"""
    scans = {}
    root = run.resolve()
    for key, op in operations.items():
        if op.get('tool') != 'feature_scan' or not re.fullmatch(r'[0-9a-f]{64}', key):
            continue
        fid = op.get('feature_id') or (op.get('target') or {}).get('feature_id')
        if fid not in features or not op.get('action_id'):
            continue
        folder = root / 'workflow' / 'ops' / key[:16] / 'scan'
        if not folder.resolve().is_relative_to(root):
            continue
        tiles = []
        for path in sorted(folder.glob('tile_*/locate.json')):
            match = re.fullmatch(r'tile_([0-9]+)', path.parent.name)
            returned = path.parent / 'detection' / 'region_result.json'
            if not match or not all(p.resolve().is_relative_to(root) for p in (path, returned)):
                continue
            try:
                ctx = optional_json(path)
            except (OSError, ValueError):
                continue
            if not isinstance(ctx, dict):
                continue
            roi = ctx.get('roi')
            if ctx.get('tool') != 'locate_feature' or ctx.get('feature_id') != fid or not _source_box(roi):
                continue
            brief = ctx.get('visual_brief')
            source = 'project_preset'
            if brief:
                source = 'project_preset_and_agent' if brief == op.get('visual_brief') else 'additional_brief_unattributed'
            tile = {'tile': int(match[1]), 'roi_source_px': roi, 'guidance_source': source, 'status': 'prepared'}
            try:
                receipt = optional_json(returned)
            except (OSError, ValueError):
                receipt = {}
            if not isinstance(receipt, dict):
                receipt = {}
            count = receipt.get('count')
            if receipt.get('feature_id') == fid and receipt.get('roi') == roi and type(count) is int and count >= 0:
                tile.update(status='worker_returned', crop_count=count)
            if include_text:
                for name, value in (('feature_hint', ctx.get('feature_hint')), ('visual_brief', brief)):
                    if _can_show_visual_text(value):
                        tile[name] = value
            tiles.append(tile)
        if tiles:
            scans[op['action_id']] = {'feature': fid, 'tiles': tiles}
    return scans


def _tool_span(key, row, start, parent, views, choices, results, vocabulary, observations, *,
               completed_actions=None, include_text=False, scans=None):
    tool = row.get("tool")
    if tool not in _TOOLS:
        tool = "workflow_tool"
    action = row.get("selected_action_id") or row.get("action_id")
    view = _view(row, views)
    inputs = {"tool": tool}
    if view:
        inputs["view"] = view
    if action in choices:
        inputs["planning_request"] = choices[action][0]
    outputs = _feedback(results.get(action, {}), vocabulary)
    if tool == 'feature_scan' and action in (scans or {}):
        outputs['scan_requests'] = scans[action]
    result = results.get(action, {}).get('result') or row
    if result.get('accepted') and result.get('status') == 'done':
        outputs.update(_visual_reading(result.get('visual_observation'), include_text))
        if result.get('cache_hit'):
            outputs['cache_hit'] = True
    if "accepted" not in outputs and type(row.get("accepted")) is bool:
        outputs["accepted"] = row["accepted"]
    if action in observations:
        _add_observations(outputs, [observations[action]])
    status = row.get("status")
    if status in _TERMINAL:
        outputs["status"] = status
    error = _code(row.get("error_code"), "TOOL_FAILED") if status in {"failed", "canceled", "cancelled"} else None
    feedback_receipts = results if completed_actions is None else completed_actions
    return Span(key, f"{view} · {tool}" if view else tool, start, _end(start, row.get("ended_at")),
                "tool", parent, inputs, outputs, error, awaiting_feedback=action not in feedback_receipts)


def _plan_a_summary(prediction, candidates):
    def labelled(row):
        label, probability = row.get("label"), row.get("probability")
        result = {"label": label} if label in candidates else {"label_available": False}
        if finite_number(probability, maximum=1):
            result["probability"] = probability
        return result

    views = []
    ensemble = prediction.get("ensemble_method") == "mean_softmax"
    predictions = prediction.get("view_predictions") or []
    policy = prediction.get("candidate_policy") or {}
    per_model = policy.get("multiview_per_model_top_k") if ensemble and len(predictions) > 1 else None
    if type(per_model) is not int or not 1 <= per_model <= 86:
        per_model = None
    for index, row in enumerate(predictions, 1):
        folds = [{"fold": n, **labelled(ranking[0])} for n, ranking in enumerate(row.get("fold_rankings") or [], 1)
                 if isinstance(ranking, list) and ranking and isinstance(ranking[0], dict)]
        average = [labelled(item) for item in (row.get("ranking") or [])[:3] if isinstance(item, dict)]
        if folds or average:
            view = {"view": f"view_{index:02d}", "average_top3": average}
            if not ensemble:
                view["fold_top1"] = folds
            elif per_model is not None:
                view["model_top_k"] = [{"model": n, "ranking": [
                    labelled(item) for item in (model.get("ranking") or [])[:per_model]
                    if isinstance(item, dict)]}
                    for n, model in enumerate(row.get("model_predictions") or [], 1)]
            views.append(view)
    result = {"views": views} if views else {}
    if ensemble:
        result["ensemble_method"] = "mean_softmax"
        if per_model is not None:
            result.update(candidate_source="multiview_model_top_k", per_model_top_k=per_model)
        else:
            k = policy.get("top_k", 8)
            if type(k) is int and 1 <= k <= 68:
                result[f"candidate_top{k}"] = [
                    labelled(item) for item in (prediction.get("ranking") or [])[:k] if isinstance(item, dict)]
    return result


def _criterion_measurements(rows, vocabulary):
    candidates, features = vocabulary
    result = []
    for row in rows:
        data = row.get("data") or {}
        cid, feature = data.get("candidate_id"), data.get("feature_id")
        if (row.get("kind") != "criterion" or data.get("dimension_eligible") is not True
                or cid not in candidates or feature not in features
                or data.get("state") not in {"IN_RANGE", "EXTREME", "OUT_OF_RANGE"}):
            continue
        stats, bounds = data.get("stats") or {}, data.get("quant_bounds") or {}
        dimension, median = bounds.get("dim"), stats.get("median")
        minimum, maximum = bounds.get("min"), bounds.get("max")
        if (dimension not in _DIMENSIONS or not finite_number(median) or median <= 0
                or not finite_number(minimum) or not finite_number(maximum, minimum=minimum)):
            continue
        measurement = {"candidate": cid, "feature": feature, "feature_name": _feature_name(feature, features),
                       "dimension": dimension, "unit": "µm²" if dimension == "area" else "µm",
                       "median": median, "reference_min": minimum, "reference_max": maximum, "state": data["state"]}
        if data.get("role") in {"required", "supporting"}:
            measurement["role"] = data["role"]
        if type(stats.get("n_measured")) is int and stats["n_measured"] > 0:
            measurement["n_measured"] = stats["n_measured"]
        result.append(measurement)
    return result[:24]


def _run_metrics(trace, spans, assessment, elapsed):
    metrics = {key: trace[key] for key in ("steps", "planning_requests", "observation_requests")
               if type(trace.get(key)) is int and trace[key] >= 0}
    if finite_number(elapsed):
        metrics["wall_time_seconds"] = round(elapsed, 3)
    reports = [span.outputs["usage_metadata"] for span in spans if span.kind == "llm" and "usage_metadata" in span.outputs]
    if reports:
        metrics["planning_returned_tokens"] = {
            key: sum(usage[key] for usage in reports if key in usage)
            for key in ("input_tokens", "output_tokens", "total_tokens") if any(key in usage for usage in reports)}
        metrics["planning_calls_with_usage"] = len(reports)
    if usage := _usage(assessment):
        metrics["final_assessment_returned_tokens"] = usage
    return metrics


def _final_details(root_outputs, output, assessment, trace, spans, elapsed, vocabulary):
    candidates = vocabulary[0]
    details = {}
    comparison = []
    for row in assessment.get("candidate_comparison", []):
        if row.get("candidate_id") in candidates:
            scores = {key: row[key] for key in ("rule_score", "fused_score", "verified_feature_count")
                      if finite_number(row.get(key))}
            comparison.append({"candidate": row["candidate_id"], **scores})
    if comparison:
        details["candidate_scores"] = comparison[:12]
    ranking = [{"candidate": row["id"], "score": row["score"]} for row in output.get("fusion_ranking", [])
               if row.get("id") in candidates and finite_number(row.get("score"))]
    if ranking:
        details["fusion_ranking"] = ranking[:12]
    version = output.get("rule_score_version")
    if isinstance(version, str) and re.fullmatch(r"b-rule/[0-9]{1,3}", version):
        details["rule_score_version"] = version
    if assessment.get("formal_verdict") in _DECISIONS:
        details["formal_b_verdict"] = assessment["formal_verdict"]

    audit = assessment.get("candidate_audit") or output.get("candidate_audit") or {}
    rows = [row for row in audit.get("candidates", []) if row.get("candidate_id") in candidates]
    if rows or audit.get("unmapped"):
        details["candidate_review"] = {"mapped": len(rows), "complete": sum(row.get("review_complete") is True for row in rows),
                                       "unmapped": len(audit.get("unmapped") or [])}
    for source, key in (("supporting_evidence", "supporting_measurements"), ("counterevidence", "counterevidence_measurements")):
        if measurements := _criterion_measurements(assessment.get(source, []), vocabulary):
            details[key] = measurements
    if metrics := _run_metrics(trace, spans, assessment, elapsed):
        details["run_metrics"] = metrics

    candidate, decision = root_outputs.get("candidate"), root_outputs.get("decision")
    if candidate or decision:
        parts = [" ".join(filter(None, (_DECISIONS.get(decision), candidate)))]
        if root_outputs.get("review_required") is True:
            parts.append("需复核")
        selected = next((row for row in comparison if row["candidate"] == candidate), {})
        if "rule_score" in selected:
            parts.append(f"B 分 {selected['rule_score']:.4f}")
        if "fused_score" in selected:
            parts.append(f"融合分 {selected['fused_score']:.4f}")
        if review := details.get("candidate_review"):
            parts.append(f"完整审查 {review['complete']}/{review['mapped']} 个映射候选")
            if review["unmapped"]:
                parts.append(f"{review['unmapped']} 项未映射")
        details = {"summary": " · ".join(parts), **details}
    return details


def project_trace(run, vocabulary, *, include_decision_summary=False, include_visual_observations=False):
    """返回一个可更新的父子调用快照，缺少可证实的开始时间时等待下一次写盘。"""
    job_file = run / "job.json"
    job = read_json(job_file)
    job_id = job.get("job_id")
    if not isinstance(job_id, str) or not _RUN_ID.fullmatch(job_id) or job_id != run.name:
        raise IntegrationError("TRACE_RUN_ID", "job.json 与融合运行目录不匹配。")
    case = run.parent.parent if run.parent.name == "runs" else run
    trace = optional_json(run / "agent_trace.json")
    timeline = optional_json(run / "workflow" / "action_timeline.json")
    plans, choices, results, execution, parallel = _events(trace.get("events", []))
    pipeline = _stamp(optional_json(run / "code_signature_start.json").get("at"))
    interface = optional_json(case / "interface_status.json")
    start = pipeline or _stamp(interface.get("started_at")) or _stamp(timeline.get("started_at"))
    if start is None:
        if job.get("status") in _TERMINAL:
            raise IntegrationError("TRACE_START_MISSING", "已结束的任务缺少可核对的开始时间。")
        return job_id, []
    root = Span("case", "TCM_Fusion · 样品鉴别", start, parent=None, inputs={"fusion_run_id": job_id})
    spans = [root]
    views = optional_json(case / "input_proof.json").get("views", [])
    a = optional_json(run / "a_prediction.json")
    hashes = [view.get("sha256") for view in views] or [row.get("input_sha256") for row in a.get("view_predictions", [])]
    image_views = {f"img_{sha}": f"view_{i:02d}" for i, sha in enumerate(hashes, 1)
                   if isinstance(sha, str) and re.fullmatch(r"[0-9a-f]{64}", sha)}
    observations = _observation_receipts(trace.get("observations", []), image_views, vocabulary[1],
                                         include_text=include_visual_observations)
    view_count = interface.get("view_count") or len(hashes)
    if type(view_count) is int and view_count > 0:
        root.inputs["view_count"] = view_count
    if type(trace.get("simulation")) is bool:
        root.outputs["simulation"] = trace["simulation"]

    a_end = _end(start, optional_json(run / "code_signature_before_agent.json").get("at"))
    if pipeline and (job.get("stage") == "a" or a_end is not None):
        a_outputs = {"status": "done"} if a_end is not None else {}
        a_outputs.update(_plan_a_summary(a, vocabulary[0]))
        spans.append(Span("plan-a", "Plan A · 候选提名及入口检查", start, a_end, outputs=a_outputs))

    for number, rows in sorted(plans.items()):
        timing = rows.get("planning_timing", {})
        began = _stamp(rows.get("planning_started", {}).get("started_at")) or _stamp(timing.get("started_at"))
        if began is None:
            continue
        error = _code(rows.get("planning_error", {}).get("reason"), "PLANNING_ERROR") if "planning_error" in rows else None
        offer = rows.get("planner_offer") or {}
        menu = offer.get("menu") if offer.get("actual_llm_short_ids") is True else None
        options = _menu_options(menu, image_views, vocabulary) if menu else {}
        spans.append(Span(f"plan-{number}", f"规划请求 {number}", began, _end(began, timing.get("ended_at")),
                          "llm", inputs=_planning_inputs(number, menu, options, vocabulary,
                                                        include_visual_observations=include_visual_observations),
                          outputs=_plan_choice(rows, options, include_decision_summary=include_decision_summary,
                                               include_visual_observations=include_visual_observations,
                                               view_names={v: image_views[v] for v in ((menu or {}).get('view_id_map') or {}).values()
                                                           if isinstance(v, str) and v in image_views}),
                          error=error, awaiting_feedback="thought" not in rows and error is None))

    operations = optional_json(run / "workflow" / "operations.json").get("entries", {})
    scans = _scan_requests(run, operations, vocabulary[1], include_visual_observations)
    groups = {}
    for key, row in operations.items():
        if row.get("epoch") and _stamp(row.get("selected_at")) is not None:
            groups.setdefault(row["epoch"], []).append((key, row))
    staged_ids = set()
    for epoch, rows in groups.items():
        began = min(_stamp(row["selected_at"]) for _, row in rows)
        group_key = f"d4-{epoch}"
        children = []
        for key, row in rows:
            worker_start = _stamp(row.get("started_at"))
            if worker_start is None and row.get("status") not in _TERMINAL:
                continue
            child = _tool_span(f"{group_key}-{key}", row, worker_start or _stamp(row["selected_at"]),
                               group_key, image_views, choices, results, vocabulary, observations,
                               include_text=include_visual_observations, scans=scans)
            if worker_start is None:
                child.inputs["timing"] = "reservation_without_worker_start"
            children.append(child)
        if not children:
            continue
        complete = len(children) == len(rows) and all(child.end is not None for child in children)
        ended = max(child.end for child in children) if complete else None
        spans.append(Span(group_key, "D4 · 跨视野并发", began, ended,
                          inputs={"branch_count": len(rows)},
                          outputs={"failed_branches": sum(child.error is not None for child in children)},
                          awaiting_feedback=any(child.awaiting_feedback for child in children)))
        spans.extend(children)
        staged_ids.update(row.get("action_id") for _, row in rows)
    completed_actions = {row.get("action_id") for row in execution if _stamp(row.get("ended_at")) is not None}
    for key, row in timeline.get("actions", {}).items():
        began = _stamp(row.get("started_at"))
        action = row.get("selected_action_id") or row.get("action_id")
        if began is not None and action not in staged_ids:
            spans.append(_tool_span(f"tool-{key}-{began}", row, began, "case", image_views,
                                    choices, results, vocabulary, observations, completed_actions=completed_actions,
                                    include_text=include_visual_observations, scans=scans))
            staged_ids.add(action)
    for row in execution:
        action = row.get("action_id")
        began = _stamp(row.get("started_at"))
        if began is None or action in staged_ids or not isinstance(action, str):
            continue
        request, thought = choices.get(action, (None, {}))
        tool = observations.get(action, {}).get("tool") or thought.get("tool")
        name = "并行观察" if row.get("parallel") is True else (tool if tool in _TOOLS else "Agent 执行动作")
        observed = [observations[key] for key in parallel.get(action, [action]) if key in observations]
        feedback = _add_observations(_feedback(results.get(action, {}), vocabulary), observed)
        if action in scans:
            feedback['scan_requests'] = scans[action]
        result = results.get(action, {}).get('result') or {}
        if result.get('accepted') and result.get('status') == 'done':
            feedback.update(_visual_reading(result.get('visual_observation'), include_visual_observations))
        spans.append(Span(f"exec-{action}-{began}", name, began,
                          _end(began, row.get("ended_at")), "tool", inputs={"planning_request": request},
                          outputs=feedback))

    status = job.get("status")
    if status in _TERMINAL:
        ends = [span.end for span in spans if span.end is not None]
        elapsed = job.get("elapsed_seconds")
        end_basis = "recorded_pipeline_elapsed" if pipeline and finite_number(elapsed) else "terminal_job_mtime"
        estimate = start + elapsed if end_basis == "recorded_pipeline_elapsed" else job_file.stat().st_mtime
        root.end = max(estimate, *(span.start for span in spans), *ends)
        root.outputs["status"] = status
        root.outputs["end_time_basis"] = end_basis
        if status != "completed":
            root.error = _code(job.get("error_code"), "WORKFLOW_FAILED")
            if job.get("stage") == "a" and a_end is None and len(spans) > 1 and spans[1].key == "plan-a":
                spans[1].end = root.end
                spans[1].error = root.error
                spans[1].outputs = {"status": "failed", "end_time_basis": end_basis}
        unresolved = sum(span.end is None for span in spans[1:])
        if unresolved:
            root.outputs["spans_without_end_receipt"] = unresolved
        output = optional_json(run / "output.json")
        result = output.get("status")
        if result in _DECISIONS:
            root.outputs["decision"] = result
        if type(output.get("review_required")) is bool:
            root.outputs["review_required"] = output["review_required"]
        candidate = (output.get("top_candidate") or {}).get("id")
        if candidate in vocabulary[0]:
            root.outputs["candidate"] = candidate
        assessment = optional_json(run / "final_assessment.json")
        reasons = [value for value in assessment.get("review_reasons", []) if _code(value)]
        if reasons:
            root.outputs["review_reasons"] = reasons
        details = _final_details(root.outputs, output, assessment, trace, spans, elapsed, vocabulary)
        root.outputs = {**details, **root.outputs}
    return job_id, spans


class LangSmithPublisher:
    """实时调用提交一次结束。历史回放直接写入完整调用，按快照复用 ID。"""

    def __init__(self, client, destination, *, project_name=None, address=None):
        self.client = client
        self.destination = destination
        self.project_name = project_name
        self.address = address
        self.sent = set()
        self.closed = set()
        self.replay_needed = set()

    @staticmethod
    def _finish(tree, span, offset=timedelta()):
        usage = span.outputs.get("usage_metadata")
        tree.end(outputs=span.outputs, error=span.error,
                 end_time=datetime.fromtimestamp(span.end, timezone.utc) + offset,
                 metadata={"usage_metadata": usage} if usage else None)

    def _post_spans(self, trace_id, spans, *, replay=False, offset=timedelta()):
        from langsmith.run_trees import RunTree
        from langsmith.utils import LangSmithConflictError

        trees = {}
        for span in spans:
            start = datetime.fromtimestamp(span.start, timezone.utc)
            args = {"name": span.name, "run_type": span.kind, "inputs": span.inputs,
                    "start_time": start + offset}
            if offset:
                original_end = None if span.end is None else datetime.fromtimestamp(span.end, timezone.utc)
                args["extra"] = {"metadata": {"replay_time_shifted": True,
                                             "original_start_time": start.isoformat(),
                                             "original_end_time": original_end.isoformat() if original_end else None}}
            if span.parent is None:
                if replay:
                    args["name"] = "TCM_Fusion · 样品鉴别回放"
                tree = RunTree(id=trace_id, ls_client=self.client, project_name=self.project_name,
                               address=self.address, **args)
            else:
                tree = trees[span.parent].create_child(run_id=uuid5(trace_id, span.key), **args)
            trees[span.key] = tree
            if tree.id not in self.sent:
                if replay and span.end is not None:
                    self._finish(tree, span, offset)
                try:
                    tree.post()
                except LangSmithConflictError:
                    pass
                self.sent.add(tree.id)
        return trees

    def _publish_replay(self, original_id, spans):
        snapshot = digest([asdict(span) for span in spans])
        trace_id = uuid5(original_id, f"archive-replay/v1/{snapshot}")
        now = datetime.now(timezone.utc)
        earliest = datetime.fromtimestamp(min(span.start for span in spans), timezone.utc)
        latest_start = datetime.fromtimestamp(max(span.start for span in spans), timezone.utc)
        # Leave room for clock skew and the time it takes to submit the whole tree.
        window = timedelta(hours=23, minutes=55)
        offset = timedelta()
        if earliest < now - window or latest_start > now + window:
            if latest_start - earliest > window * 2:
                raise IntegrationError("TRACE_REPLAY_TIME_RANGE", "归档跨度过长，无法在同一次回放中保留所有调用的时间间隔。")
            latest_end = datetime.fromtimestamp(max(span.end or span.start for span in spans), timezone.utc)
            offset = max(now - latest_end, now - window - earliest)
        self._post_spans(trace_id, spans, replay=True, offset=offset)
        return trace_id

    def sync(self, job_id, spans):
        from langsmith.utils import LangSmithConflictError

        trace_id = uuid5(NAMESPACE_URL, f"TCM_Fusion:{self.destination}:{job_id}")
        terminal = spans[0].end is not None
        if terminal and trace_id not in self.sent:
            return self._publish_replay(trace_id, spans)

        trees = self._post_spans(trace_id, spans)
        for span in reversed(spans):
            tree = trees[span.key]
            if span.end is None or tree.id in self.closed or (span.awaiting_feedback and not terminal):
                continue
            self._finish(tree, span)
            try:
                tree.patch(exclude_inputs=False)
            except LangSmithConflictError as exc:
                reason = str(exc).casefold()
                if "payload already received" not in reason and "payloads already received" not in reason:
                    raise IntegrationError("TRACE_UPLOAD_CONFLICT",
                        f"LangSmith 拒绝结束调用「{span.name}」（HTTP 409）。") from None
                # A previous observer may have closed it. Publish the complete snapshot on exit.
                self.replay_needed.add(trace_id)
            self.closed.add(tree.id)
        if terminal and trace_id in self.replay_needed:
            return self._publish_replay(trace_id, spans)
        return trace_id
