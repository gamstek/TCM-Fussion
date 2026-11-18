"""A/B 只读协作与有界影子 Agent：融合两路结果并给出可选补查建议，不改写原工件。

不修改 EvidenceCard、measurements 或原 Verdict；合法补查仅取原 B 明确 NOT_DETECTED
的 P1/C required 判据，A 不扩张补查权限。本版不执行新增图像观察，LLM 只选合法动作，
失败保留确定性建议；规划预算只覆盖已记录判药请求与本次规划。

主要定义：
- knowledge_base: 读取药材判据知识库 JSON。
- fuse: 比较 A 首选与 B 结论的关系，给出是否需复核，不修改 B 裁决。
- propose_actions: 由 B 判据生成候选补查动作及 finish/review 选项。
- select_shadow: 在预算内输出确定性或 LLM 影子选择，最多一次规划请求。

关联：
- 调用方 「lib/agent/react_graph.py」、「lib/application/pipeline.py」、「lib/application/workflow_entry.py」、「lib/offline/confidence_gate.py」、「tests/integration/test_integration.py」、「tests/offline/test_confidence_gate.py」。
- 直接依赖 「lib/runtime/storage.py」。
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import paths

from TCM_Fusion.lib.runtime.storage import IntegrationError, digest, file_hash, read_json


def knowledge_base():
    return read_json(Path(paths.JUDGE_DIR)/"data"/"herb_criteria.json")


def fuse(a, b, kb, aliases=None):
    """分类分数不参与B评分，组成员命中也不升级为药材级confirmed。"""
    original = copy.deepcopy(b)
    aliases = aliases or {}
    a_name = a.get("ranking", [{}])[0].get("label") if a.get("ranking") else None
    mapped = aliases.get(a_name, a_name)
    herbs = {h["herb"] for h in kb["herbs"] if not h.get("no_criteria")}
    conclusion = b.get("conclusion") or {}
    bid = conclusion.get("id")
    members = kb.get("groups", {}).get(bid, {}).get("members", [bid] if bid else [])
    if a.get("status") != "ok":
        relation = "a_unavailable"
    elif mapped not in herbs:
        relation = "a_unsupported"
    elif not bid or b.get("verdict") == "insufficient_evidence":
        relation = "b_insufficient"
    elif mapped in members:
        relation = "group_compatible" if bid in kb.get("groups", {}) else "agree"
    else:
        relation = "conflict"
    return {"agreement": relation, "a_label": a_name, "a_mapped_label": mapped,
            "a_probability": a.get("ranking", [{}])[0].get("probability") if a.get("ranking") else None,
            "b_verdict": original, "source": "original_b_rules",
            "needs_review": bool(b.get("needs_review")) or relation in {"conflict", "b_insufficient"},
            "a_is_independent_evidence": False, "verdict_modified": False}


def propose_actions(b):
    """候选提案与图像存在性校验；不声称已执行图像观察。"""
    evidence_version = digest(b)
    terms = (b.get("score_breakdown") or {}).get("terms", [])
    actions = []
    seen = set()
    for image in b.get("source_images", []):
        path = Path(image)
        if not path.is_file():
            continue
        for term in terms:
            if (term.get("role") != "required" or term.get("tier") not in {"P1", "C"}
                    or term.get("state") != "NOT_DETECTED"):
                continue
            key = (file_hash(path), term["feature_id"])
            if key in seen:
                continue
            seen.add(key)
            action = {"kind": "presence_check", "feature_id": term["feature_id"],
                      "image_ref": str(path.resolve()), "image_sha256": key[0],
                      "evidence_version": evidence_version,
                      "requires_history_verification": True}
            action["action_id"] = "check_"+digest(action)[:16]
            actions.append(action)
    actions.sort(key=lambda a: (a["feature_id"], a["image_sha256"]))
    actions += [{"action_id": "finish", "kind": "finish", "evidence_version": evidence_version},
                {"action_id": "review", "kind": "request_review", "evidence_version": evidence_version}]
    return actions


def select_shadow(fusion, b, config, *, allow_network=False, client=None):
    """真实LLM最多一次规划调用；全模式无新增Observation/无证据注入。

原B缺完整所有阶段API计数，故此预算明确只覆盖已记录判药请求＋新规划，非整链总成本。
"""
    mode = config.get("mode", "deterministic_shadow")
    if mode not in {"off", "deterministic_shadow", "llm_shadow"}:
        raise IntegrationError("AGENT_MODE_UNSUPPORTED", "本版Agent仅支持off/确定性shadow/LLM shadow")
    budget = config.get("judgement_request_budget", 6)
    if isinstance(budget, bool) or not isinstance(budget, int) or budget < 0:
        raise IntegrationError("CONFIG_INVALID", "judgement_request_budget须为非负整数")
    used = (b.get("agent") or {}).get("vlm_calls", 0)
    if isinstance(used, bool) or not isinstance(used, int) or used < 0:
        raise IntegrationError("B_VERDICT_INVALID", "B判药调用计数非法")
    used += len((b.get("shadow_bounded_adjudicator") or {}).get("adjudications", []))
    actions = propose_actions(b)
    need = fusion["needs_review"]
    checks = [a for a in actions if a["kind"] == "presence_check"]
    choice = checks[0]["action_id"] if need and checks else ("review" if need else "finish")
    result = {"mode": mode, "shadow_only": True, "observations_executed": 0,
              "budget_scope": "recorded_judgement_requests_plus_planning",
              "existing_requests": used, "planning_requests": 0, "budget": budget,
              "history_complete": False, "actions": actions, "choice": choice,
              "reason": "确定性建议；原B未记录全部历史，执行补查前需补齐去重轨迹",
              "degraded": False}
    if mode == "off":
        return {**result, "choice": None, "reason": "Agent关闭"}
    if used >= budget:
        return {**result, "choice": "review" if need else "finish", "reason": "无剩余判药规划预算"}
    if mode != "llm_shadow" or not need:
        return result
    if not allow_network and client is None:
        raise IntegrationError("NETWORK_DISABLED", "LLM规划须显式allow_network=true")
    result["planning_requests"] = 1  # 失败也算，不自动重试
    try:
        if client is not None:
            reply = client(actions)
        else:
            # 复用原B文本传输，不复制HTTP原语、不使用其仲裁提示词。
            # _post为现有非公开接口，本包回归对该耦合做显式覆盖。
            from adjudicator import AdjudicatorClient
            cfg_path = Path(paths.DETECTION_CONFIG_DIR)/"vlm_config.json"
            vlm = read_json(cfg_path)["vlm"]
            transport = AdjudicatorClient(cfg_path)
            if not transport.api_key:
                raise IntegrationError("CONFIG_MISSING", "缺LLM密钥")
            # 不传A猜测答案或图像，避免把分类先验混入事实观察。
            payload = {"model": transport.model, "temperature": 0.0,
                       "max_tokens": min(int(vlm.get("max_tokens", 4096)), 4096),
                       "messages": [{"role": "system", "content": "只选择合法action_id；不判断药材，不编造观察。输出JSON: action_id, reason。"},
                                    {"role": "user", "content": json.dumps({"actions": actions, "needs_review": need}, ensure_ascii=False)}]}
            if vlm.get("reasoning_effort"):
                payload["reasoning_effort"] = vlm["reasoning_effort"]
            body = transport._post(payload)
            reply = json.loads(body["choices"][0]["message"]["content"])
        action_id = reply["action_id"]
        if action_id not in {a["action_id"] for a in actions}:
            raise ValueError("非法动作")
        result.update(choice=action_id, reason=str(reply.get("reason", ""))[:300])
    except Exception as exc:
        result.update(degraded=True, error_type=type(exc).__name__, reason="规划失败，保留确定性影子建议")
    return result
