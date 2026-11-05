"""离线反思→候选→开发集工具回放→版本化经验；不更新权重，不在识别时调用模型。

首版只允许把连续无增益停止阈值2收紧到1；不改判据、证据类型、图像工具权限。
回放采用显式模拟观察，发布版本仅允许simulation运行使用，不冒充真实识别增益。
"""
from __future__ import annotations

import uuid
from pathlib import Path

from .storage import IntegrationError, digest, output_root, read_json, write_json


def save_artifact(kind, value):
    path = output_root()/"reflection"/kind/(uuid.uuid4().hex+".json")
    write_json(path, value)
    return path


def propose(trace, feedback, *, llm=False, allow_network=False):
    if feedback.get("trace_sha256") != digest(trace):
        raise IntegrationError("REFLECTION_FEEDBACK", "反馈未绑定该轨迹摘要")
    if feedback.get("source") != "program_check" or feedback.get("issue") != "repeated_no_gain":
        raise IntegrationError("REFLECTION_FEEDBACK", "首版只接受可程序复核的连续无增益反馈")
    events = [e for e in trace.get("events", []) if e.get("phase") == "action_observation"]
    if not any(not x.get("evidence_changed", True) and not y.get("evidence_changed", True)
               for x,y in zip(events, events[1:])):
        raise IntegrationError("REFLECTION_FEEDBACK", "轨迹不能证明连续两次无增益")
    reason = "已观察到连续无增益；提出更早停止候选，需开发集验证是否损害结论。"
    mode = "program_feedback"
    if llm:
        if not allow_network:
            raise IntegrationError("NETWORK_DISABLED", "离线LLM反思须显式允许联网")
        from adjudicator import AdjudicatorClient
        import paths
        import json
        client = AdjudicatorClient(Path(paths.DETECTION_CONFIG_DIR)/"vlm_config.json")
        if not client.api_key:
            raise IntegrationError("CONFIG_MISSING", "缺反思LLM凭据")
        # 仅发结构化执行信号，无原图、图像路径、药材真值或模型自由执行权限。
        body = client._post({"model": client.model, "temperature": 0, "max_tokens": 2048,
            "messages": [{"role": "system", "content":
                "分析连续无增益工具调用的执行问题。输出一句候选经验；不能改药典、尺寸、结论或宣称已验证提升。"},
                {"role": "user", "content": json.dumps([
                    {"accepted": e.get("accepted"), "evidence_changed": e.get("evidence_changed")}
                    for e in events])}]})
        reason = str(body["choices"][0]["message"]["content"])[:500]
        mode = "llm_reflection"
    return {"schema": "reflection-candidate/v1", "status": "candidate", "mode": mode,
            "source_trace_sha256": digest(trace), "feedback_sha256": digest(feedback),
            "policy": {"no_gain_limit": 1}, "lesson": reason,
            "scope": "simulation", "weight_update": False}


def check_candidate(candidate):
    if (candidate.get("schema") != "reflection-candidate/v1" or candidate.get("status") != "candidate"
            or candidate.get("policy") != {"no_gain_limit": 1} or candidate.get("scope") != "simulation"):
        raise IntegrationError("REFLECTION_POLICY", "不支持的候选策略；禁止自由改写规则")


async def validate(candidate, cases):
    from .agent_tools import FixtureBackend, LocalToolbox
    from .evidence_runtime import EvidenceRuntime
    from .react_graph import run_react
    check_candidate(candidate)
    if cases.get("split") != "development" or not cases.get("cases"):
        raise IntegrationError("REFLECTION_SPLIT", "需要非空development回放集，禁止用test调优")
    directory = output_root()/"reflection"/"replays"/uuid.uuid4().hex
    rows = []
    for i, case in enumerate(cases["cases"]):
        expected = case.get("expected")
        if not isinstance(expected, dict) or set(expected) != {"verdict", "conclusion_id"}:
            raise IntegrationError("REFLECTION_TRUTH", "每例须显式提供参考verdict和conclusion_id")
        results = []
        for threshold in (2, 1):
            runtime = EvidenceRuntime(case["records"], case["images"], sample_id=f"replay_{i}",
                                      a=case.get("a"), simulation=True)
            tools = LocalToolbox(runtime, ledger_path=directory/f"{i}_{threshold}.json",
                                 backend=FixtureBackend(case["replies"]))
            result = await run_react(runtime, tools, planner_mode="heuristic", no_gain_limit=threshold)
            d = result["decision"]
            signature = {"verdict": d["verdict"], "conclusion_id": (d.get("conclusion") or {}).get("id")}
            results.append({"signature": signature, "requests": result["requests"],
                            "evidence_digest": runtime.score(result["observations"])["evidence_digest"],
                            "needs_review": result["needs_review"]})
        baseline, changed = results
        rows.append({"case": i, "baseline": baseline, "candidate": changed,
                     "pass": baseline["signature"] == changed["signature"] == expected
                     and baseline["evidence_digest"] == changed["evidence_digest"]
                     and baseline["needs_review"] == changed["needs_review"]
                     and changed["requests"] <= baseline["requests"]})
    passed = all(r["pass"] for r in rows) and any(r["candidate"]["requests"] < r["baseline"]["requests"] for r in rows)
    report = {"schema": "reflection-validation/v1", "candidate_sha256": digest(candidate),
              "cases_sha256": digest(cases), "split": "development", "scope": "simulation",
              "passed": passed, "rows": rows, "real_accuracy_validated": False}
    # 凭证与实际回放输入一起归档，发布时可再次回放，而不是相信手填passed。
    report["replay_cases"] = cases
    return report


async def publish(candidate, report):
    check_candidate(candidate)
    if report.get("candidate_sha256") != digest(candidate):
        raise IntegrationError("REFLECTION_VALIDATION", "验证报告与候选不匹配")
    repeated = await validate(candidate, report.get("replay_cases", {}))
    if not report.get("passed") or not repeated["passed"] or repeated["cases_sha256"] != report.get("cases_sha256"):
        raise IntegrationError("REFLECTION_VALIDATION", "未通过重新回放，禁止发布")
    body = {"schema": "reflection-memory/v1", "scope": "simulation", "policy": candidate["policy"],
            "lesson": candidate["lesson"], "candidate": candidate,
            "validation": repeated, "version": uuid.uuid4().hex}
    body["sha256"] = digest(body)
    return save_artifact("versions", body)


def load_memory(path, *, simulation):
    value = read_json(path)
    body = {k:v for k,v in value.items() if k != "sha256"}
    if value.get("schema") != "reflection-memory/v1" or digest(body) != value.get("sha256"):
        raise IntegrationError("REFLECTION_MEMORY", "经验版本校验失败")
    check_candidate(value["candidate"])
    if (not simulation or value.get("scope") != "simulation" or not value["validation"].get("passed")
            or value["policy"] != value["candidate"]["policy"]):
        raise IntegrationError("REFLECTION_SCOPE", "模拟回放经验不得进入真实识别")
    return value
