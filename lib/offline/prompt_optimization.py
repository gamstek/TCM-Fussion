"""LangMem 离线提示优化与封闭工具回放：发布显式版本，在线只加载提示。

优化器只看脱敏执行信号；回放规划器只看当时已发生的观察，未录制的动作令验证失败。
返回模型配置文件的哈希，用于绑定优化/验证时的配置。由「cli/offline.py」、「lib/offline/truth_prompt_validation.py」调用，结果按调用方的任务阶段消费。

主要定义：
- save。
- chat_model。
- optimize。
- RecordedTools。
- validate。
- publish。

关联：
- 调用方 「cli/offline.py」、「lib/offline/truth_prompt_validation.py」、「tests/offline/test_prompt_optimization.py」、「tests/offline/test_workflow_prompt.py」。
- 直接依赖 「lib/agent/prompt_defaults.py」、「lib/agent/prompt_release.py」、「lib/agent/react_graph.py」、「lib/offline/truth_prompt_validation.py」、「lib/offline/workflow_prompt_validation.py」、「lib/runtime/storage.py」、「lib/workflow/evidence_runtime.py」。
"""
from __future__ import annotations

import json
import os
import uuid
from pathlib import Path

import paths

from TCM_Fusion.lib.runtime.storage import IntegrationError, digest, file_hash, output_root, read_json, write_json
from TCM_Fusion.lib.agent.prompt_defaults import DEFAULT_INSTRUCTION, GUARD
from TCM_Fusion.lib.agent.prompt_release import profile, check_candidate, runtime_context, load_release


def save(kind, value):
    path = output_root()/"prompt_optimization"/kind/(uuid.uuid4().hex+".json")
    write_json(path, value)
    return path


def chat_model():
    from langchain_openai import ChatOpenAI
    cfg = read_json(Path(paths.DETECTION_CONFIG_DIR)/"vlm_config.json")["vlm"]
    key = cfg.get("api_key") or os.environ.get(cfg.get("api_key_env", "VLM_API_KEY"))
    if not key:
        raise IntegrationError("CONFIG_MISSING", "缺LangMem离线优化模型凭据")
    return ChatOpenAI(model=cfg["model"], base_url=cfg["base_url"], api_key=key,
                      temperature=0, max_tokens=4096, max_retries=0, timeout=cfg.get("timeout_s", 60))


async def optimize(training, *, allow_network=False, model=None):
    from langmem import create_prompt_optimizer
    from langmem.prompts.types import AnnotatedTrajectory
    if training.get("split") != "training" or not training.get("examples"):
        raise IntegrationError("PROMPT_SPLIT", "优化仅接受非空training轨迹，不能使用验证/测试集")
    if model is None and not allow_network:
        raise IntegrationError("NETWORK_DISABLED", "LangMem文本优化需要显式允许联网")
    trajectories = []
    groups = set()
    for item in training["examples"]:
        if not item.get("group_id") or not isinstance(item.get("feedback"), str):
            raise IntegrationError("PROMPT_DATA", "每条训练轨迹须有group_id和反馈文字")
        trace = item["trace"]
        if item.get("trace_sha256") != digest(trace):
            raise IntegrationError("PROMPT_DATA", "训练反馈与轨迹摘要未绑定")
        groups.add(item["group_id"])
        # 仅白名单统计信号，剥离路径、原图、API内容、完整识别结论及内部推理。
        signals = [{k: e.get(k) for k in ("phase", "accepted", "evidence_changed", "pipeline_progress", "error_code", "repair_completed", "feedback")}
                   for e in trace.get("events", []) if e.get("phase") in {"action_observation", "recompute"}]
        outcome=item.get('outcome')
        if outcome is not None:
            if outcome.get('schema')!='outcome-feedback/v1' or outcome.get('trace_sha256')!=digest(trace) or outcome.get('split')!='training' or outcome.get('group_id')!=item['group_id']:
                raise IntegrationError('PROMPT_FEEDBACK','真值反馈与训练轨迹/分组不匹配')
            signals.append({'outcome':{k:outcome.get(k) for k in ('correct','abstained','a_correct','corrected_a','broke_correct_a','failure_attribution')}})
        trajectories.append(AnnotatedTrajectory(
            messages=[{"role": "user", "content": json.dumps(signals)}],
            feedback={"developer_feedback": item["feedback"][:2000]}))
    optimizer = create_prompt_optimizer(model or chat_model(), kind="metaprompt",
                    config={"max_reflection_steps": 1, "min_reflection_steps": 1})
    instruction = await optimizer.ainvoke({"trajectories": trajectories,
        "prompt": {"name": "evidence_scheduling", "prompt": DEFAULT_INSTRUCTION,
                   "update_instructions": "仅改进动作选择、停止和复核的调度建议；禁止改药典判据、尺度、工具权限和输出schema。"}})
    if not isinstance(instruction, str) or not instruction.strip() or len(instruction) > 6000:
        raise IntegrationError("PROMPT_OUTPUT", "LangMem未返回合法长度的新提示词")
    return {"schema": "prompt-candidate/v1", "generator": "langmem.metaprompt",
            "instruction": instruction, "baseline": DEFAULT_INSTRUCTION, "guard_sha256": digest(GUARD),
            "training_sha256": digest(training), "training_groups": sorted(groups),
            "profile": profile(), "simulation": model is not None}


class RecordedTools:
    def __init__(self, rows):
        self.rows = {}
        self.misses = []
        for row in rows:
            action = row["action"]
            key = action["action_id"]
            if key in self.rows and self.rows[key] != row:
                raise IntegrationError("REPLAY_CONFLICT", "同动作录制了不同响应，拒绝静默覆盖")
            self.rows[key] = row

    async def observe(self, action):
        row = self.rows.get(action["action_id"])
        if row is None or row["action"] != action:
            self.misses.append(action["action_id"])
            raise IntegrationError("REPLAY_UNKNOWN", "未录制动作，回放不能验证")
        return {"action_id": action["action_id"], "accepted": row["accepted"],
                "result": row["result"], "cache_hit": True, "new_requests": 0}


async def validate(candidate, dataset, *, allow_network=False, planner_factory=None):
    from TCM_Fusion.lib.agent.react_graph import LLMPlanner, run_react
    from TCM_Fusion.lib.workflow.evidence_runtime import EvidenceRuntime
    check_candidate(candidate)
    if candidate.get("profile") != profile():
        raise IntegrationError("PROMPT_PROFILE", "模型配置已改变，须重新优化/验证")
    if dataset.get('mode') in {'workflow','workflow_truth'}:
        from TCM_Fusion.lib.offline.workflow_prompt_validation import validate_workflow
        return await validate_workflow(candidate,dataset,allow_network=allow_network,planner_factory=planner_factory)
    if dataset.get('mode')=='truth':
        from TCM_Fusion.lib.offline.truth_prompt_validation import validate_truth
        return await validate_truth(candidate,dataset,allow_network=allow_network,planner_factory=planner_factory)
    if dataset.get("split") != "development" or not dataset.get("cases"):
        raise IntegrationError("PROMPT_SPLIT", "仅接受非空development回放集")
    if planner_factory is None and not allow_network:
        raise IntegrationError("NETWORK_DISABLED", "真实规划提示对照需允许文本模型调用；不会发图")
    simulation = bool(candidate.get("simulation") or dataset.get("simulation") or planner_factory is not None)
    groups, rows, contexts = set(), [], []
    for case in dataset["cases"]:
        group = case.get("group_id")
        if not group or group in candidate["training_groups"] or group in groups:
            raise IntegrationError("PROMPT_LEAKAGE", "开发样品须唯一且与训练group_id隔离")
        groups.add(group)
        expected = case.get("expected")
        if not isinstance(expected, dict) or set(expected) != {"verdict", "conclusion_id"}:
            raise IntegrationError("REPLAY_TRUTH", "回放须提供独立参考裁决与结论")
        results = []
        for instruction in (candidate["baseline"], candidate["instruction"]):
            runtime = EvidenceRuntime(case["records"], case["images"], sample_id=group, a=case.get("a"),
                config_identity=case["config_identity"], simulation=simulation)
            contexts.append(runtime_context(runtime))
            tools = RecordedTools(case["observations"])
            planner = planner_factory(instruction) if planner_factory else LLMPlanner()
            trace = await run_react(runtime, tools, planner_mode="llm", planner=planner,
                allow_network=allow_network, planning_instruction=instruction)
            decision = trace["decision"]
            results.append({"signature": {"verdict": decision["verdict"],
                "conclusion_id": (decision.get("conclusion") or {}).get("id")},
                "requests": trace["requests"], "unknown_actions": tools.misses,
                "degraded": trace["planner_degraded"], "needs_review": trace["needs_review"],
                "mandatory_complete": trace["mandatory_complete"],
                "evidence": runtime.score(trace["observations"])["evidence_digest"]})
        base, new = results
        passed = (not base["unknown_actions"] and not new["unknown_actions"] and not base["degraded"]
            and not new["degraded"] and base["signature"] == new["signature"] == expected
            and base["evidence"] == new["evidence"] and base["needs_review"] == new["needs_review"]
            and base["mandatory_complete"] == new["mandatory_complete"] and new["requests"] <= base["requests"])
        rows.append({"group_id": group, "baseline": base, "candidate": new, "passed": passed})
    return {"schema": "prompt-validation/v1", "candidate_sha256": digest(candidate),
        "dataset_sha256": digest(dataset), "dataset": dataset, "contexts": contexts,
        "simulation": simulation, "rows": rows,
        "passed": all(r["passed"] for r in rows) and any(
            r["candidate"]["requests"] < r["baseline"]["requests"] for r in rows),
        "metric": "paired_request_attempts_with_equal_evidence_and_verdict",
        "real_accuracy_validated": False}


async def publish(candidate, report, *, allow_network=False, planner_factory=None):
    if report.get("candidate_sha256") != digest(candidate) or not report.get("passed"):
        raise IntegrationError("PROMPT_RELEASE", "候选未通过绑定的回放验证")
    repeated = await validate(candidate, report["dataset"], allow_network=allow_network, planner_factory=planner_factory)
    if not repeated["passed"] or repeated["dataset_sha256"] != report["dataset_sha256"]:
        raise IntegrationError("PROMPT_RELEASE", "发布重新回放失败")
    value = {"schema": "prompt-release/v1", "version": uuid.uuid4().hex,
             "candidate": candidate, "validation": repeated, "simulation": repeated["simulation"]}
    value["sha256"] = digest(value)
    return save("versions", value)
