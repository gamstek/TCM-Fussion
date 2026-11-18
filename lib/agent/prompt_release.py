"""验证现有提示发布工件。校验格式、摘要、模型档案与运行时上下文，返回可用于规划的版本。
由「lib/agent/react_graph.py」、「lib/offline/prompt_optimization.py」、「lib/offline/truth_prompt_validation.py」调用，结果按调用方的任务阶段消费。

主要定义：
- profile。
- check_candidate。
- runtime_context。
- load_release。

关联：
- 调用方 「lib/agent/react_graph.py」、「lib/offline/prompt_optimization.py」、「lib/offline/truth_prompt_validation.py」、「lib/offline/workflow_replay.py」。
- 直接依赖 「lib/agent/prompt_defaults.py」、「lib/runtime/storage.py」。
"""
from pathlib import Path
import paths
from TCM_Fusion.lib.runtime.storage import IntegrationError, digest, file_hash, read_json
from TCM_Fusion.lib.agent.prompt_defaults import DEFAULT_INSTRUCTION, GUARD


def profile():
    path = Path(paths.DETECTION_CONFIG_DIR)/"vlm_config.json"
    return file_hash(path) if path.is_file() else "unconfigured"


def check_candidate(candidate):
    if (candidate.get("schema") != "prompt-candidate/v1" or candidate.get("baseline") != DEFAULT_INSTRUCTION
            or candidate.get("guard_sha256") != digest(GUARD) or not isinstance(candidate.get("instruction"), str)
            or not 0 < len(candidate["instruction"]) <= 6000):
        raise IntegrationError("PROMPT_CANDIDATE", "候选提示或固定守卫不合法")


def runtime_context(runtime):
    context = {"kb": runtime.kb_sha256, "skills": runtime.skills.list(),
               "tools": runtime.tools.signature(), "candidate_policy": runtime.candidate_policy,
               "mapping": digest(runtime.aliases)}
    workflow = getattr(runtime, "workflow", None)
    if workflow is not None:
        context["workflow_config"] = digest(workflow.config)
        context["workflow_code"] = workflow.code_signature
    return context


def load_release(path, runtime):
    value = read_json(path)
    if value.get("schema") != "prompt-release/v1" or value.get("sha256") != digest(
            {k:v for k,v in value.items() if k != "sha256"}):
        raise IntegrationError("PROMPT_RELEASE", "提示版本校验失败")
    candidate, report = value["candidate"], value["validation"]
    check_candidate(candidate)
    if (not report.get("passed") or report["candidate_sha256"] != digest(candidate)
            or candidate["profile"] != profile() or (value["simulation"] and not runtime.simulation)):
        raise IntegrationError("PROMPT_SCOPE", "版本不匹配或模拟提示禁止进入真实识别")
    if any(c != runtime_context(runtime) for c in report["contexts"]):
        raise IntegrationError("PROMPT_CONTEXT", "知识库/技能改变，须重新验证")
    return value
