"""主动Agent接入：独立B基线测量→LangGraph工具闭环→另存新裁决，原工件不可变。"""
from __future__ import annotations

from pathlib import Path

from .agent_tools import LocalToolbox
from .evidence_runtime import from_b_artifacts
from .mcp_bridge import connect_stdio
from .react_graph import run_react
from .storage import IntegrationError, write_json, project_path


async def execute_active(b, a, agent_config, *, sample_id, directory, aliases=None, allow_network=False):
    if not allow_network:
        raise IntegrationError("NETWORK_DISABLED", "主动模式真实规划和观察需要显式启用网络")
    runtime = from_b_artifacts(b, agent_config, sample_id=sample_id, a=a, aliases=aliases)
    transport = agent_config.get("transport", "local")
    options = {"max_requests": agent_config.get("judgement_request_budget", 6),
               "max_steps": agent_config.get("max_steps", 6),
               "planner_mode": agent_config.get("planner", "llm"),
               "allow_network": allow_network,
               "trace_path": Path(directory)/"agent_trace.json"}
    if agent_config.get("prompt_release"):
        options["prompt_release"] = project_path(agent_config["prompt_release"])
    if transport == "mcp":
        context_path = Path(directory)/"mcp_context.json"
        write_json(context_path, {"sample_id": sample_id, "b_snapshot": str(Path(directory)/"b_snapshot.json"),
                                 "a_prediction": str(Path(directory)/"a_prediction.json"),
                                 "agent": agent_config, "aliases": aliases or {}, "allow_network": allow_network,
                                 "max_requests": options["max_requests"], "simulation": False})
        async with connect_stdio(context_path) as toolbox:
            result = await run_react(runtime, toolbox, **options)
    elif transport == "local":
        toolbox = LocalToolbox(runtime, ledger_path=Path(directory)/"agent_tool_ledger.json",
                               max_requests=options["max_requests"], allow_network=allow_network)
        result = await run_react(runtime, toolbox, **options)
    else:
        raise IntegrationError("CONFIG_INVALID", "transport须为local/mcp")
    # 新裁决在集成工件中另存；原B verdict与测量不覆盖。
    import contracts
    verdict = {"sample_id": sample_id, **result["decision"], "source_images": b["verdict"].get("source_images", []),
               "agent": {"mode": "langgraph_react", "vlm_calls": result["observation_requests"],
                         "planning_calls": result["planning_requests"]},
               "needs_review": result["needs_review"] or result["decision"].get("needs_review", False)}
    contracts.Verdict.from_obj(verdict)
    write_json(Path(directory)/"agent_verdict.json", verdict)
    result["verdict"] = verdict
    return result
