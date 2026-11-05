"""主动闭环离线演示：真实LangGraph/可选真实MCP协议，VLM/LLM用显式fixture替身。"""
from __future__ import annotations

import uuid
from pathlib import Path

import paths

from .agent_tools import FixtureBackend, LocalToolbox
from .evidence_runtime import EvidenceRuntime
from .agent_evaluation import compare_fixed_fixture
from .mcp_bridge import connect_stdio
from .react_graph import ScriptedPlanner, run_react
from .storage import output_root, read_json, write_json


async def demo_react(*, transport="local", base=None, top_k=None):
    directory = Path(base or output_root())/"runs"/("react_demo_"+uuid.uuid4().hex)
    directory.mkdir(parents=True)
    fixture = read_json(Path(paths.PROJECT_ROOT)/"TCM_Fusion/tests/fixtures/react_case.json")
    # 测试占位图：不解码、不作为真实VLM输入。所有模拟观察来自fixture。
    image = directory/"fov.png"
    image.write_bytes(b"simulation-placeholder-not-a-real-microscope-image")
    runtime = EvidenceRuntime(fixture["records"], [image], sample_id=fixture["sample_id"],
                              a=fixture["a"], simulation=True,
                              candidate_policy={"top_k": top_k} if top_k is not None else None)
    options = {"planner_mode": "scripted", "planner": ScriptedPlanner(),
               "trace_path": directory/"agent_trace.json"}
    if transport == "mcp":
        context_path = directory/"mcp_context.json"
        write_json(context_path, {"sample_id": fixture["sample_id"], "simulation": True,
                                 "config_identity": runtime.config_identity,
                                 "fixture": {**fixture, "images": [str(image)]}, "allow_network": False,
                                 "candidate_policy": runtime.candidate_policy})
        async with connect_stdio(context_path) as tools:
            skills = await tools.call("list_skills")
            write_json(directory/"mcp_skills.json", skills)
            result = await run_react(runtime, tools, **options)
    else:
        tools = LocalToolbox(runtime, ledger_path=directory/"tool_ledger.json",
                            backend=FixtureBackend(fixture["replies"]))
        result = await run_react(runtime, tools, **options)
    write_json(directory/"result.json", result)
    comparison = compare_fixed_fixture(runtime, FixtureBackend(fixture["replies"]), result)
    write_json(directory/"comparison.json", comparison)
    before, after = runtime.original, result["decision"]
    (directory/"report.md").write_text(
        "# 主动Agent闭环演示（模拟观察）\n\n"
        f"- 引擎：LangGraph；工具传输：{transport}\n"
        f"- 原规则top1：{before['ranking'][0]['id']}，裁决：{before['verdict']}\n"
        f"- 补查后top1：{after['ranking'][0]['id']}，裁决：{after['verdict']}\n"
        f"- 规划请求：{result['planning_requests']}；观察请求：{result['observation_requests']}\n"
        f"- 停止：{result['stop_reason']}；强制检查完成：{result['mandatory_complete']}\n\n"
        f"- 原固定策略观察请求：{comparison['fixed']['observation_requests']}；"
        f"ReAct观察＋规划请求：{result['requests']}\n\n"
        "工具响应与规划为测试替身；闭环执行和规则重评分为真实代码，不是准确率提升证据。\n",
        encoding="utf-8")
    return directory
