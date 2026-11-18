"""MCP stdio 服务与真实客户端：工具调用与本地 ReAct 共用同一个 LocalToolbox。

服务一次绑定一个样品/配置；不接受任意路径、shell 或药材结论作为工具参数。

主要定义：
- load_bound_context: 从 context 文件恢复工作流运行时与工具账本。
- make_server: 创建 FastMCP 服务并注册技能、证据、观察与工具读写端点。
- MCPToolbox: 客户端侧包装 call/recompute/read/observe。
- connect_stdio: 以 stdio 子进程方式连接服务，返回可用工具箱。

关联：
- TCM_Fusion.lib.agent_tools: 复用 LocalToolbox 与 FixtureBackend。
- TCM_Fusion.lib.evidence_runtime: 提供证据读写与运行时基类。
- TCM_Fusion.lib.workflow_runtime: from_task 恢复工作流运行时。
- TCM_Fusion.lib.storage: context/账本读取与错误类型。
- TCM_Fusion.lib.workflow_entry: transport=mcp 时调用 connect_stdio。
"""
from __future__ import annotations

import asyncio
import json
import sys
from contextlib import asynccontextmanager
from datetime import timedelta
from pathlib import Path

import paths

from .agent_tools import FixtureBackend, LocalToolbox
from .evidence_runtime import EvidenceRuntime, from_b_artifacts
from .storage import IntegrationError, digest, output_root, project_path, read_json


def load_bound_context(path):
    context = read_json(path)
    if "workflow_config" in context:
        from .workflow_runtime import from_task
        runtime = from_task(context["workflow_config"],read_json(context["a_prediction"]),project_path(context["directory"]))
        backend = None
    elif context.get("simulation") is True:
        fixture = context["fixture"]
        runtime = EvidenceRuntime(fixture["records"], fixture["images"],
                                  sample_id=context["sample_id"], a=fixture.get("a"),
                                  config_identity=context.get("config_identity", digest(fixture)), simulation=True,
                                  candidate_policy=context.get("candidate_policy"))
        backend = FixtureBackend(fixture.get("replies"))
    else:
        b = read_json(project_path(context["b_snapshot"]))
        a = read_json(project_path(context["a_prediction"]))
        runtime = from_b_artifacts(b, context["agent"], sample_id=context["sample_id"], a=a,
                                   aliases=context.get("aliases"))
        backend = None
    ledger = output_root()/"mcp"/digest(context)/"tool_ledger.json"
    toolbox = LocalToolbox(runtime, ledger_path=ledger,
                           max_requests=context.get("max_requests", 6),
                           allow_network=context.get("allow_network", False), backend=backend)
    return runtime, toolbox


def make_server(context_path):
    from mcp.server.fastmcp import FastMCP
    from mcp.types import ToolAnnotations
    runtime, toolbox = load_bound_context(context_path)
    server = FastMCP("microscope-evidence", log_level="ERROR")

    @server.tool(annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False))
    def list_skills() -> dict:
        """列出本样品可用领域技能、工具权限与版本哈希。"""
        return {"skills": runtime.skills.list(), "simulation": runtime.simulation}

    @server.tool(annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False))
    def get_skill(skill_id: str) -> dict:
        """读取已注册技能；不能用文件路径代替skill_id。"""
        return {"skill_id": skill_id, "content": runtime.skills.get(skill_id)}

    @server.tool(annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False))
    def read_evidence() -> dict:
        """只读原测量加本服务已接受观察的规则快照；不接收药材真值。"""
        history = [r for r in toolbox.ledger["entries"].values() if "action" in r]
        return runtime.summary({"observations": history})

    @server.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, openWorldHint=True))
    async def observe(action: dict) -> dict:
        """执行白名单scene/presence观察；会消耗API，受配置、预算和持久去重约束。"""
        try:
            return await toolbox.observe(action)
        except IntegrationError as exc:
            # 保留领域码；只对明确的前置网络禁令声明未发请求，其余计费未知。
            return {'action_id':action.get('action_id'),'accepted':False,'status':'failed',
                    'result':{'ok':False,'error_code':exc.code},'error_code':exc.code,
                    'new_requests':0 if exc.code=='NETWORK_DISABLED' else None}

    @server.tool(annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False))
    def list_tools_registry() -> dict:
        """读取与本地执行器一致的注册表；元数据不能扩权。"""
        return {"tools": runtime.tools.describe(), "sha256": runtime.tools.signature()}

    @server.tool(annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False))
    def read_tool(name: str, arguments: dict) -> dict:
        """通过统一注册表查询证据；只读，不接受客户端注入观察历史。"""
        from .ledger_lock import acquire, release
        handle = acquire(toolbox.ledger_path)
        try:
            ledger = read_json(toolbox.ledger_path) if toolbox.ledger_path.exists() else toolbox.ledger
            if ledger.get("identity") != toolbox.identity:
                raise IntegrationError("LEDGER_MISMATCH", "账本身份不匹配")
            history = [r for r in ledger["entries"].values() if "action" in r]
            return runtime.tools.read(runtime, name, arguments, {"observations": history})
        finally:
            release(handle)

    @server.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, openWorldHint=True))
    async def recompute_tool(action: dict) -> dict:
        """执行已绑定工件的真实阶段；共享预算、幂等与证据版本提交。"""
        try:
            return await toolbox.recompute(action)
        except IntegrationError as exc:
            # FastMCP通用异常文本会丢失领域错误码；不把预算/权限失败说成服务或图像故障。
            return {'action_id':action.get('action_id'),'accepted':False,'status':'failed',
                    'repair_completed':False,'error_code':exc.code}

    @server.resource("microscope://skills")
    def skill_resource() -> str:
        return json.dumps(runtime.skills.list(), ensure_ascii=False)

    @server.prompt()
    def collect_evidence() -> str:
        return "按已注册技能查看证据、选择合法观察；最终裁决交回规则引擎。不得从A预测推造观察或尺寸。"

    return server


class MCPToolbox:
    def __init__(self, session):
        self.session = session

    async def call(self, name, arguments=None):
        value = await self.session.call_tool(name, arguments or {})
        if value.isError:
            raise IntegrationError("MCP_TOOL_ERROR", "MCP工具执行失败")
        structured = getattr(value, "structuredContent", None)
        if structured is not None:
            return structured
        for block in value.content:
            if block.type == "text":
                return json.loads(block.text)
        raise IntegrationError("MCP_OUTPUT_INVALID", "MCP返回缺结构化内容")

    async def recompute(self, action):
        return await self.call("recompute_tool", {"action": action})

    async def read(self, name, args, state):
        return await self.call("read_tool", {"name": name, "arguments": args})

    async def observe(self, action):
        return await self.call("observe", {"action": action})


@asynccontextmanager
async def connect_stdio(context_path):
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    import os
    forwarded = {}
    cfg_path=Path(paths.DETECTION_CONFIG_DIR)/"vlm_config.json"
    names={"VLM_API_KEY","IGM_API_KEY"}
    if cfg_path.is_file():
        cfg=read_json(cfg_path)
        for section in (cfg.get("vlm",{}),cfg.get("igm",{})):
            if isinstance(section.get("api_key_env"),str):names.add(section["api_key_env"])
    for name in names:
        if name in os.environ:forwarded[name]=os.environ[name]
    # MCP默认环境白名单不包含数值库线程限制；否则父进程限流而子worker仍大量分配栈。
    # 仅透传调用方已明确设置的资源环境，不改机器/用户环境或生成新路径权威。
    for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS',
                 'TEMP','TMP'):
        if name in os.environ:
            forwarded[name]=os.environ[name]
    parameters = StdioServerParameters(command=sys.executable,
        args=["-X", "utf8", "-m", "TCM_Fusion.cli.mcp_server", "--context", str(project_path(context_path))],
        cwd=str(paths.PROJECT_ROOT), env=forwarded)
    async with stdio_client(parameters) as (read, write):
        context=read_json(project_path(context_path))
        deadline=context.get('workflow_config',{}).get('agent',{}).get('workflow',{}).get('max_seconds',600)
        async with ClientSession(read, write, read_timeout_seconds=timedelta(seconds=max(360,deadline+10))) as session:
            await session.initialize()
            yield MCPToolbox(session)
