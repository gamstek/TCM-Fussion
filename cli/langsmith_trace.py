"""LangSmith 只读观察入口，从项目根以 python -m TCM_Fusion.cli.langsmith_trace 启动。

主要定义：main 选择一个 case 或 run。缺省预览结构化字段，--follow 跟随落盘，
--upload 明确选择把追踪摘要发送到当前配置的 LangSmith 目的地。

关联：lib.offline.langsmith_trace。观察者单独运行，只读取工件，不启动模型或工具。
"""

import argparse
import json
import os
from pathlib import Path
import sys
import time

import paths
from TCM_Fusion.lib.offline.langsmith_trace import (
    LangSmithPublisher, find_run, load_vocabulary, optional_json, project_trace,
)
from TCM_Fusion.lib.runtime.storage import IntegrationError, file_revision


_POLL_SECONDS = 2
_INTERFACE_FINISHED = {"completed", "done", "failed", "cancelled", "canceled", "prepared"}


def _publisher():
    from langsmith import Client, agent_address
    from langsmith.utils import LangSmithUserError

    local = optional_json(Path(paths.PROJECT_ROOT) / "config" / "langsmith.local.json")
    fields = ("endpoint", "agent_id", "environment", "workspace_id", "api_key")
    if not isinstance(local, dict) or any(not isinstance(local.get(name, ""), str) for name in fields):
        raise IntegrationError("TRACE_CONFIG", "config/langsmith.local.json 的追踪配置字段须为字符串。")
    key = os.environ.get("LANGSMITH_API_KEY", "").strip() or local.get("api_key", "").strip()
    if not key:
        raise IntegrationError("TRACE_KEY_REQUIRED", "请在 config/langsmith.local.json 的 api_key 中配置密钥。")
    project = next((value for name in ("LANGSMITH_PROJECT", "LANGCHAIN_PROJECT", "LANGCHAIN_SESSION")
                    if (value := os.environ.get(name, "").strip())), "")
    agent = os.environ.get("LANGSMITH_AGENT_ID", "").strip()
    environment = os.environ.get("LANGSMITH_AGENT_ENVIRONMENT", "").strip()
    if not (project or agent or environment):
        agent, environment = local.get("agent_id", "").strip(), local.get("environment", "").strip()
    if bool(agent) != bool(environment) or (agent and project):
        raise IntegrationError("TRACE_DESTINATION", "请填写 Project，或同时填写 Agent ID 和环境，并选择其中一种。")
    try:
        address = agent_address(agent, environment) if agent else None
    except LangSmithUserError:
        raise IntegrationError("TRACE_DESTINATION", "请核对 LangSmith 的 Agent 标识符及环境。") from None
    project = (project or "TCM-Fusion") if address is None else None
    endpoint = os.environ.get("LANGSMITH_ENDPOINT") or local.get("endpoint") or None
    workspace = os.environ.get("LANGSMITH_WORKSPACE_ID") or local.get("workspace_id") or None
    client = Client(api_url=endpoint, api_key=key, workspace_id=workspace,
                    auto_batch_tracing=False, tracing_sampling_rate=1,
                    omit_traced_runtime_info=True, timeout_ms=8000)
    destination = (client.api_url, workspace, project, agent, environment.lower())
    return LangSmithPublisher(client, destination, project_name=project, address=address)


def _revision(run):
    case = run.parent.parent if run.parent.name == "runs" else run
    files = [case / "input_proof.json", case / "interface_status.json"]
    files.extend(run / name for name in (
        "job.json", "code_signature_start.json", "code_signature_before_agent.json",
        "a_prediction.json", "agent_trace.json", "output.json", "final_assessment.json",
        "workflow/action_timeline.json", "workflow/operations.json",
    ))
    return tuple(file_revision(path) for path in files)


def _preview(job_id, spans):
    print(f"{job_id} · {len(spans)} 个调用，仅本地预览")
    depth = {None: -1}
    for span in spans:
        depth[span.key] = depth[span.parent] + 1
        state = span.error or ("进行中" if span.end is None else f"{span.end - span.start:.3f} 秒")
        details = {**span.inputs, **span.outputs}
        body = json.dumps(details, ensure_ascii=False, separators=(",", ":"))
        print(f"{'  ' * depth[span.key]}{span.name} · {state} · {body}")


def observe(directory, *, follow, publisher=None, include_decision_summary=False, include_visual_observations=False):
    vocabulary = load_vocabulary(Path(paths.PROJECT_ROOT))
    run = None
    last_revision = None
    last_update = None
    waiting = None
    read_failures = 0
    while True:
        spans = None
        try:
            if run is None:
                run = find_run(directory)
            if run is None:
                if not follow or directory.is_file():
                    raise IntegrationError("TRACE_RUN_MISSING", "请选择样品目录或含 job.json 的运行目录。")
                state = optional_json(directory / "interface_status.json")
                if state.get("status") in _INTERFACE_FINISHED:
                    run = find_run(directory)
                    if run is None:
                        raise IntegrationError("TRACE_RUN_MISSING", "样品准备已结束，目录中没有可观察的运行工件。")
            if run is not None:
                revision = _revision(run)
                if revision != last_revision:
                    job_id, spans = project_trace(run, vocabulary, include_decision_summary=include_decision_summary,
                                                  include_visual_observations=include_visual_observations)
                    last_revision = revision
        except (FileNotFoundError, PermissionError, json.JSONDecodeError) as exc:
            if not follow:
                raise
            read_failures += 1
            if read_failures >= 4:
                raise IntegrationError("TRACE_READ_FAILED",
                    f"连续四次读取运行工件失败（{type(exc).__name__}）。") from None
            if read_failures == 1:
                print("运行工件暂时不可读，等待下次刷新。", flush=True)
            time.sleep(_POLL_SECONDS)
            continue
        read_failures = 0

        if follow and run is None and waiting != "run":
            print("等待样品目录写入 job.json。按 Ctrl+C 停止观察。", flush=True)
            waiting = "run"
        if spans == []:
            if not follow:
                raise IntegrationError("TRACE_START_MISSING", "此任务尚无可用的开始时间工件。")
            if waiting != "start":
                print("已找到运行目录，等待开始时间记录。", flush=True)
                waiting = "start"
        if spans:
            terminal = spans[0].end is not None
            if publisher is None:
                _preview(job_id, spans)
            else:
                try:
                    trace_id = publisher.sync(job_id, spans)
                except IntegrationError:
                    raise
                except Exception as exc:
                    raise IntegrationError("TRACE_UPLOAD_FAILED",
                        f"LangSmith 同步失败（{type(exc).__name__}）。请检查连接、工作区和密钥，再重试同一目录。") from None
                ended = sum(span.end is not None and (terminal or not span.awaiting_feedback) for span in spans)
                update = trace_id, terminal, len(spans), ended
                if update != last_update:
                    status = "已结束" if terminal else f"进行中，{ended} 个调用已结束"
                    print(f"{trace_id} · {status} · {len(spans)} 个调用", flush=True)
                    last_update = update
            if terminal:
                return
        if not follow:
            return
        time.sleep(_POLL_SECONDS)


def main(argv=None):
    parser = argparse.ArgumentParser(description="预览或同步一个融合运行的 LangSmith 追踪")
    parser.add_argument("directory", type=Path, help="样品 case 或 runs/fusion_<id> 目录")
    parser.add_argument("--follow", action="store_true", help="每两秒检查已指定任务的工件更新")
    parser.add_argument("--upload", action="store_true", help="将预览字段发送到已配置的 LangSmith 目的地")
    parser.add_argument("--include-decision-summary", action="store_true", help="附带已有工件中的模型选择简述")
    parser.add_argument("--include-visual-observations", action="store_true", help="附带已记录的视觉委托与实见短答")
    args = parser.parse_args(argv)
    publisher = None
    try:
        directory = args.directory.resolve()
        if directory.is_file() or (not args.follow and not directory.is_dir()):
            raise IntegrationError("TRACE_RUN_MISSING", "请输入可观察的目录路径。")
        publisher = _publisher() if args.upload else None
        observe(directory, follow=args.follow, publisher=publisher,
                include_decision_summary=args.include_decision_summary,
                include_visual_observations=args.include_visual_observations)
    except KeyboardInterrupt:
        return 130
    except (IntegrationError, OSError, ValueError, ImportError) as exc:
        code = exc.code if isinstance(exc, IntegrationError) else type(exc).__name__
        info = str(exc) if isinstance(exc, IntegrationError) else "无法读取工件或初始化追踪配置。"
        print(f"[{code}] {info}", file=sys.stderr)
        return 2
    finally:
        if publisher is not None:
            publisher.client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
