"""根据固定命令、预算上下文和日志路径启动阶段子进程。轮询 HTTP 账本、取消和截止，结束时记录进程耗时并清理子进程。受控启动钩子将 worker 内的 HTTP 请求纳入同一预算。

主要定义：
- safe_log: 日志上下文管理器，落盘后把环境与配置中的凭据替换为 [REDACTED]。
- kill_tree: 有界整树终止；Windows 用 taskkill /T /F，未退出立即 kill/reap，失败则显式报错。
- run: 启动 worker，写入 command.json，轮询 HTTP 失败/超时/取消并返回失败码。

关联：
- 调用方 「lib/runtime/hard_calls.py」、「lib/workflow/workflow_backend.py」、「tests/integration/test_agent_efficiency.py」、「tests/runtime/test_service_failures.py」、「tests/vision/test_presence_localization.py」、「tests/workflow/test_workflow.py」。
- 直接依赖 「lib/runtime/resource_budget.py」、「lib/runtime/storage.py」。
"""
import os
import signal
import subprocess
import time
from pathlib import Path
import paths
from TCM_Fusion.lib.runtime.storage import IntegrationError
from contextlib import contextmanager


@contextmanager
def safe_log(path, env):
    try:
        with Path(path).open("w",encoding="utf-8") as stream:yield stream
    finally:
        if Path(path).exists():
            text=Path(path).read_text("utf-8",errors="replace")
            for key,value in env.items():
                if len(value)>=8 and any(word in key.upper() for word in ("API_KEY","TOKEN","SECRET")):
                    text=text.replace(value,"[REDACTED]")
            config=Path(paths.DETECTION_CONFIG_DIR)/'vlm_config.json'
            if config.is_file():
                from TCM_Fusion.lib.runtime.storage import read_json
                try:
                    cfg=read_json(config)
                    for section in ('vlm','igm','labelstudio'):
                        for key in ('api_key','token'):
                            value=cfg.get(section,{}).get(key)
                            if isinstance(value,str) and value:text=text.replace(value,'[REDACTED]')
                except (ValueError,OSError,AttributeError):
                    text='[LOG WITHHELD: credential config could not be read]'
            Path(path).write_text(text,"utf-8")


def kill_tree(p):
    """Bound tree termination, then immediately kill/reap a surviving worker.

Windows taskkill can return a failure without stopping the process. Do not
spend the old five-second grace period waiting for a worker still running.
Tree termination remains best effort if the OS denies taskkill; a surviving
parent is an explicit failure rather than a silently successful cleanup.
"""
    if p.poll() is not None:
        return
    if os.name=='nt':
        try:
            subprocess.run(['taskkill','/PID',str(p.pid),'/T','/F'],capture_output=True,timeout=1.5)
        except (subprocess.TimeoutExpired,OSError):
            pass
    else:
        try:os.killpg(p.pid,signal.SIGKILL)
        except ProcessLookupError:pass
    if p.poll() is None:
        try:p.kill()
        except OSError:
            if p.poll() is None:raise
    try:p.wait(timeout=1)
    except subprocess.TimeoutExpired as exc:
        raise IntegrationError('PROCESS_TERMINATION_FAILED','worker未在清理时限内退出') from exc


def run(argv, *, context_path, timeout, log_path, cancel_event=None):
    if not isinstance(context_path, (str, os.PathLike)):
        raise IntegrationError('BUDGET_CONTEXT_INVALID', 'worker预算上下文必须是文件路径')
    hook = Path(paths.PROJECT_ROOT) / 'TCM_Fusion/lib/runtime/worker_bootstrap'
    if not (hook / 'sitecustomize.py').is_file():
        raise IntegrationError('WORKER_BOOTSTRAP_MISSING', '受控子进程缺少预算启动钩子')
    env=dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONUNBUFFERED='1', FUSION_BUDGET_CONTEXT=str(context_path),
             PYTHONPATH=os.pathsep.join([str(hook),str(paths.PROJECT_ROOT),*map(str,paths.MODULE_DIRS)]))
    Path(log_path).parent.mkdir(parents=True,exist_ok=True)
    from TCM_Fusion.lib.runtime.storage import write_json
    from TCM_Fusion.lib.runtime.resource_budget import operation_failure
    def check_http_failure():
        try:
            failure=operation_failure(context_path)
        except (OSError, TypeError, KeyError, ValueError) as exc:
            raise IntegrationError('BUDGET_CONTEXT_INVALID', 'worker预算上下文不可读取或字段不完整') from exc
        if failure:
            status=failure.get('http_status')
            code='UPSTREAM_HTTP_'+str(status) if isinstance(status,int) else 'UPSTREAM_NETWORK_FAILED'
            raise IntegrationError(code,'原B模型请求失败；停止本阶段且不接受空检测产物，详见资源账本')
    check_http_failure()  # 无效上下文在启动子进程前失败，不产生幽灵worker。
    write_json(Path(log_path).parent/'command.json', {'argv':list(map(str,argv)),'timeout_seconds':timeout})
    with safe_log(log_path,env) as stream:
        started = time.monotonic(); started_at = time.time()
        p=subprocess.Popen(list(map(str,argv)),cwd=paths.PROJECT_ROOT,env=env,
            stdin=subprocess.DEVNULL,stdout=stream,stderr=stream,start_new_session=os.name!='nt',
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name=='nt' else 0)
        try:
            deadline=time.monotonic()+max(0.1,timeout)
            while p.poll() is None:
                check_http_failure()
                if time.monotonic()>deadline or (cancel_event is not None and cancel_event.is_set()):
                    raise TimeoutError()
                time.sleep(.05)
            code=p.returncode
            check_http_failure()
        except BaseException as exc:
            kill_tree(p)
            if isinstance(exc,IntegrationError):raise
            if isinstance(exc, TimeoutError):
                raise IntegrationError('WORKFLOW_TIMEOUT','任务超时或取消；进程树已终止，不自动重发') from exc
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            raise IntegrationError('WORKER_CONTROL_FAILED',
                                   '本地worker监管异常：'+type(exc).__name__+'；已清理，不自动重发') from exc
        finally:
            write_json(Path(log_path).parent/'process_timing.json', {
                'started_at':started_at,'ended_at':time.time(),
                'elapsed_seconds':round(time.monotonic()-started,6),
                'returncode':p.poll(),'scope':'subprocess including local startup and cleanup'})
    if code:raise IntegrationError('WORKER_FAILED',f'受控阶段退出码{code}；日志仅保留本地')
