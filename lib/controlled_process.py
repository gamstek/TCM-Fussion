"""固定argv运行原入口；超时/取消杀进程树，预算钩子由paths派生环境注入。"""
import os
import signal
import subprocess
import time
from pathlib import Path
import paths
from .storage import IntegrationError
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
                from .storage import read_json
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
    """受控子进程通用终止原语；取消不代表远端退费。"""
    if os.name=='nt':
        subprocess.run(['taskkill','/PID',str(p.pid),'/T','/F'],capture_output=True)
    else:
        try:os.killpg(p.pid,signal.SIGKILL)
        except ProcessLookupError:pass
    p.wait()


def run(argv, *, context_path, timeout, log_path, cancel_event=None):
    hook=Path(paths.PROJECT_ROOT)/'TCM_Fusion/lib/worker_bootstrap'
    env=dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONUNBUFFERED='1', FUSION_BUDGET_CONTEXT=str(context_path),
             PYTHONPATH=os.pathsep.join([str(hook),str(paths.PROJECT_ROOT),*map(str,paths.MODULE_DIRS)]))
    Path(log_path).parent.mkdir(parents=True,exist_ok=True)
    from .storage import write_json
    from .resource_budget import operation_failure
    def check_http_failure():
        failure=operation_failure(context_path)
        if failure:
            status=failure.get('http_status')
            code='UPSTREAM_HTTP_'+str(status) if isinstance(status,int) else 'UPSTREAM_NETWORK_FAILED'
            raise IntegrationError(code,'原B模型请求失败；停止本阶段且不接受空检测产物，详见资源账本')
    write_json(Path(log_path).parent/'command.json', {'argv':list(map(str,argv)),'timeout_seconds':timeout})
    with safe_log(log_path,env) as stream:
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
            raise IntegrationError('WORKFLOW_TIMEOUT','任务超时或取消；进程树已终止，不自动重发')
    if code:raise IntegrationError('WORKER_FAILED',f'受控阶段退出码{code}；日志仅保留本地')
