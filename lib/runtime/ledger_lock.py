"""标准库实现的 OS 文件锁：串行化同一账本上的跨实例/跨进程观察写入。

锁在进程崩溃时由操作系统自动释放；超时只报 LEDGER_BUSY，不自动重试模型请求，
以免在状态不明时重复发出计费调用。

主要定义：
- acquire: 非阻塞尝试获取账本锁文件句柄，超时后抛错并关闭句柄。
- release: 解锁并关闭锁文件句柄。

关联：
- 调用方 「lib/decision/decision_llm.py」、「lib/evidence/evidence_store.py」、「lib/mcp/mcp_bridge.py」、「lib/runtime/resource_budget.py」、「lib/tools/agent_tools.py」、「lib/workflow/workflow_session.py」。
- 直接依赖 「lib/runtime/storage.py」。
"""
import os
import time
from pathlib import Path

from TCM_Fusion.lib.runtime.storage import IntegrationError


def acquire(ledger_path, timeout=600, *, cancel_event=None):
    path = Path(str(ledger_path)+".lock")
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+b")
    if path.stat().st_size == 0:
        handle.write(b"0")
        handle.flush()
    start = time.monotonic()
    while True:
        if cancel_event is not None and cancel_event.is_set():
            handle.close()
            raise IntegrationError('WORKFLOW_CANCELLED', '等待文件锁时任务已取消')
        try:
            handle.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return handle
        except OSError:
            if time.monotonic()-start >= timeout:
                handle.close()
                raise IntegrationError("LEDGER_BUSY", "观察账本锁超时；未自动重试模型请求")
            time.sleep(.05)


def release(handle):
    try:
        handle.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    finally:
        handle.close()
