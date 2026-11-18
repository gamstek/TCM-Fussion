"""标准库实现的 OS 文件锁：串行化同一账本上的跨实例/跨进程观察写入。

锁在进程崩溃时由操作系统自动释放；超时只报 LEDGER_BUSY，不自动重试模型请求，
以免在状态不明时重复发出计费调用。

主要定义：
- acquire: 非阻塞尝试获取账本锁文件句柄，超时后抛错并关闭句柄。
- release: 解锁并关闭锁文件句柄。

关联：
- .storage: 使用 IntegrationError 表达锁超时错误。
- .evidence_store / .workflow_session / .resource_budget / .agent_tools / .mcp_bridge:
  在提交证据、写账本或预算时调用 acquire/release 互斥。
"""
import os
import time
from pathlib import Path

from .storage import IntegrationError


def acquire(ledger_path, timeout=600):
    path = Path(str(ledger_path)+".lock")
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+b")
    if path.stat().st_size == 0:
        handle.write(b"0")
        handle.flush()
    start = time.monotonic()
    while True:
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
