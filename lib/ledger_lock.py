"""标准库OS文件锁：序列化同账本的跨实例/进程观察；崩溃由OS释放，pending不重发。"""
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
