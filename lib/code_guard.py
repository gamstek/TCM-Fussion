# -*- coding: utf-8 -*-
"""运行期代码签名冻结检查：记录 baseline，供运行前、运行中和结束前复核。

只做签名比对，不读写代码、不修改证据；签名函数由调用方注入（生产用 WorkflowBackend.code_signature）；写盘失败不阻断主流程，但签名变化不能被静默吞掉。

主要定义：
- CodeSignatureGuard: 保存 baseline/current，提供 check、snapshot、start、stop、finish；变化时抛 CODE_CHANGED 或置位 cancel_event。

关联：
- .storage: IntegrationError 与 write_json。
- .workflow_entry: 构造 CodeSignatureGuard 并在各阶段复核。
- .workflow_backend: 提供生产用 code_signature 作为签名函数。
- tests/test_code_guard.py: 回归签名变化检测。
"""
from __future__ import annotations

import threading
import time
from pathlib import Path

from .storage import IntegrationError, write_json


class CodeSignatureGuard:
    """记录 baseline，供运行前、运行中和结束前复核。"""

    def __init__(self, path, signature_fn, *, baseline=None, interval=10.0):
        if not callable(signature_fn):
            raise IntegrationError("CONFIG_INVALID", "签名函数不可调用")
        self.path = Path(path)
        self.signature_fn = signature_fn
        self.interval = max(0.01, float(interval))
        self.baseline = baseline if baseline is not None else signature_fn()
        self.current = self.baseline
        self.changed = False
        self.changed_at = None
        self.checks = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self._cancel_event = None
        self._started = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        self._write("start", None)

    def _write(self, phase, where):
        try:
            write_json(self.path, {
                "schema": "code-signature-guard/v1",
                "baseline": self.baseline,
                "current": self.current,
                "changed": self.changed,
                "changed_at": self.changed_at,
                "checks": self.checks,
                "phase": phase,
                "where": where,
                "started_at": self._started,
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            })
        except OSError:
            # 记录失败不应掩盖真正的签名变化；check() 仍会抛 CODE_CHANGED。
            pass

    def check(self, where="check"):
        """复核签名；变化时抛 CODE_CHANGED，并保留首次变化位置。"""
        with self._lock:
            current = self.signature_fn()
            self.checks += 1
            self.current = current
            if current != self.baseline:
                if not self.changed:
                    self.changed = True
                    self.changed_at = where
                self._write("changed", where)
                raise IntegrationError(
                    "CODE_CHANGED",
                    f"代码签名在运行期间变化（{where}）；已停止采信本轮新产物")
            self._write("check", where)
            return current

    def snapshot(self):
        return {
            "baseline": self.baseline,
            "current": self.current,
            "changed": self.changed,
            "changed_at": self.changed_at,
            "checks": self.checks,
            "started_at": self._started,
            "path": str(self.path),
        }

    def start(self, cancel_event=None):
        """启动周期复核；变化时置位 cancel_event 并结束线程。"""
        if self._thread is not None:
            return
        self._cancel_event = cancel_event

        def loop():
            while not self._stop.wait(self.interval):
                try:
                    self.check("watchdog")
                except IntegrationError:
                    if self._cancel_event is not None:
                        self._cancel_event.set()
                    return

        self._thread = threading.Thread(target=loop, name="code-signature-guard", daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)
        self._thread = None

    def finish(self):
        """结束前再复核一次并停止线程；变化只记录，不覆盖主异常。"""
        try:
            self.check("finish")
        except IntegrationError:
            pass
        finally:
            self.stop()
            self._write("finish", None)
        return self.snapshot()
