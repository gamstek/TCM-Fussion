# -*- coding: utf-8 -*-
"""代码签名冻结检查的离线回归：签名漂移即中止，不联网、不执行真实阶段。
该组用例由项目根 selftest 发现。涉及模型回复的用例使用可控输入，涉及进程的用例以各断言规定的执行边界为准。

主要定义：
- CodeSignatureGuardTest: 覆盖未变化时记录检查次数、变化时报 CODE_CHANGED、watchdog 置取消事件与显式基线拒绝漂移。
- CodeSignatureGuardTest.test_pipeline_aborts_before_agent_on_signature_change: 流水线在启动 Agent 前中止并写失败 job 与起止签名。
- _FakeSignature: 可切换返回值的签名替身，模拟代码被修改。

关联：
- 直接依赖 「lib/application/pipeline.py」、「lib/runtime/code_guard.py」、「lib/runtime/storage.py」、「lib/workflow/workflow_backend.py」。
- 统一执行见 tests/README.md。
"""
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from TCM_Fusion.lib.runtime.code_guard import CodeSignatureGuard
from TCM_Fusion.lib.application.pipeline import run_pipeline
from TCM_Fusion.lib.runtime.storage import IntegrationError, read_json
from TCM_Fusion.lib.workflow.workflow_backend import WorkflowBackend


class _FakeSignature:
    def __init__(self, value="sig-a"):
        self.value = value

    def __call__(self):
        return self.value


class CodeSignatureGuardTest(unittest.TestCase):
    def test_unchanged_records_checks(self):
        with tempfile.TemporaryDirectory() as root:
            fake = _FakeSignature()
            guard = CodeSignatureGuard(Path(root) / "signature.json", fake, interval=1)
            self.assertEqual(guard.check("start"), "sig-a")
            snapshot = guard.finish()
            self.assertFalse(snapshot["changed"])
            self.assertEqual(snapshot["checks"], 2)
            doc = read_json(Path(root) / "signature.json")
            self.assertFalse(doc["changed"])
            self.assertEqual(doc["phase"], "finish")

    def test_change_raises_and_is_recorded(self):
        with tempfile.TemporaryDirectory() as root:
            fake = _FakeSignature()
            guard = CodeSignatureGuard(Path(root) / "signature.json", fake, interval=1)
            fake.value = "sig-b"
            with self.assertRaises(IntegrationError) as ctx:
                guard.check("after_edit")
            self.assertEqual(ctx.exception.code, "CODE_CHANGED")
            snapshot = guard.finish()
            self.assertTrue(snapshot["changed"])
            self.assertEqual(snapshot["changed_at"], "after_edit")
            doc = read_json(Path(root) / "signature.json")
            self.assertTrue(doc["changed"])

    def test_watchdog_sets_cancel_event(self):
        with tempfile.TemporaryDirectory() as root:
            fake = _FakeSignature()
            guard = CodeSignatureGuard(Path(root) / "signature.json", fake, interval=0.02)
            cancel = threading.Event()
            guard.start(cancel)
            fake.value = "sig-b"
            self.assertTrue(cancel.wait(2.0), "watchdog did not set cancel_event")
            guard.stop()
            self.assertTrue(guard.changed)

    def test_explicit_baseline_rejects_immediate_drift(self):
        with tempfile.TemporaryDirectory() as root:
            fake = _FakeSignature("sig-b")
            guard = CodeSignatureGuard(Path(root) / "signature.json", fake,
                                       baseline="sig-a", interval=1)
            with self.assertRaises(IntegrationError):
                guard.check("workflow_start")
            guard.finish()

    def test_watchdog_cancels_when_signature_cannot_be_read(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "signature.json"
            cancel = threading.Event()
            with patch.object(_FakeSignature, "__call__", side_effect=FileNotFoundError):
                guard = CodeSignatureGuard(path, _FakeSignature(), baseline="sig-a", interval=0.02)
                try:
                    guard.start(cancel)
                    self.assertTrue(cancel.wait(2.0))
                    self.assertEqual(read_json(path)["read_error"], "FileNotFoundError")
                finally:
                    guard.stop()

    def test_pipeline_aborts_before_agent_on_signature_change(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            image = root / "fov.png"
            image.write_bytes(b"explicit-offline-test")
            calls = {"n": 0}

            def fake_signature():
                calls["n"] += 1
                return "sig-a" if calls["n"] == 1 else "sig-b"

            config = {
                "sample_id": "sig_guard_case",
                "allow_network": False,
                "a": {"mode": "unavailable", "required": False},
                "b": {"mode": "run", "input_mode": "images", "input": str(image),
                      "enable_vlm": False},
                "agent": {"mode": "active", "planner": "heuristic", "transport": "local",
                          "image_roots": [str(image)], "max_steps": 2,
                          "candidate_policy": {"top_k": 3},
                          "workflow": {"execution": "active",
                                       "python": ".venv/vision/Scripts/python.exe",
                                       "max_seconds": 30}},
            }
            with patch.object(WorkflowBackend, "code_signature", side_effect=fake_signature):
                with self.assertRaises(IntegrationError) as ctx:
                    run_pipeline(config, base=root / "case")
            self.assertEqual(ctx.exception.code, "CODE_CHANGED")
            run = next((root / "case" / "runs").iterdir())
            job = read_json(run / "job.json")
            self.assertEqual(job["status"], "failed")
            self.assertTrue(job.get("code_changed"))
            self.assertTrue((run / "code_signature_start.json").is_file())
            self.assertTrue((run / "code_signature_end.json").is_file())


if __name__ == "__main__":
    unittest.main()
