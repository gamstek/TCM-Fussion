# -*- coding: utf-8 -*-
"""工具级有界重试策略的离线回归：验证失败计数只在允许范围内放大为熔断。

主要定义：
- ToolRetryPolicyTest: 覆盖 locate 一次失败不阻断、两次失败阻断，以及非可重试服务错误立即阻断。
- ToolRetryPolicyTest.test_classify_failure_has_one_retry: 分类失败仅保留一次重试。
- ToolRetryPolicyTest.test_initial_detection_and_feature_scan_have_one_service_retry: 初检与 feature_scan 各允许一次服务重试。
- _runtime / _failed: 写入最小 operations 账本并构造失败条目，隔离文件系统。

关联：
- TCM_Fusion.lib.workflow_session: 被测的 blocked_tools 重试判定。
- TCM_Fusion.lib.storage.write_json: 构造临时账本文件。
- 位于 tests/ 下，随 selftest 递归发现。
"""
import tempfile
import unittest
from pathlib import Path

from TCM_Fusion.lib.storage import write_json
from TCM_Fusion.lib.workflow_session import WorkflowSession


class _Runtime:
    pass


def _runtime(root, entries):
    ops = Path(root) / "operations.json"
    write_json(ops, {"entries": entries})
    obj = _Runtime()
    obj.operations = ops
    obj.config = {"max_tool_failures": 2}
    return obj


def _failed(tool):
    return {"tool": tool, "status": "failed", "error_code": "UPSTREAM_NETWORK_FAILED"}


class ToolRetryPolicyTest(unittest.TestCase):
    def test_one_locate_failure_does_not_block_tool(self):
        with tempfile.TemporaryDirectory() as root:
            obj = _runtime(root, {"l1": _failed("locate_feature")})
            self.assertEqual(WorkflowSession.blocked_tools(obj), {})

    def test_two_locate_failures_block_tool(self):
        with tempfile.TemporaryDirectory() as root:
            obj = _runtime(root, {"l1": _failed("locate_feature"),
                                  "l2": _failed("locate_feature")})
            self.assertIn("locate_feature", WorkflowSession.blocked_tools(obj))

    def test_non_retryable_service_failure_blocks_immediately(self):
        with tempfile.TemporaryDirectory() as root:
            obj = _runtime(root, {"s1": _failed("segment_instance")})
            self.assertIn("segment_instance", WorkflowSession.blocked_tools(obj))

    def test_classify_failure_has_one_retry(self):
        with tempfile.TemporaryDirectory() as root:
            obj = _runtime(root, {"c1": _failed("classify_target")})
            self.assertEqual(WorkflowSession.blocked_tools(obj), {})

    def test_initial_detection_and_feature_scan_have_one_service_retry(self):
        for tool in ('detect_fov_initial', 'feature_scan'):
            with self.subTest(tool=tool), tempfile.TemporaryDirectory() as root:
                first = _runtime(root, {'1': _failed(tool)})
                self.assertNotIn(tool, WorkflowSession.blocked_tools(first))
                twice = _runtime(root, {'1': _failed(tool), '2': _failed(tool)})
                self.assertIn(tool, WorkflowSession.blocked_tools(twice))


if __name__ == "__main__":
    unittest.main()
