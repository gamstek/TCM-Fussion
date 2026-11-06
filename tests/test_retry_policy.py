# -*- coding: utf-8 -*-
"""工具级有界重试策略的离线回归。"""
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

    def test_initial_detection_and_roi_screen_have_one_service_retry(self):
        for tool in ('detect_fov_initial', 'roi_screen'):
            with self.subTest(tool=tool), tempfile.TemporaryDirectory() as root:
                first = _runtime(root, {'1': _failed(tool)})
                self.assertNotIn(tool, WorkflowSession.blocked_tools(first))
                twice = _runtime(root, {'1': _failed(tool), '2': _failed(tool)})
                self.assertIn(tool, WorkflowSession.blocked_tools(twice))


if __name__ == "__main__":
    unittest.main()
