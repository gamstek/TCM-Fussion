"""工作流时间预算回归：schema 接受十分钟级上限，实时模板保持参考时间预算。

主要定义：
- TimeBudgetTest.test_schema_accepts_ten_minute_budget: 校验 WORKFLOW_SCHEMA 接受 600 秒级总时长与单次调用上限。
- TimeBudgetTest.test_live_template_preserves_reference_time_budget: 校验 multi_view.json 的 max_seconds/max_observation_seconds/max_call_seconds 取值。

关联：
- TCM_Fusion.lib.workflow_entry.WORKFLOW_SCHEMA: 被验证的工作流配置 schema。
- TCM_Fusion.lib.tool_registry.validate_arguments: 执行 schema 校验。
- TCM_Fusion/config/multi_view.json: 实时模板时间预算来源。
- 位于 tests/ 下，随 selftest 递归发现。
"""
import json
import unittest
from pathlib import Path

import paths
from TCM_Fusion.lib.storage import read_json
from TCM_Fusion.lib.tool_registry import validate_arguments
from TCM_Fusion.lib.workflow_entry import WORKFLOW_SCHEMA


class TimeBudgetTest(unittest.TestCase):
    def test_schema_accepts_ten_minute_budget(self):
        validate_arguments(WORKFLOW_SCHEMA, {
            "max_seconds": 600,
            "max_observation_seconds": 600,
            "max_call_seconds": 600,
        })

    def test_live_template_preserves_reference_time_budget(self):
        path = Path(paths.PROJECT_ROOT) / "TCM_Fusion/config/multi_view.json"
        workflow = read_json(path)["agent"]["workflow"]
        self.assertEqual(workflow["max_seconds"], 2400)
        self.assertEqual(workflow["max_observation_seconds"], 150)
        self.assertEqual(workflow["max_call_seconds"], 180)
        validate_arguments(WORKFLOW_SCHEMA, workflow)


if __name__ == "__main__":
    unittest.main()
