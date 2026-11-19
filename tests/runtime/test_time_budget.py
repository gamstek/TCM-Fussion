"""工作流时间预算回归：schema 接受十分钟级上限，实时模板保持参考时间预算。
该组用例由项目根 selftest 发现。涉及模型回复的用例使用可控输入，涉及进程的用例以各断言规定的执行边界为准。

主要定义：
- TimeBudgetTest.test_schema_accepts_ten_minute_budget: 校验 WORKFLOW_SCHEMA 接受 600 秒级总时长与单次调用上限。
- TimeBudgetTest.test_live_template_preserves_reference_time_budget: 校验 multi_view.json 的 max_seconds/max_observation_seconds/max_call_seconds 取值。

关联：
- 直接依赖 「lib/application/workflow_entry.py」、「lib/runtime/storage.py」、「lib/tools/tool_registry.py」。
- 统一执行见 tests/README.md。
"""
import json
import unittest
from pathlib import Path

import paths
from TCM_Fusion.lib.runtime.storage import read_json
from TCM_Fusion.lib.tools.tool_registry import validate_arguments
from TCM_Fusion.lib.application.workflow_entry import WORKFLOW_SCHEMA


class TimeBudgetTest(unittest.TestCase):
    def test_schema_accepts_ten_minute_budget(self):
        validate_arguments(WORKFLOW_SCHEMA, {
            "max_seconds": 600,
            "max_observation_seconds": 600,
            "max_call_seconds": 600,
        })

    def test_live_template_preserves_reference_time_budget(self):
        path = Path(paths.PROJECT_ROOT) / "config/multi_view.json"
        workflow = read_json(path)["agent"]["workflow"]
        self.assertEqual(workflow["max_seconds"], 2400)
        self.assertEqual(workflow["max_observation_seconds"], 150)
        self.assertEqual(workflow["max_call_seconds"], 180)
        validate_arguments(WORKFLOW_SCHEMA, workflow)


if __name__ == "__main__":
    unittest.main()
