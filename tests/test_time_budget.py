"""十分钟时间预算：schema、示例配置与客户端上限。"""
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
        path = Path(paths.PROJECT_ROOT) / "TCM_Fusion/config/five_view.json"
        workflow = read_json(path)["agent"]["workflow"]
        self.assertEqual(workflow["max_seconds"], 1800)
        self.assertEqual(workflow["max_observation_seconds"], 150)
        self.assertEqual(workflow["max_call_seconds"], 180)
        validate_arguments(WORKFLOW_SCHEMA, workflow)


if __name__ == "__main__":
    unittest.main()
