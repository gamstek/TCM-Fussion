"""显式模拟上下文的视觉响应后端。接收录制回复或确定性替身，供模拟工具箱和 MCP 协议测试使用。
由「lib/tools/agent_tools.py」调用，结果按调用方的任务阶段消费。

主要定义：
- FixtureBackend。

关联：
- 调用方 「lib/tools/agent_tools.py」。
"""
import copy


class FixtureBackend:
    def __init__(self, replies=None):
        self.replies = replies or {}
        self.calls = []

    def call(self, runtime, action, path):
        self.calls.append(action["action_id"])
        key = action["args"].get("feature_id", "scene_check")
        value = self.replies.get(key)
        if value is None and action["tool"] == "scene_check":
            value = {"ok": True, "conf": 0.9, "cell_wall_visible": False,
                     "crystal_location": "free", "crystal_location_raw": "free", "guarded": False,
                     "arrangement": "scattered", "abundance": "many", "reason": "模拟场景观察"}
        if value is None:
            value = {"ok": True, "present": key == "hypha", "conf": 0.9, "reason": "模拟观察"}
        if isinstance(value, Exception):
            raise value
        return copy.deepcopy(value), None
