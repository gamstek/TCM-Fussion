"""按动作 ID 和完整动作内容回放已录制的观察。

保存已知动作及其响应，返回零新请求的观察结果。重复录制内容冲突或缺少动作时拒绝回放。

主要定义：
- RecordedTools 维护只读录制索引和回放缺口。

关联：
- prompt_optimization 和 truth_prompt_validation 复用同一观察接口。
- react_graph 将该接口用作离线工具箱。
"""
from TCM_Fusion.lib.runtime.storage import IntegrationError


class RecordedTools:
    def __init__(self, rows):
        self.rows = {}
        self.misses = []
        for row in rows:
            action = row["action"]
            key = action["action_id"]
            if key in self.rows and self.rows[key] != row:
                raise IntegrationError("REPLAY_CONFLICT", "同动作录制了不同响应，拒绝静默覆盖")
            self.rows[key] = row

    async def observe(self, action):
        row = self.rows.get(action["action_id"])
        if row is None or row["action"] != action:
            self.misses.append(action["action_id"])
            raise IntegrationError("REPLAY_UNKNOWN", "未录制动作，回放不能验证")
        return {"action_id": action["action_id"], "accepted": row["accepted"],
                "result": row["result"], "cache_hit": True, "new_requests": 0}
