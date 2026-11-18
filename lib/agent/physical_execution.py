"""把 ReAct 明确选中的两个动作交给工作流预约，经 toolbox 并发准备，按回执顺序提交。
普通回执一轮汇总；快支解锁实例操作时允许提前规划。会话退出时先隔离未提交分支。

主要定义：
- PhysicalRunner 维护一次 Agent 调用的工作票据、并发任务和独占协调者锁。

关联：
- react_graph 管理模型授权、步数和反思，physical_schedule 管理预算、工件和持久化。
"""
import asyncio
from itertools import combinations
from TCM_Fusion.lib.runtime.ledger_lock import acquire, release
from TCM_Fusion.lib.workflow import physical_schedule as schedule


class PhysicalRunner:
    def __init__(self, session, toolbox):
        self.session = session
        self.toolbox = toolbox
        self.tasks = {}
        self.lease = acquire(session.root/'physical_coordinator',timeout=0)
        try:
            with schedule.locked(session):
                schedule.fence_locked(session,'coordinator_restart')
        except BaseException:
            release(self.lease)
            raise

    def options(self, actions, targets, budget):
        pairs = []
        for a,b in combinations((a for a in actions if a['tool'] in schedule.POOLS),2):
            tool = a['tool']
            if tool != b['tool'] or budget[schedule.POOLS[tool]] < 2:
                continue
            network = tool == 'classify_target' or not self.session.runtime.simulation
            if budget['total'] < 2 + 2*network or (network and budget['network'] < 2):
                continue
            if targets[a['args']['target_id']]['image_id'] != targets[b['args']['target_id']]['image_id']:
                pairs.append([a['action_id'],b['action_id']])
        return pairs

    def start(self, actions):
        tickets = schedule.reserve_pair(self.session, actions)
        for a,ticket in zip(actions,tickets):
            task = asyncio.create_task(self.toolbox.prepare_stage(*ticket))
            self.tasks[task] = (a,ticket)

    def can_follow(self, action):
        targets = self.session.targets()
        if action['args'].get('target_id') not in targets:
            return False
        pending = [{'epoch':ticket[1], 'image_id':targets[a['args']['target_id']]['image_id']}
                   for a,ticket in self.tasks.values() if a['args']['target_id'] in targets]
        return schedule.allows_single(pending,action['tool'],targets[action['args']['target_id']])

    async def completed(self):
        done,_ = await asyncio.wait(self.tasks,return_when=asyncio.FIRST_COMPLETED,
                                    timeout=self.session.remaining_seconds())
        for task in done:
            action,ticket = self.tasks.pop(task)
            try:
                task.result()
                error = None
            except Exception as exc:
                error = getattr(exc,'code','TOOL_TRANSPORT_FAILED')
            yield action,schedule.publish(self.session,*ticket,transport_error=error)

    async def close(self, reason):
        try:
            with schedule.locked(self.session):
                schedule.fence_locked(self.session,reason)
            for task in self.tasks:
                task.cancel()
            await asyncio.gather(*self.tasks,return_exceptions=True)
            self.tasks.clear()
        finally:
            release(self.lease)
