# 09｜DAG 与旧“短计划”的边界

同步版本 v1.0｜2026-11-17

本文只解决一个问题：

> 现在引入 DAG / 局部执行图时，如何避免重新落回已经删除、且已被证明不可行的“短计划”。

---

## 1. 先给结论

```text
旧短计划 = 模型在观察前预生成具体工具调用序列
新 DAG   = 运行时按当前证据编译的语义任务依赖图
```

两者形似，但不是一回事。

如果新 DAG 由模型一次性写出：

```text
segment crop A
→ classify crop A
→ measure seg A
```

那它就是旧短计划的变体，不应重新引入。

---

## 2. 旧短计划是什么

旧设计大致是：

```text
模型看到当前上下文
→ 一次生成多个 supporting probe / 工具步骤
→ 执行器按计划连续执行
→ 被拒绝或证据变化后再回退
```

典型问题：

```text
计划在证据之前生成
target 在生成时写死
候选名进入计划
多个 probe 共享同一计划
执行时依赖/身份/预算已经变化
```

实际失败码包括：

```text
PLAN_SCOPE
PLAN_PROBE_DENIED
PLAN_NO_LONGER_LEGAL
```

最终删除原因不是“计划不够长”，而是：

```text
模型预规划的具体工具动作
跟不上运行时证据变化
```

---

## 3. 新 DAG 是什么

新 DAG 只承载“语义任务”和“依赖”：

```text
确认实例身份
→ 获取可用 mask
→ 检查测量用途资格
→ 测量
→ 更新判据
```

具体 target 不在计划生成时绑定，  
由运行时在执行前解析：

```text
instance_id
→ 当前合法 crop
→ 当前合法 segmentation
→ 当前合法 measurement
```

---

## 4. 核心区别

| 维度 | 旧短计划 | 新 DAG |
|---|---|---|
| 生成者 | LLM 直接生成工具序列 | LLM 可生成语义目标/补丁；运行时编译执行节点 |
| 生成时机 | 观察之前 | 基于当前证据；执行前可重新解析 |
| 节点内容 | 具体工具 + target | 语义任务 + 条件 |
| target | 写死 crop/seg id | 执行时按实例和版本解析 |
| 候选绑定 | 候选名进入计划 | 候选只消费结果，不进入物理请求身份 |
| 失效处理 | 回退/拒绝旧计划 | 节点 `invalid_if`，立即停止或局部重规划 |
| 证据变化 | 计划容易过期 | 依赖版本变化只影响相关节点 |
| 并行 | 计划顺序执行 | 只并行独立工具，不并行提交证据 |
| 模型调用 | 计划一次，执行整段 | 只在关键事件/需要重规划时调用 |
| 停止 | 计划结束才收尾 | 事件驱动，可提前 review/final |
| 单写者 | 容易竞争 | 证据头只由 Single Writer 提交 |

---

## 5. 允许的 DAG

```text
1. 节点是语义任务，不是具体工具调用
2. target 在执行前解析
3. 每个节点有前置条件和失效条件
4. 关键证据变化时节点可失效
5. 可随时取消未开始节点
6. 不自动取消候选组
7. 不增加每 observation 一次 LLM
8. 不并行提交证据
```

示例：

```json
{
  "node_id": "measure_width",
  "semantic_task": "measure_width",
  "depends_on": ["identity", "mask", "eligibility"],
  "preconditions": [
    "identity.verified",
    "mask.quality=valid",
    "eligibility.width=true"
  ],
  "invalid_if": [
    "identity.rejected",
    "mask.review",
    "evidence_version changed"
  ]
}
```

---

## 6. 不允许的 DAG

```text
1. 模型一次生成 3–5 个具体工具调用
2. 节点写死 crop/seg target
3. 节点跨身份门预先规划 segment/measure
4. 计划生成后不重新检查合法性
5. 失败后继续执行旧计划的后续节点
6. DAG 完成后才允许 ReAct 决策
7. 用 DAG 完成自动替代 review/finish 判断
8. 因某个 GOAL_SATISFIED 自动取消候选组
9. 并行写 evidence head
10. 为每个节点单独调用 LLM
```

---

## 7. 停止与取消

```text
全局 stop：
  ReAct/LLM 决定 continue / review / finish
  运行时只校验 finish 资格
  没有合法动作时自动 review

候选组分支取消：
  ReAct/LLM 决定
  不能因 top1 required 满足自动取消

精确重复动作：
  运行时可以自动剪枝
  例如同一实例同一用途同一输入

节点失效：
  运行时可以自动 invalidate
  但不等同于取消竞争候选
```

---

## 8. 何时才值得引入 DAG

只有满足：

```text
1. 确实存在独立分支可并行
2. 确实存在可预见的局部依赖链
3. 运行时可解析 target
4. 有明确失效条件
5. 能减少实际 planning 请求
6. 不增加额外模型调用
7. A/B 证明优于单步 ReAct
```

否则：

```text
DAG 只是旧短计划的换名
```

---

## 9. 代码审查清单

看到“新增 DAG”时，逐条检查：

```text
[ ] 节点是否是语义任务，而不是固定工具调用？
[ ] target 是否在执行前解析？
[ ] 是否有 invalid_if？
[ ] 证据版本变化时是否只影响依赖节点？
[ ] 是否仍由 ReAct 决定调查目标和停止？
[ ] 候选组是否不会因局部目标满足被取消？
[ ] 是否只有 Single Writer 提交证据？
[ ] 是否避免每个 observation 调用 LLM？
[ ] 是否避免把候选名写进视觉请求？
[ ] 是否有真实 A/B，而不是只看代码结构？
```

---

## 10. 一句话

```text
旧短计划：
  模型在证据之前写死工具动作。

新 DAG：
  模型决定调查目标，
  运行时按当前证据编译语义节点并执行。

如果 DAG 由模型预先生成具体工具序列，
它就还是旧短计划，不应重新引入。
```
