# 10｜ReAct 主导的证据优化与语义 DAG 实施部署方案

版本 v1.5.1｜2026-11-17｜状态：**D1–D3 已实施；D3见 [14 交付与验收](14_D3交付与验收.md)。调查自主性与质量门以13及LOW-DETAIL-001为准；D4、D5仍待实施，语义执行默认关闭。**

先读 [09 DAG 与旧短计划边界](09_DAG与旧短计划边界.md)。

本稿按评审意见修订：

1. 原v1.3采用方案A自动required测量；本次按新授权调整：模型选择调查顺序/对象，保留已选分割对象的合格自动测量，不再自动挑top required实例。
2. 修复 `pending_review` 首次 GLM 复核被误拦。
3. 增加实例关联歧义、语义任务封闭注册、并发 epoch/幂等、Reflection 后验、S7 go/no-go。
4. 把 20 人日改为**单人 + AI 辅助的一周垂直实施计划**。
5. 一周内完成“可合并代码 + 离线验收 + 单次烟测”；不承诺一周内完成生产级多案例统计验收。

---

## 一、目标、范围与一周边界

### 1.1 目标

保留 ReAct 对调查目标、方向、下一动作和自主停止的控制；减少同一目标重复注册、同一物理问题重复扫描、无输入变化的失败重试和冗余规划。正式结论仍由原规则与证据资格授权。

DAG 的加入必须取得额外收益。若实例共享与上下文优化已经解决问题，而语义 DAG 没有改善质量—成本关系，保留单步 ReAct，不为架构完整性强行启用。

### 1.2 一周交付边界

一周内完成：

```text
S0 基线修复
S1 实例账本 / 语义尝试键
S2 区分问题 / 证据资格投影
S3 紧凑上下文 / Reflection Controller v1
S4 串行语义执行（默认不启用）
S5 单写者 / 有限并行（视野准备）
S6 最小事件队列 / 取消 / 迟到隔离
S7 离线回归、单次烟测、回滚演练
```

一周内**不承诺**：

```text
多案例统计显著性
真实 API 性能提升证明
生产默认切换
全量并发压力测试
跨任务 Reflection Memory 发布
```

### 1.3 当前代码已具备的基础与缺口

| 已确认现状 | 仍需实施/验证 |
|---|---|
| A 每图五折各自 Top-K，默认每折 3 个 | 候选来源完整保留与恢复调查 |
| ToolSpec、受信 handler/precondition/guard、MCP/本地入口 | 语义任务类型、效果、依赖和晚绑定 |
| 证据快照、哈希、父版本 CAS、依赖失效传播 | 单写者协调与 read_set 重验 |
| presence 共享、crop/高 IoU 去重 | 跨分类/分割/测量的统一实例与问题账本 |
| stop_decision、review 出口、最终引用校验 | pending_review 首次 GLM 修复、循环内资格投影 |
| 部分并发、上下文压缩、反思卡 | 语义 DAG、事件合并、可验证反思补丁 |

方案起点的383项不替代本轮新基线；D3及单图修复后最新466项（465通过、1历史case缺失跳过），原B59/21/125通过。  
工具清理后的当前代码必须先重新跑 `selftest`。

---

## 二、当前授权：调查顺序自主、已选对象自动测量

本节旧方案A的自动步骤在D1/D2阶段保留；ADVISORY-001已替换自主路径中的强制调度，详见[13](13_调查建议与硬约束分离.md)。以下列出当前边界。

### 2.1 自动执行的确定性阶段

```text
文件读取 / 哈希 / 输入校验
→ 模型在当前合法集合选择detect/classify/observe/segment等调查动作
→ 选定segment对象并通过原资格后，可自动measure同一对象
→ 场景未完成不能授予正式确认，但可随时结束为review报告
```

配置保持：

```text
auto_prepare_stage = false
auto_required_measurement = false
auto_measure_after_segment = true
```

资料完整性和既定对象的数值计算由程序完成；调查哪个对象、先执行哪个步骤由模型选择。

### 2.2 交回 ReAct 的情况

```text
存在多个候选的 required 取舍
实例关联歧义
候选组比较
新 ROI / 新视野是否值得重开
scan / repair / 放弃候选
stop / review / finish 意图
```

### 2.3 明确写入边界

```text
自动测量只跟随模型已选定的分割对象，不自动挑选top required的新对象。
```

---

## 三、必须先修的 P0 问题

### 3.1 pending_review 首次 GLM 复核

当前 `_classify_attempted` 不能只看：

```text
classification artifact valid
```

正确规则：

```text
CNN batch pending_review
→ 允许一次 classify_target GLM 复核

classify_target 已执行
→ 不再自动重复

GLM 返回 unreadable/other
→ 不再自动重试
```

落点：

```text
workflow_session._classify_attempted
test_stage2 / test_workflow
```

### 3.2 stop / review 权限

```text
LLM 决定 continue / review / finish
Runtime 只校验 finish 资格
review 始终合法
没有合法动作时自动 review
```

### 3.3 精确重复动作

```text
同一实例 + 同一用途 + 同一方法 + 同一输入指纹
→ 运行时可以剪枝
```

竞争候选分支不得自动取消。

---

## 四、总体架构

```text
用户目标 / 验收约束
        ↓
Goal & Constraint Manager（程序）
        ↓
Evidence View
  ├─ Instance Ledger
  ├─ Evidence Store
  ├─ Eligibility Projection
  └─ Comparison Problems
        ↓
ReAct Controller（LLM）
  ├─ 调查目标
  ├─ continue / review / finish
  ├─ 是否重规划
  └─ 是否取消某个竞争分支
        ↓
LLM Planner（按需）
  └─ 语义 DAG / 局部补丁
        ↓
Plan Compiler（程序）
  ├─ instance_ref 晚绑定
  ├─ 合法性 / 前置条件
  ├─ 语义去重 / 预算预约
  └─ invalid_if
        ↓
DAG Scheduler（程序）
  ├─ 依赖
  ├─ any_of / quorum / all_of
  ├─ 节点失效 / 取消
  └─ 单写者提交顺序
        ↓
Async Executor（独立工具并行）
        ↓
Single Writer（证据头提交）
        ↓
Execution Monitor / Event Bus
        ↓
ReAct Controller / Reflection Controller
```

ReAct 不等整个 DAG。它等待：

```text
关键事件
明确依赖节点
any_of / quorum 条件
```

不是：

```text
all_of 全部完成
```

---

## 五、核心数据契约

### 5.1 InstanceLedger

```json
{
  "instance_id": "...",
  "source_image_sha256": "...",
  "source_bbox": [x1, y1, x2, y2],
  "lineage": {
    "parent_crop": "...",
    "parent_instance": "..."
  },
  "crop_versions": [],
  "classification_attempts": [],
  "mask_versions": [],
  "measurements": [],
  "eligibility": {
    "identity": "verified|pending|rejected",
    "mask": "valid|review|invalid",
    "measurement": "available|unavailable"
  }
}
```

尝试 key：

```text
instance_id
+ purpose
+ method/model version
+ input fingerprint
+ repair parameters
```

实例关联：

```text
父 crop
+ 原图坐标
+ IoU / containment / center distance
+ 已有身份
```

规则：

```text
IoU >= 0.85 且源图/特征一致 → 视为同一实例候选
多解 / 低置信 → ambiguous，不默认合并
ambiguous 改变调查含义 → 交 ReAct
不同原图 → 默认不同实例
```

### 5.2 ComparisonProblem

```json
{
  "problem_id": "...",
  "active_candidates": ["甘草", "苦参"],
  "shared_questions": [],
  "difference_questions": [],
  "evidence_refs": [],
  "capability": "supported|missing_prerequisite|insufficient_input|unsupported_tool|kb_indistinguishable|kb_incomplete|budget_blocked",
  "expected_impact": "...",
  "legal_action_ids": []
}
```

规则：

```text
共同物理问题只执行一次
候选分别解释同一观察
不同形态/量纲/范围不强行合并
kb_gap 不自动创造组结论
宽度重叠不宣布区分
```

### 5.3 EvidenceEligibilityProjection

```json
{
  "candidate_id": "甘草",
  "formal_support": [],
  "formal_counter": [],
  "diagnostic_only": [],
  "invalid_dimensions": [],
  "gaps": [],
  "allowed_conclusion": "insufficient_evidence|group_level|probable|confirmed",
  "evidence_version": "..."
}
```

使用规则：

```text
每次有效提交后更新
规划器下一轮可见
final_assessment 使用同一投影
引用绑定 artifact/observation 身份与评估版本
```

---

## 六、语义 DAG 最小契约

```json
{
  "node_id": "width_question",
  "semantic_task": "measure_width",
  "problem_id": "Q1",
  "instance_ref": "I1",
  "depends_on": ["identity", "mask", "eligibility"],
  "required_facts": ["identity_eligible", "width_eligible", "calibration_trusted"],
  "invalid_if": ["bound_input_changed", "qualification_revoked", "task_closed"]
}
```

规则：

```text
semantic_task 是封闭枚举
不接受模型自由文本
instance_ref 执行前解析当前合法工件
required_facts 是受信谓词，不执行模型表达式
semantic_task → tool 的映射受信且可审计
多解且改变含义 → 回交 ReAct
身份/资格不满足 → 不执行后继
invalid_if 只检查实际输入/资格/任务状态
```

不允许：

```text
模型预生成本次调查的具体 tool+target 序列
节点写死未来 crop/seg ID
DAG 完成才允许 ReAct 决策
```

---

## 七、Reflection Controller

### 输入

```text
原调查目标
实际执行与成本
工件 / 资格 / 判据变化
预期未实现的部分
```

### 输出

```json
{
  "reflection_id": "...",
  "scope": "task",
  "failure_type": "no_valid_mask",
  "affected_candidates": ["甘草"],
  "evidence_refs": [],
  "change_condition": "new_crop_has_complete_endpoints",
  "proposed_patch": {
    "avoid_action_keys": [],
    "activate_candidates": [],
    "recommended_exit": "continue|review"
  },
  "expected_gain": "...",
  "stop_condition": "...",
  "evidence_effect": "none"
}
```

触发：

```text
同一语义问题连续失败 2 次
同一 action key 重复
segment 连续 review
候选 gap 长时间不变
no_gain 超过阈值
```

执行：

```text
不修改证据
→ 校验 patch
→ 应用动作守卫 / DAG 补丁
→ 下一状态变化检查是否有效
```

后验：

```text
patch 是否应用
是否减少重复
是否改变结果/结论
无效 → review
```

默认不增加独立模型调用；  
与下一次必要 ReAct 合并。

---

## 八、并发与事件

### 8.1 可并行

```text
不同视野 detect / classify
已授权独立 observation
```

### 8.2 不可并行

```text
segment
measure
证据头提交
候选决策
```

### 8.3 单写者

每个 worker：

```text
独立目录
共享预算预约
返回 read_set、epoch、结果
不直接写 evidence head
```

Single Writer：

```text
校验 read_set
检查 epoch / input fingerprint
串行提交
```

必须记录：

```text
branch_epoch
plan_revision
idempotency_key
unknown billing
late result quarantine
```

### 8.4 事件

```text
NODE_SUCCEEDED
NODE_FAILED_RETRYABLE
NODE_FAILED_TERMINAL
EVIDENCE_COMMITTED
LOCAL_GOAL_SATISFIED
COMPARISON_PENDING
CONFLICT
NEW_INFORMATION
BUDGET_THRESHOLD
LATE_RESULT
```

普通成功自动传播；  
只有关键事件唤醒 ReAct。

---

## 九、一周实施计划

前提：

```text
1 名熟悉项目的工程师
+ AI 辅助编码/测试
+ 已有环境和依赖可用
```

不是承诺日历日期；  
真实 API 调用需额外预算与时间。

### D1：基线与分类/实例

执行记录：[11 D1交付与验收](11_D1交付与验收.md)。只完成本日范围，不代表语义 DAG 或新版 Reflection 已落地。

```text
S0 新基线、pending_review 修复
S1 InstanceLedger 骨架、语义尝试键、pending/done/failed
```

验收：

```text
pending_review 首次 GLM 可达
同实例换名不重复
既有失败路径可复现
```

### D2：区分问题与资格

执行记录：[12 D2交付与验收](12_D2交付与验收.md)。共享查询、资格引用和关联歧义已接入单步 ReAct；未实施 D3–D5。

```text
S2 ComparisonProblem
EvidenceEligibilityProjection
实例关联 ambiguous 规则
```

验收：

```text
共享问题只执行一次
非法支持下一轮可见
kb_gap 不死循环
```

### D3：上下文、Reflection、串行语义执行

执行记录：[14 D3交付与验收](14_D3交付与验收.md)。S3及默认关闭的最小串行S4已接入真实ReAct入口；未实施D4并发/事件，不代表真实API效益已验证。

```text
S3 紧凑上下文 + Reflection Controller v1
S4 串行语义执行器（默认关闭，实验开关）
```

验收：

```text
反思有 patch 和后验
语义节点晚绑定
身份/资格失败不执行后继
默认仍单步 ReAct
```

### D4：单写者与有限并行

```text
S5 单写者/read_set/epoch/幂等
独立视野 detect/classify 并行
S6 最小事件队列、取消、迟到隔离
```

验收：

```text
无冲突结果都提交
重叠写入明确冲突
迟到结果不发布
不并行 evidence head
```

### D5：集成、回归、A/B 烟测

```text
S7 离线回归
同输入单步基线 vs MVP B/C
单次烟测
回滚演练
```

验收：

```text
业务回归通过
质量不劣化
重复 action/planning 请求可量化
能在新任务回到单步路径
```

---

## 十、测试与 Go/No-Go

### 10.1 必测

```text
pending_review → 一次 GLM
同实例换 crop 不重复
新端点/新视野允许重开
共享结果不重复计 support
非法支持提前可见
语义节点晚绑定
身份/资格失败不执行后继
并发无版本冲突
迟到结果隔离
review 不被 coverage veto
```

### 10.2 对照

```text
A：冻结原链
B：S0–S3 单步 ReAct MVP
C：串行语义图
D：有限并行/事件
```

先比较 A vs B；  
只有 B 仍有明确执行瓶颈，  
才比较 C/D。

### 10.3 Go/No-Go

在实验前写定：

```text
质量非劣界
成本/时延可接受改善
重复动作下降
合法证据不丢失
回滚可执行
```

未定标准前，  
只能称“实验可运行”，  
不能称生产验收通过。

---

## 十一、风险与暂不做

### 风险

```text
实例关联误合并
pending future 被错误共享
语义任务被模型发明
并发 read_set 失效
迟到结果污染
Reflection patch 扩权
```

### 暂不做

```text
多实例分布式调度
默认多 Sub-Agent
生产默认切换 semantic_dag
跨任务自动记忆
全量并行压力验收
```

---

## 十二、完成定义

一周结束时必须能：

```text
1. 在新任务默认运行单步 ReAct
2. 受控开启语义图实验
3. 证明 pending_review 修复
4. 证明同实例不重复
5. 证明非法支持提前可见
6. 证明 review 合法
7. 证明单写者无冲突
8. 回滚到单步路径
9. 文档记录新基线、失败、未验证项
```

---

## 十三、变更记录

v1.5｜2026-11-17｜D3紧凑上下文/字段审计、可校验Reflection补丁与后验、默认关闭的串行实例语义目标执行；新详情kind与stdio MCP连通，458项回归（1 skip）和原B通过。D4/D5待实施。

v1.4｜2026-11-17｜按新授权分离调查建议与硬约束；保留无收益继续上限，并把不够确认的停止意图直接转复核，详见13。完整Reflection/DAG仍未实施。

v1.3.2｜2026-11-17｜实施 D2 区分问题、共用资格投影、稳定引用与关联歧义；详情工具、skill、模型菜单及 MCP 转发闭环。实际验收见 12，不代表真实效益已验证。

v1.3.1｜2026-11-17｜实施 D1 首次 GLM 复核、精确实例索引、语义尝试键与自动测量审计；其余阶段保持待实施，验证结论见 11。

```text
v1.3｜2026-11-17
- 采用方案 A：保留自动 required 测量
- 修复 pending_review 首次 GLM 复核
- 增加实例歧义、语义任务封闭注册、epoch/幂等、Reflection 后验
- 20 人日改为单人+AI 的一周垂直计划
- 明确一周只完成代码/离线验收/烟测，不承诺生产统计验收
```

v1.5.1｜2026-11-17｜D3真实单图暴露的自动测量上下文/伪超时、交付候选引用一致性和反思拒绝诊断已修复；466项回归（1 skip）及原B通过。真实复测簇晶尺寸入账、反思2补丁有后验但无增益证明；仍需复核，D4/D5未完成。

### EFF-001实施边界补记（2026-11-17）

D3真实运行后已实施[15 调查效率与反思收益优化](15_调查效率与反思收益优化.md)：稳定语义问题、分轴收益、可核验补丁no-op/退出、判据状态去重、分段计时和模型授权双路presence。它只补D4的一种独立观察执行能力；通用物理任务并行、增量DAG更新、read_set/epoch、事件提前取消与独立D5仍未交付。semantic_execution保持模板false，实验由模型选择目标，不恢复旧工具短计划。最新回归485项（484通过/1历史case缺失跳过），B59/21/125通过；真实效益应另看新任务而非引用历史样本。
