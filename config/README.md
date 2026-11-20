# 融合配置

multi_view.json 用于构建新任务。detector.json 记录检测推理覆盖项，负例库和示例库向特征提示及视觉审计提供参考。字段语义和兼容要求见[重组方案](../融合链路文档/21_TCM_Fusion组件重组方案.md)。

## 单视野定量问题

`multi-view` 使用正式模板创建新任务时，默认允许 ReAct 授权单视野定量问题。要做关闭组对照，在同一模板副本中将 `agent.workflow.problem_goal_execution` 设为 `false`，并通过 `--config-template` 传入。旧模板缺少该字段时也采用关闭行为。[导管测宽与回交条件](../融合链路文档/09_DAG与旧短计划边界.md#单视野鉴别问题的授权入口)
