# 自主B工具与A融合

- 裁决：insufficient_evidence
- 交付输出：tentative / 白芍
- 内部复核标记：True；交付层 review_required=True
- 强观察：2 条；备选：4 个
- 停止原因：runtime_request_review
- 证据版本：aaffec2fd5210b9d7132e6565d75f66dda7d3d1dd9224bc0a8c04cbc62dc8208
- 执行终止原因：PLANNER_QUESTION_UNAVAILABLE；类型：runtime
- 交付复核原因：['FORMAL_REQUIRES_REVIEW', 'FORMAL_VERDICT_NOT_AUTHORIZED']（与执行中断原因分开记录）
- 原图及原B目录未覆盖；阶段失败/资源明细见agent_trace.json。

## 收尾综合研判

现有证据不足以给出可靠的药材倾向

依据全部已收集证据及原规则综合排序给出当前倾向。判据状态：无可采信支持。原规则说明：候选池为空或最高分候选无有效支持分（<0.50）。正式判据未满足不等于原图中不存在该目标；已检测或分类但缺少合格尺寸的实例仍待核验。

未选择药材候选，无法给出候选分数。

- 解释来源：deterministic；文本警告：[]
- 研判来源：deterministic；模型失败原因：ASSESSMENT_INVALID
- 支持证据：[]
- 反证：[]
- 未解决：[]
- 全部证据及采信边界：final_evidence.json；结论与依据：final_assessment.json。

| 候选 | B规则分 | 融合排序分（非正确率） | 支持判据数 | 必需判据缺口数 |
|---|---|---|---|---|
| 白芍 | 0.0 | 0.472168 | 0 | 1 |
| 郁金 | 0.459 | 0.190423 | 1 | 0 |
| 茯苓 | 0.0 | 0.001309 | 0 | 2 |
| 枳壳 / 陈皮 | 0.0 | 7e-05 | 0 | 1 |
| 赤芍 / 牡丹皮 | 0.0 | 5.7e-05 | 0 | 1 |
