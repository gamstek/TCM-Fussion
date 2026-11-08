# evidence_stop

工具：`finish`、`request_review`。强制场景检查未完成时不得 finish。

`adaptive_v2` 按原规则证据和未完成义务给出停止建议；逐折模式不以平均榜首决定停止。覆盖软建议时填 `override_reason`；步数、预算、时间和合法动作受运行时硬约束。

当存在未分类的低分 `identity_required` crop 时，`identity_pending` 会清空 stop，先强制 classify 最高 `det_score` crop；classify 尝试后才允许继续或复核。`det_score >= 0.8` 的 `locate_override` 例外。

无收益可触发任务反思及复核建议，但不能用停止码代替诊断。最后裁决来自原 `decide`，不来自 LLM 的结论文字。冲突、强制检查缺失、未知观察和工具失败都保留复核标记。

日志仅写简短 `decision_summary`、action、observation、规则快照与请求统计；不要求或保存模型内部思维链。
