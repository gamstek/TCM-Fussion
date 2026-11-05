# evidence_stop

工具：finish、request_review。强制场景检查未完成时不得finish。
adaptive_v2：按原规则证据和未完成义务给出停止建议；逐折模式不以平均榜首决定停止。覆盖建议填 override_reason。步数、预算、时间和合法动作受运行时硬约束。无收益可触发任务反思及复核建议。
最后裁决来自原decide，不来自LLM的结论文字。冲突、强制检查缺失、未知观察和工具失败都保留复核标记。
日志仅写简短decision_summary、action、observation、规则快照与请求统计；不要求或保存模型内部思维链。
