# evidence_stop

工具：`finish`、`request_review`。强制场景检查未完成时不得 finish。

`adaptive_v2` 按原规则证据给出建议；逐折模式不以平均榜首决定停止。调查优先级和其他候选未查完只是建议。可覆盖建议并说明理由；步数、预算、时间和事实资格仍是硬约束。

`request_review`始终可申请；不具备确认资格却提出finish时，直接转review输出已有证据报告，不反复请求规划。identity_pending只阻止对应对象的segment/measure，不阻止其他合法调查或复核。高分不绕过身份门。

建议停止后仍可继续；若连续两次继续执行均无实际进展，程序以no_progress_after_advice收尾。新证据或可执行能力可重置计数，改文件名、只读切换或假设切换不能单独重置。计数不需要新模型请求。

无收益可触发任务反思及复核建议，但不能用停止码代替诊断。最后裁决来自原 `decide`，不来自 LLM 的结论文字。冲突、强制检查缺失、未知观察和工具失败都保留复核标记。

日志仅写简短 `decision_summary`、action、observation、规则快照与请求统计；不要求或保存模型内部思维链。

CANDIDATE-AUDIT-001补充：调查顺序仍是建议；确认资格增加候选组审查完整性校验。required/supporting未解决或A候选未映射时，不能以Top1完成升级为完整鉴别。模型可随时request_review；缺口完整落盘，不为补格式额外调用模型，也不强制每个特征必须阳性。stop_advisory与stop_decision统一建议；原规则分/判据保持，交付层单独标CANDIDATE_AUDIT_INCOMPLETE。

FAIR-INVESTIGATION-001：候选账本不是逐项付费清单；主解释未取证与竞争支持增加分别说明。lead_evidence_gap只是建议，可按新证据换假设或review。新的decision_readiness_shadow仅审计判定充分性，不替代当前保守确认校验；不能因为影子结果would_be_ready而确认。停止/格式修复不是证据，不用便宜但不区分候选的观察替代真正取证。
