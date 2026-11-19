# evidence_stop

工具：`finish`、`request_review`。强制场景检查未完成时不得 finish。

`adaptive_v2` 按 B 规则分、必需判据核验状态与候选组审查给出停止建议。明确胜出要求 B 分差至少 0.10。支持类数用于证据广度，逐折 A 先验用于调查。可调整调查顺序，步数、预算、时间和证据资格继续约束执行。

`request_review`始终可申请；不具备确认资格却提出finish时，直接转review输出已有证据报告，不反复请求规划。identity_pending只阻止对应对象的segment/measure，不阻止其他合法调查或复核。高分不绕过身份门。

建议停止后仍可继续；若连续两次继续执行均无实际进展，程序以no_progress_after_advice收尾。新证据或可执行能力可重置计数，改文件名、只读切换或假设切换不能单独重置。计数不需要新模型请求。

无收益可触发任务反思及复核建议。最后裁决由 `rule_scoring.decide` 根据已采信证据计算。冲突、强制检查缺失、未知观察和工具失败都保留复核标记。

日志仅写简短 `decision_summary`、action、observation、规则快照与请求统计；不要求或保存模型内部思维链。

候选组审查完整性是确认条件。首选的 required 全齐时可以获得独立 supporting 加分，候选组中未处理的判据或未映射的 A 候选仍需记录。模型可随时 request_review，交付层用 CANDIDATE_AUDIT_INCOMPLETE 标记审查缺口。

FAIR-INVESTIGATION-001：候选账本不是逐项付费清单；主解释未取证与竞争支持增加分别说明。lead_evidence_gap只是建议，可按新证据换假设或review。新的decision_readiness_shadow仅审计判定充分性，不替代当前保守确认校验；不能因为影子结果would_be_ready而确认。停止/格式修复不是证据，不用便宜但不区分候选的观察替代真正取证。
