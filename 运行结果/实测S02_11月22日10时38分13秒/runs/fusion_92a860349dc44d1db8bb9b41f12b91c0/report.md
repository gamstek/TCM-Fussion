# 自主B工具与A融合

- 裁决：insufficient_evidence
- 交付输出：undetermined / 无
- 内部复核标记：True；交付层 review_required=True
- 强观察：0 条；备选：4 个
- 停止原因：runtime_request_review
- 证据版本：2c9bc91d6f57ec4975039288f90728f456efa3454f316d8d12e9ddffaeb96fcc
- 执行终止原因：UPSTREAM_NETWORK_FAILED；类型：service
- 交付复核原因：['FORMAL_REQUIRES_REVIEW', 'CANDIDATE_AUDIT_INCOMPLETE', 'FORMAL_VERDICT_NOT_AUTHORIZED']（与执行中断原因分开记录）
- 原图及原B目录未覆盖；阶段失败/资源明细见agent_trace.json。

## 裁决与两路排序

正式结论未确认（insufficient_evidence）
融合首位 白芍（排序分 0.488869）
  - 支持：暂无可引用判据
  - 反证：暂无可引用判据
B 证据首位 赤芍 / 牡丹皮（排序分 0.0）
  - 支持：暂无可引用判据
  - 反证：暂无可引用判据

## 收尾综合研判

现有证据不足以给出可靠的药材倾向

依据全部已收集证据及原规则综合排序给出当前倾向。判据状态：无可采信支持。原规则说明：必需判据未全部核验或 B 规则分未达倾向门槛。。正式判据未满足不等于原图中不存在该目标；已检测或分类但缺少合格尺寸的实例仍待核验。

未选择药材候选，无法给出候选分数。

- 解释来源：deterministic；文本警告：[]
- 研判来源：deterministic；模型失败原因：UPSTREAM_NETWORK_FAILED
- 研判引用的合格判据：[]
- 模型原提案逐条引用及剔除原因见 [收尾引用审计](final_assessment.json) 的 model_citation_audit。
- 局部存在性观察（仅说明该视野）：[]
- 局部未见（仅说明该视野）：[]
- 判据反证：[]
- 未解决：[]
- 候选组审查完成：False；未完成候选：['白芍', 'G4', 'G3', 'G1', '香附']；未映射：[]
- 判定充分性影子结果：{'schema': 'decision-readiness/shadow-v1', 'mode': 'shadow', 'selected_candidate_id': '白芍', 'audit_complete': False, 'formal_authorized': False, 'selected_required_gaps': ['cluster_crystal'], 'competing_required_gaps': {'G4': ['needle_crystal'], 'G3': ['single_crystal'], 'G1': ['cluster_crystal'], '香附': ['secretory_tissue']}, 'supporting_gaps': {'白芍': ['starch_grain', 'vessel', 'fiber'], 'G4': ['fiber', 'stone_cell', 'inulin'], 'G3': ['stoma'], 'G1': ['starch_grain', 'vessel', 'cork_cell', 'fiber']}, 'unmapped': [], 'would_be_ready': False, 'production_authorization_changed': False, 'limit': '未查supporting不单独判否；required观察完成不等于排除竞争候选，仍需独立风险验证'}（不提升生产确认资格）
- 上述未解决引用不等于全部候选缺口清零；完整required/supporting状态见candidate_audit。
- 全部证据及采信边界：final_evidence.json；结论与依据：final_assessment.json。

B 规则分为 required 基础分加 supporting 佐证分，再扣分离区间反证。逐项反证见 agent_verdict.json 的 score_components。

| 候选 | B规则分 | required 基础分 | supporting 加分 | 分离区间反证扣分 | 融合排序分（非正确率） | 支持判据数 | 必需判据缺口数 |
|---|---|---|---|---|---|---|---|
| 白芍 | 0.0 | 0.0 | 0 | 0.0 | 0.488869 | 0 | 1 |
| 白术 / 苍术 | 0.0 | 0.0 | 0 | 0.0 | 0.010369 | 0 | 1 |
| 赤芍 / 牡丹皮 | 0.0 | 0.0 | 0 | 0.0 | 0.000358 | 0 | 1 |
| 枳壳 / 陈皮 | 0.0 | 0.0 | 0 | 0.0 | 0.000214 | 0 | 1 |
| 香附 | 0.0 | 0.0 | 0 | 0.0 | 0.000113 | 0 | 1 |
