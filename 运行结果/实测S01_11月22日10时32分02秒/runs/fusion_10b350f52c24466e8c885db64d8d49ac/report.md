# 自主B工具与A融合

- 裁决：insufficient_evidence
- 交付输出：tentative / 滑石
- 内部复核标记：True；交付层 review_required=True
- 强观察：5 条；备选：4 个
- 停止原因：planner_request_review
- 证据版本：afb46b8d6dca506df226dac2f527c01eeed24803c11e8087da9ea3db2c7b66b5
- 执行终止原因：planner_request_review；类型：model
- 交付复核原因：['FORMAL_REQUIRES_REVIEW', 'CANDIDATE_AUDIT_INCOMPLETE', 'INSUFFICIENT_FEATURE_TYPES', 'CANDIDATE_NOT_FORMAL_CONCLUSION', 'FORMAL_VERDICT_NOT_AUTHORIZED']（与执行中断原因分开记录）
- 原图及原B目录未覆盖；阶段失败/资源明细见agent_trace.json。

## 裁决与两路排序

正式结论未确认（insufficient_evidence）
融合首位 滑石（排序分 0.215058）
  - 支持：暂无可引用判据
  - 反证：暂无可引用判据
B 证据首位 赤芍 / 牡丹皮（排序分 0.0）
  - 支持：暂无可引用判据
  - 反证：暂无可引用判据

## 收尾综合研判

现有证据暂倾向：滑石；尚不能确认，需要复核。；中间证据显示 mineral_feature 待核验，正式证据不足。

依据已采信判据分析当前候选，未通过的身份、质量及尺寸仍保留不确定性。判据状态：mineral_feature=MORPHOLOGY_UNVERIFIED。

滑石：B规则排序分=0.0；融合排序分=0.215058；已验证独立特征类数=0；B规则计分特征类数=0；B规则最低计分类数=2。上述分数不是校准正确率；交付复核要求见 review_reasons。

- 解释来源：deterministic_numeric_guard；文本警告：['MODEL_NUMERIC_CLAIM_UNVERIFIED']
- 研判来源：model；模型失败原因：None
- 研判引用的合格判据：[]
- 模型原提案逐条引用及剔除原因见 [收尾引用审计](final_assessment.json) 的 model_citation_audit。
- 局部存在性观察（仅说明该视野）：['mineral_feature @ view_01 (observation:475d8922a41b493788fabba6)']
- 局部未见（仅说明该视野）：[]
- 判据反证：[]
- 未解决：['criterion:47bc398a9a05ccff6eb5de36']
- 候选组审查完成：False；未完成候选：['秦皮', '大青叶', '滑石', '雄黄']；未映射：[]
- 判定充分性影子结果：{'schema': 'decision-readiness/shadow-v1', 'mode': 'shadow', 'selected_candidate_id': '滑石', 'audit_complete': False, 'formal_authorized': False, 'selected_required_gaps': ['mineral_feature'], 'competing_required_gaps': {'秦皮': ['sand_crystal'], '大青叶': ['indigo_crystal'], '雄黄': ['mineral_feature']}, 'supporting_gaps': {}, 'unmapped': [], 'would_be_ready': False, 'production_authorization_changed': False, 'limit': '未查supporting不单独判否；required观察完成不等于排除竞争候选，仍需独立风险验证'}（不提升生产确认资格）
- 上述未解决引用不等于全部候选缺口清零；完整required/supporting状态见candidate_audit。
- 全部证据及采信边界：final_evidence.json；结论与依据：final_assessment.json。

B 规则分为 required 基础分加 supporting 佐证分，再扣分离区间反证。逐项反证见 agent_verdict.json 的 score_components。

| 候选 | B规则分 | required 基础分 | supporting 加分 | 分离区间反证扣分 | 融合排序分（非正确率） | 支持判据数 | 必需判据缺口数 |
|---|---|---|---|---|---|---|---|
| 滑石 | 0.0 | 0.0 | 0 | 0.0 | 0.215058 | 0 | 1 |
| 大青叶 | 0.0 | 0.0 | 0 | 0.0 | 0.17782 | 0 | 1 |
| 秦皮 | 0.0 | 0.0 | 0 | 0.0 | 0.055361 | 0 | 1 |
| 穿心莲 | 0.0 | 0.0 | 0 | 0.0 | 0.009746 | 0 | 1 |
| 乌梅 | 0.0 | 0.0 | 0 | 0.0 | 0.008448 | 0 | 1 |
| 雄黄 | 0.0 | 0.0 | 0 | 0.0 | 0.005004 | 0 | 1 |
