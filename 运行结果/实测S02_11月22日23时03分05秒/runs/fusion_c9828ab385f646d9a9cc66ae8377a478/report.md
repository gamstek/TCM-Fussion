# 自主B工具与A融合

- 裁决：confirmed
- 交付输出：tentative / 白芍
- 内部复核标记：True；交付层 review_required=True
- 强观察：1 条；备选：4 个
- 停止原因：planner_request_review
- 证据版本：db6a079b933c0c6823ac4420e892e01984a04be1effc24cd683810eb081fce23
- 执行终止原因：planner_request_review；类型：model
- 交付复核原因：['FORMAL_REQUIRES_REVIEW', 'CANDIDATE_AUDIT_INCOMPLETE', 'INSUFFICIENT_FEATURE_TYPES']（与执行中断原因分开记录）
- 原图及原B目录未覆盖；阶段失败/资源明细见agent_trace.json。

## 裁决与两路排序

正式结论未确认（confirmed）
融合首位 白芍（排序分 0.888869）
  - 支持：cluster_crystal required，median 30.2μm ∈ [11, 35]μm [criterion:45402fefba504a8503a18c66]
  - 反证：暂无可引用判据
B 证据首位 白芍（排序分 0.8）
  - 支持：cluster_crystal required，median 30.2μm ∈ [11, 35]μm [criterion:45402fefba504a8503a18c66]
  - 反证：暂无可引用判据

## 收尾综合研判

现有证据暂倾向：白芍；尚不能确认，需要复核。

依据已采信判据分析当前候选，未通过的身份、质量及尺寸仍保留不确定性。判据状态：cluster_crystal=IN_RANGE；starch_grain=NOT_DETECTED；vessel=NOT_DETECTED；fiber=NOT_DETECTED。

白芍：B规则排序分=0.8；融合排序分=0.888869；已验证独立特征类数=1；B规则计分特征类数=1；B规则最低计分类数=2。上述分数不是校准正确率；交付复核要求见 review_reasons。

- 解释来源：deterministic_numeric_guard；文本警告：['MODEL_NUMERIC_CLAIM_UNVERIFIED']
- 研判来源：model；模型失败原因：None
- 研判引用的合格判据：['criterion:45402fefba504a8503a18c66']
- 模型原提案逐条引用及剔除原因见 [收尾引用审计](final_assessment.json) 的 model_citation_audit。
- 局部存在性观察（仅说明该视野）：[]
- 局部未见（仅说明该视野）：['fiber @ view_01 (observation:e2ebc3e9e4cd26cfcfa4a411)']
- 判据反证：[]
- 未解决：['criterion:b368406c90876f780af56cd8', 'criterion:8011a56c9185db4044780a4e', 'criterion:489ad0b8c6ebdef396b3ea9d']
- 候选组审查完成：False；未完成候选：['白芍', 'G4', 'G3', 'G1', '香附']；未映射：[]
- 判定充分性影子结果：{'schema': 'decision-readiness/shadow-v1', 'mode': 'shadow', 'selected_candidate_id': '白芍', 'audit_complete': False, 'formal_authorized': False, 'selected_required_gaps': [], 'competing_required_gaps': {'G4': ['needle_crystal'], 'G3': ['single_crystal'], '香附': ['secretory_tissue']}, 'supporting_gaps': {'白芍': ['starch_grain', 'vessel'], 'G4': ['stone_cell', 'inulin'], 'G3': ['stoma'], 'G1': ['starch_grain', 'vessel', 'cork_cell']}, 'unmapped': [], 'would_be_ready': False, 'production_authorization_changed': False, 'limit': '未查supporting不单独判否；required观察完成不等于排除竞争候选，仍需独立风险验证'}（不提升生产确认资格）
- 上述未解决引用不等于全部候选缺口清零；完整required/supporting状态见candidate_audit。
- 全部证据及采信边界：final_evidence.json；结论与依据：final_assessment.json。

B 规则分为 required 基础分加 supporting 佐证分，再扣分离区间反证。逐项反证见 agent_verdict.json 的 score_components。

| 候选 | B规则分 | required 基础分 | supporting 加分 | 分离区间反证扣分 | 融合排序分（非正确率） | 支持判据数 | 必需判据缺口数 |
|---|---|---|---|---|---|---|---|
| 白芍 | 0.8 | 0.8 | 0.0 | 0.0 | 0.888869 | 1 | 0 |
| 赤芍 / 牡丹皮 | 0.48 | 0.48 | 0.0 | 0.0 | 0.240358 | 1 | 0 |
| 虎杖 | 0.48 | 0.48 | 0.0 | 0.0 | 0.24 | 1 | 0 |
| 白术 / 苍术 | 0.0 | 0.0 | 0 | 0.0 | 0.010369 | 0 | 1 |
| 枳壳 / 陈皮 | 0.0 | 0.0 | 0 | 0.0 | 0.000214 | 0 | 1 |
| 香附 | 0.0 | 0.0 | 0 | 0.0 | 0.000113 | 0 | 1 |
