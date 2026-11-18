# 自主B工具与A融合

- 裁决：confirmed
- 交付输出：tentative / 白芍
- 内部复核标记：True；交付层 review_required=True
- 强观察：5 条；备选：4 个
- 停止原因：planner_request_review
- 证据版本：05881a3cd3b3d5e897f380ea3627ae0177d9f6deb608c78813c46e4e918e75e3
- 执行终止原因：planner_request_review；类型：model
- 交付复核原因：['FORMAL_REQUIRES_REVIEW', 'CANDIDATE_AUDIT_INCOMPLETE', 'UNMAPPED_A_CANDIDATES', 'INSUFFICIENT_FEATURE_TYPES']（与执行中断原因分开记录）草酸钙簇晶直径1
- 原图及原B目录未覆盖；阶段失败/资源明细见agent_trace.json。

## 收尾综合研判

现有证据暂倾向：白芍；尚不能确认，需要复核。

依据已采信判据分析当前候选，未通过的身份、质量及尺寸仍保留不确定性。判据状态：cluster_crystal=IN_RANGE；starch_grain=NOT_DETECTED；vessel=NOT_DETECTED；fiber=NOT_DETECTED。

白芍：B规则排序分=1.0；融合排序分=0.872168；已验证独立特征类数=1；最低类数=2。上述分数不是校准正确率；交付复核要求见 review_reasons。

- 解释来源：deterministic_numeric_guard；文本警告：['MODEL_NUMERIC_CLAIM_UNVERIFIED']
- 研判来源：model；模型失败原因：None
- 支持证据：['criterion:359f38026291c8524d8ca276']
- 反证：['observation:c6a8e652a9318a088fee79b2']
- 未解决：['criterion:2de3260b61c2458d804fa831', 'criterion:ccfbeb0b2f8187a370e3a24d', 'observation:2bd4f4534d074dfeb572dbe3', 'criterion:336ad51efb41e3c8509c2370']
- 候选组审查完成：False；未完成候选：['白芍', 'G1', 'G3', '茯苓']；未映射：['玄明粉']
- 判定充分性影子结果：{'schema': 'decision-readiness/shadow-v1', 'mode': 'shadow', 'selected_candidate_id': '白芍', 'audit_complete': False, 'formal_authorized': False, 'selected_required_gaps': [], 'competing_required_gaps': {'G3': ['single_crystal'], '茯苓': ['fungal_mass']}, 'supporting_gaps': {'白芍': ['starch_grain', 'vessel'], 'G1': ['starch_grain', 'vessel', 'cork_cell']}, 'unmapped': ['玄明粉'], 'would_be_ready': False, 'production_authorization_changed': False, 'limit': '未查supporting不单独判否；required观察完成不等于排除竞争候选，仍需独立风险验证'}（不提升生产确认资格）
- 上述未解决引用不等于全部候选缺口清零；完整required/supporting状态见candidate_audit。
- 全部证据及采信边界：final_evidence.json；结论与依据：final_assessment.json。

| 候选 | B规则分 | 融合排序分（非正确率） | 支持判据数 | 必需判据缺口数 |
|---|---|---|---|---|
| 白芍 | 1.0 | 0.872168 | 1 | 0 |
| 赤芍 / 牡丹皮 | 0.6 | 0.240057 | 1 | 0 |
| 郁金 | 0.4131 | 0.172063 | 1 | 0 |
| 茯苓 | 0.0 | 0.001309 | 0 | 2 |
| 枳壳 / 陈皮 | 0.0 | 7e-05 | 0 | 1 |
