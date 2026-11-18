# 自主B工具与A融合

- 裁决：confirmed
- 交付输出：tentative / 白芍
- 内部复核标记：True；交付层 review_required=True
- 强观察：4 条；备选：4 个
- 停止原因：workflow_deadline
- 证据版本：fbed39a41ad4bbad5399fe1bc95aa1507aec8218c16f33555dd65c7b2b653f02
- 执行终止原因：workflow_deadline；类型：runtime
- 交付复核原因：['FORMAL_REQUIRES_REVIEW', 'CANDIDATE_AUDIT_INCOMPLETE', 'UNMAPPED_A_CANDIDATES', 'INSUFFICIENT_FEATURE_TYPES']（与执行中断原因分开记录）
- 原图及原B目录未覆盖；阶段失败/资源明细见agent_trace.json。

## 收尾综合研判

现有证据暂倾向：白芍；尚不能确认，需要复核。

依据全部已收集证据及原规则综合排序给出当前倾向。判据状态：cluster_crystal=IN_RANGE；starch_grain=NOT_DETECTED；vessel=NOT_DETECTED；fiber=NOT_DETECTED。原规则说明：D8b 单类证据条件放行：独立证据仅 1 类（cluster_crystal），但 top1 白芍 conf 1.00 ≥ 0.75 且与 top2 已区分（分差 0.40）；按 v1.6 出结论，证据类数单薄故仍标人工复核。正式判据未满足不等于原图中不存在该目标；已检测或分类但缺少合格尺寸的实例仍待核验。

白芍：B规则排序分=1.0；融合排序分=0.872168；已验证独立特征类数=1；最低类数=2。上述分数不是校准正确率；交付复核要求见 review_reasons。

- 解释来源：deterministic；文本警告：[]
- 研判来源：deterministic；模型失败原因：WORKFLOW_CLOSED
- 支持证据：['criterion:d51745bd7d543b90fcf12cf5']
- 反证：['observation:6a71d964698496b2b537ab11']
- 未解决：['observation:03f5ac591f47badc0a2df95b']
- 候选组审查完成：False；未完成候选：['白芍', 'G1', 'G3']；未映射：['玄明粉']
- 判定充分性影子结果：{'schema': 'decision-readiness/shadow-v1', 'mode': 'shadow', 'selected_candidate_id': '白芍', 'audit_complete': False, 'formal_authorized': False, 'selected_required_gaps': [], 'competing_required_gaps': {'G3': ['single_crystal']}, 'supporting_gaps': {'白芍': ['starch_grain', 'vessel'], 'G1': ['starch_grain', 'vessel', 'cork_cell'], 'G3': ['stoma']}, 'unmapped': ['玄明粉'], 'would_be_ready': False, 'production_authorization_changed': False, 'limit': '未查supporting不单独判否；required观察完成不等于排除竞争候选，仍需独立风险验证'}（不提升生产确认资格）
- 上述未解决引用不等于全部候选缺口清零；完整required/supporting状态见candidate_audit。
- 全部证据及采信边界：final_evidence.json；结论与依据：final_assessment.json。

| 候选 | B规则分 | 融合排序分（非正确率） | 支持判据数 | 必需判据缺口数 |
|---|---|---|---|---|
| 白芍 | 1.0 | 0.872168 | 1 | 0 |
| 赤芍 / 牡丹皮 | 0.6 | 0.240057 | 1 | 0 |
| 郁金 | 0.4131 | 0.172063 | 1 | 0 |
| 茯苓 | 0.0971 | 0.040149 | 1 | 1 |
| 枳壳 / 陈皮 | 0.0 | 7e-05 | 0 | 1 |
