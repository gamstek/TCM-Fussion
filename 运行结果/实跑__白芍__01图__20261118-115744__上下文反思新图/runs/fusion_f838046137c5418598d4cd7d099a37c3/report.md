# 自主B工具与A融合

- 裁决：confirmed
- 交付输出：tentative / 白芍
- 内部复核标记：True；交付层 review_required=True
- 强观察：5 条；备选：4 个
- 停止原因：planner_request_review
- 证据版本：6d42fa60291046bdb1f670c7732a2d25cb36696628f4ce1e4f051e4dc9ad6dfd
- 执行终止原因：planner_request_review；类型：model
- 交付复核原因：['FORMAL_REQUIRES_REVIEW', 'CANDIDATE_AUDIT_INCOMPLETE', 'INSUFFICIENT_FEATURE_TYPES']（与执行中断原因分开记录）
- 原图及原B目录未覆盖；阶段失败/资源明细见agent_trace.json。

## 收尾综合研判

现有证据暂倾向：白芍；尚不能确认，需要复核。

依据已采信判据分析当前候选，未通过的身份、质量及尺寸仍保留不确定性。判据状态：cluster_crystal=IN_RANGE；starch_grain=NOT_DETECTED；vessel=NOT_DETECTED；fiber=NOT_DETECTED。

白芍：B规则排序分=1.0；融合排序分=0.986643；已验证独立特征类数=1；最低类数=2。上述分数不是校准正确率；交付复核要求见 review_reasons。

- 解释来源：deterministic_numeric_guard；文本警告：['MODEL_NUMERIC_CLAIM_UNVERIFIED']
- 研判来源：model；模型失败原因：None
- 支持证据：['criterion:a47f1f7365a155e636b32426']
- 反证：['observation:7443bdefa668f922972c6bb6']
- 未解决：['criterion:2de3260b61c2458d804fa831', 'criterion:c3d2157706b5fed64bc3eeb9', 'criterion:d65f39b215a55ea87893efbe', 'criterion:c06bfe92c2e85e59e2053f06', 'criterion:d2a7519a834e440946d3437d', 'observation:5f2f8e8fc0a6c6060b301735']
- 候选组审查完成：False；未完成候选：['白芍', 'G4', 'G3', 'G1', '香附']；未映射：[]
- 判定充分性影子结果：{'schema': 'decision-readiness/shadow-v1', 'mode': 'shadow', 'selected_candidate_id': '白芍', 'audit_complete': False, 'formal_authorized': False, 'selected_required_gaps': [], 'competing_required_gaps': {'G4': ['needle_crystal'], 'G3': ['single_crystal'], '香附': ['secretory_tissue']}, 'supporting_gaps': {'白芍': ['starch_grain', 'vessel'], 'G4': ['stone_cell'], 'G1': ['starch_grain', 'vessel', 'cork_cell']}, 'unmapped': [], 'would_be_ready': False, 'production_authorization_changed': False, 'limit': '未查supporting不单独判否；required观察完成不等于排除竞争候选，仍需独立风险验证'}（不提升生产确认资格）
- 上述未解决引用不等于全部候选缺口清零；完整required/supporting状态见candidate_audit。
- 全部证据及采信边界：final_evidence.json；结论与依据：final_assessment.json。

| 候选 | B规则分 | 融合排序分（非正确率） | 支持判据数 | 必需判据缺口数 |
|---|---|---|---|---|
| 白芍 | 1.0 | 0.986643 | 1 | 0 |
| 赤芍 / 牡丹皮 | 0.66 | 0.26443 | 1 | 0 |
| 香附 | 0.3825 | 0.153136 | 0 | 1 |
| 白术 / 苍术 | 0.0 | 0.012443 | 1 | 1 |
| 枳壳 / 陈皮 | 0.0 | 0.000256 | 0 | 1 |
