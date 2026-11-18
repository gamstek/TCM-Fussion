# 自主B工具与A融合

- 裁决：confirmed
- 交付输出：tentative / 白芍
- 内部复核标记：True；交付层 review_required=True
- 强观察：7 条；备选：4 个
- 停止原因：planner_request_review
- 证据版本：ec9ec3a48d5ec3f95edffc7c4275ef8c5b06981f6161823994224266a4bde64d
- 执行终止原因：planner_request_review；类型：model
- 交付复核原因：['FORMAL_REQUIRES_REVIEW', 'INSUFFICIENT_FEATURE_TYPES']（与执行中断原因分开记录）
- 原图及原B目录未覆盖；阶段失败/资源明细见agent_trace.json。

## 收尾综合研判

现有证据暂倾向：白芍；尚不能确认，需要复核。

依据已采信判据分析当前候选，未通过的身份、质量及尺寸仍保留不确定性。判据状态：cluster_crystal=IN_RANGE；starch_grain=NOT_DETECTED；vessel=NOT_DETECTED；fiber=NOT_DETECTED。

白芍：B规则排序分=1.0；融合排序分=0.872168；已验证独立特征类数=1；最低类数=2。上述分数不是校准正确率；交付复核要求见 review_reasons。

- 解释来源：deterministic_numeric_guard；文本警告：['MODEL_NUMERIC_CLAIM_UNVERIFIED']
- 研判来源：model；模型失败原因：None
- 支持证据：['criterion:a8d3c5599d907c5d7a5dbe02']
- 反证：['observation:27f17e9b5b3c794c731ed84d']
- 未解决：['criterion:2de3260b61c2458d804fa831', 'criterion:ccfbeb0b2f8187a370e3a24d', 'criterion:f125c4b5e984fe1dc2557f67']
- 全部证据及采信边界：final_evidence.json；结论与依据：final_assessment.json。

| 候选 | B规则分 | 融合排序分（非正确率） | 支持判据数 | 必需判据缺口数 |
|---|---|---|---|---|
| 白芍 | 1.0 | 0.872168 | 1 | 0 |
| 赤芍 / 牡丹皮 | 0.6 | 0.240057 | 1 | 0 |
| 郁金 | 0.459 | 0.190423 | 1 | 0 |
| 茯苓 | 0.0 | 0.001309 | 0 | 2 |
| 枳壳 / 陈皮 | 0.0 | 7e-05 | 0 | 1 |
