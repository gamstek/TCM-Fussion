# 自主B工具与A融合

- 裁决：confirmed
- 交付输出：tentative / 白芍
- 内部复核标记：True；交付层 review_required=True
- 强观察：8 条；备选：4 个
- 停止原因：runtime_request_review
- 证据版本：df464fa441bacf785b717d9799305653e4233097b0eb4fb99b3ef29fab644986
- 执行终止原因：finalization_time_reserve；类型：runtime
- 交付复核原因：['FORMAL_REQUIRES_REVIEW', 'INSUFFICIENT_FEATURE_TYPES']（与执行中断原因分开记录）
- 原图及原B目录未覆盖；阶段失败/资源明细见agent_trace.json。

## 收尾综合研判

现有证据暂倾向：白芍；尚不能确认，需要复核。

依据已采信判据分析当前候选，未通过的身份、质量及尺寸仍保留不确定性。判据状态：cluster_crystal=IN_RANGE；starch_grain=NOT_DETECTED；vessel=NOT_DETECTED；fiber=NOT_DETECTED。

白芍：B规则排序分=1.0；融合排序分=0.8823；已验证独立特征类数=1；最低类数=2。上述分数不是校准正确率；交付复核要求见 review_reasons。

- 解释来源：deterministic_numeric_guard；文本警告：['MODEL_NUMERIC_CLAIM_UNVERIFIED']
- 研判来源：model；模型失败原因：None
- 支持证据：['criterion:1', 'a_prior:37', 'a_prior:41', 'a_prior:42', 'a_prior:43', 'a_prior:44', 'a_prior:47', 'a_prior:48', 'a_prior:49', 'a_prior:51', 'a_prior:52', 'a_prior:53', 'a_prior:54', 'a_prior:55']
- 反证：[]
- 未解决：['criterion:7', 'criterion:2', 'criterion:3', 'criterion:4', 'classification:97', 'hypothesis:149']
- 全部证据及采信边界：final_evidence.json；结论与依据：final_assessment.json。

| 候选 | B规则分 | 融合排序分（非正确率） | 支持判据数 | 必需判据缺口数 |
|---|---|---|---|---|
| 白芍 | 1.0 | 0.8823 | 1 | 0 |
| 赤芍 / 牡丹皮 | 0.6 | 0.24041 | 1 | 0 |
| 白术 / 苍术 | 0.3 | 0.195871 | 1 | 0 |
| 知母 | 0.0 | 0.002787 | 0 | 1 |
| 郁金 | 0.0 | 0.001753 | 0 | 1 |
| 桔梗 | 0.0 | 0.001547 | 0 | 1 |
| 牡蛎 | 0.0 | 0.001261 | 0 | 1 |
| 苦参 | 0.0 | 0.00065 | 0 | 1 |
| 茯苓 | 0.0 | 0.000344 | 0 | 2 |
| 枳壳 / 陈皮 | 0.0 | 0.00016 | 0 | 1 |
| 香附 | 0.0 | 0.000156 | 0 | 1 |
| 滑石 | 0.0 | 3.1e-05 | 0 | 1 |
