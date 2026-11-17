# 自主B工具与A融合

- 裁决：confirmed
- 交付输出：tentative / 白芍
- 内部复核标记：True；交付层 review_required=True
- 强观察：14 条；备选：4 个
- 停止原因：runtime_request_review
- 证据版本：76df62cda5af157f71cc0cf1df9454da58da529be86a09ab044a2c6a70dd2552
- 原图及原B目录未覆盖；阶段失败/资源明细见agent_trace.json。

## 收尾综合研判

现有证据暂倾向：白芍；尚不能确认，需要复核。

依据全部已收集证据及原规则综合排序给出当前倾向。判据状态：cluster_crystal=IN_RANGE；starch_grain=NOT_DETECTED；vessel=NOT_DETECTED；fiber=NOT_DETECTED。原规则说明：D8b 单类证据条件放行：独立证据仅 1 类（cluster_crystal），但 top1 白芍 conf 1.00 ≥ 0.75 且与 top2 已区分（分差 0.40）；按 v1.6 出结论，证据类数单薄故仍标人工复核。未完成的身份或尺寸仍保留不确定性。

- 研判来源：deterministic；模型失败原因：429
- 支持证据：['criterion:1', 'a_prior:31', 'a_prior:32', 'a_prior:33', 'a_prior:35', 'a_prior:36', 'a_prior:37', 'a_prior:38', 'a_prior:39', 'a_prior:41', 'a_prior:42', 'a_prior:43', 'a_prior:44', 'a_prior:45', 'a_prior:46', 'a_prior:47', 'a_prior:48', 'a_prior:49', 'a_prior:50']
- 反证：['observation:54', 'observation:59', 'observation:61', 'observation:65']
- 未解决：['classification:80', 'classification:87', 'classification:88']
- 全部证据及采信边界：final_evidence.json；结论与依据：final_assessment.json。

| 候选 | B规则分 | 融合排序分（非正确率） | 支持判据数 | 必需判据缺口数 |
|---|---|---|---|---|
| 白芍 | 1.0 | 0.8823 | 1 | 0 |
| 赤芍 / 牡丹皮 | 0.6 | 0.24041 | 1 | 0 |
| 郁金 | 0.4131 | 0.166993 | 1 | 0 |
| 白术 / 苍术 | 0.0 | 0.075871 | 0 | 1 |
| 知母 | 0.0 | 0.002787 | 0 | 1 |
| 桔梗 | 0.0 | 0.001547 | 0 | 1 |
| 牡蛎 | 0.0 | 0.001261 | 0 | 1 |
| 苦参 | 0.0 | 0.00065 | 0 | 1 |
| 茯苓 | 0.0 | 0.000344 | 0 | 2 |
| 枳壳 / 陈皮 | 0.0 | 0.00016 | 0 | 1 |
| 香附 | 0.0 | 0.000156 | 0 | 1 |
| 滑石 | 0.0 | 3.1e-05 | 0 | 1 |
