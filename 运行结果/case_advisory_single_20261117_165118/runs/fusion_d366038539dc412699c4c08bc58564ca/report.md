# 自主B工具与A融合

- 裁决：insufficient_evidence
- 交付输出：tentative / 白芍
- 内部复核标记：True；交付层 review_required=True
- 强观察：6 条；备选：4 个
- 停止原因：no_legal_action
- 证据版本：d84d7dcd0769219702cb783b8dc553f418628b4ffb12d0a73669937d9aa34404
- 执行终止原因：no_legal_action；类型：runtime
- 交付复核原因：['FORMAL_REQUIRES_REVIEW', 'INSUFFICIENT_FEATURE_TYPES', 'CANDIDATE_NOT_FORMAL_CONCLUSION', 'FORMAL_VERDICT_NOT_AUTHORIZED']（与执行中断原因分开记录）
- 原图及原B目录未覆盖；阶段失败/资源明细见agent_trace.json。

## 收尾综合研判

暂定排序倾向：白芍；尚不能确认，需要复核。；中间证据显示 cluster_crystal 待核验，正式证据不足。

依据已采信判据分析当前候选，未通过的身份、质量及尺寸仍保留不确定性。判据状态：cluster_crystal=NOT_DETECTED；starch_grain=NOT_DETECTED；vessel=NOT_DETECTED；fiber=NOT_DETECTED。

白芍：B规则排序分=0.0；融合排序分=0.472168；已验证独立特征类数=0；最低类数=2。上述分数不是校准正确率；交付复核要求见 review_reasons。

- 解释来源：deterministic_numeric_guard；文本警告：['MODEL_NUMERIC_CLAIM_UNVERIFIED']
- 研判来源：model；模型失败原因：None
- 支持证据：['criterion:98498fdf5396005658b6c32b', 'observation:676f6953dba7605103610516']
- 反证：[]
- 未解决：['classification:5f0af8884092e2a362d4f31d', 'classification:f0b89fd9f72e20b392744df2', 'hypothesis:584e3444a9e596ab0e8c6c7c', 'hypothesis:800a493729ee7337f4bf45bc', 'hypothesis:0405b557fff3a337ece8a3d8']
- 全部证据及采信边界：final_evidence.json；结论与依据：final_assessment.json。

| 候选 | B规则分 | 融合排序分（非正确率） | 支持判据数 | 必需判据缺口数 |
|---|---|---|---|---|
| 白芍 | 0.0 | 0.472168 | 0 | 1 |
| 郁金 | 0.4131 | 0.172063 | 1 | 0 |
| 茯苓 | 0.0 | 0.001309 | 0 | 2 |
| 枳壳 / 陈皮 | 0.0 | 7e-05 | 0 | 1 |
| 赤芍 / 牡丹皮 | 0.0 | 5.7e-05 | 0 | 1 |
