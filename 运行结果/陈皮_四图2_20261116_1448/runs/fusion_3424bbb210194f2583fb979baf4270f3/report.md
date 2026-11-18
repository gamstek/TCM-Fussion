# 自主B工具与A融合

- 裁决：group_level
- 交付输出：tentative / 枳壳 / 陈皮
- 内部复核标记：True；交付层 review_required=True
- 强观察：14 条；备选：4 个
- 停止原因：no_legal_action
- 证据版本：d73a105ee78ea24e192621e5887446250095d79670a5391cefef4d73ee1c15bf
- 执行终止原因：no_legal_action；类型：runtime
- 交付复核原因：['FORMAL_REQUIRES_REVIEW']（与执行中断原因分开记录）
- 原图及原B目录未覆盖；阶段失败/资源明细见agent_trace.json。

## 收尾综合研判

现有证据暂倾向：枳壳 / 陈皮；尚不能确认，需要复核。

依据已采信判据分析当前候选，未通过的身份、质量及尺寸仍保留不确定性。判据状态：single_crystal=IN_RANGE；stoma=UNCAL。

枳壳 / 陈皮：B规则排序分=1.0；融合排序分=0.991607；已验证独立特征类数=2；最低类数=2。上述分数不是校准正确率；交付复核要求见 review_reasons。

- 解释来源：deterministic_numeric_guard；文本警告：['MODEL_NUMERIC_CLAIM_UNVERIFIED']
- 研判来源：model；模型失败原因：None
- 支持证据：['criterion:1', 'criterion:2', 'observation:50', 'a_prior:25', 'a_prior:35']
- 反证：['observation:49']
- 未解决：['classification:91', 'observation:54', 'observation:55']
- 全部证据及采信边界：final_evidence.json；结论与依据：final_assessment.json。

| 候选 | B规则分 | 融合排序分（非正确率） | 支持判据数 | 必需判据缺口数 |
|---|---|---|---|---|
| 枳壳 / 陈皮 | 1.0 | 0.991607 | 2 | 0 |
| 猪苓 | 0.8197 | 0.32788 | 2 | 0 |
| 白术 / 苍术 | 0.0 | 0.002192 | 0 | 1 |
| 马齿苋 | 0.0 | 0.000801 | 0 | 1 |
| 茯苓 | 0.0 | 0.00076 | 2 | 1 |
| 地黄 | 0.0 | 0.000428 | 1 | 0 |
| 麻黄 | 0.0 | 0.000139 | 1 | 0 |
| 桔梗 | 0.0 | 0.000132 | 0 | 1 |
| 当归 | 0.0 | 0.0 | 1 | 0 |
| 泽泻 | 0.0 | 0.0 | 1 | 0 |
