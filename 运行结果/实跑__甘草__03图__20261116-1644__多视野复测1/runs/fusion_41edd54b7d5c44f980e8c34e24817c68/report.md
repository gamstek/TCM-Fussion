# 自主B工具与A融合

- 裁决：insufficient_evidence
- 交付输出：tentative / 甘草
- 内部复核标记：True；交付层 review_required=True
- 强观察：2 条；备选：4 个
- 停止原因：runtime_request_review
- 证据版本：2a38a9a6bfba3f77732a99660d988183289e7ff643bae7674816d772bae89451
- 执行终止原因：planning_attempt_limit；类型：runtime
- 交付复核原因：['FORMAL_REQUIRES_REVIEW', 'INSUFFICIENT_FEATURE_TYPES', 'CANDIDATE_NOT_FORMAL_CONCLUSION', 'FORMAL_VERDICT_NOT_AUTHORIZED']（与执行中断原因分开记录）
- 原图及原B目录未覆盖；阶段失败/资源明细见agent_trace.json。

## 收尾综合研判

现有证据暂倾向：甘草；尚不能确认，需要复核。；中间证据显示 crystal_fiber 待核验，正式证据不足。

依据全部已收集证据及原规则综合排序给出当前倾向。判据状态：crystal_fiber=UNCAL。原规则说明：候选池为空或最高分候选无有效支持分（<0.50）。未完成的身份或尺寸仍保留不确定性。

甘草：B规则排序分=0.3825；融合排序分=0.503822；已验证独立特征类数=1；最低类数=2。上述分数不是校准正确率；交付复核要求见 review_reasons。

- 解释来源：deterministic；文本警告：[]
- 研判来源：deterministic；模型失败原因：ASSESSMENT_INVALID
- 支持证据：['criterion:11', 'a_prior:29', 'a_prior:30', 'a_prior:31', 'a_prior:32', 'a_prior:33', 'a_prior:34', 'a_prior:35', 'a_prior:36', 'a_prior:37', 'a_prior:38', 'a_prior:39', 'a_prior:40', 'a_prior:41', 'a_prior:43', 'observation:46']
- 反证：[]
- 未解决：['criterion:11', 'observation:45', 'observation:47', 'classification:56', 'classification:61', 'classification:63', 'classification:64', 'classification:66', 'classification:67', 'classification:68', 'classification:69', 'classification:70']
- 全部证据及采信边界：final_evidence.json；结论与依据：final_assessment.json。

| 候选 | B规则分 | 融合排序分（非正确率） | 支持判据数 | 必需判据缺口数 |
|---|---|---|---|---|
| 甘草 | 0.3825 | 0.503822 | 1 | 1 |
| 苦参 | 0.3825 | 0.398712 | 1 | 1 |
| 连翘 | 0.459 | 0.183601 | 1 | 0 |
| 黄芪 | 0.3825 | 0.15346 | 1 | 1 |
| 白术 / 苍术 | 0.0 | 0.001288 | 2 | 1 |
| 龙胆 | 0.0 | 0.001248 | 0 | 1 |
| 马齿苋 | 0.0 | 2.6e-05 | 0 | 1 |
