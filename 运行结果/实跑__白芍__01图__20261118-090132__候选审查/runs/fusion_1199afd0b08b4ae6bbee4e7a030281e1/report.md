# 自主B工具与A融合

- 裁决：insufficient_evidence
- 交付输出：tentative / 白芍
- 内部复核标记：True；交付层 review_required=True
- 强观察：4 条；备选：4 个
- 停止原因：runtime_request_review
- 证据版本：a72580e303fec4a56785fb64865082e11180f497dc23fe297de842ad01b0d4ad
- 执行终止原因：PLANNER_CHOICE_INVALID；类型：runtime
- 交付复核原因：['FORMAL_REQUIRES_REVIEW', 'CANDIDATE_AUDIT_INCOMPLETE', 'UNMAPPED_A_CANDIDATES', 'INSUFFICIENT_FEATURE_TYPES', 'CANDIDATE_NOT_FORMAL_CONCLUSION', 'FORMAL_VERDICT_NOT_AUTHORIZED']（与执行中断原因分开记录）
- 原图及原B目录未覆盖；阶段失败/资源明细见agent_trace.json。

## 收尾综合研判

现有证据暂倾向：郁金；尚不能确认，需要复核。；中间证据显示 gelatinized_starch 待核验，正式证据不足。

依据已采信判据分析当前候选，未通过的身份、质量及尺寸仍保留不确定性。判据状态：gelatinized_starch=NO_REF。

郁金：B规则排序分=0.4131；融合排序分=0.172063；已验证独立特征类数=1；最低类数=2。上述分数不是校准正确率；交付复核要求见 review_reasons。

- 解释来源：deterministic_numeric_guard；文本警告：['MODEL_NUMERIC_CLAIM_UNVERIFIED']
- 研判来源：model；模型失败原因：None
- 支持证据：['criterion:4733d24862fdcfe404f37197', 'observation:dc6b615c4abd8ff2b43388b5']
- 反证：[]
- 未解决：['criterion:18ea1eeca2f02d9c12227892', 'criterion:2de3260b61c2458d804fa831', 'criterion:ccfbeb0b2f8187a370e3a24d', 'criterion:806ae1fd5efd40b1b1f4d1bb', 'criterion:c06bfe92c2e85e59e2053f06', 'criterion:d7ebf5f1026f31752154b955', 'criterion:a4ee8c48a788846505079341', 'observation:6f34045ef2abf50fedbc5ea9', 'observation:f4bab6652b5527e719a98cd2']
- 候选组审查完成：False；未完成候选：['白芍', 'G1', 'G3']；未映射：['玄明粉']
- 上述未解决引用不等于全部候选缺口清零；完整required/supporting状态见candidate_audit。
- 全部证据及采信边界：final_evidence.json；结论与依据：final_assessment.json。

| 候选 | B规则分 | 融合排序分（非正确率） | 支持判据数 | 必需判据缺口数 |
|---|---|---|---|---|
| 白芍 | 0.0 | 0.472168 | 0 | 1 |
| 郁金 | 0.4131 | 0.172063 | 1 | 0 |
| 茯苓 | 0.0971 | 0.040149 | 1 | 1 |
| 枳壳 / 陈皮 | 0.0 | 7e-05 | 0 | 1 |
| 赤芍 / 牡丹皮 | 0.0 | 5.7e-05 | 0 | 1 |
