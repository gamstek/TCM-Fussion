# 自主B工具与A融合

- 裁决：confirmed
- 交付输出：tentative / 白芍
- 内部复核标记：True；交付层 review_required=True
- 强观察：4 条；备选：4 个
- 停止原因：runtime_request_review
- 证据版本：5e3f3d881d92123c9c4a86a29b27bf0df8e0a53169b504ca6d2eae7c692f8e49
- 原图及原B目录未覆盖；阶段失败/资源明细见agent_trace.json。

## 收尾综合研判

现有证据暂倾向：白芍；尚不能确认，需要复核。

支持：criterion:1簇晶required阳性（calibration:30校准ok，measurement_record:29仅1例feret_max≈13.8μm），且五折A先验a_prior:17/18/19/20均以极高概率排白芍首位。冲突/压制点：observation:25否定纤维存在（0.85置信），是白芍eligible_counter且同时反制G1；但单一视野无法将NOT_DETECTED当成缺席定论。缺口：白芍与G1共享簇晶特征（criterion:7同为NO_REF），区分依赖的fiber/cork_cell等P1 supporting均未检出（criterion:2/3/4），且criterion:11的starch_grain确认因classification:34虚焦不可判；D8b条件放行下支持证据仅1类，故维持needs_review，倾向白芍但融合分0.24仅为未校准先验+证据加权结果，非校准正确率。

- 研判来源：model；模型失败原因：None
- 支持证据：['criterion:1', 'a_prior:17', 'a_prior:18', 'a_prior:19', 'a_prior:20']
- 反证：['observation:25']
- 未解决：['criterion:2', 'criterion:3', 'criterion:4', 'classification:34']
- 全部证据及采信边界：final_evidence.json；结论与依据：final_assessment.json。

| 候选 | B规则分 | 融合排序分（非正确率） | 支持判据数 | 必需判据缺口数 |
|---|---|---|---|---|
| 白芍 | 1.0 | 0.872168 | 1 | 0 |
| 赤芍 / 牡丹皮 | 0.6 | 0.240057 | 1 | 0 |
| 郁金 | 0.4131 | 0.172063 | 1 | 0 |
| 茯苓 | 0.0 | 0.001309 | 0 | 2 |
| 枳壳 / 陈皮 | 0.0 | 7e-05 | 0 | 1 |
