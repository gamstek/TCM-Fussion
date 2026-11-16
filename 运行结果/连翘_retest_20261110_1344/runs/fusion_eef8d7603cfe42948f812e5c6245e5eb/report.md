# 自主B工具与A融合

- 裁决：insufficient_evidence
- 交付输出：tentative / 连翘
- 内部复核标记：True；交付层 review_required=True
- 强观察：0 条；备选：4 个
- 停止原因：finalization_time_reserve
- 证据版本：fd6381b10ae907728572192f114cc343adc9d33c6dcece69e0d15e838fc1036b
- 原图及原B目录未覆盖；阶段失败/资源明细见agent_trace.json。

## 收尾综合研判

现有证据暂倾向：连翘；仅有A先验，缺少独立显微支持，不构成鉴定。

多视图A先验（a_prior:34、a_prior:37、a_prior:44–48等）持续高概率指向连翘，为当前唯一有先验支持的候选，但其主判特征内果皮纤维未检出（criterion:6），B侧无任何检出支持，故维持insufficient_evidence。view_03/view_04检出多个石细胞（classification:61、63、67、69）与疑似气孔（classification:60），与连翘的石细胞/内果皮纤维形态部分相容但均无量化核验（calibration:49–52有效，px_per_um=15.55）；classification:53检出纤维但artifact未索引到候选支持集。缺口：连翘必需的纤维特征及其尺寸/形态量化判定仍未完成，融合分数0.407667为未校准先验+未支持证据的组合，不可当作已校准正确率。

- 研判来源：model；模型失败原因：None
- 支持证据：['a_prior:34', 'a_prior:37', 'a_prior:38', 'a_prior:44', 'a_prior:45', 'a_prior:46', 'a_prior:47', 'a_prior:48']
- 反证：[]
- 未解决：['criterion:6', 'classification:53', 'classification:60', 'classification:61', 'classification:63', 'classification:67', 'classification:69']
- 全部证据及采信边界：final_evidence.json；结论与依据：final_assessment.json。

| 候选 | B规则分 | 融合排序分（非正确率） | 支持判据数 | 必需判据缺口数 |
|---|---|---|---|---|
| 连翘 | 0.0 | 0.407667 | 0 | 1 |
| 诃子 | 0.0 | 0.059216 | 0 | 1 |
| 甘草 | 0.0 | 0.040354 | 0 | 1 |
| 龙胆 | 0.0 | 0.021874 | 0 | 1 |
| 黄芩 | 0.0 | 0.01619 | 0 | 1 |
| 黄芪 | 0.0 | 0.011059 | 0 | 1 |
| 厚朴 | 0.0 | 0.009905 | 0 | 1 |
| 麻黄 | 0.0 | 0.003211 | 0 | 1 |
| 栀子 | 0.0 | 0.000679 | 0 | 1 |
