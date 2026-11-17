# 自主B工具与A融合

- 裁决：insufficient_evidence
- 交付输出：tentative / 连翘
- 内部复核标记：True；交付层 review_required=True
- 强观察：0 条；备选：4 个
- 停止原因：finalization_time_reserve
- 证据版本：c42c4c0cf3915dfa8cd66573afe1c31ee79422d27e10f1b9656eb5225424655f
- 原图及原B目录未覆盖；阶段失败/资源明细见agent_trace.json。

## 收尾综合研判

现有证据暂倾向：连翘；仅有A先验，缺少独立显微支持，不构成鉴定。

A先验中连翘在view_02/03/04多折反复Top1（如a_prior:27、37达0.97-0.99），融合分0.408远高于次名诃子0.059，但该先验未校准、不能当正确率。Plan B侧criterion:6显示连翘必需特征fiber NOT_DETECTED，observation:42的presence_check因UPSTREAM_REMOTE_DISCONNECTED失败，属未知而非缺失。同时view_04检出多个石细胞（classification:49/50/51/52/54/56）与view_03簇晶（classification:58/59），这些观察supports_candidate_ids为空、未纳入正式判定，未核验身份与形态归属。多数分类图像不可判（classification:47/48/55），证据质量受限。综合：当前倾向连翘（先验占优、无候选获得对抗证据），但正式verdict为insufficient_evidence合理——必需纤维判据未确认，石细胞/簇晶线索与诃子/栀子等候选判据重叠，缺口是纤维的量化核验（segment失败operation:86）与石细胞身份核验。

- 研判来源：model；模型失败原因：None
- 支持证据：['a_prior:23', 'a_prior:27', 'a_prior:28', 'a_prior:30', 'a_prior:31', 'a_prior:33', 'a_prior:34', 'a_prior:35', 'a_prior:37', 'a_prior:38', 'a_prior:39', 'a_prior:40', 'a_prior:41']
- 反证：[]
- 未解决：['criterion:6', 'observation:42', 'classification:49', 'classification:50', 'classification:58']
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
