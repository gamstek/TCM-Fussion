# 自主B工具与A融合

- 裁决：insufficient_evidence
- 交付输出：tentative / 白芍
- 内部复核标记：True；交付层 review_required=True
- 强观察：5 条；备选：4 个
- 停止原因：no_legal_action
- 证据版本：7f20b5f3d39c28a5d26ca8654ae92865f7b8ddd8f44d50d78905249d66bd4fba
- 执行终止原因：no_legal_action；类型：runtime
- 交付复核原因：['FORMAL_REQUIRES_REVIEW', 'INSUFFICIENT_FEATURE_TYPES', 'CANDIDATE_NOT_FORMAL_CONCLUSION', 'FORMAL_VERDICT_NOT_AUTHORIZED']（与执行中断原因分开记录）
- 原图及原B目录未覆盖；阶段失败/资源明细见agent_trace.json。

## 收尾综合研判

暂定排序倾向：白芍；尚不能确认，需要复核。；中间证据显示 cluster_crystal 待核验，正式证据不足。

郁金是唯一有正式规则支持的候选：糊化淀粉粒判据(criterion:ce955ff2c7e76ffa0b265553)与视野内观察(observation:e16918f4d9be213f44e77559)相互印证，且无针对郁金的反驳事实。茯苓有真菌团块支持(observation:d07f3fd96daeba4e06140685)，但其必需特征菌丝未检出且被observation:2bd7a75f1bfd2ad739b8bd96明确反驳，证据强度弱于郁金。白芍仅有A先验排名靠前，规则支持为空，且其必需的簇晶判据(criterion:5928e7c984550e9f2e74091a)未检出、observation:894a7603ec023e8fadf46d02对纤维阴性构成反驳；两条CNN高置信簇晶分类(classification:440b3440095f4e1ae38d494d、classification:88536a8b0570ac43b5ff5105)与未检出结论存在未消解的冲突，可能指向样本处理或视野覆盖缺口。operation:79f9a97ed7bf9714f2dd53ef/operation:cbbd3087ced383a49577bf79的measure_instances失败使尺寸类量化判据全部缺失，观察仅限单一视野，无法排除视野外特征，故整体仍属证据不足而非确定结论。停止原因是no_legal_action，属运行时终止，与正式规则的needs_review需人工复核相独立。

白芍：B规则排序分=0.0；融合排序分=0.472168；已验证独立特征类数=0；最低类数=2。上述分数不是校准正确率；交付复核要求见 review_reasons。

- 解释来源：model；文本警告：[]
- 研判来源：model；模型失败原因：None
- 支持证据：['criterion:ce955ff2c7e76ffa0b265553', 'observation:e16918f4d9be213f44e77559']
- 反证：[]
- 未解决：['observation:2bd7a75f1bfd2ad739b8bd96', 'observation:894a7603ec023e8fadf46d02', 'classification:440b3440095f4e1ae38d494d', 'classification:88536a8b0570ac43b5ff5105']
- 全部证据及采信边界：final_evidence.json；结论与依据：final_assessment.json。

| 候选 | B规则分 | 融合排序分（非正确率） | 支持判据数 | 必需判据缺口数 |
|---|---|---|---|---|
| 白芍 | 0.0 | 0.472168 | 0 | 1 |
| 郁金 | 0.459 | 0.190423 | 1 | 0 |
| 茯苓 | 0.0971 | 0.040149 | 1 | 1 |
| 枳壳 / 陈皮 | 0.0 | 7e-05 | 0 | 1 |
| 赤芍 / 牡丹皮 | 0.0 | 5.7e-05 | 0 | 1 |
