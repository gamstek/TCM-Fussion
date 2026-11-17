# 自主B工具与A融合

- 裁决：confirmed
- 交付输出：tentative / 白芍
- 内部复核标记：True；交付层 review_required=True
- 强观察：4 条；备选：4 个
- 停止原因：no_legal_action
- 证据版本：1ebcc2bc6fe93869cf95ac4cfaf83026c503e854bced8efa6bed6017d6ae3829
- 原图及原B目录未覆盖；阶段失败/资源明细见agent_trace.json。

## 收尾综合研判

现有证据暂倾向：白芍；尚不能确认，需要复核。

当前倾向白芍：P0必需特征簇晶检出且判据criterion:1为IN_RANGE，测量记录measurement_record:28（feret_max≈15.7μm，标定calibration:29/30正常）与G1的criterion:7仅NO_REF状态、且G1形态描述为赤芍/牡丹皮（直径更大的簇晶、导管、木栓细胞均未检出，criterion:8~11），郁金仅糊化淀粉粒弱证据（criterion:12、observation:23，conf 0.65），G3方晶与气孔均未检出（criterion:15/16，observation:25反证），茯苓菌丝/菌团未检出（criterion:13/14）。A先验a_prior:17~20在5折中4折rank1白芍（但A为先验、不可当已校准正确率）。冲突/缺口：observation:24检出疑似单个厚壁细胞但否定纤维（对白芍仅P1支持特征，影响有限）；判定仅依赖单视野单类证据（support_feature_count=1，D8b放行），白芍 Supporting 特征淀粉粒/导管/纤维全部NOT_DETECTED（criterion:2~4），且视野仅含一个团块（observation:22），证据覆盖单薄，需人工复核补充视野确认。

- 研判来源：model；模型失败原因：None
- 支持证据：['criterion:1', 'a_prior:17', 'a_prior:18', 'a_prior:19', 'a_prior:20']
- 反证：['observation:24']
- 未解决：['criterion:2', 'criterion:3', 'criterion:4', 'criterion:7', 'criterion:12', 'criterion:13', 'criterion:14', 'measurement_record:28']
- 全部证据及采信边界：final_evidence.json；结论与依据：final_assessment.json。

| 候选 | B规则分 | 融合排序分（非正确率） | 支持判据数 | 必需判据缺口数 |
|---|---|---|---|---|
| 白芍 | 1.0 | 0.872168 | 1 | 0 |
| 赤芍 / 牡丹皮 | 0.6 | 0.240057 | 1 | 0 |
| 郁金 | 0.4131 | 0.172063 | 1 | 0 |
| 茯苓 | 0.0 | 0.001309 | 0 | 2 |
| 枳壳 / 陈皮 | 0.0 | 7e-05 | 0 | 1 |
