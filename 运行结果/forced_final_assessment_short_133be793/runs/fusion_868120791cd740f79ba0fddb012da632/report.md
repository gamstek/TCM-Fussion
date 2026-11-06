# 自主B工具与A融合

- 裁决：insufficient_evidence
- 交付输出：tentative / 秦皮
- 内部复核标记：True；交付层 review_required=True
- 强观察：1 条；备选：1 个
- 停止原因：max_steps
- 证据版本：a02866d8b4f7f8e3b7d595d348583b6ad87efd55d0ad0f6ceee03975905ae3c8
- 原图及原B目录未覆盖；阶段失败/资源明细见agent_trace.json。

## 收尾综合研判

现有证据暂倾向：秦皮；尚不能确认，需要复核。

当前倾向为秦皮。直接显微证据仅一项：criterion:1 的必需特征草酸钙砂晶检出（tier C，贡献0.1377），并由 observation:18 独立观察确认（present=true, conf=0.75，颗粒远小于20μm标尺，形态符合砂晶）。A先验 a_prior:13/14/15/17 四折均将秦皮排第一（0.51～0.99），仅 a_prior:16 一折以桔梗居首，故先验整体偏向秦皮但仅为未校准的先验，不构成校准正确率。反面情况：桔梗的主判特征 secretory_tissue（criterion:2，联结乳管）NOT_DETECTED，按通道灵敏度打折计罚0.06，但'视野内未检出'不等于不存在，属于通道灵敏度不确定而非否定证据。主要缺口：支持特征仅1项且来自单一视野，未对其他结晶类候选（G1簇晶/G2针晶/G3单晶的P0特征 criterion:3/8/11）完成排查，多种必需特征（簇晶、针晶、单晶、乳管等）均未检出或未核查。融合分0.5946属'最高分候选无有效支持分(<0.50)备注'之外但置信不足，formal verdict 为 insufficient_evidence，倾向秦皮但需人工复核与补充视野确认。

- 研判来源：model；模型失败原因：None
- 支持证据：['criterion:1', 'observation:18', 'a_prior:13', 'a_prior:14', 'a_prior:15', 'a_prior:16', 'a_prior:17']
- 反证：[]
- 未解决：['criterion:2']
- 全部证据及采信边界：final_evidence.json；结论与依据：final_assessment.json。

| 候选 | B规则分 | 融合排序分（非正确率） | 支持判据数 | 必需判据缺口数 |
|---|---|---|---|---|
| 秦皮 | 0.459 | 0.5946 | 1 | 0 |
| 桔梗 | 0.0 | 0.167496 | 0 | 1 |
