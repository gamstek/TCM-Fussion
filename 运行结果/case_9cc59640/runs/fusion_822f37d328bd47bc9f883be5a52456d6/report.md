# 自主B工具与A融合

- 裁决：insufficient_evidence
- 交付输出：tentative / 龙胆
- 内部复核标记：True；交付层 review_required=True
- 强观察：1 条；备选：4 个
- 停止原因：override_exhausted
- 证据版本：41c0ca89cbb41a906492408f869b3fd6cd6b042efe760f66a0690c494f558b47
- 原图及原B目录未覆盖；阶段失败/资源明细见agent_trace.json。

## 收尾综合研判

现有证据暂倾向：龙胆；仅有A先验，缺少独立显微支持，不构成鉴定。

依据全部已收集证据及原规则综合排序给出当前倾向。判据状态：needle_crystal=NOT_DETECTED；epidermis_cell=NOT_DETECTED。原规则说明：候选池为空或最高分候选无有效支持分（<0.50）。未完成的身份或尺寸仍保留不确定性。

- 研判来源：deterministic；模型失败原因：ASSESSMENT_INVALID
- 支持证据：['a_prior:20', 'a_prior:21', 'a_prior:22', 'a_prior:23', 'a_prior:24', 'a_prior:25', 'a_prior:26', 'a_prior:27', 'a_prior:28', 'a_prior:29', 'a_prior:35', 'a_prior:36', 'a_prior:37', 'a_prior:38', 'a_prior:39', 'a_prior:40', 'a_prior:41', 'a_prior:42', 'a_prior:43', 'a_prior:44']
- 反证：[]
- 未解决：['criterion:18', 'classification:49', 'classification:50']
- 全部证据及采信边界：final_evidence.json；结论与依据：final_assessment.json。

| 候选 | B规则分 | 融合排序分（非正确率） | 支持判据数 | 必需判据缺口数 |
|---|---|---|---|---|
| 龙胆 | 0.0 | 0.383256 | 0 | 1 |
| 秦皮 | 0.0 | 0.082235 | 0 | 1 |
| 桔梗 | 0.0 | 0.080481 | 0 | 1 |
| 石膏 | 0.0 | 0.006632 | 0 | 1 |
| 菟丝子 | 0.0 | 0.001011 | 0 | 1 |
| 黄芪 | 0.0 | 0.000541 | 0 | 1 |
| 苦参 | 0.0 | 0.000471 | 0 | 1 |
| 麦芽 | 0.0 | 0.000317 | 0 | 1 |
| 乌梅 | 0.0 | 5.3e-05 | 0 | 1 |
