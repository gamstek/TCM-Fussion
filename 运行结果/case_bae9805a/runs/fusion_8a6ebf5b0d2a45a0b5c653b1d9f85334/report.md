# 自主B工具与A融合

- 裁决：insufficient_evidence
- 交付输出：tentative / 龙胆
- 内部复核标记：True；交付层 review_required=True
- 强观察：7 条；备选：4 个
- 停止原因：planner_request_review
- 证据版本：c2cffd66505d5231589d40f17c0f59772c60558b08654e559242fe3f3512e8e0
- 原图及原B目录未覆盖；阶段失败/资源明细见agent_trace.json。

## 收尾综合研判

现有证据暂倾向：龙胆；仅有A先验，缺少独立显微支持，不构成鉴定。

依据全部已收集证据及原规则综合排序给出当前倾向。判据状态：needle_crystal=NOT_DETECTED；epidermis_cell=NOT_DETECTED。原规则说明：D8 证据数量门槛：独立证据特征仅 1 类（sand_crystal），不足以支撑 probable 结论（top1 秦皮 conf 0.51）；降级为证据不足、top1 保留为首个备选，补齐其他判据特征或经 VLM 补证据后重评。未完成的身份或尺寸仍保留不确定性。

- 研判来源：deterministic；模型失败原因：ASSESSMENT_INVALID
- 支持证据：['a_prior:23', 'a_prior:24', 'a_prior:25', 'a_prior:26', 'a_prior:27', 'a_prior:28', 'a_prior:29', 'a_prior:30', 'a_prior:31', 'a_prior:32', 'a_prior:38', 'a_prior:39', 'a_prior:40', 'a_prior:41', 'a_prior:42', 'a_prior:43', 'a_prior:44', 'a_prior:45', 'a_prior:46', 'a_prior:47']
- 反证：['observation:49', 'observation:54']
- 未解决：['criterion:18', 'observation:55', 'classification:61', 'classification:62']
- 全部证据及采信边界：final_evidence.json；结论与依据：final_assessment.json。

| 候选 | B规则分 | 融合排序分（非正确率） | 支持判据数 | 必需判据缺口数 |
|---|---|---|---|---|
| 龙胆 | 0.0 | 0.383256 | 0 | 1 |
| 秦皮 | 0.51 | 0.286235 | 1 | 0 |
| 桔梗 | 0.3825 | 0.233481 | 1 | 1 |
| 石膏 | 0.0 | 0.006632 | 0 | 1 |
| 菟丝子 | 0.0 | 0.001011 | 0 | 1 |
| 黄芪 | 0.0 | 0.000541 | 0 | 1 |
| 苦参 | 0.0 | 0.000471 | 0 | 1 |
| 麦芽 | 0.0 | 0.000317 | 0 | 1 |
| 乌梅 | 0.0 | 5.3e-05 | 0 | 1 |
