# 自主B工具与A融合

- 裁决：insufficient_evidence
- 交付输出：tentative / 秦皮
- 内部复核标记：True；交付层 review_required=True
- 强观察：3 条；备选：1 个
- 停止原因：max_steps
- 证据版本：8df69e45a67dbdd758aa13997a154ec13ad373e6b835ee3fc98a471de2edc04d
- 原图及原B目录未覆盖；阶段失败/资源明细见agent_trace.json。

## 收尾综合研判

现有证据暂倾向：秦皮；尚不能确认，需要复核。

依据全部已收集证据及原规则综合排序给出当前倾向。判据状态：sand_crystal=NO_REF。原规则说明：D8 证据数量门槛：独立证据特征仅 1 类（sand_crystal），不足以支撑 probable 结论（top1 秦皮 conf 0.51）；降级为证据不足、top1 保留为首个备选，补齐其他判据特征或经 VLM 补证据后重评。未完成的身份或尺寸仍保留不确定性。

- 研判来源：deterministic；模型失败原因：ASSESSMENT_INVALID
- 支持证据：['criterion:1', 'a_prior:141', 'a_prior:142', 'a_prior:143', 'a_prior:144', 'a_prior:145', 'observation:146']
- 反证：[]
- 未解决：[]
- 全部证据及采信边界：final_evidence.json；结论与依据：final_assessment.json。

| 候选 | B规则分 | 融合排序分（非正确率） | 支持判据数 | 必需判据缺口数 |
|---|---|---|---|---|
| 秦皮 | 0.51 | 0.615 | 1 | 0 |
| 桔梗 | 0.0 | 0.167496 | 0 | 1 |
