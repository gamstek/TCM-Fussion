# 自主B工具与A融合

- 裁决：confirmed
- 交付输出：tentative / 白芍
- 内部复核标记：True；交付层 review_required=True
- 强观察：4 条；备选：4 个
- 停止原因：runtime_request_review
- 证据版本：4a8cc4766f0a3621dde9df04b0d90ac905366842142762c9109000b7cfd487fb
- 原图及原B目录未覆盖；阶段失败/资源明细见agent_trace.json。

## 收尾综合研判

现有证据暂倾向：白芍；尚不能确认，需要复核。

倾向白芍。criterion:1 簇晶 required IN_RANGE，measurement_record:60（feret_max≈22.7μm，校准 calibration:61 ok）与白芍尺寸域一致；a_prior 多个视野高概率支持（a_prior:34/38/40/43/45/48/50/52），且与 top2 G1 分差 0.40（notes D8b 放行）。冲突：observation:54 纤维未检出（conf 0.55，仅限单视野）反证白芍 supporting 特征缺失，但属弱证据且非 required。G1 被 observation:56 木栓细胞阴性削弱，G4 缺 needle_crystal（criterion:18）而菊糖仅 supporting（criterion:21, observation:53 conf 0.62）。缺口：仅 1 类独立证据（cluster_crystal），淀粉粒/导管未检出不可判为不存在；needs_review 保持合理。

- 研判来源：model；模型失败原因：None
- 支持证据：['criterion:1', 'a_prior:34', 'a_prior:38', 'a_prior:40', 'a_prior:43', 'a_prior:45', 'a_prior:48', 'a_prior:50', 'a_prior:52']
- 反证：['observation:54']
- 未解决：['observation:53', 'observation:56', 'observation:57']
- 全部证据及采信边界：final_evidence.json；结论与依据：final_assessment.json。

| 候选 | B规则分 | 融合排序分（非正确率） | 支持判据数 | 必需判据缺口数 |
|---|---|---|---|---|
| 白芍 | 1.0 | 0.8823 | 1 | 0 |
| 赤芍 / 牡丹皮 | 0.6 | 0.24041 | 1 | 0 |
| 白术 / 苍术 | 0.0 | 0.075871 | 1 | 1 |
| 知母 | 0.0 | 0.002787 | 0 | 1 |
| 郁金 | 0.0 | 0.001753 | 0 | 1 |
| 桔梗 | 0.0 | 0.001547 | 0 | 1 |
| 牡蛎 | 0.0 | 0.001261 | 0 | 1 |
| 苦参 | 0.0 | 0.00065 | 0 | 1 |
| 茯苓 | 0.0 | 0.000344 | 0 | 2 |
| 枳壳 / 陈皮 | 0.0 | 0.00016 | 0 | 1 |
| 香附 | 0.0 | 0.000156 | 0 | 1 |
| 滑石 | 0.0 | 3.1e-05 | 0 | 1 |
