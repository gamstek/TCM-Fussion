# 自主B工具与A融合

- 裁决：insufficient_evidence
- 交付输出：tentative / 龙胆
- 内部复核标记：True；交付层 review_required=True
- 强观察：7 条；备选：4 个
- 停止原因：planner_request_review
- 证据版本：1f8f308f317f10eeb7f84c2e9bbc5aadd2cba664bb844cae047fb1911aa6e289
- 原图及原B目录未覆盖；阶段失败/资源明细见agent_trace.json。

## 收尾综合研判

现有证据暂倾向：秦皮；尚不能确认，需要复核。

秦皮是当前倾向：criterion:1 记录 sand_crystal（草酸钙砂晶）C级required阳性，observation:46 独立presence_check以conf 0.8确认细胞腔内密集砂晶，且view_03五个fold（a_prior:29-33）均以秦皮为先验首位，支持方向一致。相比之下滑石/石膏/雄黄/牡蛎仅共享同一矿物feature阳性observation:47（conf仅0.65，VLM自述需与植物纤维区分，无法区分具体矿物），区分度极低；龙胆虽有强A先验（a_prior:19等）但其P0主判针晶criterion:14未检出且b_confidence为0。冲突/缺口：与滑石分差仅0.05且各类仅1类独立证据（notes亦提示双候选人工复核）；observation:47的矿物片状结晶与砂晶是否分属不同视野未澄清（hypothesis:71），view_01针晶分类阳性（classification:55，medium置信、identity未核实）与龙胆/G2假设（hypothesis:67、76）悬置；view_01等存在虚焦不可判裁片（classification:56），D8证据数量门槛未达probable，维持证据不足而非排除。融合分数为未校准先验与证据的合成，非正确率。

- 研判来源：model；模型失败原因：None
- 支持证据：['criterion:1', 'observation:46', 'a_prior:29', 'a_prior:30', 'a_prior:31', 'a_prior:32', 'a_prior:33']
- 反证：[]
- 未解决：['observation:47', 'classification:55', 'hypothesis:67', 'hypothesis:71']
- 全部证据及采信边界：final_evidence.json；结论与依据：final_assessment.json。

| 候选 | B规则分 | 融合排序分（非正确率） | 支持判据数 | 必需判据缺口数 |
|---|---|---|---|---|
| 龙胆 | 0.0 | 0.383256 | 0 | 1 |
| 秦皮 | 0.51 | 0.286235 | 1 | 0 |
| 石膏 | 0.459 | 0.190232 | 1 | 0 |
| 滑石 | 0.459 | 0.183619 | 1 | 0 |
| 雄黄 | 0.459 | 0.183601 | 1 | 0 |
| 牡蛎 | 0.4131 | 0.165301 | 1 | 0 |
| 桔梗 | 0.0 | 0.080481 | 0 | 1 |
| 菟丝子 | 0.0 | 0.001011 | 0 | 1 |
| 黄芪 | 0.0 | 0.000541 | 0 | 1 |
| 苦参 | 0.0 | 0.000471 | 0 | 1 |
| 麦芽 | 0.0 | 0.000317 | 0 | 1 |
| 乌梅 | 0.0 | 5.3e-05 | 0 | 1 |
