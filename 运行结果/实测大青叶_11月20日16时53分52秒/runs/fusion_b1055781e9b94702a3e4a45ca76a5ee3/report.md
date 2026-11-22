# 自主B工具与A融合

- 裁决：insufficient_evidence
- 交付输出：tentative / 大青叶
- 内部复核标记：True；交付层 review_required=True
- 强观察：8 条；备选：4 个
- 停止原因：runtime_request_review
- 证据版本：2f1d3c83794261d77dea60d9bad0f18be11a2c43eaa9a3600e676c0de2435c88
- 执行终止原因：UPSTREAM_NETWORK_FAILED；类型：service
- 交付复核原因：['FORMAL_REQUIRES_REVIEW', 'CANDIDATE_AUDIT_INCOMPLETE', 'INSUFFICIENT_FEATURE_TYPES', 'CANDIDATE_NOT_FORMAL_CONCLUSION', 'FORMAL_VERDICT_NOT_AUTHORIZED']（与执行中断原因分开记录）
- 原图及原B目录未覆盖；阶段失败/资源明细见agent_trace.json。

## 裁决与两路排序

正式结论未确认（insufficient_evidence）
融合首位 大青叶（排序分 0.289656）
  - 支持：暂无可引用判据
  - 反证：暂无可引用判据
B 证据首位 赤芍 / 牡丹皮（排序分 0.0）
  - 支持：暂无可引用判据
  - 反证：暂无可引用判据

## 收尾综合研判

现有证据暂倾向：大青叶；尚不能确认，需要复核。；中间证据显示 indigo_crystal 待核验，正式证据不足。

依据全部已收集证据及原规则综合排序给出当前倾向。判据状态：indigo_crystal=MORPHOLOGY_UNVERIFIED；epidermis_cell=NOT_DETECTED；stoma=NOT_DETECTED。原规则说明：必需判据未全部核验或 B 规则分未达倾向门槛。。正式判据未满足不等于原图中不存在该目标；已检测或分类但缺少合格尺寸的实例仍待核验。

大青叶：B规则排序分=0.0；融合排序分=0.289656；已验证独立特征类数=0；B规则计分特征类数=0；B规则最低计分类数=2。上述分数不是校准正确率；交付复核要求见 review_reasons。

- 解释来源：deterministic；文本警告：[]
- 研判来源：deterministic；模型失败原因：UPSTREAM_NETWORK_FAILED
- 研判引用的合格判据：[]
- 模型原提案逐条引用及剔除原因见 [收尾引用审计](final_assessment.json) 的 model_citation_audit。
- 局部存在性观察（仅说明该视野）：['indigo_crystal @ view_01 (observation:9ef3e95e71b53c2b6f13f4b1)']
- 局部未见（仅说明该视野）：['indigo_crystal @ view_02 (observation:adcf9e641c4841da13606e85)', 'indigo_crystal @ view_03 (observation:10af4eb920a6a4c29d20c6c0)']
- 判据反证：[]
- 未解决：['criterion:6dab74c06db9699546423c80']
- 候选组审查完成：False；未完成候选：['秦皮', '大青叶', '穿心莲', '滑石', '雄黄', '乌梅', '连翘', '桔梗', '牡蛎', '菟丝子', '白芍', '地黄', '龙胆']；未映射：[]
- 判定充分性影子结果：{'schema': 'decision-readiness/shadow-v1', 'mode': 'shadow', 'selected_candidate_id': '大青叶', 'audit_complete': False, 'formal_authorized': False, 'selected_required_gaps': ['indigo_crystal'], 'competing_required_gaps': {'秦皮': ['sand_crystal'], '穿心莲': ['calcium_carbonate'], '滑石': ['mineral_feature'], '雄黄': ['mineral_feature'], '乌梅': ['epidermis_cell'], '连翘': ['fiber'], '桔梗': ['secretory_tissue'], '牡蛎': ['mineral_feature'], '菟丝子': ['palisade_cell'], '白芍': ['cluster_crystal'], '地黄': ['parenchyma_cell'], '龙胆': ['needle_crystal']}, 'supporting_gaps': {'大青叶': ['epidermis_cell', 'stoma'], '穿心莲': ['non_glandular_hair', 'glandular_hair', 'stoma'], '乌梅': ['non_glandular_hair'], '白芍': ['starch_grain', 'vessel', 'fiber'], '地黄': ['secretory_tissue', 'vessel'], '龙胆': ['epidermis_cell']}, 'unmapped': [], 'would_be_ready': False, 'production_authorization_changed': False, 'limit': '未查supporting不单独判否；required观察完成不等于排除竞争候选，仍需独立风险验证'}（不提升生产确认资格）
- 上述未解决引用不等于全部候选缺口清零；完整required/supporting状态见candidate_audit。
- 全部证据及采信边界：final_evidence.json；结论与依据：final_assessment.json。

B 规则分为 required 基础分加 supporting 佐证分，再扣分离区间反证。逐项反证见 agent_verdict.json 的 score_components。

| 候选 | B规则分 | required 基础分 | supporting 加分 | 分离区间反证扣分 | 融合排序分（非正确率） | 支持判据数 | 必需判据缺口数 |
|---|---|---|---|---|---|---|---|
| 大青叶 | 0.0 | 0.0 | 0 | 0.0 | 0.289656 | 0 | 1 |
| 秦皮 | 0.0 | 0.0 | 0 | 0.0 | 0.057259 | 0 | 1 |
| 滑石 | 0.0 | 0.0 | 0 | 0.0 | 0.053886 | 0 | 1 |
| 桔梗 | 0.0 | 0.0 | 0 | 0.0 | 0.027736 | 0 | 1 |
| 地黄 | 0.0 | 0.0 | 0 | 0.0 | 0.0196 | 0 | 1 |
| 牡蛎 | 0.0 | 0.0 | 0 | 0.0 | 0.009357 | 0 | 1 |
| 菟丝子 | 0.0 | 0.0 | 0 | 0.0 | 0.008984 | 0 | 1 |
| 乌梅 | 0.0 | 0.0 | 0 | 0.0 | 0.005132 | 0 | 1 |
| 连翘 | 0.0 | 0.0 | 0 | 0.0 | 0.005018 | 0 | 1 |
| 穿心莲 | 0.0 | 0.0 | 0 | 0.0 | 0.003533 | 0 | 1 |
| 白芍 | 0.0 | 0.0 | 0 | 0.0 | 0.002041 | 0 | 1 |
| 雄黄 | 0.0 | 0.0 | 0 | 0.0 | 0.00126 | 0 | 1 |
| 龙胆 | 0.0 | 0.0 | 0 | 0.0 | 0.0007 | 0 | 1 |
