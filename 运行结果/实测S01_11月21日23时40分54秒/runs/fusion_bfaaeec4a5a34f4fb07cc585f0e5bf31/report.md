# 自主B工具与A融合

- 裁决：insufficient_evidence
- 交付输出：tentative / 大青叶
- 内部复核标记：True；交付层 review_required=True
- 强观察：8 条；备选：4 个
- 停止原因：planner_request_review
- 证据版本：ac35a0745195942ba6ce135c8332d2786a833b01404f9344aae56552211c5c5c
- 执行终止原因：planner_request_review；类型：model
- 交付复核原因：['FORMAL_REQUIRES_REVIEW', 'CANDIDATE_AUDIT_INCOMPLETE', 'INSUFFICIENT_FEATURE_TYPES', 'CANDIDATE_NOT_FORMAL_CONCLUSION', 'FORMAL_VERDICT_NOT_AUTHORIZED']（与执行中断原因分开记录）
- 原图及原B目录未覆盖；阶段失败/资源明细见agent_trace.json。

## 裁决与两路排序

正式结论未确认（insufficient_evidence）
融合首位 滑石（排序分 0.215058）
  - 支持：暂无可引用判据
  - 反证：暂无可引用判据
B 证据首位 赤芍 / 牡丹皮（排序分 0.0）
  - 支持：暂无可引用判据
  - 反证：暂无可引用判据

## 收尾综合研判

暂定排序倾向：大青叶；尚不能确认，需要复核。

依据全部已收集证据及原规则综合排序给出当前倾向。判据状态：indigo_crystal=NOT_DETECTED；epidermis_cell=MORPHOLOGY_UNVERIFIED；stoma=NOT_DETECTED。原规则说明：必需判据未全部核验或 B 规则分未达倾向门槛。。正式判据未满足不等于原图中不存在该目标；已检测或分类但缺少合格尺寸的实例仍待核验。

大青叶：B规则排序分=0.0；融合排序分=0.17782；已验证独立特征类数=0；B规则计分特征类数=0；B规则最低计分类数=2。上述分数不是校准正确率；交付复核要求见 review_reasons。

- 解释来源：deterministic_selection_guard；文本警告：['ASSESSMENT_SELECTION_REBOUND']
- 研判来源：model；模型失败原因：None
- 研判引用的合格判据：[]
- 模型原提案逐条引用及剔除原因见 [收尾引用审计](final_assessment.json) 的 model_citation_audit。
- 局部存在性观察（仅说明该视野）：['epidermis_cell @ view_01 (observation:da5c7920ecee6b6c91570a16)']
- 局部未见（仅说明该视野）：['indigo_crystal @ view_01 (observation:96565460d670525bf6231018)', 'stoma @ view_01 (observation:c33b0c57acd19b50f2043fa2)']
- 判据反证：[]
- 未解决：['criterion:fb09debaa8b237b9f8efa8f1', 'classification:92126033d77c28d22ac41254', 'classification:2c3a6c6d122a1e463b5352a4', 'classification:c3a9ed267b99c5f90df74590', 'classification:32139f44da7ebd6dd09bbf46']
- 候选组审查完成：False；未完成候选：['秦皮', '大青叶', '乌梅']；未映射：[]
- 判定充分性影子结果：{'schema': 'decision-readiness/shadow-v1', 'mode': 'shadow', 'selected_candidate_id': '滑石', 'audit_complete': False, 'formal_authorized': False, 'selected_required_gaps': [], 'competing_required_gaps': {'秦皮': ['sand_crystal'], '乌梅': ['epidermis_cell']}, 'supporting_gaps': {'大青叶': ['epidermis_cell']}, 'unmapped': [], 'would_be_ready': False, 'production_authorization_changed': False, 'limit': '未查supporting不单独判否；required观察完成不等于排除竞争候选，仍需独立风险验证'}（不提升生产确认资格）
- 上述未解决引用不等于全部候选缺口清零；完整required/supporting状态见candidate_audit。
- 全部证据及采信边界：final_evidence.json；结论与依据：final_assessment.json。

B 规则分为 required 基础分加 supporting 佐证分，再扣分离区间反证。逐项反证见 agent_verdict.json 的 score_components。

| 候选 | B规则分 | required 基础分 | supporting 加分 | 分离区间反证扣分 | 融合排序分（非正确率） | 支持判据数 | 必需判据缺口数 |
|---|---|---|---|---|---|---|---|
| 滑石 | 0.0 | 0.0 | 0 | 0.0 | 0.215058 | 0 | 1 |
| 大青叶 | 0.0 | 0.0 | 0 | 0.0 | 0.17782 | 0 | 1 |
| 秦皮 | 0.0 | 0.0 | 0 | 0.0 | 0.055361 | 0 | 1 |
| 穿心莲 | 0.0 | 0.0 | 0 | 0.0 | 0.009746 | 0 | 1 |
| 乌梅 | 0.0 | 0.0 | 0 | 0.0 | 0.008448 | 0 | 1 |
| 雄黄 | 0.0 | 0.0 | 0 | 0.0 | 0.005004 | 0 | 1 |
