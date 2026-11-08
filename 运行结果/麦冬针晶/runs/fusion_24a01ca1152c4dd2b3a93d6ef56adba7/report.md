# 自主B工具与A融合

- 裁决：insufficient_evidence
- 交付输出：tentative / 知母
- 内部复核标记：True；交付层 review_required=True
- 强观察：0 条；备选：4 个
- 停止原因：planner_request_review
- 证据版本：00453ff8d57427d7f88485047dea8f230b7bad50c5143c9695511f4a5d641ca0
- 原图及原B目录未覆盖；阶段失败/资源明细见agent_trace.json。

## 收尾综合研判

现有证据暂倾向：知母；仅有A先验，缺少独立显微支持，不构成鉴定。

正式判定为insufficient_evidence。A先验在全部5个视图25个折中一致将知母排第一（a_prior:24–a_prior:48，约0.975–0.999），但所有A概率仅为未校准先验且同源视图不构成独立投票。B通道证据全部为阴性：知母主判P0特征针晶NOT_DETECTED（criterion:21），麦冬针晶（criterion:22、criterion:23）与G4针晶（criterion:17）亦未检出，support_count=0，B置信0。分类产物classification:54将view_03疑似结构判为'纤维'（medium置信，second为'图像不可判'），与针晶判据不符且未量化尺寸，不能据calibration:49–53的标定换算物理尺寸。hypothesis:84/89/90仍处investigating，针晶三分候选（知母26–110、G4 5–32、G2 20–144μm feret_max）未获分割测量数据，判别悬置。当前倾向知母（仅先验支撑），但缺少针晶形态与尺寸确认，维持证据不足结论。

- 研判来源：model；模型失败原因：None
- 支持证据：['a_prior:24', 'a_prior:25', 'a_prior:26', 'a_prior:27', 'a_prior:28', 'a_prior:29', 'a_prior:30', 'a_prior:31', 'a_prior:32', 'a_prior:33', 'a_prior:34', 'a_prior:35', 'a_prior:36', 'a_prior:37', 'a_prior:38', 'a_prior:39', 'a_prior:40', 'a_prior:41', 'a_prior:42', 'a_prior:43', 'a_prior:44', 'a_prior:45', 'a_prior:46', 'a_prior:47', 'a_prior:48']
- 反证：[]
- 未解决：['criterion:21', 'criterion:22', 'criterion:23', 'classification:54', 'hypothesis:84', 'hypothesis:89', 'hypothesis:90']
- 全部证据及采信边界：final_evidence.json；结论与依据：final_assessment.json。

| 候选 | B规则分 | 融合排序分（非正确率） | 支持判据数 | 必需判据缺口数 |
|---|---|---|---|---|
| 知母 | 0.0 | 0.596794 | 0 | 1 |
| 黄芪 | 0.0 | 0.000329 | 0 | 1 |
| 白术 / 苍术 | 0.0 | 0.000208 | 0 | 1 |
| 益母草 | 0.0 | 0.000165 | 0 | 1 |
| 苦参 | 0.0 | 0.000143 | 0 | 1 |
| 菟丝子 | 0.0 | 0.00014 | 0 | 1 |
| 甘草 | 0.0 | 0.000131 | 0 | 1 |
| 雄黄 | 0.0 | 0.000112 | 0 | 1 |
| 白头翁 | 0.0 | 8.5e-05 | 0 | 1 |
| 大青叶 | 0.0 | 6e-05 | 0 | 1 |
| 诃子 | 0.0 | 5e-05 | 0 | 1 |
