# 自主B工具与A融合

- 裁决：insufficient_evidence
- 交付输出：tentative / 龙胆
- 内部复核标记：True；交付层 review_required=True
- 强观察：2 条；备选：4 个
- 停止原因：runtime_request_review
- 证据版本：b98698de2c70176574968cf58c0af69a811362a20cfde4448fab1ec167878746
- 原图及原B目录未覆盖；阶段失败/资源明细见agent_trace.json。

## 收尾综合研判

现有证据暂倾向：龙胆；尚不能确认，需要复核。

当前倾向为龙胆。A先验在五个视角的多折中高度一致支持龙胆（a_prior:33、a_prior:34、a_prior:36、a_prior:38、a_prior:40、a_prior:53、a_prior:54、a_prior:57，多折概率>0.86）；B通道中龙胆的支持特征外皮层/内皮层细胞符合（criterion:27，NO_REF态），observation:59也在独立视野确认长短细胞交互排列，与龙胆外皮层特征形态相容（置信中等）。桔梗虽获criterion:12+observation:58（联结乳管阳性），但其A概率仅约0.13且先验折内多为第二位，与龙胆先验强度差距大，更可能是共享分泌组织导致的同源混淆。关键冲突点在于龙胆P0必需的针晶（criterion:26）判为NOT_DETECTED，但classification:65-69显示各候选crop均因严重虚焦被判'图像不可判'，即该视野不可读，未检出不能视为真实缺失；正式verdict为insufficient_evidence，融合分数为未校准先验加权，不构成正确率。缺口：需重新取清晰焦面确认草酸钙针晶，并测量乳管宽度（桔梗14-25μm vs 柴胡8-25μm）以在龙胆/桔梗/柴胡间裁决；同一视野多折非独立票。

- 研判来源：model；模型失败原因：None
- 支持证据：['criterion:27', 'observation:59', 'a_prior:33', 'a_prior:34', 'a_prior:36', 'a_prior:38', 'a_prior:40', 'a_prior:53', 'a_prior:54', 'a_prior:57']
- 反证：[]
- 未解决：['criterion:26']
- 全部证据及采信边界：final_evidence.json；结论与依据：final_assessment.json。

| 候选 | B规则分 | 融合排序分（非正确率） | 支持判据数 | 必需判据缺口数 |
|---|---|---|---|---|
| 龙胆 | 0.0 | 0.383256 | 1 | 1 |
| 桔梗 | 0.425 | 0.250481 | 1 | 1 |
| 麦芽 | 0.459 | 0.183917 | 1 | 0 |
| 乌梅 | 0.436 | 0.174453 | 1 | 0 |
| 秦皮 | 0.0 | 0.082235 | 0 | 1 |
| 石膏 | 0.0 | 0.006632 | 0 | 1 |
| 菟丝子 | 0.0 | 0.001011 | 0 | 1 |
| 黄芪 | 0.0 | 0.000541 | 0 | 1 |
| 苦参 | 0.0 | 0.000471 | 0 | 1 |
