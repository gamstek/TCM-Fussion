# 00_INDEX｜融合链路文档索引

同步版本 v5.0｜2026-11-08｜以 TCM_Fusion 当前代码为准。历史设计、试跑和指标按[修改记录](../修改记录.md)中的当时配置解释。

## 阅读顺序

1. [01 架构](01_架构总览.md) → [02 数据流](02_数据流与状态机.md) → [03 契约](03_接口契约.md)
2. [04 工具](04_工具清单.md) → [05 决策](05_决策协议.md) → [06 硬约束](06_硬约束与回退.md)
3. [07 运行与验收](07_验收与运行.md) → [多图接口](../多图测试运行方案.md) → [单图手册](../单图测试操作手册.md)
4. [08 针晶个体与组织干扰改进指导](08_针晶个体与组织干扰_改进指导.md)：按简单到复杂、低风险到高收益排列的后续修改路线

## 术语

| 术语 | 当前含义 |
|---|---|
| A | 每张图独立五折推理，每折各自 Top-K；默认每折 3 个 |
| B | YOLO 检测、GLM screen/locate/classify、标定、分割、测量及原规则裁决 |
| C | LangGraph 编排、MCP/本地工具、证据版本与预算控制 |
| locate | GLM 在指定 ROI 中找指定特征候选框 |
| classify | GLM 对 locate crop 做身份复核；YOLO crop 复核是辅助 |
| identity_pending | 未分类的低分 `identity_required` crop；必须先 classify |
| screen batch | 多视野 `roi_screen_batch`，worker 并发、Session 一次提交 |
| 候选兴趣 | 逐折榜单映射、去重后供补查的药材集合；不裁剪 B 规则候选 |
| 任务内 Reflection | 从本次动作反馈生成一次结构化提示卡 |
| 离线 Reflection / LangMem | 显式命令生成候选、开发集回放、验证与发布 |

## 当前主链

```text
原图（任意 N 张）
→ A 逐图五折 Top-K
→ YOLO 初检
→ 本地 ROI 提议
→ GLM screen（可批量并发）
→ GLM locate
→ GLM classify（身份门）
→ 分割 / 标定 / 测量
→ 原规则 decide
→ A/B 融合
→ final_assessment
```

## 边界

- 图片数量不固定为 5；由 manifest 决定，至少 1 张。
- 药材种类来自 Plan A；YOLO 输出显微特征类别和框。
- P0 在 `adaptive_v2` 只是建议；身份完整性门是数据完整性约束，不是 P0 强制路线。
- `roi_screen_batch` 已实现；`classify_batch` 和 YOLO batch/GPU 尚未实现。
- 无合格身份、mask 或标定时，可能没有 μm。

维护说明｜CODE-071｜2026-11-08：同步多图接口、批量 screen、身份门及 current 边界。历史归档不作为在线输入。
