# 00_INDEX｜融合链路文档索引

同步版本 v6.0｜2026-11-10｜以 TCM_Fusion 当前代码为准。历史设计、试跑和指标按[修改记录](../修改记录.md)中的当时配置解释。

## 阅读顺序

1. [01 架构](01_架构总览.md) → [02 数据流](02_数据流与状态机.md) → [03 契约](03_接口契约.md)
2. [04 工具](04_工具清单.md) → [05 决策](05_决策协议.md) → [06 硬约束](06_硬约束与回退.md)
3. [07 运行与验收](07_验收与运行.md) → [多图接口](../多图测试运行方案.md) → [单图手册](../单图测试操作手册.md)
4. [当前实现状态](../当前实现状态.md) → 已验证/未验证边界与最近运行
4. [08 针晶个体与组织干扰改进指导](08_针晶个体与组织干扰_改进指导.md)：具体特征与后续修改路线

## 术语

| 术语 | 当前含义 |
|---|---|
| A | 每张图独立五折推理，每折各自 Top-K；默认每折 3 个 |
| B | YOLO 检测、GLM locate/classify、CNN stage2、标定、分割、测量及原规则裁决 |
| C | LangGraph 编排、MCP/本地工具、证据版本与预算控制 |
| fast_yolo_stage2 | 当前默认：YOLO 先检 + CNN batch 分类；不默认 propose/screen |
| required feature | A 候选判据需要的 feature；fast 只在缺失时触发 GLM fallback |
| locate | GLM 在指定 ROI 中找指定特征候选框 |
| classify | CNN stage2 先判，低置信由 GLM 兜底；YOLO crop 复核只是辅助 |
| identity_pending | 未分类的低分 `identity_required` crop；必须先 classify |
| screen batch | 多视野 `roi_screen_batch`，worker 并发、Session 一次提交；非 fast 默认 |
| 候选兴趣 | 逐折榜单映射、去重后供补查的药材集合；不裁剪 B 规则候选 |
| fallback | 同一样本的整组视野仍缺某个 required feature 时，按视野提供 GLM hybrid 补扫 |
| full / prescreen | fallback tile 策略：全 4 tile 并发 / 先 pilot 后决定 |
| provisional | 用中间 classification 生成暂定排序；1.0/0.5/0.25/0，正式判定不变 |
| required_states | 候选级 required 状态；DETECTED_PENDING/PARTIAL_MATCH 不等于已满足 |

## 当前主链

```text
原图（任意 N 张）
→ A 逐图五折 Top-K
→ fast：YOLO 初检
→ CNN stage2 batch 分类全部 YOLO crop
→ required feature 覆盖
   ├─ 有：标定 / 分割 / 测量
   └─ 无：一次 GLM hybrid fallback
→ 原规则 decide
→ A/B 融合
→ final_assessment
```

非 fast 兼容链：

```text
detect → propose_regions → roi_screen → locate_feature
→ classify_crops_batch / classify_target → segment / measure
```

## 边界

- 图片数量不固定为 5；由 manifest 决定，至少 1 张。
- 药材种类来自 Plan A；YOLO 输出显微特征类别和框。
- P0 在 `adaptive_v2` 只是建议；身份完整性门是数据完整性约束，不是 P0 强制路线。
- `roi_screen_batch` 已实现；fast 默认不走。
- `classify_crops_batch` 已提供逐实例批量 CNN；YOLO batch/GPU 尚未实现。
- fallback 是 required feature 的有限补扫，不是零漏检保证。
- 高分 `det_score>=0.8` 可绕过质量门；低分 blank 仍阻断 segment。
- 无合格身份、mask 或标定时，可能没有 μm。

维护说明｜CODE-158｜2026-11-13：整理当前实现状态、batch/context/测量门/supporting/locate 文档与 335 项回归口径。
