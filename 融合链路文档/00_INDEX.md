# 00_INDEX｜融合链路文档索引

同步版本 v8.1｜2026-11-18｜以当前 `TCM_Fusion` 代码为准；“待实施方案”单独标注，不代表运行能力。历史运行和旧配置按[修改记录](../修改记录.md)当时的版本解释。

## 第一次阅读

- [项目文件地图：按文件粒度理解整个项目](../../项目文件地图.md)

1. **[00 新手快速入门](00_新手快速入门.md)** ← 先读这一页
2. [01 架构总览](01_架构总览.md)
3. [02 数据流与状态机](02_数据流与状态机.md)
4. [03 接口契约](03_接口契约.md)
5. [04 工具清单](04_工具清单.md)
6. [05 决策协议](05_决策协议.md)
7. [06 硬约束与回退](06_硬约束与回退.md)
8. [07 验收与运行](07_验收与运行.md)

深入某类特征/误检时再读：

- [08 针晶个体与组织干扰改进指导](08_针晶个体与组织干扰_改进指导.md)
- [09 DAG 与旧短计划边界](09_DAG与旧短计划边界.md)
- [10 ReAct 主导的证据优化与语义 DAG 实施部署方案](10_ReAct主导的证据优化与语义DAG实施部署方案.md)
- [11 D1交付与验收](11_D1交付与验收.md)：首次 GLM、实例账本、语义尝试键及本次离线基线。
- [12 D2交付与验收](12_D2交付与验收.md)：区分问题、资格投影、稳定引用、关联歧义和共享查询。
- [13 调查建议与硬约束分离](13_调查建议与硬约束分离.md)：最新调查自主性、停止投影与无收益防循环。
- [14 D3交付与验收](14_D3交付与验收.md)：上下文大小审计、反思补丁/后验、默认关闭的串行语义执行。
- [15 调查效率与反思收益优化](15_调查效率与反思收益优化.md)：稳定问题、分轴进展、上下文去重、模型授权的独立观察并发与计时；不代表完整D4。

- [16 主选择协议与规划次数分析](16_主选择协议与规划次数分析.md)：依据真实14轮轨迹分析，A/G/Q同级选择、实际送模菜单审计及边界。

- [17 候选审查完整性与效率边界](17_候选审查完整性与效率边界.md)：最新候选账本、跨对象补查、停止一致性与交付限制。

待实施方案（不代表当前代码已支持）：

- [10 ReAct 主导的证据优化与语义 DAG 实施部署方案](10_ReAct主导的证据优化与语义DAG实施部署方案.md) v1.6 已前置 D5 历史基线并完成 D4.0 跨视野初检乱序回放。物理并行与线上效果对照见 §九后续验收步骤。

运行操作：

- [多图测试运行方案](../多图测试运行方案.md)
- [单图测试操作手册](../单图测试操作手册.md)
- [当前实现状态](../当前实现状态.md)

## 一句话架构

```text
A 给候选
B 给证据
C 决定查什么
```

```text
原图（N 张）
→ A 每图五折 Top-K
→ ReAct 根据上下文选择合法调查动作
  ├─ 出框：YOLO 初检 → crop分类 → 必要时GLM定位补救
  ├─ 原图：GLM presence_check / scene_check
  └─ 选定实例：segment → 合格时同对象自动measure
→ 更新证据与 required/竞争候选差异，再由ReAct决定继续或停止
→ 原规则 decide
→ A/B 融合
→ final_assessment
```

## 当前关键参数

```text
execution = active
agent_policy = adaptive_v2
planner = llm
transport = mcp

fast_yolo_stage2 = true
auto_prepare_stage = false
auto_required_measurement = false
auto_measure_after_segment = true
suppress_repeated_measurement = true
candidate_group_verification = true

max_seconds = 2400
fast_fallback_max_features = 2
fast_fallback_max_locate_calls = 6
feature_scan_tile_policy = full
feature_scan_max_tiles = 4
feature_scan_concurrency = 2

supporting_probe_total = 6
supporting_probe_per_candidate = 2
```

## 术语速查

| 术语 | 当前含义 |
|---|---|
| A | 每张图独立五折推理，每折各自 Top-K；默认每折 3 个 |
| B | YOLO 检测、GLM locate/classify、CNN stage2、标定、分割、测量及原规则裁决 |
| C | LangGraph 编排、MCP/本地工具、证据版本与预算控制 |
| fast_yolo_stage2 | 当前默认：YOLO 先检 + CNN batch 分类；不默认 propose/screen |
| required | 候选成立必须满足的判据 |
| supporting | 辅助判据；可增加 support_count，但不是必要唯一条件 |
| support_count | 正式支持的不同 feature 类数；同一 feature 多个实例仍算 1 类 |
| min_support_features | D8 门槛，默认 2 类；只有 1 类通常降级 |
| D8b | 单类证据放行：confidence 达标且与 top2 已区分，仍强制复核 |
| provisional | 基于有效 classification 的暂定排序；不进入正式 support_count |
| pending / DETECTED_PENDING | 有线索但身份/质量/定量未完成，不是阳性也不是阴性 |
| UNCAL | 有特征但缺少可信定量 |
| NO_REF | 有特征但候选缺少参考量程 |
| identity_pending | 未分类的 `identity_required` crop；必须先 classify |
| shared ledger | 同一 image+feature 的观察共享；候选只做消费者 |
| stop_decision | `can_finish` / `can_review` / `blockers[]` / `recommended_exit` |
| fallback | 整组视野仍缺 required feature 时的有限 GLM 补扫 |
| full / prescreen | fallback tile 策略：全 4 tile 并发 / 先 pilot 后决定 |

## 当前边界

- 图片数量不固定为 5；由 manifest 决定，至少 1 张。
- 药材种类来自 Plan A；YOLO 输出显微特征类别和框。
- P0 在 `adaptive_v2` 只是建议；身份完整性门是数据完整性约束，不是 P0 强制路线。
- `classify_crops_batch` 已提供逐实例批量 CNN；YOLO batch/GPU 尚未实现。
- `roi_screen_batch` 已随标准路径移除，不再注册。
- fallback 是 required feature 的有限补扫，不是零漏检保证。
- 高分 `det_score` 只影响定位优先级，不能绕过身份门。
- 无合格身份、mask 或标定时，可能没有 μm。
- 普通存在性未检出只表示当前扫描范围未知/未见，不等于全样本不存在。
- 最新回归466项：465通过，1历史参考case缺失跳过；原B三组59/21/125断言通过。未调用真实视觉API，D3详情见14。

- [18 主解释与竞争解释公平取证](18_主解释与竞争解释公平取证.md)：前置入口可达性、逐折先验分歧、受限格式恢复与唯一交付；充分性策略尚在影子验证。

- [19 传输诊断、反思精简与有界调度方案](19_传输诊断反思精简与有界调度方案.md)：待实施；断连边界、当前Reflection机制、上下文/并行交接的具体修改包。

- [20 判据分值解释与反思策略增量方案](20_判据分值解释与反思策略增量方案.md)：上下文/Reflection已接入；B原评分不变，稳定问题、同轮策略响应及用途后验见§9，独立调度后置。

- [21 TCM_Fusion 组件重组方案](21_TCM_Fusion组件重组方案.md)记录修改范围、依赖关系、兼容项和验收。
