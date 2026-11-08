# TCM_Fusion｜真实五折CNN＋VLM Agent融合链路

现役入口：

```text
cli.main
→ pipeline
→ workflow_entry
→ workflow_runtime / workflow_session
→ LangGraph ReAct
→ MCP / 本地工具
→ YOLO、GLM、分割、标定、测量及原规则
→ A/B 融合与 final_assessment
```

A 每张图独立五折推理，每折各自 Top-3；不取平均榜替代逐折输入。药材种类由 Plan A 给出；YOLO 给的是显微特征类别和框，不是药材结论。

## 一条命令跑任意数量多图

从项目根运行。先做本地预检：

```powershell
.venv\Scripts\python.exe -X utf8 -m TCM_Fusion.cli.main multi-view `
  --source-dir <含input_proof.json的目录> `
  --a-python D:/Anaconda/python.exe `
  --dry-run --json
```

真实运行：

```powershell
.venv\Scripts\python.exe -X utf8 -m TCM_Fusion.cli.main multi-view `
  --source-dir <含input_proof.json的目录> `
  --a-python D:/Anaconda/python.exe `
  --name 我的测试 `
  --json
```

`--name` 只决定 case 目录名；配置里的 `sample_id` 会另做 ASCII 清洗。`five-view` 仍保留为兼容别名，但不再强制 5 张图。

接口行为：

- 图片数由 `input_proof.json` 或 `--manifest` 决定，至少 1 张；
- 每张图必须有 SHA256，路径必须在 source-dir 内，路径和内容都不能重复；
- 新 case、空 A 缓存、中性 `view_01...` 文件名；
- 必须显式提供 `--a-python` 或 `TCM_A_PYTHON`，不自动使用 Agent venv；
- 参考标签和人工框不进入规划配置；
- dry-run 只准备、复制、生成配置和 doctor，不调用模型。

详细输入格式见 [多图测试运行方案](多图测试运行方案.md)。

## 当前执行设计

```text
detect_fov_initial
→ propose_regions
→ roi_screen_batch
→ locate_feature
→ classify_target
→ segment_instance
→ measure_instances
```

关键行为：

- **YOLO 初检**：全视野特征检测；输出组织/特征框，不是药材类别。
- **propose_regions**：本地把原图切成固定 ROI 网格并计算线纹理分，不调用模型。
- **roi_screen_batch**：多视野 contact sheet 并发筛选，默认 `screen_concurrency=3`；每个视野返回 top ROI，不返回框。
- **locate_feature**：GLM 在指定 ROI 找指定特征候选框；同一 `(image, ROI)` 已有 locate 时不再因不同 feature 重复执行。
- **classify_target**：GLM 对 locate crop 做身份复核；required crop 还会做一次 YOLO crop 复核。YOLO 未命中时保留 GLM 结论。
- **identity_pending**：存在未分类的低分 `identity_required` crop 时，规划器必须先 classify 最高 `det_score` crop，不能继续 locate，也不能直接暂停或 review；classify 失败/不可判会留下操作记录，不会无限重试。
- **locate_override**：`det_score >= glm_locate_threshold=0.8` 的 crop 可绕过 classify 进入诊断性 IGM/测量；低于 0.8 不享受该绕过。
- **locate_expand=1.5**：GLM tight box 先扩框再裁 crop。
- **segment_instance / measure_instances**：仍受真实 crop、身份、标定、IGM 质量和用途资格约束。

当前模板：

```text
max_parallel = 2              # 仅独立观察并发，不是 screen/locate 并发
screen_top_k = 2
screen_concurrency = 3
locate_attempts_per_view = 2
locate_attempt_reserve = 2
initialization_reserve = 1
glm_identity_threshold = 0.8
glm_locate_threshold = 0.8
locate_expand = 1.5
identity_min_confidence = medium
candidate_scope = open_strict
a_weight / b_weight = 0.6 / 0.4
min_support_features = 2
```

动态预算：

```text
max_initializations = N + initialization_reserve
max_locate_attempts = max(4, locate_attempts_per_view * N + locate_attempt_reserve)
```

固定安全上限仍保留：`max_planning=32`、`max_observations=10`、`max_api_calls=60`、`max_resource_units=220`。

## 当前边界

- `roi_screen_batch` 已支持多视野并发；`classify_target` 仍是单 crop 串行，尚未实现 `classify_batch`。
- YOLO 仍是逐图执行，默认 `device=cpu`；尚未实现批量 YOLO/GPU 自动切换。
- `locate_feature` 仍是逐 ROI 动作；只做了同 ROI 去重，没有 locate 并发。
- 图片数不受 5 张限制，但总规划、总观察、总 API 和资源上限仍是固定安全阈值。
- 分类通过不等于针晶通过；`纤维`、`花粉粒`、`图像不可判` 都不会自动变成针晶。
- 没有合格 identity、mask 或标定时，可能没有 μm；这是约束，不是程序异常。

## 保留的组件

| 位置 | 职责 |
|---|---|
| `cli/main.py` | run、doctor、selftest、multi-view；five-view 为兼容别名 |
| `lib/multi_view.py`、`config/multi_view.json` | 任意数量原图接口及规范模板 |
| `lib/workflow_*`、`react_graph`、`tool_*`、`agent_tools`、`mcp_bridge` | 主动流程、真实 MCP、前置条件、模型规划与反馈 |
| `cli/*_worker`、`lib/worker_bootstrap` | 隔离解释器、API、YOLO、分类和 B 阶段执行 |
| `lib/final_assessment`、`output_projection` | 调查停止后综合判断；证据不足不强制确认 |
| `skills`、`requirements-agent*.txt` | 注册技能与锁定依赖 |
| `yolo-results` | 当前 best.pt、class_map 及训练/评测来源指标 |
| `tests`、`tests/offline` | 主链契约与离线功能测试，统一由 selftest 递归发现 |

## 在线与离线分开

```powershell
.venv\Scripts\python.exe -X utf8 -m TCM_Fusion.cli.main selftest
.venv\Scripts\python.exe -X utf8 -m TCM_Fusion.cli.offline --help
```

离线入口保留 LangMem 提示优化/验证/发布、任务经验发布、数据划分、真值隔离评估、质量回放、校准门和工作流回放。离线功能不参与当前多图在线推理，但不能删除；在线仍可显式加载通过验证的 `prompt_release`，默认模板不加载未知提示。

旧 demo、无 workflow 的 active 运行器和 retired 入口不再提供命令；历史录制只作为 `tests/fixtures/recorded_pipeline.json` 的契约样本。

## 输出与审计

主要工件：

```text
job.json
a_prediction.json
agent_trace.json
workflow/operations.json
workflow/ops/
agent_verdict.json
fusion.json
output.json
final_evidence.json
final_assessment.json
report.md
```

`output.json` 始终交付 top 候选、备选、强观察、检出框和缺失证据；`review_required` 如实保留。`fusion.json` 的 `fused_score` 是排序分，不是校准正确率；正式确认仍由原规则授权。

历史 `case_9cc59640` 只作参考，不能宣称验证了其后的代码。新测试应使用新 case、空 A 缓存和中性的输入副本。

详见 [修改记录](修改记录.md)、[融合链路文档索引](融合链路文档/00_INDEX.md)、[多图接口](多图测试运行方案.md)、[单图测试](单图测试操作手册.md)、[针晶个体与组织干扰改进指导](融合链路文档/08_针晶个体与组织干扰_改进指导.md)。
