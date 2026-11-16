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

当前默认是 fast 路径：

```text
detect_fov_initial（YOLO）
→ classify_target（CNN batch 分类全部 YOLO crop）
→ required feature 覆盖检查
   ├─ 有：calibrate_fov → segment_instance → measure_instances
   └─ 无：一次 hybrid_scan fallback（3 feature / 12 locate / full 4 tile）
→ 原规则 / A-B 融合
```

非 fast 兼容路径：

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
- **propose_regions（非 fast 兼容）**：本地把原图切成固定 ROI 网格并计算线纹理分，不调用模型；fast 默认不跑。
- **roi_screen_batch（非 fast 兼容）**：多视野 contact sheet 并发筛选，默认 `screen_concurrency=3`；每个视野返回 top ROI，不返回框；fast 默认不跑，`screen_max_rois=4` 只截断输出。
- **locate_feature**：GLM 在指定 ROI 找指定特征候选框；fast 只用于 missing required feature 的 fallback；同一 `(image, ROI)` 已有 locate 时不再因不同 feature 重复执行。
- **目标条件提示**：`feature_profiles` 按 `feature_id` 提供该特征的正例/负例描述，附加到 screen、locate 和 classify 提示；taxonomy 为 14 类提供默认 profile，配置只做覆盖；不再使用针晶专用 hint 字段。
- **locate 允许为空**：`locate_feature` 提示词明确“没有目标必须返回 objects=[]”，禁止用背景纹理/色差/模糊边缘凑框；空结果不裁 crop，不伪造证据。
- **`hybrid_scan` fallback（fast）**：仅对 missing required feature 创建一次；3 个 feature、总 locate cap 12，每个 feature 最多 4 tile；full 策略 4 tile 全并发，任一有 crop 合并，全空即弃。非 fast 时仍可作为兼容扫描。
- **fast_yolo_stage2**：开启时每图先走 YOLO detect，所有 YOLO crop 进入 CNN batch；在每图 crop 未全部分类前，不创建 propose/screen/普通 hybrid 目标；若该样本的整组视野仍缺某个 A 候选 required feature，才创建一次至多 `fast_fallback_max_features` 的 GLM hybrid 兜底。YOLO 框全部保留，CNN 高置信直接采用，低置信仍由 classify_target 回退 GLM。
- **GLM fallback tile policy**：`feature_scan_tile_policy=full` 时一次把 4 个 tile 全部并发提交，任一找到 crop 就合并继续；全部为空则立即返回空、不重试。`prescreen` 则先扫 pilot，pilot 空即停。`fast_fallback_max_locate_calls` 限制一次 hybrid 兜底的总 locate 次数。
- `feature_scan` / `needle_scan` 是兼容工具；fast 模式主要用 `hybrid_scan` 做 missing required feature fallback，非 fast 模式才按旧模板使用 hybrid/feature scan。
- **feature_profiles**：以 `feature_taxonomy.json` 为全集生成默认 profile；针晶的负例只属于针晶 profile，不会污染其他特征。
- **negative_feature_bank.json**：按 P0 test 错误归因构建的 target-conditioned 负例特征库；淀粉粒因漏检高且类别混淆为 0，不作为针晶硬负例。
- **needle_audit / feature_audit**：workflow 结束后离线读取已有 crop/classification/mask，生成通用 `feature_audit/feature_audit.json`；针晶兼容名仍写 `needle_audit/`；只做非阻断审计和实例拆分统计。
- **image_exemplar_audit**：读取 `feature_exemplar_bank.json`（14 类 target profile，每类正/负示例），把目标 crop 与正/负示例拼成 contact sheet，由 VLM 输出 `target_score`（针晶兼容 `needle_score`）/ `negative_margin`；当前只审计 `det_score>=0.8` 且未分类的 crop，最多 6 个；只写 `image_audit` 工件，不阻断高分 mask/measure。
- **classify_target**：先用 14 类 ResNet18 对全部待分类 crop 做一次 batch 推理；`top1_prob>=cnn_threshold` 且熵低且不是 `other` 时直接写 `source=cnn`，否则回退 GLM；required crop 的 GLM 结果仍会做 YOLO crop 复核。
- **CNN 特征分类器**：权重与 meta 在 `分类器设计与权重/cls_runs/abl_320_lb_e40_ls01_mix01_s20261031/`，使用 `.venv/yolo`（有 torch/torchvision）；只做 14 类特征归类，不判药材、不替代 locate、不生成 mask。分类器没有 `unreadable` 类，低置信/`other` 必须回退 GLM。
- **identity_pending**：存在未分类的低分 `identity_required` crop 时，规划器必须先 classify 最高 `det_score` crop，不能继续 locate，也不能直接暂停或 review；classify 失败/不可判会留下操作记录，不会无限重试。
- **locate_override**：`det_score >= glm_locate_threshold=0.8` 的 crop 可绕过 classify 进入诊断性 IGM/测量；低于 0.8 不享受该绕过。
- **locate_expand / min size**：GLM tight box 先扩 1.5 倍；观察 ROI 至少 `locate_min_roi_side=256`，tight box 至少 `locate_min_box_side=90`（围绕中心扩展，不直接丢弃小目标），减少 50–70px 小框和过度放大模糊。
- **box_quality**：每个 crop 记录 `box_source`、`box_quality`、`box_quality_flags`、`box_quality_metrics`；`blank` 只阻断低分分割，不阻断 stage2/CNN 分类；`det_score>=glm_locate_threshold` 的高分 crop 即使 `blank` 也保留后续路线（高分解锁优先）；`low_detail` 只标记 review；同一特征的 GLM 框若与 YOLO 框 IoU>=0.5，保留 YOLO、跳过重复 GLM crop；`blank/low_detail` 可触发一次基于 `loose_box` 的 `locate:roi:boxq:*` 重定位。阈值刻意宽松，避免单图所有框被清空。
- **样本级特征覆盖**：`coverage` 同时统计 `useful` / `missing_useful` 和 `required_features` / `required_all` / `missing_required`；fast 模式把同一样本各视野的有效分类取并集。任一视野检出某类特征，该类即覆盖，其他视野无需重复补扫。逐图覆盖仍用于审计，正式判药继续按原证据规则执行。高分 crop 也提供非阻断 classify，不再长期停留在 `high_score_unclassified`。
- **agent_driven_coverage**：开启时 planner 从全部合法动作自行选择；`coverage`/`required_features`/`missing_required`/`remaining_seconds` 只作为观察和规划目标，不再由运行时强制“先 detect、再 propose、再 screen、再 locate”。仅当 planner 失效时用 required coverage 做兜底。
- **检测/标定解耦**：`detect_fov_initial` 先完成 YOLO；自动 scalebar 失败只置 `px_per_um=None`，保留检测框和 crop，后续可 `calibrate_fov`，不再把整个 detect 操作回滚为失败。
- **segment_instance / measure_instances**：仍受真实 crop、身份、标定、IGM 质量和用途资格约束。

当前模板：

```text
max_parallel = 2              # 仅独立观察并发，不是 screen/locate 并发
screen_top_k = 2
screen_max_rois = 4            # 只截断 screen 输出；不减少 192 个输入候选
fast_yolo_stage2 = true
fast_fallback_max_features = 3
fast_fallback_max_locate_calls = 12
feature_scan_tile_policy = "full"
screen_concurrency = 3
locate_attempts_per_view = 2
locate_attempt_reserve = 2
initialization_reserve = 1
glm_identity_threshold = 0.8
glm_locate_threshold = 0.8
locate_expand = 1.5
locate_min_box_side = 90
locate_min_roi_side = 256
box_quality_enabled = true
box_quality_retry = true
agent_driven_coverage = true
identity_min_confidence = medium
needle_audit = true
needle_scan_enabled = false
needle_scan_tile_size = 1024
needle_scan_overlap = 0.15
needle_scan_max_tiles = 9
needle_scan_concurrency = 2
feature_scan_enabled = false
feature_scan_features = []
feature_scan_from_candidates = true
feature_scan_max_features = 2
feature_scan_tile_size = 1024
feature_scan_overlap = 0.15
feature_scan_max_tiles = 4
feature_scan_concurrency = 4
hybrid_scan_enabled = true
hybrid_scan_max_features = 2   # 非 fast 兼容路径；fast fallback 用 fast_fallback_max_features
feature_audit = true
feature_audit_features = 14 类 taxonomy（默认全部）
image_exemplar_audit = true
image_exemplar_audit_min_det_score = 0.8
image_exemplar_audit_max_crops = 6
image_exemplar_audit_skip_classified = true
image_exemplar_bank = "TCM_Fusion/config/feature_exemplar_bank.json"
image_exemplar_positive = 4
image_exemplar_negative = 4
image_exemplar_seconds = 60
cnn_weights = "分类器设计与权重/cls_runs/abl_320_lb_e40_ls01_mix01_s20261031/best.pt"
cnn_meta = "分类器设计与权重/cls_runs/abl_320_lb_e40_ls01_mix01_s20261031/model_meta.json"
cnn_threshold = 0.75
cnn_entropy_threshold = 0.8
cnn_batch_size = 16
cnn_other_fallback = true
candidate_scope = open_strict
a_weight / b_weight = 0.6 / 0.4
min_support_features = 2
```

动态预算：

```text
max_initializations = N + initialization_reserve
max_locate_attempts = max(4, locate_attempts_per_view * N + locate_attempt_reserve)
```

固定安全上限：`max_planning=60`、`max_observations=40`、`max_api_calls=200`、`max_resource_units=400`。

## 当前边界

- `roi_screen_batch` 已支持多视野并发；`classify_target` 对外仍是单 crop 动作，但首次调用会用 CNN 对全部待分类 crop 做一次 batch 缓存；尚未实现独立的 `classify_batch` 工具。
- YOLO 仍是逐图执行，默认 `device=cpu`；尚未实现批量 YOLO/GPU 自动切换。
- `locate_feature` 仍是逐 ROI 动作；只做了同 ROI 去重，没有 locate 并发。
- fast fallback 使用 `full` 4 tile 全并发；4 tile 全空即弃，不重试，不是零漏检保证。
- 自动标定失败不会丢 YOLO 框，但该视野仍需显式 `calibrate_fov` 才可能有 μm。
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
needle_audit/needle_audit.json
feature_audit/feature_audit.json
image_audit/*/audit.json
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
