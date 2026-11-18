# TCM_Fusion｜真实五折CNN＋VLM Agent融合链路

开发者从[源码导航](lib/README.md)、[回归导航](tests/README.md)和[组件重组方案](融合链路文档/21_TCM_Fusion组件重组方案.md)进入。

FAIR-INVESTIGATION-001（2026-11-18）：540项（539通过/1历史参考缺失跳过）、B59/21/125通过；新单图425.578秒，主/竞争均调查并产出17.274558μm，交付一致暂定白芍仍复核。统一鉴别问题卡，主解释与竞争支持分列；无crop时簇晶可关联真实YOLO入口。受限说明字段本地归一、结构错误至多纠正一次，交付跟随唯一研判候选。充分性策略仅shadow，原保守确认不放松；不强制调查顺序。见[18](融合链路文档/18_主解释与竞争解释公平取证.md)。

CANDIDATE-AUDIT-001（2026-11-17）：最终回归521项（520通过、1历史case缺失跳过）及原B59/21/125通过；候选审查优先于时延/token。五折并集及未映射、暂停候选的required/supporting逐项保留；独立补查不再被别的crop未分类阻塞。审查不完整时可review，交付不得升级为完整鉴别。旧245.828秒只属历史运行，不能证明同覆盖提效。见[17 候选审查完整性与效率边界](融合链路文档/17_候选审查完整性与效率边界.md)。

PLANNER-CHOICES-001（2026-11-17）：ReAct现将单动作、实例目标、独立观察组作为同级主选择，按本轮A/G/Q编号明确授权；保存实际送模菜单和模型选择模式，不强制并行或固定调查顺序。最新503项（502通过、1历史case缺失跳过）及原B59/21/125通过。详见[16 主选择协议与规划次数分析](融合链路文档/16_主选择协议与规划次数分析.md)。以下EFF/D3计数为历史口径。

EFF-001（2026-11-17）：按语义问题而非工具名跟踪Reflection，分开记录证据/处理能力/诊断及退出；去重判据上下文，补齐分段计时。模型可显式选两路独立presence问题并行；可选假设字段错误不废弃合法动作。最新回归485项（484通过、1历史case缺失跳过），原B59/21/125通过。详见[15 调查效率与反思收益优化](融合链路文档/15_调查效率与反思收益优化.md)；语义执行仍默认关闭，通用D4/D5未完成。

D3-LIVE-FIX-001（2026-11-17）：修复自动测量上下文与伪超时、收尾候选/引用不一致，补充反思拒绝与自动子阶段反馈；真实单图已产出合格尺寸，仍暂定白芍需复核。见[测试总结](运行结果/case_d3_fixed_single_20261117_192513/测试总结.md)。离线selftest与真实服务测试分别报告，不能混称零API验证。

D3-001（2026-11-17）：默认仍单步ReAct；紧凑上下文保留五折并集并记录字段字节数，Reflection在正常规划中提交补丁并回查效果。实验semantic_execution=false，显式开启且模型选择实例语义目标才逐节点晚绑定执行；不接受具体工具序列。交付见[14](融合链路文档/14_D3交付与验收.md)，D4/D5待实施。

2026-11-13｜PRESENCE-001：存在性补查增加同请求可选定位、原图映射、紧框/上下文裁图及影子质量状态，进入后续 Agent 上下文。没有核验集时不把新框直接升为身份或尺寸证据；旧 P1/C 观察保持兼容。详见 [接口契约：presence 可选定位扩展](融合链路文档/03_接口契约.md#presence-可选定位扩展presence-001) 与 [修改记录](修改记录.md)。

PRESENCE-002 系统核验：修复关闭/取消/源图异常后的终态与迟到采信、Windows 父 worker 延迟终止；后续失败路径修复后统一 383 项（1 skip）。真实 MCP/框架测试使用受控视觉响应，不代表外部模型识别准确率已验证；新框仍未自动进入正式实例核验。

现役入口：

短计划（`supporting_probes_batch`）、`supporting_plans_enabled`、`supporting_plan_max_probes`、`parallel_supporting_observations` 已移除；supporting 只保留逐动作预算。2026-11-17最新完整回归503项（502通过、1历史case缺失跳过），原B59/21/125断言通过。当前调查自主性、停止直接转复核及有界防空转以[13 调查建议与硬约束分离](融合链路文档/13_调查建议与硬约束分离.md)为准；不代表真实视觉API、准确率或总token收益已验证。详见[修改记录](修改记录.md)。

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

### stage2 模型接入（STAGE2-001，2026-11-16）

- 用户确认药材 A 仍使用 `MICROSCOPE-main/v3_resnet50_full`，本次只更新 crop 显微特征分类接口。
- `config/multi_view.json` 的 `cnn_weights/cnn_meta` 指向项目根 `分类器设计权重与CNN权重/cls_runs/abl_320_lb_e40_ls01_mix01_s20261031/`。采用原配置对应的单个 ResNet18、320 letterbox、14 类，不从测试集挑选其他消融模型或擅自启用三模型集成。
- 真实入口：`classify_crops_batch/classify_target → WorkflowBackend → cli.cnn_classifier_worker → lib.stage2_classifier → 目标检测.lib.cls_net`。输入为绑定原图的 crop；输出为逐 crop 的特征 Top-3、概率、熵和模型指纹。外层药材五折 Top-K 不经此模块。
- 训练预处理为 `cls_net.build_transforms`：letterbox 灰度 128、BILINEAR、等比缩放含小图放大。移除融合 worker 中灰度 114/BICUBIC/thumbnail 的重复实现。checkpoint 和 sidecar 的架构、类别顺序、尺寸、输入模式与归一化必须一致。
- 默认概率 ≥0.75、归一化熵 ≤0.8 且非 `other` 时沿用高置信身份路径；这不是已校准正确率。低置信/other 的批量结果保留 Top-K 并标记 `pending_review`，由 Agent 决定后续动作。模型加载失败单列 `failures`，不当作低置信或阴性。
- 同任务 CNN 缓存绑定权重、meta、预处理/适配代码、解释器和 crop 内容；旧无指纹缓存不采信。新任务仍使用自己的新目录。模型/元信息纳入任务身份，运行中变化拒绝混用。
- 本次已验证真实本地 CNN 与训练 API 对拍，不等于重跑检测、GLM、分割、测量和最终药材识别；详见 [修改记录](修改记录.md) 的 STAGE2-001。

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
→ classify_crops_batch / classify_target（CNN 分类 YOLO crop）
→ required feature 覆盖检查
   ├─ 有：calibrate_fov → segment_instance → measure_instances
   └─ 无：一次 fast fallback（2 feature / 6 locate / full 4 tile）
→ 原规则 / A-B 融合
```

关键行为：

- **YOLO 初检**：全视野特征检测；输出组织/特征框，不是药材类别。
- **locate_feature**：GLM 在指定 ROI 找指定特征候选框；fast 只用于 missing required feature 的 fallback；同一 image/feature 已有 done locate 时不再换 ROI 立即重试；ROI 已被同 feature 有效 crop 覆盖时不再生成 locate。
- **目标条件提示**：`feature_profiles` 按 `feature_id` 提供该特征的正例/负例描述，附加到 locate 和 classify 提示；taxonomy 为 14 类提供默认 profile，配置只做覆盖；不再使用针晶专用 hint 字段。
- **locate 允许为空**：`locate_feature` 提示词明确“没有目标必须返回 objects=[]”，禁止用背景纹理/色差/模糊边缘凑框；空结果不裁 crop，不伪造证据。
- **`hybrid_scan` fallback（fast）**：仅对 missing required feature 创建一次；2 个 feature、总 locate cap 6，每个 feature 最多 4 tile；full 策略 4 tile 全并发，任一有 crop 合并，全空即弃。legacy 兼容路径下仍可显式使用扫描工具。
- **fast（标准路径，无 false 开关）**：每图先走 YOLO detect，所有 YOLO crop 进入 CNN batch；在每图 crop 未全部分类前，不创建 propose/screen/普通 hybrid 目标；若该样本的整组视野仍缺某个 A 候选 required feature，才创建一次至多 `fast_fallback_max_features` 的 GLM hybrid 兜底。YOLO 框全部保留，CNN 高置信直接采用，低置信仍由 classify_target 回退 GLM；批量动作只登记 pending，不自动 GLM。`fast_yolo_stage2` 字段只为兼容历史配置保留，只允许 `true`，不再作为开关。
- **GLM fallback tile policy**：`feature_scan_tile_policy=full` 时一次把 4 个 tile 全部并发提交，任一找到 crop 就合并继续；全部为空则立即返回空、不重试。`prescreen` 则先扫 pilot，pilot 空即停。`fast_fallback_max_locate_calls` 限制一次 hybrid 兜底的总 locate 次数。
- `feature_scan` / `hybrid_scan` 是兜底扫描工具；标准路径只在 missing required feature 时按 `fast_recovery_selection` 创建，`hybrid_scan` 仅旧配置 `legacy_bundle` 使用。
- **feature_profiles**：以 `feature_taxonomy.json` 为全集生成默认 profile；针晶的负例只属于针晶 profile，不会污染其他特征。
- **negative_feature_bank.json**：按 P0 test 错误归因构建的 target-conditioned 负例特征库；淀粉粒因漏检高且类别混淆为 0，不作为针晶硬负例。
- **feature_audit**：workflow 结束后离线读取已有 crop/classification/mask，生成通用 `feature_audit/feature_audit.json`。
- **image_exemplar_audit**：读取 `feature_exemplar_bank.json`（14 类 target profile，每类正/负示例），把目标 crop 与正/负示例拼成 contact sheet，由 VLM 输出 `target_score`（针晶兼容 `needle_score`）/ `negative_margin`；当前只审计 `det_score>=0.8` 且未分类的 crop，最多 6 个；只写 `image_audit` 工件，不阻断高分 mask/measure；worker 的 PIL/numpy 输入 bug 已修复，修复后未在线复验。
- **classify_crops_batch / classify_target**：批量动作逐实例提交；底层仍用 14 类 ResNet18 对全部待分类 crop 做一次 batch 推理；`top1_prob>=cnn_threshold` 且熵低且不是 `other` 时直接写 `source=cnn`，否则进入 pending_review；单实例 classify_target 路径可回退 GLM；required crop 的 GLM 结果仍会做 YOLO crop 复核。
- **CNN 特征分类器**：权重与 meta 在 `分类器设计权重与CNN权重/cls_runs/abl_320_lb_e40_ls01_mix01_s20261031/`，使用 `.venv/yolo`（有 torch/torchvision）；只做 14 类特征归类，不判药材、不替代 locate、不生成 mask。分类器没有 `unreadable` 类；批量低置信/`other` 保留 `pending_review`，由 Agent 决定后续 GLM 复核。
- **identity_pending**：存在未分类的低分 `identity_required` crop 时，规划器必须先 classify 最高 `det_score` crop，不能继续 locate，也不能直接暂停或 review；classify 失败/不可判会留下操作记录，不会无限重试。
- **identity gate**：`identity_required` crop 必须先通过 classify/身份确认，高分 `det_score` 不能绕过身份门进入 IGM/测量。
- **locate_expand / min size**：GLM tight box 先扩 1.5 倍；观察 ROI 至少 `locate_min_roi_side=256`，tight box 至少 `locate_min_box_side=90`（围绕中心扩展，不直接丢弃小目标），减少 50–70px 小框和过度放大模糊。
- **box_quality**：每个 crop 记录 `box_source`、`box_quality`、`box_quality_flags`、`box_quality_metrics`；`blank`阻断分割；`low_detail`保留质量提示，同类别身份确认后可由模型选择分割，mask/标定/量纲资格另查，不因`det_score`高低绕过身份；同一特征的 GLM 框若与 YOLO 框 IoU>=0.5，保留 YOLO、跳过重复 GLM crop；`blank/low_detail` 可触发一次基于 `loose_box` 的 `locate:roi:boxq:*` 重定位；ROI 已有同 feature 有效 crop 或已有 done locate 时不再生成。纹理阈值尚未由独立显微核验集校准，不作为不可分割的证明。
- **样本级特征覆盖**：`coverage` 同时统计 `useful` / `missing_useful` 和 `required_features` / `required_all` / `missing_required`；fast 模式把同一样本各视野的有效分类取并集。任一视野检出某类特征，该类即覆盖，其他视野无需重复补扫。逐图覆盖仍用于审计，正式判药继续按原证据规则执行。高分 crop 也提供非阻断 classify，不再长期停留在 `high_score_unclassified`。
- **agent_driven_coverage**：自主路径由 planner 在受信合法菜单中选择；`coverage`/`required_features`/`missing_required` 是调查信息，不强制路线。规划失败或预算不足直接收尾复核，不默认选择第一个工具。
- **检测/标定解耦**：`detect_fov_initial` 先完成 YOLO；自动 scalebar 失败只置 `px_per_um=None`，保留检测框和 crop，后续可 `calibrate_fov`，不再把整个 detect 操作回滚为失败。
- **暂定排序**：`output.json` 增加 `provisional`，只读已产出的 classification，按 `VERIFIED_MATCH/1.0`、`DETECTED_PENDING/0.5`、`PARTIAL_MATCH/0.25`、`CONFLICT/0` 生成候选排序；`formal_verdict`、`support_count` 和正式证据类数不变。0.5/0.25 不进入正式确认。
- **每图 classify 上限**：fast 模式普通 YOLO crop 默认每图最多 `fast_classify_max_per_view=3` 次；identity_required 仍优先，不被该上限挡住。
- **segment_instance / measure_instances**：仍受真实 crop、身份、标定、IGM质量和用途资格约束。模型选定分割对象后，条件满足可自动测量同一对象；自主路径不因其他实例已有量程内测量就隐藏此实例，但同输入同方法去重保留。

当前模板：

```text
max_parallel = 2              # 仅独立观察并发，不是 locate 并发
fast_fallback_max_features = 2
fast_fallback_max_locate_calls = 6
fast_classify_max_per_view = 3
classify_batch_enabled = true
context_compact = true
context_recent_observations = 6
context_recent_operations = 8
context_max_actions = 20
auto_measure_after_segment = true
suppress_repeated_measurement = true
supporting_probe_total = 6
supporting_probe_per_candidate = 2
tool_timeout_seconds = 180
feature_scan_tile_policy = "full"
locate_attempts_per_view = 2
locate_attempt_reserve = 2
initialization_reserve = 1
glm_identity_threshold = 0.8
locate_expand = 1.5
locate_min_box_side = 90
locate_min_roi_side = 256
box_quality_enabled = true
box_quality_retry = true
agent_driven_coverage = true
identity_min_confidence = medium
feature_scan_features = []
feature_scan_max_tiles = 4
feature_scan_concurrency = 2
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
cnn_weights = "分类器设计权重与CNN权重/cls_runs/abl_320_lb_e40_ls01_mix01_s20261031/best.pt"
cnn_meta = "分类器设计权重与CNN权重/cls_runs/abl_320_lb_e40_ls01_mix01_s20261031/model_meta.json"
cnn_threshold = 0.75
cnn_entropy_threshold = 0.8
cnn_batch_size = 16
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

- `classify_target` 保留为单 crop 路径，`classify_crops_batch` 提供逐实例批量 CNN 动作，二者共享底层 CNN 缓存；批量动作不自动 GLM/segment。
- YOLO 仍是逐图执行，默认 `device=cpu`；尚未实现批量 YOLO/GPU 自动切换。
- `locate_feature` 仍是逐 ROI 动作；已增加同 feature/ROI 已有框或 done locate 的过滤，没有 locate 并发。
- fast fallback 使用 `full` 4 tile 全并发；当前 2 feature / 6 locate；全空即弃，不重试，不是零漏检保证。
- 自动标定失败不会丢 YOLO 框，但该视野仍需显式 `calibrate_fov` 才可能有 μm。
- 图片数不受 5 张限制，但总规划、总观察、总 API 和资源上限仍是固定安全阈值。
- 分类通过不等于针晶通过；`纤维`、`花粉粒`、`图像不可判` 都不会自动变成针晶。
- 没有合格 identity、mask 或标定时，可能没有 μm；这是约束，不是程序异常。

## 保留的组件

| 位置 | 职责 |
|---|---|
| `cli/main.py` | run、doctor、selftest、multi-view；five-view 为兼容别名 |
| `lib/application/multi_view.py`、`config/multi_view.json` | 任意数量原图接口及规范模板 |
| [代码组件](lib/README.md) | 应用编排、ReAct、MCP、视觉、证据、工作流及离线接口 |
| `cli/*_worker`、[预算启动钩子](lib/runtime/worker_bootstrap/README.md) | 隔离解释器、API、YOLO、分类和 B 阶段执行 |
| `lib/decision/final_assessment.py`、`output_projection` | 调查停止后综合判断；证据不足不强制确认 |
| `skills`、`requirements-agent*.txt` | 注册技能与锁定依赖 |
| `yolo-results` | 当前 best.pt、class_map 及训练/评测来源指标 |
| [回归目录](tests/README.md) | 主链契约与离线功能测试，统一由 selftest 递归发现 |

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
workflow/action_timeline.json
workflow/ops/
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

项目文件级地图见 [项目文件地图](../项目文件地图.md)；新手先看 [融合链路：新手快速入门](融合链路文档/00_新手快速入门.md)；其余详见 [当前实现状态](当前实现状态.md)、[修改记录](修改记录.md)、[融合链路文档索引](融合链路文档/00_INDEX.md)、[多图接口](多图测试运行方案.md)、[单图测试](单图测试操作手册.md)、[针晶个体与组织干扰改进指导](融合链路文档/08_针晶个体与组织干扰_改进指导.md)、[DAG 与旧短计划边界](融合链路文档/09_DAG与旧短计划边界.md)、[实施部署方案 v1.3](融合链路文档/10_ReAct主导的证据优化与语义DAG实施部署方案.md)。


### 通用逐类补扫（CODE-121）

新模板显式配置 `fast_recovery_selection=per_feature`、`fast_recovery_roles=[required,supporting]`。
YOLO 初检及当前视野 crop 分类后，活动候选的受支持微粒以单类 `feature_scan:<image_id>:<feature_id>` 动作提供给 Agent。
候选判据去重共享，`workflow.recovery_options` 给出 candidate_id/role 来源；不再按候选插入顺序截掉菜单尾部，也不重复跑 YOLO。
模型可选择现有实例核验或补扫，不要求遍历所有选项。每视野最多执行 `fast_fallback_max_features` 类，预留 tile 数之和不超过 `fast_fallback_max_locate_calls`；pending/失败也占用，未执行选项不占用。
跨视野已有分类时不重复补同类；这只避免重复检出，不表示候选 required 已 VERIFIED。后续核验仍用现有身份、标定及质量门。
请求、完成文件与所有 crop 的 feature_id 必须一致，缺省/不受支持 ID 报错，不能隐式变成针晶。
字段缺省或显式 `legacy_bundle` 保留旧任务的 required 合并扫描；历史配置与工件不修改。
本补丁未修改模型权重、判据或正式评分；新策略的真实定位/分类收益仍须独立在线对照。


### 提前核验第一步（CODE-124）

新模板 early_verification=true。初检后不因coverage缺口过滤合法分割/测量；verification_opportunities给出活动候选相关定量核验建议，模型仍决定顺序。planning_menu记录实际送模菜单，区别于原始propose offered。旧配置字段缺省或false保留旧策略；本批没有自动连续执行分割/测量。详见《提前核验第一步说明.md》。

EFF-001真实复测：同图新任务402.375秒、14次规划/7次观察，第二个mask合格并测量16.619808μm，交付暂定白芍；4次补丁有3次新观察/反证，不能称自进化。本例模型未选语义目标/双路观察；对旧408.594秒仅少1.52%，未证明明显提速。完整路径及工件见[本次测试总结](运行结果/case_eff_single_20261117_214135/测试总结.md)。


PLANNER-CHOICES-001最终真实复测：[七轮完整路径](运行结果/case_choices_fixed_single_20261117_223156/测试总结.md)。模型自主选3组观察并行及1个量纲目标；规划14→7、观察仍7、耗时402.375→245.828秒，规划＋收尾token346885→182248。交付仍暂定白芍，1类支持<最低2；单样本/生成路线变化，非因果泛化验收。首轮编号错误和修复记录完整保留。
