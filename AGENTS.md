# 融合链路开发与测试规范

本目录遵守项目根 `AGENTS.md` 和 `重构方案.md` 的优先级与数据契约。规则冲突时，以根目录规范和用户当前指令为准。

### 可选 presence 定位边界（PRESENCE-001，2026-11-13）

- `presence_check` 的可选 `box` 仅是同一次观察的定位假设；坐标必须以实际下发图像为准，来源与逆变换由主机生成。不得采信模型自报的 transform、SHA、身份验证或量纲。
- `presence_localization_shadow_v1` 分开记录定位、质量、身份和用途资格。小目标、细长、贴边、面积和重叠只作诊断；不可读/不可解释映射明确阻断新工件，不依赖旧 `block_segment` 的兼容值。
- 没有人工核验集时，新定位工件仅作影子线索，身份保持 pending、用途资格不新增。旧 P1/C 规则保留，不能将“兼容采信旧观察”误写成“新 box 已验证”。
- 影子裁图不写入正式证据版本，不带 CropMeta，不能借 `locate_override` 进入分割/测量；后续调查由 Agent 从已有合法工具中自主选择，不强制每个阳性追加 GLM。
- 回归包含 `tests/test_presence_localization.py`，继续使用现有 `selftest` 总入口；离线结构验证不代表身份准确率或线上增益已验证。

PRESENCE-002 补充：观察须在请求前、定位发布前及账本提交时检查任务关闭/取消/截止；终态记账允许在任务关闭后完成，但不得采信迟到证据。取消、源图删除或替换必须落失败/取消终态，不得因异常长期留 pending，也不得自动重发未知计费请求。Windows 整树终止失败时应立即停止仍存活的 worker，不能先等五秒；父 worker 退出不等于已证明所有后代均退出。并发协议测试使用同步信号验证重叠，不依赖固定短 sleep 窗口。

## 1. 输入与测试

- 单图是多图的特例，按 [单图测试操作手册](单图测试操作手册.md) 执行；多图按 [多图测试运行方案](多图测试运行方案.md) 执行。
- 输入图片数量由 manifest 决定，至少 1 张，不再固定 5 张。
- 每张图必须有 SHA256；路径必须在 source-dir 内，路径和内容都不能重复。
- 新在线实验必须使用新 case、空 A 缓存和中性输入名；只有用户明确要求时才复用历史工件。
- A 解释器必须由 `--a-python` 或 `TCM_A_PYTHON` 显式提供，不默认使用 Agent venv。
- 参考 herb/feature 标签、人工框和参考结论只用于事后核对，不进入规划配置，也不作为扩展候选的依据。

## 2. A、B、C 的边界

- 真实 A 只以各折 Top-K 为 Agent 候选输入，默认每折 3 个；平均榜属于路由及对照报告。验收注明实际 K、逐折榜单和映射失败。
- 药材种类来自 Plan A；YOLO 输出显微特征类别和框，不输出药材结论。
- GLM `locate_feature` 是在指定 ROI 中找候选框；GLM `classify_target` 是对 crop 做身份复核；YOLO crop 复核是 classify 的辅助，不替代 GLM。
- 当前真实 Agent 测试显式选择 `adaptive_v2 + llm + mcp + active`。复现旧运行时保留其策略和预算。
- 规划器按当轮短编号选择注册动作；代码校验权限、前置条件、证据版本和资源预算，并可否决无充分证据的确认。不得用固定首项、参考答案或降级执行冒充真实模型决策。
- 分别记录模型、离线替身和运行时接管来源；usage 只记录实际返回值。

## 3. 当前工作流约束

- 当前默认 `fast_yolo_stage2=true`：每图 YOLO 先检，所有 YOLO crop 进入 CNN batch；默认不创建 `propose_regions` / `roi_screen` / 普通 `hybrid_scan`。
- 旧配置 `fast_recovery_selection=legacy_bundle`（字段缺省同义）保留 required 合并补扫、每图一次；新模板显式使用 `per_feature`，把活动候选的 required/supporting 逐类开放为 feature_scan，模型选择目标，预算内每图最多 fast_fallback_max_features 类、共 fast_fallback_max_locate_calls 个请求 tile。候选角色只作来源解释，不创造证据、不强制扫描、不满足 required；已尝试同视野同类不重复。旧参数：`fast_fallback_max_features=2`、`fast_fallback_max_locate_calls=6`、`feature_scan_tile_policy=full`。`full` 为 4 tile 全并发，全空即弃，不重试。
- YOLO 框全部保留；CNN stage2 先分类，低置信/other 才回退 GLM classify；required crop 的 YOLO crop 复核只是辅助。新模板 `auto_prepare_stage=true`：detect 后自动 classify_crops_batch，再自动 mandatory scene_check，不占 planning 请求。
- `adaptive_v2` 的 P0 缺口只能作为建议，不得按假设名称或 P0 等级强制调查路线；模型可以提出、调查、暂停、撤销或恢复假设，状态不是证据。新模板 `candidate_group_verification=true`：top required 满足后，竞争候选 required 未尝试且有合法动作时，`finish_veto` 加 `candidate_group_incomplete`，防止直接收尾；完成一次候选组核验后才解除。
- 身份完整性门是数据完整性约束，不是 P0 路线：存在 `identity_required` 且未分类的低分 locate crop 时，必须先 classify 最高 `det_score` crop，再允许继续 locate、暂停或 review。
- `det_score >= glm_locate_threshold=0.8` 的 crop 可走 `locate_override` 进入诊断性 IGM/测量，不强制 classify；低于 0.8 不享受绕过。`blank` 只阻断低分 segment；高分 crop 仍保留后续路线。
- `locate_min_roi_side=256`、`locate_min_box_side=90`；locate 无目标必须返回空，不伪造 crop。
- `detect_fov_initial` 的 YOLO 与自动标定解耦：scalebar 失败只置 `px_per_um=None`，不回滚 YOLO 框。
- `roi_screen_batch` 仍支持多视野并发；fast 默认不走。`classify_target` 对外仍是单 crop 动作；`classify_crops_batch` 已提供逐实例批量 CNN 动作；YOLO batch/GPU 尚未实现，不得在文档或验收中声称已完成。
- 只采信合法、可追溯的观察、定位、标定和测量。服务失败和无效输出按未知处理；任务结束、HTTP 成功、工具通过校验、药材判定和准确率是不同层次的结果。
- 调查停止码不等于诊断。正常收尾必须固定证据并生成 `final_evidence` / `final_assessment`；文本失败、非法引用或硬截止回退程序化研判。不得把失败当阴性、把先验或假设当显微证据、把融合分数当正确率。

## 4. 回归与记录

- 主链测试在 `tests/`，离线 Reflection/LangMem/数据划分/置信门/路由测试在 `tests/offline/`，两组均由 `selftest` 递归发现。不得通过删除离线断言伪造主链回归通过。
- 代码改动、测试结果和范围登记在 [修改记录](修改记录.md)；区分真实调用、零 API 续跑、模拟回归、源码审查与待验证假设。
- 调试信息统一维护在 [测试与调试经验](../融合链路/测试与调试经验.md)；保留独立运行的原始证据，不记录凭据、请求头、图像载荷或含凭据的原始响应。
- `case_9cc59640` 等历史 case 只作参考，不宣称验证了其后的代码。

## 5. 当前入口

```text
python -m TCM_Fusion.cli.main run
python -m TCM_Fusion.cli.main doctor
python -m TCM_Fusion.cli.main selftest
python -m TCM_Fusion.cli.main multi-view
python -m TCM_Fusion.cli.main five-view   # 兼容别名
```

离线回放、评估、LangMem 优化与发布统一走 `TCM_Fusion.cli.offline`；不得当废弃代码删除。

## 6. 变更摘要

- CODE-066：主/离线分层；主 CLI 保留 run/doctor/selftest；离线功能保留。
- CODE-067：A 解释器必须显式提供；主/离线测试分层。
- CODE-068：五图接口泛化为任意数量多图；动态初始化与 locate 预算。
- CODE-069：locate 后身份完整性门；未分类低分 crop 先 classify。
- CODE-070：同 ROI locate 去重；`roi_screen_batch` 并发筛选。
- CODE-071：按当前设计同步全部描述性 Markdown；不改变运行逻辑。
- CODE-101–110：fast_yolo_stage2、required coverage、full/prescreen tile policy、locate 最小尺寸、高分绕过、检测/标定解耦；当前默认 `fast_yolo_stage2=true`、`fast_fallback_max_features=2`、`fast_fallback_max_locate_calls=6`、`feature_scan_tile_policy=full`。
- CODE-111–157：YOLO crop 分类可进入 evidence；`provisional` 暂定排序只读有效 snapshot；`required_states`/`discriminating_gaps`；`fast_classify_max_per_view=3`、`tool_timeout_seconds=180`；hybrid 复用同 YOLO crop，避免 classification 被字段漂移失效。0.5/0.25 不进入正式 support_count；UNCAL/NO_REF 不再算 VERIFIED。

- CODE-121：通用逐类补扫、候选判据绑定及请求预算审计；未提供 feature_id 或不支持的 ID 在执行前报错。旧运行配置仍走原路径。

- CODE-124：新模板 early_verification=true；初检后coverage仅为调查信息，不得隐藏合法核验动作。verification_opportunities是候选相关定量建议，不授予证据资格；实际送模菜单见planning_menu，不能将propose offered直接解释为模型可见。

- CODE-126–157：batch 分类逐实例/pending；context 压缩（skills 摘要、actions≤20、evidence 摘要+index、feedback 去重）；自动测量与 required 测量门；supporting presence/feature 预算 6/2、共享一次；stop 冷却；重复 ROI locate 过滤；只读 catalog/detail 已注册但不自动进菜单。当前测试 363 项。
