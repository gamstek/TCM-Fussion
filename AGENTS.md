# 融合链路开发与测试规范

同步：2026-11-19｜独立运行根、判据资格与候选区间。当前任务内反思接口和审计口径见融合链路文档20 §9。

本目录是独立运行根。路径由本目录 `paths.py` 定义，源图、数据契约和知识库按下述所有权约束维护。搬迁及回归的环境准备见 [独立部署](独立部署.md)。

自主融合工作流使用 `b-rule/4`。B 排序、A+B 融合和报告读取同一 `rule_score`。required 权重 80%，全部满足后才给独立 supporting 最多 20% 的加分。相同合格量纲实测支持一个分离区间并超出另一侧时，对被反证候选按特征扣分，保留实测、实例、参考区间和双方候选。通用存在性观察按视野保留，具体形态待核验时记 `MORPHOLOGY_UNVERIFIED`。原判药模块提供逐判据核验及独立基线，历史条目按运行版本解释。[计分与验收口径](lib/decision/README.md)

### 可选 presence 定位边界（PRESENCE-001，2026-11-13）

- `presence_check` 的可选 `box` 仅是同一次观察的定位假设；坐标必须以实际下发图像为准，来源与逆变换由主机生成。不得采信模型自报的 transform、SHA、身份验证或量纲。
- `presence_localization_shadow_v1` 分开记录定位、质量、身份和用途资格。小目标、细长、贴边、面积和重叠只作诊断；不可读/不可解释映射明确阻断新工件，不依赖旧 `block_segment` 的兼容值。
- 没有人工核验集时，新定位工件仅作影子线索，身份保持 pending、用途资格不新增。旧 P1/C 规则保留，不能将“兼容采信旧观察”误写成“新 box 已验证”。
- 影子裁图不写入正式证据版本，不带 CropMeta，不能绕过身份分类进入分割/测量；后续调查由 Agent 从已有合法工具中自主选择，不强制每个阳性追加 GLM。
- 回归包含 `tests/vision/test_presence_localization.py`，继续使用现有 `selftest` 总入口；离线结构验证不代表身份准确率或线上增益已验证。

PRESENCE-002 补充：观察须在请求前、定位发布前及账本提交时检查任务关闭/取消/截止；终态记账允许在任务关闭后完成，但不得采信迟到证据。取消、源图删除或替换必须落失败/取消终态，不得因异常长期留 pending，也不得自动重发未知计费请求。Windows 整树终止失败时应立即停止仍存活的 worker，不能先等五秒；父 worker 退出不等于已证明所有后代均退出。并发协议测试使用同步信号验证重叠，不依赖固定短 sleep 窗口。

## 1. 输入与测试

- 单图是多图的特例，按 [单图测试操作手册](单图测试操作手册.md) 执行；多图按 [多图测试运行方案](多图测试运行方案.md) 执行。
- 输入图片数量由 manifest 决定，至少 1 张，不再固定 5 张。
- 每张图必须有 SHA256；路径必须在 source-dir 内，路径和内容都不能重复。
- 新在线实验必须使用新 case、空 A 缓存和中性输入名；只有用户明确要求时才复用历史工件。
- A 解释器必须由 `--a-python` 或 `TCM_A_PYTHON` 显式提供，不默认使用 Agent venv。
- 参考 herb/feature 标签、人工框和参考结论只用于事后核对，不进入规划配置，也不作为扩展候选的依据。

## 2. A、B、C 的边界

- 新任务的 A 使用 `herb_cls_runs` 的三个 `v2_resnet18_448` 模型。单图取三模型平均榜 Top 8。多图每视野、每模型各取 Top 5，跨模型和视野并集经原映射进入 Agent。保留未映射项、逐模型概率与资产指纹。A+B 继续使用完整样品平均榜。历史配置按当时的候选策略运行。
- 药材种类来自 Plan A；YOLO 输出显微特征类别和框，不输出药材结论。
- STAGE2-001（2026-11-16）：`分类器设计权重与CNN权重/cls_runs` 中的 ResNet18 是 crop 显微特征分类器。stage2 预处理复用训练侧 `cls_net.build_transforms`，不得重抄 letterbox；checkpoint/meta 不符必须失败。换权重/meta/变换/crop 内容后不可复用旧 CNN 预测；单任务工件绑定 stage2 指纹。低置信/other 与模型加载失败分别记录，不把后者当阴性。
- GLM `locate_feature` 是在指定 ROI 中找候选框；GLM `classify_target` 是对 crop 做身份复核；YOLO crop 复核是 classify 的辅助，不替代 GLM。
- 当前真实 Agent 测试显式选择 `adaptive_v2 + llm + mcp + active`。复现旧运行时保留其策略和预算。
- 规划器按当轮短编号选择注册动作；代码校验权限、前置条件、证据版本和资源预算，并可否决无充分证据的确认。不得用固定首项、参考答案或降级执行冒充真实模型决策。
- 分别记录模型、离线替身和运行时接管来源；usage 只记录实际返回值。

## 3. 当前工作流约束

- `fast` 是唯一标准路径（不再提供 false 开关）：每图显微特征出框先用YOLO，CNN batch为可选合法动作，原图存在性观察可独立选择；标准路径不创建 `propose_regions` / `roi_screen` / 普通 `hybrid_scan`，只在缺 required 时按 `fast_recovery_selection` 生成 `feature_scan`（或旧配置的 `legacy_bundle` hybrid）。`fast_yolo_stage2` 仅为历史配置兼容保留，schema 只接受 `true`，缺失时按 `true` 补齐。
- 旧配置 `fast_recovery_selection=legacy_bundle`（字段缺省同义）保留 required 合并补扫、每图一次；新模板显式使用 `per_feature`，把活动候选的 required/supporting 逐类开放为 feature_scan，模型选择目标，预算内每图最多 fast_fallback_max_features 类、共 fast_fallback_max_locate_calls 个请求 tile。候选角色只作来源解释，不创造证据、不强制扫描、不满足 required；已尝试同视野同类不重复。旧参数：`fast_fallback_max_features=2`、`fast_fallback_max_locate_calls=6`、`feature_scan_tile_policy=full`。`full` 执行已分配的局部 tile，模板最多 4 块、并发上限 2。仅余 1 块按局部覆盖记账，全空不重试。显式 `hybrid_scan_enabled=false` 关闭旧合并补扫。
- YOLO 框全部保留；CNN stage2 先分类，低置信/other 才回退 GLM classify；required crop 的 YOLO crop 复核只是辅助。调查顺序与对象由模型从合法菜单选择。选定 segment 对象后仍可在资格满足时自动 measure。auto_prepare_stage 和 auto_required_measurement 仅供历史配置读取。
- `adaptive_v2` 的 P0 缺口只能作为建议，不得按假设名称或 P0 等级强制调查路线；模型可以提出、调查、暂停、撤销或恢复假设，状态不是证据。新模板 `candidate_group_verification=true`：top required 满足后，竞争候选 required 只做一次核验；每个候选最多一次，阴性/未知后不再扩散；阳性交给 React 决定是否改变研究对象。候选组缺口只作建议，不再以candidate_group_incomplete否决结束调查。
- 身份完整性门是数据完整性约束，不是 P0 路线：identity_required crop未确认时禁止该对象segment/measure；不因此禁止其他合法调查或review。
- 任何 `identity_required` crop 未通过分类/身份确认时不得进入 segment/measure；高分 `det_score` 只用于定位优先级，不绕过身份门。`blank`阻断分割；`low_detail`保留提示，须有有效同类别identity_verified分类后才可由模型选择分割。它不授予mask或量纲资格；旧low_detail阻断开关已删除，不接受该过时配置字段。
- `locate_min_roi_side=256`、`locate_min_box_side=90`；locate 无目标必须返回空，不伪造 crop。
- `detect_fov_initial` 的 YOLO 与自动标定解耦：scalebar 失败只置 `px_per_um=None`，不回滚 YOLO 框。
- `roi_screen_batch`、`roi_screen`、`propose_regions`、`needle_scan` 已随标准路径移除，不再注册或提供；`classify_target` 对外仍是单 crop 动作；`classify_crops_batch` 已提供逐实例批量 CNN 动作；YOLO batch/GPU 尚未实现，不得在文档或验收中声称已完成。
- 只采信合法、可追溯的观察、定位、标定和测量。服务失败和无效输出按未知处理；任务结束、HTTP 成功、工具通过校验、药材判定和准确率是不同层次的结果。
- 调查停止码不等于诊断。正常收尾必须固定证据并生成 `final_evidence` / `final_assessment`；文本失败、非法引用或硬截止回退程序化研判。不得把失败当阴性、把先验或假设当显微证据、把融合分数当正确率。
- FINAL-001：`adaptive_v2` 只读文本规划遇 `UPSTREAM_REMOTE_DISCONNECTED` 在本轮两次尝试上限内最多重试一次，逐次扣 planning/network/total，保留收尾额度与时间；不得将它扩展成图像工具自动重试。连续断连或额度/鉴权失败后停止，不再借初检回退发付费工具。`termination.reason` 是执行原因，`review_reasons` 是交付授权原因，二者不得合并；D8b 单类复核不因此放松。候选分数按 ID 从结构化表输出，模型自由文本中的数值声明保留原提案审计并回退程序化解释。

## 4. 回归与记录

当前模板试行 8 次 supporting 探查，分 5 次存在性观察与 3 次定向补扫。关键入口仍用于菜单排序，`protect_investigation_budget: false` 关闭其挡掉其他合法查询的行为，缺省保留旧策略。每候选分别可发起 2 次与 1 次，真实 required 不占 supporting 额度。共享一次只扣一个发起者，重复物理查询仍拦截。并行观察先联合预检专项额度，余额只够一项时保留模型所选的第一项。所有动作仍受原任务预算、实例资格和截止约束。

- 新增模块同步维护同目录 README 与三段式说明。组件分工、接口复核与迁移见 [组件重组方案](融合链路文档/21_TCM_Fusion组件重组方案.md)。
- 新测试的目录命名固定为 `实测<药材名>_M月D日HH时mm分ss秒` 或 `模拟<药材名>_M月D日HH时mm分ss秒`。例如 `实测龙胆_11月19日14时30分25秒`。使用北京时间开始时刻，同秒重名时依次追加 `-02`、`-03`。图数、图号和目的登记于运行记录。盲测用 `S01` 等中性代号。内部 `runs/fusion_<id>` 用于工件绑定。仅在用户明确要求时变更命名规则，并同步入口、回归与 [运行结果索引](运行结果/README.md)。

- 主链用例按组件归类，跨组件回归在 `tests/integration/`，离线用例在 `tests/offline/`，均由 `selftest` 递归发现。不得通过删除离线断言伪造主链回归通过。
- 代码改动、测试结果和范围登记在 [修改记录](修改记录.md)；区分真实调用、零 API 续跑、模拟回归、源码审查与待验证假设。
- 真跑、回放、评估和重要联调结束后均登记 `运行摘要.md`，包括失败、中止和复核。按 [统一摘要模板](运行结果/README.md#运行摘要模板)记录测试目的、对象和输入、版本与边界、实际路径、证据与交付、开销、异常及产物。结束后将目录与摘要登记到测试运行总报告。归档核对原始工件 SHA256、历史文件地址和本地链接。
- 调试信息统一维护在 [测试运行总报告](测试运行总报告.md)；保留独立运行的原始证据，不记录凭据、请求头、图像载荷或含凭据的原始响应。
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
- CODE-101–110：fast_yolo_stage2、required coverage、full/prescreen tile policy、locate 最小尺寸、高分绕过、检测/标定解耦；当前 `fast` 为标准路径、`fast_fallback_max_features=2`、`fast_fallback_max_locate_calls=6`、`feature_scan_tile_policy=full`。
- CODE-111–157：YOLO crop 分类可进入 evidence；`provisional` 暂定排序只读有效 snapshot；`required_states`/`discriminating_gaps`；`fast_classify_max_per_view=3`、`tool_timeout_seconds=180`；hybrid 复用同 YOLO crop，避免 classification 被字段漂移失效。0.5/0.25 不进入正式 support_count；UNCAL/NO_REF 不再算 VERIFIED。

- CODE-121：通用逐类补扫、候选判据绑定及请求预算审计；未提供 feature_id 或不支持的 ID 在执行前报错。旧运行配置仍走原路径。

- CODE-124：新模板 early_verification=true；初检后coverage仅为调查信息，不得隐藏合法核验动作。verification_opportunities是候选相关定量建议，不授予证据资格；实际送模菜单见planning_menu，不能将propose offered直接解释为模型可见。

- CODE-126–157：batch 分类逐实例/pending；context 压缩（skills 摘要、actions≤20、evidence 摘要+index、feedback 去重）；自动测量与 required 测量门；supporting presence/feature 预算 6/2、共享一次；stop 冷却；重复 ROI locate 过滤；只读 catalog/detail 已注册但不自动进菜单。当前测试 383 项。

- D1-001（2026-11-17）：最新离线基线为391项（390通过、1项因历史参考case缺失跳过），另原B59/21/125通过，替代上文历史计数。CNN pending_review允许首次GLM classify_target；GLM同输入pending/done/failed不因换名重发。`instance_ledger.json`只是任务内工件索引，执行状态以operations为准；语义键绑定源图SHA/精确坐标、输入和方法/参数，不以候选名扩增支持，不给别名自动授予身份或尺寸资格。保持单步ReAct及原自动required测量，DAG/新版Reflection待后续阶段；离线fixture通过不等于真实API或准确率通过。详见融合链路文档/11_D1交付与验收.md。

- D2-001（2026-11-17）：最新回归411项（410通过、1历史case缺失跳过），B59/21/125通过。自主presence默认使用taxonomy通用问题，多候选共享；专门形态/量纲/范围不同不可强行共享。循环与收尾共用evidence_projection资格及来源绑定ID，先验/失败/未标定尺寸不能作正式支持。高IoU/包含只提示ambiguous，不自动合并或授权。完整资格/事实/区分问题/实例由read_evidence_detail v2分页读取，skill映射、模型菜单及stdio MCP均已验证；完整五折候选并集不裁成Top2。D3–D5未实施，真实效果/成本未验收，详见12_D2交付与验收.md。

- ADVISORY-001（2026-11-17）：最新基线424项（423通过、1历史case缺失跳过），原B59/21/125通过，覆盖前述历史计数。adaptive_v2的调查优先级/覆盖/修复为建议，模型选对象与顺序；review始终可选，无资格finish同轮转review，不重复付费重问。取消自主auto_prepare与首项接管，旧配置true不恢复该强制顺序；同对象合格分割后自动测量保留。有效进展可多次偏离建议，停止建议后连续两次执行无新进展则no_progress_after_advice收尾，不能只看override次数。精确去重/来源/坐标/身份/QA/标定/预算/原规则仍硬校验。Windows瞬时读冲突仅有限本地重试，不重发API。交付融合链路文档/13_调查建议与硬约束分离.md；无真实视觉API、无线上效益结论，D3–D5待实施。

- LOW-DETAIL-001（2026-11-17）：最新回归433项（432通过、1历史case缺失跳过）、原B59/21/125通过。low_detail是质量建议，同类别有效identity_verified后可供模型选择segment，blank仍拦；不修改纹理状态或授予量纲资格。旧box_quality_block_low_detail已从schema/模板/代码删除，旧case仅作审计，新任务使用当前模板。准入信息必须在压缩后的classification_summary仍可见；不新增模型调用。两张实际crop本地准入通过，纹理12组对拍不变；未重跑真实视觉API/生成mask，不作为准确率验证。


- D3-001（2026-11-17）：最新回归458项（457通过/1历史参考case缺失跳过）及B59/21/125通过，覆盖以上历史计数。planning_context只改送模视图，保留逐图五折候选并集和全KB索引；6候选详细窗口不影响权限或候选状态，详情工具v3支持candidate_catalog/provisional/observations，当前显式页不可强制隐藏。字节画像不得当token，usage仍取真实返回。
- 任务内reflection_patch只能在封闭schema内修改调查策略，精确避让已知失败键、通过旧假设入口激活已知候选；不得改证据/资格/判据。与下次正常ReAct规划合并，不独立加付费反思调用；后验检查有效条件变化与新信息，不将文字总结/effective当因果收益或跨任务学习。
- semantic_execution默认false，不恢复旧具体工具短计划。开启且模型本轮选择后，只串行执行该实例语义目标，按当前工件晚绑定，保存required_facts受信名称；几何/像素/身份/标定/用途/预算/截止变化立即停止后继并回交ReAct。测量文件存在但请求量纲无有限正值，不得记目标成功。既有并行组合逐子动作反馈，不用伙伴成功掩盖失败。D4并行调度/事件和D5真实评估待实施；不得将默认关闭的最小S4称为完整并发DAG。交付见融合链路文档/14_D3交付与验收.md。


- D3-LIVE-FIX-001（2026-11-17）：最新466项（465通过/1历史case缺失跳过），B59/21/125通过。普通及自动子阶段经同一worker_context产生真实预算上下文路径；禁止向WorkflowBackend/controlled_process传{}冒充上下文。无效上下文在启动worker前拒绝；本地监管错误不能统一记WORKFLOW_TIMEOUT。自动测量成败分别反馈到ReAct/Reflection，不能由分割成功遮蔽。
- 收尾若按已有provisional策略改选交付候选，必须同步重建该候选的解释、支持/反证引用及has_admitted_b_support；原模型提案仅作审计，不能把另一个药材的支持拼到当前结果。DETECTED_PENDING和缺少合格尺寸不等于原图不存在目标。
- Reflection拒绝仅记录已知ID、计数和摘要，下一正常规划可见原因及完整proposed_patch；禁止用额外付费重问修正可选补丁，不自动扩权或把模型未采用反思说成自进化。

- D3-LIVE-FIX-001复测：真实单图簇晶测量已入账，2次反思补丁有后验但无新增有效证据，不称自进化；原规则单类confirmed+needs_review，对外交付仍tentative。一次网络断连保持unknown；GLM新crop身份unreadable不能以放松low_detail绕过。原case不覆盖，最终schema字段诊断在运行结束后补入并全量回归。

- EFF-001（2026-11-17）：最新485项（484通过/1历史case缺失跳过）及B59/21/125通过。跨工具调查使用主机绑定的physical-question ID，执行前冻结输入归因；候选/工具名称变化不等于换问题。进展分evidence/capability/diagnostic，诊断不可判不当阴性；补丁应用/no-op/退出/新增证据分别记账，不称自进化。
- 当前判据默认摘要合并到workflow.criterion_states；五折Top-K并集、资格投影和显式详情读取保持可达。可选假设/并行伙伴元数据错误只拒绝该部分，保留独立合法动作，不额外付费重问。
- parallel_question_id只允许模型选择两个不同语义问题的当前合法presence，max_parallel及各预算仍硬校验；父循环顺序确认观察，失败不抹掉成功伙伴。不并发同实例修复/分割/测量，不宣称已实现完整D4。semantic_execution默认false，verify_identity只做身份目标，不承诺mask/测量。
- 性能审计分planning_timing/execution_timing/process_timing/HTTP阶段；外层跨度含自动测量，并行子项不可重复累加。MCP只透传调用方显式设置的线程与TEMP/TMP；低内存回归需记录线程条件，不改用户/系统环境、不删失败日志或放松断言。selftest回收用例循环引用不改变生产GC。详见融合链路文档/15_调查效率与反思收益优化.md。

- PLANNER-CHOICES-001（2026-11-17）：实际LLM主回复为decision三选一：action/A、instance_goal/G、observe_questions/Q组。只解码模型明确选择，未知/矛盾主选择不能猜首项；旧action_id仍兼容但不混填。execution_options从当前可见合法动作关联目标及问题，不能拿全choices冒充实际菜单。
- 允许两路当前合法presence/scene观察（同源不代表独立证据），是否同时值得执行由模型选择；第二问题依赖首项结果时选单步。实例目标遇身份/QA/标定/量纲/预算失败回交ReAct；verify_identity不能授权segment/measure。目标表的规划节省上界扣除既有自动测量，不代表少API。
- planner_offer记录真实LLM短编号/目标/问题/预算，thought.execution_choice记录模式；未保留的历史请求正文不可伪称已完整重建。当前503项（502通过/1历史case缺失跳过），B59/21/125通过；语义默认false、完整D4/D5未交付，见文档16。


- CANDIDATE-AUDIT-001（2026-11-17）：候选审查充分性优先于token/时延。candidate_audit必须保留五折Top-K并集的mapped/unmapped和暂停候选的required/supporting；暂停不删除阳性事实、不等于排除。Top1 required满足不能替代全组比较。
- 逐项区分unchecked、失败/unknown、定向搜索未解决、存在但量纲待核验及原规则合法支持。单视野阴性不代替其他视野；全部提供视野未见也只限这些视野，不排除标本。
- adaptive_v2的独立特征搜索不再要求先分完别的crop；相应对象segment/measure仍按真实身份/QA/标定前置。旧固定路径行为保留；不强制候选顺序/逐项付费打卡。
- 完整审查是确认资格，review仍随时允许；缺口完整落盘、不为填格式额外重问。原B判据/分数保持，交付增加CANDIDATE_AUDIT_INCOMPLETE/UNMAPPED_A_CANDIDATES，不将未映射高分候选静默删除。
- 原unresolved字段不是全候选缺口；以candidate_audit和unresolved_scope为准。新版本须独立验证覆盖、错误确认及拒答后再报告效率，不以七轮为硬目标，旧七轮/245.828秒不证明同覆盖提效。详见文档17。

- FAIR-INVESTIGATION-001（2026-11-18）：主解释与竞争解释公平取证。comparison_problems的entry_paths只关联本轮实际动作；discovery_prerequisite不授予阳性、身份或量纲，不承诺后继执行。逐折先验领先者及未映射折首选不能被竞争弱阳性挤出状态视图，模型仍自主决定顺序。
- 已知decision说明字段可受限归一；未知/混合执行字段不忽略。格式纠正与断连重试共用原两次尝试上限，坏提案无工具执行；每次模型请求真实扣预算/保留usage，不能当零成本恢复。review不因主解释未查而禁止，不能强制首项。
- 输出/报告遵循同一已校验final_assessment候选；融合榜另列，未知交付候选须失效为undetermined。支持变化、主解释required推进、合格量纲反证分别反馈；不得把竞争支持或局部阴性说成已排除标本。
- decision_readiness_shadow仅比较审计完整与判定充分，生产授权不变。新版确认策略必须独立风险/覆盖验证后才发布；不为使测试通过而放宽身份、QA、标定、预算或原B。详见文档18。

- CONTEXT-REFLECTION-001：原B评分和判据不改。NO_REF是参考缺失，重测不补KB、分差不是排他反证；range_relation缺参考为not_comparable。共享实例事实一份、各候选解释分别引用。
- Reflection按稳定issue/revision与物理反馈去重，能力推进不因排名不变判停滞；服务等显式失败不被诊断文件进展掩盖。模型在下一正常规划自主apply/maintain/defer/reject，问题级strategy只引用当轮真实入口，不指定药材/特征固定顺序；缺失/非法可选响应不得废弃合法主动作或新增付费重问。
- 反思验收分别统计实际送模、响应、接受、策略增量、条件落实及合格证据/缺口改善；接受补丁或换文件不能称自进化。新意图不改变工具菜单/权限；旧reflection_patch兼容，完整DAG并发/部分交接仍后置。

- CONTEXT-REFLECTION-001最终：最新564项（563通过/1历史case缺失skip）及B59/21/125通过，新增24项。并行子反馈仅恢复自身真实进展，不借伙伴收益；A别名仅认当轮合法Q关联，其他工具不混入观察组；格式/服务修正共享原两次上限，每次真实扣预算。新图B20240866.jpg实测暂定白芍需复核，Reflection1卡但0结构化应用，不称自进化或已验证策略收益。换图实验不可作配对提速；联合候选预算预检与目标身份失败反馈仍待后续，不放松原B/身份/配额。见文档20 §9.1及本次测试总结。
