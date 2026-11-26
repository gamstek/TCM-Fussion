# 融合配置

multi_view.json 用于构建新任务。detector.json 记录检测推理覆盖项，负例库和示例库向特征提示及视觉审计提供参考。字段语义和兼容要求见[重组方案](../融合链路文档/21_TCM_Fusion组件重组方案.md)。

## 多视野模板与历史配置

Plan A 用 `a.checkpoint_dir` 和 `a.model_runs` 固定三份 448 px ResNet50 药材分类权重。模板的 `candidate_policy.top_k=8` 用于单图平均榜，`candidate_policy.multiview_per_model_top_k=5` 用于两张及以上视野的逐模型候选并集。历史三模型配置缺该字段时沿用样品平均榜 Top-K。完整类别概率逐模型、逐视野等权平均，用于计算 A 分。显式指向历史五折目录且省略 `model_runs` 时走原入口。

`observation_group_execution=true` 允许 adaptive_v2 从当轮可见的 Q 菜单选择超过两个不同问题。可选数受步数和剩余观察、网络、总额度限制，执行时每批最多两项；`max_parallel=1` 时逐项执行。缺省或 false 保留原双问题菜单。

`identity_group_execution` 为布尔字段。当前多图模板显式开启，缺省保持原菜单。该设置向 adaptive_v2 增加可选的顺序身份核验，宽度受实际可见独立对象、剩余步数和逐张预算约束。

`compact_view_ids=true` 在正式规划请求中以当轮 `view_id_map` 表示重复视野 ID。主机反解回复后校验动作与工件。历史配置缺省沿用长 ID。当前 `per_feature` 模板在只余一次定位时仍下发局部 ROI，结果按部分覆盖解释。旧配置省略恢复选择器时沿用 `legacy_bundle`，显式关闭 `hybrid_scan_enabled` 将同时关闭该动作。旧示例配置和 v1 交付的迁移见[接口契约](../融合链路文档/03_接口契约.md)。

## LangSmith 本机配置

`langsmith.local.json` 集中保存 `api_key`、`endpoint`、`agent_id`、`environment` 和可选的 `workspace_id`。观察命令直接读取本机配置，显式环境变量可覆盖对应字段。本机 JSON 由 Git 忽略。[接入步骤](../融合链路文档/LangSmith追踪接入.md)

## 形态视窗

`feature_crop_policy: structure_v1` 用于当前模板。簇晶、方晶、石细胞、淀粉粒、导管、纤维和分泌组织按特征选择原像素观察窗。粒状目标四周留边。细长目标由框或方向稳定的纹理给出轴向，在该方向多留上下文。已有分类输入在新裁图中保留为子窗。

对上述特征，`locate_expand` 和 `locate_min_box_side` 进入观察窗，原始定位紧框用于分割关联。其他类别、显式 `legacy` 或省略该项的历史配置使用原裁图口径。[留边规则与子窗分工](../lib/vision/README.md#feature-window-method)

## 单实例视觉身份复核

`identity_review_bank` 指向[人工确认示例索引](identity_review_bank.json)。当前模板只在 `classify_target` 进入原有 GLM 复核后读取对照计划。出框类别与 CNN 首位冲突时优先比较这两类。两者一致时，`identity_review_cnn_margin` 默认 0.15，用于判断 CNN 次位是否值得对照。

当前索引 `identity-review-bank/v2` 收录用户确认的 7 组、14 张裁图。每组通过 `example_ids` 指定两张图，按待判类别与竞争类别排入 P、N。同一特征在不同组可以有各自的示例。索引保存原图、像素框、可见部位、裁图 ResNet18 分数及原图和裁图 SHA256。训练重叠字段为 `unverified`。

同组示例须来自已确认的不同来源。本任务原图与示例源图字节相同时，该对走目标图与形态提示。早期示例及人工状态保留在索引中供追溯。历史 `v1` 库沿用按类别匹配的规则。

复核沿用单图请求的实际高和宽。原生目标分辨率装不下、示例缩小后难以辨认、找不到已确认的成对示例或资产校验失败时，发送完整目标裁图和简短形态提示。类别需有目标区内匹配的结构代码及相对部位。不可判时记录“身份待定”。

## 单视野定量问题

`multi-view` 使用正式模板创建新任务时，默认允许 ReAct 授权单视野定量问题。要做关闭组对照，在同一模板副本中将 `agent.workflow.problem_goal_execution` 设为 `false`，并通过 `--config-template` 传入。旧模板缺少该字段时也采用关闭行为。[导管测宽与回交条件](../融合链路文档/09_DAG与旧短计划边界.md#单视野鉴别问题的授权入口)
