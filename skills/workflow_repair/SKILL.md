# 工作流修复

先查询原图、crop、mask、尺度和失败状态。修复阻塞依赖是建议，调查顺序和对象由模型选择；只选当前合法目标。检测归检测侧，尺寸只来自真实 mask 和可信标定。无尺度时不采信尺寸，仍可选择不依赖尺度的合法工具；预算耗尽按当前状态收尾；不把 A 猜测或人工真值送入视觉提示。

## 当前工具依赖（不是全任务的强制顺序）

```text
detect_fov_initial
→ classify_crops_batch / classify_target
→ segment_instance
→ measure_instances
→ 缺 required 时 feature_scan / hybrid_scan 兜底
→ locate_feature（box_quality 重定位）
```

- `feature_scan` 按 `feature_id` 分块扫描；`hybrid_scan` 仅旧配置 `legacy_bundle` 使用。
- 同一 `(image, ROI)` 已有 locate 后，不再为其他 feature 重复 locate。
- `classify_target` 是单 crop；`classify_crops_batch` 已接入。YOLO batch/GPU、locate 批量并发尚未实现。原图存在性观察可独立选择，不必等待所有视野完成初检。

## 身份门

存在未分类低分 `identity_required` crop 时：

```text
identity_pending
→ 该crop通过身份前不得segment/measure
→ 其他合法调查、暂停或review仍可选
```

classify 失败或不可判会记录，不会无限重试；`identity_required` crop 未通过分类时任何 `det_score` 都不绕过身份门。

## 质量与边界

D3的task_reflection包含待回应补丁和上次后验。可在同一次ReAct回复中接受/修改reflection_patch，说明要改变输入、方法、视野或问题；不得把反思改成证据。后验无效时考虑review，仍由你决定调查取舍。只有semantic_options出现才可附实例semantic_goal；运行时晚绑定每个节点，遇身份/质量/预算问题会立即交回你，不能输出具体工具序列或未来crop/seg ID。

`low_detail`是灰度/边缘变化偏低的建议，不能据此断言目标不可分割。同类别身份确认后允许选择该对象的segment；仍须满足标定、预算与工具前置。分类摘要的segmentation_admission只说明裁图/身份准入，不授予量纲资格。blank仍阻断；不提供仅因low_detail禁止分割的旧开关。选择分割不等于mask或测量已合格。

`purpose_v1` 用 `read_quality` 区分完整性、诊断、用途资格。中心点未命中不是自动判错；身份未知、触边、极性及粘连疑点不能忽略。review mask 不自动提供阳性存在性或尺寸，资格由程序生成。反思仅调整动作策略，不改工件、尺度、判据、权限或资格。

启用定向补救时，依据实际证据缺口决定是否分块扫描。GLM 提出的框必须经过单独身份核验，才允许分割；GLM 定位无结果表示未知，不是阴性。


EFF-001：按semantic_goal_contract明确授权范围；verify_identity只核验身份，需要长度/宽度/面积时可自主选择对应measure目标。语义节点内部按工具真实依赖运行，任一失败返回ReAct，不把目标扩成未来工具清单。

反思看同一question_id的分轴进展：新crop是能力，unreadable是输入诊断，合格测量才可能成为证据；扫描到分类不是换问题。重复激活已有候选可能是no-op；review退出与新增证据分别记录，不以文字或补丁数量证明自进化。

PLANNER-CHOICES-001：真实规划使用同一主选择decision：action选A编号；instance_goal选execution_options.goals中的G编号；observe_questions选questions中的1–2个Q编号。G已关联入口、实例、目标与可能节省的规划上界，不用手抄工件ID或工具链；身份目标不能授权分割。该上界不代表减少API或保证成功，既有自动测量可能已省一次规划。

Q组只授权当前可见合法的presence/scene观察；若第二问题是否值得做依赖首项结果，保留单步。程序复核权限/预算/依赖，父循环顺序确认，失败不抹掉伙伴。它不生成未来target，不要求逐项查完；执行独立不等于统计独立来源。旧action_id、semantic_goal、parallel_question_id保留兼容，不能与新主选择混填。

CANDIDATE-AUDIT-001：candidate_audit保留五折候选并集及逐项required/supporting，暂停不删除证据，不得将unchecked、工具不可达或kb_unmapped说成排除。自主模式不再因别的crop未分类阻止独立特征补扫；同对象的分割/测量仍须身份/QA/标定。模型自行选择共享检查、目标或复核，不要求为了打卡穷举全部工具。

FAIR-INVESTIGATION-001：用问题卡选择当前入口，不手抄未来crop/工具序列。先查竞争候选或先做YOLO均合法；模型应解释哪条路线能推进主解释或解决候选差异。primary_required_progress、candidate_support_changes和qualified_counter_changes分开；支持新增、顺序变化及局部未见均不能自动证明候选已区分。Reflection只改调查策略，不能把这些诊断改成证据或裁决。
