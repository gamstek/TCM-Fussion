# 工作流修复

先查询原图、crop、mask、尺度和失败状态。优先修复阻塞依赖，再做鉴别观察；只选当前合法目标。检测归检测侧，尺寸只来自真实 mask 和可信标定。无尺度时不采信尺寸，仍可选择不依赖尺度的合法工具；预算耗尽按当前状态收尾；不把 A 猜测或人工真值送入视觉提示。

## 当前工具链

```text
detect_fov_initial
→ propose_regions
→ roi_screen / roi_screen_batch
→ locate_feature
→ classify_target
→ segment_instance
→ measure_instances
```

- 多视野 screen 可用 `roi_screen_batch` 按 `screen_concurrency` 并发；批量只在 worker 并发，证据由 Session 一次提交。
- 同一 `(image, ROI)` 已有 locate 后，不再为其他 feature 重复 locate。
- 当前 `classify_target` 仍是单 crop 串行；`classify_batch`、YOLO batch/GPU、locate 批量并发尚未实现。

## 身份门

存在未分类低分 `identity_required` crop 时：

```text
identity_pending
→ 先 classify 最高 det_score crop
→ 再允许继续 locate / 暂停 / review
```

classify 失败或不可判会记录，不会无限重试；`det_score >= 0.8` 可走 `locate_override`，低于 0.8 不绕过身份。

## 质量与边界

`purpose_v1` 用 `read_quality` 区分完整性、诊断、用途资格。中心点未命中不是自动判错；身份未知、触边、极性及粘连疑点不能忽略。review mask 不自动提供阳性存在性或尺寸，资格由程序生成。反思仅调整动作策略，不改工件、尺度、判据、权限或资格。

启用定向补救时，依据实际证据缺口决定是否提议区域。区域线索是纹理评分，不能当作特征存在证据。GLM 提出的框必须经过单独身份核验，才允许分割；GLM 定位无结果表示未知，不是阴性。
