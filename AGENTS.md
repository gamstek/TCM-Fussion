# 融合链路开发与测试规范

本目录遵守项目根 `AGENTS.md` 和 `重构方案.md` 的优先级与数据契约。规则冲突时，以根目录规范和用户当前指令为准。

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

- `adaptive_v2` 的 P0 缺口只能作为建议，不得按假设名称或 P0 等级强制调查路线；模型可以提出、调查、暂停、撤销或恢复假设，状态不是证据。
- 身份完整性门是数据完整性约束，不是 P0 路线：存在 `identity_required` 且未分类的低分 locate crop 时，必须先 classify 最高 `det_score` crop，再允许继续 locate、暂停或 review。
- `det_score >= glm_locate_threshold=0.8` 的 crop 可走 `locate_override` 进入诊断性 IGM/测量，不强制 classify；低于 0.8 不享受绕过。
- `roi_screen_batch` 可按 `screen_concurrency` 并发筛选多个视野；批量只在 worker 并发，证据由 `WorkflowSession` 一次提交。
- 同一 `(image, ROI)` 已有 locate 后，不再为其他 feature 重复 locate。
- `classify_target` 当前仍是单 crop 串行；`classify_batch` 和 YOLO batch/GPU 尚未实现，不得在文档或验收中声称已完成。
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
