# agent

规划与反馈，从 react_graph 跟踪可见动作、模型选择、观察回写以及任务内反思。

当可授权的问题多于单批容量时，实际送模菜单同时显示本轮可选数与每批可执行数。系统提示用占位编号演示跨批列表，Agent 按调查价值排序，可选子集、暂缓有依赖的问题或申请复核。

新模板下，`observe_questions.question_ids` 可跨批授权多个当轮独立原图问题。每批最多两项，重新校验当前权限、剩余步数、专项和收尾额度。每条观察沿用自己的 MCP 调用、资源扣账和终态。仅当本批都获得可信局部未见、候选与调查建议稳定时继续。阳性、低置信、失败、场景回执或调查方向变化即交还 Agent；同批已派发伙伴的回执正常保留。多批选择由 `observation_group_wave` 和 `observation_group_return` 记录实际派发与余项。

在 adaptive_v2 中，最近四条只读回执内已展示的固定视野清单和同候选 KB 判据，可跨证据版本继续使用。移出可见窗口后显式调阅会恢复同任务回执，轨迹标记 `static_receipt`。动态标定及证据详情仍按当前版本读取。扣除本轮规划预算后若仅余复核、确认资格不足，且无待回收分支或可能复用的量测，运行时记录 `only_review_affordable`，直接进入复核收尾。

新模板启用 `identity_group_execution`。Agent 可在一个 decision 内按顺序选取多个独立 `classify_target`。未读清且继续有价值时逐张晚绑定；身份确认、其他分类结果、工具失败、调查建议或候选活动范围变化、输入/预算/截止变化时回交。每张走原执行、扣费、证据和反思回执路径。组本身没有显微证据资格。下一轮可读 `last_identity_group`，轨迹区分已尝试和未派发对象。

压缩后的操作总数取自同一账本。待核验摘要保留原请求特征、返回类别与拒绝原因。启用 `compact_view_ids` 时，真实发送的菜单及 `view_id_map` 可在本轮 planner_offer 核对；回复绑定和本地轨迹投影还原主机引用。

| 文件 | 阅读指引 |
| --- | --- |
| [identity_group.py](identity_group.py) | 从当轮实际展示的单裁图动作绑定身份、像素和来源框，检查所选对象是否独立，按单张正式回执决定继续或回交。 |
| [measurement_gain.py](measurement_gain.py) | 从合格实例的量纲、来源及跨候选边界生成可选复测理由，供直接动作和语义目标共用。 |
| [action_feedback.py](action_feedback.py) | 比较动作目标与动作前后的证据、候选判据和可用能力。 |
| [investigation_control.py](investigation_control.py) | 调查进展的本地指纹：约束无收益循环，不规定模型应调查哪个候选。 |
| [planning_choices.py](planning_choices.py) | ReAct执行选择视图：单动作、实例目标、单视野定量问题和独立观察组使用同一主选择接口。 |
| [problem_goals.py](problem_goals.py) | 将 D2 定量问题、原图与可用入口绑定，解析实例歧义，复用带来源的合格主判测量。 |
| [planning_context.py](planning_context.py) | 规划请求按证据变化更新；同轮历史提问、影子定位及其用途资格可按来源逐条还原。 |
| [prompt_defaults.py](prompt_defaults.py) | 提供 ReAct 规划默认指令及固定守卫文本。 |
| [prompt_release.py](prompt_release.py) | 验证现有提示发布工件。 |
| [react_graph.py](react_graph.py) | 从工作流状态读取当前合法动作和只读证据，构建 LangGraph 的评估、规划和执行节点。 |
| [reflection_controller.py](reflection_controller.py) | D3任务内反思控制器：实际反馈→补丁建议→校验应用→下一动作后验。 |
| [reflection_memory.py](reflection_memory.py) | 加载通过回放验证的经验工件。 |
| [semantic_execution.py](semantic_execution.py) | 现有实例目标和可选的单视野鉴别问题共用串行 DAG。每个节点按当前证据绑定合法工具，失败回交 ReAct。 |
| [physical_execution.py](physical_execution.py) | D4 跨视野调度。模型可单选，也可在同一工具组的不同视野中各选一个初检或分类。可选数量受并发上限、剩余步数和资源约束。协调者逐支提交，快支解锁实例操作时可提前交回 ReAct。 |

单步选择的 `decision` 内填写 `kind` 和 `action_id`。同视野伴随观察填写顶层 `parallel_observation_id`。运行时会归位已知的错层字段，以当前菜单核对主动作和伴随观察。伴随项有冲突或不可用时，有效的主动作可单独执行，轨迹记录处置结果。主选择无法确定时按本轮规划预算重试。

试验配置 `agent.workflow.criterion_impact_menu: true` 让规划动作关联本轮判据索引。共享判据只发送一次，关联保留候选、角色、状态、量纲与已登记的量程。阶段标签区分查找、观察、核验身份和直接取得量纲的入口。量纲复测沿用现有收益表，同物理问题的无增益反馈沿用反思账本。观察与工具资格继续由现有契约校验。

直接依赖 decision、evidence、runtime、tools。

对应回归见 [tests/agent](../../tests/agent/README.md)。

[返回上级导航](../README.md)
