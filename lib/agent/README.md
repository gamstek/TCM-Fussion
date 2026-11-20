# agent

规划与反馈，从 react_graph 跟踪可见动作、模型选择、观察回写以及任务内反思。

| 文件 | 阅读指引 |
| --- | --- |
| [measurement_gain.py](measurement_gain.py) | 从合格实例的量纲、来源及跨候选边界生成可选复测理由，供直接动作和语义目标共用。 |
| [action_feedback.py](action_feedback.py) | 比较动作目标与动作前后的证据、候选判据和可用能力。 |
| [investigation_control.py](investigation_control.py) | 调查进展的本地指纹：约束无收益循环，不规定模型应调查哪个候选。 |
| [planning_choices.py](planning_choices.py) | ReAct执行选择视图：单动作、实例目标和独立观察组使用同一主选择接口。 |
| [planning_context.py](planning_context.py) | 规划请求按证据变化更新；同轮历史提问、影子定位及其用途资格可按来源逐条还原。 |
| [prompt_defaults.py](prompt_defaults.py) | 提供 ReAct 规划默认指令及固定守卫文本。 |
| [prompt_release.py](prompt_release.py) | 验证现有提示发布工件。 |
| [react_graph.py](react_graph.py) | 从工作流状态读取当前合法动作和只读证据，构建 LangGraph 的评估、规划和执行节点。 |
| [reflection_controller.py](reflection_controller.py) | D3任务内反思控制器：实际反馈→补丁建议→校验应用→下一动作后验。 |
| [reflection_memory.py](reflection_memory.py) | 加载通过回放验证的经验工件。 |
| [semantic_execution.py](semantic_execution.py) | D3实验串行语义图：模型选实例目标，程序逐节点晚绑定当前合法工具。 |
| [physical_execution.py](physical_execution.py) | D4 跨视野调度。模型可单选，也可在同一工具组的不同视野中各选一个初检或分类。可选数量受并发上限、剩余步数和资源约束。协调者逐支提交，快支解锁实例操作时可提前交回 ReAct。 |

直接依赖 decision、evidence、runtime、tools。

对应回归见 [tests/agent](../../tests/agent/README.md)。

[返回上级导航](../README.md)
