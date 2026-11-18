# agent

动作选择、调查控制、优先级和任务内反思。

| 文件 | 阅读指引 |
| --- | --- |
| [test_investigation_control.py](test_investigation_control.py) | 建议/硬约束分离回归：真实LangGraph与明确fixture，验证自主选择及无收益收尾。 |
| [test_p0_priority.py](test_p0_priority.py) | P0 优先缺口判定的离线回归：只读工作流状态推断下一焦点，不联网、不执行阶段。 |
| [test_planning_choices.py](test_planning_choices.py) | 核对规划选项到实际模型短编号的映射以及模型响应解码。 |
| [test_reflection_strategy.py](test_reflection_strategy.py) | 验证问题标识和证据前后条件怎样影响反思建议、显式回应和后验记录。 |

统一命令和测试输入边界见 [tests 导航](../README.md)。
