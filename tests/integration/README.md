# integration

跨组件闭环，包含真实 LangGraph、MCP stdio 或原几何 worker 配合可控输入。

| 文件 | 阅读指引 |
| --- | --- |
| [test_adaptive_quality.py](test_adaptive_quality.py) | 分层 QA、按用途采信、真值反馈、并发与硬截止的离线回归；不伪造在线收益。 |
| [test_agent_efficiency.py](test_agent_efficiency.py) | Agent效率回归：语义问题、分轴反馈、上下文/计时和模型授权并发；无真实API。 |
| [test_association_policy.py](test_association_policy.py) | 连通域关联策略回归：规则表、主导连通域、粒度守卫与保守回退。 |
| [test_component_boundaries.py](test_component_boundaries.py) | 检查组件之间的导入方向和在线边界。 |
| [test_d3.py](test_d3.py) | 在真实 LangGraph 和工作流守卫上验证规划上下文、单任务反思、实例语义目标和部分失败反馈。 |
| [test_decision_view.py](test_decision_view.py) | D2 离线回归：共享查询、区分问题、资格引用及歧义关联，随既有 selftest 发现。 |
| [test_fair_investigation.py](test_fair_investigation.py) | 检验先验领先者、竞争解释和实际合法工具入口在 ReAct 中保持可见。 |
| [test_integration.py](test_integration.py) | 可控输入与假子进程下的适配边界回归，绝不把模拟结果当真实模型效果。 |
| [test_react.py](test_react.py) | 真实 LangGraph 与 stdio MCP 边界回归；模型响应使用可控替身，绝不联网推理。 |

统一命令和测试输入边界见 [tests 导航](../README.md)。

[test_measurement_gain.py](test_measurement_gain.py) 对拍冻结连翘四图的逐轮测量菜单和最终引用，并用独立构造输入验证跨视野重开。
