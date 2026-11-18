# tools

工具注册和执行边界，处理权限、观察与只读详情。模拟后端只接受显式模拟上下文。

| 文件 | 阅读指引 |
| --- | --- |
| [agent_tools.py](agent_tools.py) | 接收当前合法的观察或重算动作，检查样品绑定和预算，在调用前登记去重项并在结束时记账。 |
| [simulation.py](simulation.py) | 显式模拟上下文的视觉响应后端。 |
| [tool_builtins.py](tool_builtins.py) | 读取会话提供的证据、质量、标定和候选状态。 |
| [tool_registry.py](tool_registry.py) | 将技能、权限、输入输出 schema 与受信 handler 和前置条件绑定为工具规格。 |

直接依赖 evidence、runtime、vision。

[返回上级导航](../README.md)
