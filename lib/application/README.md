# application

任务与交付编排，从此处追踪 run、doctor、multi-view 进入的 A/B 链路。

| 文件 | 阅读指引 |
| --- | --- |
| [multi_view.py](multi_view.py) | 通用多图入口：原图/manifest→新建 case→doctor→既有工作流；图片数量由 manifest 决定。 |
| [pipeline.py](pipeline.py) | 接收 A/B 与 Agent 配置并先校验运行模式和样品绑定。 |
| [workflow_entry.py](workflow_entry.py) | 将校验过的本次任务交给实际工作流，创建本地或 MCP 工具箱并执行 ReAct。 |

直接依赖 agent、decision、evidence、mcp、runtime、tools、vision、workflow。

对应回归见 [tests/application](../../tests/application/README.md)。

[返回上级导航](../README.md)
