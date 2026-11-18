# support

可复用的测试替身和工件构造器。测试包引用这些辅助函数。

| 文件 | 阅读指引 |
| --- | --- |
| [fixed_baseline.py](fixed_baseline.py) | 同起点对照原 B 固定补查策略；仅用于显式模拟环境，不把回放当在线收益。 |
| [prompt_support.py](prompt_support.py) | 离线提示测试替身；保留真实 LangMem 优化器，仅替换外部聊天模型与规划响应。 |
| [result_builders.py](result_builders.py) | 构造候选投影测试所需的规则裁决、Agent 结果和 A 预测。 |
| [workflow_fixtures.py](workflow_fixtures.py) | 组合临时测试目录、阶段替身和可控候选场景，供工作流及跨组件用例复用。 |

统一命令和测试输入边界见 [tests 导航](../README.md)。
