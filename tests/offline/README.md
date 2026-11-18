# offline

离线划分、路由测试、置信门、提示优化、反思和冻结工件的回归。

| 文件 | 阅读指引 |
| --- | --- |
| [test_confidence_gate.py](test_confidence_gate.py) | 统计门控离线单测；临时构造数据不证明真实分类效果或样品独立性。 |
| [test_dataset_split.py](test_dataset_split.py) | 数据集划分回归：group 隔离、确定性、显式覆盖与泄漏检查。 |
| [test_evaluation.py](test_evaluation.py) | 离线评估与报告回归，不属于线上五视图链路。 |
| [test_prompt_optimization.py](test_prompt_optimization.py) | 真实 LangMem 优化、提示消费、封闭回放/发布隔离与共享账本回归。 |
| [test_reflection.py](test_reflection.py) | 反思发布的有效性、数据隔离与回放反例回归；不调用真实模型。 |
| [test_routing.py](test_routing.py) | 路由真实编排分支、候选权限与零 API 只读验证；不把替身当识别效果。 |
| [test_stage_replay.py](test_stage_replay.py) | 批量分类与示例审计往返、摘要校验、历史录制缺件。 |
| [test_workflow_prompt.py](test_workflow_prompt.py) | 离线 LangMem/prompt 发布工作流回归，被 selftest 递归发现但不属于主链测试。 |

统一命令和测试输入边界见 [tests 导航](../README.md)。
