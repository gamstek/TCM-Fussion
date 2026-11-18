# decision

概率分流、候选证据解释、A/B 融合和对外交付。

| 文件 | 阅读指引 |
| --- | --- |
| [ab_fusion.py](ab_fusion.py) | A/B 候选融合：把 A 的逐折概率与 B 的规则置信度按权重合并后重排候选。 |
| [candidate_audit.py](candidate_audit.py) | 逐项对照候选判据、当前用途资格、任务动作和观察账本。 |
| [candidate_policy.py](candidate_policy.py) | 把 A 的逐折 Top-K 预测映射为 Agent 候选兴趣清单，仅作查证入口而非显微证据。 |
| [candidate_scope.py](candidate_scope.py) | 消费 A 候选、原裁决和融合配置，生成允许交付的候选范围。 |
| [collaboration.py](collaboration.py) | A/B 只读协作与有界影子 Agent：融合两路结果并给出可选补查建议，不改写原工件。 |
| [comparison_problems.py](comparison_problems.py) | D2 候选区分问题：共享物理查询，分别解释判据，不生成强制调查剧本。 |
| [confidence_contract.py](confidence_contract.py) | 候选置信度工件的共同约束。 |
| [decision_llm.py](decision_llm.py) | 接收分割复核原因或候选证据摘要，通过有界文本请求选择修复和候选准入策略。 |
| [final_assessment.py](final_assessment.py) | 只读收尾研判：把冻结的 A/B 证据转成有界文本判断，失败时回退程序化汇总。 |
| [output_projection.py](output_projection.py) | 把规则裁决投影成“永远有输出”的交付视图，不修改 decide/阈值/证据卡，并保留真实复核状态。 |
| [routing_policy.py](routing_policy.py) | 把 A 的原始完整排序转成启发式分流建议；只消费推理结果，不校准、不造 B 证据。 |

直接依赖 evidence、runtime、tools、vision。

对应回归见 [tests/decision](../../tests/decision/README.md)。

[返回上级导航](../README.md)
