# decision

概率分流、候选证据解释、A/B 融合和对外交付。

| 文件 | 阅读指引 |
| --- | --- |
| [rule_scoring.py](rule_scoring.py) | 复用判药模块的逐条核验，以当前有效证据计算正式 B 分，按分差、必需判据和独立证据广度裁决。 |
| [ab_fusion.py](ab_fusion.py) | 将 A 的逐折平均先验与同一候选的正式 B 分加权，生成融合排序。 |
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

## 从判据读懂 B 分

正式工作流写出 `b-rule/2`。先按候选自己的全部 required 权重计算达成度 R，计入有效的定性或量纲匹配、缺失与冲突罚项，再施加场景修正和证据上限。有量程要求却尚未标定的检出可以指导调查。它对该条 required 的正贡献为零。宽区间主导度只在已核验的 required 正贡献中计算。场景相悖仍会降分，相符也不能突破上限。

基础分为 `0.80 × R`。全部 required 均核验通过后，取每种独立 supporting 的最高合格贡献，排除已用于 required 的同类特征，把各类贡献相加得到 E。佐证加分为 `0.20 × R × E / (1 + E)`。没有任何有效 required 时 B 为零，部分满足时只给基础分。知识库没有 supporting 的候选仍可凭高质量 required 得到 0.80。

正式 B 排序、裁决分差、A+B 融合以及报告中的「B规则分」都取 `rule_score`。融合结果单列 `fused_score`，兼容字段 `confidence` 指向同一个 B 分。逐项贡献与罚项计算出的 `raw_score` 供审计。每个候选的 `score_components` 保留 required 命中数、达成度、基础分、逐类 supporting 贡献和实际加分。

倾向和确认分带下限分别为 0.50、0.75，明确区分候选的最小 B 分差为 0.10。独立特征类数决定证据广度，不能代替分差。单类放行仍须复核。Agent 可以在证据不足时请求复核，确认还受候选组审查约束。[离线对照与实算例子](../../运行结果/评估__B规则__30用例__20261119-095522__佐证计分/README.md)

对应回归见 [tests/decision](../../tests/decision/README.md)。

[返回上级导航](../README.md)
