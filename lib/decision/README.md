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

正式工作流写出 `b-rule/3`。先按候选自己的全部 required 权重计算达成度 R，计入合格的定性或量纲匹配、缺失与冲突罚项，再施加场景修正和证据上限。只有通用视野阳性时，具体形态判据记为 `MORPHOLOGY_UNVERIFIED`，保留视野来源和定性线索。存在性、具体形态和物理量纲按各自证据范围呈现。无合格量纲时，对定量 required 的正贡献为零。宽区间主导度只在已核验的 required 正贡献中计算。场景相悖仍会降分，相符也不能突破上限。

基础分为 `0.80 × R`。全部 required 均核验通过后，取每种独立 supporting 的最高合格贡献，排除已用于 required 的同类特征，把各类贡献相加得到 E。佐证加分为 `0.20 × R × E / (1 + E)`。没有任何有效 required 时 B 为零，部分满足时只给基础分。知识库没有 supporting 的候选仍可凭高质量 required 得到 0.80。

正式 B 排序、裁决分差、A+B 融合以及报告中的「B规则分」都取 `rule_score`。融合结果单列 `fused_score`，兼容字段 `confidence` 指向同一个 B 分。`raw_score` 记录原 B 逐项结果。`score_components` 保留 required 命中、达成度、缺失扣分及独立佐证。缺失扣分表示当前没有可采信的必需证据，局部未见按视野记账。离线影子对照可在相同已核验行上将 `absent_required` 置为空列表后复算，保持 required 数量和实测超范围罚项一致。

区分问题中的 `range_pairs` 给出相关候选两两的 `overlap`、`disjoint` 或 `not_comparable`。汇总的 `mixed` 表示既有重叠候选对，也有互斥候选对。定量反证仍须来自相同量纲的合格实测。规划摘要每题最多显示四对，详情由 `read_evidence_detail(kind=comparison)` 提供。

倾向和确认分带下限分别为 0.50、0.75，明确区分候选的最小 B 分差为 0.10。独立特征类数判断证据广度。单类放行仍须复核。Agent 可以在证据不足时请求复核，确认还受候选组审查约束。[b-rule/2 历史离线对照](../../测试运行总报告.md#recent-095522)

对应回归见 [tests/decision](../../tests/decision/README.md)。

[返回上级导航](../README.md)

### 收尾模型的引用规则

候选的 `eligible_support_refs` 和 `eligible_counter_refs` 由证据资格生成。模型引用的一项合格支持足以保留候选提议。存在性线索仅限对应视野，定量判据仍需合格的测量。其他候选的判据和 A 先验可以用于解释比较。系统逐项登记采信与剔除原因，零条合格支持时转入本地回退。正式确认与复核沿用原 B 和候选审查规则。

`final_assessment.json` 的 `model_proposal` 保存模型原话，`model_citation_audit` 按原提议候选和当时证据版本记录各栏通过的 ID 及剔除明细。若交付层改选候选，其证据另按交付候选绑定。混合引用触发 `validated_references` 说明文字，避免把已剔除的内容再次当作支持。
