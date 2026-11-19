# evidence_inspection · 查阅已有证据

在申请新的视觉调用前，先查清某条判据已有的事实、适用范围和仍缺的条件。本卡提供的工具均为只读。

## 适用条件

当规划摘要缺少来源、候选对同一观察的解释不同，或需要确认实例是否具备某项用途时使用。若当前摘要已能回答问题，可直接据此规划。紧凑视图保留逐视野逐折候选并集，未在详情窗口显示的候选仍可查阅。

## 行动选择

| 想核对什么 | 从哪里读 |
| --- | --- |
| 候选需要什么、与谁共享或区分 | `query_criteria` 查指定候选，`criteria_diff` 比较判据。`read_evidence_detail` 的 `candidate_catalog`、`comparison` 可展开候选和问题。 |
| 实际取得什么、可为谁作证 | `list_features` 看特征卡，`read_evidence` 看已有记录。用 `read_evidence_detail` 的 `eligibility` 查候选审查和资格，`eligibility_facts` 查事实引用，`observations` 查原观察，`provisional` 查暂定排序。 |
| 来源、身份、标定、掩码是否适用 | `list_fovs`、`read_scale` 查视野与尺度，`list_artifacts`、`read_artifact_status` 查工件，`read_quality` 查已有质量评估。`read_evidence_detail` 的 `instances`、`classifications`、`quality` 和 `measurements` 可追到实例及实际测量。 |
| 现在能做什么、先前做过什么 | `read_action_catalog` 查当下的阶段动作，`read_evidence_detail` 的 `operations` 查操作记录。要复核某一轮模型实际看到了什么，查当轮 `planner_offer`。 |

详情按 `total`、`offset`、`limit` 分页，读到能回答本轮问题即可。已经有正式引用时，保留其候选、视野或实例以及证据版本，便于收尾核对。

## 结果与边界

- “已经检查”（`assessed`）和“满足判据”（`satisfies_criterion`）要分别读。某候选得到支持，不会自动给竞争候选增加反证。未映射、暂停和未查的候选各自保留其状态。
- `present_pending` 表示仍缺正式资格。实例关联为 `ambiguous` 时不能合并，缺少质量评估或标定时相应用途保持未知。重复裁图不增加独立特征类别数。
- 共享观察可被多个候选分别解释。区间重叠、量纲不一致或知识库只支持组级判断时，在缺口中写明能区分到哪一层。影子评估仅供审查，正式结论以有效事实和判据为准。

## 示例

裁图已经标为簇晶，问题却是“直径是否在候选范围内”。先查该实例的身份、同视野标定、对应测量和用途资格。如只有分类结果，就保留“可继续测量”的调查入口，直径判据仍待核验。
