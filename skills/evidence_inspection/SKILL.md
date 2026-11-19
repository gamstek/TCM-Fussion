# 证据检查

D3紧凑视图保留全部逐折候选和知识库ID，详细显示窗口不是候选排除。可用read_evidence_detail的candidate_catalog/provisional/observations分别分页取回完整目录、暂定排序和原观察；当前显式读回的详情不会被再次隐藏。字节统计不是模型usage，不据此声称准确率或费用改善。

先查已有证据、尺度与视野，再决定是否需要新观察。只读结果不是新证据；缺失不等于不存在，服务失败也不等于阴性。

可调用注册的 `read_evidence`、`list_features`、`query_criteria`、`read_scale`、`list_fovs`、`criteria_diff`、`list_artifacts`、`read_artifact_status`。正式 B 分由判据计算。全部 required 均已核验时，独立 supporting 按质量加分。同类重复观察仍按一类计。

`read_quality` 可读取受信评估的原因码与存在性/形态/长度/宽度/面积资格。没有评估时为未知，不把旧工件自动升级。原始记录和诊断测量与规则实际采用的证据要区分，不能把文件哈希变化称为新证据或正确性收益。

判据差异可参考 A 逐折 Top-K 的映射候选及 B 已有支持；候选分数只用于规划查证，不能补写证据。当前多图接口接受任意 N 张图；单图是 N=1 的特例。

D2：先读上下文的 `comparison_problems`、`eligibility_projection` 和 `instance_associations`。
通用存在性只证明某类特征在该视野可见；不同候选对形态和量纲的要求仍需分别解释。
`ambiguous` 不表示已证明同一目标，也不授予身份/尺寸资格。KB 只支持组内不可区分时，不靠重复扫描猜成员。

摘要被截断时，可用已有 `read_evidence_detail` 分页读取 `eligibility`、`eligibility_facts`、`comparison`、`instances`；以返回的 `total/offset/limit` 为准。
`read_action_catalog` 可查询完整阶段动作。它们只是已注册的只读能力，权限仍由 ToolSpec 和程序守卫决定；Markdown 不扩权。


EFF-001：压缩视图的workflow.criterion_states集中保存判据状态、来源实例及局部观察状态，supporting_states/supporting_gaps只是候选-特征索引；顶层重复镜像已移除。完整五折并集、资格投影和当前显式读取结果继续保留，其他细节按需读取。

action_objectives里的增量说明和实测耗时范围只是调查建议；同类多个crop不能自动增加独立特征类别数。查看planning_timing/execution_timing和子阶段process_timing时区分嵌套与并行，不能把耗时相加重复计算，也不能把JSON字节换成token。

PLANNER-CHOICES-001：execution_options把目标/问题直接关联当前A动作。shared_consumers说明一次观察可服务谁，不表示通用存在性足以区分这些候选的形态/尺寸。选择组前判断是否真的需要各结果；不为unchecked打卡。planner_offer保存实际送模短编号和目标/问题表，全部合法集合或timeline offered不能替代它证明模型看见了什么。

CANDIDATE-AUDIT-001：evidence.workflow.candidate_audit是全候选状态摘要，不随六候选详情窗口裁掉；read_evidence_detail(kind=eligibility)可读完整账本。assessed与satisfies_criterion分开：局部未见不等于全样本排除；有存在性但无定量资格仍为present_pending。候选暂停不能抹掉程序已采信的阳性支持。

多图阴性审查必须保留视野覆盖：local_absent_partial只说明部分视野有效未见，其余未完成；只有全部提供视野均有效未见才记local_absent，仍不排除整个标本。阳性资格继续以原规则与具体用途为准。

FAIR-INVESTIGATION-001：comparison_problems的问题卡同时展示直接步骤和discovery_prerequisite。没有crop不等于没有路线；当前YOLO初检可以推进实例发现，但不保证发现某类，也不构成阳性。entry_paths只引用本轮菜单，后继必须重新满足真实前置条件。

investigation_focus保留逐折先验领先者及其required缺口、未映射折首选和竞争支持；这些是解释信息，不要求先查A首选。竞争阳性不自动成为主解释反证。候选审计完整与判定充分分开；decision_readiness_shadow未经独立验证，不得用于生产确认。
