# differential_presence

触发：仍有冲突、probable、insufficient 或 needs_review，且存在可区分的 required 特征。

候选来自 B 有正支持的前三和 A 逐折 Top-K（默认每折 3）映射去重后的兴趣集合；不缩小 B 规则候选池。P0 在 `adaptive_v2` 只是建议，不能按假设名称或 P0 等级强制调查路线。

工具：`presence_check(image_id, feature_id, morph_hint)`。只注册知识库和运行时允许的 required P1/C；具体权限以 ToolSpec 为准。

选法：优先候选 required 集合的差集，再按稳定顺序；`adaptive_v2` 的 LLM 从当轮预算内合法动作选短编号，由运行时校验和绑定参数。

有效观察：严格 `present` 布尔、`conf` 有限且 0～1。positive 且 conf≥0.5 只添加 `uncal_refs/conf_assumed`；禁止产生微米尺寸、标定或药材结论。

negative 不视为模型证明不存在，不写阴性卡；失败与低置信均计预算并去重。合格的新负观察可记为当前视野的动作进展。连续无收益可触发反思和停止建议，但不能把阴性或失败当成排他证据。
