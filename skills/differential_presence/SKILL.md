# differential_presence

触发：仍有冲突、probable、insufficient或needs_review。候选来自有正支持的 B 前三及 A 逐折 Top-K（默认每折 3）映射去重后的兴趣集合；不缩小B规则候选池。
工具：presence_check(image_id, feature_id, morph_hint)。仅注册知识库中的required P1/C；禁止P0。
选法：候选required集合的差集优先，再按固定稳定顺序；adaptive_v2 的 LLM 从当轮预算内合法动作选短编号，由运行时校验和绑定参数。
有效观察：严格present布尔、conf有限且0～1。positive且conf≥0.5只添加uncal_refs/conf_assumed；禁止产生微米尺寸、标定或药材结论。
negative不视为模型证明不存在，不写阴性卡；失败与低置信均计预算并去重。合格的新负观察可记为当前视野的动作进展。连续无收益可触发反思和停止建议。
