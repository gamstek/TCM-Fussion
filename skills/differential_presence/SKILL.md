# differential_presence

触发：仍有冲突、probable、insufficient 或 needs_review，且存在可区分的 required 特征。

候选来自 B 有正支持的前三和 A 逐折 Top-K（默认每折 3）映射去重后的兴趣集合；不缩小 B 规则候选池。P0 在 `adaptive_v2` 只是建议，不能按假设名称或 P0 等级强制调查路线。

工具：`presence_check(image_id, feature_id, morph_hint)`。只注册知识库和运行时允许的 required P1/C；具体权限以 ToolSpec 为准。

选法：优先候选 required 集合的差集，再按稳定顺序；`adaptive_v2` 的 LLM 从当轮预算内合法动作选短编号，由运行时校验和绑定参数。

有效观察：严格 `present` 布尔、`conf` 有限且 0～1。positive 且 conf≥0.5 只添加 `uncal_refs/conf_assumed`；禁止产生微米尺寸、标定或药材结论。

negative 不视为模型证明不存在，不写阴性卡；失败与低置信均计预算并去重。合格的新负观察可记为当前视野的动作进展。连续无收益可触发反思和停止建议，但不能把阴性或失败当成排他证据。

## 可选定位线索（PRESENCE-001，2026-11-13）

真实补查在同一次请求中可选返回 `box={bbox:[x1,y1,x2,y2],coordinate_space:"model_input_pixels"}`，坐标以实际发送图像为准。没有把握就不返回框；工具不会因此强制再调一次 GLM。运行时从实际载荷和原图建立映射、绑定 SHA256，不接受模型生成标定、变换或采信资格。

`presence_localization` 是独立影子诊断：`localization / quality / identity` 三轴与用途资格分别记录。合法框只意味着可解释坐标；小框、极端长宽比、贴边和重叠是线索，不能证明身份或完整长度。crop 紧框和扩展上下文分别保存，质量评估针对紧框，避免背景纹理掩盖空目标。

`present_pending`、`present_unlocalized` 不等于阴性；`not_observed_local` 只描述本次发送视野中的未观察到，失败、低置信保持 `unknown`。旧 P1/C 存在性规则继续兼容，影子用途资格不取代旧规则。当前没有人工核验集，不产生新的 `present_verified`；是否继续调用已有定位/分类工具由 Agent 结合缺口、预算及合法工具自行决定。影子 PNG 不是正式 CropMeta，不能经 `locate_override` 直接进入分割或测量。
