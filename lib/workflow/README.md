# workflow

领域运行时、阶段会话与后端执行，处理动作提议、工件生命周期和资源约束。

分类回执保留原请求特征、拒绝原因和有效粒度。原框或分类指明束、片、团时，分割交付沿用粗粒度，测量账本保留存在性及 `coarse_quant`。量纲目标到此回交，后到的粗粒度复核也会撤回原测量的单体资格。新分割与有效复核工件绑定。

CNN 批量预测按裁图复用。同一任务内，模型、预处理、解释器、裁图 ID、像素和分类子窗的签名一致时读取既有预测。新增或替换裁图只派发缺失部分，完整批次按请求顺序组装。重叠批次按固定顺序获得逐裁图锁，预测验证后原子发布；完整旧缓存仍须匹配整批签名。D4 轨迹的 `physical_epoch_timing` 分别记录组内分支秒数之和、执行区间并集和重叠分支秒数。缺时间戳的分支只计入总分支数。该指标用于定位并行工作，整案净提速通过配对测试衡量。

| 文件 | 阅读指引 |
| --- | --- |
| [adapters.py](adapters.py) | 通过子进程调用原 A/原 B 入口，绝不 import 入口模块或复制其预处理算法。 |
| [evidence_runtime.py](evidence_runtime.py) | 构造 Agent 领域运行时：原测量→规则卡池→动作/观察→原 decide，只读原图、不修改原 B 工件。 |
| [workflow_backend.py](workflow_backend.py) | 接收会话授权的单视野或单裁图目标并分派实际阶段。 |
| [workflow_runtime.py](workflow_runtime.py) | 以原图、A 候选与工作流配置创建调查运行时。 |
| [workflow_session.py](workflow_session.py) | 绑定样品、输入、运行策略与阶段后端，维护可恢复的操作记录和时间线。 |
| [physical_schedule.py](physical_schedule.py) | 为显式选择的 2–4 个跨视野动作同时预约额度，冻结各自依赖和写入范围。worker 准备私有工件，协调者逐支发布。关闭或接管时按已发布版本中的回执对账。 |

直接依赖 decision、evidence、runtime、tools、vision。

对应回归见 [tests/workflow](../../tests/workflow/README.md)。

[返回上级导航](../README.md)
