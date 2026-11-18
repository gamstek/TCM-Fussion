# offline

离线数据划分、回放、评估、优化与发布。由 cli.offline 进入。

| 文件 | 阅读指引 |
| --- | --- |
| [ablation_evaluation.py](ablation_evaluation.py) | B0—B3 既有报告的配对评估与用途资格误拦/误放统计，不生成观察、不泄漏答案。 |
| [confidence_gate.py](confidence_gate.py) | 离线温度缩放与冻结策略的风险验证：真实放行只消费通过复验的版本，不自动造真值。 |
| [dataset_split.py](dataset_split.py) | 按 group_key 做确定性的训练/开发/测试划分，只产出数据清单，不推断标签。 |
| [outcome_evaluation.py](outcome_evaluation.py) | 运行后的独立真值评估：把轨迹结论与外部参考标签配对，不用模型自评充当真值。 |
| [prompt_optimization.py](prompt_optimization.py) | LangMem 离线提示优化与封闭工具回放：发布显式版本，在线只加载提示。 |
| [recorded_tools.py](recorded_tools.py) | 提示验证共享的观察回放接口，校验动作身份与录制冲突。 |
| [reflection.py](reflection.py) | 离线闭环反思：从程序可复核的轨迹反馈提出候选经验，经开发集回放后版本化发布。 |
| [truth_prompt_validation.py](truth_prompt_validation.py) | 带独立真值的提示策略双回放：新旧指令配对比较，允许有证据的纠错。 |
| [workflow_prompt_validation.py](workflow_prompt_validation.py) | 工作流提示的开发集双回放门：重放原 B 物理工件，不新发任何图像请求。 |
| [workflow_replay.py](workflow_replay.py) | 重算闭环的封闭回放：复用录制的物理工件与视觉观察，重跑规划链做对照评估。 |

直接依赖 agent、decision、runtime、tools、workflow。

对应回归见 [tests/offline](../../tests/offline/README.md)。

阶段回放按工具、对象和依赖指纹匹配。批量分类恢复逐裁图结果和失败码，示例审计恢复 JSON 和拼图。旧录制缺少必需字段时返回 REPLAY_MISS。

[返回上级导航](../README.md)
