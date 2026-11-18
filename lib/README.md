# 融合组件导航

从 application 开始追踪完整运行，从具体组件定位单项维护。组件依赖和验收见 [重组方案](../融合链路文档/21_TCM_Fusion组件重组方案.md)。

| 组件 | 职责 |
| --- | --- |
| [application](application/README.md) | 任务与交付编排，从此处追踪 run、doctor、multi-view 进入的 A/B 链路。 |
| [agent](agent/README.md) | 规划与反馈，从 react_graph 跟踪可见动作、模型选择、观察回写以及任务内反思。 |
| [mcp](mcp/README.md) | MCP 的 stdio 协议适配，绑定本次样品并提供正式工具箱的调用入口。 |
| [tools](tools/README.md) | 工具注册和执行边界，处理权限、观察与只读详情。模拟后端只接受显式模拟上下文。 |
| [workflow](workflow/README.md) | 领域运行时、阶段会话与后端执行，处理动作提议、工件生命周期和资源约束。 |
| [vision](vision/README.md) | 图像与模型适配，处理出框、分块、特征身份、裁图质量和诊断审计。 |
| [evidence](evidence/README.md) | 证据与工件，包括内容寻址版本、实例关联、用途投影和录制元信息。 |
| [decision](decision/README.md) | 概率分流、候选证据解释、A/B 融合和对外交付。 |
| [runtime](runtime/README.md) | 工作流运行基础，包括路径转发、进程、文件锁、预算、签名和内部 schema 校验。 |
| [offline](offline/README.md) | 离线数据划分、回放、评估、优化与发布。由 cli.offline 进入。 |

[返回融合入口](../README.md)
