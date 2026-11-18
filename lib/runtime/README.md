# runtime

工作流运行基础，包括路径转发、进程、文件锁、预算、签名和内部 schema 校验。

| 文件 | 阅读指引 |
| --- | --- |
| [code_guard.py](code_guard.py) | 持有任务开始时的运行签名，由入口显式检查并可在调查期间定期复查。 |
| [controlled_process.py](controlled_process.py) | 根据固定命令、预算上下文和日志路径启动阶段子进程。 |
| [hard_calls.py](hard_calls.py) | 为视觉编码与模型请求提供可终止的子进程执行通道，临时载荷用完即删。 |
| [ledger_lock.py](ledger_lock.py) | 标准库实现的 OS 文件锁：串行化同一账本上的跨实例/跨进程观察写入。 |
| [probe_budget.py](probe_budget.py) | 候选绑定的 supporting 探测到共享 resources.json 预留：只是控制元数据，不是显微证据。 |
| [resource_budget.py](resource_budget.py) | 以文件锁保护共享资源账本。 |
| [runtime_readiness.py](runtime_readiness.py) | 读取模型配置与调用环境，检查模型名、服务根地址和凭据配置的一致性。 |
| [schema.py](schema.py) | 组件内部 JSON Schema 校验。 |
| [service_errors.py](service_errors.py) | 将阶段和模型异常归为有限诊断码。 |
| [storage.py](storage.py) | 集成层的基础设施：路径权威、原子 JSON 读写、哈希/摘要与输入快照工具。 |

对应回归见 [tests/runtime](../../tests/runtime/README.md)。

[worker 启动钩子](worker_bootstrap/README.md)由进程管理层加载。

[返回上级导航](../README.md)
