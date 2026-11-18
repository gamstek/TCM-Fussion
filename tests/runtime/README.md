# runtime

进程、资源预算、环境就绪、代码签名及异常行为。

| 文件 | 阅读指引 |
| --- | --- |
| [test_code_guard.py](test_code_guard.py) | 代码签名冻结检查的离线回归：签名漂移即中止，不联网、不执行真实阶段。 |
| [test_probe_budget.py](test_probe_budget.py) | 支持性探测预算与 presence 上下文压缩的离线回归，不调用任何模型 API。 |
| [test_readiness.py](test_readiness.py) | 本地配置检查与启动前失败回归；替身只覆盖检查分支，绝不发起模型请求。 |
| [test_service_failures.py](test_service_failures.py) | 真实服务故障的离线反例：余额/鉴权失败停机、首个 HTTP 失败留痕、断连重试上限。 |
| [test_time_budget.py](test_time_budget.py) | 工作流时间预算回归：schema 接受十分钟级上限，实时模板保持参考时间预算。 |

统一命令和测试输入边界见 [tests 导航](../README.md)。
