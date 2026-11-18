# workflow

工作流权限、阶段执行、几何工件、重试与证据提交。

| 文件 | 阅读指引 |
| --- | --- |
| [test_retry_policy.py](test_retry_policy.py) | 工具级有界重试策略的离线回归：验证失败计数只在允许范围内放大为熔断。 |
| [test_workflow.py](test_workflow.py) | 在真实会话和注册表上检查工件提交、身份门、预算、重试与去重。 |
| [test_physical_schedule.py](test_physical_schedule.py) | 用两个视野验证 D4 的实际执行重叠、分类、快支提前推进、慢支撤销、额度、幂等、过期与提交中断恢复。阶段模型使用替身，图、文件锁、预算和证据库走正式实现。 |

统一命令和测试输入边界见 [tests 导航](../README.md)。
