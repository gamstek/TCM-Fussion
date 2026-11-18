# worker_bootstrap

由受控子进程作为 Python 启动钩子加载，启用按任务记账的 HTTP 守卫。

| 文件 | 阅读指引 |
| --- | --- |
| [sitecustomize.py](sitecustomize.py) | worker 子进程的启动引导钩子：仅由显式 PYTHONPATH 启用，初始化失败直接退出。 |

直接依赖 runtime。

[返回上级导航](../README.md)
