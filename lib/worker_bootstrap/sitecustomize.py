"""worker 子进程的启动引导钩子：仅由显式 PYTHONPATH 启用，初始化失败直接退出。

主要定义：
- 模块级启动逻辑: 读取环境变量 `FUSION_BUDGET_CONTEXT`，存在时安装 HTTP 守卫。
- FUSION_BUDGET_CONTEXT: 指向资源账本上下文的真实环境变量名，缺失则跳过安装。
- install_http_guard: 从 `lib.resource_budget` 引入的守卫安装函数；安装异常时 `os._exit(73)` 终止子进程。

关联：
- TCM_Fusion.lib.resource_budget: 提供 `install_http_guard`，把 worker 的出网请求记入统一账本。
- TCM_Fusion.lib.worker_bootstrap: 本文件所属的引导包，由 worker 启动命令注入 PYTHONPATH。
- TCM_Fusion.cli.api_worker、TCM_Fusion.cli.classify_worker 等: 受该钩子约束的 HTTP 请求子进程。
"""
import os
if os.environ.get('FUSION_BUDGET_CONTEXT'):
    try:
        from TCM_Fusion.lib.resource_budget import install_http_guard
        install_http_guard(os.environ['FUSION_BUDGET_CONTEXT'])
    except Exception:
        os._exit(73)
