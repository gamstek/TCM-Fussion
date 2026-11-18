"""worker 子进程的启动引导钩子：仅由显式 PYTHONPATH 启用，初始化失败直接退出。
读取环境变量 `FUSION_BUDGET_CONTEXT`，存在时安装 HTTP 守卫。

主要定义：
- 模块级启动逻辑: 读取环境变量 `FUSION_BUDGET_CONTEXT`，存在时安装 HTTP 守卫。
- FUSION_BUDGET_CONTEXT: 指向资源账本上下文的真实环境变量名，缺失则跳过安装。
- install_http_guard: 从 `lib.resource_budget` 引入的守卫安装函数；安装异常时 `os._exit(73)` 终止子进程。

关联：
- 直接依赖 「lib/runtime/resource_budget.py」。
- 由 runtime/controlled_process.py 设置 PYTHONPATH 触发。
"""
import os
if os.environ.get('FUSION_BUDGET_CONTEXT'):
    try:
        from TCM_Fusion.lib.runtime.resource_budget import install_http_guard
        install_http_guard(os.environ['FUSION_BUDGET_CONTEXT'])
    except Exception:
        os._exit(73)
