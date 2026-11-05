"""仅由显式PYTHONPATH启用的受控worker启动钩子；初始化失败直接退出。"""
import os
if os.environ.get('FUSION_BUDGET_CONTEXT'):
    try:
        from TCM_Fusion.lib.resource_budget import install_http_guard
        install_http_guard(os.environ['FUSION_BUDGET_CONTEXT'])
    except Exception:
        os._exit(73)
