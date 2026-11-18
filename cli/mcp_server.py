"""项目根启动绑定样品的 MCP stdio 服务 worker；协议 stdout 仅由官方 SDK 写入。
解析 `--context`，先 import cv2 完成原生库冷加载，再用 `lib.mcp_bridge.make_server(context).run(transport="stdio")` 提供显微证据工具。从项目根以模块方式启动。读取命令参数或任务上下文，并通过约定的结果文件或退出码交付。

主要定义：
- main: 解析 `--context`，先 import cv2 完成原生库冷加载，再用 `lib.mcp_bridge.make_server(context).run(transport="stdio")` 提供显微证据工具。

关联：
- 直接依赖 「lib/mcp/mcp_bridge.py」。
- 从项目根通过 python -m 启动。
"""
import argparse

from TCM_Fusion.lib.mcp.mcp_bridge import make_server


def main(argv=None):
    parser = argparse.ArgumentParser(description="显微证据MCP stdio server")
    parser.add_argument("--context", required=True)
    args = parser.parse_args(argv)
    # Native DLL cold imports can stall in Windows asyncio workers. Load the
    # existing numerical runtime on the main thread before accepting tools.
    import cv2  # noqa: F401
    make_server(args.context).run(transport="stdio")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
