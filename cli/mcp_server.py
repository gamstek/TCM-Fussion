"""项目根启动绑定样品的 MCP stdio 服务 worker；协议 stdout 仅由官方 SDK 写入。

主要定义：
- main: 解析 `--context`，先 import cv2 完成原生库冷加载，再用 `lib.mcp_bridge.make_server(context).run(transport="stdio")` 提供显微证据工具。

关联：
- TCM_Fusion.lib.mcp_bridge: 提供 `make_server`，并以 `-m TCM_Fusion.cli.mcp_server --context ...` 子进程拉起本入口。
- TCM_Fusion.cli.workflow_worker、TCM_Fusion.cli.yolo_worker 等: 被 MCP 工具间接调用的检测/分类 worker 兄弟入口。
- 检测侧库（cv2 等）: 主线程预加载，避免 Windows asyncio worker 中的 DLL 冷启动阻塞。
"""
import argparse

from TCM_Fusion.lib.mcp_bridge import make_server


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
