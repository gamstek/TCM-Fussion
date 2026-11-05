"""项目根启动绑定样品的MCP stdio服务；协议stdout仅由官方SDK写入。"""
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
