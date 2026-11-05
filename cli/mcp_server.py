"""项目根启动绑定样品的MCP stdio服务；协议stdout仅由官方SDK写入。"""
import argparse

from TCM_Fusion.lib.mcp_bridge import make_server


def main(argv=None):
    parser = argparse.ArgumentParser(description="显微证据MCP stdio server")
    parser.add_argument("--context", required=True)
    args = parser.parse_args(argv)
    make_server(args.context).run(transport="stdio")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
