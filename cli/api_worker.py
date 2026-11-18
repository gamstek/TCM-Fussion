"""固定 HTTP worker：由 hard_calls 以子进程启动，隔离网络阻塞并支撑整次硬截止。

主要定义：
- main: 解析 `--input` / `--output` 两个路径参数，转交 `lib.hard_calls.worker`。

关联：
- TCM_Fusion.lib.hard_calls: 提供 `worker` 实现，并以 `-m TCM_Fusion.cli.api_worker` 拉起本入口。
- TCM_Fusion.lib.worker_bootstrap: 子进程启动钩子，把 HTTP 请求纳入统一资源账本。
"""
import argparse
from TCM_Fusion.lib.hard_calls import worker


def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True)
    a=p.parse_args(argv);worker(a.input,a.output);return 0


if __name__=='__main__':raise SystemExit(main())
