"""读取调用方写入的请求 JSON，运行图像编码或模型请求，并将结果写入指定路径。负责命令行参数与进程退出码，实际处理由 runtime.hard_calls 提供。

主要定义：
- main: 解析 `--input` / `--output` 两个路径参数，转交 `lib.hard_calls.worker`。

关联：
- 直接依赖 「lib/runtime/hard_calls.py」。
- 从项目根通过 python -m 启动。
"""
import argparse
from TCM_Fusion.lib.runtime.hard_calls import worker


def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True)
    a=p.parse_args(argv);worker(a.input,a.output);return 0


if __name__=='__main__':raise SystemExit(main())
