"""由hard_calls启动的固定HTTP worker；隔离网络阻塞以执行整次硬截止。"""
import argparse
from TCM_Fusion.lib.hard_calls import worker


def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True)
    a=p.parse_args(argv);worker(a.input,a.output);return 0


if __name__=='__main__':raise SystemExit(main())
