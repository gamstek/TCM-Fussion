"""项目主 CLI 入口：`python -m TCM_Fusion.cli.main run|doctor|selftest|multi-view|five-view`。

主要定义：
- main: 构建 argparse 子命令并按 `args.command` 分发。
- run: 要求 `agent.mode=active` 且含 `agent.workflow`，转交 `lib.pipeline.run_pipeline`。
- doctor: 转交 `lib.pipeline.doctor` 输出就绪报告，未就绪返回码 2。
- selftest: 用 `unittest` 递归发现 `TCM_Fusion/tests`，并额外执行原 B 的 rule_engine/adjudicator/scene 三组测试。
- multi-view: 转交 `lib.multi_view.run_multi_view`；图片数量由 manifest 决定，`--dry-run` 只准备与本地 doctor。
- five-view: 兼容旧名的等价别名，不带独立逻辑。
- 错误处理: 捕获 `IntegrationError/OSError/ValueError/KeyError`，按 `[CODE] message` 写 stderr 并返回 2。

关联：
- TCM_Fusion.lib.pipeline: 提供 doctor 与 run_pipeline。
- TCM_Fusion.lib.multi_view: 任意数量原图的建任务、配置准备与工作流启动。
- TCM_Fusion.lib.storage: 配置读取、路径解析与 IntegrationError。
- TCM_Fusion.cli.offline: 离线回放/评估/优化/发布统一入口，不在本文件。
"""
from __future__ import annotations
import argparse
import json
import subprocess
import sys
import unittest
from pathlib import Path
import paths
from TCM_Fusion.lib.pipeline import doctor, run_pipeline
from TCM_Fusion.lib.storage import IntegrationError, project_path, read_json


def main(argv=None):
    parser = argparse.ArgumentParser(description="真实A/B融合与VLM Agent")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "doctor"):
        item = sub.add_parser(name)
        item.add_argument("--config", required=True)
    sub.add_parser("selftest", help="离线回归含原B三组测试")
    for name, help_text in (("multi-view", "任意数量原图→新任务→doctor→真实MCP工作流"),
                            ("five-view", "兼容旧名；等价于multi-view")):
        item = sub.add_parser(name, help=help_text)
        item.add_argument("--source-dir", required=True)
        item.add_argument("--manifest")
        item.add_argument("--output-root")
        item.add_argument("--config-template")
        item.add_argument("--name", help="case目录名；sample_id会另做ASCII清洗")
        item.add_argument("--a-python", help="A模型解释器，或设TCM_A_PYTHON；不自动切环境")
        item.add_argument("--dry-run", action="store_true", help="只准备和本地doctor，不调用模型")
        item.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "selftest":
        import gc
        class RegressionResult(unittest.TextTestResult):
            def stopTest(self, test):
                super().stopTest(test)
                # 用例构造闭环运行时/图回调；结束后回收循环引用，避免整套累计占用。
                # 不改断言、时限或生产运行的GC/线程配置。
                gc.collect()
        package = Path(paths.PROJECT_ROOT) / "TCM_Fusion"
        suite = unittest.defaultTestLoader.discover(str(package / "tests"), top_level_dir=paths.PROJECT_ROOT)
        passed = unittest.TextTestRunner(verbosity=2, resultclass=RegressionResult).run(suite).wasSuccessful()
        for name in ("test_rule_engine.py", "test_adjudicator.py", "test_scene.py"):
            cp = subprocess.run([sys.executable, "-X", "utf8", str(Path(paths.JUDGE_DIR) / "tests" / name)], cwd=paths.PROJECT_ROOT)
            passed = cp.returncode == 0 and passed
        return 0 if passed else 1
    try:
        if args.command in {"multi-view", "five-view"}:
            from TCM_Fusion.lib.multi_view import run_multi_view
            directory = run_multi_view(args.source_dir, manifest=args.manifest,
                output_root=args.output_root, config_template=args.config_template,
                a_python=args.a_python, dry_run=args.dry_run, case_name=args.name)
            value = {"directory": str(directory), "dry_run": args.dry_run,
                     "status": "prepared" if args.dry_run else "completed"}
            print(json.dumps(value, ensure_ascii=False) if args.json else str(directory))
            return 0
        config = read_json(project_path(args.config))
        if args.command == "doctor":
            report = doctor(config)
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return 0 if report["ready"] else 2
        if config.get('agent', {}).get('mode') != 'active' or 'workflow' not in config.get('agent', {}):
            raise IntegrationError('CONFIG_INVALID', '真实run入口要求active + agent.workflow；旧模拟仅在回归测试中使用')
        directory = run_pipeline(config)
        print(json.dumps({"directory": str(directory), "report": str(directory / "report.md")}, ensure_ascii=False))
        return 0
    except (IntegrationError, OSError, ValueError, KeyError) as exc:
        print(f"[{getattr(exc, 'code', 'CONFIG_OR_IO_ERROR')}] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
