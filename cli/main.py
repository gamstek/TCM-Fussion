"""提供真实任务、就绪检查、多图准备和统一回归命令。解析用户参数后分发给应用层。selftest 从项目根递归发现融合用例，并执行原 B 三套断言。

主要定义：
- main: 构建 argparse 子命令并按 `args.command` 分发。
- run: 要求 `agent.mode=active` 且含 `agent.workflow`，转交 `lib.application.pipeline.run_pipeline`。
- doctor: 转交 `lib.application.pipeline.doctor` 输出就绪报告，未就绪返回码 2。
- selftest: 用 `unittest` 递归发现 `TCM_Fusion/tests`，并额外执行原 B 的 rule_engine/adjudicator/scene 三组测试。
- multi-view: 转交 `lib.application.multi_view.run_multi_view`；图片数量由 manifest 决定，`--dry-run` 只准备与本地 doctor。
- five-view: 兼容旧名的等价别名，不带独立逻辑。
- 错误处理: 捕获 `IntegrationError/OSError/ValueError/KeyError`，按 `[CODE] message` 写 stderr 并返回 2。

关联：
- 直接依赖 「lib/application/multi_view.py」、「lib/application/pipeline.py」、「lib/runtime/storage.py」。
- 从项目根通过 python -m 启动。
"""
from __future__ import annotations
import argparse
import json
import subprocess
import sys
import unittest
from pathlib import Path
import paths
from TCM_Fusion.lib.application.pipeline import doctor, run_pipeline
from TCM_Fusion.lib.runtime.storage import IntegrationError, project_path, read_json


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
        item.add_argument("--subject", help="目录中的测试对象；盲测使用中性样本代号，默认未标注")
        item.add_argument("--purpose", help="目录中的测试目的，默认链路测试")
        item.add_argument("--name", help="兼容旧脚本的完整目录名；与 --subject/--purpose 互斥")
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
            from TCM_Fusion.lib.application.multi_view import run_multi_view
            directory = run_multi_view(args.source_dir, manifest=args.manifest,
                output_root=args.output_root, config_template=args.config_template,
                a_python=args.a_python, dry_run=args.dry_run, case_name=args.name,
                subject=args.subject, purpose=args.purpose)
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
