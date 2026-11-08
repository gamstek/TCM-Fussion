"""真实入口：项目根 python -m TCM_Fusion.cli.main run|doctor|selftest|multi-view。

five-view 是兼容别名；任意数量原图统一走 lib.multi_view。离线命令统一在 cli.offline。
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
        package = Path(paths.PROJECT_ROOT) / "TCM_Fusion"
        suite = unittest.defaultTestLoader.discover(str(package / "tests"), top_level_dir=paths.PROJECT_ROOT)
        passed = unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful()
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
