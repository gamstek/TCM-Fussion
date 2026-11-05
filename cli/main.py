"""项目根执行：python -m TCM_Fusion.cli.main demo|doctor|run|selftest。

原 A/B 通过子进程调用，融合代码归 TCM_Fusion。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
import unittest
from pathlib import Path

import paths
from TCM_Fusion.lib.pipeline import doctor, run_pipeline
from TCM_Fusion.lib.storage import IntegrationError, project_path, read_json


def main(argv=None):
    parser = argparse.ArgumentParser(description="A/B融合与LangGraph主动ReAct证据闭环")
    sub = parser.add_subparsers(dest="command", required=True)
    quality=sub.add_parser('quality-replay',help='零API重放已有生成图，写新QA/用途资格')
    quality.add_argument('--generation-dir',required=True);quality.add_argument('--crop',required=True)
    quality.add_argument('--out-dir',required=True)
    outcome=sub.add_parser('outcome-evaluate',help='运行后独立真值反馈，不给在线Agent答案')
    outcome.add_argument('--trace',required=True);outcome.add_argument('--reference',required=True)
    outcome.add_argument('--a-label');outcome.add_argument('--out',required=True)
    ablate=sub.add_parser('ablation-report',help='B0—B3已完成报告的配对评估，不联网')
    ablate.add_argument('--dataset',required=True);ablate.add_argument('--out',required=True)
    quality_report=sub.add_parser('quality-report',help='人工用途参考的误拦/误放统计，不拟合阈值')
    quality_report.add_argument('--dataset',required=True);quality_report.add_argument('--out',required=True)
    for command in ("run", "doctor"):
        p = sub.add_parser(command)
        p.add_argument("--config", required=True, help="JSON配置；内部相对路径以项目根为基准")
    for name,fields in (("gate-fit",["dataset"]),("gate-validate",["calibration","policy","dataset"]),("gate-publish",["report"])):
        gate=sub.add_parser(name,help="离线校准/独立风险验证/复验发布，不调用模型")
        for field in fields:gate.add_argument("--"+field,required=True)
    replay=sub.add_parser("workflow-replay",help="封闭物理工件及观察回放，不新发看图请求")
    replay.add_argument("--report",required=True)
    split=sub.add_parser("dataset-split",help="按group_key确定性划分训练/开发/测试，防止调参泄漏")
    split.add_argument("--attributes",nargs="+",required=True,help="一个或多个attributes.csv")
    split.add_argument("--out",required=True,help="输出dataset-split/v1 JSON")
    split.add_argument("--seed",type=int,default=20261104)
    split.add_argument("--ratios",default="0.7,0.15,0.15")
    split.add_argument("--explicit",default="",help="可选JSON：group_key->split 人工覆盖")
    sub.add_parser("demo", help="纯离线模拟，不执行真实模型")
    sub.add_parser("reflection-demo", help="离线反思→回放验证→发布→后续运行复用，不联网")
    sub.add_parser("prompt-demo", help="实际LangMem优化器与封闭回放演示；模型为替身，不联网")
    for name, fields in (("prompt-optimize", ["training"]), ("prompt-validate", ["candidate", "dataset"]),
                         ("prompt-publish", ["candidate", "report"])):
        command = sub.add_parser(name, help="LangMem离线文本优化/封闭工具回放/版本发布")
        for field in fields:
            command.add_argument("--"+field, required=True)
        command.add_argument("--allow-network", action="store_true")
    reflect = sub.add_parser("reflect", help="从轨迹和程序反馈生成候选经验")
    reflect.add_argument("--trace", required=True)
    reflect.add_argument("--feedback", required=True)
    reflect.add_argument("--llm", action="store_true")
    reflect.add_argument("--allow-network", action="store_true")
    verify = sub.add_parser("reflection-validate")
    verify.add_argument("--candidate", required=True)
    verify.add_argument("--cases", required=True)
    release = sub.add_parser("reflection-publish")
    release.add_argument("--candidate", required=True)
    release.add_argument("--report", required=True)
    react_demo = sub.add_parser("agent-demo", help="主动ReAct离线闭环，规划/观察为模拟替身")
    react_demo.add_argument("--transport", choices=["local", "mcp"], default="local")
    react_demo.add_argument("--top-k", type=int, default=None, help="启用Top-K候选与只读工具，1～68")
    sub.add_parser("selftest", help="聚合本包测试与原B三条回归，不联网")
    args = parser.parse_args(argv)
    package = Path(paths.PROJECT_ROOT)/"TCM_Fusion"
    if args.command == "selftest":
        suite = unittest.defaultTestLoader.discover(str(package/"tests"), top_level_dir=paths.PROJECT_ROOT)
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        passed = result.wasSuccessful()
        for name in ("test_rule_engine.py", "test_adjudicator.py", "test_scene.py"):
            cp = subprocess.run([sys.executable, "-X", "utf8", str(Path(paths.JUDGE_DIR)/"tests"/name)],
                                cwd=paths.PROJECT_ROOT)
            passed = passed and cp.returncode == 0
        return 0 if passed else 1
    try:
        if args.command=='quality-replay':
            import contracts
            from quality_eval import replay
            value=replay(project_path(args.generation_dir),contracts.CropMeta.from_json(str(project_path(args.crop))),project_path(args.out_dir))
            print(json.dumps(value,ensure_ascii=False,indent=2));return 0
        if args.command in {'outcome-evaluate','ablation-report','quality-report'}:
            from TCM_Fusion.lib.storage import write_json
            if args.command=='outcome-evaluate':
                from TCM_Fusion.lib.outcome_evaluation import evaluate
                value=evaluate(read_json(project_path(args.trace)),read_json(project_path(args.reference)),a_label=args.a_label)
            elif args.command=='ablation-report':
                from TCM_Fusion.lib.ablation_evaluation import compare
                value=compare(read_json(project_path(args.dataset)))
            else:
                from TCM_Fusion.lib.ablation_evaluation import quality_compare
                value=quality_compare(read_json(project_path(args.dataset)))
            write_json(project_path(args.out),value);print(args.out);return 0
        if args.command in {"gate-fit","gate-validate","gate-publish"}:
            from TCM_Fusion.lib import confidence_gate as G
            if args.command == "gate-fit":
                value=G.fit(read_json(project_path(args.dataset)));kind="calibrations"
            elif args.command == "gate-validate":
                value=G.validate(read_json(project_path(args.calibration)),read_json(project_path(args.policy)),
                                 read_json(project_path(args.dataset)));kind="validations"
            else:
                value=G.publish(read_json(project_path(args.report)));kind="releases"
            print(G.save(kind,value))
            return 2 if value.get("passed") is False else 0
        if args.command == "workflow-replay":
            from TCM_Fusion.lib.workflow_replay import replay_report
            from TCM_Fusion.lib.storage import output_root
            import uuid
            directory=output_root()/"workflow_replay"/uuid.uuid4().hex
            value=asyncio.run(replay_report(read_json(project_path(args.report)),directory))
            print(directory/"report.json")
            return 0 if value["covered"] else 2
        if args.command == "dataset-split":
            from TCM_Fusion.lib import dataset_split as S
            explicit = read_json(project_path(args.explicit)) if args.explicit else None
            value = S.build(args.attributes, seed=args.seed, ratios=args.ratios, explicit=explicit)
            print(S.write(value, args.out))
            return 0
        if args.command == "prompt-demo":
            from TCM_Fusion.lib.prompt_demo import demo
            print(asyncio.run(demo()))
            return 0
        if args.command in {"prompt-optimize", "prompt-validate", "prompt-publish"}:
            from TCM_Fusion.lib import prompt_optimization as P
            if args.command == "prompt-optimize":
                value = asyncio.run(P.optimize(read_json(project_path(args.training)), allow_network=args.allow_network))
                print(P.save("candidates", value))
            elif args.command == "prompt-validate":
                value = asyncio.run(P.validate(read_json(project_path(args.candidate)),
                    read_json(project_path(args.dataset)), allow_network=args.allow_network))
                print(P.save("validations", value))
                return 0 if value["passed"] else 2
            else:
                print(asyncio.run(P.publish(read_json(project_path(args.candidate)),
                    read_json(project_path(args.report)), allow_network=args.allow_network)))
            return 0
        if args.command == "reflection-demo":
            from TCM_Fusion.lib.reflection_demo import demo
            print(asyncio.run(demo()))
            return 0
        if args.command in {"reflect", "reflection-validate", "reflection-publish"}:
            from TCM_Fusion.lib.reflection import propose, validate, publish, save_artifact
            if args.command == "reflect":
                value = propose(read_json(project_path(args.trace)), read_json(project_path(args.feedback)),
                                llm=args.llm, allow_network=args.allow_network)
                print(save_artifact("candidates", value))
            elif args.command == "reflection-validate":
                value = asyncio.run(validate(read_json(project_path(args.candidate)), read_json(project_path(args.cases))))
                print(save_artifact("validations", value))
                return 0 if value["passed"] else 2
            else:
                print(asyncio.run(publish(read_json(project_path(args.candidate)), read_json(project_path(args.report)))))
            return 0
        if args.command == "agent-demo":
            from TCM_Fusion.lib.agent_demo import demo_react
            directory = asyncio.run(demo_react(transport=args.transport, top_k=args.top_k))
            print(f"主动闭环演示：{directory / 'report.md'}")
            print("LangGraph/规则重评分实际执行；LLM/VLM观察为模拟，非识别效果证据。")
            return 0
        if args.command == "demo":
            config = read_json(package/"config"/"demo.json")
        else:
            config = read_json(project_path(args.config))
        if args.command == "doctor":
            report = doctor(config)
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return 0 if report["ready"] else 2
        directory = run_pipeline(config)
        print(f"完成：{directory}")
        print(f"报告：{directory / 'report.md'}")
        print(f"Agent模式：{config.get('agent', {}).get('mode', 'deterministic_shadow')}；demo/agent-demo不代表真实模型效果。")
        return 0
    except (IntegrationError, OSError, ValueError, KeyError) as exc:
        print(f"[{getattr(exc, 'code', 'CONFIG_OR_IO_ERROR')}] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
