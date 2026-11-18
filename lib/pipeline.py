"""统一集成入口：配置→A 缓存/原 CLI→原 B/完成工件→融合→影子 Agent→独立报告。

结果全部位于 paths.UI_OUT_ROOT；原代码、图片与 B verdict 均不被修改。

主要定义：
- validate_config: 校验配置字段与取值，拒绝未知键。
- run_pipeline: 建独立任务目录，执行 A 推理、路由、工作流或融合，并写 job/fusion/report。
- doctor: 只读环境检查，不加载模型、不下载权重、不联网。

关联：
- TCM_Fusion.lib.routing_policy / candidate_policy: 路由与候选策略校验。
- TCM_Fusion.lib.adapters: predict_a/execute_b 调用原 A/B 入口。
- TCM_Fusion.lib.collaboration: fuse/knowledge_base/select_shadow。
- TCM_Fusion.lib.workflow_entry: execute_workflow 执行自主 B 工作流。
- TCM_Fusion.lib.workflow_backend: 启动前后比对代码签名。
- TCM_Fusion.lib.stage2_classifier: doctor 检查 stage2 模型/元信息。
- TCM_Fusion.lib.multi_view: 多图建任务后转交 run_pipeline/doctor。
- TCM_Fusion.cli.main: CLI 的 run/doctor 子命令最终调用本模块。
"""
from __future__ import annotations

import copy
import asyncio
import json
import re
import subprocess
import time
import uuid
from pathlib import Path

import paths

from .routing_policy import validate_policy, route
from .candidate_policy import validate_candidates
from .adapters import execute_b, predict_a
from .collaboration import fuse, knowledge_base, select_shadow
from .storage import (IntegrationError, digest, output_root, project_path,
                      read_json, write_json)


def validate_config(config):
    if not isinstance(config, dict):
        raise IntegrationError("CONFIG_INVALID", "配置须为JSON对象")
    allowed = {"sample_id", "a", "b", "agent", "allow_network", "allow_recorded", "aliases", "routing"}
    if set(config)-allowed:
        raise IntegrationError("CONFIG_INVALID", "存在未知顶层配置字段")
    for name in ("a", "b", "agent"):
        if not isinstance(config.get(name, {}), dict):
            raise IntegrationError("CONFIG_INVALID", f"{name}须为对象")
    for name in ("allow_network", "allow_recorded"):
        if not isinstance(config.get(name, False), bool):
            raise IntegrationError("CONFIG_INVALID", f"{name}须为布尔值")
    validate_policy(config.get("routing"))
    validate_candidates(config.get("agent", {}).get("candidate_policy"))
    sample = config.get("sample_id", "sample")
    if not isinstance(sample, str) or not re.fullmatch(r"[\w-]{1,64}", sample):
        raise IntegrationError("CONFIG_INVALID", "sample_id只允许1～64个字母/数字/中文/下划线/连字符")
    modes = (config.get("a", {}).get("mode", "cli"), config.get("b", {}).get("mode", "run"))
    if "recorded" in modes and not config.get("allow_recorded", False):
        raise IntegrationError("RECORDED_DISABLED", "模拟记录必须显式allow_recorded=true")
    if not isinstance(config.get("aliases", {}), dict) or any(
            not isinstance(k, str) or not isinstance(v, str) for k, v in config.get("aliases", {}).items()):
        raise IntegrationError("CONFIG_INVALID", "aliases须为字符串映射")
    fields = {
        "a": {"mode", "required", "image", "images", "checkpoint_dir", "python", "record", "timeout_seconds"},
        "b": {"mode", "input", "input_mode", "images", "manifest", "python", "enable_vlm", "record", "run_dir"},
        "agent": {"mode", "judgement_request_budget", "max_steps", "planner", "transport", "image_roots", "prompt_release", "candidate_policy", "workflow"},
    }
    for section, names in fields.items():
        if set(config.get(section, {}))-names:
            raise IntegrationError("CONFIG_INVALID", f"{section}含未支持字段")
    for section, field in (("a", "required"), ("b", "enable_vlm")):
        if field in config.get(section, {}) and not isinstance(config[section][field], bool):
            raise IntegrationError("CONFIG_INVALID", f"{section}.{field}须为布尔值")
    a, b = config.get("a", {}), config.get("b", {})
    agent = config.get("agent", {})
    if "images" in a:
        views = a["images"]
        if (a.get("mode", "cli") != "cli" or "image" in a or not isinstance(views, list)
                or len(views) < 1 or any(not isinstance(p, str) or not p for p in views)
                or len(set(views)) != len(views) or config.get("routing")
                or agent.get("mode") != "active" or "workflow" not in agent):
            raise IntegrationError("CONFIG_INVALID", "A.images 仅用于无路由的主动多视野工作流，须为不同原图")
    for section, mode, choices in (
        ("A", a.get("mode", "cli"), {"cli", "recorded", "unavailable"}),
        ("B", b.get("mode", "run"), {"run", "existing", "recorded"}),
        ("Agent", agent.get("mode", "deterministic_shadow"), {"off", "deterministic_shadow", "llm_shadow", "active"}),
        ("B.input_mode", b.get("input_mode", "images"), {"images", "crops", "masks"}),
    ):
        if mode not in choices:
            raise IntegrationError("CONFIG_INVALID", f"{section}模式不支持")
    budget = agent.get("judgement_request_budget", 6)
    if isinstance(budget, bool) or not isinstance(budget, int) or budget < 0:
        raise IntegrationError("CONFIG_INVALID", "Agent预算须为非负整数")
    if agent.get("mode") == "llm_shadow" and not config.get("allow_network", False):
        raise IntegrationError("NETWORK_DISABLED", "LLM shadow需要显式允许联网")
    if "workflow" in agent:
        from .workflow_entry import WORKFLOW_SCHEMA
        from .tool_registry import validate_arguments
        validate_arguments(WORKFLOW_SCHEMA,agent["workflow"])
        if agent["workflow"].get("targeted_recovery") and (
                agent.get("planner", "llm") != "llm"
                or agent["workflow"].get("agent_policy", "adaptive_v2") != "adaptive_v2"):
            raise IntegrationError("CONFIG_INVALID", "定向补救需要 adaptive_v2 模型规划")
        if agent.get("mode") != "active":
            raise IntegrationError("CONFIG_INVALID", "workflow需要active模式")
        if a.get('mode')=='recorded':
            raise IntegrationError('CONFIG_INVALID','真实workflow不接受模拟A；可显式设置A不可用')
    if agent.get("mode") == "active":
        if "workflow" not in agent:
            raise IntegrationError("CONFIG_INVALID", "主动模式必须提供 agent.workflow；旧运行器已退役")
        if (not config.get("allow_network", False)
            and ("workflow" not in agent or agent.get("planner", "llm") != "heuristic")):
            raise IntegrationError("NETWORK_DISABLED", "主动模式须显式允许联网")
        if not isinstance(agent.get("image_roots"), list) or not agent["image_roots"]:
            raise IntegrationError("CONFIG_INVALID", "主动模式须提供同样品image_roots")
        if b.get("mode") == "recorded" or b.get("enable_vlm", False):
            raise IntegrationError("CONFIG_INVALID", "主动模式禁止录制B与旧VLM预补查；workflow可从原图/部分工件开始，主动运行统一走workflow")
        if agent.get("planner", "llm") not in {"llm", "heuristic"} or agent.get("transport", "local") not in {"local", "mcp"}:
            raise IntegrationError("CONFIG_INVALID", "主动planner/transport非法")
        steps = agent.get("max_steps", 6)
        if not isinstance(steps, int) or isinstance(steps, bool) or not 1 <= steps <= 100 or budget > 100:
            raise IntegrationError("CONFIG_INVALID", "主动步数1～100、请求预算0～100")
    if (a.get("mode", "cli") == "cli" and b.get("mode", "run") == "run"
            and b.get("input_mode", "images") == "images" and a.get("image") and b.get("input")):
        if not project_path(a["image"]).is_relative_to(project_path(b["input"])):
            raise IntegrationError("SAMPLE_MISMATCH", "A图像须位于本次B原图目录，不能融合不同样品")
    validated = copy.deepcopy(config)
    if validated.get("routing") and "candidate_policy" not in validated.get("agent", {}):
        validated.setdefault("agent", {})["candidate_policy"] = {"top_k": validated["routing"].get("top_k", 3)}
    return validated


def run_pipeline(config, *, base=None):
    cfg = validate_config(config)
    identifier = "fusion_"+uuid.uuid4().hex
    base = Path(base) if base is not None else output_root()
    directory = base/"runs"/identifier
    directory.mkdir(parents=True, exist_ok=False)
    status_path = directory/"job.json"
    job = {"job_id": identifier, "sample_id": cfg.get("sample_id", "sample"),
           "status": "running", "stage": "a", "config_sha256": digest(cfg),
           "result": str(directory/"fusion.json")}
    write_json(status_path, job)
    code_baseline = None
    if "workflow" in cfg.get("agent", {}):
        from .workflow_backend import WorkflowBackend
        code_baseline = WorkflowBackend.code_signature()
        write_json(directory/"code_signature_start.json",
                   {"schema": "code-signature/v1", "stage": "pipeline_start",
                    "code_signature": code_baseline,
                    "at": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
    start = time.perf_counter()
    try:
        try:
            a_cfg = cfg.get("a", {})
            if "images" in a_cfg:
                settings = {k: v for k, v in a_cfg.items() if k != "images"}
                views = [predict_a({**settings, "image": image}, base/"a_cache")
                         for image in a_cfg["images"]]
                a = {"status": "ok", "ranking": [], "provenance": "live_multiview",
                     "view_predictions": views}
            else:
                a = predict_a(a_cfg, base/"a_cache")
        except IntegrationError as exc:
            if cfg.get("a", {}).get("required", True):
                raise
            a = {"status": "unavailable", "ranking": [], "provenance": "unavailable", "error_code": exc.code}
        write_json(directory/"a_prediction.json", a)
        routing = route(a, cfg.get("routing"), knowledge_base(), cfg.get("aliases"))
        write_json(directory/"routing.json", routing)
        if routing["skip"]:
            calibrated = routing["mode"] == "validated"
            limitation = ("仅按已发布的独立风险验证门放行；保证限于其模型、分布和样品独立性条件，不代表B confirmed"
                          if calibrated else "启发式暂定分类；B未执行，概率未校准；不代表B confirmed")
            value = {"schema": "ab-fusion/v2", "sample_id": job["sample_id"],
                     "simulation": a.get("provenance") == "recorded_simulation",
                     "real_model_validation": False if a.get('provenance')=='recorded_simulation' else None,
                     "gate_risk_validated": calibrated, "route": "a_only_"+routing["mode"], "routing": routing,
                     "a": a, "b": None, "agent": {"executed": False},
                     "needs_review": not calibrated, "b_executed": False,
                     "limitations": [limitation],
                     "elapsed_seconds": time.perf_counter()-start}
            write_json(directory/"fusion.json", value)
            (directory/"report.md").write_text(
                "# A分类分流\n\nB未执行，Agent未执行。模式："+routing["mode"]+"\n\n类别："
                + a["ranking"][0]["label"] + "\n" + limitation + "。\n", encoding="utf-8")
            job.update(status="completed", stage="done", needs_review=value["needs_review"],
                       route=value["route"], elapsed_seconds=value["elapsed_seconds"])
            write_json(status_path, job)
            return directory
        if "workflow" in cfg.get("agent", {}):
            from .workflow_entry import execute_workflow
            from .workflow_backend import WorkflowBackend
            current = WorkflowBackend.code_signature()
            write_json(directory/"code_signature_before_agent.json",
                       {"schema": "code-signature/v1", "stage": "before_agent",
                        "code_signature": current,
                        "baseline": code_baseline,
                        "at": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
            if code_baseline is not None and current != code_baseline:
                raise IntegrationError("CODE_CHANGED", "A 推理期间代码签名变化；Agent 未启动")
            return asyncio.run(execute_workflow(cfg,a,directory,routing=routing,start=start,job=job,status_path=status_path,code_baseline=code_baseline))
        job.update(stage="b")
        write_json(status_path, job)
        b_cfg = cfg.get("b", {})
        if b_cfg.get("mode", "run") == "run" and b_cfg.get("input_mode", "images") == "images":
            raise IntegrationError("CONFIG_INVALID", "从原图运行 B 须配置 agent.workflow，由 YOLO 出框")
        b = execute_b(b_cfg, job["sample_id"], identifier,
                      allow_network=cfg.get("allow_network", False))
        write_json(directory/"b_snapshot.json", b)
        job.update(stage="fusion_and_shadow")
        write_json(status_path, job)
        fusion = fuse(a, b["verdict"], knowledge_base(), cfg.get("aliases"))
        # Read-only historical artifact integration remains for contract regression.
        # All active executions have already entered execute_workflow above.
        agent = select_shadow(fusion, b["verdict"], cfg.get("agent", {}),
                              allow_network=cfg.get("allow_network", False))
        simulated = any(x.get("provenance") == "recorded_simulation" for x in (a, b))
        value = {"schema": "ab-fusion/v1", "sample_id": job["sample_id"],
                 "simulation": simulated, "real_model_validation": False if simulated else None,
                 "a": a, "b": b, "fusion": fusion, "agent": agent,
                 "routing": routing, "route": "agent_or_baseline", "b_executed": True,
                 "elapsed_seconds": time.perf_counter()-start,
                 "limitations": ["Agent只做影子动作选择，不执行新观察、不修改B裁决",
                                 "A原CLI概率保留4位小数；尚未实现模型常驻",
                                 "B检测/IGM完整API计数不可得，不声称全链预算已受控"]}
        write_json(directory/"fusion.json", value)
        label = "模拟演示" if simulated else "真实工件集成（不代表准确率已验证）"
        report = (f"# A/B融合报告：{job['sample_id']}\n\n{label}\n\n"
                  f"- A：{fusion['a_label'] or '不可用'}\n"
                  f"- B：{b['verdict']['verdict']} / {(b['verdict'].get('conclusion') or {}).get('display', '证据不足')}\n"
                  f"- 关系：{fusion['agreement']}\n- 需要复核：{fusion['needs_review']}\n"
                  f"- 影子动作：{agent['choice']}\n"
                  f"- 观察执行数：{agent['observations_executed']}；原B工件未修改。\n\n"
                  + "\n".join('- '+x for x in value['limitations'])+"\n")
        if routing["mode"] != "off":
            report += f"\n- 启发式路由：{routing['mode']}；建议跳过：{routing['recommend_skip']}；本次仍执行B。\n"
        (directory/"report.md").write_text(report, encoding="utf-8")
        job.update(status="completed", stage="done", simulation=simulated,
                   needs_review=fusion["needs_review"], elapsed_seconds=value["elapsed_seconds"])
        write_json(status_path, job)
        return directory
    except Exception as exc:
        job.update(status="failed", error_code=getattr(exc, "code", "INTEGRATION_ERROR"),
                   error_type=type(exc).__name__, elapsed_seconds=time.perf_counter()-start)
        write_json(status_path, job)
        if isinstance(exc, IntegrationError):
            raise IntegrationError(exc.code, f"{exc}；失败记录: {status_path}") from exc
        raise IntegrationError("INTEGRATION_ERROR", f"集成失败({type(exc).__name__})；详见状态: {status_path}") from exc
    finally:
        if code_baseline is not None:
            try:
                from .workflow_backend import WorkflowBackend
                current = WorkflowBackend.code_signature()
                write_json(directory/"code_signature_end.json",
                           {"schema": "code-signature/v1", "stage": "pipeline_end",
                            "code_signature": current, "baseline": code_baseline,
                            "changed": current != code_baseline,
                            "at": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
                if current != code_baseline:
                    job["code_changed"] = True
                    if job.get("status") == "completed":
                        job["needs_review"] = True
                    write_json(status_path, job)
            except Exception:
                # 结束复核不能覆盖主流程的失败原因。
                pass


def doctor(config):
    """只读检查，不加载模型、不下载权重、不联网。"""
    cfg = validate_config(config)
    checks = []
    def check(name, ok, detail):
        checks.append({"name": name, "ok": bool(ok), "detail": str(detail)})
    a, b = cfg.get("a", {}), cfg.get("b", {})
    if cfg.get("agent", {}).get("mode") == "active":
        import importlib.util
        check("LangGraph", importlib.util.find_spec("langgraph") is not None, "requirements-agent.txt")
        check("JSON Schema", importlib.util.find_spec("jsonschema") is not None, "受信注册表依赖jsonschema")
        check("图像工具依赖", all(importlib.util.find_spec(m) is not None for m in ("cv2", "numpy")),
              "真实VLM图像编码复用原B，需要OpenCV与NumPy；离线fixture不需要")
        if cfg["agent"].get("transport") == "mcp":
            check("MCP SDK", importlib.util.find_spec("mcp") is not None, "mcp 1.x stdio")
        for root in cfg["agent"]["image_roots"]:
            check("Agent样品原图", project_path(root).exists(), root)
    check("A原脚本", (Path(paths.PROJECT_ROOT)/"MICROSCOPE-main/predictv3_resnet50.py").is_file(), "只通过子进程执行")
    if a.get("mode", "cli") == "cli":
        checkpoint = project_path(a.get("checkpoint_dir", "MICROSCOPE-main/runs/v3_resnet50_full"))
        image = project_path(a.get("image") or (a.get("images") or ["MICROSCOPE-main/显微镜像/甘草/example.jpg"])[0])
        check("A五折参考落点", all((checkpoint/f"fold{i}_best.pt").is_file() for i in range(1, 6)), checkpoint)
        check("A原图", image.is_file(), image)
        py = project_path(a.get("python", paths.VENV_PY))
        check("A解释器", py.is_file(), py)
        if py.is_file():
            code = "import importlib.util,json; print(json.dumps({m:importlib.util.find_spec(m) is not None for m in ['torch','torchvision','numpy','PIL']}))"
            cp = subprocess.run([str(py), "-X", "utf8", "-c", code], capture_output=True,
                                text=True, encoding="utf-8", timeout=30)
            try:
                modules = json.loads(cp.stdout)
                check("A依赖可发现（非运行验证）", cp.returncode == 0 and all(modules.values()), modules)
            except ValueError:
                check("A依赖探针", False, "解释器未返回可解析结果")
    workflow = cfg.get('agent',{}).get('workflow')
    from .runtime_readiness import profile_checks, planner_preflight
    if workflow is not None:
        checks.extend(planner_preflight(cfg)['checks'])
        if workflow.get('cnn_weights') or workflow.get('cnn_meta'):
            from .stage2_classifier import model_profile
            try:
                profile=model_profile(workflow)
                check('stage2模型/元信息',True,
                      '显微特征 '+profile['metadata']['arch']+'；仅文件/元信息检查，未加载权重')
            except (IntegrationError,OSError,ValueError) as exc:
                check('stage2模型/元信息',False,getattr(exc,'code','CNN_META_INVALID'))
    if workflow is None and b.get("mode", "run") == "run":
        check("B输入", project_path(b.get("input", "MICROSCOPE-main/显微镜像/甘草")).exists(), b.get("input"))
        py = project_path(b.get("python", paths.VENV_PY))
        check("B解释器", py.is_file(), py)
        if b.get("input_mode", "images") != "masks" or b.get("enable_vlm"):
            check("B配置", (Path(paths.DETECTION_CONFIG_DIR)/"vlm_config.json").is_file(), "images/crops/enable_vlm需要配置")
            check("网络显式启用", cfg.get("allow_network", False), "参考配置默认关闭付费调用")
    elif b.get("mode") == "existing":
        if "workflow" in cfg.get("agent", {}):
            check("B部分工件目录", project_path(b["run_dir"]).is_dir(), "workflow不要求完成manifest")
        else:
            check("B完成manifest", (project_path(b["run_dir"])/"run_manifest.json").is_file(), b["run_dir"])
    if workflow is not None:
        from .adapters import interpreter
        py=interpreter(cfg["agent"]["workflow"].get("python"))
        probe=subprocess.run([py,"-X","utf8","-c","import cv2,numpy; print('ok')"],
                             capture_output=True,timeout=30)
        check("重算解释器实际导入",probe.returncode==0,"opencv/numpy；不代表在线模型可用")
        if set(workflow.get('enabled_tools',{'detect_fov_initial','calibrate_fov','segment_instance','detect_region'})) - {'measure_instances'}:
            probe=subprocess.run([py,'-X','utf8','-c','import PIL,rapidocr_onnxruntime; print("ok")'],
                                 capture_output=True,timeout=30)
            check('检测侧环境依赖',probe.returncode==0,'Pillow/RapidOCR；未初始化OCR模型、未联网')
            check('工作流视觉配置',(Path(paths.DETECTION_CONFIG_DIR)/'vlm_config.json').is_file(),'沿用原B配置')
            check('工作流网络显式启用',cfg.get('allow_network',False),'视觉阶段需要允许网络')
            sections=['vlm']
            if 'segment_instance' in workflow.get('enabled_tools',{'segment_instance'}):sections.append('igm')
            checks.extend(profile_checks(sections))
    return {"ready": all(x["ok"] for x in checks), "checks": checks,
            "note": "doctor不构成模型/图像/网络运行验证；缓存和演示不替代真实联调"}
