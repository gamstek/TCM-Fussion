"""通过子进程调用原 A/原 B 入口，绝不 import 入口模块或复制其预处理算法。

A 平均榜沿用四位小数输出，Agent 候选使用机器可读的逐折原始概率排名；
B 始终走原 run_pipeline 或只读已完成 run，交付校验复用 contracts。
失败不自动重试，A 结果按图像/权重/脚本/解释器指纹缓存。

主要定义：
- interpreter: 解析并校验 A/B 使用的解释器路径，缺失即报错。
- run_process: 以 UTF-8 环境运行子进程入口，超时或退出码非零时抛结构化错误。
- parse_a_stdout: 严格解析原 A 完整 Top-N 文本，缺行、重名或概率非法即失败。
- predict_a: 按 cli/recorded/unavailable 三种模式产出 A 排序与逐折排名，并写缓存。
- read_b_verdict: 读取并校验 B 裁决，检查裁决带与 sample_id。
- execute_b: 按 run/existing/recorded 模式执行或读取 B run，并校验完成度与样品唯一性。

关联：
- contracts: 读取 B verdict 与测量记录，做交付级校验。
- paths: 提供项目根、run 目录、解释器与配置路径。
- .storage: 使用 IntegrationError、digest、file_hash、fingerprint 等做校验与缓存键。
- .pipeline / .workflow_session: 分别调用 predict_a、execute_b 与 interpreter。
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import time
from pathlib import Path

import contracts
import paths

from .storage import (IntegrationError, digest, file_hash, fingerprint,
                      finite_number, project_path, read_json, write_json)


def interpreter(value=None):
    p = project_path(value or paths.VENV_PY)
    if not p.is_file():
        raise IntegrationError("PYTHON_MISSING", f"解释器不存在: {p}")
    return p


def run_process(argv, *, timeout=None):
    """Python入口不套PS流合并；失败不自动重试，尤其不重发付费请求。"""
    try:
        cp = subprocess.run([str(x) for x in argv], cwd=paths.PROJECT_ROOT,
                            capture_output=True, text=True, encoding="utf-8",
                            errors="replace", env=dict(os.environ, PYTHONIOENCODING="utf-8"),
                            timeout=timeout, shell=False)
    except subprocess.TimeoutExpired as exc:
        raise IntegrationError("PROCESS_TIMEOUT", "A推理超时；未自动重试") from exc
    if cp.returncode:
        # 原始异常可能含配置或凭据；不把stderr原文写入结果。
        raise IntegrationError("PROCESS_FAILED",
                               f"子进程退出码{cp.returncode}，入口{Path(argv[3] if len(argv)>3 else argv[0]).name}")
    return cp.stdout


def parse_a_stdout(text):
    """严格解析原predict脚本的完整Top-N，缺行/重名/非有限值则失败。"""
    count_match = re.search(r"^类别数:\s*(\d+)\s*$", text, re.M)
    folds_match = re.search(r"(\d+)-fold ensemble", text)
    if not count_match or not folds_match:
        raise IntegrationError("A_OUTPUT_FORMAT", "原A输出缺类别数或折数")
    count = int(count_match.group(1))
    rows = []
    for match in re.finditer(r"^\s+(\d+)\.\s+(.+?)\s+(\d+\.\d+)\s+\(", text, re.M):
        rank, label, probability = match.groups()
        rows.append({"rank": int(rank), "label": label.strip(), "probability": float(probability)})
    if (count < 1 or len(rows) != count or
            [x["rank"] for x in rows] != list(range(1, count + 1)) or
            len({x["label"] for x in rows}) != count or
            any(not finite_number(x["probability"], maximum=1) for x in rows) or
            any(rows[i]["probability"] < rows[i+1]["probability"] for i in range(len(rows)-1)) or
            abs(sum(x["probability"] for x in rows) - 1) > count * 0.000051 + 1e-7):
        raise IntegrationError("A_OUTPUT_FORMAT", "A完整概率输出缺失或不合法")
    return {"status": "ok", "provenance": "original_cli", "ranking": rows,
            "folds": int(folds_match.group(1)), "probability_precision": 4,
            "calibrated": False, "resident_models": False, "coverage": "single_view",
            "warning": "原CLI四位小数；保留原排序，不能冒充原始float概率或已校准置信度"}


def _validate_ranking(value, code):
    rows = value.get("ranking") if isinstance(value, dict) else None
    if not isinstance(rows, list) or not rows:
        raise IntegrationError(code, "A记录缺完整ranking")
    if (any(not isinstance(x, dict) or not isinstance(x.get("label"), str) or not x["label"] or
            not finite_number(x.get("probability"), maximum=1) for x in rows)
            or len({x["label"] for x in rows}) != len(rows)
            or abs(sum(x["probability"] for x in rows) - 1) > 0.01):
        raise IntegrationError(code, "A记录类别或概率非法")
    if any(rows[i]["probability"] < rows[i + 1]["probability"] for i in range(len(rows)-1)):
        raise IntegrationError(code, "A记录必须保留降序")


def _recorded_a(path):
    value = read_json(path)
    _validate_ranking(value, "A_RECORD_INVALID")
    return {**value, "status": "ok", "provenance": "recorded_simulation",
            "record_sha256": file_hash(path)}


def _validate_fold_rankings(value, code):
    folds = value.get("fold_rankings")
    labels = {row["label"] for row in value["ranking"]}
    if not isinstance(folds, list) or len(folds) != value.get("folds"):
        raise IntegrationError(code, "A逐折排名缺失或折数不符")
    for rows in folds:
        _validate_ranking({"ranking": rows}, code)
        if {row["label"] for row in rows} != labels or any(
            row.get("rank") != rank for rank, row in enumerate(rows, 1)
        ):
            raise IntegrationError(code, "A逐折排名类别或名次不符")


def predict_a(config, cache_dir):
    mode = config.get("mode", "cli")
    if mode == "unavailable":
        return {"status": "unavailable", "provenance": "unavailable", "ranking": []}
    if mode == "recorded":
        return _recorded_a(project_path(config["record"]))
    if mode != "cli":
        raise IntegrationError("CONFIG_INVALID", "A.mode须为cli/recorded/unavailable")
    image = project_path(config["image"])
    if not image.is_file():
        raise IntegrationError("A_IMAGE_MISSING", f"A原图未提供: {image}")
    checkpoint_dir = project_path(config["checkpoint_dir"])
    weights = sorted(checkpoint_dir.glob("fold*_best.pt"), key=lambda p: p.name)
    numbered = {}
    for p in weights:
        m = re.fullmatch(r"fold([1-9]\d*)_best\.pt", p.name)
        if m:
            numbered[int(m.group(1))] = p
    if not numbered or sorted(numbered) != list(range(1, len(numbered)+1)):
        raise IntegrationError("A_CHECKPOINT_MISSING", "须提供从fold1开始连续编号的完整checkpoint集合")
    script = Path(paths.PROJECT_ROOT) / "MICROSCOPE-main" / "predictv3_resnet50.py"
    py = interpreter(config.get("python"))
    timeout = config.get("timeout_seconds", 0)
    if not finite_number(timeout):
        raise IntegrationError("CONFIG_INVALID", "A.timeout_seconds须为非负有限值")
    # 指定的A环境与B环境可不同；不互相安装冲突的Torch版本。
    signature = {"image": file_hash(image), "weights": [file_hash(numbered[i]) for i in sorted(numbered)],
                 "script": file_hash(script), "python": str(py),
                 "python_sha256": file_hash(py), "adapter_version": 3}
    key = digest(signature)
    cache_path = Path(cache_dir) / (key + ".json")
    if cache_path.exists():
        try:
            cached = read_json(cache_path)
        except ValueError as exc:
            raise IntegrationError("CACHE_INVALID", "A缓存JSON损坏") from exc
        _validate_ranking(cached, "CACHE_INVALID")
        _validate_fold_rankings(cached, "CACHE_INVALID")
        if (cached.get("cache_key") == key and cached.get("signature") == signature
                and cached.get("provenance") == "original_cli" and cached.get("status") == "ok"):
            return {**cached, "cache_hit": True}
        raise IntegrationError("CACHE_INVALID", "A缓存键不匹配")
    start = time.perf_counter()
    text = run_process([py, "-X", "utf8", script, "--checkpoint-dir", checkpoint_dir,
                        "--image", image, "--top-k", "1000000", "--emit-fold-rankings"],
                       timeout=timeout or None)
    value = parse_a_stdout(text)
    if value["folds"] != len(numbered):
        raise IntegrationError("A_CHECKPOINT_MISMATCH", "元信息声明折数与实际checkpoint集合不一致")
    prefix = "FOLD_RANKINGS_JSON="
    fold_lines = [line[len(prefix):] for line in text.splitlines() if line.startswith(prefix)]
    try:
        if len(fold_lines) != 1:
            raise ValueError("expected exactly one fold ranking record")
        value["fold_rankings"] = json.loads(fold_lines[0])
    except ValueError as exc:
        raise IntegrationError("A_OUTPUT_FORMAT", "A逐折排名输出缺失或无法解析") from exc
    _validate_fold_rankings(value, "A_OUTPUT_FORMAT")
    if file_hash(image) != signature["image"] or any(
            file_hash(numbered[i]) != signature["weights"][i-1] for i in sorted(numbered)):
        raise IntegrationError("INPUT_CHANGED", "预测期间图像或权重发生变化")
    value.update(cache_key=key, cache_hit=False, image=str(image),
                 input_sha256=signature["image"], signature=signature,
                 elapsed_seconds=time.perf_counter()-start)
    write_json(cache_path, value)
    return value


def read_b_verdict(path, expected_sample=None):
    v = contracts.load_verdict(str(path))
    allowed = {"confirmed", "probable", "group_level", "insufficient_evidence"}
    if v.verdict not in allowed or (expected_sample and v.sample_id != expected_sample):
        raise IntegrationError("B_VERDICT_INVALID", "B裁决带或sample_id不匹配")
    if v.verdict != "insufficient_evidence" and not v.conclusion:
        raise IntegrationError("B_VERDICT_INVALID", "非弃权B结果缺conclusion")
    return v.raw


def execute_b(config, sample_id, run_id, *, allow_network=False):
    mode = config.get("mode", "run")
    if mode == "recorded":
        path = project_path(config["record"])
        return {"verdict": read_b_verdict(path, sample_id), "provenance": "recorded_simulation",
                "record_sha256": file_hash(path), "run_dir": None}
    if mode == "existing":
        run_dir = project_path(config["run_dir"])
    elif mode == "run":
        input_mode = config.get("input_mode", "images")
        if input_mode not in {"images", "crops", "masks"}:
            raise IntegrationError("CONFIG_INVALID", "B.input_mode须为images/crops/masks")
        if (input_mode != "masks" or config.get("enable_vlm")) and not allow_network:
            raise IntegrationError("NETWORK_DISABLED", "检测/IGM/VLM会消耗API，须显式配置allow_network=true")
        source = project_path(config["input"])
        before = fingerprint(source)
        py = interpreter(config.get("python"))
        run_dir = Path(paths.RUNS_ROOT) / run_id
        if run_dir.exists():
            raise IntegrationError("RUN_EXISTS", "拒绝覆盖既有B run")
        command = [py, "-X", "utf8", Path(paths.PROJECT_ROOT)/"run_pipeline.py",
                   "--"+input_mode, source, "--run", run_id, "--sample", sample_id,
                   "--python", py]
        if config.get("manifest"):
            command += ["--manifest", project_path(config["manifest"])]
        if config.get("images") and input_mode != "images":
            command += ["--images", project_path(config["images"])]
        if config.get("enable_vlm"):
            command += ["--enable-vlm"]
        # 不设置B强制超时：杀父进程并不能证明孙进程停止，不提供自动重试。
        run_process(command)
        if fingerprint(source) != before:
            raise IntegrationError("INPUT_CHANGED", "B执行期间输入发生变化，结果不可发布为成功")
    else:
        raise IntegrationError("CONFIG_INVALID", "B.mode须为run/existing/recorded")
    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.is_file():
        raise IntegrationError("B_INCOMPLETE", "B缺完成manifest，不能凭verdicts目录视为完成")
    manifest = read_json(manifest_path)
    if manifest.get("run_id") != run_dir.name:
        raise IntegrationError("B_INCOMPLETE", "B manifest标识与目录不匹配")
    measurements = sorted((run_dir/"measurements").glob("*_measurements.jsonl"))
    if not measurements or not any(list(contracts.iter_measurement_records(str(p))) for p in measurements):
        raise IntegrationError("B_INCOMPLETE", "B没有可校验的测量记录")
    verdict_paths = list((run_dir/"verdicts"/"reports").glob("*/verdict.json"))
    values = [(p, read_b_verdict(p)) for p in verdict_paths]
    matches = [(p, v) for p, v in values if v.get("sample_id") == sample_id]
    if len(matches) != 1:
        raise IntegrationError("B_SAMPLE_AMBIGUOUS", "B run须恰有一份指定样品裁决")
    path, verdict = matches[0]
    return {"verdict": verdict, "provenance": "original_b_run", "run_dir": str(run_dir),
            "record_sha256": file_hash(path)}
