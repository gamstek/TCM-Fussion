"""通过子进程执行 Plan A 与原 B，复用模型的推理预处理。

A 对三个模型的完整 softmax 等权求平均，历史五折保留原输出契约；
B 始终走原 run_pipeline 或只读已完成 run，交付校验复用 contracts。
失败不自动重试，A 结果按图像/权重/脚本/解释器指纹缓存。

主要定义：
- interpreter: 解析并校验 A/B 使用的解释器路径，缺失即报错。
- run_process: 以 UTF-8 环境运行子进程入口，超时或退出码非零时抛结构化错误。
- parse_a_stdout: 严格解析原 A 完整 Top-N 文本，缺行、重名或概率非法即失败。
- resolve_a_assets / predict_a: 校验模型路径、执行 CLI 并缓存完整概率或历史逐折排名。
- combine_a_views: 对新 A 各视野的完整概率等权求平均。
- read_b_verdict: 读取并校验 B 裁决，检查裁决带与 sample_id。
- execute_b: 按 run/existing/recorded 模式执行或读取 B run，并校验完成度与样品唯一性。

关联：
- 调用方 「lib/application/pipeline.py」、「lib/workflow/workflow_session.py」、「tests/integration/test_integration.py」。
- 直接依赖 「lib/runtime/storage.py」。
"""
from __future__ import annotations

import json
import math
import os
import re
import subprocess
import time
from pathlib import Path

import contracts
import paths

from TCM_Fusion.lib.runtime.storage import (IntegrationError, digest, file_hash, fingerprint,
                      finite_number, project_path, read_json, write_json)


def interpreter(value=None):
    p = project_path(value or paths.VENV_PY)
    if not p.is_file():
        raise IntegrationError("PYTHON_MISSING", f"解释器不存在: {p}")
    return p


def a_interpreter(config):
    value = config.get("python") or os.environ.get("TCM_A_PYTHON")
    if not value:
        raise IntegrationError("PYTHON_MISSING", "须显式指定 A 解释器（a.python 或 TCM_A_PYTHON）")
    return interpreter(value)


def run_process(argv, *, timeout=None):
    """Python入口不套PS流合并；失败不自动重试，尤其不重发付费请求。"""
    try:
        cp = subprocess.run([str(x) for x in argv], cwd=paths.PROJECT_ROOT,
                            capture_output=True, text=True, encoding="utf-8",
                            errors="replace", env=dict(paths.python_environment(), PYTHONIOENCODING="utf-8"),
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


def resolve_a_assets(config):
    """定位三模型与元信息；历史显式权重目录仍可调用原五折入口。"""
    directory = project_path(config.get("checkpoint_dir", paths.A_CHECKPOINT_DIR))
    if "model_runs" in config or directory == project_path(paths.A_CHECKPOINT_DIR):
        names = config.get("model_runs", list(paths.A_MODEL_RUNS))
        if (not isinstance(names, list) or len(names) != 3
                or any(not isinstance(name, str) or not re.fullmatch(r"[\w-]+", name) for name in names)
                or len(set(names)) != 3):
            raise IntegrationError("CONFIG_INVALID", "A.model_runs 须指定三个不同的模型子目录名")
        models = [(directory / name).resolve() for name in names]
        if any(model.parent != directory or model.name != name for model, name in zip(models, names)):
            raise IntegrationError("CONFIG_INVALID", "A 模型须位于声明的权重根目录内")
        weights = [model / "best.pt" for model in models]
        for weight in weights:
            if not weight.is_file() or not weight.with_name("model_meta.json").is_file():
                raise IntegrationError("A_CHECKPOINT_MISSING", f"A 权重或元信息缺失: {weight.parent}")
        return Path(paths.A_PREDICT), weights, models
    numbered = {}
    for weight in directory.glob("fold*_best.pt"):
        match = re.fullmatch(r"fold([1-9]\d*)_best\.pt", weight.name)
        if match and weight.is_file():
            numbered[int(match.group(1))] = weight
    if not numbered or sorted(numbered) != list(range(1, len(numbered) + 1)):
        raise IntegrationError("A_CHECKPOINT_MISSING", "须提供从 fold1 开始连续编号的历史 checkpoint")
    return Path(paths.A_LEGACY_PREDICT), [numbered[i] for i in sorted(numbered)], []


def _rank_probabilities(probabilities):
    ordered = sorted(probabilities.items(), key=lambda item: (-item[1], item[0]))
    return [{"rank": i, "label": label, "probability": prob}
            for i, (label, prob) in enumerate(ordered, 1)]


def _mean_ranking(rankings, code):
    by_label = []
    for rows in rankings:
        _validate_ranking({"ranking": rows}, code)
        probabilities = {row["label"]: row["probability"] for row in rows}
        if (abs(math.fsum(probabilities.values()) - 1) > 1e-5
                or any(row.get("rank") != i for i, row in enumerate(rows, 1))):
            raise IntegrationError(code, "A 集成概率榜归一化或名次不合法")
        if by_label and probabilities.keys() != by_label[0].keys():
            raise IntegrationError(code, "A 概率榜类别集合不一致")
        by_label.append(probabilities)
    if not by_label:
        raise IntegrationError(code, "A 缺少可集成的概率榜")
    return _rank_probabilities({label: math.fsum(row[label] for row in by_label) / len(by_label)
                                for label in by_label[0]})


def _parse_ensemble(text, names):
    code = "A_OUTPUT_FORMAT"
    try:
        output = json.loads(text)
        labels = output["classes"]
        predictions = output["model_predictions"]
        if (not isinstance(labels, list) or len(labels) < 8
                or any(not isinstance(label, str) or not label.strip() for label in labels)
                or len(set(labels)) != len(labels)
                or not isinstance(predictions, list) or [row["model_id"] for row in predictions] != names):
            raise ValueError
        models = []
        for row in predictions:
            values = row["probabilities"]
            if (not isinstance(values, list) or len(values) != len(labels)
                    or any(not finite_number(prob, maximum=1) for prob in values)):
                raise ValueError
            models.append({"model_id": row["model_id"],
                           "ranking": _rank_probabilities(dict(zip(labels, values)))})
    except (KeyError, TypeError, ValueError) as exc:
        raise IntegrationError(code, "A 三模型原始概率输出不合法") from exc
    ranking = _mean_ranking([model["ranking"] for model in models], code)
    return {"status": "ok", "provenance": "original_cli", "ranking": ranking,
            "ensemble_method": "mean_softmax", "model_predictions": models,
            "calibrated": False, "resident_models": False, "coverage": "single_view"}


def _validate_ensemble_cache(value, names):
    models = value.get("model_predictions")
    if (value.get("ensemble_method") != "mean_softmax" or not isinstance(models, list)
            or any(not isinstance(model, dict) for model in models)
            or [model.get("model_id") for model in models] != names):
        raise IntegrationError("CACHE_INVALID", "A 缓存模型不匹配")
    mean = _mean_ranking([model.get("ranking") for model in models], "CACHE_INVALID")
    if len(mean) < 8 or value.get("ranking") != mean:
        raise IntegrationError("CACHE_INVALID", "A 缓存平均榜无法复算")


def combine_a_views(views):
    value = {"status": "ok", "ranking": [], "provenance": "live_multiview",
             "view_predictions": views}
    methods = {view.get("ensemble_method") for view in views}
    if len(methods) > 1:
        raise IntegrationError("INPUT_CHANGED", "各视野的 A 集成方法不一致")
    if methods == {"mean_softmax"}:
        revisions = [{key: item for key, item in view["signature"].items() if key != "image"}
                     for view in views]
        if any(revision != revisions[0] for revision in revisions[1:]):
            raise IntegrationError("INPUT_CHANGED", "各视野的 A 模型、预处理或入口不一致")
        value.update(ensemble_method="mean_softmax", calibrated=False,
                     ranking=_mean_ranking([view["ranking"] for view in views], "A_OUTPUT_FORMAT"))
    return value


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
    script, weights, models = resolve_a_assets(config)
    py = a_interpreter(config)
    timeout = config.get("timeout_seconds", 0)
    if not finite_number(timeout):
        raise IntegrationError("CONFIG_INVALID", "A.timeout_seconds须为非负有限值")
    # 指定的A环境与B环境可不同；不互相安装冲突的Torch版本。
    def input_signature():
        value = {"image": file_hash(image), "weights": [file_hash(weight) for weight in weights],
                 "script": file_hash(script), "python": str(py),
                 "python_sha256": file_hash(py), "adapter_version": 3}
        if models:
            value.update(adapter_version=4, transforms=file_hash(Path(paths.DETECTION_LIB_DIR) / "cls_net.py"),
                         models=[{"id": model.name, "weight_sha256": sha,
                                  "meta_sha256": file_hash(model / "model_meta.json")}
                                 for model, sha in zip(models, value["weights"])])
        return value

    signature = input_signature()
    key = digest(signature)
    cache_path = Path(cache_dir) / (key + ".json")
    if cache_path.exists():
        try:
            cached = read_json(cache_path)
        except ValueError as exc:
            raise IntegrationError("CACHE_INVALID", "A缓存JSON损坏") from exc
        _validate_ranking(cached, "CACHE_INVALID")
        if models:
            _validate_ensemble_cache(cached, [model.name for model in models])
        else:
            _validate_fold_rankings(cached, "CACHE_INVALID")
        if (cached.get("cache_key") == key and cached.get("signature") == signature
                and cached.get("provenance") == "original_cli" and cached.get("status") == "ok"):
            return {**cached, "cache_hit": True}
        raise IntegrationError("CACHE_INVALID", "A缓存键不匹配")
    start = time.perf_counter()
    argv = [py, "-X", "utf8", script, "--image", image]
    if models:
        for model in models:
            argv.extend(["--model-dir", model])
        value = _parse_ensemble(run_process(argv, timeout=timeout or None), [model.name for model in models])
    else:
        argv.extend(["--checkpoint-dir", weights[0].parent, "--top-k", "1000000", "--emit-fold-rankings"])
        text = run_process(argv, timeout=timeout or None)
        value = parse_a_stdout(text)
        if value["folds"] != len(weights):
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
    try:
        unchanged = input_signature() == signature
    except OSError as exc:
        raise IntegrationError("INPUT_CHANGED", "预测期间 A 输入文件不可读取") from exc
    if not unchanged:
        raise IntegrationError("INPUT_CHANGED", "预测期间 A 输入或模型资产发生变化")
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
