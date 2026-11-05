"""按 group_key 做确定性训练/开发/测试划分；只做数据清单，不推断标签。

用途：调提示词、关联阈值和边界策略时，开发集用于反复试验，冻结测试集只在最后评估；
禁止同一 group_key 跨 split，禁止同一个图像文件落进两个组。
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from .storage import IntegrationError, project_path

SPLITS = ("training", "development", "test")


def _ratio(value):
    if isinstance(value, str):
        value = value.split(",")
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise IntegrationError("SPLIT_RATIO", "ratios 须为 training,development,test 三个数")
    try:
        nums = [float(v) for v in value]
    except (TypeError, ValueError) as exc:
        raise IntegrationError("SPLIT_RATIO", "ratios 必须是数字") from exc
    if any(n < 0 for n in nums) or abs(sum(nums) - 1.0) > 1e-6:
        raise IntegrationError("SPLIT_RATIO", "ratios 须非负且和为 1")
    return dict(zip(SPLITS, nums))


def read_groups(paths):
    """读一个或多个 attributes.csv；返回 group_key -> 组内图像/药材/特征清单。"""
    groups = {}
    image_owner = {}
    for raw_path in paths:
        path = project_path(raw_path)
        if not path.is_file():
            raise IntegrationError("SPLIT_INPUT", f"attributes 不存在：{raw_path}")
        with path.open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        if not rows:
            raise IntegrationError("SPLIT_INPUT", f"attributes 为空：{raw_path}")
        for row in rows:
            group = (row.get("group_key") or "").strip()
            image = (row.get("image") or "").strip()
            if not group or not image:
                raise IntegrationError("SPLIT_INPUT", "attributes 缺 group_key/image")
            # 同名文件可能出现在不同特征目录；身份必须带目录，不能只看 basename。
            candidate = path.parent / "images" / image
            image_key = str((candidate if candidate.exists() else path.parent / image).resolve())
            if image_key in image_owner and image_owner[image_key] != group:
                raise IntegrationError("SPLIT_LEAKAGE", f"同一图像出现在不同 group：{image_key}")
            image_owner[image_key] = group
            entry = groups.setdefault(group, {"herb": set(), "features": set(), "images": []})
            entry["herb"].add((row.get("herb") or "").strip())
            entry["features"].add((row.get("feature_id") or "").strip())
            entry["images"].append(image_key)
    return groups


def _bucket(group, seed):
    digest = hashlib.sha256(f"{seed}:{group}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16) / float(16 ** 16)


def assign(groups, *, seed=20261104, ratios=(0.7, 0.15, 0.15), explicit=None):
    """确定性划分；explicit 是 group_key -> split 的人工覆盖。"""
    ratio = _ratio(ratios)
    explicit = dict(explicit or {})
    unknown = set(explicit) - set(groups)
    if unknown:
        raise IntegrationError("SPLIT_EXPLICIT", f"explicit 含未知 group：{sorted(unknown)}")
    bad = {g: s for g, s in explicit.items() if s not in SPLITS}
    if bad:
        raise IntegrationError("SPLIT_EXPLICIT", f"explicit split 非法：{bad}")
    assignment = {}
    for group in sorted(groups):
        if group in explicit:
            assignment[group] = explicit[group]
            continue
        value = _bucket(group, seed)
        cumulative = 0.0
        for split, share in zip(SPLITS, (ratio[s] for s in SPLITS)):
            cumulative += share
            if value < cumulative - 1e-12:
                assignment[group] = split
                break
        else:
            assignment[group] = SPLITS[-1]
    counts = {s: sum(1 for v in assignment.values() if v == s) for s in SPLITS}
    if any(counts[s] == 0 for s in SPLITS):
        raise IntegrationError("SPLIT_EMPTY", f"划分后有空 split：{counts}；请调整 ratios 或 explicit")
    validate(assignment, groups)
    return assignment


def validate(assignment, groups):
    if set(assignment) != set(groups):
        raise IntegrationError("SPLIT_ASSIGNMENT", "划分未覆盖全部 group 或含未知 group")
    if any(split not in SPLITS for split in assignment.values()):
        raise IntegrationError("SPLIT_ASSIGNMENT", "存在非法 split")
    seen = {}
    for group, split in assignment.items():
        for image in groups[group]["images"]:
            if image in seen and seen[image] != split:
                raise IntegrationError("SPLIT_LEAKAGE", f"图像跨 split：{image}")
            seen[image] = split
    return True


def build(attributes_paths, *, seed=20261104, ratios=(0.7, 0.15, 0.15), explicit=None):
    groups = read_groups(attributes_paths)
    assignment = assign(groups, seed=seed, ratios=ratios, explicit=explicit)
    summary = {}
    for split in SPLITS:
        members = [g for g, s in assignment.items() if s == split]
        summary[split] = {
            "groups": len(members),
            "images": sum(len(groups[g]["images"]) for g in members),
            "herbs": sorted({h for g in members for h in groups[g]["herb"] if h}),
            "features": sorted({f for g in members for f in groups[g]["features"] if f}),
        }
    return {
        "schema": "dataset-split/v1",
        "seed": seed,
        "ratios": dict(zip(SPLITS, [float(v) for v in (
            ratios.split(",") if isinstance(ratios, str) else ratios)])),
        "groups": {g: {"split": assignment[g], "images": sorted(groups[g]["images"]),
                       "herbs": sorted(h for h in groups[g]["herb"] if h),
                       "features": sorted(f for f in groups[g]["features"] if f)}
                   for g in sorted(groups)},
        "summary": summary,
        "source_files": [str(project_path(p)) for p in attributes_paths],
        "note": "group_key 级别隔离；训练/开发用于调参，测试集只做最终评估，不得回流调参",
    }


def write(manifest, path):
    path = project_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load(path):
    path = project_path(path)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise IntegrationError("SPLIT_INPUT", f"划分文件不可读：{path}") from exc
    if value.get("schema") != "dataset-split/v1" or not isinstance(value.get("groups"), dict):
        raise IntegrationError("SPLIT_INPUT", "不是有效的 dataset-split/v1")
    return value


def split_of(manifest, group):
    row = manifest.get("groups", {}).get(group)
    if not row:
        raise IntegrationError("SPLIT_UNKNOWN", f"group 不在划分文件：{group}")
    return row["split"]
