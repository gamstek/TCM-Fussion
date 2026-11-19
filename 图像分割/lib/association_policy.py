"""可配置的连通域关联策略：规则表 + 显式审计 + 用途资格。

为什么不是 if/else 堆叠：
- 每个规则只回答“我是否适用 + 为什么”，evaluate 只做一次有序匹配；
- 阈值、特征形态、粒度、允许用途全部在 JSON 策略里，不在代码里散落；
- 决策对象带 selected/dropped/metrics/reasons/trace，落盘后可复核；
- 默认仍是 ambiguous/review；dominant 是显式候选策略，不静默接管。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable

import paths
from contracts import GRANULARITY_DEFAULT, QUALITY_USES


POLICY_PATH = Path(paths.SEGMENTATION_CONFIG_DIR) / "association_policy.json"

DEFAULT_POLICY = {
    "version": "association-policy-v1-trial",
    "calibrated": False,
    "dominant": {
        "enabled": True,
        "features": ["fiber", "vessel"],
        "shapes": ["elongated"],
        "granularities": ["instance"],
        "min_components": 2,
        "max_components": 4,
        "largest_share_min": 0.95,
        "second_share_max": 0.03,
        "dropped_area_ratio_max": 0.05,
        "max_dropped_components": 2,
        "min_tight_overlap": 0.5,
        "require_center_hit": True,
        "drop_touching_border": True,
        "allow_width": True,
        "allow_length": False,
        "allow_area": False,
    },
    "multipart": {
        "shapes": ["multipart"],
        "granularities": ["bundle", "sheet", "clump"],
    },
}


@dataclass(frozen=True)
class Component:
    """一个候选连通域；由调用方从 labels/stats 构造，策略本身不碰像素。"""

    id: int
    area: int
    bbox: tuple[int, int, int, int]  # x, y, w, h（crop 局部坐标）
    inside_ratio: float              # 落在 ROI/框内的像素比例
    tight_overlap: float             # 落在 tight_box 内的像素比例
    center_hit: bool
    touches_border: bool


@dataclass(frozen=True)
class AssociationContext:
    components: tuple[Component, ...]
    feature_id: str | None
    shape: str
    granularity: str = GRANULARITY_DEFAULT
    tight_box: tuple[int, int, int, int] | None = None
    center_region: tuple[int, int, int, int] | None = None


@dataclass(frozen=True)
class Decision:
    method: str
    selected: tuple[int, ...]
    dropped: tuple[int, ...]
    status: str
    reasons: tuple[str, ...]
    metrics: dict
    allowed_uses: dict
    policy_version: str
    trace: tuple[dict, ...] = ()


_SHAPES = None


def shape_of(feature_id):
    """feature_taxonomy.qa_shape 的只读缓存；未知特征返回 unknown（不猜形态）。"""
    global _SHAPES
    if _SHAPES is None:
        try:
            table = json.loads((Path(paths.PROJECT_ROOT) / "feature_taxonomy.json").read_text(
                encoding="utf-8"))
            _SHAPES = {row["feature_id"]: row.get("qa_shape", "unknown")
                       for row in table.get("features", [])}
        except (OSError, ValueError, KeyError):
            _SHAPES = {}
    return _SHAPES.get(feature_id or "", "unknown")


def _deep_merge(base, override):
    out = {k: (dict(v) if isinstance(v, dict) else v) for k, v in base.items()}
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _validate(policy):
    if not isinstance(policy, dict):
        raise ValueError("association policy must be an object")
    dominant = policy.get("dominant", {})
    if not isinstance(dominant.get("features", []), list) or not isinstance(
            dominant.get("shapes", []), list) or not isinstance(
            dominant.get("granularities", []), list):
        raise ValueError("association policy: features/shapes/granularities must be lists")
    for key in ("largest_share_min", "second_share_max", "dropped_area_ratio_max",
                "min_tight_overlap"):
        value = dominant.get(key)
        if not isinstance(value, (int, float)) or not 0 <= value <= 1:
            raise ValueError(f"association policy: {key} must be in [0,1]")
    for key in ("min_components", "max_components", "max_dropped_components"):
        value = dominant.get(key)
        if type(value) is not int or value < 0:
            raise ValueError(f"association policy: {key} must be a non-negative integer")
    if dominant["min_components"] > dominant["max_components"]:
        raise ValueError("association policy: min_components > max_components")
    return policy


def load_policy(path=None):
    target = Path(path) if path else POLICY_PATH
    if not target.is_file():
        return _validate(_deep_merge(DEFAULT_POLICY, {}))
    try:
        override = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"association policy 不可读：{target}") from exc
    return _validate(_deep_merge(DEFAULT_POLICY, override))


def _uses(**allowed):
    out = {use: False for use in QUALITY_USES}
    out.update({k: bool(v) for k, v in allowed.items()})
    return out


def _metrics(ctx: AssociationContext, selected=(), dropped=()):
    total = sum(c.area for c in ctx.components)
    by_id = {c.id: c for c in ctx.components}
    selected_area = sum(by_id[i].area for i in selected if i in by_id)
    dropped_area = sum(by_id[i].area for i in dropped if i in by_id)
    ordered = sorted(ctx.components, key=lambda c: c.area, reverse=True)
    largest_share = (ordered[0].area / total) if total and ordered else 0.0
    second_share = (ordered[1].area / total) if total and len(ordered) > 1 else 0.0
    return {
        "components": len(ctx.components),
        "total_area": total,
        "selected_area": selected_area,
        "dropped_area": dropped_area,
        "dropped_area_ratio": (dropped_area / total) if total else 0.0,
        "largest_share": largest_share,
        "second_share": second_share,
        "dropped_components": len(dropped),
        "center_hit": bool(ordered and ordered[0].center_hit),
        "tight_overlap": ordered[0].tight_overlap if ordered else 0.0,
    }


def _single(ctx: AssociationContext, policy: dict):
    if len(ctx.components) != 1:
        return None, ("NOT_SINGLE_COMPONENT",)
    comp = ctx.components[0]
    method = "center_region" if comp.center_hit else "unique_box_component"
    return Decision(
        method=method, selected=(comp.id,), dropped=(), status="ok",
        reasons=("SINGLE_COMPONENT",), metrics=_metrics(ctx, (comp.id,), ()),
        allowed_uses=_uses(morphology=True, length=True, width=True, area=True),
        policy_version=policy["version"]), ()


def _multipart(ctx: AssociationContext, policy: dict):
    config = policy.get("multipart", {})
    shapes = set(config.get("shapes", []))
    granularities = set(config.get("granularities", []))
    if ctx.shape not in shapes and ctx.granularity not in granularities:
        return None, ("NOT_MULTIPART",)
    ids = tuple(c.id for c in ctx.components)
    return Decision(
        method="multipart", selected=ids, dropped=(), status="review",
        reasons=("MULTIPART_KEPT",) + (("NON_INSTANCE_GRANULARITY",)
                                       if ctx.granularity != GRANULARITY_DEFAULT else ()),
        metrics=_metrics(ctx, ids, ()),
        # 多部件仍可做存在性/形态；定量由 coarse_quant/逐用途资格另行拦截。
        allowed_uses=_uses(morphology=True, length=True, width=True, area=True),
        policy_version=policy["version"]), ()


def _dominant(ctx: AssociationContext, policy: dict):
    cfg = policy["dominant"]
    reasons = []
    if not cfg.get("enabled", False):
        return None, ("DOMINANT_DISABLED",)
    if ctx.feature_id not in set(cfg.get("features", [])):
        reasons.append("FEATURE_NOT_ELIGIBLE")
    if ctx.shape not in set(cfg.get("shapes", [])):
        reasons.append("SHAPE_NOT_ELIGIBLE")
    if ctx.granularity not in set(cfg.get("granularities", [])):
        reasons.append("GRANULARITY_NOT_ELIGIBLE")
    if not cfg["min_components"] <= len(ctx.components) <= cfg["max_components"]:
        reasons.append("COMPONENT_COUNT_OUT_OF_RANGE")
    if reasons:
        return None, tuple(reasons)

    ordered = sorted(ctx.components, key=lambda c: c.area, reverse=True)
    total = sum(c.area for c in ordered)
    if total <= 0:
        return None, ("EMPTY_COMPONENTS",)
    largest, rest = ordered[0], ordered[1:]
    largest_share = largest.area / total
    second_share = max((c.area / total for c in rest), default=0.0)
    dropped_area = sum(c.area for c in rest)
    dropped_ratio = dropped_area / total
    if largest_share < cfg["largest_share_min"]:
        reasons.append("LARGEST_SHARE_TOO_SMALL")
    if second_share > cfg["second_share_max"]:
        reasons.append("SECOND_SHARE_TOO_LARGE")
    if len(rest) > cfg["max_dropped_components"]:
        reasons.append("TOO_MANY_DROPPED_COMPONENTS")
    if dropped_ratio > cfg["dropped_area_ratio_max"]:
        reasons.append("DROPPED_AREA_TOO_LARGE")
    if cfg.get("require_center_hit", True) and not largest.center_hit:
        reasons.append("LARGEST_NOT_CENTERED")
    if largest.tight_overlap < cfg["min_tight_overlap"]:
        reasons.append("LARGEST_NOT_IN_TIGHT_BOX")
    if cfg.get("drop_touching_border", True) and any(c.touches_border for c in rest):
        reasons.append("DROPPED_COMPONENT_TOUCHES_BORDER")
    if reasons:
        return None, tuple(reasons)

    width_only = bool(largest.touches_border or not cfg.get("allow_length", False))
    allowed = _uses(morphology=True, width=bool(cfg.get("allow_width", True)),
                    length=bool(cfg.get("allow_length", False)),
                    area=bool(cfg.get("allow_area", False)))
    metrics = _metrics(ctx, (largest.id,), tuple(c.id for c in rest))
    metrics.update(largest_share=largest_share, second_share=second_share,
                   dropped_area_ratio=dropped_ratio, width_only=width_only,
                   dropped_touching_border=sum(1 for c in rest if c.touches_border))
    return Decision(
        method="dominant_component", selected=(largest.id,),
        dropped=tuple(c.id for c in rest), status="ok",
        reasons=("DOMINANT_COMPONENT_SELECTED",) + (("WIDTH_ONLY",) if width_only else ()),
        metrics=metrics, allowed_uses=allowed, policy_version=policy["version"]), ()


def _ambiguous(ctx: AssociationContext, policy: dict):
    ids = tuple(c.id for c in ctx.components)
    return Decision(
        method="ambiguous", selected=ids, dropped=(), status="review",
        reasons=("ASSOCIATION_AMBIGUOUS",), metrics=_metrics(ctx, ids, ()),
        allowed_uses=_uses(), policy_version=policy["version"]), ()


_RULES: tuple[tuple[str, Callable], ...] = (
    ("unique", _single),
    ("multipart", _multipart),
    ("dominant", _dominant),
    ("ambiguous", _ambiguous),
)


def evaluate(ctx: AssociationContext, policy=None) -> Decision:
    """按规则表顺序匹配；返回第一个命中的 Decision，trace 保留全部规则的判定。"""
    cfg = _validate(_deep_merge(DEFAULT_POLICY, policy or {}))
    trace = []
    for code, rule in _RULES:
        decision, reasons = rule(ctx, cfg)
        trace.append({"rule": code, "matched": decision is not None,
                      "reasons": list(reasons)})
        if decision is not None:
            return replace(decision, trace=tuple(trace))
    # _ambiguous 恒命中；这里只是防御未来规则表被改坏。
    return Decision(method="ambiguous", selected=(), dropped=(), status="review",
                    reasons=("NO_RULE_MATCHED",), metrics=_metrics(ctx),
                    allowed_uses=_uses(), policy_version=cfg["version"],
                    trace=tuple(trace))


def components_from_cv(labels, stats, *, tight_box, roi, keep_overlap=0.5,
                       center_point=None):
    """从 cv2.connectedComponentsWithStats 的结果构造 Component 列表。

    只构造“落在 tight_box 内且 ROI 占比 >= keep_overlap”的候选；调用方仍负责
    决定哪些是 keep/candidate。labels/stats 由同一张 mask 求得。
    """
    import numpy as np

    x1, y1, x2, y2 = [int(round(float(v))) for v in tight_box]
    h, w = labels.shape
    x1, y1 = max(0, min(x1, w - 1)), max(0, min(y1, h - 1))
    x2, y2 = max(x1 + 1, min(x2, w)), max(y1 + 1, min(y2, h))
    tight = np.zeros((h, w), bool)
    tight[y1:y2, x1:x2] = True
    inside = np.bincount(labels[roi].ravel(), minlength=len(stats)).astype(float) if roi is not None else None
    inside_tight = np.bincount(labels[tight].ravel(), minlength=len(stats)).astype(float)
    candidates = [int(i) for i in np.unique(labels[tight]) if i > 0]
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    hit = int(labels[min(h - 1, cy), min(w - 1, cx)])
    out = []
    for i in candidates:
        x, y, cw, ch, area = [int(v) for v in stats[i]]
        ratio = (inside[i] / area) if inside is not None and area else 1.0
        if ratio < keep_overlap:
            continue
        comp = (labels == i)
        touches = bool(comp[0].any() or comp[-1].any() or comp[:, 0].any() or comp[:, -1].any())
        out.append(Component(
            id=i, area=area, bbox=(x, y, cw, ch),
            inside_ratio=ratio,
            tight_overlap=(inside_tight[i] / area) if area else 0.0,
            center_hit=(i == hit), touches_border=touches))
    return tuple(out)
