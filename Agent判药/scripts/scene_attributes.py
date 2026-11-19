# -*- coding: utf-8 -*-
"""scene_attributes — 场景证据提取（三层模型第 2/3 层，纯几何，无 VLM）。

三层模型（2026-09-22 设计决议）：
  第 1 层 检测原子 = 结晶本体（feature_taxonomy P0 已如此，不在本模块范围）；
  第 2 层 宿主语境（晶在胞内/散在）= 关系属性 → 由 VLM 场景问句提供
          （agent_loop.VLMClient.scene_check），不建检测类；
  第 3 层 排列（质心共线性）/ 丰度（每视野检出数）= 场景级判读 → 本模块从
          测量记录直接计算。

铁律：场景证据只作备选排序（rule_engine.scene_fit）与报告展示，不进置信度、
不进 support_count、不计入 D8 证据类数（rule_engine._counts_toward_max 的
"supporting 不进分母"语义不破坏）。

输入对齐 几何测量/measure.py 的测量契约：每记录含 image_stem、
bbox_x/bbox_y/bbox_w_px/bbox_h_px（质心 = bbox 中心）、feret_max_um /
primary_value_um。纯标准库。
"""
import json
import math
import statistics
from collections import Counter

# ---- 丰度档位：每视野检出数中位数 → 措辞档位（估计值，报告须标注"受检测召回影响"）----
ABUNDANCE_FEW_MAX = 1       # ≤1 个/视野 → 可见
ABUNDANCE_MANY_MAX = 9      # 2~9 个/视野 → 众多；≥10 → 极多
ABUNDANCE_FEW = "可见"
ABUNDANCE_MANY = "众多"
ABUNDANCE_VERY = "极多"

# ---- 共线性启发式：单视野 ≥ROW_MIN_INSTANCES 个实例才判；质心到拟合直线垂距 RMS
#      与平均近邻间距之比 ≤ROW_RMS_RATIO → 判"排列成行" ----
ROW_MIN_INSTANCES = 3
ROW_RMS_RATIO = 0.25

# VLM 场景问句的英文档位 → 中文丰度措辞（merge_vlm_scene 用）
VLM_ABUNDANCE_MAP = {"few": ABUNDANCE_FEW, "many": ABUNDANCE_MANY, "very_many": ABUNDANCE_VERY}

# 结晶类特征：scene_check 强制触发的条件（样本检出任一此类即逐视野发起场景判读，
# 2026-09-22 用户决议）。钙盐之外的结晶不判位置（钟乳体/升华物语境不同）。
CRYSTAL_FEATURES = ("cluster_crystal", "single_crystal", "needle_crystal", "sand_crystal")


def abundance_level(median_count):
    """每视野检出数中位数 → 丰度措辞档位。"""
    if median_count <= ABUNDANCE_FEW_MAX:
        return ABUNDANCE_FEW
    if median_count <= ABUNDANCE_MANY_MAX:
        return ABUNDANCE_MANY
    return ABUNDANCE_VERY


def _centroid(rec):
    """bbox 中心作为实例质心的近似（测量契约无现成质心字段）。"""
    try:
        x = float(rec["bbox_x"]) + float(rec["bbox_w_px"]) / 2.0
        y = float(rec["bbox_y"]) + float(rec["bbox_h_px"]) / 2.0
        return (x, y)
    except (KeyError, TypeError, ValueError):
        return None


def _tls_line(pts):
    """总体最小二乘直线拟合 → (单位方向向量 d, 单位法向量 n, 中心点)。"""
    n = len(pts)
    mx = sum(p[0] for p in pts) / n
    my = sum(p[1] for p in pts) / n
    sxx = sum((p[0] - mx) ** 2 for p in pts)
    syy = sum((p[1] - my) ** 2 for p in pts)
    sxy = sum((p[0] - mx) * (p[1] - my) for p in pts)
    theta = 0.5 * math.atan2(2.0 * sxy, sxx - syy)   # 主方向角（最大特征值方向）
    d = (math.cos(theta), math.sin(theta))
    nvec = (-math.sin(theta), math.cos(theta))
    return d, nvec, (mx, my)


def row_verdict(recs):
    """单视野实例 → (verdict, rms_ratio)。

    verdict ∈ {"rows", "scattered", "insufficient"}；insufficient 表示实例数
    不足（<ROW_MIN_INSTANCES）或质心缺失，不作排列声明。
    """
    pts = [c for c in (_centroid(r) for r in recs) if c]
    if len(pts) < ROW_MIN_INSTANCES:
        return "insufficient", None
    d, nvec, (mx, my) = _tls_line(pts)
    perp = [(p[0] - mx) * nvec[0] + (p[1] - my) * nvec[1] for p in pts]
    rms = math.sqrt(sum(v * v for v in perp) / len(perp))
    proj = sorted(p[0] * d[0] + p[1] * d[1] for p in pts)
    gaps = [b - a for a, b in zip(proj, proj[1:])]
    spacing = sum(gaps) / len(gaps) if gaps else 0.0
    if spacing < 1e-6:
        return "scattered", None
    ratio = rms / spacing
    return ("rows" if ratio <= ROW_RMS_RATIO else "scattered"), round(ratio, 3)


def _fov_sort_key(k):
    return (0, int(k), "") if str(k).isdigit() else (1, 0, str(k))


def empty_scene():
    """空场景骨架（VLM-only 路径兜底，schema 与 build_scene 一致）。"""
    return {
        "per_fov_counts": {}, "n_total": 0, "abundance_obs": None,
        "size_stats": None, "arrangement": [],
        "sources": {"arrangement": "geometry", "abundance": "geometry"},
    }


def build_scene(sample_records):
    """样本的测量记录 → 场景证据 dict（纯几何）。

    按 image_stem 分视野聚合：每视野检出数、丰度档位、尺寸分布、
    逐视野排列判读（质心共线性启发式）。
    """
    fovs = {}
    for rec in sample_records or []:
        fovs.setdefault(str(rec.get("image_stem", "?")), []).append(rec)
    fov_items = sorted(fovs.items(), key=lambda kv: _fov_sort_key(kv[0]))

    per_fov_counts = {k: len(v) for k, v in fov_items}
    counts = [len(v) for _, v in fov_items]
    abundance_obs = abundance_level(statistics.median(counts)) if counts else None

    sizes = []
    for rec in sample_records or []:
        v = rec.get("feret_max_um")
        if not isinstance(v, (int, float)) or v <= 0:
            v = rec.get("primary_value_um")
        if isinstance(v, (int, float)) and v > 0:
            sizes.append(float(v))
    size_stats = None
    if sizes:
        size_stats = {"min": round(min(sizes), 1), "max": round(max(sizes), 1),
                      "median": round(statistics.median(sizes), 1)}

    arrangement = []
    for k, recs in fov_items:
        verdict, ratio = row_verdict(recs)
        arrangement.append({"fov": k, "n": len(recs), "verdict": verdict, "rms_ratio": ratio})

    return {
        "per_fov_counts": per_fov_counts,
        "n_total": len(sample_records or []),
        "abundance_obs": abundance_obs,
        "size_stats": size_stats,
        "arrangement": arrangement,
        "sources": {"arrangement": "geometry", "abundance": "geometry"},
    }


def merge_vlm_scene(scene, answers):
    """VLM 场景判读结果并入 scene（返回新 dict，不改入参）。

    归并策略（设计决议）：VLM 与几何**并存记录、VLM 优先参与拟合**——
    * arrangement：逐视野把 VLM 判读写入 `vlm_verdict`（几何 verdict 保留），
      sources.arrangement 置 "vlm"（有有效 VLM 判读时）；
    * abundance：几何观测保留在 `abundance_obs_geometry`，`abundance_obs` 被
      VLM 多数票覆盖，sources.abundance 置 "vlm"；
    * crystal_location（胞内/游离/混合）：多数票汇总进 `vlm` 块（纯第 2 层属性，
      当前仅报告展示，不参与 scene_fit）。
    """
    s = json.loads(json.dumps(scene or empty_scene(), ensure_ascii=False))
    answers = [a for a in (answers or []) if isinstance(a, dict)]
    vlm_block = s.setdefault("vlm", {"per_fov": {}, "crystal_location": "unknown",
                                     "arrangement": "unknown", "abundance": None, "n_calls": 0})
    by_index = {a["fov"]: a for a in s.get("arrangement", []) if isinstance(a, dict) and "fov" in a}
    loc_votes, loc_confs, arr_votes, abd_votes = [], [], [], []
    for a in answers:
        fov = str(a.get("fov", "?"))
        vlm_block["per_fov"][fov] = a
        entry = by_index.get(fov)
        if a.get("arrangement") and a["arrangement"] != "unknown":
            if entry is not None:
                entry["vlm_verdict"] = a["arrangement"]   # 几何条目存在 → 并存记录
            arr_votes.append(a["arrangement"])            # 票 regardless：VLM-only 路径也计
        if a.get("crystal_location") and a["crystal_location"] != "unknown":
            loc_votes.append(a["crystal_location"])
            try:
                loc_confs.append(float(a.get("conf", 0.5)))
            except (TypeError, ValueError):
                loc_confs.append(0.5)
        ab = VLM_ABUNDANCE_MAP.get(a.get("abundance"))
        if ab:
            abd_votes.append(ab)
    vlm_block["n_calls"] = len(answers)
    if loc_votes:
        vlm_block["crystal_location"] = Counter(loc_votes).most_common(1)[0][0]
        # 保守取最小置信度（rule_engine.SCENE_LOC_CONF_GATE 评分门限用）
        vlm_block["crystal_location_conf"] = round(min(loc_confs), 2)
    if arr_votes:
        vlm_block["arrangement"] = Counter(arr_votes).most_common(1)[0][0]
        s.setdefault("sources", {})["arrangement"] = "vlm"
    if abd_votes:
        if s.get("abundance_obs") is not None:
            s["abundance_obs_geometry"] = s["abundance_obs"]
        s["abundance_obs"] = Counter(abd_votes).most_common(1)[0][0]
        s.setdefault("sources", {})["abundance"] = "vlm"
    return s
