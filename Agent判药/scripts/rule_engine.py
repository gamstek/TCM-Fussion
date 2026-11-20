# -*- coding: utf-8 -*-
"""rule_engine — Agent 判药·符号规则引擎（确定性，纯标准库）。

职责（设计方案 §5.2）：
  1) 候选生成：判据特征有正证据的药材/组进入候选池（不读 manifest.herb，防泄漏）；
  2) 逐候选评分：S(h) = Σ w(tier)·q(state)·morph·m(abundance)·rel·s(scene)
                 − Σ 错误特征出现罚 0.4·rel
                 − Σ required 缺失罚 0.3·w·sens(tier)
       置信度 C(h) = min(1, max(0, S_required) / Σ_required w) —— 只由必备判据决定；
              佐证(supporting)不影响置信度，只进 support_count（D8 广度门槛）与并列排序（见 _counts_toward_max）
  3) 组级归并 + 裁决带：confirmed / probable / group_level / insufficient_evidence；D8 证据数量门槛：独立证据特征不足 MIN_SUPPORT_FEATURES 类时阳性和解降级 insufficient_evidence，
     但 D8b 条件放行（v1.6，2026-09-22）：药典只给 1 个显微判据的品种（59/85 味）在仅 1 类证据时，
     若该类证据已把 top1 与并列者区分开（decisive）且置信度达 confirmed 带，则放行出结论（仍标人工复核）

排他层拆除记录（2026-09-22）：原「候选 exclusions 命中 → 退出排名/升级 suspect」及其 scope、
升级前件、suspect_subject 一并删除——库内唯二两条否决规则（EX-DH-1 石细胞、EX-DH-2 簇晶<45μm）
的唯一出处是一份**示例文档**（《一中药显微鉴别标准示例大黄粉末显微特征.docx》），非药典依据，
不足以支撑否决真品。裁决只由判据评分 + 裁决带 + D8 门槛做出。

设计铁律：本模块不得 import 任何第三方库；所有数字结论可溯源到证据卡。
"""
import json
import statistics
from pathlib import Path

# ---------------------------------------------------------------- 常量
W_TIER = {"P0": 1.0, "P1": 0.5, "C": 0.3}          # 通道权重（设计方案 §4.2）
SENS_TIER = {"P0": 0.8, "P1": 0.4, "C": 0.1}       # 阴性证据灵敏度（§4.4）
Q_STATE = {                                        # 量化核验五态得分（§5.2）
    "IN_RANGE": 1.0, "EXTREME": 0.7, "NO_REF": 0.6, "UNCAL": 0.5, "OUT_OF_RANGE": 0.0,
}
OUT_PENALTY = 0.5          # OUT_OF_RANGE 的负分系数
WRONG_PRESENT_PENALTY = 0.4
ABSENT_REQUIRED_PENALTY = 0.3
M_ABUNDANCE = {"甚多": 1.0, "极多": 1.0, "较多": 1.0, "众多": 1.0,
               "多见": 0.95, "可见": 0.9, "偶见": 0.9, "少见": 0.9}

BAND_CONFIRMED = 0.75
BAND_PROBABLE = 0.50
MARGIN_DELTA = 0.10
MIN_SUPPORT_FEATURES = 2   # Baseline D8; fusion may explicitly opt into a versioned experiment.
NONOFF_MAX_CONF = 0.49     # 判据依据非药典正文的候选置信度封顶（低于 BAND_PROBABLE，不得作为结论）

# ---- 候选级"宽区间主导"置信度压制 ----
# 若某候选的 score 被"单一宽区间判据"主导（贡献占比过高），其置信度被设上限。
# 针对马齿苋 cluster 7~108 这类"宽区间 + 判据少"的候选；多判据真药（如上段大黄
# cluster+vessel+starch）贡献分散，不触发。sp 只在贡献占比达阈值且区间足够宽时生效。
DOM_SHARE_THRESHOLD = 0.55    # 单一判据贡献占总分比率阈值
WIDE_SPAN_THRESHOLD = 60.0    # 判据区间跨度（μm）阈值：超过视为"宽"
DOM_MAX_FACTOR = 0.60         # 命中后在置信带上限（≤probable 上沿，防武断 confirmed）

PRIMARY_DIMS = ("feret_max", "mean_width")         # 与 profiles.py 对齐的两个判据量纲
SHAPE_KEYS = ("defect_count", "mean_defect_depth_ratio", "solidity", "aspect_ratio",
              "rectangularity", "roundness", "circularity", "tortuosity",
              "dt_peak_count", "skeleton_len_um", "feret_max_um")

# ---- 场景证据（三层模型第 2/3 层，2026-09-22）：有界参与评分 + 平分排序 ----
# 场景证据 = scene_attributes.build_scene（几何）+ VLM 场景问句（agent_loop.scene_check）。
# 两条通道、职责不同：
#   ① 宿主语境维（第 2 层，仅 VLM）→ 以有界乘数 s(scene) 作用于**置信度**：
#      相符 ×SCENE_MATCH_FACTOR、相悖 ×SCENE_CONTRA_FACTOR，仅 VLM 置信度达
#      SCENE_LOC_CONF_GATE 时生效，且只作用于 morph 声明了"细胞内含/散在"的判据
#      （虎杖类正文无表述者天然不受影响）；不改 support_count、不计入 D8/D8b、
#      cap/decisive 结构不动。
#      施加位置：**所有置信度封顶之后**（2026-09-23 修正）。原先并入封顶前的判据贡献分，
#      乘法调制会被 min(conf, DOM_MAX_FACTOR) 吸收——封顶值 0.60 与相悖乘数 0.60 同值，
#      宽区间判据（马齿苋 7~108μm）的惩罚完全归零，而不触发封顶的候选（赤芍 quant=null）
#      乘数才生效，等效于"乘数只对无药典量程的候选有效"。修正后两个机制相乘，
#      结果记在 candidate.scene_factor_eff（逐判据的 s_loc 仍记在 check.scene_factor）；
#   ② 丰度维 + 排列维（第 3 层）→ 仅 (confidence, support_count) 完全并列时作为
#      排序第三键（先于 score），不改 conf。
# 共同铁律：场景证据不进 support_count、不计入 D8 证据类数。
SCENE_ROW_KEYWORDS = ("排列成行", "成行")            # 判据 morph 中的"行列排列"表述
SCENE_SCATTER_KEYWORDS = ("散在", "散落")            # 判据 morph 中的"散在"表述
SCENE_INTRA_KEYWORDS = ("薄壁细胞", "厚角细胞", "细胞中", "细胞内", "细胞含")
#                                   ↑ 判据 morph 中的"细胞内含"表述（宿主语境维）
SCENE_FREE_KEYWORDS = ("散在", "散落")               # 判据 morph 中的"游离散在"表述
SCENE_MATCH_FACTOR = 1.10     # 宿主语境相符：置信度乘数（小幅加分，末端 conf≤1 封顶）
SCENE_CONTRA_FACTOR = 0.60    # 宿主语境相悖：置信度乘数（减分；VLM 误判风险由置信门限压制）
SCENE_LOC_CONF_GATE = 0.70    # VLM crystal_location 判读置信度门限（低于则只记录不评分）
ABUNDANCE_RANK = {"偶见": 0, "少见": 0, "可见": 1, "多见": 1,
                  "众多": 2, "较多": 2, "甚多": 3, "极多": 3}


# ---------------------------------------------------------------- 知识库
class KnowledgeBase:
    def __init__(self, kb_path):
        with open(kb_path, encoding="utf-8") as f:
            self.raw = json.load(f)
        self.meta = self.raw["meta"]
        self.tier_of = {}
        for tier, feats in self.raw["feature_vocab"].items():
            for fid in feats:
                self.tier_of[fid] = tier
        self.groups = self.raw["groups"]
        self.herbs = {h["herb"]: h for h in self.raw["herbs"]}
        self.group_of = {h: gid for gid, g in self.groups.items() for h in g["members"]}

    def tier(self, feature_id):
        return self.tier_of.get(feature_id, "C")

    def candidates(self):
        """候选 = 组（4个） ∪ 非组成员药材。组成员以组为单位参评。"""
        out = []
        for gid, g in self.groups.items():
            out.append({
                "id": gid, "kind": "group", "display": " / ".join(g["members"]),
                "members": g["members"], "criteria": g["criteria"],
            })
        for name, h in self.herbs.items():
            if h.get("no_criteria"):
                continue
            if name in self.group_of:
                continue
            out.append({
                "id": name, "kind": "herb", "display": name,
                "members": [name], "criteria": h["criteria"],
                # 判据依据非药典正文（地方标准/教材口径）→ 置信度封顶，仅作备选
                "non_official_basis": bool(h.get("non_official_basis")),
            })
        return out


# ---------------------------------------------------------------- 证据卡
class EvidenceCard:
    """一个特征在样本级的聚合证据（设计方案 §4.1）。"""

    def __init__(self, feature_id):
        self.feature_id = feature_id
        self.instance_refs = []          # (image_stem, instance_id)
        self.confs = []                  # 检出置信度（缺失时用中性默认并在卡上标记）
        self.conf_assumed = False
        self.uncal_refs = []             # 未标定实例
        self.dims = {}                   # dim -> {"values":[...], "uncal_only":bool}
        self.shape = {}                  # shape key -> median
        self.n_components_multi = 0      # multi_component 实例数

    @property
    def present(self):
        """检出即 present（含量化实例与仅存在性/未标定实例）。"""
        return len(self.instance_refs) + len(self.uncal_refs) > 0

    @property
    def n_instances(self):
        """可量化实例数（标定 ok 且主判尺寸有效）。"""
        return len(self.instance_refs)

    @property
    def n_seen(self):
        return len(self.instance_refs) + len(self.uncal_refs)

    @property
    def median_conf(self):
        return statistics.median(self.confs) if self.confs else None

    def dim_stats(self, dim):
        """返回 (median, p25, p75, min, max, n_measured, n_uncal)；无测量值返回 None。"""
        d = self.dims.get(dim)
        if not d or not d["values"]:
            return None
        vals = sorted(d["values"])
        n = len(vals)
        med = statistics.median(vals)
        p25 = vals[int(round(0.25 * (n - 1)))]
        p75 = vals[int(round(0.75 * (n - 1)))]
        return dict(median=med, p25=p25, p75=p75, min=vals[0], max=vals[-1],
                    n_measured=n, n_uncal=len(self.uncal_refs))

def pool_evidence(records):
    """measurements 记录列表 → {feature_id: EvidenceCard}。

    安全纪律：丢弃记录中的 herb 字段（防泄漏，设计方案 §1.3）。
    """
    cards = {}
    for rec in records:
        fid = rec.get("feature")
        if not fid or fid in ("suspect", "疑似待定"):
            continue
        card = cards.setdefault(fid, EvidenceCard(fid))
        stem, iid = rec.get("image_stem", "?"), rec.get("instance_id", "?")
        flags = rec.get("flags") or []
        # IGM 分割状态非 ok（框先验异常 / 极性相悖 / 中心块越界）→ 该实例的**尺寸不可信**，
        # 整条不进证据池：既不进量化值，也不当"存在性"证据。2026-10-09 实测：菊花样本的两个
        # 花粉实例 igm_status 均为 review，却照样参与打分 → 产出 conf=1.00 的「鉴别成立 → 红花」
        # 假阳（判药取实例中位数 46.0μm，落进红花卡片的单侧窗 [None,60]）。
        # 代价是"检到但不可信"退化成"未检出"，这是**保守方向**：宁可不出结论，不出错结论。
        if "igm_review" in flags:
            continue
        calib_ok = rec.get("calibration_status") in ("ok", "median_herb") and rec.get("calibration_px_per_um")
        conf = rec.get("conf", rec.get("confidence"))
        primary = rec.get("primary_value_um")
        # 量化可用性闸门（2026-10-09）：几何对"束/片/团"这类**粗粒度框**打 coarse_quant
        # （granularity != instance）——其尺寸不对应药典单值量纲（束宽≠单根直径、整片尺寸≠
        # 单个细胞直径）。此时剔除量化值，让该实例与"未标定"同构地走 uncal_refs = **只作存在性**。
        # 与 igm_review（整条丢弃）相邻但更轻：检到了就仍算一类独立证据，只是不参与尺寸核验。
        coarse = "coarse_quant" in flags
        prim_ok = primary is not None and primary > 0 and calib_ok and not coarse
        if prim_ok:
            card.instance_refs.append((stem, iid))
            card.confs.append(float(conf) if conf is not None else 0.85)
            if conf is None:
                card.conf_assumed = True
            dim = None
            pd = rec.get("primary_dimension") or ""
            if "feret_max" in pd:
                dim = "feret_max"
            elif "mean_width" in pd or "median_width" in pd:
                dim = "mean_width"
            if dim is None:
                # 回退：主判尺寸按数值语义归入 feret_max（直径/长类）
                dim = "feret_max"
            card.dims.setdefault(dim, {"values": [], "uncal_only": False})["values"].append(float(primary))
        else:
            # 检出但无法量化（未标定/尺寸无效）→ 仅存在性证据
            card.uncal_refs.append((stem, iid))
            card.confs.append(float(conf) if conf is not None else 0.85)
            if conf is None:
                card.conf_assumed = True
        if "multi_component" in flags:
            card.n_components_multi += 1
        for k in SHAPE_KEYS:
            v = rec.get(k)
            if isinstance(v, (int, float)):
                card.shape.setdefault(k, []).append(float(v))
    # 形状描述子取中位数
    for card in cards.values():
        card.shape = {k: statistics.median(v) for k, v in card.shape.items() if v}
    return cards


# ---------------------------------------------------------------- 核验
def check_quant_state(quant, stats):
    """量化核验五态（设计方案 §5.2 表）。"""
    if quant is None:
        return "NO_REF", None
    if stats is None:
        return "UNCAL", None
    med = stats["median"]
    umin, umax, extreme = quant.get("min"), quant.get("max"), quant.get("extreme")
    if umin is None and umax is None:
        return "NO_REF", None
    if umin is not None and med < umin:
        return "OUT_OF_RANGE", f"median {med:.1f}μm < 下界 {umin}μm"
    if umax is not None and med > umax:
        if extreme is not None and med <= extreme:
            return "EXTREME", f"median {med:.1f}μm 落入'个别可达'区间 (≤{extreme}μm)"
        return "OUT_OF_RANGE", f"median {med:.1f}μm > 上界 {umax}μm"
    note = f"median {med:.1f}μm ∈ [{umin}, {umax}]μm"
    if extreme is not None and stats["max"] > extreme:
        note += f"，但有实例 max {stats['max']:.1f}μm 超 extreme {extreme}μm（记录 flag）"
    return "IN_RANGE", note


# 形态签名软核验（明场几何代理；阈值与评测联动，先宽后严）
def morphology_plausibility(feature_id, card):
    if not card or not card.shape:
        return 0.85  # 无形状信息 → 中性
    s = card.shape
    if feature_id == "cluster_crystal":
        return 1.0 if s.get("defect_count", 0) >= 3 else (0.6 if s.get("defect_count", 0) >= 1 else 0.4)
    if feature_id == "single_crystal":
        if s.get("rectangularity", 0) >= 0.55 or s.get("aspect_ratio", 0) >= 2:
            return 1.0
        return 0.5
    if feature_id == "needle_crystal":
        a = s.get("aspect_ratio", 0)
        return 1.0 if a >= 4 else (0.7 if a >= 2.5 else 0.4)
    if feature_id in ("fiber", "crystal_fiber"):
        a = s.get("aspect_ratio", 0)
        return 1.0 if a >= 5 else (0.8 if a >= 3 else 0.5)
    if feature_id == "vessel":
        if s.get("aspect_ratio", 0) >= 3 or s.get("tortuosity", 0) >= 2 or s.get("skeleton_len_um", 0) >= 50:
            return 1.0
        return 0.6
    if feature_id == "starch_grain":
        if s.get("roundness", 0) >= 0.65 or s.get("dt_peak_count", 0) >= 2:
            return 1.0
        return 0.7
    if feature_id == "secretory_tissue":
        return 1.0 if s.get("circularity", 0) >= 0.5 else 0.7
    if feature_id == "pollen_grain":
        return 1.0 if s.get("roundness", 0) >= 0.6 else 0.7
    if feature_id == "hypha":
        return 1.0 if s.get("aspect_ratio", 0) >= 5 else 0.6
    if feature_id == "non_glandular_hair":
        if s.get("aspect_ratio", 0) >= 4 or s.get("feret_max_um", 0) >= 100:
            return 1.0
        return 0.6
    return 0.85


def conf_factor(card):
    mc = card.median_conf if card else None
    if mc is None:
        return 0.9
    if mc >= 0.8:
        return 1.0
    if mc >= 0.6:
        return 0.9
    if mc >= 0.5:
        return 0.75
    return 0.4


def quality_factor(card):
    q = 1.0
    if card and card.n_components_multi:
        q -= 0.05 * min(card.n_components_multi, 4)   # 过分割/多连通域轻罚
    return max(0.8, q)


def reliability(card, kb):
    """rel = conf_factor × quality_factor（证据质量，通道无关）。

    注意：通道权重 w(tier) 只在判据贡献 (w·q·morph·m·rel) 与罚项中体现一次；
    若 rel 再含 w 会造成弱通道双重降权，使 P1/C-only 药材永远无法进入 probable 带，
    与设计方案"弱通道结论上限封顶为倾向"矛盾。
    """
    if card is None:
        return 0.0
    return round(conf_factor(card) * quality_factor(card), 4)


# ---------------------------------------------------------------- 评分
def _counts_toward_max(role):
    """该判据是否计入置信度分母 —— **只计 required**，supporting 完全不进置信度。

    required 是"必须具备"，无论检出与否都占分母的 w（未检出另有缺失罚，比值被压低）。
    supporting 的语义是"有则加分、无则不影响"，且按 v1.2 立库意图，补录 supporting 是为了
    **增强证据广度、助力 D8≥2 类门槛**，不是抬置信度。两种口径都不该进这个分母：
      * 未检出的 supporting 进分母 → "知识库给这味药多记了几条 supporting"变成纯稀释，
        记录越详尽的药材排名越靠后（实测白芍：簇晶判据同样拿满 +1.000，却因 4 条未检出的
        supporting 把分母撑到 3.80，置信度 0.263 排第 9；只记 2 条判据的辣蓼同样得分
        却得 0.769 排第 1）；
      * 已检出的 supporting 进分母 → 分子增量 w·q·morph·rel ≤ w 而分母增量恒为 w，
        于是"多检出佐证反而降分"（实测猪苓：方晶 required 单独 1.0000，加上已检出的
        菌丝 supporting 后反降到 0.8667）。
    故 supporting 的净贡献（sup_net）从置信度分子里剔除，只在排序 tiebreak（support_count
    → score）与 D8 广度门槛中生效。这样 conf 恒等于"必备判据的达成度"：required 全中→1.0，
    required 未检出（如川楝子：晶纤维未检出、仅簇晶作佐证）→ 归 0，不会被佐证顶起来。
    """
    return role == "required"


def _crit_span(cand, feature_id):
    """候选某判据的量化区间跨度（max-min）；无量化/单边返回 None。"""
    for cr in cand.get("criteria") or []:
        if cr["feature"] == feature_id:
            q = cr.get("quant")
            if q and q.get("min") is not None and q.get("max") is not None:
                return q["max"] - q["min"]
    return None


def _support_feature_ids(res):
    """实际支持该候选的独立证据特征集合（present、非 OUT、贡献分>0）。

    同一特征的多个量纲判据（如针晶长+宽）只算一类，不构成独立证据。
    用于 D8 门槛与排序 tiebreak（判据依据相同时，佐证类数多者优先）。
    """
    return sorted({c["feature_id"] for c in res["checks"]
                   if c["present"] and c["state"] not in ("OUT_OF_RANGE", "NOT_DETECTED")
                   and c["score_contribution"] > 0})


def evaluate_candidate(cand, cards, kb, scene=None):
    """对单个候选打分，返回结果 dict（含逐判据核验明细，供报告引用）。

    scene（可选）：场景证据。宿主语境维以有界乘数（_scene_location_factor）**在置信度
    封顶之后**作用于 required 判据，不改 support_count/D8。
    """
    checks, score, s_max, sup_net = [], 0.0, 0.0, 0.0
    loc_num, loc_den = 0.0, 0.0                  # 宿主语境乘数的加权效力（required 内）
    cand_features = {c["feature"] for c in cand["criteria"]}

    for crit in cand["criteria"]:
        fid = crit["feature"]
        tier = kb.tier(fid)
        w = W_TIER[tier]
        card = cards.get(fid)
        present = bool(card and card.present)
        quant = crit.get("quant")
        dim = (quant or {}).get("dim", "feret_max")
        stats = card.dim_stats(dim) if present else None
        state, qnote = check_quant_state(quant, stats)
        morph = morphology_plausibility(fid, card) if present else 0.0
        rel = reliability(card, kb) if present else 0.0
        m = M_ABUNDANCE.get(crit.get("abundance") or "", 1.0)
        s_loc = _scene_location_factor(crit, scene) if present else 1.0
        q = Q_STATE[state] if present else 0.0
        # 判据贡献分**不含**宿主语境乘数（2026-09-23 修正）：纳入贡献分时它会被
        # 置信度封顶的 min() 吃掉——宽区间判据（马齿苋 7~108μm）conf 恒被压到
        # DOM_MAX_FACTOR=0.60，而相悖乘数 0.60 恰与之同值，惩罚就此归零；
        # 反之不触发封顶的候选（赤芍 quant=null）乘数才有效，等效于"乘数只对
        # 无药典量程的候选生效"。乘数改在封顶之后以 s_loc_eff 施加（见下）。
        base = w * q * morph * m * rel
        if present and state == "OUT_OF_RANGE":
            contrib = 0.0
            penalty = OUT_PENALTY * w * rel
            score -= penalty
        else:
            contrib = base
            penalty = 0.0
            score += contrib
        if crit["role"] == "required":
            s_max += w                      # required 恒在分母：缺了要压低比值（另有缺失罚）
            if present and state != "OUT_OF_RANGE":
                loc_num += base * s_loc     # 乘数的加权效力：Σ(base×s_loc)/Σbase
                loc_den += base
        else:
            sup_net += contrib - penalty    # 佐证的净贡献：只用于排序 tiebreak 与证据广度，
                                            # 不进置信度（见 _counts_toward_max）
        checks.append({
            "feature_id": fid, "role": crit["role"], "tier": tier,
            "present": present, "state": state if present else "NOT_DETECTED",
            "quant_note": qnote, "morph_plausibility": round(morph, 3) if present else None,
            "reliability": rel,
            # 评分公式逐项（S(h)=Σ w·q·morph·m·rel − 罚项）：报告"评分计算过程"直接复算用
            "weight": w,                       # w(tier)：通道权重（P0 1.0 / P1 0.5 / C 0.3）
            "q_state": q,                      # q(state)：量化核验五态得分
            "abundance": crit.get("abundance"),        # 判据声明的丰度措辞（未声明为 None）
            "m_abundance": m,                  # m(abundance)：丰度乘数（未声明取 1.0）
            "counts_toward_conf": _counts_toward_max(crit["role"]),   # 是否进置信度（supporting 不进）
            "score_contribution": round(contrib, 4) if present else 0.0,
            "penalty": round(penalty, 4),
            "n_instances": card.n_instances if present else 0,
            "stats": stats,
            "morph_text": crit.get("morph"),
            # 结构化判据区间（报告"测量分布 vs 候选区间"表用；quant 为 null 时不输出）
            "quant_bounds": ({"dim": dim, "min": quant.get("min"), "max": quant.get("max"),
                              "extreme": quant.get("extreme")} if quant else None),
            # 宿主语境乘数（第 2 层评分通道；1.0 或无 scene 时不出现在报告语义中）
            "scene_factor": (round(s_loc, 2) if present else None),
            "instance_refs": [f"{s}#{i}" for s, i in card.instance_refs[:8]] if present else [],
        })

    # 错误特征出现罚：P0 特征出现且不在判据中，且实例数≥3
    wrong = []
    for fid, card in cards.items():
        if fid in cand_features or not card.present:
            continue
        if kb.tier(fid) != "P0" or card.n_instances < 3:
            continue
        pen = WRONG_PRESENT_PENALTY * W_TIER[kb.tier(fid)] * reliability(card, kb)
        score -= pen
        wrong.append({"feature_id": fid, "n_instances": card.n_instances,
                      "penalty": round(pen, 4),
                      "note": "该药判据不含此 P0 特征但大量检出"})

    # required 缺失罚
    absent = []
    for crit in cand["criteria"]:
        if crit["role"] != "required":
            continue
        card = cards.get(crit["feature"])
        if card and card.present:
            continue
        tier = kb.tier(crit["feature"])
        pen = ABSENT_REQUIRED_PENALTY * W_TIER[tier] * SENS_TIER[tier]
        score -= pen
        absent.append({"feature_id": crit["feature"], "penalty": round(pen, 4),
                       "note": "主判特征未检出（按通道灵敏度打折）"})

    # 置信度 = 必备判据的达成度：分子只取 required 部分（佐证净贡献 sup_net 已剔除），
    # 分母只含 required 的满配权重；佐证的作用体现在 support_count（证据广度）与并列排序
    res = {
        "id": cand["id"], "kind": cand["kind"], "display": cand["display"],
        "score": round(score, 4), "score_max": round(s_max, 4),
        "confidence": 0.0,
        "checks": checks, "wrong_present": wrong, "absent_required": absent,
        # 封顶标志恒存在（布尔），供报告/审计统一读取；无 required 判据的候选保持 False
        "cap_dom_wide": False, "cap_non_official": False,
    }
    res["support_count"] = len(_support_feature_ids(res))
    if s_max > 0:
        # 评分计算过程（报告"评分计算过程"节直接复算用，2026-09-28 新增）：
        #   S(h) = Σ w·q·morph·m·rel − Σ 罚项；conf = (Σrequired w·q·morph·m·rel) / Σrequired w
        #         − sup_net 从分子剔除（supporting 只作 tiebreak 与 D8 广度）
        #         − 宽区间主导 cap（DOM_MAX_FACTOR）/ 非药典依据 cap（NONOFF_MAX_CONF）
        #         − 宿主语境乘数 s_loc_eff 在**所有封顶之后**施加
        res["conf_required_num"] = round(score - sup_net, 4)   # 分子：required 判据贡献合计
        res["conf_denom"] = round(s_max, 4)                    # 分母：required 满配权重合计
        res["sup_net"] = round(sup_net, 4)                     # 佐证净贡献（不进置信度）
        conf = min(max(0.0, (score - sup_net) / s_max), 1.0)
        res["conf_raw"] = round(conf, 4)                       # 封顶/场景乘数之前的置信度
        if score > 0 and checks:
            dom = max(checks, key=lambda c: c["score_contribution"])
            if (dom["score_contribution"] / score >= DOM_SHARE_THRESHOLD
                    and _crit_span(cand, dom["feature_id"]) is not None
                    and _crit_span(cand, dom["feature_id"]) >= WIDE_SPAN_THRESHOLD):
                conf = min(conf, DOM_MAX_FACTOR)
                res["cap_dom_wide"] = True
        # 判据依据非药典正文（地方标准/教材口径）→ 封顶到 probable 带以下，
        # 可作备选列出，但不得据此下 confirmed/probable 结论
        if cand.get("non_official_basis"):
            conf = min(conf, NONOFF_MAX_CONF)
            res["cap_non_official"] = True
        # 宿主语境乘数在**所有封顶之后**施加（2026-09-23 修正）：若并入封顶前的贡献分，
        # 乘法调制会被 min(conf, DOM_MAX_FACTOR) 吸收——封顶值 0.60 与相悖乘数 0.60
        # 同值，宽区间判据（如马齿苋）的惩罚会完全消失。放在最后则两个机制相乘：
        # 马齿苋 0.60×0.6=0.36、虎杖 0.60×1.0=0.60，接线不再被吃掉。
        # s_loc_eff = 乘数对 required 达成度的加权效力，无有效项时取 1.0。
        s_loc_eff = (loc_num / loc_den) if loc_den > 0 else 1.0
        conf = min(conf * s_loc_eff, 1.0)
        res["scene_factor_eff"] = round(s_loc_eff, 4)
        res["confidence"] = round(conf, 4)
    return res


# ---------------------------------------------------------------- 场景证据拟合
def _scene_location_factor(crit, scene):
    """宿主语境乘数 s(scene)（第 2 层，仅 VLM 来源）。

    VLM 判读 crystal_location（intracellular/free，置信度 ≥SCENE_LOC_CONF_GATE）
    与该判据 morph 的"细胞内含/散在"表述比对：相符 ×SCENE_MATCH_FACTOR、
    相悖 ×SCENE_CONTRA_FACTOR；mixed/无判读/置信不足/morph 无相关表述 → 1.0。
    返回值按 required 达成度加权后，在**置信度封顶之后**施加于候选 confidence
    （evaluate_candidate 的 s_loc_eff）——不可并入判据贡献分，否则会被封顶吸收；
    不改 support_count、不计入 D8。几何链无法判读胞内/游离，故纯几何 scene 恒 1.0。
    """
    if not scene:
        return 1.0
    vlm = scene.get("vlm") or {}
    loc = vlm.get("crystal_location")
    if loc not in ("intracellular", "free"):
        return 1.0
    conf = vlm.get("crystal_location_conf")
    if conf is None or float(conf) < SCENE_LOC_CONF_GATE:
        return 1.0
    morph_text = crit.get("morph") or ""
    has_intra = any(k in morph_text for k in SCENE_INTRA_KEYWORDS)
    has_free = any(k in morph_text for k in SCENE_FREE_KEYWORDS)
    if loc == "intracellular":
        if has_intra:
            return SCENE_MATCH_FACTOR
        if has_free:
            return SCENE_CONTRA_FACTOR
    else:  # free：游离散在
        if has_free:
            return SCENE_MATCH_FACTOR
        if has_intra:
            return SCENE_CONTRA_FACTOR
    return 1.0


def scene_fit(cand, scene, feature_ids=None):
    """场景证据 → (fit, reasons)。**平分排序专用**（丰度 + 排列两维）。

    fit ∈ [-2, +2]，两维各 ±1：
      * 丰度维：观测档位（scene.abundance_obs，几何或 VLM 来源）与候选判据声明
        （限 feature_ids 内检出的判据，缺省取全部）比较——观测 ≥ 声明 +1；
        低一档 0；低两档及以上 -1；
      * 排列维：观测到 rows 且候选 morph 含"排列成行/成行" → +1；观测到 rows
        且候选 morph 明确"散在"而无成行表述 → -1；其余 0。
    宿主语境（胞内/游离）已升级为评分乘数 _scene_location_factor（在封顶后作用于
    候选置信度），此处不再计入，避免同一证据双重计分。
    reasons 为人类可读依据列表，进报告/notes。VLM 判读优先（arrangement 条目的
    vlm_verdict 覆盖几何 verdict，见 scene_attributes.merge_vlm_scene）。
    """
    if not scene or cand is None:
        return 0, []
    fit, reasons = 0, []
    morphs = "；".join(filter(None, (c.get("morph") or ""
                                     for c in (cand.get("criteria") or []))))

    # 丰度维
    obs = scene.get("abundance_obs")
    obs_rank = ABUNDANCE_RANK.get(obs) if obs else None
    if obs_rank is not None:
        crits = (cand.get("criteria") or [])
        if feature_ids:
            scoped = [c for c in crits if c.get("feature") in feature_ids]
            crits = scoped or crits
        claimed = max((ABUNDANCE_RANK.get(c.get("abundance") or "", -1) for c in crits),
                      default=-1)
        if claimed >= 0:
            if obs_rank >= claimed:
                fit += 1
                reasons.append(f"丰度观测（{obs}）不低于判据声明")
            elif claimed - obs_rank == 1:
                pass
            else:
                fit -= 1
                reasons.append(f"丰度观测（{obs}）明显低于判据声明（检出密度不足以支撑）")

    # 排列维（VLM 判读优先，几何 verdict 兜底）
    arr = scene.get("arrangement") or []
    verdicts = [(a.get("vlm_verdict") or a.get("verdict")) for a in arr]
    if any(v == "rows" for v in verdicts):
        has_row_kw = any(k in morphs for k in SCENE_ROW_KEYWORDS)
        has_scatter_kw = any(k in morphs for k in SCENE_SCATTER_KEYWORDS)
        if has_row_kw:
            fit += 1
            reasons.append("观测到结晶排列成行，与判据 morph 表述相符")
        elif has_scatter_kw:
            fit -= 1
            reasons.append("观测到结晶排列成行，与判据『散在』表述不符")
    return fit, reasons


# ---------------------------------------------------------------- 裁决
def decide(kb, cards, candidates=None, scene=None, *, min_support_features=None):
    """主入口：证据卡池 → 最终裁决 dict。纯确定性。

    scene（可选）：scene_attributes.build_scene / merge_vlm_scene 产出的场景证据。
    两通道：宿主语境乘数在置信度封顶之后作用于 confidence（不改 support_count/D8/cap 结构）；
    丰度+排列仅在 (confidence, support_count) 完全并列时作排序键与 notes。
    """
    cands = candidates if candidates is not None else kb.candidates()
    ranking = [evaluate_candidate(c, cards, kb, scene) for c in cands]
    if scene:
        cand_map = {c["id"]: c for c in cands}
        for r in ranking:
            f, reasons = scene_fit(cand_map.get(r["id"]), scene, _support_feature_ids(r))
            r["scene_fit"] = f
            r["scene_reasons"] = reasons
    ranking.sort(key=lambda r: (r["confidence"], r["support_count"],
                                r.get("scene_fit", 0), r["score"]), reverse=True)
    top = ranking[0] if ranking else None
    second = ranking[1] if len(ranking) > 1 else None

    out = {
        "verdict": "insufficient_evidence",
        "conclusion": None,
        "alternatives": [],
        "ranking": ranking,
        "notes": [],
    }

    sup = _support_feature_ids(top) if top else []
    out["support_feature_count"] = len(sup)
    if top is None or not sup or top["confidence"] < BAND_PROBABLE:
        out["verdict"] = "insufficient_evidence"
        out["notes"].append("候选池为空或最高分候选无有效支持分（<%.2f）" % BAND_PROBABLE)
        return out

    margin = top["confidence"] - (second["confidence"] if second else 0.0)
    out["conclusion"] = {
        "id": top["id"], "kind": top["kind"], "display": top["display"],
        "confidence": top["confidence"],
    }
    out["alternatives"] = [
        {"id": r["id"], "display": r["display"], "confidence": r["confidence"]}
        for r in ranking[1:4]
    ]
    out["margin"] = round(margin, 4)

    # 排序键是 (confidence, support_count, score)，故"分差"不能只看置信度：
    # 置信度并列但独立证据类数不同时，top1 是被明确判出来的（如猪苓 2 类 vs 枳壳/陈皮 1 类），
    # 不该再按"窄分差"降级成 probable + 复核。
    decisive = margin >= MARGIN_DELTA or (
        second is not None and top["support_count"] != second["support_count"])
    if top["confidence"] >= BAND_CONFIRMED and decisive:
        out["verdict"] = "group_level" if top["kind"] == "group" else "confirmed"
    else:
        out["verdict"] = "group_level" if top["kind"] == "group" else "probable"
        if not decisive:
            out["notes"].append(
                f"top1({top['display']}) 与 top2({second['display']}) 分差 {margin:.2f} < {MARGIN_DELTA}"
                f" 且独立证据类数相同（各 {top['support_count']} 类），输出双候选，建议人工复核")
            out["needs_review"] = True
            # 场景证据排序提示（三层模型第 2/3 层）：平分带内任一候选 fit 非 0 时，
            # 完整列出平分带排序依据（含被 fit 降权降到带内的候选，如丰度不足的虎杖）；
            # 不改 verdict / conf / D8（铁律）。
            if scene:
                tied = [r for r in ranking
                        if r["confidence"] == top["confidence"]
                        and r["support_count"] == top["support_count"]][:4]
                if any(r.get("scene_fit", 0) != 0 for r in tied):
                    detail = "、".join(
                        f"{r['display']} fit {r.get('scene_fit', 0):+d}"
                        + (f"（{'；'.join(r.get('scene_reasons') or [])}）" if r.get("scene_reasons") else "")
                        for r in tied)
                    out["notes"].append(
                        f"场景证据排序：{detail}；场景证据不改变置信度与证据门槛，仍建议人工复核")

    # Keep fixed-B baseline intact; relaxed experiments are explicit per-call policy.
    threshold = MIN_SUPPORT_FEATURES if min_support_features is None else min_support_features
    if type(threshold) is not int or threshold not in (1, 2):
        raise ValueError('min_support_features must be 1 or 2')
    solo_release = (len(sup) == 1 and decisive and top["confidence"] >= BAND_CONFIRMED)
    if len(sup) < threshold and not solo_release:
        would = out["verdict"]
        out["verdict"] = "insufficient_evidence"
        out["conclusion"] = None
        out["alternatives"] = ([{"id": top["id"], "display": top["display"],
                                 "confidence": top["confidence"]}]
                               + out["alternatives"])[:4]
        out["needs_review"] = True
        out["notes"].append(
            f"D8 证据数量门槛：独立证据特征仅 {len(sup)} 类（{'、'.join(sup)}），"
            f"不足以支撑 {would} 结论（top1 {top['display']} conf {top['confidence']:.2f}）；"
            f"降级为证据不足、top1 保留为首个备选，补齐其他判据特征或经 VLM 补证据后重评")
    elif solo_release and len(sup) < threshold:
        out["notes"].append(
            f"D8b 单类证据条件放行：独立证据仅 1 类（{'、'.join(sup)}），但 top1 {top['display']} "
            f"conf {top['confidence']:.2f} ≥ {BAND_CONFIRMED} 且与 top2 已区分（分差 {margin:.2f}）；"
            f"按 v1.6 出结论，证据类数单薄故仍标人工复核")
        out["needs_review"] = True
    return out
