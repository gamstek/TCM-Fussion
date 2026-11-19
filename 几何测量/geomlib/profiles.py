# -*- coding: utf-8 -*-
"""profiles — 每特征测量剖面 + 药典参考尺寸范围。

per-feature 测量剖面的意义：85 味药典措辞里每个特征"关心的几何量"不同——
  簇晶关心"直径"，针晶关心"长"，纤维/导管关心"径（宽度）"，
  淀粉粒关心"直径+复粒分粒数"，石细胞关心"直径+形状（分枝状）"。
primary_dims 决定该特征对外报告的主判尺寸（primary_value_um），
也是 validate_ranges.py 对照药典范围时用的量纲。

REFERENCE_RANGES 依据《85味中药显微特征_修正版》（药典原文式术语表）录入，
仅收录药典给出量化尺寸的品种；无数值者（如"枳壳方晶成片"）不设范围，
validate_ranges 对其返回 no_reference 而非误判。
"""

PROFILES = {
    "cluster_crystal": {   # 草酸钙簇晶
        "zh": "草酸钙簇晶",
        "primary_dims": ["feret_max_um", "eq_diameter_um"],
        "shape_focus": ["solidity", "defect_count", "mean_defect_depth_ratio",
                        "circularity", "convexity"],
        "note": "直径=长径 Feret；solidity 低+凹角多 → 多瓣星状集合体；"
                "凹角深度比大 → 棱角偏钝（大黄判据的明场形态代理）",
    },
    "single_crystal": {    # 草酸钙方晶
        "zh": "草酸钙方晶",
        "primary_dims": ["feret_max_um", "eq_diameter_um"],
        "shape_focus": ["aspect_ratio", "rectangularity", "solidity"],
        "note": "长=feret_max，宽=feret_min；rectangularity 高 → 菱形/长方形",
    },
    "needle_crystal": {    # 草酸钙针晶（含柱晶、针晶束）
        "zh": "草酸钙针晶/柱晶",
        "primary_dims": ["feret_max_um"],
        "shape_focus": ["aspect_ratio", "mean_width_um", "n_components"],
        "note": "长=feret_max（束=整束 Feret，多连通域时 flags 带 multi_component）",
    },
    "vessel": {            # 导管
        "zh": "导管",
        "primary_dims": ["mean_width_um", "max_inscribed_dia_um"],
        "shape_focus": ["skeleton_len_um", "tortuosity", "aspect_ratio"],
        "note": "管径=沿骨架宽度中位数更稳；导管是连续单一结构，长度=骨架长",
    },
    "stone_cell": {        # 石细胞
        "zh": "石细胞",
        "primary_dims": ["feret_max_um", "eq_diameter_um"],
        "shape_focus": ["solidity", "aspect_ratio", "defect_count"],
        "note": "solidity 低+深凹角 → 分枝状石细胞（厚朴等）；壁厚/层纹需灰度信息，非纯几何",
    },
    "starch_grain": {      # 淀粉粒
        "zh": "淀粉粒",
        "primary_dims": ["feret_max_um", "eq_diameter_um"],
        "shape_focus": ["roundness", "dt_peak_count"],
        "note": "dt_peak_count≥2 提示复粒/聚粒（近似）；脐点类型需灰度信息，非纯几何",
    },
    "fiber": {             # 纤维/木纤维/晶纤维
        "zh": "纤维",
        "primary_dims": ["mean_width_um", "median_width_um"],
        "shape_focus": ["feret_max_um", "aspect_ratio", "tortuosity"],
        "note": "药典'直径14~25μm'类判据对照沿骨架宽度；长细比=feret_max/mean_width",
    },
    "crystal_fiber": {     # 晶鞘纤维
        "zh": "晶鞘纤维",
        "primary_dims": ["mean_width_um", "median_width_um"],
        "shape_focus": ["feret_max_um", "aspect_ratio"],
        "note": "几何上同纤维；'外方附方晶'需灰度/邻接信息确认",
    },
    "secretory_tissue": {  # 油室/油管/油细胞/乳汁管/分泌细胞（2026-10-09 主判量纲改 mean_width）
        "zh": "分泌组织",
        "primary_dims": ["mean_width_um", "feret_max_um", "eq_diameter_um"],
        "shape_focus": ["aspect_ratio", "tortuosity"],
        "note": "药典量化给的是『管径/直径』，故主判取沿骨架中位宽 mean_width（长管上 feret_max=管段长度"
                "会恒 OUT_OF_RANGE；类圆油细胞/分泌细胞 mean_width≈直径，实测圆盘仅差约 3.5%）；"
                "feret_max 退为后备（椭圆油细胞可回看长径）",
    },
    "pollen_grain": {
        "zh": "花粉粒",
        "primary_dims": ["feret_max_um", "eq_diameter_um"],
        "shape_focus": ["roundness", "circularity"],
    },
    "epidermis_cell": {    # 2026-10-09 L1 量化闸门：整片一框（非单个细胞）→ 恒不产 primary_value_um
        "zh": "表皮细胞（含种皮/外果皮）",
        "primary_dims": ["feret_max_um", "eq_diameter_um"],
        "shape_focus": ["aspect_ratio", "solidity"],
        "coarse_quant": True,   # 框粒度天然是"片"，尺寸不对应药典单细胞直径 → 只作存在性
    },
    "cork_cell": {
        "zh": "木栓细胞",
        "primary_dims": ["feret_max_um", "eq_diameter_um"],
        "shape_focus": ["aspect_ratio"],
    },
    "stoma": {             # 2026-10-09 L1 量化闸门：气孔是"整器/成片"观察对象，无量纲单值 → 只作存在性
        "zh": "气孔（含保卫细胞形态）",
        "primary_dims": ["feret_max_um", "eq_diameter_um"],
        "shape_focus": ["aspect_ratio", "circularity"],
        "coarse_quant": True,
    },
    "hypha": {
        "zh": "菌丝",
        "primary_dims": ["mean_width_um", "median_width_um"],
        "shape_focus": ["skeleton_len_um", "tortuosity"],
        "note": "菌丝径 3~8μm 为强判据；分枝多 → 骨架长而弯曲",
    },
    "silica_body": {
        "zh": "硅质块",
        "primary_dims": ["feret_max_um", "eq_diameter_um"],
        "shape_focus": ["roundness"],
    },
    "inulin": {
        "zh": "菊糖",
        "primary_dims": ["feret_max_um", "eq_diameter_um"],
        "shape_focus": ["solidity"],
    },
    "glandular_hair": {    # 腺毛/腺鳞（2026-10-09 扩类第 2 批补齐：此前 PROFILES 缺此项 → 退 generic）
        "zh": "腺毛/腺鳞",
        "primary_dims": ["feret_max_um", "eq_diameter_um"],   # 药典给的是"头部直径"
        "shape_focus": ["roundness", "solidity"],
        "note": "量头部直径；柄不计入主判尺寸。有无膨大腺头是与非腺毛的唯一分界（需灰度/形态，非纯几何）",
    },
    "generic": {           # 未知特征 / 通用回退
        "zh": "通用",
        "primary_dims": ["feret_max_um", "eq_diameter_um", "mean_width_um"],
        "shape_focus": [],
    },
}


def get_profile(feature_label):
    return PROFILES.get(feature_label, PROFILES["generic"])


# ---------------------------------------------------------------------------
# 药典参考尺寸范围（《85味中药显微特征_修正版》逐条录入）
# dim ∈ feret_max(直径/长) | mean_width(径/宽)
# extreme: 药典"个别可达/有的至"值，单独判 extreme 而非直接 out_of_range
# ---------------------------------------------------------------------------
REFERENCE_RANGES = [
    # --- 簇晶 ---
    dict(herb="大黄",   feature="cluster_crystal", dim="feret_max", umin=20,  umax=160, extreme=190,
         note="草酸钙簇晶直径20~160μm，有的至190μm"),
    dict(herb="白芍",   feature="cluster_crystal", dim="feret_max", umin=11,  umax=35,
         note="簇晶直径11~35μm，常排列成行"),
    dict(herb="虎杖",   feature="cluster_crystal", dim="feret_max", umin=30,  umax=100,
         note="簇晶极多，较大，直径30~100μm"),
    dict(herb="牡丹皮", feature="cluster_crystal", dim="feret_max", umin=None, umax=None,
         note="簇晶存在于薄壁细胞中（无量化范围）"),
    # --- 针晶/柱晶 ---
    dict(herb="半夏",   feature="needle_crystal", dim="feret_max", umin=20,  umax=144,
         note="针晶长20~144μm"),
    dict(herb="姜半夏", feature="needle_crystal", dim="feret_max", umin=20,  umax=144,
         note="针晶长20~144μm"),
    dict(herb="法半夏", feature="needle_crystal", dim="feret_max", umin=32,  umax=144,
         note="针晶长32~144μm"),
    dict(herb="白术",   feature="needle_crystal", dim="feret_max", umin=10,  umax=32,
         note="针晶细小，长10~32μm；少数直径至4μm"),
    dict(herb="苍术",   feature="needle_crystal", dim="feret_max", umin=5,   umax=30,
         note="针晶细小，长5~30μm"),
    dict(herb="知母",   feature="needle_crystal", dim="feret_max", umin=26,  umax=110,
         note="针晶成束或散在，长26~110μm"),
    dict(herb="射干",   feature="needle_crystal", dim="feret_max", umin=49,  umax=240, extreme=315,
         note="柱晶长49~240(315)μm，直径约至49μm"),
    dict(herb="射干",   feature="needle_crystal", dim="mean_width", umin=None, umax=49,
         note="柱晶直径约至49μm"),
    # --- 方晶 ---
    dict(herb="陈皮",   feature="single_crystal", dim="feret_max", umin=5,   umax=53,
         note="方晶直径3~34μm，长5~53μm（长径取 feret_max）"),
    dict(herb="猪苓",   feature="single_crystal", dim="feret_max", umin=3,   umax=60, extreme=68,
         note="方晶直径3~60μm，长至68μm（双锥八面体等）"),
    # --- 淀粉粒 ---
    dict(herb="大黄",   feature="starch_grain", dim="feret_max", umin=3,   umax=45,
         note="单粒直径3~45μm，脐点星状；复粒2~8分粒"),
    # --- 石细胞 ---
    dict(herb="山楂",   feature="stone_cell", dim="feret_max", umin=19,  umax=125,
         note="石细胞直径19~125μm"),
    dict(herb="肉桂",   feature="stone_cell", dim="feret_max", umin=32,  umax=88,
         note="石细胞直径32~88μm"),
    dict(herb="桂枝",   feature="stone_cell", dim="feret_max", umin=30,  umax=64,
         note="石细胞直径30~64μm"),
    dict(herb="玄参",   feature="stone_cell", dim="feret_max", umin=None, umax=94,
         note="石细胞直径约至94μm"),
    dict(herb="五味子", feature="stone_cell", dim="feret_max", umin=18,  umax=83,
         note="种皮表皮石细胞18~50μm、内层石细胞约至83μm（两类合并记录）"),
    # --- 纤维 ---
    dict(herb="板蓝根", feature="fiber", dim="mean_width", umin=14,  umax=25,
         note="木纤维直径14~25μm"),
    dict(herb="木香",   feature="fiber", dim="mean_width", umin=16,  umax=24,
         note="木纤维直径16~24μm"),
    dict(herb="黄柏",   feature="fiber", dim="mean_width", umin=16,  umax=38,
         note="纤维直径16~38μm，常成束（晶纤维）"),
    dict(herb="川楝子", feature="fiber", dim="mean_width", umin=9,   umax=36,
         note="果皮纤维直径9~36μm（晶纤维）"),
    dict(herb="桑白皮", feature="fiber", dim="mean_width", umin=13,  umax=26,
         note="纤维直径13~26μm"),
    dict(herb="黄芪",   feature="fiber", dim="mean_width", umin=8,   umax=30,
         note="纤维直径8~30μm"),
    dict(herb="大腹皮", feature="fiber", dim="mean_width", umin=8,   umax=15,
         note="中果皮纤维直径8~15μm"),
    # --- 导管（大黄/常山药典文未给量化管径，不设范围） ---
    # --- 分泌组织 ---（2026-10-09：dim 由 feret_max 改 mean_width，与卡片主判量纲对齐；药典给的是管径/直径，
    #     长管上 feret_max=管段长度会恒 OUT_OF_RANGE。类圆油细胞/分泌细胞 mean_width≈直径，区间维持不变）
    dict(herb="柴胡",   feature="secretory_tissue", dim="mean_width", umin=8,  umax=25,
         note="油管直径8~25μm（管径）"),
    dict(herb="防风",   feature="secretory_tissue", dim="mean_width", umin=17, umax=60,
         note="油管直径17~60μm（管径）"),
    dict(herb="桔梗",   feature="secretory_tissue", dim="mean_width", umin=14, umax=25,
         note="联结乳管直径14~25μm（管径）"),
    dict(herb="鱼腥草", feature="secretory_tissue", dim="mean_width", umin=28, umax=104,
         note="油细胞直径28~104μm（类圆油细胞 mean_width≈直径）"),
    dict(herb="香附",   feature="secretory_tissue", dim="mean_width", umin=35, umax=72,
         note="分泌细胞直径35~72μm（类圆分泌细胞 mean_width≈直径）"),
    dict(herb="薄荷",   feature="glandular_hair", dim="feret_max", umin=None, umax=90,
         note="腺鳞头部直径约至90μm（2026-10-09 扩类：由 secretory_tissue 迁到 glandular_hair，与 KB 判据对齐）"),
    # --- 表皮细胞 ---
    dict(herb="葶苈子",   feature="epidermis_cell", dim="feret_max", umin=15, umax=42,
         note="种皮内表皮细胞直径15~42μm"),
    dict(herb="王不留行", feature="epidermis_cell", dim="feret_max", umin=50, umax=120,
         note="种皮表皮细胞直径50~120μm"),
    # --- 菌丝/硅质块 ---
    dict(herb="茯苓",   feature="hypha", dim="mean_width", umin=4,  umax=6,
         note="菌丝直径4~6μm"),
    dict(herb=None,     feature="hypha", dim="mean_width", umin=3,  umax=8,
         note="通用：菌丝直径3~8μm（特征全集）"),
    dict(herb="大腹皮", feature="silica_body", dim="feret_max", umin=4, umax=12,
         note="圆簇状硅质块直径约8μm"),
    # --- 花粉粒 ---
    dict(herb="菊花",   feature="pollen_grain", dim="feret_max", umin=32, umax=37,
         note="花粉粒直径32~37μm"),
    dict(herb="红花",   feature="pollen_grain", dim="feret_max", umin=None, umax=60,
         note="花粉粒直径约至60μm"),
    # --- 非腺毛（taxonomy P1 扩展键 non_glandular_hair） ---
    dict(herb="广藿香", feature="non_glandular_hair", dim="feret_max", umin=None, umax=590,
         note="非腺毛长约至590μm"),
    dict(herb="淫羊藿", feature="non_glandular_hair", dim="feret_max", umin=200, umax=1000,
         note="非腺毛长200~1000μm，多碎断"),
    dict(herb="茵陈",   feature="non_glandular_hair", dim="feret_max", umin=600, umax=1700,
         note="非腺毛T字形，长600~1700μm"),
    dict(herb="白头翁", feature="non_glandular_hair", dim="mean_width", umin=13, umax=33,
         note="非腺毛单细胞直径13~33μm"),
]

# 2026-09-22：原 EXCLUSION_GEOMETRIC（大黄"簇晶<45μm 存疑"，源自 feature_taxonomy.json
# exclusion_rules）已删除——该规则唯一出处是示例文档《一中药显微鉴别标准示例大黄粉末显微特征.docx》，
# 非药典依据；其"尺寸出界"语义已由 REFERENCE_RANGES 的 out_of_range 核查覆盖。
