# -*- coding: utf-8 -*-
"""contracts.py — 五个数据契约的唯一事实源与统一装载器（P2-1；2026-09-29 C1a 增第 ⑤ 段）。

**数据链**（唯一一条，见 重构方案.md §二）：
    crops(+px_per_um) → mask + `<stem>__measure_manifest.csv` → measurements.jsonl → verdicts

第 ⑤ 段（prelabel）**不在这条链上**，而是链的更上游语料：它既是"喂人工标注的预标注交付件"，
也是"全项目 px_per_um 的原始生产记录"。之所以值得成契约，是因为它有 10 个模块在读，
是仓内被读得最广的自由格式（详见文件末 ⑤ 段说明）。

**为什么要它**：此前这四段契约靠"随行 json 裸 `json.load()` 取字典 + 口口相传"传递 ——
字段拼错不报错（只会静默拿到 `None`），缺失检查散落在各自模块（`px_per_um` 的强校验全项目只有
`measure.py` 与 `igm_segment_batch.check_items` 两处）。本模块把四段契约写成 dataclass，
**校验只在这里发生**；每条记录同时保留 `raw`，未知/新增字段原样透传，字段演化不必同步改契约。

**px_per_um 的三条口径**（数字、字符串、缺失）在历史产物里同时存在，故：
  - `px_per_um`      → float 或 None（供判空/计算）
  - `px_per_um_text` → 原样字符串（清单 CSV 必须原样带出，`igm_segment_batch` 自检断言 "10.0"）
  - `require_px_per_um()` → 缺失即抛 ContractError，绝不静默补默认值

各契约**不得**在此做业务判断（不裁剪、不换算、不改写），只负责装载与校验。
"""
import ast
import csv
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

__all__ = [
    "ContractError", "CropMeta", "MeasureManifestRow", "MeasurementRecord", "Verdict",
    "load_crop_metas", "crop_sidecar", "check_crops", "load_manifest_rows", "iter_measurement_records",
    "load_verdict", "MANIFEST_SUFFIX", "MANIFEST_COLUMNS",
    "coerce_measurement_row", "MEASUREMENT_TEXT_COLUMNS", "parse_flags",
    "GRANULARITY_LEVELS", "GRANULARITY_DEFAULT",
    # ⑤ prelabel（2026-09-29 C1a）
    "DetBox", "ScaleReading", "PrelabelRecord", "load_prelabel", "check_prelabel",
    "PRELABEL_REQUIRED",
]


class ContractError(ValueError):
    """契约违约（字段缺失/类型不可用）。消息面向"一次修完"，含全部违约项。"""


QUALITY_USES = ('existence', 'morphology', 'length', 'width', 'area')


def quality_assessment(obj):
    """跨阶段用途资格契约；完整性校验不等于授权，调用方须验证受信生成及工件绑定。"""
    import math
    if not isinstance(obj, dict) or obj.get('schema') != 'quality-assessment/v1':
        raise ContractError('质量评估schema非法')
    required = {'bindings','findings','uses','policy_version','evaluator_version','integrity'}
    if not required <= set(obj) or type(obj['integrity']) is not bool:
        raise ContractError('质量评估缺字段')
    if not isinstance(obj['bindings'],dict) or not obj['bindings']:
        raise ContractError('质量评估必须绑定工件')
    if not {'image','crop','crop_meta','mask'}<=set(obj['bindings']):
        raise ContractError('质量评估缺原图/crop/侧车/mask绑定')
    for key,value in obj['bindings'].items():
        if not isinstance(key,str) or not isinstance(value,str) or len(value)!=64 or any(c not in '0123456789abcdef' for c in value):
            raise ContractError('工件绑定必须为SHA256')
    if not isinstance(obj['uses'],dict) or set(obj['uses'])!=set(QUALITY_USES):
        raise ContractError('用途资格键不完整')
    for use in obj['uses'].values():
        if not isinstance(use,dict) or type(use.get('allowed')) is not bool or not isinstance(use.get('reasons'),list):
            raise ContractError('用途资格类型非法')
        if not obj['integrity'] and use['allowed']:raise ContractError('完整性失败不得授予资格')
    if not isinstance(obj['findings'],list) or any(not isinstance(f,dict) or not isinstance(f.get('code'),str) for f in obj['findings']):
        raise ContractError('质量原因非法')
    if not all(isinstance(obj[k],str) and obj[k] for k in ('policy_version','evaluator_version')):
        raise ContractError('质量版本非法')
    def finite(v):
        if isinstance(v,float) and not math.isfinite(v):raise ContractError('质量评估含非有限数')
        if isinstance(v,dict):
            for x in v.values():finite(x)
        if isinstance(v,list):
            for x in v:finite(x)
    finite(obj)
    return obj


def _pos_float(v) -> Optional[float]:
    """正浮点或 None（沿用历史口径：<=0 与不可解析一律视为无效）。"""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if f > 0 else None


def _json_list(v) -> Optional[List]:
    """CSV 里以 JSON 文本存的列表（如 tight_bbox "[31, 31, 639, 714]"）；已是列表则原样。"""
    if v is None or v == "":
        return None
    if isinstance(v, list):
        return v
    try:
        out = json.loads(v)
    except (TypeError, ValueError):
        return None
    return out if isinstance(out, list) else None


# ──────────────────────────────────────────────────────────────
# 量化可用性闸门（Phase 2，2026-10-09）：粗粒度实例（束/片/团）不产 primary_value_um
# ──────────────────────────────────────────────────────────────
# 为什么需要：有的特征在图上**有时是单实例、有时是束/片/团**（一根纤维 vs 一束纤维；整片表皮细胞）。
# 框粒度 ≠ 药典单值量纲（束宽被当单根直径、整片尺寸被当单个细胞直径），几何一律产 primary_value_um
# 就会污染定量核验。这里把"粒度"作为**逐实例标注**随交付链下传，几何遇 `granularity != instance`
# 打 `coarse_quant` 旗标 → 判药按"只作存在性"处理（与 igm_review 同构，区别是不丢弃）。
# 默认 `instance`：无 stage-2（单阶段检测）时行为与改动前**逐字一致**。
GRANULARITY_LEVELS = ("instance", "bundle", "sheet", "clump")
GRANULARITY_DEFAULT = "instance"
FEATURE_WINDOW_SCHEMA = "feature-window/v1"


# ──────────────────────────────────────────────────────────────
# ① 检测 → 分割：crop 随行 json（`<crop_id>.json`，单图自包含契约）
# ──────────────────────────────────────────────────────────────
@dataclass
class CropMeta:
    """crop 小图元数据。写方：detection/cli/detect_sample.py；读方：segmentation/cli/igm_segment_batch.py。

    `png` 由 json 的 png 字段与 crop_id 共同决定（json 未写则取同名 png），故由装载器解析后传入。
    """
    crop_id: str
    json_path: str
    png: str = ""
    source_image_path: str = ""
    stem: str = ""
    feature_id: str = ""
    object_label: str = ""
    herb: str = ""
    px_per_um: Optional[float] = None
    px_per_um_text: str = ""
    tight_box: Optional[List] = None
    tight_box_local: Optional[List] = None
    loose_box: Optional[List] = None
    crop_size: str = ""
    pad_info: str = ""
    granularity: str = GRANULARITY_DEFAULT   # 逐个实例的框粒度（instance/bundle/sheet/clump）
    raw: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_json(cls, json_path: str, crops_dir: str = "") -> "CropMeta":
        with open(json_path, encoding="utf-8") as f:
            cj = json.load(f)
        crop_id = os.path.splitext(os.path.basename(json_path))[0]
        base = crops_dir or os.path.dirname(json_path)
        png = os.path.join(base, cj.get("png") or crop_id + ".png")
        src = cj.get("source_image_path") or ""
        return cls(
            crop_id=crop_id, json_path=json_path, png=png,
            source_image_path=src,
            stem=str(cj.get("stem") or "").strip() or os.path.splitext(os.path.basename(str(src)))[0],
            feature_id=str(cj.get("feature_id") or "").strip(),
            object_label=str(cj.get("object_label") or "").strip(),
            herb=str(cj.get("herb") or "").strip(),
            px_per_um=_pos_float(cj.get("px_per_um")),
            px_per_um_text="" if cj.get("px_per_um") is None else str(cj.get("px_per_um")).strip(),
            tight_box=_json_list(cj.get("tight_box")),
            tight_box_local=_json_list(cj.get("tight_box_local")),
            loose_box=_json_list(cj.get("loose_box")),
            crop_size=str(cj.get("crop_size") or ""),
            pad_info=str(cj.get("pad_info") or ""),
            granularity=str(cj.get("granularity") or "").strip().lower() or GRANULARITY_DEFAULT,
            raw=cj,
        )

    def require_px_per_um(self) -> float:
        if self.px_per_um is None:
            raise ContractError(
                "crop 随行 json 没有有效 px_per_um（几何标定唯一来源，缺失不降级）：{}".format(self.json_path))
        return self.px_per_um


def load_crop_metas(crops_dir: str) -> List[CropMeta]:
    """读目录下全部 `<crop_id>.json` → CropMeta 列表（按 crop_id 升序）。

    不做目录级猜测：json 指向的子图不存在即报错（原 load_crops 行为）。
    缺 feature_id / stem / px_per_um 的聚合校验由 `check_crops()` 负责（一次列全）。
    """
    out = []
    for fn in sorted(os.listdir(crops_dir)):
        if not fn.lower().endswith(".json"):
            continue
        meta = CropMeta.from_json(os.path.join(crops_dir, fn), crops_dir=crops_dir)
        if not os.path.exists(meta.png):
            raise ContractError("[FAIL] {}.json 指向的子图不存在: {}".format(meta.crop_id, meta.png))
        out.append(meta)
    return out


def crop_sidecar(crop_id, png, source_image_path, stem, feature_id, object_label, herb,
                 px_per_um, tight_box, tight_box_local, loose_box, crop_size,
                 pad_info="", granularity=GRANULARITY_DEFAULT) -> Dict[str, Any]:
    """构造 crop 随行 json（**写侧唯一实现**，键序即 `CropMeta.from_json` 读取的那 14 项）。

    这是「检测 → 分割」的输入契约：`<crop_id>.json` + `<crop_id>.png` 成对，缺 pair 则
    `igm_segment_batch` 直接报错（不静默降级）。写方原只有 UI 链 `ui/seg_stage.build_crops`，
    2026-09-29 起 `detection/cli/detect_sample.py`（VLM 检测产物）也走这里，故把键表收到契约层。
    `granularity` 默认 `instance`（无 stage-2 时沿用旧行为）；两阶段检测由 stage 2 逐 crop 给出。
    """
    return {
        "crop_id": crop_id, "png": png,
        "source_image_path": source_image_path, "stem": stem,
        "feature_id": feature_id, "object_label": object_label, "herb": herb,
        "px_per_um": px_per_um,
        "tight_box": tight_box, "tight_box_local": tight_box_local,
        "loose_box": loose_box, "crop_size": crop_size, "pad_info": pad_info,
        "granularity": granularity,
    }


def effective_granularity(crop_granularity, classification_granularity=None):
    """原框及身份复核都支持单实例时才可作单值量测。旧记录缺复核粒度时沿用原框。"""
    if crop_granularity and crop_granularity != GRANULARITY_DEFAULT:
        return crop_granularity
    return classification_granularity or GRANULARITY_DEFAULT


def check_crops(metas: Iterable[CropMeta]) -> None:
    """入口聚合校验：feature_id / stem / px_per_um 缺一不可，**一次列全**（便于批量补）。"""
    metas = list(metas)
    groups = [
        ("没有 feature_id（IGM 靠它取特征描述）",
         [m.crop_id for m in metas if not m.feature_id]),
        ("推不出源图 stem（随行 json 缺 stem 且 source_image_path 为空）",
         [m.crop_id for m in metas if not m.stem]),
        ("没有有效 px_per_um（几何标定唯一来源，缺失不降级）",
         [m.crop_id for m in metas if m.px_per_um is None]),
    ]
    msgs = ["[FAIL] 以下 crop 的随行 json {}:\n  {}".format(desc, "\n  ".join(ids))
            for desc, ids in groups if ids]
    if msgs:
        raise ContractError("\n".join(msgs))


# ──────────────────────────────────────────────────────────────
# ② 分割 → 几何：`<stem>__measure_manifest.csv` 交付清单
# ──────────────────────────────────────────────────────────────
MANIFEST_SUFFIX = "__measure_manifest.csv"
MANIFEST_COLUMNS = ["instance_id", "inst_mask", "crop_id", "source_image_path", "feature_id",
                    "object_label", "herb", "tight_bbox", "px_per_um",
                    "igm_status", "igm_area_ratio", "granularity",
                    "feature_window_schema", "igm_associated_clipped_px"]


@dataclass
class MeasureManifestRow:
    """交付清单一行。写方：segmentation/cli/igm_segment_batch.py；读方：几何测量 measure.py。"""
    instance_id: Optional[int] = None
    inst_mask: str = ""
    crop_id: str = ""
    source_image_path: str = ""
    feature_id: str = ""
    object_label: str = ""
    herb: str = ""
    tight_bbox: Optional[List] = None
    px_per_um: Optional[float] = None
    px_per_um_text: str = ""
    igm_status: str = ""
    igm_area_ratio: Optional[float] = None
    granularity: str = GRANULARITY_DEFAULT   # 逐实例框粒度（缺列/空 → instance，旧清单兼容）
    raw: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_row(cls, row: Dict[str, Any]) -> "MeasureManifestRow":
        try:
            iid = int(str(row.get("instance_id")).strip())
        except (TypeError, ValueError):
            iid = None
        return cls(
            instance_id=iid,
            inst_mask=str(row.get("inst_mask") or "").strip(),
            crop_id=str(row.get("crop_id") or "").strip(),
            source_image_path=str(row.get("source_image_path") or "").strip(),
            feature_id=str(row.get("feature_id") or "").strip(),
            object_label=str(row.get("object_label") or "").strip(),
            herb=str(row.get("herb") or "").strip(),
            tight_bbox=_json_list(row.get("tight_bbox")),
            px_per_um=_pos_float(row.get("px_per_um")),
            px_per_um_text="" if row.get("px_per_um") is None else str(row.get("px_per_um")).strip(),
            igm_status=str(row.get("igm_status") or "").strip(),
            igm_area_ratio=_pos_float(row.get("igm_area_ratio")),
            granularity=str(row.get("granularity") or "").strip().lower() or GRANULARITY_DEFAULT,
            raw=dict(row),
        )

    def require_px_per_um(self) -> float:
        if self.px_per_um is None:
            raise ContractError(
                "交付清单行缺 px_per_um（几何只认清单实测值，缺失不降级）："
                "inst_mask={} source_image_path={}".format(self.inst_mask, self.source_image_path))
        return self.px_per_um


def load_manifest_rows(masks_dir: str, *, by_mask=False) -> Dict[str, MeasureManifestRow]:
    """加载分割交付清单，可按实际 mask 文件名精确索引。

    默认保留原图 stem 首行的历史口径，供原图级 mask 或旧清单使用。
    """
    out: Dict[str, MeasureManifestRow] = {}
    if not masks_dir or not os.path.isdir(masks_dir):
        return out
    for fn in sorted(os.listdir(masks_dir)):
        if not fn.lower().endswith("measure_manifest.csv"):
            continue
        with open(os.path.join(masks_dir, fn), encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                mask_name = (row.get("inst_mask") or "").strip()
                key = mask_name if by_mask else mask_name.split("__")[0] or fn.split("__")[0]
                if by_mask and key in out and out[key].raw != row:
                    raise ContractError(f"mask 对应多条不一致的交付清单记录：{key}")
                if key:
                    out.setdefault(key, MeasureManifestRow.from_row(row))
    return out


def parse_flags(fl) -> List[str]:
    """flags 三种历史载体 → 字符串列表（缺一即会丢历史产物里的质量标记）：

      ① JSONL 原生数组                        ["uncalibrated"]
      ② CSV 分号连接串（现行 CSV 契约）        "uncalibrated;multi_component"
      ③ CSV 内 Python list repr（更早的写法）  "['uncalibrated', 'multi_component']"
    """
    if isinstance(fl, list):
        return [str(x) for x in fl]
    if not isinstance(fl, str):
        return []
    s = fl.strip()
    if not s:
        return []
    if s.startswith("["):
        try:
            v = ast.literal_eval(s)
        except (ValueError, SyntaxError):
            return [t.strip("[]' ") for t in s.split(";") if t.strip("[]' ")]
        return [str(x) for x in v] if isinstance(v, (list, tuple)) else [s]
    return [t for t in (p.strip() for p in s.split(";")) if t]


# ──────────────────────────────────────────────────────────────
# ③ 几何 → 判药：measurements.jsonl / .csv
# ──────────────────────────────────────────────────────────────
@dataclass
class MeasurementRecord:
    """单实例测量记录（46 键，尺寸字段按需取 raw）。写方：几何测量 measure.py；
    读方：Agent判药 scripts/evidence_pool.py。

    `flags` 两种落盘形态：jsonl 原生数组、csv 分号连接串；本类统一成 list，并保留 `flags_text`。
    """
    image_stem: str = ""
    instance_id: Optional[int] = None
    feature: str = ""
    herb: Optional[str] = None
    calibration_px_per_um: Optional[float] = None
    calibration_status: str = ""
    flags: List[str] = field(default_factory=list)
    raw: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_obj(cls, obj: Dict[str, Any]) -> "MeasurementRecord":
        try:
            iid = int(str(obj.get("instance_id")).strip())
        except (TypeError, ValueError):
            iid = None
        return cls(
            image_stem=str(obj.get("image_stem") or "").strip(),
            instance_id=iid,
            feature=str(obj.get("feature") or "").strip(),
            herb=obj.get("herb"),
            calibration_px_per_um=_pos_float(obj.get("calibration_px_per_um")),
            calibration_status=str(obj.get("calibration_status") or "").strip(),
            flags=parse_flags(obj.get("flags")),
            raw=dict(obj),
        )

    @property
    def flags_text(self) -> str:
        return ";".join(self.flags)

    @property
    def key(self) -> Tuple[str, Optional[int]]:
        """主键：交付清单与测量记录都按 (stem, instance_id) 取行（2026-09-29 修串号缺陷的键）。"""
        return (self.image_stem, self.instance_id)

    def require(self, name: str):
        if self.raw.get(name) in (None, ""):
            raise ContractError(
                "测量记录缺必需字段 {}（{}(inst{})）".format(name, self.image_stem, self.instance_id))
        return self.raw[name]


def iter_measurement_records(path: str) -> Iterator[MeasurementRecord]:
    """逐行读 measurements（.jsonl 逐行 JSON；.csv 经 `coerce_measurement_row` 数值化）。"""
    if str(path).lower().endswith(".csv"):
        with open(path, encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                yield MeasurementRecord.from_obj(coerce_measurement_row(row))
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield MeasurementRecord.from_obj(json.loads(line))


# 文本列白名单：这些列即便内容全是数字也必须保持字符串
# （典型坑：image_stem 为 "1"、"2"，数值化后与 str 比较/拼接全部失配）。
MEASUREMENT_TEXT_COLUMNS = (
    "image_stem", "feature", "herb", "calibration_source", "calibration_status",
    "primary_dimension", "group_key", "split",
)


def coerce_measurement_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """measurements CSV 单元格 → 原类型（JSONL 读路不需要，本就带类型）。

    口径（与 JSONL 读路必须一致，否则按数值比较的判据在两种输入下行为不同）：
      空串 → None；文本列 → 原样；flags → 原样字符串（由 MeasurementRecord 归一成 list）；
      其余 → int / float。
    """
    out: Dict[str, Any] = {}
    for k, v in row.items():
        if v is None or v == "":
            out[k] = None
            continue
        if k in MEASUREMENT_TEXT_COLUMNS or k == "flags":
            out[k] = v
            continue
        try:
            out[k] = float(v) if ("." in v or "e" in v.lower()) else int(float(v))
        except (TypeError, ValueError):
            out[k] = v
    return out


# ──────────────────────────────────────────────────────────────
# ④ 判药 → 报告：verdict.json
# ──────────────────────────────────────────────────────────────
@dataclass
class Verdict:
    """单样本判决。写方：Agent判药 scripts/judge.py；读方：报告/审计/演示。"""
    sample_id: str = ""
    verdict: str = ""
    verdict_zh: str = ""
    conclusion: Dict[str, Any] = field(default_factory=dict)
    margin: Optional[float] = None
    support_feature_count: Optional[int] = None
    needs_review: Any = None
    alternatives: Optional[List] = None
    notes: Optional[List] = None
    schema: str = ""
    generated_at: str = ""
    raw: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_obj(cls, obj: Dict[str, Any]) -> "Verdict":
        if not obj.get("verdict"):
            raise ContractError("verdict 记录缺 `verdict` 字段（样本 {}）".format(obj.get("sample_id")))
        return cls(
            sample_id=str(obj.get("sample_id") or ""),
            verdict=str(obj.get("verdict") or ""),
            verdict_zh=str(obj.get("verdict_zh") or ""),
            conclusion=obj.get("conclusion") or {},
            margin=obj.get("margin"),
            support_feature_count=obj.get("support_feature_count"),
            needs_review=obj.get("needs_review"),
            alternatives=obj.get("alternatives"),
            notes=obj.get("notes"),
            schema=str(obj.get("schema") or ""),
            generated_at=str(obj.get("generated_at") or ""),
            raw=dict(obj),
        )

    @property
    def conclusion_display(self) -> str:
        """结论一句话（报告标题与 CLI 摘要共用）。"""
        return str((self.conclusion or {}).get("display") or "")

    @property
    def is_confirmed(self) -> bool:
        return self.verdict == "confirmed"


def load_verdict(path: str) -> Verdict:
    with open(path, encoding="utf-8") as f:
        return Verdict.from_obj(json.load(f))


# ──────────────────────────────────────────────────────────────
# ⑤ 检测：prelabel 记录（`<stem>.json`，一图一份；语料级，非数据链环节）
# ──────────────────────────────────────────────────────────────
#
# 为什么值得成契约（2026-09-29 C1a）：它是仓内**被读得最广**的自由格式 —— `ls_common`（4 个函数）、
# `det_eval`、`import_to_labelstudio`、`render_prelabels`、`refresh_task_data`、`sync_criteria`、
# `sync_prelabels_to_ls`、`build_tif_proxies`、`vlm_prelabel` 自身，各自 `json.load` 后按记忆取字段；
# 而这份语料**跨 v7.1~v7.4 多代口径**（`scale.grade` / `image_path_legacy` / `image_status` 都是后加的）
# → 字段缺失只会静默拿到 `None`。
#
# 本批只收口最值钱的两层：**框层**（`DetBox`，writer 与 reader 共用一份字段名）与
# **标定层**（`ScaleReading` —— 全项目 px_per_um 的原始来源，取值必须单点）。
# 顶层 `PrelabelRecord` + `check_prelabel` 提供"一次列全缺口"，但**不抛错**（宽容装载 + `raw` 透传），
# 因为存量里确实存在旧代记录，抛错会淹没真问题。

PRELABEL_REQUIRED = ("image_path", "prompt_version", "original_size", "objects")


@dataclass
class DetBox:
    """VLM 检测框（`prelabel.objects[i]`）。writer 与 reader 共用这一份字段名。

    `tight_bbox` = **原图像素** [x1,y1,x2,y2]（0~1000 归一化坐标已由
    `detect_core.detect_once` 映射完毕；原始 `bbox_2d` 只在 `raw` 里）。
    """
    label: str = ""
    tight_bbox: Optional[List[float]] = None
    confidence: str = "medium"
    score: Optional[float] = None
    morphology: str = ""
    note: str = ""
    raw: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_det(cls, det: Dict[str, Any], raw: Optional[Dict[str, Any]] = None) -> "DetBox":
        """**写侧**：由 `detect_core.detect_once` 的框 + 其对应的原始 object 构造
        （`note` 只在原始 object 里，故必须一起传）。"""
        raw = raw if isinstance(raw, dict) else {}
        return cls(label=det.get("label", ""), tight_bbox=det.get("tight_bbox"),
                   confidence=det.get("confidence", "medium"), score=det.get("score"),
                   morphology=det.get("morphology", ""), note=raw.get("note", ""),
                   raw=dict(raw))

    @classmethod
    def from_dict(cls, o: Optional[Dict[str, Any]]) -> "DetBox":
        """**读侧**：由 `prelabel.objects[i]`（或 crops.json 的框）构造。只搬字段，不校验。"""
        o = o if isinstance(o, dict) else {}
        return cls(label=o.get("label", ""), tight_bbox=o.get("tight_bbox"),
                   confidence=o.get("confidence", "medium"), score=o.get("score"),
                   morphology=o.get("morphology", ""), note=o.get("note", ""), raw=dict(o))

    def to_dict(self) -> Dict[str, Any]:
        """键序固定 label/tight_bbox/confidence/score/morphology/note（writer 沿此序落盘）。"""
        return {"label": self.label, "tight_bbox": self.tight_bbox,
                "confidence": self.confidence, "score": self.score,
                "morphology": self.morphology, "note": self.note}


@dataclass
class ScaleReading:
    """比例尺双引擎标定结果（`prelabel.scale`）—— 全项目 px_per_um 的**原始生产记录**。

    2026-09-20 起取代离线 `scale_table.csv`：检测阶段现测（A 本地形态学+OCR / B VLM
    交叉验证），结果随 prelabel JSON 流转。`status` 里的 grade 编码与"可信与否"判定见
    `ls_common.is_trusted_status()`（OK / OK-consensus 可信；OK-medium/OK-low/manual 不可信）。

    `px_per_um` **原样保留**（历史产物里 `"11.7"` 与 `11.7` 并存，下游有按字符串取值的口径）；
    需要数值时用 `px_per_um_value()`（<=0 与不可解析一律 None）。
    """
    px_per_um: Any = ""
    status: str = ""
    grade: str = ""
    source: str = ""
    bar_px: Any = None
    bar_um: Any = None
    note: str = ""
    raw: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_info(cls, info: Dict[str, Any]) -> "ScaleReading":
        """**写侧**：由 `scalebar_duo.measure_img_px_per_um` 的 info 构造
        （引擎字段名 `bar_px_cv` → 契约字段名 `bar_px`）。**刻意不做 `or ""` 兜底**，
        以免改动写盘字节。"""
        return cls(px_per_um=info["px_per_um"], status=info["status"],
                   grade=info.get("grade", ""), source=info["source"],
                   bar_px=info["bar_px_cv"], bar_um=info["bar_um"], note=info["note"],
                   raw=dict(info))

    @classmethod
    def from_dict(cls, sc: Optional[Dict[str, Any]]) -> "ScaleReading":
        """**读侧**：宽容 —— 早期 JSON 无 `scale` 字段（或为 null）→ 全空记录，不报错。"""
        sc = sc if isinstance(sc, dict) else {}
        return cls(px_per_um=sc.get("px_per_um") or "", status=sc.get("status") or "",
                   grade=sc.get("grade") or "", source=sc.get("source") or "",
                   bar_px=sc.get("bar_px"), bar_um=sc.get("bar_um"),
                   note=sc.get("note") or "", raw=dict(sc))

    @property
    def px_per_um_value(self) -> Optional[float]:
        return _pos_float(self.px_per_um)

    def to_dict(self) -> Dict[str, Any]:
        return {"px_per_um": self.px_per_um, "status": self.status, "grade": self.grade,
                "source": self.source, "bar_px": self.bar_px, "bar_um": self.bar_um,
                "note": self.note}


@dataclass
class PrelabelRecord:
    """一图一份的 VLM 预标注记录（`results/detection/prelabels/<stem>.json`）。"""
    image_path: str = ""
    prompt_version: str = ""
    original_size: Optional[List[int]] = None
    inference_size: Optional[List[int]] = None
    model: str = ""
    feature_label_folder: str = ""
    herb: str = ""
    group_key: str = ""
    needs_review: str = ""
    dir_prior: Optional[Dict[str, Any]] = None
    objects: List[DetBox] = field(default_factory=list)
    dropped_objects: List[Dict[str, Any]] = field(default_factory=list)
    raw_response_excerpt: str = ""
    scale: ScaleReading = field(default_factory=ScaleReading)
    raw: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, pre: Optional[Dict[str, Any]]) -> "PrelabelRecord":
        pre = pre if isinstance(pre, dict) else {}
        return cls(
            image_path=pre.get("image_path") or "",
            prompt_version=pre.get("prompt_version") or "",
            original_size=pre.get("original_size"),
            inference_size=pre.get("inference_size"),
            model=pre.get("model") or "",
            feature_label_folder=pre.get("feature_label_folder") or "",
            herb=pre.get("herb") or "",
            group_key=pre.get("group_key") or "",
            needs_review=pre.get("needs_review") or "",
            dir_prior=pre.get("dir_prior"),
            objects=[DetBox.from_dict(o) for o in (pre.get("objects") or [])],
            dropped_objects=list(pre.get("dropped_objects") or []),
            raw_response_excerpt=pre.get("raw_response_excerpt") or "",
            scale=ScaleReading.from_dict(pre.get("scale")),
            raw=dict(pre))


def load_prelabel(path: str) -> PrelabelRecord:
    with open(path, encoding="utf-8") as f:
        return PrelabelRecord.from_dict(json.load(f))


def check_prelabel(pre) -> List[str]:
    """**一次列全缺口**（不抛）：返回违约项列表，空列表 = 齐。

    接受原始 dict 或 `PrelabelRecord`。只查两件事：① 装载必需字段（存在且非空，
    `objects` 允许为空列表 —— v7.4 明确允许"确无单颗可输出空 objects"）；
    ② 每个框的 label 非空、`tight_bbox` 是 4 元且不退化为零/负宽高。
    **刻意不报**旧代差异（如无 `scale`、无 `image_path_legacy`）—— 存量里确实存在，
    报出来会淹没真问题。
    """
    rec = pre if isinstance(pre, PrelabelRecord) else PrelabelRecord.from_dict(pre)
    gaps = []
    for k in PRELABEL_REQUIRED:
        if k not in rec.raw:
            gaps.append(f"缺字段 {k}")
        elif k != "objects" and not rec.raw[k]:
            gaps.append(f"字段为空 {k}")
    for i, b in enumerate(rec.objects):
        if not b.label:
            gaps.append(f"objects[{i}] 缺 label")
        if not b.tight_bbox or len(b.tight_bbox) != 4:
            gaps.append(f"objects[{i}] tight_bbox 非法: {b.tight_bbox!r}")
        elif b.tight_bbox[2] <= b.tight_bbox[0] or b.tight_bbox[3] <= b.tight_bbox[1]:
            gaps.append(f"objects[{i}] 退化框: {b.tight_bbox!r}")
    return gaps
