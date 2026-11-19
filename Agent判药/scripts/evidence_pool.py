# -*- coding: utf-8 -*-
"""evidence_pool — 几何测量输出 → 样本级证据卡池。

输入：measure.py 产出的 <prefix>_measurements.jsonl / .csv（支持多文件、目录）。
职责：
  1) 读取并规整测量记录（丢弃 herb 字段——防泄漏，设计方案 §1.3）；
  2) 按 image_stem → 样本（文件夹）归并（依赖 manifest，可选；无 manifest 时每图一样本）；
  3) 调 rule_engine.pool_evidence 聚合证据卡。

对测量表的解析纪律（与 几何测量/measure.py 输出契约对齐）：
  * 目录扫描时同一 prefix 的 .csv 与 .jsonl 只取一个（优先 .jsonl），
    否则每条记录被加载两遍，实例计数/否决条件全部失真；
  * image_stem 若带图像扩展名（历史一次性产物如 "20250378.tif"）自动剥除，
    保证与 manifest 的 stem 键一致；
  * flags 兼容三种载体：JSONL 数组、CSV 分号连接串（现行契约）、
    CSV 内 Python list repr（历史 measure.py 写法）。

本模块对 标注体系/、几何测量/ 只读。
"""
import csv
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))   # 项目根：contracts.py
import contracts  # noqa: E402 测量记录契约（flags 三载体解析 / CSV 数值化口径唯一事实源）

from rule_engine import pool_evidence

IMG_STEM_KEYS = ("image_stem", "stem")
_IMG_EXT_RX = re.compile(r"\.(jpe?g|png|tiff?|bmp)$", re.I)


def _normalize_rec(rec):
    """测量记录入口规整：stem 去扩展名、flags 统一为字符串列表。

    2026-09-29（P2-1）：flags 三种载体的解析已上收到契约层 `contracts.parse_flags()`
    （JSONL 数组 / CSV 分号串 / CSV 内 list repr），此处只做 stem 归一后转调。
    """
    st = rec.get("image_stem")
    if isinstance(st, str):
        rec["image_stem"] = _IMG_EXT_RX.sub("", st)
    rec["flags"] = contracts.parse_flags(rec.get("flags"))
    return rec


def _scan_files(p):
    """目录 → 每个 <prefix>_measurements 只保留一个格式（优先 jsonl）。"""
    files = sorted(p.rglob("*_measurements.*"))
    by_prefix = {}
    for f in files:
        if f.suffix.lower() not in (".jsonl", ".csv"):
            continue
        key = (f.parent, f.stem)
        cur = by_prefix.get(key)
        if cur is None or (f.suffix.lower() == ".jsonl" and cur.suffix.lower() == ".csv"):
            by_prefix[key] = f
    return sorted(by_prefix.values())


def load_measurements(paths):
    """读取一个或多个 measurements 文件/目录，返回记录列表。"""
    records = []
    for p in paths:
        p = Path(p)
        files = _scan_files(p) if p.is_dir() else [p]
        for f in files:
            if f.suffix.lower() == ".jsonl":
                with open(f, encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if line:
                            records.append(json.loads(line))
            elif f.suffix.lower() == ".csv":
                with open(f, encoding="utf-8-sig", newline="") as fh:
                    for row in csv.DictReader(fh):
                        records.append(_coerce_csv_row(row))
            else:
                # 显式指定的单文件：.csv 走数值化，其余按 jsonl 逐行读
                if f.suffix.lower() not in (".json", ".jsonl", ".txt"):
                    continue
                with open(f, encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if line:
                            records.append(json.loads(line))
    # 防泄漏：测量记录携带的 herb 列一律丢弃（该列来自 manifest，属评测 ground truth）
    for rec in records:
        rec.pop("herb", None)
        _normalize_rec(rec)
    return records


def _coerce_csv_row(row):
    """CSV 行 → 数值化 dict（jsonl 无需）。

    2026-09-29（P2-1）：口径上收到契约层 `contracts.coerce_measurement_row()` ——
    它带着"文本列白名单"（image_stem 为 "1" 这类数字串必须保持字符串），
    与 JSONL 读路的类型一致性因此只有一个地方需要维护。
    """
    return contracts.coerce_measurement_row(row)


def load_manifest_index(manifest_path):
    """manifest.csv → {image_stem: row}。只读。"""
    idx = {}
    if not manifest_path:
        return idx
    with open(manifest_path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            stem = Path(row["image_path"]).stem
            idx.setdefault(stem, row)
    return idx


def group_samples(records, manifest_idx=None):
    """记录 → {sample_id: [records]}。

    manifest 可用时样本 = 药材文件夹（生产语义=同一份送检粉末的多视野）；
    不可用时每张图独立成样本。sample_id 不携带任何药材语义进入推理。
    """
    samples = {}
    for rec in records:
        stem = rec.get("image_stem", "?")
        mrow = (manifest_idx or {}).get(stem)
        sid = mrow["folder_name"] if mrow and mrow.get("folder_name") else stem
        samples.setdefault(sid, []).append(rec)
    return samples


def build_cards(records):
    """records → {feature_id: EvidenceCard}（转调 rule_engine.pool_evidence）。"""
    return pool_evidence(records)
