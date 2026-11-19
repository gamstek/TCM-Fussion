# -*- coding: utf-8 -*-
"""calibration — 每图 px/μm 标定。

依据（项目文档 §3.3/§4.5）：
  * 全库倍率不统一（实测 CV 31.9%），绝不能假设全库一个倍率 → 必须逐图标定；
  * **标定来源唯一**：分割交付清单 `*_measure_manifest.csv` 的 `px_per_um`，
    由 `load_delivery_manifest()` 读取（原生尺寸图另有交付清单随行的同名字段）；
  * **2026-09-20 起 `scale_table.csv` 全项目停用**：不再离线批量标定，也不再有
    "右下角比例尺自动探测 + 默认 20μm" 的兜底路径。比例尺测定改由检测阶段的双引擎
    （`标注体系/common/scalebar_duo.py`：A 本地形态学+OCR / B VLM，交叉验证）负责，
    结果随 prelabel JSON 与交付清单流转。
  * 拿不到 px_per_um 时**由调用方直接报错**，不静默降级（见 measure.py）。
"""
import os
import csv
from dataclasses import dataclass

import numpy as np


@dataclass
class Calibration:
    px_per_um: float | None      # None = 未标定
    bar_px: float | None
    bar_um: float | None
    source: str                  # delivery_manifest | uncalibrated
    status: str                  # ok | failed
    note: str = ""

    @property
    def ok(self) -> bool:
        return self.px_per_um is not None and self.px_per_um > 0


def detect_scale_bar_px(gray):
    """右下角 ROI 内找最长暗色水平游程 = 比例尺线段像素长度。

    仅供对照/实验使用（`common/scalebar_duo.py` 把它作为旧算法基线记录在报告里）；
    标定链路已不使用本函数（2026-09-20 起自动探测兜底退役）。
    必须排除图像暗边框行与近全宽游程（边框伪影）。返回 0 表示检测失败。
    """
    h, w = gray.shape
    x0, y0 = int(w * 0.72), int(h * 0.90)
    roi = gray[y0:h - 3, x0:w - 3]
    if roi.size == 0:
        return 0
    rw = roi.shape[1]
    dark = (roi < 100).astype(np.uint8)
    best = 0
    for row in dark:
        run = cur = 0
        for v in row:
            cur = cur + 1 if v else 0
            if cur > best:
                best = cur
    if best > rw * 0.9 or best < 8:      # 全宽游程=边框伪影；<8px 不可信
        return 0
    return best


def load_delivery_manifest(masks_dir):
    """读分割交付清单 `*_measure_manifest.csv` → {源图stem: row(原始 dict)}。

    分割小图本身不含任何比例尺像素（裁剪步约定；此前提的原实现 build_seg_crops.py 已于 2026-09-29 归档），crop 级 mask 的
    标定只能来自交付清单中人工实测的 px_per_um；清单同时携带 herb/object_label 与
    source_image_path（原图出处），是逐图元数据的最高优先级来源，并可作 manifest
    关联的消歧线索（全局 manifest 同名文件极多，裸 stem 不可用）。
    兼容两种命名：`<stem>__measure_manifest.csv` 与 `<stem>__<feature>__measure_manifest.csv`。

    2026-09-29（P2-1）：扫描与字段解析已上收到项目根 `contracts.py`（契约唯一事实源）；
    本函数保留原签名与"返回原始 dict"的行为，需要类型化访问（`px_per_um` 浮点、
    `require_px_per_um()` 报错）的调用方请直接用 `contracts.load_manifest_rows()`。
    """
    from contracts import load_manifest_rows   # 延迟导入：几何模块的可选依赖
    return {stem: row.raw for stem, row in load_manifest_rows(masks_dir).items()}
