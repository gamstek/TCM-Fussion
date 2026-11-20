# -*- coding: utf-8 -*-
"""
geomlib — 中药显微鉴别 · 几何测量模块核心库

输入：实例分割产出的 Mask（本模块不关心 mask 是谁产的，只认约定的几种格式）
输出：每个实例的几何特征（形状 + 尺寸，像素测量后按 px/μm 标定值换算成真实尺寸）

设计依据（见项目《中药显微鉴别项目全流程说明.md》）：
  * mask 是尺寸测量的唯一真值，bbox 降为由 mask 派生的量；
  * 逐图 px/μm 标定由 measure.py 通过 contracts 读取分割交付清单。
    `scale_table.csv` 已于 2026-09-20 全项目停用，比例尺测定由检测阶段的双引擎负责；
  * 量化属性只对 test 集全标、训练集抽 20%（供 Agent 裁决，不进感知模型）；
  * 明视野亮背景下 Otsu 会产生假溢出轮廓——那是分割环节的坑；
    本模块收到的 mask 已成形，只做几何计算，不做阈值分割。
"""
from .imutil import imread, imwrite
from .calibration import Calibration
from .mask_io import load_instances, InstanceMask
from .descriptors import measure_instance, describe_mask
from .profiles import get_profile, REFERENCE_RANGES

__all__ = [
    "imread", "imwrite",
    "Calibration",
    "load_instances", "InstanceMask",
    "measure_instance", "describe_mask",
    "get_profile", "REFERENCE_RANGES",
]
