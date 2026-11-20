# -*- coding: utf-8 -*-
"""定义单视野的像素到微米换算信息。

measure.py 通过 contracts 读取分割交付清单中的 px_per_um，再构造本模块的标定对象供几何描述子换算。标定读数由检测阶段提供。

主要定义：Calibration 保存比例尺、来源和状态，ok 检查是否有正的 px_per_um。
关联：measure.py 和 geomlib.descriptors 消费这个对象。
"""
from dataclasses import dataclass


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
