# -*- coding: utf-8 -*-
"""imutil — 图像读写统一封装。

项目已踩过的坑：Windows 上 cv2.imread 不支持中文路径，而且静默失败
（不报错、只返回空值）。全工具链统一走 np.fromfile + cv2.imdecode。
"""
import os
import numpy as np
import cv2


def imread(path, flags=cv2.IMREAD_COLOR):
    """读图（支持中文路径）。失败返回 None，绝不静默给出空数组以外的假象。"""
    if not os.path.isfile(path):
        return None
    data = np.fromfile(path, dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)


def imwrite(path, img):
    """写图（支持中文路径）。目录不存在会自动创建。"""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    ext = os.path.splitext(path)[1] or ".png"
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        raise IOError(f"imencode 失败: {path}")
    buf.tofile(path)
    return path
