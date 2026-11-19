# -*- coding: utf-8 -*-
"""
textvis — OpenCV 图像上的中文文字渲染（cv2.putText 只支持 ASCII，中文会变 ?）。
用 Pillow + 系统中文字体（黑体→雅黑→宋体 自动回退）绘制，输入输出均为 BGR ndarray。

用法（批量画字，避免逐条转换整幅图）:
    from textvis import draw_texts_cn
    ov = draw_texts_cn(ov, [("草酸钙簇晶(high)", (120, 30), (0, 0, 255), 20)])
颜色为 BGR 元组，org 为文字左上角像素坐标，size 为字号(px)。
"""
import os

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

_FONT_PATH = None
_FONT_CACHE = {}


def _find_font():
    global _FONT_PATH
    if _FONT_PATH is None:
        for p in (r"C:\Windows\Fonts\simhei.ttf",
                  r"C:\Windows\Fonts\msyh.ttc",
                  r"C:\Windows\Fonts\simsun.ttc"):
            if os.path.exists(p):
                _FONT_PATH = p
                break
        _FONT_PATH = _FONT_PATH or ""          # 找不到时置空串，走 PIL 默认字体
    return _FONT_PATH


def _font(size):
    if size not in _FONT_CACHE:
        path = _find_font()
        _FONT_CACHE[size] = (ImageFont.truetype(path, size) if path
                             else ImageFont.load_default())
    return _FONT_CACHE[size]


def draw_texts_cn(img_bgr, items):
    """items: [(text, (x, y), (b, g, r), size?), ...] 返回新 BGR 图。"""
    if not items:
        return img_bgr
    pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(pil)
    for it in items:
        text, org, color = it[0], it[1], it[2]
        size = it[3] if len(it) > 3 else 18
        # BGR → PIL RGB
        d.text(org, text, font=_font(size), fill=(color[2], color[1], color[0]))
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def put_text_cn(img_bgr, text, org, color=(0, 0, 255), size=18):
    """单条文字的便捷封装。"""
    return draw_texts_cn(img_bgr, [(text, org, color, size)])
