# -*- coding: utf-8 -*-
"""igm_exp_common — IGM 分割实验（灰度提示 / 集成采样）的公共工具。

定位：**实验代码**，不属于生产管线。生产脚本仍为 ../lib/igm_segment.py（本次实验未改动它）。
本模块只提供三类复用工具：
  1) 灰度概率提示词构造（与生产模板仅最后一条输出格式不同）；
  2) 检测框推导：兼容 crops/*.json 的两种历史 schema；
  3) 软概率图读取/聚合度量（把 run 目录的 gen_raw 回缩到原图尺寸，取 0~1 浮点）。

术语：
  soft map   —— 把生成图的（内容区）灰度回缩到原图尺寸后归一化到 0~1 的浮点图，
                即"每像素属于前景的软证据"；n 次平均后即为经验概率图。
  hard mask  —— 走 igm_segment 生产策略（各自 Otsu + 自动极性）得到的 0/255 二值图。
"""
import json
import os
import sys

import cv2
import numpy as np

SEG_DIR = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
COMMON_DIR = os.path.join(os.path.dirname(SEG_DIR), "common")
for _d in (SEG_DIR, COMMON_DIR):
    if _d not in sys.path:
        sys.path.insert(0, _d)
import igm_segment as ig  # noqa: E402
from geom import iou_mask as iou  # noqa: E402 掩码 IoU 原语（保留 iou 别名，使本目录 4 个实验零改动）

RESULTS_SEG = os.path.join(os.path.dirname(SEG_DIR), "results", "segmentation")
PROD_DIR = os.path.join(RESULTS_SEG, "igm_seg")          # 生产产物（对照组）
EXP_A = os.path.join(RESULTS_SEG, "igm_seg_expA")        # 实验A：灰度提示词单跑
EXP_B = os.path.join(RESULTS_SEG, "igm_seg_expB")        # 实验B：集成投票/软图平均
EXP_C = os.path.join(RESULTS_SEG, "igm_seg_expC")        # 实验C：多样本扩大验证
METHOD_TAG = "实验产物，勿当预标注使用"

# 灰度概率提示词：与 igm_segment.PROMPT_TEMPLATE 只有最后一条不同
# （生产版要求"纯白/纯黑、不做灰度渐变"，实验版允许灰度过渡表示不确定）
PROMPT_GRAY_TAIL = ("输出灰度概率图：越接近前景内部的像素越白（越接近 255），越接近背景越黑"
                    "（越接近 0）；边界与不确定处用中间灰度过渡表示前景概率，"
                    "不要画轮廓线，也不要把整幅画成均匀灰。")


def gray_prompt(feature_id="cluster_crystal"):
    """构造灰度概率版提示词（特征描述取自 feature_taxonomy 的单一事实源）。"""
    desc = ig.load_feature_desc(feature_id)
    return (f"请完成前景概率图任务（图中只有一个前景和背景），图中的前景是{desc}。\n"
            "请严格按图中所见该实体的实际轮廓，把它从背景中完整分离出来，保留原本形状，"
            "不要简化和复杂化，不要凭空生成图中不存在的实体。\n"
            + PROMPT_GRAY_TAIL)


def prod_meta(stem):
    """读生产产物 meta（用于定位源 crop、原图尺寸、以及作为对照组）。"""
    with open(os.path.join(PROD_DIR, stem, "meta.json"), encoding="utf-8") as f:
        return json.load(f)


def load_crop_json(img_path):
    """读同目录同名 crop json（兼容三种历史 schema）。"""
    p = os.path.splitext(img_path)[0] + ".json"
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def box_from_crop(cj, h, w):
    """crop json → 检测框布尔掩码（图像尺寸 h×w 下的局部坐标）。

    兼容三种 schema（历史上 crop 构建脚本迭代过）：
      A. tight_box_local                —— 已是局部坐标
      B. tight_box + loose_box          —— 全图坐标，用 loose 原点换算
      C. tight_box_px + loose_box_px    —— 同上但为浮点
    返回 None 表示该 json 推不出框（调用方应跳过依赖框的指标）。
    """
    if not cj:
        return None
    if "tight_box_local" in cj:
        x0, y0, x1, y1 = cj["tight_box_local"]
    elif "tight_box" in cj and "loose_box" in cj:
        tb, lb = cj["tight_box"], cj["loose_box"]
        x0, y0, x1, y1 = tb[0] - lb[0], tb[1] - lb[1], tb[2] - lb[0], tb[3] - lb[1]
    elif "tight_box_px" in cj and "loose_box_px" in cj:
        tb, lb = cj["tight_box_px"], cj["loose_box_px"]
        x0, y0, x1, y1 = tb[0] - lb[0], tb[1] - lb[1], tb[2] - lb[0], tb[3] - lb[1]
    else:
        return None
    box = np.zeros((h, w), bool)
    xa, ya = int(max(0, round(x0))), int(max(0, round(y0)))
    xb, yb = int(min(w, round(x1))), int(min(h, round(y1)))
    if xb <= xa or yb <= ya:
        return None
    box[ya:yb, xa:xb] = True
    return box


def cover(a, b):
    """a 覆盖 b 的比例（用于回答"漏掉多少目标"）。"""
    if b is None:
        return float("nan")
    return float((a & b).sum()) / max(int(b.sum()), 1)


def soft_map(run_dir, h, w):
    """run 目录的 gen_raw.png → 原图尺寸(h,w)的 0~1 软概率图（亮度=前景）。"""
    with open(os.path.join(run_dir, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    gen = ig.imread_any(os.path.join(run_dir, "gen_raw.png"))
    g = cv2.cvtColor(gen, cv2.COLOR_BGR2GRAY) if gen.ndim == 3 else gen
    roi = ig._crop_content(g, meta["letterbox"])
    return cv2.resize(roi, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0, meta


def mid_tone_ratio(roi_gray):
    """中间灰阶(40~210)占比：二值提示词下应≈0，灰度提示词下应显著>0。"""
    return float(((roi_gray >= 40) & (roi_gray <= 210)).mean())


def hard_mask(run_dir):
    """run 目录的 mask.png（生产策略二值结果）→ 布尔图。"""
    m = ig.imread_any(os.path.join(run_dir, "mask.png"))
    return (m > 0) if m is not None else None


def run_igm(img, prompt, out_dir, reuse=True):
    """调 igm_segment.main() 生成一次产物。

    幂等：out_dir 已有 meta.json 时直接复用（**不花 API 额度**），便于离线复算与复现。
    reuse=False（或传 --force）才重新调用。
    返回 (ok, meta)；ok=False 表示调用失败/被跳过。
    """
    if reuse and os.path.exists(os.path.join(out_dir, "meta.json")):
        with open(os.path.join(out_dir, "meta.json"), encoding="utf-8") as f:
            return True, json.load(f)
    argv = ["igm_segment.py", "--image", img, "--prompt", prompt, "--out-dir", out_dir]
    sys.argv = argv
    try:
        rc = ig.main()
    except SystemExit as e:                        # main 用 sys.exit 报错
        rc = e.code
    if rc != 0 or not os.path.exists(os.path.join(out_dir, "meta.json")):
        return False, None
    with open(os.path.join(out_dir, "meta.json"), encoding="utf-8") as f:
        return True, json.load(f)


def print_usage_warning(calls):
    """实验会消耗 IGM API 额度，运行前明确提示。"""
    print(f"[实验] 本次将可能发起最多 {calls} 次 IGM 调用（已有产物会自动复用，不重复扣费）")