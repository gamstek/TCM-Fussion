# -*- coding: utf-8 -*-
"""descriptors — 从单个实例 mask 计算几何描述子（像素空间）。

输出分四组（像素单位，换算 μm 在 measure_instance 中统一做）：
  尺寸组: area, perimeter, bbox, feret_max/min(+角度), eq_diameter,
          max_inscribed_dia, minAreaRect 宽高
  细长组: skeleton_length, mean/median width along skeleton, tortuosity
          —— 对针晶(束)/纤维/导管等细长目标有效，圆粒目标仅参考
  形状组: aspect_ratio, eccentricity, solidity, extent, convexity,
          circularity, roundness, rectangularity
  结构组: convexity defect 计数与深度（簇晶"棱角钝度"代理）、
          dt_peak_count（复粒分粒数 / 多分粒聚集的近似）
"""
import math

import numpy as np
import cv2

from .profiles import get_profile

_SKELETON_MAX_AREA = 1_200_000     # 骨架计算前 bbox 超过此面积则降采样（防慢）
_EPS = 1e-9                        # 浮点零阈值：参与除法/比值前判断是否为 0，避免除零与伪值


# ---------------------------------------------------------------- 基础几何
def _contours(mask):
    """提取 mask 的外轮廓集合。

    RETR_EXTERNAL：只取最外层轮廓，忽略孔洞；
    CHAIN_APPROX_NONE：保留全部边界像素点，保证周长/凸包/缺陷计算精度。
    少于 3 个点的轮廓无法构成面积，直接丢弃。
    """
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return [c for c in cnts if len(c) >= 3]


def _hull_points(cnts):
    """求所有轮廓点的凸包顶点（N×2，float64）。

    多连通域（针晶束/聚集复粒）先合并全部点再求包，
    得到的是整体外包围凸包，供 Feret 长/短径与 solidity 使用。
    """
    pts = np.concatenate([c.reshape(-1, 2) for c in cnts]).astype(np.float32)  # 各轮廓点拼成 (N,2)
    hull = cv2.convexHull(pts)                    # 凸包顶点（逆时针，仅保留极点）
    return hull.reshape(-1, 2).astype(np.float64)


def _feret_max(P):
    """Feret 长径（最大卡尺径）：凸包顶点两两之间的最大距离。

    P：凸包顶点数组 (M,2)；M 通常远小于轮廓点数，故 O(M²) 全对枚举可接受。
    d2：M×M 距离平方矩阵；开方取最大值即像素单位的 feret_max。
    """
    if len(P) < 2:
        return 0.0                                # 退化：不足两点无距离可言
    d2 = ((P[:, None, :] - P[None, :, :]) ** 2).sum(-1)   # 两两差向量平方和 → M×M
    return float(math.sqrt(d2.max()))


def _feret_min(P):
    """Feret 短径（最小卡尺径）：旋转卡壳法。

    原理：卡尺的最小宽度必出现在某条凸包边的法线方向上，
    故遍历每条边，把所有凸包点投到该边法线，投影跨度即为该方向的宽度，取全局最小。
    P：凸包顶点数组 (M,2)。
    """
    if len(P) < 2:
        return 0.0
    if len(P) == 2:
        return 0.0                                # 仅两点无法定义宽度
    best = float("inf")                           # 当前最小投影跨度
    for i in range(len(P)):
        a, b = P[i], P[(i + 1) % len(P)]          # 凸包的第 i 条边（首尾相连）
        d = b - a                                 # 边向量
        n = np.array([-d[1], d[0]])               # 边法线
        nn = math.hypot(n[0], n[1])               # 法线模长（用于归一化）
        if nn < _EPS:
            continue                              # 退化边（两点重合）跳过
        proj = P @ (n / nn)                       # 所有顶点在单位法线上的投影值
        best = min(best, float(proj.max() - proj.min()))   # 该方向的宽度
    return best if math.isfinite(best) else 0.0   # 无有效边时回退 0


def _zhang_suen(img):
    """Zhang-Suen 形态学细化（向量化实现），输入/输出均为 uint8 的 0/1 图。

    每轮迭代分两个子步骤，反复剥除满足条件的边界像素，直到不再变化，
    最终把粗 mask 收缩成单像素宽的中心线（骨架），供细长组的长度/宽度统计使用。
    命名对照：P2~P9 为当前像素的 8 邻域（顺时针），
      B = 前景邻居总数，A = 邻域 0→1 的跳变次数。
    """
    I = np.pad(img, 1).astype(np.uint8)           # 补 1 圈 0 边界，便于整块向量化取邻域
    for _ in range(10000):                        # 迭代上限，正常几轮即收敛
        changed = False                           # 本轮是否有像素被删除
        for step in (0, 1):                       # 子迭代：交替剥离两个方向的边界
            C = I[1:-1, 1:-1]                     # 中心像素
            P2, P3 = I[:-2, 1:-1], I[:-2, 2:]     # 上、右上
            P4, P5 = I[1:-1, 2:], I[2:, 2:]       # 右、右下
            P6, P7 = I[2:, 1:-1], I[2:, :-2]      # 下、左下
            P8, P9 = I[1:-1, :-2], I[:-2, :-2]    # 左、左上
            B = P2 + P3 + P4 + P5 + P6 + P7 + P8 + P9    # 前景邻居个数
            seq = (P2, P3, P4, P5, P6, P7, P8, P9, P2)   # 首尾相接，便于统计跳变
            A = np.zeros_like(C)                  # 邻域 0→1 跳变次数（反映端点/连接性）
            for i in range(8):
                A += ((seq[i] == 0) & (seq[i + 1] == 1)).astype(np.uint8)
            if step == 0:
                # 子迭代 1：上/左/右条件（避免腐蚀出断点）
                cond = (P2 * P4 * P6 == 0) & (P4 * P6 * P8 == 0)
            else:
                # 子迭代 2：下/右/左条件
                cond = (P2 * P4 * P8 == 0) & (P2 * P6 * P8 == 0)
            # 删除条件：自身为前景、B∈[2,6]、A==1 且满足该子迭代的拓扑约束
            dele = (C == 1) & (B >= 2) & (B <= 6) & (A == 1) & cond
            if dele.any():
                I[1:-1, 1:-1][dele] = 0           # 批量置 0（剥离边界）
                changed = True
        if not changed:                           # 无像素可删 → 已收敛
            break
    return I[1:-1, 1:-1]                          # 去掉补边，还原原尺寸


def _skeleton(mask):
    """对 mask 做细化求骨架，返回 (skel, y0, x0, scale)。

    先按前景 bbox 裁剪（去掉大片空白，减少细化开销）；
    若裁剪区面积超过 _SKELETON_MAX_AREA，则按比例降采样后再细化，
    此时 scale>1，(y0,x0) 为裁剪原点，用于把骨架坐标映射回原 mask 坐标系。
    """
    m = (mask > 0).astype(np.uint8)
    ys, xs = np.nonzero(m)
    if len(ys) == 0:
        return m, 0, 0, 1.0                       # 空 mask：无骨架
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1   # 前景包围盒
    crop = m[y0:y1, x0:x1]                        # 裁剪到目标区域
    h, w = crop.shape
    scale = 1.0                                   # 缩放因子：1 表示未降采样
    if h * w > _SKELETON_MAX_AREA:
        scale = math.sqrt(h * w / _SKELETON_MAX_AREA)     # 使面积降到阈值附近的缩放比
        nw, nh = max(8, int(math.ceil(w / scale))), max(8, int(math.ceil(h / scale)))
        # INTER_AREA 缩小的抗混叠效果最好；>0 重新二值化，保持 0/1 语义
        crop = (cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA) > 0).astype(np.uint8)
    skel = _zhang_suen(crop)                      # 细化得到单像素宽中心线
    return skel, int(y0), int(x0), scale


def _skeleton_length_px(skel):
    """按 8 邻域无向边计数估算骨架长度（像素）：

    水平/垂直相邻边记 1，对角相邻边记 √2，累加即骨架的近似折线长度。
    s：布尔化的骨架图；n_h/n_v 为水平/垂直边数，n_d1/n_d2 为两个方向的对角边数。
    """
    s = skel.astype(bool)
    n_h = int((s[:, :-1] & s[:, 1:]).sum())       # 水平相邻点对
    n_v = int((s[:-1, :] & s[1:, :]).sum())       # 垂直相邻点对
    n_d1 = int((s[:-1, :-1] & s[1:, 1:]).sum())   # 主对角（左上-右下）
    n_d2 = int((s[:-1, 1:] & s[1:, :-1]).sum())   # 副对角（右上-左下）
    return n_h + n_v + math.sqrt(2.0) * (n_d1 + n_d2)


# ---------------------------------------------------------------- 主入口
def describe_mask(mask):
    """对单个实例的 0/1 uint8 mask 计算全部描述子，返回像素空间 dict。

    返回的键值均以像素为单位（面积 px²、长度 px），换算成 μm 由 measure_instance 统一处理。
    out["flags"] 记录质量标记：empty / multi_component / degenerate_tiny 等。
    """
    m = (mask > 0).astype(np.uint8)               # 规范化成 0/1，避免非二值输入干扰统计
    out = {"n_components": 0, "flags": []}        # 结果容器 + 质量标记列表
    ys, xs = np.nonzero(m)                        # 所有前景像素坐标（面积用 len(ys)）
    if len(ys) == 0:
        out["flags"].append("empty")              # 空 mask，直接返回
        return out

    cnts = _contours(m)
    out["n_components"] = len(cnts)               # 连通域（轮廓）个数
    if out["n_components"] > 1:
        out["flags"].append("multi_component")   # 针晶束/成束纤维/复粒等
    area = float(len(ys))                         # 面积 = 前景像素计数（px²）
    if area < 8 or not cnts:
        out["flags"].append("degenerate_tiny")    # 过小目标，后续比值类指标不可靠

    # 尺寸组
    out["area_px"] = area
    out["perimeter_px"] = float(sum(cv2.arcLength(c, True) for c in cnts)) if cnts else 0.0  # 各外轮廓周长之和（True=闭合）
    out["bbox_x"], out["bbox_y"] = int(xs.min()), int(ys.min())          # 外接矩形左上角（mask 坐标系）
    out["bbox_w"], out["bbox_h"] = int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)  # 外接矩形宽高（含端点像素，故 +1）

    hullP = _hull_points(cnts) if cnts else np.zeros((0, 2))   # 凸包顶点；无轮廓则为空数组
    fmax, fmin = _feret_max(hullP), _feret_min(hullP)          # Feret 长径 / 短径（像素）
    out["feret_max_px"], out["feret_min_px"] = fmax, fmin
    if len(hullP) >= 2:
        # 长径方向角（度，相对 x 轴，[0,180)）
        d2max, pair = 0.0, (hullP[0], hullP[1])   # d2max：当前最大距离平方；pair：构成该距离的点对
        for i in range(len(hullP)):
            d2 = ((hullP[i + 1:] - hullP[i]) ** 2).sum(-1)     # 第 i 点到其后各点的距离平方
            if len(d2) and d2.max() > d2max:
                d2max = float(d2.max())
                j = int(d2.argmax()) + i + 1       # 还原成 hullP 中的真实下标
                pair = (hullP[i], hullP[j])
        ang = math.degrees(math.atan2(pair[1][1] - pair[0][1], pair[1][0] - pair[0][0]))  # 长径倾角（度）
        out["feret_angle_deg"] = float((ang + 180.0) % 180.0)  # 归一到 [0,180)，消除方向二义性
        # 长径端点：凸包上构成最大距离的那一对点（与 mask 同坐标系）。
        # 叠加图直接连这两点，红线长度/位置即 feret_max 本身，不再用"质心+方向+估长"示意。
        out["feret_p1_px"] = [float(pair[0][0]), float(pair[0][1])]
        out["feret_p2_px"] = [float(pair[1][0]), float(pair[1][1])]
    else:
        out["feret_angle_deg"] = 0.0
        out["feret_p1_px"] = out["feret_p2_px"] = None    # 无法确定长径端点
    out["eq_diameter_px"] = 2.0 * math.sqrt(area / math.pi)   # 等效圆直径：与目标等面积圆的直径

    DT = cv2.distanceTransform(m, cv2.DIST_L2, 5)     # 每个前景像素到最近背景的距离（欧氏，5×5 掩码近似）
    dt_max = float(DT.max())                          # DT 峰值 = 最大内切圆半径
    out["max_inscribed_dia_px"] = 2.0 * dt_max        # 最大内切圆直径（目标最粗处的宽度）

    rect = cv2.minAreaRect(np.concatenate([c for c in cnts]).astype(np.float32)) if cnts \
        else ((0, 0), (0, 0), 0)                      # 最小面积旋转外接矩形（多轮廓合并成一个整体求）
    (_, _), (rw, rh), rang = rect                     # 中心(忽略)、(宽,高)、旋转角（度）
    out["rect_w_px"], out["rect_h_px"], out["rect_angle_deg"] = float(rw), float(rh), float(rang)

    # 形状组
    hull_area = float(cv2.contourArea(hullP.astype(np.float32))) if len(hullP) >= 3 else area   # 凸包面积（不足 3 点时退回面积本身）
    hull_perim = float(cv2.arcLength(hullP.astype(np.float32), True)) if len(hullP) >= 3 else out["perimeter_px"]  # 凸包周长
    P = out["perimeter_px"]                       # 实例真实外轮廓周长（下文比值共用）
    out["solidity"] = min(1.0, area / hull_area) if hull_area > _EPS else 0.0
    # 注: area 为像素计数、hull_area 为多边形面积，小目标上两者口径差可致比值略>1，截断到 1
    out["extent"] = area / (out["bbox_w"] * out["bbox_h"])                              # 充盈度：面积占外接正矩形比例
    out["convexity"] = hull_perim / P if P > _EPS else 0.0          # 凸度：凸包周长/实际周长，越小越皱
    out["circularity"] = 4 * math.pi * area / (P * P) if P > _EPS else 0.0             # 圆度(4πA/P²)：圆为 1，越皱越小
    out["roundness"] = 4 * area / (math.pi * fmax * fmax) if fmax > _EPS else 0.0      # 圆钝度(4A/πDmax²)：圆粒约 1，针晶趋 0
    out["rectangularity"] = area / (fmax * fmin) if fmax * fmin > _EPS else 0.0        # 矩形度：面积占 Feret 外接矩形比例
    out["aspect_ratio"] = fmax / fmin if fmin > _EPS else 0.0                          # 长宽比：细长目标的判据主量

    Mo = cv2.moments(m, True)                     # 二值图矩（True 表示输入按 0/1 计算，而非灰度加权）
    a, b, c = Mo["mu20"], Mo["mu02"], Mo["mu11"]  # 二阶中心矩：x 方差、y 方差、协方差
    tr, com = a + b, math.sqrt(max((a - b) ** 2 + 4 * c * c, 0.0))   # 迹 与 判别式项（用于求特征值）
    e1, e2 = (tr + com) / 2, (tr - com) / 2       # 惯性椭圆的长/短半轴平方（两个特征值）
    out["eccentricity"] = math.sqrt(max(1 - e2 / e1, 0.0)) if e1 > _EPS else 0.0       # 偏心率：0=圆，越接近 1 越扁长

    # 结构组：凹角（凸包缺陷）—— 簇晶"棱角"钝锐、多瓣外缘的代理量
    out["defect_count"], out["mean_defect_depth_ratio"] = 0, 0.0   # 显著凹角个数 / 平均凹深与 fmax 之比
    if cnts:
        main = max(cnts, key=cv2.contourArea)     # 只分析面积最大的轮廓（主体）
        if len(main) >= 6:                        # 点数过少时凸包缺陷不可靠
            hull_idx = cv2.convexHull(main, returnPoints=False)    # 凸包顶点的索引（非坐标）
            defects = cv2.convexityDefects(main, hull_idx)          # 凸包缺陷四元组 (起点,终点,最远点,深度)
            if defects is not None and len(defects):
                # reshape(-1,4) 兼容 cv2 版本差异：旧版 (M,1,4)、新版 (M,4)
                depths = defects.reshape(-1, 4)[:, 3] / 256.0       # 第 4 列为深度，cv2 内部放大 256 倍，需还原
                sig = depths[depths > max(2.0, 0.03 * fmax)]        # 只保留显著凹陷（>2px 且 >3% 长径）
                out["defect_count"] = int(len(sig))
                if len(sig):
                    out["mean_defect_depth_ratio"] = float(sig.mean() / fmax) if fmax > _EPS else 0.0   # 凹深/长径：越小越"圆钝"

    # 复粒/多粒聚集近似：DT 高值区连通块数（仅类圆目标有意义，剖面层取舍）
    if dt_max > 1:
        num, _ = cv2.connectedComponents((DT >= 0.5 * dt_max).astype(np.uint8))   # 取半径≥峰值一半的区域，看有几个"芯"
        out["dt_peak_count"] = int(num - 1)       # 减去背景标签
    else:
        out["dt_peak_count"] = 0                  # 目标过薄，无法区分分粒

    # 细长组：骨架长 / 沿骨架宽度 / 弯曲度
    skel, oy, ox, sc = _skeleton(m)               # skel：细化后的骨架；oy/ox：裁剪原点；sc：降采样比率
    if skel.any():
        out["skeleton_len_px"] = _skeleton_length_px(skel) * sc    # 还原到原分辨率下的骨架长度
        sy, sx = np.nonzero(skel)                 # 骨架点在裁剪（并可能降采样）坐标系中的位置
        # 映射回原 mask 坐标系：先还原缩放，再加裁剪原点，最后夹到图像边界内防越界
        fy = np.clip(np.round(sy * sc + oy).astype(int), 0, m.shape[0] - 1)
        fx = np.clip(np.round(sx * sc + ox).astype(int), 0, m.shape[1] - 1)
        wvals = 2.0 * DT[fy, fx]                  # 骨架点处的局部宽度 = 2×该点到背景的距离
        wvals = wvals[wvals > 0]                  # 丢掉落在背景上的无效采样
        if len(wvals):
            out["mean_width_px"] = float(wvals.mean())            # 平均宽度
            out["median_width_px"] = float(np.median(wvals))      # 中位宽度（抗分支/毛刺影响）
            out["width_p25_px"] = float(np.percentile(wvals, 25)) # 25 分位：统计偏细段
            out["width_p75_px"] = float(np.percentile(wvals, 75)) # 75 分位：统计偏粗段
        out["tortuosity"] = out["skeleton_len_px"] / fmax if fmax > _EPS else 0.0   # 弯曲度：骨架长/直线长径，1=笔直
    else:
        out["skeleton_len_px"] = 0.0              # 无骨架（退化目标）
        out["tortuosity"] = 0.0

    return out


# ---------------------------------------------------------------- 剖面换算
def measure_instance(inst, calib, herb=None):
    """InstanceMask + Calibration → 单实例完整测量记录（px + μm + 主判尺寸）。

    inst：单实例（含 mask/label/image_stem/instance_id）；
    calib：标定结果（提供 px_per_um，标定失败则为 None，此时所有 μm 字段为 None）；
    herb：药材名，仅作记录用。
    """
    d = describe_mask(inst.mask)                  # 像素空间描述子
    s = calib.px_per_um if calib and calib.ok else None   # 每 μm 像素数；标定失败置 None
    profile = get_profile(inst.label)             # 该特征类型（如"针晶"）的剖面配置

    def U(key_px, mul=1.0):
        """把 d 中的 px 值换算成 μm；mul 用于面积等非一维量的幂次修正。"""
        v = d.get(key_px)
        if v is None:
            return None
        return (v / s) * mul if s else None        # 未标定时一律返回 None

    rec = {
        "image_stem": inst.image_stem,            # 来源图名（不含扩展名）
        "instance_id": inst.instance_id,          # 实例编号（同图内唯一）
        "feature": inst.label or "generic",       # 特征类别名，未标注则记为 generic
        "herb": herb,                             # 药材名（记录用，不参与计算）
        "calibration_px_per_um": s,               # 标定比：每 μm 像素数
        "calibration_source": calib.source if calib else "none",      # 标定来源（如 scalebar/ocr）
        "calibration_status": calib.status if calib else "failed",    # 标定状态
        "flags": list(d.get("flags", [])),        # 质量标记（empty/multi_component 等）
        # 尺寸组
        "area_px": d.get("area_px"),              # 面积（像素计数）
        "area_um2": U("area_px", 1.0 / s) if s else None,   # px² → μm²：/s²
        "perimeter_um": U("perimeter_px"),        # 周长（μm）
        "bbox_x": d.get("bbox_x"), "bbox_y": d.get("bbox_y"),          # 外接矩形左上角（px）
        "bbox_w_px": d.get("bbox_w"), "bbox_h_px": d.get("bbox_h"),    # 外接矩形宽高（px）
        "feret_max_um": U("feret_max_px"),        # Feret 长径（μm）
        "feret_min_um": U("feret_min_px"),        # Feret 短径（μm）
        "feret_angle_deg": d.get("feret_angle_deg"),   # 长径方向角（度，[0,180)）
        # 长径端点（px，与 bbox_x/bbox_y 同坐标系）：叠加图据此画红线，不换算 μm
        "feret_p1_px": d.get("feret_p1_px"),
        "feret_p2_px": d.get("feret_p2_px"),
        "eq_diameter_um": U("eq_diameter_px"),             # 等效圆直径（μm）
        "max_inscribed_dia_um": U("max_inscribed_dia_px"), # 最大内切圆直径（μm）
        "rect_w_um": U("rect_w_px"), "rect_h_um": U("rect_h_px"),   # 最小外接旋转矩形的宽/高（μm）
        "n_components": d.get("n_components"),    # 轮廓（连通域）个数
        # 细长组
        "skeleton_len_um": U("skeleton_len_px"),  # 骨架长度（μm）
        "mean_width_um": U("mean_width_px"),      # 沿骨架的平均宽度（μm）
        "median_width_um": U("median_width_px"),  # 沿骨架的中位宽度（μm）
        "width_p25_um": U("width_p25_px"),        # 宽度 25 分位（μm）
        "width_p75_um": U("width_p75_px"),        # 宽度 75 分位（μm）
        "tortuosity": d.get("tortuosity"),        # 弯曲度（无量纲，1=笔直）
        # 形状组（无量纲，不换算）
        "aspect_ratio": d.get("aspect_ratio"),    # 长宽比 = 长径/短径
        "eccentricity": d.get("eccentricity"),    # 偏心率（惯性椭圆）
        "solidity": d.get("solidity"),            # 实心度 = 面积/凸包面积
        "extent": d.get("extent"),                # 充盈度 = 面积/外接正矩形
        "convexity": d.get("convexity"),          # 凸度 = 凸包周长/实际周长
        "circularity": d.get("circularity"),      # 圆度 = 4πA/P²
        "roundness": d.get("roundness"),          # 圆钝度 = 4A/(π·Dmax²)
        "rectangularity": d.get("rectangularity"),# 矩形度 = 面积/(长径×短径)
        # 结构组
        "defect_count": d.get("defect_count"),    # 凸包显著凹角个数（棱角/瓣数代理）
        "mean_defect_depth_ratio": d.get("mean_defect_depth_ratio"),   # 平均凹深/长径
        "dt_peak_count": d.get("dt_peak_count"),  # DT 高值区个数（复粒分粒数近似）
    }

    # 主判尺寸：按特征剖面选 primary（供判据核验/范围核查使用）
    prim_dim, prim_val = None, None               # 选中的字段名与取值（μm）
    for dim in profile["primary_dims"]:           # 按剖面给出的优先顺序依次尝试
        v = rec.get(dim)
        if v is not None and v > 0:               # 取第一个有有效值的指标
            prim_dim, prim_val = dim, float(v)
            break
    if profile.get("coarse_quant"):               # L1 量化闸门：整片/无量纲单值的特征（表皮细胞/气孔）
        prim_dim, prim_val = None, None           # 不产 primary_value_um → 判药按"只作存在性"（coarse_quant）
        rec["flags"].append("coarse_quant")
    rec["primary_dimension"] = prim_dim           # 该实例的主判尺寸字段名
    rec["primary_value_um"] = prim_val            # 主判尺寸数值（μm），无则 None
    if not s:
        rec["flags"].append("uncalibrated")       # 未标定：μm 字段全为 None，需上层拦截
    return rec
