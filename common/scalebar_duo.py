# -*- coding: utf-8 -*-
"""
scalebar_duo.py — 比例尺双引擎标定（本地形态学 + 本地OCR + VLM 交叉验证）。

**定位：库 + 维护入口**。生产链路只用它的两个公共 API（`measure_px_per_um` /
`measure_img_px_per_um`），由检测阶段（`prompts`/`detect_core` 侧）与 UI（`ui/tools.py`、
`ui/px_read_cli.py`）调用；文件末尾的 `--images` 批量对照入口是**开发/调试用的维护入口**，
不在生产链路上（2026-09-29 P3 复核：原计划"拆除该入口"，因 `px_read_cli` 只做单图读数、
不产多图对照 CSV，拆除会丢掉唯一的批量验证能力，故改为保留并在此标注用途）。

引擎分工（每引擎各自给出"完整标定：像素长 + μm 数"，再做交叉验证）：
  A 引擎（全本地，不调 API）—— 文字优先（2026-09-20 二次实测重构）
     - RapidOCR **全分辨率**读烧录文字里的 μm 数字 bar_um，文字 bbox 同时作为线段锚点
     - 在文字邻域窄带内以"近纯黑长游程 + 两端端帽 + 行间一致"定线段像素长 bar_px
     - 拿不到文字时退回红尺/黑尺形态学路径（选序见 engine_a 文档串）
  B 引擎（VLM API 一次推理读出两者；模型与思考强度见 detection/config/vlm_config.json）
     - 读线段 bbox → 换算 bar_px
     - 读烧录文字 → bar_um
（2026-09-29 P1-4）原报告里的"旧算法基线 orig"（复用 几何测量/geomlib/calibration.py 的
detect_scale_bar_px）已移除：它使声明的共享层反向依赖上层模块，且该基线仅为报告对照用。
**本模块不依赖 几何测量/ 下任何代码**——依赖方向保持 几何测量/geomlib ← 项目根 common/。
对应地，几何侧自带同名函数与其自测用例（几何测量/selftest_synthetic.py）不受影响。

交叉验证规则（2026-09-20 重做，2026-09-28 补 OK-consensus；详见 cross_validate 判定表）：
     μm 以 A 的本地 OCR 为权威；A 的 px 过四层判据 + ppm 物理窗后即可信，
     B 的框只作单边证实（B 惯性过框）。status 分级：
       OK（双读一致）| OK-consensus（A 可信、B 未证实，但 A 的 bar_px 已被同批精确 OK
       样本旁证——实测 119 份 OK-medium 里 118 份属此列，靠它免重跑即可用于测量）|
       OK-medium（A 可信但 B 未证实且 bar_px 无旁证）| OK-low（族先验 px_fam/um，
       无族先验时 B 框过物理窗兜底 px_B/um）| manual（须人工）| n/a（无有效长）。
     可信档 = ls_common.TRUSTED_STATUSES（OK / OK-consensus）；其余下游 criteria_hint_for
     自动追加"μm 仅供参考"提示。

公共 API：`measure_px_per_um(image_path, cfg, px_fam=…, px_ok_set=…)` —— 检测阶段
（vlm_prelabel / detect_sample）与 UI（ui/tools.py）共用，是全项目 px_per_um 的唯一
生产方式（离线 scale_table.csv 已停用）。px_ok_set 由 ls_common.confirmed_bar_px() 提供。

输出：results/detection/scalebar_duo/scalebar_duo_report.csv（逐图对照，仅供目检）

用法（venv python）：
  python scalebar_duo.py --images p1,p2 [--tol 0.10] [--max-side 1536]
`--images` 必填（逗号分隔）：早先默认取 archive/demo_output/input/ 的 3 张黑尺图，
归档目录不该被库代码当输入源，已改为显式传参（2026-09-29 P1-4）。
"""
import argparse
import os
import csv
import re

import cv2
import numpy as np

import paths                                          # 路径唯一权威（项目根 paths.py）

# 输出目录取路径权威，**不再按自身位置推深度**（2026-10-09：本模块自 标注体系/common 上提到
# 项目根 common/ 后，`dirname(HERE)` 会退到项目根、`results/` 落到错误位置 —— 与 P4-14/P4-15 同类坏点）。
OUT = os.path.join(paths.DET_RESULTS, "scalebar_duo")

from ls_common import (STATUS_OK, STATUS_OK_CONSENSUS, TRUSTED_STATUSES,  # noqa: E402
                       imread_cn, load_config, resolve_key)
import vlm_api                                               # call_vlm / encode_jpeg_b64 / parse_vlm_json

B_PROMPT = """你是中药显微图像标注助手。这张图右下角或图内某处烧录了比例尺（一条**线段** + 旁边文字如"长度：20.00um"）。

请只针对这条比例尺线段（【不含】旁边的文字）：
- present：是否存在可辨认的比例尺线段
- color：线段颜色（black / red / other / none）
- bbox_2d：比例尺**线段**的紧外接矩形，0~1000 归一化坐标 [x1,y1,x2,y2]（左上/右下），务必只包住线段本身不包文字
- um_label：烧录文字里标注的微米数值（数字，如 20；读不到给 null）
- confidence：high / medium

只输出 JSON，不要输出其他文字：
{"present": true, "color": "black", "bbox_2d": [x1,y1,x2,y2], "um_label": 20, "confidence": "high"}"""


# ---------------- 本地 OCR（RapidOCR，惰性单例） ----------------
_OCR = None
# OCR 侧分辨率（2026-09-20 实测）：烧录文字里的"20.00"在 1536 下会被糊成"2000"，
# 出数值窗被判无效（蝉蜕 B20132527 实测 1536→"长度：2000um"、2048→"长度：20:00um"），
# 故按 2048 图族取全分辨率；3840 图族缩到 2048（0.53×）实测仍稳定读出"20μm"。
_OCR_MAX_SIDE = 2048
# VLM 侧分辨率单独固定：engine_b 沿用线上既有输入尺度，避免改 OCR 分辨率时连带改变 B 的行为。
VLM_MAX_SIDE = 1536


def _get_ocr():
    global _OCR
    if _OCR is None:
        from rapidocr_onnxruntime import RapidOCR
        _OCR = RapidOCR()
    return _OCR


_DIGIT = re.compile(r"(\d+(?:\.\d+)?)")


def ocr_read_um(img_bgr, bar_bbox, max_side=_OCR_MAX_SIDE):
    """全图下采样做一次 OCR，取『靠近比例尺 bbox』的文字框，解析其中 μm 数值。

    返回临近比例尺的数字（能跟'μm'关联优先）；找不到返回 None。
    bar_bbox = [x1,y1,x2,y2] 原图坐标。
    """
    h, w = img_bgr.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    small = img_bgr if scale >= 1.0 else \
        cv2.resize(img_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    res, _ = _get_ocr()(small)
    if not res:
        return None
    # 以比例尺 bbox 为中心放大做近邻窗
    cx = (bar_bbox[0] + bar_bbox[2]) / 2.0
    cy = (bar_bbox[1] + bar_bbox[3]) / 2.0
    bw = bar_bbox[2] - bar_bbox[0] + 1
    half_x = max(bw, 600) * 0.9
    half_y = max((bar_bbox[3] - bar_bbox[1] + 1) * 5, 150)
    best = None
    best_text = None
    for box, text, conf in res:
        xs = [p[0] for p in box]; ys = [p[1] for p in box]
        tx = np.mean(xs) / scale; ty = np.mean(ys) / scale
        if abs(tx - cx) > half_x or abs(ty - cy) > half_y:
            continue
        if abs(ty - cy) > half_y:   # 位置过滤：通常文字就在尺旁
            continue
        m = _DIGIT.search(text)
        if not m:
            continue
        val = float(m.group(1))
        if not (1.0 <= val <= 1000):   # μm 数值合理窗
            continue
        # 倾向与 'μm'/'um' 直接关联的文本
        boosted = 0.0 if re.search(r"[μµm]m?", text, re.I) else 0.5
        if best is None or (conf - boosted) > (best_conf - best_boost):
            best, best_text, best_conf, best_boost = val, text, conf, boosted
    return best


# ---------------- A 引擎：形态学量长 + 本地OCR读μm ----------------
# 黑尺位置先验（2026-09-20 实测 2048×1536/3840×2160 图族）：黑尺固定在右下角，
# x/w ≈ 0.83~0.98、y/h ≈ 0.88~0.93，位置小幅浮动 → ROI 取 (0.60, 0.80) 留足余量。
# 红尺位置**完全不固定**（实测 3840×2160 图族的红尺 y/h 在 0.55~0.94 间漂移）→ 不加 ROI，全图搜索。
BLACK_ROI_X = 0.60
BLACK_ROI_Y = 0.80
# 红尺资格（两项同时满足才算"实心红尺"）：
#   ① 候选框内"红像素占比" ≥ RED_SOLID —— 实测真红尺 0.537，误入红掩膜的杂散物多为 0.09~0.36；
#   ② 若黑尺候选也在，红尺长度不得短于其 RED_MIN_RATIO 倍 —— 图上确实存在"很红但很短"的
#      红色杂质（实测 red 占比 0.425~0.592、却只有 12~30px），会抢走正确的 234px 黑尺；
#      而同一图族的尺长稳定在数百 px，短块不可能是尺。
RED_SOLID = 0.40
RED_MIN_RATIO = 0.5
# 烧录标注是"画"上去的近纯黑覆盖层（实测线段 gray≈0~25），而组织自身的深色区多在 27 以上。
# 旧版用 gray<90 做掩膜，线段压到深色组织上时两者会经闭运算黏成一团（蝉蜕 B20132527 实测
# 黑掩膜连通域合并成 w/h=1.3 与 2.1 的两大块，被"细长"判据全部淘汰 → 右下 ROI 候选为空）。
INK_MAX = 30
# 端帽：线段两端各有竖向短笔画，实测竖游程 25px ≈ 8 倍线宽；组织纹理的竖游程多在 12~18
# 且逐行漂移 → 用"端帽成对 + 行间一致"两道约束把纹理噪声挡掉。
CAP_MIN = 12
# 线段最短像素长：实测图族真尺 233~373px；红色杂散物 9~66px。
MIN_BAR_PX = 80
# 行间一致性容差：真尺是 2~4 行几乎等起止的常值游程，纹理噪声的游程逐行漂移。
ROW_TOL = 4
# 线体粗细：真尺是一条细线，整段内部各列竖游程 ≈ 线宽（实测 3，偶有组织杂点穿过抬到 21）；
# 而深色组织纹理在整段内部的竖游程普遍 20~40（苦参 B20164003(3) 实测整段恒为 37）。
# 取内部 7 点竖游程的**中位数** ≤ THIN_MAX 作"这确实是一条线"的判据（中位数对偶发穿越点稳健）。
# 上限取 10：275 张真尺实测厚度只有两个值——3px（274 张）与 9px（连翘 20130915，粗墨烧录），
# 旧值 8 恰好把 9px 的真尺误杀、退回弱红路径量到 115px 假阳性；纹理块实测 20~40，余量仍足。
THIN_MAX = 10


def _elongated_components(mask, img_w, img_h, roi=None):
    """闭运算→连通域，筛细长候选（非全宽边框、长宽比高、面积适中）。返回候选 bbox 列表。

    roi=(x_min, y_min)：只保留外接矩形左上角落在该像素坐标之后（右下区域）的候选，
    用于黑尺的位置先验；roi=None 表示全图搜索（红尺用）。
    """
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(closed, 8)
    cands = []
    for i in range(1, n):
        x, y, w, h, area = stats[i, :5]
        if w < 8 or area < 60:                          # 太短/太小
            continue
        if w > img_w * 0.9:                              # 全宽边框伪影
            continue
        if h < 2 or w / max(h, 1) < 2.5:                 # 非细长（比例尺是长条）
            continue
        if roi and (x < roi[0] or y < roi[1]):            # 位置先验：只要右下区域
            continue
        cands.append((x, y, w, h))
    return cands


def _bar_px_in_bbox(mask, x, y, w, h):
    """候选块内取最长水平暗/亮游程作为线段像素长。"""
    roi = mask[y:y + h, x:x + w]
    if roi.size == 0:
        return 0
    best = 0
    for row in roi:
        run = cur = 0
        for v in row:
            cur = cur + 1 if v else 0
            if cur > best:
                best = cur
    return best


def _red_frac(img_bgr, bbox):
    """bbox 内"红像素占比"（R 显著高于 G/B 的像素比例），与红掩膜同一判据。"""
    x1, y1, x2, y2 = [int(v) for v in bbox]
    roi = img_bgr[y1:y2 + 1, x1:x2 + 1]
    if roi.size == 0:
        return 0.0
    b, g, r = cv2.split(roi)
    return float((r.astype(int) - np.maximum(g.astype(int), b.astype(int)) > 40).mean())


def _bbox_color(img_bgr, bbox):
    """判定 bbox 内线段真实颜色：红像素占比高 → red，否则 black。"""
    if _red_frac(img_bgr, bbox) > 0.05:
        return "red"
    x1, y1, x2, y2 = [int(v) for v in bbox]
    roi = img_bgr[y1:y2 + 1, x1:x2 + 1]
    if roi.size == 0:
        return "none"
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    return "black" if (gray < 90).mean() > 0.05 else "other"


def _vertical_extent(gray, x, y, win=18, ink_max=INK_MAX):
    """列 x 在 y±win 内的最长连续"近纯黑"竖游程（端帽 ≈15~30，细线 ≈2~4）。"""
    lo = max(0, y - win)
    hi = min(gray.shape[0], y + win + 1)
    col = gray[lo:hi, x] < ink_max
    best = cur = 0
    for v in col:
        cur = cur + 1 if v else 0
        if cur > best:
            best = cur
    return int(best)


def _cap_extent(gray, x, y, win_x=3):
    """端帽强度：列 x±win_x 内最大的竖游程。

    端帽是 ~2~3px 宽的竖笔画，而游程末端列会因 JPEG 模糊在 ±2px 抖动（苦参 17.jpg 实测
    同一端帽：行 1416 末端列竖游程 25，行 1415/1417 末端列却只有 4）→ 只在单一末端列
    查会漏检，故取小窗口最大值。
    """
    lo = max(0, x - win_x)
    hi = min(gray.shape[1] - 1, x + win_x)
    return max(_vertical_extent(gray, xx, y) for xx in range(lo, hi + 1))


def _is_thin_line(gray, px0, px1, py, thin_max=THIN_MAX):
    """线体判据：整段内部 7 点竖游程的**中位数**须接近线宽。

    真尺是细线（内部 3~7），深色组织纹理是厚块（内部 20~40）；取中位数可容忍偶发
    组织杂点穿过线段（蝉蜕 B20132527 内部 7 点中有 1 点为 21，中位数仍是 7）。
    """
    span = px1 - px0
    if span <= 0:
        return False
    vals = [_vertical_extent(gray, int(px0 + span * f), py)
            for f in (0.15, 0.27, 0.38, 0.5, 0.62, 0.73, 0.85)]
    return float(np.median(vals)) <= thin_max


def _scan_bar_lines(gray, y0, y1, x0, x1):
    """在 [y0,y1)×[x0,x1) 内找"长横线 + 两端端帽 + 行间一致 + 线体细"的线段 → (bar_px, bbox)。

    四层判据缺一不可（阈值见 INK_MAX / CAP_MIN / MIN_BAR_PX / ROW_TOL / THIN_MAX 注释）：
      ① 行内近纯黑连续游程 ≥ MIN_BAR_PX —— 烧录标注近纯黑，深色组织够不上这个阈值；
      ② 游程两端各有竖向笔画 ≥ CAP_MIN  —— 尺的端帽；纹理很难恰好两端成对；
      ③ 相邻行游程起止一致（≤ROW_TOL）  —— 尺是 2~4 行等宽粗线，纹理逐行漂移；
      ④ 线体中位数 ≤ THIN_MAX          —— 尺是细线，纹理是厚块（苦参 B20164003(3) 靠这层挡掉）。
    候选按长度降序逐个过检，取第一条同时满足四层的；都不满足返回 None。
    """
    H, W = gray.shape[:2]
    y0, y1 = max(0, int(y0)), min(H, int(y1))
    x0, x1 = max(1, int(x0)), min(W - 1, int(x1))
    if y1 - y0 < 2 or x1 - x0 < MIN_BAR_PX:
        return None
    sub = gray[y0:y1, x0:x1] < INK_MAX
    per_row = {}
    for i in range(sub.shape[0]):
        row = sub[i]
        j = 0
        while j < row.size:
            if not row[j]:
                j += 1
                continue
            s = j
            while j < row.size and row[j]:
                j += 1
            e = j - 1
            if e - s + 1 < MIN_BAR_PX:
                continue
            ly = y0 + i
            if _cap_extent(gray, x0 + s, ly) < CAP_MIN:
                continue
            if _cap_extent(gray, x0 + e, ly) < CAP_MIN:
                continue
            per_row.setdefault(i, []).append((s, e))
    # 行间一致性：相邻两行起止相近的游程算一组，同组取最长游程
    groups = []
    for i, runs in per_row.items():
        for s, e in runs:
            for s2, e2 in per_row.get(i + 1, ()):
                if abs(s - s2) <= ROW_TOL and abs(e - e2) <= ROW_TOL:
                    groups.append((e - s + 1, s, e, i))
                    break
    for bar_px, s, e, i in sorted(groups, key=lambda g: -g[0]):
        if _is_thin_line(gray, x0 + s, x0 + e, y0 + i):
            return bar_px, [x0 + s, y0 + i, x0 + e, y0 + i + 1]
    return None


def _find_um_text(img_bgr):
    """全分辨率 OCR 找烧录的 μm 文字 → {um, bbox, text, conf} / None。

    文字优先（text-first）：97 张实测样本里 95 张带这段文字，锚点几乎永远可用，
    比"先瞎找线段候选、再回头拿候选框去配文字"稳得多 —— 旧版正是后者翻的车
    （候选被图上方红色杂散物抢走，真正的文字落在距离窗之外 → bar_um=None）。
    兼容两种烧录风格：2048 图族"长度：20.00um"、3840 图族"20μm"。
    """
    res, _ = _get_ocr()(img_bgr)
    if not res:
        return None
    best = None
    for box, text, conf in res:
        if not re.search(r"(?:[μµ]|u)m", text, re.I):   # 必须带 μm/um 单位，否则只是普通数字
            continue
        m = _DIGIT.search(text)
        if not m:
            continue
        val = float(m.group(1))
        if not (1.0 <= val <= 1000):                    # μm 数值合理窗
            continue
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        bb = [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]
        if best is None or conf > best["conf"]:
            best = {"um": val, "bbox": bb, "text": text, "conf": float(conf)}
    return best


# 文字窄带的横向放宽（倍字宽）。实测（2026-09-20）：绝大多数图线段与文字左对齐、右端还略短于
# 文字（右溢出 −0.20~0.03 倍字宽），但 um=50 图族的尺长 579px、比文字宽得多，会向文字右侧多
# 伸出 1.06 倍字宽（蝉蜕 1.jpg：文字 x[1321,1597] 宽 276、尺 x[1311,1889]）——原 0.4 倍系数
# 把该游程截断在窗口边缘，端帽判据因此不过，只能靠右下 ROI 兜底。故左取 0.5、右取 1.2。
BAR_ZONE_XL = 0.5
BAR_ZONE_XR = 1.2


def _bar_zones(tb):
    """由文字锚点推出线段可能所在的两条窄带：正上方为主、正下方兜底。

    实测线段紧贴文字上沿（蝉蜕 B20132527：尺 y1339-1341、文字 y1347-1398，间距仅 6px），
    故主带取文字上方 2.5 倍字高；横向按 BAR_ZONE_XL/XR 放宽，越界由 _scan_bar_lines 兜。
    下界只留 1px：原按 0.08 倍字高留空隙（≈4px），而蝉蜕 4.jpg 的尺贴在 y1391~1393、
    文字自 y1396 起，空隙恰好把尺下半行切在窗口外 → "行间一致"配不成对而漏检。
    """
    x1, y1, x2, y2 = tb["bbox"]
    th, tw = y2 - y1, x2 - x1
    bx0, bx1 = x1 - int(BAR_ZONE_XL * tw), x2 + int(BAR_ZONE_XR * tw)
    return [(y1 - int(2.5 * th), y1 - 1, bx0, bx1),             # 主：文字正上方
            (y2 + 1, y2 + int(1.5 * th), bx0, bx1)]             # 辅：文字正下方


def engine_a(img_bgr, read_um=True):
    """A 引擎：文字优先定 μm + 严格阈值/端帽定长；红/黑形态学兜底。
    返回 {bar_px, scale_type, bbox, bar_um, picked_by}。

    选序（2026-09-20 二次实测驱动）：
      ① **文字锚定**（`_find_um_text` 拿 μm+锚点，`_scan_bar_lines` 扫文字邻域窄带）→ 采用：
         97 张样本 95 张有烧录文字，锚点最稳；且同一趟 OCR 就给出 μm，不再依赖
         "候选框附近找文字"那套脆弱逻辑；
      ② 否则 **合格红尺**（红全图候选且满足 `RED_SOLID` + `RED_MIN_RATIO`）→ 采用红尺：
         红尺位置完全不固定，只能靠颜色与长度自信度判定；
      ③ 否则 **黑尺**（严格阈值 + 端帽，右下 ROI）→ 采用：黑尺位置有强先验；
      ④ 都没有 → 返回 none。旧版此处会退回"弱红候选"（实测红占比仅 0.09~0.36、
         长 9~66px 的杂散物），量出的错值会污染下游 → 宁判 manual。
    """
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    b, g, r = cv2.split(img_bgr)
    red_mask = (r.astype(int) - np.maximum(g.astype(int), b.astype(int)) > 40).astype(np.uint8) * 255

    def _red_pick():
        """全图最宽的合格红候选 → (bar_px, bbox) / None。"""
        cands = _elongated_components(red_mask, w, h, roi=None)
        if not cands:
            return None
        x, y, cw, ch = max(cands, key=lambda c: c[2])   # 取最宽候选
        bar_px = _bar_px_in_bbox(red_mask, x, y, cw, ch)
        return (bar_px, [int(x), int(y), int(x + cw), int(y + ch)]) if bar_px >= 8 else None

    def _pack(picked, bar_px, bbox, um=None):
        out = {"bar_px": bar_px, "scale_type": _bbox_color(img_bgr, bbox),
               "bbox": bbox, "bar_um": um, "picked_by": picked}
        if read_um and out["bar_um"] is None:
            out["bar_um"] = ocr_read_um(img_bgr, bbox)
        return out

    # ① 文字锚定：OCR 先给 μm 与锚点，再在锚点邻域用严格阈值 + 端帽定线长
    tb = _find_um_text(img_bgr)
    if tb:
        for (zy0, zy1, zx0, zx1) in _bar_zones(tb):
            hit = _scan_bar_lines(gray, zy0, zy1, zx0, zx1)
            if hit:
                return _pack("text", hit[0], hit[1], um=tb["um"])

    # ②/③ 兜底：合格红尺优先（位置不固定），再试右下 ROI 黑尺
    red_c = _red_pick()
    red_ok = red_c is not None and _red_frac(img_bgr, red_c[1]) >= RED_SOLID
    black_hit = _scan_bar_lines(gray, h * BLACK_ROI_Y, h, w * BLACK_ROI_X, w)
    if red_ok and black_hit and red_c[0] < RED_MIN_RATIO * black_hit[0]:
        red_ok = False      # 很红但很短 → 红色杂质，让位黑尺（详见 RED_MIN_RATIO 注释）
    if red_ok:
        return _pack("red", red_c[0], red_c[1])
    if black_hit:
        return _pack("black", black_hit[0], black_hit[1])
    # 没量到尺也要把已读到的烧录 μm 带出去：cross_validate 的 OK-low（族先验兜底）
    # 正是"μm 有了、线没量到"的场景——这里丢成 None 的话 OK-low 永远不会触发。
    return {"bar_px": 0, "scale_type": "none", "bbox": None,
            "bar_um": (tb["um"] if tb else None), "picked_by": "none"}


# ---------------- B 引擎：VLM 读长度+数字 ----------------
def engine_b(img_bgr, cfg_vlm, api_key, max_side):
    img = img_bgr if max(img_bgr.shape[:2]) <= max_side else \
        cv2.resize(img_bgr, None, fx=max_side / max(img_bgr.shape[:2]),
                   fy=max_side / max(img_bgr.shape[:2]), interpolation=cv2.INTER_AREA)
    b64 = vlm_api.encode_jpeg_b64(img, max_side)[0]
    text, err = vlm_api.call_vlm(cfg_vlm, api_key, b64, B_PROMPT)
    if text is None:
        return {"error": err}
    try:
        parsed = vlm_api.parse_vlm_json(text)
    except Exception as e:
        return {"error": f"parse:{e}"}
    W = img_bgr.shape[1]
    pres = bool(parsed.get("present", False))
    bbox = parsed.get("bbox_2d")
    bar_px = 0
    if pres and bbox and len(bbox) == 4:
        bar_px = round(max(0.0, float(bbox[2]) - float(bbox[0])) / 1000.0 * W)
    um = parsed.get("um_label")
    return {
        "present": pres,
        "color": parsed.get("color", ""),
        "bbox": [round(float(v), 0) for v in bbox] if bbox else None,
        "bar_px": bar_px,
        "bar_um": um,
        "confidence": parsed.get("confidence", ""),
    }


# ---------------- 交叉验证 ----------------
SOURCE = "duo(A文字优先+OCR / B=VLM, 交叉验证)"
# px_per_um 物理合理窗（2026-09-20 实测 277 张：合法值分布 [2.91, 11.65]，2.91 为黄芪
# 低倍率真值、已目视验证）。窗口留一个量级余量：拦得住退化错值（如 9px 杂散块量出的
# ppm≈0.4），又不误杀未来可能出现的更低倍率图。注意这是保险网不是修复手段——离群值
# 的根因多是 A 选错块，先修选块再看本闸门。
PPM_MIN = 0.5
PPM_MAX = 50.0


def cross_validate(name, ipath, A, B, tol, px_fam=None, img_w=None, px_ok_set=None):
    """综合 A/B 出判定，返回 row dict（逐图对照报告的一行）。

    判定表（2026-09-20 重做，2026-09-28 补 OK-consensus；grade 编码进 status 便于下游零改动穿透）：
      um_A_ok : A 的 μm 有效（数值窗内）——本地 OCR 是 μm 的权威来源
      conflict: A/B 都读到 μm 但不相等 —— 真冲突，必须人工
      B_ok   : B_px>0 且 ≤ min(2.5×px_A, 0.5W) —— B 的框惯性过框（把端帽包进去，
               实测 B_px/px_A=1.04~1.18），故只设单边上界
      px_A_ok: px_A≥MIN_BAR_PX 且 ≤0.5W（尺不可能横过半幅图，实测最长真尺 0.286W）
               且 ppm_A ∈ [PPM_MIN, PPM_MAX]
      px_ok_set: **已被旁证的 bar_px 离散集**（由 ls_common.confirmed_bar_px 从精确 OK
               样本构建）——B 未证实时的第二独立证据

      ¬um_A_ok                    → manual / n/a（保持旧语义：px_A≥8→manual，否则 n/a）
      conflict                    → manual（不出值）
      um_A_ok & px_A_ok & B_ok    → OK            px_A/um    （grade=high，双读一致）
      um_A_ok & px_A_ok & ¬B_ok & px_A∈px_ok_set
                                  → OK-consensus  px_A/um    （grade=consensus，见下）
      um_A_ok & px_A_ok & ¬B_ok   → OK-medium     px_A/um    （B 未证实且无旁证）
      um_A_ok & ¬px_A_ok & px_fam → OK-low        px_fam/um  （A 错块/未量到，族先验优先——
                                                   族内 ≥3 张精确 OK 的众数，比单次 B 框准）
      um_A_ok & ¬px_A_ok & B_usable → OK-low      px_B/um    （无族先验时宁用不准确的 B 出值，
                                                   也不卡 manual；B 须自身过物理窗）
      其余（B 也无效）             → manual（px_A≥8）/ n/a     ← 拦 ppm 出窗的错值

    OK-consensus 的依据（2026-09-28 实测 422 份）：`bar_px` 是 A 引擎的**离散可重复**读数——
    精确 OK 样本里 97% 只取 233/311 两个值（对应两个镜头档），且算式 px_per_um=bar_px/bar_um
    严格成立。实测 119 份 OK-medium 中 118 份的 bar_px 落在 OK 已验证集合内，即"B 没证实"
    只是 B 引擎噪声，A 的读数与同批 OK 样本同值。故当 bar_px 已有 ≥min_n 张精确 OK 旁证时
    判 OK-consensus（**不重跑**即可）；只有 bar_px 尚无旁证的（如实测 1 例 331）保持 medium。

    防自我放大：px_ok_set 只由 **精确 OK** 构成，OK-consensus 自身不回灌
    （见 ls_common.confirmed_bar_px：只收 status=="OK"）。

    status 语义：OK=双读一致；OK-consensus=A 可信且有跨图旁证；两者下游都视为可信
    （TRUSTED_STATUSES）；OK-medium/OK-low 会被 criteria_hint_for 自动追加 μm 仅供参考提示。
    px_ok_set 缺省 None ⇒ 行为与旧版完全一致（向后兼容，老调用方无需改）。
    """
    Ap, Bp = A["bar_px"], B["bar_px"]
    Au = A.get("bar_um"); Bu = B.get("bar_um")
    um_A_ok = Au is not None and 1.0 <= float(Au) <= 1000.0
    # um 交叉（仅供报告）：两引擎都读到且相等
    um_ok = um_A_ok and Bu is not None and abs(float(Au) - float(Bu)) < 1e-9
    conflict = um_A_ok and Bu is not None and not um_ok
    ppm = (Ap / float(Au)) if (um_A_ok and Ap) else None
    px_A_ok = (Ap >= MIN_BAR_PX
               and (img_w is None or Ap <= 0.5 * img_w)
               and ppm is not None and PPM_MIN <= ppm <= PPM_MAX)
    b_ok = (Bp > 0 and Ap > 0 and Bp <= 2.5 * Ap
            and (img_w is None or Bp <= 0.5 * img_w))
    # B 独立可用性（不依赖 px_A）：A 错块/未量到时的兜底裁判。B 是框宽、惯性偏大，
    # 故不设下限比对，只要求自身过物理窗（够长、不出半幅图、ppm 合理）。
    b_usable = (Bp >= MIN_BAR_PX
                and (img_w is None or Bp <= 0.5 * img_w)
                and um_A_ok and PPM_MIN <= Bp / float(Au) <= PPM_MAX)
    px_per_um = ""
    grade = ""
    if not um_A_ok or conflict:
        status = "manual" if Ap >= 8 else "n/a"
    elif px_A_ok:
        px_per_um = round(ppm, 3)
        if b_ok:
            status, grade = STATUS_OK, "high"
        elif px_ok_set and Ap in px_ok_set:
            status, grade = STATUS_OK_CONSENSUS, "consensus"
        else:
            status, grade = "OK-medium", "medium"
    elif px_fam and px_fam >= MIN_BAR_PX:
        # 族先验（族内 ≥3 张精确 OK 的众数）比单次 B 框更准，优先于 B
        px_per_um = round(px_fam / float(Au), 3)
        status, grade = "OK-low", "low"
    elif b_usable:
        # A 错块/未量到但 um 已读到、B 框过物理窗 → 宁用不准确的 B 出值（grade=low），
        # 也不把整图打成 manual 卡死下游（2026-09-21 实测 4 例 A 错块均由此救回）。
        px_per_um = round(Bp / float(Au), 3)
        status, grade = "OK-low", "low"
    else:
        status = "manual" if Ap >= 8 else "n/a"
    diff = (abs(Ap - Bp) / max(Ap, Bp)) if (Ap and Bp) else None
    row = dict(
        image=name, A_bar_px=Ap, A_um=Au, A_color=A["scale_type"],
        B_bar_px=Bp, B_um=Bu, B_color=B.get("color", ""),
        px_delay_pct=(f"{diff*100:.1f}" if diff is not None else ""),
        um_agree=um_ok, px_per_um=px_per_um, status=status, grade=grade,
    )
    return row


def measure_img_px_per_um(img, name, cfg=None, px_fam=None, px_ok_set=None):
    """双引擎标定（**已读入内存的 BGR 图**）→ 结果 dict，见 `measure_px_per_um`。

    供已持有图像数组的调用方复用，避免 tif 大图二次读盘（vlm_prelabel / detect_sample）。
    px_fam：同目录族先验（族内 ≥3 张精确 OK 的 bar_px 众数），A 错块/未量到但 um 已读到时
    以 px_fam/um 出 OK-low；调用方（ls_common.family_bar_px）负责收集，此处只消费。
    px_ok_set：已旁证的 bar_px 集（ls_common.confirmed_bar_px），B 未证实时出 OK-consensus；
    不传（None）则行为与旧版一致。
    """
    cfg = cfg or load_config()
    if img is None:
        return {"px_per_um": None, "bar_px_cv": 0, "bar_px_vlm": None, "bar_um": None,
                "source": SOURCE, "status": "n/a", "note": "图像读取失败"}
    api_key = resolve_key(cfg)
    A = engine_a(img)                                   # 本地：文字优先定 μm + 端帽定长
    B = engine_b(img, cfg["vlm"], api_key, VLM_MAX_SIDE)
    if "error" in B:                                    # VLM 失败 → 仅 A（本地OCR）兜底
        B = {"bar_px": 0, "bar_um": None, "color": "", "confidence": ""}
    row = cross_validate(name, "", A, B, 0.10, px_fam=px_fam, img_w=img.shape[1],
                         px_ok_set=px_ok_set)

    def _f(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    # note 里写明"为什么可信/不可信"的旁证来源，便于事后审计（prelabel 只留这一行线索）
    note = f"pxΔ={row.get('px_delay_pct','')}% um_agree={row.get('um_agree')}"
    if row["status"] == STATUS_OK_CONSENSUS:
        note += f"；bar_px={A['bar_px']} 已由同批精确 OK 样本旁证（B 未证实）"
    if px_fam:
        note += f" px_fam={px_fam}"
    return {"px_per_um": _f(row.get("px_per_um")),
            "bar_px_cv": A["bar_px"], "bar_px_vlm": B.get("bar_px"),
            "bar_um": A["bar_um"] or B.get("bar_um"),
            "source": SOURCE, "status": row["status"] or "", "grade": row.get("grade", ""),
            "note": note,
            "A": {"bar_px": A["bar_px"], "bar_um": A["bar_um"], "color": A["scale_type"]},
            "B": {"bar_px": B.get("bar_px"), "bar_um": B.get("bar_um"),
                  "color": B.get("color", "")}}


def measure_px_per_um(image_path, cfg=None, px_fam=None, px_ok_set=None):
    """逐图比例尺双引擎标定 —— **公共 API**（检测阶段与 UI 共用）。

    返回 dict：{px_per_um, bar_px_cv, bar_px_vlm, bar_um, source, status, grade, note, A, B}
    status ∈ OK（双读一致）| OK-consensus（A 可信且有跨图旁证，2026-09-28 补）|
    OK-medium（A 可信但 B 未证实且无旁证）| OK-low（族先验/B 兜底）| manual | n/a。
    可信与否请用 ls_common.is_trusted_status()，勿手拼字符串比较。

    2026-09-20 起这是全项目 px_per_um 的生产方式（离线 scale_table.csv 已停用）：
    检测阶段（vlm_prelabel / detect_sample）把结果写入 prelabel JSON，随标注流转到
    LS 任务 data 与分割交付清单，几何测量只认交付清单里的这份值。
    """
    if not os.path.isfile(str(image_path)):
        return {"px_per_um": None, "bar_px_cv": 0, "bar_px_vlm": None, "bar_um": None,
                "source": SOURCE, "status": "n/a", "note": f"文件不存在: {image_path}"}
    return measure_img_px_per_um(imread_cn(image_path), os.path.basename(str(image_path)), cfg,
                                 px_fam=px_fam, px_ok_set=px_ok_set)


# ---------------- 主流程 ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tol", type=float, default=0.10, help="A 与 B 的 bar_px 一致性容差(相对, 仅提示)")
    ap.add_argument("--max-side", type=int, default=VLM_MAX_SIDE)
    ap.add_argument("--images", default="", help="逗号分隔的待测图路径（必填；见模块头注）")
    a = ap.parse_args()

    targets = [p for p in [x.strip() for x in a.images.split(",")] if p]
    if not targets:
        ap.error("--images 必填：逗号分隔的待测图路径（原默认取 archive/demo_output/input/ 已移除）")
    names = [f"img{i+1}" for i in range(len(targets))]

    cfg = load_config()
    cfg_vlm = cfg["vlm"]
    api_key = resolve_key(cfg)

    rows = []
    print(f"{'img':6}{'A_px':>6}{'A_um':>6}{'A_col':>6}{'B_px':>6}{'B_um':>6}{'B_col':>6}"
          f"{'pxΔ%':>6}{'um_ag':>6}{'px/um':>7}  status")
    for name, p in zip(names, targets):
        img = imread_cn(p)
        if img is None:
            print(f"{name}: 读取失败 {p}"); continue
        # A 引擎（本地：形态学 + OCR）
        A = engine_a(img)
        # B 引擎（VLM）
        B = engine_b(img, cfg_vlm, api_key, a.max_side)
        if "error" in B:
            print(f"{name}:B引擎失败 {B['error']}"); continue
        row = cross_validate(name, p, A, B, a.tol, img_w=img.shape[1])
        rows.append(row)
        print(f"{name:6}{row['A_bar_px'] or 0:>6}{str(row['A_um'] or ''):>6}"
              f"{row['A_color']:>6}{row['B_bar_px'] or 0:>6}{str(row['B_um'] or ''):>6}"
              f"{row['B_color']:>6}{str(row['px_delay_pct'] or ''):>6}{str(row['um_agree']):>6}"
              f"{str(row['px_per_um'] or ''):>7}  {row['status']}")

    # 报告（仅供目检对照；线上 px_per_um 走 measure_px_per_um()，不再落任何离线标定表）
    report = os.path.join(OUT, "scalebar_duo_report.csv")
    header = ["image", "A_bar_px", "A_um", "A_color", "B_bar_px", "B_um", "B_color",
              "px_delay_pct", "um_agree", "px_per_um", "status", "grade"]
    with open(report, "w", newline="", encoding="utf-8-sig") as f:
        wcsv = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
        wcsv.writeheader()
        wcsv.writerows(rows)
    print(f"\n报告 → {report}")
    print("（2026-09-20 起不再回写 scale_table.csv；线上取值请调 measure_px_per_um()）")


if __name__ == "__main__":
    main()