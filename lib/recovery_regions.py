"""Local native-resolution region proposals. Scores are search cues only.

2026-11-05：原实现只保留 3 个最高线分窗口，龙胆 5 视野复测中 3 个窗口与人工
针晶框的 IoU 全为 0。现改为覆盖全图的 50% 重叠网格，并返回窗口覆盖审计；线分只
用于排序，不再用于丢弃窗口。
"""


def _grid(w, h):
    # 1024 窗口 + 50% 步长：2048×1536 为 3×2=6 个窗口，边界目标也至少完整落在一个窗口内。
    side = min(1024, max(256, (min(w, h) * 2) // 3))
    side = min(side, w, h)
    step = max(64, side // 2)
    xs = sorted({*range(0, max(1, w - side + 1), step), w - side})
    ys = sorted({*range(0, max(1, h - side + 1), step), h - side})
    return side, xs, ys


def propose_regions(image, *, with_audit=False):
    import cv2
    import numpy as np

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    side, xs, ys = _grid(width, height)
    if side < 48:
        return {"regions": [], "audit": {"image_size": [width, height], "window_side": side,
                                          "rows": 0, "cols": 0, "window_count": 0,
                                          "covered_bbox": [0, 0, 0, 0],
                                          "covered_area_ratio": 0.0,
                                          "policy": "full_grid"}} if with_audit else []
    edges = cv2.Canny(gray, 40, 120)
    regions = []
    for row, y in enumerate(ys):
        for col, x in enumerate(xs):
            tile = edges[y:y + side, x:x + side]
            lines = cv2.HoughLinesP(tile, 1, np.pi / 180, 22,
                                    minLineLength=side // 16, maxLineGap=4)
            score = 0.0
            if lines is not None:
                segments = lines[:, 0].astype(float)
                lengths = np.hypot(segments[:, 2] - segments[:, 0], segments[:, 3] - segments[:, 1])
                # 长贯穿线常来自比例尺或图像边缘，小目标线索只按短线段计分。
                lengths = lengths[lengths <= side / 3]
                if lengths.size:
                    score = float(np.sort(lengths)[-8:].sum() / side)
            regions.append({"box": [int(x), int(y), int(x + side), int(y + side)],
                            "line_score": round(score, 3), "row": row, "col": col})
    regions.sort(key=lambda item: (-item["line_score"], item["row"], item["col"]))
    audit = {"image_size": [width, height], "window_side": side,
             "rows": len(ys), "cols": len(xs), "window_count": len(regions),
             "covered_bbox": [0, 0, width, height], "covered_area_ratio": 1.0,
             "policy": "full_grid", "selection": "all_windows_line_score_order"}
    return {"regions": regions, "audit": audit} if with_audit else regions
