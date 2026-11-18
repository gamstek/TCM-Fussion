"""将原图像素宽高划分为覆盖全幅的网格，输出数量受限的原图坐标框。GLM 补扫据此生成分块动作。网格只使用长宽比和分块上限，旧 tile 与 overlap 参数仅兼容历史调用。

主要定义：
- uniform_tiles: 按图像长宽比选择网格，返回不超过 max_tiles 个覆盖整图的区块；4 个 tile 时即 2×2 分区。

关联：
- 调用方 「lib/workflow/workflow_backend.py」、「tests/vision/test_image_tiling.py」。
"""
from __future__ import annotations


def uniform_tiles(width, height, *, tile=1024, overlap=0.15, max_tiles=4):
    """把全图切成不超过 max_tiles 个、覆盖整图的均匀区块。

    4 个 tile 时是 2×2 分区，而不是从 3×2 网格截断 4 个角；
    这样不会留下中间或边缘空洞。tile 和 overlap 为旧接口保留。
    均匀分区仅由图像尺寸与 max_tiles 决定。
    """
    width = int(width)
    height = int(height)
    limit = max(1, int(max_tiles))
    if width <= 0 or height <= 0:
        return []
    # 选择接近图像长宽比的网格，并保证 rows*cols <= limit。
    cols = max(1, min(limit, int(round((limit * width / max(1, height)) ** 0.5))))
    rows = max(1, min(limit, (limit + cols - 1) // cols))
    while cols * rows > limit:
        if cols >= rows and cols > 1:
            cols -= 1
        elif rows > 1:
            rows -= 1
        else:
            break
    xs = [round(i * width / cols) for i in range(cols)] + [width]
    ys = [round(i * height / rows) for i in range(rows)] + [height]
    boxes = []
    for row in range(rows):
        for col in range(cols):
            x1, x2 = xs[col], xs[col + 1]
            y1, y2 = ys[row], ys[row + 1]
            if x2 - x1 >= 1 and y2 - y1 >= 1:
                boxes.append([x1, y1, x2, y2])
    return boxes[:limit]

