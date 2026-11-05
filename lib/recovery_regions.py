"""Local native-resolution region proposals. Scores are search cues only."""


def propose_regions(image):
    import cv2
    import numpy as np

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    side = min(384, height // 2, width // 2)
    if side < 48:
        return []
    edges = cv2.Canny(gray, 40, 120)
    candidates = []
    stride = side // 2
    for y in sorted({*range(0, height - side + 1, stride), height - side}):
        for x in sorted({*range(0, width - side + 1, stride), width - side}):
            tile = edges[y:y + side, x:x + side]
            lines = cv2.HoughLinesP(tile, 1, np.pi / 180, 22,
                                    minLineLength=side // 16, maxLineGap=4)
            if lines is None:
                continue
            segments = lines[:, 0].astype(float)
            lengths = np.hypot(segments[:, 2] - segments[:, 0], segments[:, 3] - segments[:, 1])
            # 长贯穿线常来自比例尺或图像边缘，小目标线索只按短线段计分。
            lengths = lengths[lengths <= side / 3]
            if not lengths.size:
                continue
            score = float(np.sort(lengths)[-8:].sum() / side)
            candidates.append({'box': [x, y, x + side, y + side], 'line_score': round(score, 3)})
    from geom import iou_box
    selected = []
    for item in sorted(candidates, key=lambda row: (-row['line_score'], row['box'])):
        if all(iou_box(item['box'], other['box']) < .2 for other in selected):
            selected.append(item)
        if len(selected) == 3:
            break
    return selected
