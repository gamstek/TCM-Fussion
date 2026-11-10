"""图片示例审计的公共部分：contact sheet、分数规范化。

只做旁路证据；不参与 rule decide，不阻断 det_score>=0.8 的 mask/measure。
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def _thumb(path, box):
    image = Image.open(path).convert('RGB')
    image.thumbnail(box)
    return image


def build_contact_sheet(target, positives, negatives, output, *, target_label='TARGET'):
    """目标 + 正/负示例拼图；示例数量由调用方控制。"""
    cells = [('target', target)] + [('positive', p) for p in positives] + [('negative', n) for n in negatives]
    if not cells:
        raise ValueError('audit contact sheet requires at least one image')
    cell_w, cell_h = 300, 230
    cols = min(5, len(cells))
    rows = (len(cells) + cols - 1) // cols
    sheet = Image.new('RGB', (cell_w * cols, (cell_h + 28) * rows), 'white')
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype('/Windows/Fonts/msyh.ttc', 15)
    except OSError:
        font = ImageFont.load_default()
    for index, (kind, path) in enumerate(cells):
        x = (index % cols) * cell_w
        y = (index // cols) * (cell_h + 28)
        try:
            image = _thumb(path, (cell_w - 16, cell_h - 30))
        except Exception:
            image = Image.new('RGB', (100, 80), 'white')
        sheet.paste(image, (x + (cell_w - image.width) // 2, y + 4))
        if kind == 'target':
            label = f'TARGET: {target_label}'
        elif kind == 'positive':
            label = f'P{index}: positive'
        else:
            label = f'N{index}: negative'
        draw.text((x + 6, y + cell_h - 4), label, fill=(150, 25, 25), font=font)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, quality=92)
    return output


def normalize_audit(parsed, *, target_id=None, positive_count=0, negative_count=0, feature_id='needle_crystal'):
    """把 VLM JSON 规范成稳定 schema；不在这里做通过/拒绝。"""
    def number(value, default=0.0):
        try:
            return max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            return default
    value = parsed if isinstance(parsed, dict) else {}
    target = value.get('target_score', value.get('needle_score'))
    needle = number(target)
    negative = number(value.get('negative_similarity'))
    margin = value.get('negative_margin')
    try:
        margin = float(margin)
    except (TypeError, ValueError):
        margin = needle - negative
    result = {
        'schema': 'image-exemplar-audit/v2',
        'target_id': target_id,
        'feature_id': str(feature_id or value.get('feature_id') or 'unknown'),
        'target_score': round(needle, 4),
        'negative_similarity': round(negative, 4),
        'negative_margin': round(max(-1.0, min(1.0, margin)), 4),
        'audit_label': str(value.get('audit_label') or 'unknown')[:80],
        'confidence': str(value.get('confidence') or 'low')[:20],
        'reason': str(value.get('reason') or '')[:500],
        'positive_examples': int(positive_count),
        'negative_examples': int(negative_count),
        'review_status': 'auto_unreviewed',
        'non_blocking': True,
    }
    # 兼容旧针晶消费方。
    if result['feature_id'] == 'needle_crystal':
        result['needle_score'] = result['target_score']
    return result
