"""单crop类别复核worker：在受控子进程内复用检测侧 classify_crop。

为什么存在：融合链路此前的工具集只有检测/分割/测量/重裁，无法在「已观察到的存在特征」与
「crop 类别」相左时复核类别。本入口只做一件事：读入一个已绑定crop，调一次分类，写结构化结果。

独立进程而非进程内调用的理由与 detect_region 相同：图像读取/编码/HTTP 失败不会拖垮主循环；
并且 `worker_bootstrap` 的HTTP守卫会把这次请求记进同一份资源账本（不另立第二套预算）。
"""
import argparse
from pathlib import Path

from TCM_Fusion.lib.storage import IntegrationError, read_json, write_json


def geometry_context(tight_box, width, height):
    """与 `detect_sample._geom_ctx` 同口径的纯几何先验。

    该函数在入口脚本内，按仓规「入口目录不得被 import」不能复用；这里只重述一段提示词文本，
    不涉及判定逻辑。若日后要改口径，两处必须一起改（或先把它提到 lib 里）。
    """
    x1, y1, x2, y2 = [float(v) for v in tight_box]
    w, h = max(1e-6, x2 - x1), max(1e-6, y2 - y1)
    ar = w / h
    shape = "细长条形" if (ar >= 3 or ar <= 1 / 3) else ("近方形" if 0.8 <= ar <= 1.25 else "长方形")
    return (f"- 候选框长宽比 {ar:.2f}（{shape}），原图内尺寸约 {w:.0f}×{h:.0f} 像素"
            f"（占画面 {100.0 * w * h / max(1.0, width * height):.1f}%）")


def main(argv=None):
    parser = argparse.ArgumentParser(description="单crop类别复核")
    parser.add_argument('--context', required=True)
    ctx = read_json(parser.parse_args(argv).context)
    from ls_common import imread_cn, load_config, resolve_key
    from classify_crop import classify_crop
    crop = imread_cn(ctx['crop_png'])
    if crop is None:
        raise IntegrationError('CROP_INVALID', '无法读取绑定crop')
    if 'tight_box_local' in ctx:
        x1, y1, x2, y2 = map(int, ctx['tight_box_local'])
        h, w = crop.shape[:2]
        if not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h):
            raise IntegrationError('CROP_INVALID', '身份核验框越出裁图')
        crop = crop[y1:y2, x1:x2]
    from PIL import Image
    with Image.open(ctx['source']) as handle:
        width, height = handle.size
    config = load_config()
    key = resolve_key(config)
    context = geometry_context(ctx['tight_box'], width, height) if ctx.get('tight_box') else ""
    hints = str(ctx.get('feature_hint') or '').strip()
    if hints:
        context = (context + "\n" + hints).strip()
    value = classify_crop(crop, config['vlm'], key, ctx=context)
    if not isinstance(value, dict):
        raise IntegrationError('CLASSIFY_OUTPUT_INVALID', '分类返回不是结构化对象')
    out = Path(ctx['output'])
    out.mkdir(parents=True, exist_ok=True)
    raw_conf = value.get('confidence')
    try:
        numeric_conf = float(raw_conf)
    except (TypeError, ValueError):
        numeric_conf = {'high': 0.90, 'medium': 0.65, 'low': 0.30}.get(str(raw_conf).strip().lower(), 0.0)
    value['confidence_score'] = round(max(0.0, min(1.0, numeric_conf)), 4)
    value['confidence_scale'] = 'glm_high_0.90_medium_0.65_low_0.30'
    write_json(out / 'classification.json', {
        **value,
        'crop_id': ctx.get('crop_id'),
        'crop_png': str(Path(ctx['crop_png']).resolve()),
        'source': str(Path(ctx['source']).resolve()),
    })
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
