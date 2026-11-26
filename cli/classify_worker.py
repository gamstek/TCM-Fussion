"""单 crop 类别复核 worker：在受控子进程内复用检测侧 `classify_crop`，从项目根以 `-m` 启动。

读取绑定裁图，对当前实例执行身份复核，或按规划器选择观察可见细节。两种请求各有自己的回执。

独立进程而非进程内调用的理由与 detect_region 相同：图像读取/编码/HTTP 失败不会拖垮主循环；并且 `worker_bootstrap` 的 HTTP 守卫会把这次请求记进同一份资源账本（不另立第二套预算）。YOLO crop 复核只是辅助，不替代 GLM。

主要定义：
- main: 解析 `--context`，读取绑定 crop（可选 `tight_box_local` 二次裁剪），调用一次分类并写 `classification.json`。
- 视觉身份复核：用已绑定的对照计划准备同图幅输入，校验模型报告的可见结构。
- inspect_crop 使用目标区原有像素，只记录实见、位置和可读性。
- geometry_context: 与 `detect_sample._geom_ctx` 同口径的纯几何提示文本（长宽比、尺寸、占画面比例），只影响提示词，不参与判定。
- 模块级约束: 文字型 confidence 按 glm high=0.90 / medium=0.65 / low=0.30 折算为 `confidence_score`。

关联：
- 直接依赖 「lib/runtime/storage.py」、「lib/vision/identity_review.py」。
- 从项目根通过 python -m 启动。
"""
import argparse
from pathlib import Path

from TCM_Fusion.lib.runtime.storage import IntegrationError, read_json, write_json


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
    brief = ctx.get('visual_brief')
    if ctx.get('tool') == 'inspect_crop':
        from vlm_api import call_vlm, encode_jpeg_b64, parse_vlm_json
        from TCM_Fusion.lib.vision.visual_observation import prompt_suffix, receipt, record
        prompt = '观察这张显微裁图。描述目标区内能辨认的结构及其位置。只返回JSON。'
        prompt += prompt_suffix(brief, open_observation=True)
        b64, _ = encode_jpeg_b64(crop, max_side=1024)
        answer, error = call_vlm({**config['vlm'], 'max_retries': 1}, key, b64, prompt)
        if error or answer is None:
            raise IntegrationError('CROP_OBSERVATION_FAILED', 'GLM 裁图观察请求失败')
        try:
            report = receipt(brief, parse_vlm_json(answer))
        except (ValueError, TypeError):
            report = receipt(brief, {})
        if report['status'] != 'reported':
            raise IntegrationError('MODEL_FIELD_INVALID', '裁图观察缺少可用回执')
        record(Path(ctx['output']), report)
        return 0
    context = geometry_context(ctx['tight_box'], width, height) if ctx.get('tight_box') else ""
    review = ctx.get('identity_review')
    if review is None:
        hints = str(ctx.get('feature_hint') or '').strip()
        if hints:
            context = (context + "\n" + hints).strip()
        value = classify_crop(crop, config['vlm'], key, ctx=context, **({'visual_brief': brief} if brief else {}))
    else:
        from TCM_Fusion.lib.vision.identity_review import finish_review, prepare_review
        value = classify_crop(crop, config['vlm'], key, **({'visual_brief': brief} if brief else {}),
                              review_input=lambda native, baseline: prepare_review(native, baseline, review, context))
    if not isinstance(value, dict):
        raise IntegrationError('CLASSIFY_OUTPUT_INVALID', '分类返回不是结构化对象')
    if brief and value.get('error'):
        raise IntegrationError('CLASSIFY_FAILED', 'GLM 未返回有效的身份复核')
    if review is not None:
        value = finish_review(value)
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
