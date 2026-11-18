"""图片示例审计 worker：拼 contact sheet 并发起一次 VLM 旁路判断，从项目根以 `-m` 启动。

不是 `classify_target` 的替代；输出只写 audit 工件，不参与 evidence/decide。

主要定义：
- main: 读取 `--context`，校验目标 crop 与正负例是否齐全，构建对照图后请求 VLM 并写规范化审计结果。

关联：
- 直接依赖 「lib/runtime/storage.py」、「lib/vision/image_exemplar_audit.py」。
- 从项目根通过 python -m 启动。
"""
import argparse
from pathlib import Path

from PIL import Image
import numpy as np

from TCM_Fusion.lib.vision.image_exemplar_audit import build_contact_sheet, normalize_audit
from TCM_Fusion.lib.runtime.storage import IntegrationError, read_json, write_json


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--context', required=True)
    ctx = read_json(parser.parse_args(argv).context)
    target = Path(ctx['target'])
    positives = [Path(x) for x in ctx.get('positives') or []]
    negatives = [Path(x) for x in ctx.get('negatives') or []]
    if not target.is_file():
        raise IntegrationError('AUDIT_TARGET_MISSING', '审计目标 crop 不存在')
    if not positives or not negatives:
        raise IntegrationError('AUDIT_EXEMPLAR_MISSING', '图片示例审计缺少正例或负例')
    out = Path(ctx['output'])
    out.mkdir(parents=True, exist_ok=True)
    sheet = build_contact_sheet(target, positives, negatives, out / 'contact.jpg',
                                target_label=ctx.get('target_label') or 'needle_crystal')
    from ls_common import load_config, resolve_key
    from vlm_api import call_vlm, encode_jpeg_b64, parse_vlm_json
    with Image.open(sheet) as image:
        # encode_jpeg_b64 按 BGR 处理；PIL 给的是 RGB，逐通道翻转后再编码。
        b64, _ = encode_jpeg_b64(np.asarray(image.convert('RGB'))[:, :, ::-1].copy(),
                                 max_side=1600, quality=90)
    feature_id = str(ctx.get('feature_id') or 'needle_crystal')
    prompt = (
        '这是一张显微图片示例对照拼图。TARGET 是待审计的裁图；'
        'P 开头的是已标注的正例（目标微粒）；N 开头的是已标注的负例（易混组织/其他微粒）。'
        '请只比较可见形态，不要根据文件名或顺序猜测。'
        f'目标特征为 {feature_id}。判断 TARGET 更接近正例还是负例，并给出 0~1 的：'
        'target_score（目标正例相似度）、negative_similarity（最相似负例相似度）、'
        'negative_margin（target_score - negative_similarity）。'
        'audit_label 取目标 feature_id / 负例 feature_id / other / unknown；'
        'confidence 取 high / medium / low；reason 一句话。'
        '只输出 JSON：{"target_score":0.0,"negative_similarity":0.0,"negative_margin":0.0,'
        '"audit_label":"unknown","confidence":"low","reason":""}'
    )
    config = load_config()
    text, error = call_vlm(config['vlm'], resolve_key(config), b64, prompt)
    if text is None:
        raise IntegrationError('AUDIT_VLM_FAILED', f'图片示例审计请求失败: {error or "unknown"}')
    try:
        parsed = parse_vlm_json(text)
    except Exception as exc:
        raise IntegrationError('AUDIT_OUTPUT_INVALID', f'图片示例审计解析失败: {exc}') from exc
    result = normalize_audit(parsed, target_id=ctx.get('target_id'),
                             positive_count=len(positives), negative_count=len(negatives),
                             feature_id=feature_id)
    result['target'] = str(target)
    result['contact_sheet'] = str(sheet)
    write_json(out / 'audit.json', result)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
