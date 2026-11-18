"""受控 YOLO 检测 worker：只输出原图坐标与模型溯源，从项目根以 `-m` 启动。
读取 `--context`，校验权重/类别表/检测配置的哈希签名，在可选 ROI 上检测并把框平移回原图坐标，最后复查资产与原图未变。从项目根以模块方式启动。读取命令参数或任务上下文，并通过约定的结果文件或退出码交付。

主要定义：
- main: 读取 `--context`，校验权重/类别表/检测配置的哈希签名，在可选 ROI 上检测并把框平移回原图坐标，最后复查资产与原图未变。
- 模块级约束: 强制 `YOLO_AUTOINSTALL=false`；类别表必须与项目类别顺序一致，否则 `YOLO_CLASS_MAP` 失败；资产或原图在推理期间变化即 `YOLO_ASSET_CHANGED`。

关联：
- 直接依赖 「lib/runtime/storage.py」、「lib/vision/detector_yolo.py」。
- 从项目根通过 python -m 启动。
"""
import argparse
import os
from pathlib import Path

from TCM_Fusion.lib.runtime.storage import IntegrationError, file_hash, read_json, write_json


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--context', required=True)
    ctx = read_json(parser.parse_args(argv).context)
    os.environ['YOLO_AUTOINSTALL'] = 'false'
    import yaml
    import paths
    from ls_common import imread_cn, load_config
    from TCM_Fusion.lib.vision.detector_yolo import detect, load_settings

    assets = {key: ctx[key] for key in ('weights', 'class_map', 'detector_config')}
    assets['taxonomy_config'] = Path(paths.DETECTION_CONFIG_DIR) / 'vlm_config.json'
    def verify_assets():
        for key, path in assets.items():
            if file_hash(path) != ctx['signature'][key]:
                raise IntegrationError('YOLO_ASSET_CHANGED', f'运行期间 {key} 发生变化')

    verify_assets()
    settings = load_settings(load_config(), ctx['detector_config'], weights=ctx['weights'])
    with Path(ctx['class_map']).open(encoding='utf-8-sig') as stream:
        class_map = yaml.safe_load(stream)
    expected = {row['train_id']: row['zh'] for row in settings['classes']}
    if class_map['names_zh'] != expected:
        raise IntegrationError('YOLO_CLASS_MAP', '训练导出类别表与项目类别顺序不一致')
    if 'confidence' in ctx:
        settings['conf'] = ctx['confidence']
    granularity = settings.pop('granularity')
    source_hash = file_hash(ctx['source'])
    image = imread_cn(ctx['source'])
    if image is None:
        raise IntegrationError('IMAGE_INVALID', '无法读取绑定原图')
    h, w = image.shape[:2]
    left, top, right, bottom = ctx.get('roi', [0, 0, w, h])
    if not (0 <= left < right <= w and 0 <= top < bottom <= h):
        raise IntegrationError('ROI_INVALID', '局部范围越出原图')
    detections = detect(image[top:bottom, left:right], **settings)
    for det in detections:
        x1, y1, x2, y2 = det['tight_bbox']
        det['tight_bbox'] = [round(x1+left, 1), round(y1+top, 1),
                             round(x2+left, 1), round(y2+top, 1)]
        det['granularity'] = granularity.get(det['label'])
    verify_assets()
    if file_hash(ctx['source']) != source_hash:
        raise IntegrationError('YOLO_ASSET_CHANGED', '推理期间原图发生变化')
    write_json(ctx['output'], {'source_sha256': source_hash, 'image_size': [w,h],
                              'signature':ctx['signature'], 'confidence':settings['conf'],
                              'roi': ctx.get('roi'), 'dets': detections})
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
