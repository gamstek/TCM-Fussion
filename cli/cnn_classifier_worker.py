"""特征分类器 CNN worker（ResNet18 / 14 类）。

只负责 crop → 14 类概率；不判药材、不做 quality gate。
在 `.venv/yolo` 下运行（该环境有 torch/torchvision）。
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


def _feature_id(class_name: str) -> str:
    name = str(class_name or '')
    return name.split('_', 1)[1] if name[:2].isdigit() and '_' in name else name


def _prepare(path: Path, size: int, mode: str, mean, std):
    image = Image.open(path).convert('RGB')
    if mode == 'letterbox':
        image.thumbnail((size, size), Image.Resampling.BICUBIC)
        canvas = Image.new('RGB', (size, size), (114, 114, 114))
        canvas.paste(image, ((size - image.width) // 2, (size - image.height) // 2))
        image = canvas
    elif mode == 'squash':
        image = image.resize((size, size), Image.Resampling.BICUBIC)
    else:
        width, height = image.size
        scale = size / min(width, height)
        image = image.resize((int(width * scale), int(height * scale)), Image.Resampling.BICUBIC)
        left = (image.width - size) // 2
        top = (image.height - size) // 2
        image = image.crop((left, top, left + size, top + size))
    tensor = transforms.ToTensor()(image)
    return transforms.Normalize(mean, std)(tensor)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--context', required=True)
    args = parser.parse_args(argv)
    context = json.loads(Path(args.context).read_text(encoding='utf-8'))
    meta = json.loads(Path(context['meta']).read_text(encoding='utf-8'))
    weights = Path(context['weights'])
    classes = list(meta['classes'])
    size = int(meta.get('imgsz', 320))
    mode = str(meta.get('input_mode', 'letterbox'))
    mean = list(meta.get('mean') or [0.485, 0.456, 0.406])
    std = list(meta.get('std') or [0.229, 0.224, 0.225])
    checkpoint = torch.load(weights, map_location='cpu', weights_only=False)
    model = getattr(models, meta.get('arch', 'resnet18'))(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    crops = context.get('crops') or []
    if not crops:
        raise SystemExit('CNN_INPUT: no crops')
    batch_size = max(1, int(context.get('batch_size', 16)))
    tensors = [_prepare(Path(item['png']), size, mode, mean, std) for item in crops]
    rows = []
    with torch.no_grad():
        for start in range(0, len(tensors), batch_size):
            batch = torch.stack(tensors[start:start + batch_size])
            probs = torch.softmax(model(batch), dim=1)
            for offset, prob in enumerate(probs):
                top = torch.topk(prob, min(3, len(classes)))
                values = [float(x) for x in top.values]
                indexes = [int(x) for x in top.indices]
                entropy = -sum(float(p) * math.log(max(float(p), 1e-12)) for p in prob)
                entropy /= math.log(max(2, len(classes)))
                rows.append({
                    'id': crops[start + offset]['id'],
                    'topk': [{'class': classes[index], 'feature_id': _feature_id(classes[index]),
                              'prob': round(value, 6)}
                             for value, index in zip(values, indexes)],
                    'top1': _feature_id(classes[indexes[0]]),
                    'top1_class': classes[indexes[0]],
                    'top1_prob': round(values[0], 6),
                    'margin': round(values[0] - values[1], 6) if len(values) > 1 else round(values[0], 6),
                    'entropy': round(entropy, 6),
                    'source': 'cnn',
                })
    output = Path(context['output'])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({'schema': 'cnn-classifier/v1', 'model': str(weights),
                                  'imgsz': size, 'input_mode': mode,
                                  'classes': classes, 'rows': rows},
                                 ensure_ascii=False, indent=2), encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
