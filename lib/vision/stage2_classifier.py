"""stage2 适配层：训练侧 cls_net → crop 显微特征身份概率 → workflow_backend。

只处理显微特征，药材五折候选不经此处。父进程仅读元信息与指纹；Torch 只在
worker 调用 infer 时加载；预处理复用训练侧唯一实现，checkpoint/meta 不符即失败。

主要定义：
- feature_id: 去除类别名前缀，得到显微特征 ID。
- model_profile: 只读校验权重/元信息并生成模型档案，不加载权重。
- request_signature: 绑定模型、变换、解释器及每个 crop 实际内容的请求指纹。
- cache_matches: 判断已有预测能否复用同一指纹。
- infer: 在 worker 中执行真实 CNN，输出逐 crop topk/entropy 与模型指纹。

关联：
- 调用方 「cli/cnn_classifier_worker.py」、「lib/application/pipeline.py」、「lib/workflow/workflow_backend.py」、「lib/workflow/workflow_session.py」、「tests/vision/test_stage2.py」。
- 直接依赖 「lib/runtime/storage.py」。
"""
from __future__ import annotations
import math
import re
from pathlib import Path
import paths
from TCM_Fusion.lib.runtime.storage import IntegrationError, digest, file_hash, finite_number, project_path, read_json


def feature_id(name):
    return re.sub(r'^\d+_', '', name)


def model_profile(config):
    """配置/sidecar 检查，不加载权重；checkpoint 一致性由 worker 核对。"""
    if not config.get('cnn_weights') or not config.get('cnn_meta'):
        raise IntegrationError('CNN_CONFIG_INVALID', '需要成对配置 cnn_weights/cnn_meta')
    weights, sidecar = (project_path(config[k]) for k in ('cnn_weights', 'cnn_meta'))
    if not weights.is_file() or not sidecar.is_file():
        raise IntegrationError('CNN_ASSET_MISSING', 'stage2 权重或 model_meta.json 不存在')
    meta = read_json(sidecar)
    classes = meta.get('classes') if isinstance(meta, dict) else None
    known = {r['feature_id'] for r in read_json(Path(paths.PROJECT_ROOT) / 'feature_taxonomy.json')['features']} | {'other'}
    if (not isinstance(classes, list) or len(classes) < 2
            or any(not isinstance(c, str) or not c for c in classes)
            or len(set(classes)) != len(classes)
            or len({feature_id(c) for c in classes}) != len(classes)
            or any(feature_id(c) not in known for c in classes)):
        raise IntegrationError('CNN_META_INVALID', 'classes 必须是唯一、已登记的显微特征类别')
    size = meta.get('imgsz')
    if (meta.get('arch') not in {'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'}
            or isinstance(size, bool) or not isinstance(size, int) or size < 1
            or meta.get('input_mode', 'center') not in {'center', 'squash', 'letterbox'}):
        raise IntegrationError('CNN_META_INVALID', 'stage2 架构、尺寸或 input_mode 不受支持')
    return {'weights': str(weights), 'meta': str(sidecar), 'metadata': meta,
            'weights_sha256': file_hash(weights), 'meta_sha256': file_hash(sidecar),
            'preprocessor_sha256': file_hash(Path(paths.DETECTION_LIB_DIR) / 'cls_net.py'),
            'adapter_sha256': file_hash(__file__)}


def request_signature(profile, crops, python):
    """同任务缓存绑定模型、解释器、crop 内容与分类子窗。"""
    items = []
    for row in crops:
        item = {'id': row['id'], 'png': str(project_path(row['png'])), 'sha256': file_hash(row['png'])}
        box = row.get('box_local')
        if box is not None:
            if (not isinstance(box, list) or len(box) != 4 or any(type(v) is not int for v in box)
                    or not (0 <= box[0] < box[2] and 0 <= box[1] < box[3])):
                raise IntegrationError('CNN_INPUT_INVALID', 'stage2 分类子窗坐标非法')
            item['box_local'] = list(box)
        items.append(item)
    if not items or len({r['id'] for r in items}) != len(items):
        raise IntegrationError('CNN_INPUT_INVALID', 'stage2 crop 不能为空或重复')
    return {'schema': 'stage2-request/v2', 'profile': profile, 'crops': items,
            'python': str(project_path(python)), 'python_sha256': file_hash(python)}


def cache_matches(value, signature):
    if not isinstance(value, dict) or value.get('signature') != signature:
        return False
    classes = signature['profile']['metadata']['classes']
    rows = value.get('rows')
    def valid_row(row):
        if not isinstance(row,dict):
            return False
        topk=row.get('topk')
        if not isinstance(topk,list) or len(topk)!=min(3,len(classes)):
            return False
        if not all(isinstance(r,dict) and r.get('class') in classes
                   and r.get('feature_id')==feature_id(r['class'])
                   and finite_number(r.get('prob'),maximum=1) for r in topk):
            return False
        return (len({r['class'] for r in topk})==len(topk)
                and all(a['prob']>=b['prob'] for a,b in zip(topk,topk[1:]))
                and row.get('top1_class')==topk[0]['class']
                and row.get('top1')==topk[0]['feature_id']
                and row.get('top1_prob')==topk[0]['prob']
                and finite_number(row.get('entropy'),maximum=1))
    return (isinstance(rows, list) and len(rows) == len(signature['crops'])
            and {r.get('id') for r in rows if isinstance(r, dict)} == {r['id'] for r in signature['crops']}
            and all(valid_row(r) for r in rows))


def infer(context):
    """真实 CNN，输出兼容 workflow 的逐 crop topk/entropy 与模型指纹。"""
    import sys
    import torch
    from PIL import Image
    from 目标检测.lib.cls_net import build_transforms, load_model
    cfg = {'cnn_weights': context['weights'], 'cnn_meta': context['meta']}
    profile = model_profile(cfg)
    crops = context.get('crops') or []
    signature = request_signature(profile, crops, sys.executable)
    if context.get('signature') is not None and context['signature'] != signature:
        raise IntegrationError('CNN_INPUT_CHANGED', 'stage2 请求的模型或 crop 指纹已变化')
    model, actual, device = load_model(profile['weights'], device='cpu')
    meta = profile['metadata']
    for key in ('arch', 'imgsz', 'classes'):
        if actual[key] != meta[key]:
            raise IntegrationError('CNN_META_MISMATCH', 'checkpoint 与 sidecar 的 ' + key + ' 不一致')
    mode = meta.get('input_mode', 'center')
    if actual['input_mode'] != mode:
        raise IntegrationError('CNN_META_MISMATCH', 'checkpoint 实际 input_mode 与配置不一致')
    _, transform, mean, std = build_transforms(actual['imgsz'], mode)
    if meta.get('mean', mean) != mean or meta.get('std', std) != std:
        raise IntegrationError('CNN_META_MISMATCH', '归一化参数与训练侧变换不一致')
    classes = actual['classes']
    batch_size = int(context.get('batch_size', 16))
    if batch_size < 1:
        raise IntegrationError('CNN_INPUT_INVALID', 'batch_size 必须大于零')
    rows = []
    with torch.inference_mode():
        for start in range(0, len(crops), batch_size):
            chunk = crops[start:start + batch_size]
            tensors = []
            for item in chunk:
                with Image.open(item['png']) as image:
                    box = item.get('box_local')
                    if box is not None:
                        if box[2] > image.width or box[3] > image.height:
                            raise IntegrationError('CNN_INPUT_INVALID', 'stage2 分类子窗越出裁图')
                        image = image.crop(box)
                    tensors.append(transform(image.convert('RGB')))
            probs = torch.softmax(model(torch.stack(tensors).to(device)).float(), dim=1).cpu()
            if not torch.isfinite(probs).all():
                raise IntegrationError('CNN_OUTPUT_INVALID', 'stage2 输出非有限概率')
            for item, prob in zip(chunk, probs):
                values, indexes = prob.topk(min(3, len(classes)))
                indexes, values = indexes.tolist(), values.tolist()
                entropy = float(-(prob * prob.clamp_min(1e-12).log()).sum()) / math.log(len(classes))
                rows.append({'id': item['id'], 'source': 'cnn',
                    'topk': [{'class': classes[i], 'feature_id': feature_id(classes[i]), 'prob': p}
                             for i, p in zip(indexes, values)],
                    'top1': feature_id(classes[indexes[0]]), 'top1_class': classes[indexes[0]],
                    'top1_prob': values[0], 'margin': values[0] - values[1],
                    'entropy': min(1., max(0., entropy))})
    if request_signature(model_profile(cfg), crops, sys.executable) != signature:
        raise IntegrationError('CNN_INPUT_CHANGED', '推理期间模型、预处理或 crop 变化')
    return {'schema': 'cnn-classifier/v2', 'model': profile['weights'],
            'imgsz': actual['imgsz'], 'input_mode': mode, 'classes': classes,
            'preprocessing': '目标检测.lib.cls_net.build_transforms',
            'signature': signature, 'cache_key': digest(signature), 'rows': rows}
