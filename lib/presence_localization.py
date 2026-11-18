"""把 presence 观察的可选 box 投影为影子定位线索，供账本与规划器参考，不产生正式工件。

源图只读；本模块不生成正式 CropMeta、身份批准、标定或测量，几何/质量提示也不改变
旧的 P1/C 观察。模型自报的 transform 一律不采信；box schema 归 ToolRegistry，
IoU/上下文扩展与图像质量原语仍留在 geom/box_quality。

主要定义：
- POLICY_VERSION: 定位策略版本 `presence_localization_shadow_v1`。
- sanitize_proposal: 丢弃畸形可选 box，不使合法观察失效。
- observation_context: 绑定实际下发栅格并记录来源哈希、尺寸与逆变换。
- assess_presence: 把观察结果拆分为定位、质量、身份与用途资格。
- `_mapped_box` / `_spatial_hints`：坐标逆映射与重叠诊断的内部工具。

关联：
- TCM_Fusion.lib.tool_registry: PRESENCE_BOX_SCHEMA 与 validate_arguments 校验 box。
- TCM_Fusion.lib.box_quality: assess 提供质量提示。
- geom: iou_box/loose_box 提供共享几何原语。
- TCM_Fusion.lib.agent_tools: 调用 sanitize_proposal/observation_context/assess_presence。
- TCM_Fusion.lib.storage: 哈希、数值校验与 JSON 写出。
"""
from __future__ import annotations

import base64
import io
import math
from pathlib import Path

from PIL import Image
from geom import iou_box, loose_box

from .box_quality import assess
from .storage import IntegrationError, file_hash, finite_number, write_json
from .tool_registry import PRESENCE_BOX_SCHEMA, validate_arguments

POLICY_VERSION = 'presence_localization_shadow_v1'


def sanitize_proposal(value):
    """Discard malformed optional hints without invalidating a valid observation."""
    if value is None:
        return None
    try:
        validate_arguments(PRESENCE_BOX_SCHEMA, value)
    except IntegrationError:
        return {'error_code': 'BOX_SCHEMA_INVALID'}
    return {'bbox': list(value['bbox']), 'coordinate_space': value['coordinate_space']}


def observation_context(path, image_url):
    """Bind to the actual sent raster; never accept model-supplied transforms.

The observation encoder uses full-image resizing without padding. Rotated EXIF
inputs remain valid legacy observations, but box mapping is refused until their
orientation is unambiguously normalized by the encoder.
"""
    try:
        before = file_hash(path)
        with Image.open(path) as source:
            width, height = source.size
            if source.getexif().get(274, 1) != 1:
                return {'error_code': 'ORIENTATION_MAPPING_UNCONFIRMED'}
        data = base64.b64decode(image_url.split(',', 1)[1], validate=True)
        with Image.open(io.BytesIO(data)) as sent:
            sent_width, sent_height = sent.size
            if sent.getexif().get(274, 1) != 1:
                return {'error_code': 'ORIENTATION_MAPPING_UNCONFIRMED'}
        if file_hash(path) != before:
            return {'error_code': 'SOURCE_CHANGED'}
        return {'source_image_sha256': before, 'source_size': [width, height],
                'model_input_size': [sent_width, sent_height],
                'transform': {'kind': 'full_image_resize',
                              'scale_xy': [sent_width / width, sent_height / height],
                              'padding_xy': [0, 0], 'roi_xyxy': [0, 0, width, height]}}
    except (OSError, ValueError, TypeError, IndexError):
        return {'error_code': 'IMAGE_MAPPING_UNAVAILABLE'}


def _mapped_box(proposal, context, size, expected_sha256):
    validate_arguments(PRESENCE_BOX_SCHEMA, proposal)
    if (context.get('source_image_sha256') != expected_sha256
            or context.get('source_size') != list(size)):
        raise IntegrationError('SOURCE_BINDING_INVALID', '定位上下文与原图不一致')
    transform = context.get('transform') or {}
    width, height = size
    sent = context.get('model_input_size')
    if (not isinstance(sent, list) or len(sent) != 2
            or any(type(n) is not int or n <= 0 for n in sent)
            or transform.get('kind') != 'full_image_resize'
            or transform.get('padding_xy') != [0, 0]
            or transform.get('roi_xyxy') != [0, 0, width, height]
            or transform.get('scale_xy') != [sent[0] / width, sent[1] / height]):
        raise IntegrationError('TRANSFORM_INVALID', '不支持或无法核实的定位映射')
    x1, y1, x2, y2 = proposal['bbox']
    if not (0 <= x1 < x2 <= sent[0] and 0 <= y1 < y2 <= sent[1]):
        raise IntegrationError('BOX_BOUNDS_INVALID', '定位框反向、退化或越界')
    # Same full-image inverse scaling as detection; do not clamp or sort bad boxes.
    return [x1 * width / sent[0], y1 * height / sent[1],
            x2 * width / sent[0], y2 * height / sent[1]]


def _spatial_hints(box, feature_id, existing):
    rows = []
    for row in existing:
        other = row.get('box')
        if (not isinstance(other, (list, tuple)) or len(other) != 4
                or any(type(n) not in (int, float) or not math.isfinite(n) for n in other)
                or not (other[0] < other[2] and other[1] < other[3])):
            continue
        overlap = iou_box(box, other)
        if overlap <= 0:
            continue
        rows.append({'artifact_id': row.get('id'), 'feature_id': row.get('feature_id'),
                     'iou': round(overlap, 6),
                     'possible_duplicate': overlap >= .5 and row.get('feature_id') == feature_id,
                     'possible_identity_conflict': overlap >= .5 and row.get('feature_id') != feature_id,
                     'identity_inference': False})
    return rows


def assess_presence(result, *, path, expected_sha256, feature_id, context=None,
                    output_dir=None, existing=(), can_publish=None):
    """Return independent diagnostic axes; no model can grant eligibility here.

output_dir is a fresh host-owned action directory under the current ledger.
Diagnostic PNGs deliberately have no formal CropMeta sidecars and cannot enter
classification or measurement through the artifact store.
"""
    row = {'policy_version': POLICY_VERSION, 'mode': 'shadow',
           'source_image_sha256': expected_sha256, 'feature_id': feature_id,
           'localization': 'unlocalized', 'quality': 'not_assessed',
           'identity': 'pending', 'verification_basis': [], 'findings': [],
           'usage_eligibility': {k: False for k in ('presence', 'morphology', 'length', 'width', 'area')},
           'legacy_presence_unchanged': True, 'state': 'unknown'}
    if (result.get('ok') is not True or type(result.get('present')) is not bool
            or not finite_number(result.get('conf'), maximum=1)):
        return row
    if result.get('conf', 0) < .5:
        row['findings'].append('LOW_OBSERVATION_CONFIDENCE')
        return row
    row['observation_scope'] = {'kind': 'sent_full_fov', 'exhaustive': False}
    if not result['present']:
        row['state'] = 'not_observed_local'
        return row
    row['state'] = 'present_unlocalized'
    proposal = sanitize_proposal(result.get('localization_proposal'))
    if proposal is None:
        return row
    try:
        if can_publish is not None and not can_publish():
            raise IntegrationError('LOCALIZATION_CANCELLED', '任务结束，禁止发布定位诊断')
        if proposal.get('error_code'):
            raise IntegrationError(proposal['error_code'], '定位提示不满足 schema')
        if (context or {}).get('error_code'):
            raise IntegrationError(context['error_code'], '原图映射未能确认')
        if file_hash(path) != expected_sha256:
            raise IntegrationError('SOURCE_CHANGED', '原图内容已变化')
        with Image.open(path) as source:
            source.load()
            box = _mapped_box(proposal, context or {}, source.size, expected_sha256)
            width, height = source.size
            bw, bh = box[2] - box[0], box[3] - box[1]
            findings = row['findings']
            if min(bw, bh) < 16:
                findings.append('small_box')
            if max(bw / bh, bh / bw) > 10:
                findings.append('extreme_aspect')
            if bw * bh > .5 * width * height:
                findings.append('large_box')
            if box[0] == 0 or box[1] == 0 or box[2] == width or box[3] == height:
                findings.append('possible_border_truncation')
            crop_box = loose_box(box, width, height)
            row.update(localization='valid', state='present_pending',
                       source_tight_bbox=box, crop_bbox=crop_box,
                       transform=context['transform'], model_input_size=context['model_input_size'],
                       spatial_agreement=_spatial_hints(box, feature_id, existing),
                       verification_basis=[{'kind': 'glm_localization_hypothesis',
                                            'independent_support': False}])
            # Assess target pixels separately: expanded context must not disguise a blank target.
            tight_raster = [math.floor(box[0]), math.floor(box[1]),
                            math.ceil(box[2]), math.ceil(box[3])]
            target = source.crop(tight_raster).convert('RGB')
            crop = source.crop(crop_box).convert('RGB')
            # PNG round trip is an integrity check even when no output is requested.
            buffer = io.BytesIO()
            target.save(buffer, format='PNG')
            with Image.open(io.BytesIO(buffer.getvalue())) as decoded:
                decoded.load()
            if output_dir is not None:
                if can_publish is not None and not can_publish():
                    raise IntegrationError('LOCALIZATION_CANCELLED', '任务结束，禁止写裁图')
                directory = Path(output_dir)
                directory.mkdir(parents=True, exist_ok=False)
                tight_path, crop_path = directory / 'target.png', directory / 'context.png'
                with tight_path.open('xb') as stream:
                    stream.write(buffer.getvalue())
                with crop_path.open('xb') as stream:
                    crop.save(stream, format='PNG')
                diagnostic = assess(tight_path, box=box, source_size=source.size)
                row['artifacts'] = {'target_png': str(tight_path), 'context_png': str(crop_path),
                                    'target_sha256': file_hash(tight_path)}
            else:
                # Reuse the existing evaluator rather than duplicating texture thresholds.
                buffer.seek(0)
                diagnostic = assess(buffer, box=box, source_size=source.size)
            status = diagnostic['status']
            row['quality'] = {'invalid': 'unreadable', 'tiny': 'low_detail'}.get(status, status)
            row['quality_diagnostic'] = diagnostic
            if status == 'invalid':
                raise IntegrationError('CROP_UNREADABLE', '裁图不可读取')
            if file_hash(path) != expected_sha256:
                raise IntegrationError('SOURCE_CHANGED', '裁图期间原图变化')
            if output_dir is not None:
                if can_publish is not None and not can_publish():
                    raise IntegrationError('LOCALIZATION_CANCELLED', '任务结束，禁止发布定位诊断')
                write_json(Path(output_dir) / 'diagnostic.json', row)
    except (IntegrationError, OSError, ValueError) as exc:
        row.update(localization='invalid', state='present_unlocalized')
        row['findings'].append(getattr(exc, 'code', 'CROP_INTEGRITY_FAILED'))
        if getattr(exc, 'code', None) in {'SOURCE_CHANGED', 'LOCALIZATION_CANCELLED'}:
            row['state'] = 'unknown'
    return row
