"""固定检测侧库包装 worker：全图/局部检测、重新标定与 GLM 定向定位，从项目根以 `-m` 启动，绝不 import CLI。
读取 `--context` 绑定上下文，按 `ctx['tool']` 分发 `calibrate_fov`、`detect_fov_initial`、`detect_region`、`locate_feature`，写回标定、裁图与框清单。从项目根以模块方式启动。读取命令参数或任务上下文，并通过约定的结果文件或退出码交付。

主要定义：
- main: 读取 `--context` 绑定上下文，按 `ctx['tool']` 分发 `calibrate_fov`、`detect_fov_initial`、`detect_region`、`locate_feature`，写回标定、裁图与框清单。
- calibrate_fov 分支: 双引擎标定，状态不属可信集合时以 `SCALE_UNTRUSTED` 失败。
- detect_fov_initial / detect_region 分支: 消费 yolo_worker 输出的框，裁出 crop 并保留原图坐标。
- locate_feature 分支: 在合法局部 ROI 内调用 GLM 定向定位，按 IOU 去重同 ROI 重复框，缺目标返回空。
- 模块级约束: `locate_min_roi_side` 用于扩窗，`locate_min_box_side` 约束框边长，`feature_hint` 只作提示词补充。

关联：
- 直接依赖 「lib/runtime/storage.py」。
- 从项目根通过 python -m 启动。
"""
import argparse
from pathlib import Path
from TCM_Fusion.lib.runtime.storage import file_hash, read_json, write_json, IntegrationError


def main(argv=None):
    parser=argparse.ArgumentParser();parser.add_argument('--context',required=True)
    args=parser.parse_args(argv);ctx=read_json(args.context)
    from ls_common import imread_cn, TRUSTED_STATUSES
    from detect_core import lookup_px_per_um
    structure_crop = ctx.get('feature_crop_policy') == 'structure_v1'
    try:
        source_sha = file_hash(ctx['source']) if structure_crop else None
    except OSError as exc:
        raise IntegrationError('IMAGE_INVALID', '无法读取绑定原图') from exc
    img=imread_cn(ctx['source'])
    if img is None:raise IntegrationError('IMAGE_INVALID','无法读取绑定原图')
    out=Path(ctx['output']);out.mkdir(parents=True,exist_ok=True)
    if ctx['tool']=='calibrate_fov':
        info=lookup_px_per_um(ctx['source'],img,full=True)
        if info.get('status') not in TRUSTED_STATUSES:
            raise IntegrationError('SCALE_UNTRUSTED','双引擎没有可信标定')
        write_json(out/'scale.json',{'px_per_um':float(info['px_per_um']),'status':info['status']})
        return 0
    initial = ctx['tool'] == 'detect_fov_initial'
    targeted = ctx['tool'] == 'locate_feature'
    if not initial and not targeted and ctx['tool']!='detect_region':raise IntegrationError('TOOL_UNKNOWN','未知worker操作')
    from detect_core import FEATURE_ID
    from geom import loose_box, iou_box
    import cv2
    import contracts
    h,w=img.shape[:2]
    if targeted:
        from detect_core import detect_once
        from prompts import CLASSIFY_HINTS
        from ls_common import load_config, resolve_key
        x1, y1, x2, y2 = map(int, ctx['roi'])
        from TCM_Fusion.lib.vision.image_tiling import local_roi
        if not local_roi((x1, y1, x2, y2), w, h):
            raise IntegrationError('ROI_INVALID', 'GLM 定向定位仅接受原图局部')
        # 观察窗口至少 min_roi_side，避免在过小 ROI 内强迫模型给框。
        min_roi = int(ctx.get('locate_min_roi_side') or 0)
        rx1, ry1, rx2, ry2 = x1, y1, x2, y2
        if min_roi > 0 and max(rx2-rx1, ry2-ry1) < min_roi:
            cx, cy = (rx1+rx2)//2, (ry1+ry2)//2
            rx1, rx2 = max(0, cx-min_roi//2), min(w, cx+min_roi//2)
            ry1, ry2 = max(0, cy-min_roi//2), min(h, cy+min_roi//2)
        if (rx2-rx1)*(ry2-ry1) == w*h:
            rx1, ry1, rx2, ry2 = x1, y1, x2, y2
        label = next((name for name, fid in FEATURE_ID.items() if fid == ctx['feature_id']), None)
        if label not in CLASSIFY_HINTS:
            raise IntegrationError('ACTION_DENIED', '该特征没有受信形态描述')
        prompt = (f'只检查所给局部是否可定位 {label}。形态：{CLASSIFY_HINTS[label]}。'
                  f'{ctx.get("feature_hint") or ""} '
                  '无法区分、目标截断或看不清时返回空 objects。'
                  '如果局部没有目标，必须返回 {"objects":[]}；不得用背景纹理、色差、模糊边缘或相邻组织凑框。'
                  '列出所有可分辨的完整目标紧框，按置信度从高到低；不要为了凑数编造目标。'
                  '坐标相对所给局部归一化到0至1000。'
                  f'仅输出JSON：{{"objects":[{{"label":"{label}","bbox_2d":[x1,y1,x2,y2],'
                  '"confidence":"high 或 medium","morphology":"可见判别特征"}]}')
        cfg = load_config()
        response, error = detect_once(img, cfg['vlm'], resolve_key(cfg), prompt, {label},
                                      region=[rx1, ry1, rx2, ry2], max_side=1024, upscale=True)
        if error:
            raise IntegrationError('DETECTION_FAILED', 'GLM 定向定位请求失败')
        dets = []
        for det in response['dets']:
            left, top, right, bottom = map(round, det['tight_bbox'])
            if rx1 < left < right < rx2 and ry1 < top < bottom < ry2:
                dets.append({**det, 'tight_bbox': [left, top, right, bottom]})
        # 同一 ROI 内 GLM 可能重复给同一目标；保留置信度高的框，避免重复注入。
        dets.sort(key=lambda d: float(d.get('score') or 0), reverse=True)
        unique = []
        for det in dets:
            if any(iou_box(det['tight_bbox'], other['tight_bbox']) >= 0.5 for other in unique):
                continue
            unique.append(det)
        dets = unique
        expand = float(ctx.get('locate_expand') or 1.0)
        if expand > 1.0:
            for det in dets:
                bx1, by1, bx2, by2 = det['tight_bbox']
                cx, cy = (bx1 + bx2) / 2.0, (by1 + by2) / 2.0
                bw, bh = (bx2 - bx1) * expand, (by2 - by1) * expand
                det['source_tight_bbox'] = det['tight_bbox']
                det['tight_bbox'] = [max(0.0, cx - bw / 2), max(0.0, cy - bh / 2),
                                     min(float(w), cx + bw / 2), min(float(h), cy + bh / 2)]
                det['locate_expand'] = expand
        min_side = int(ctx.get('locate_min_box_side') or 0)
        if min_side > 0:
            for det in dets:
                bx1, by1, bx2, by2 = det['tight_bbox']
                if max(bx2-bx1, by2-by1) >= min_side:
                    continue
                cx, cy = (bx1+bx2)/2.0, (by1+by2)/2.0
                half = min_side/2.0
                det['source_tight_bbox'] = det.get('source_tight_bbox') or det['tight_bbox']
                det['tight_bbox'] = [max(0.0, cx-half), max(0.0, cy-half),
                                     min(float(w), cx+half), min(float(h), cy+half)]
                det['min_box_side'] = min_side
                det['min_box_expanded'] = True
        result = {'dets': dets, 'roi': ctx['roi'], 'feature_id': ctx['feature_id']}
    else:
        result = read_json(ctx['boxes'])
        if (result['source_sha256'] != (source_sha or file_hash(ctx['source'])) or result['image_size'] != [w, h]
                or result['roi'] != ctx.get('roi')):
            raise IntegrationError('SAMPLE_MISMATCH', 'YOLO 框与绑定原图或局部范围不一致')
    detections = result['dets']
    if not targeted:
        from 目标检测.lib.det_dataset import split_delivery
        detections, other = split_delivery(detections)
        write_json(out/'other_detections.json', {'dets':other})
    if initial:
        # 标定是检测后的可选增强；scalebar VLM 失败不得回滚已产出的 YOLO 框。
        try:
            ctx['px_per_um'] = lookup_px_per_um(ctx['source'], img) if detections else None
        except Exception as exc:
            ctx['px_per_um'] = None
            print(f'[warn] 自动标定失败，保留检测框并置 px_per_um=None: {type(exc).__name__}: {exc}', flush=True)
    crops=out/'crops';crops.mkdir()
    from TCM_Fusion.lib.runtime.storage import digest
    from TCM_Fusion.lib.vision.feature_window import feature_window
    count=0;replacements={};used_replacements=set();seen=list(ctx.get('existing_boxes',[]))
    for d in detections:
        fid = FEATURE_ID[d['label']]
        box = d['tight_bbox']
        classification_window = loose_box(box,w,h)
        identity_box = box
        window_info = None
        if structure_crop:
            seed = d.get('source_tight_bbox', box)
            window_info = feature_window(img, seed, fid, context_box=classification_window)
            if window_info is not None:
                box = d['tight_bbox'] = seed
        # 已有同类区域不重复注入；复用框IoU原语，不把重复实例当新证据。
        overlaps=[b for b in seen if d['label']==b['label'] and iou_box(box,b['box'])>=0.5]
        replacement=next((b['id'] for b in overlaps if b.get('id')==ctx.get('replace_id')),None)
        if replacement in used_replacements:continue
        if overlaps and replacement is None:continue
        count+=1;stem=Path(ctx['source']).stem
        suffix = '' if initial else ('_glm' if targeted else '_roi')+digest(ctx['roi'])[:12]
        cid=f'{fid}_{stem}{suffix}_{count:02d}'
        if replacement:
            replacements[cid]=replacement;used_replacements.add(replacement)
        seen.append({'id':cid,'label':d['label'],'box':box})
        window, frame = window_info if window_info is not None else (classification_window, None)
        ox,oy,ex,ey=window
        crop=img[oy:ey,ox:ex];ok,encoded=cv2.imencode('.png',crop)
        if not ok:raise IntegrationError('CROP_ENCODE','裁图编码失败')
        with (crops/(cid+'.png')).open('wb') as f:f.write(encoded.tobytes())
        side=contracts.crop_sidecar(cid,cid+'.png',ctx['source'],stem,fid,d['label'],'',
            ctx.get('px_per_um'),box,[box[0]-ox,box[1]-oy,box[2]-ox,box[3]-oy],
            [ox,oy,ex,ey],f'{crop.shape[1]}x{crop.shape[0]}',
            granularity=d.get('granularity') or contracts.GRANULARITY_DEFAULT)
        # 定位分数进入 crop 契约的 raw 字段，供“高分 crop 先分类”的调度使用。
        side['det_score']=d.get('score')
        side['det_confidence']=d.get('confidence')
        if d.get('source_tight_bbox') is not None:
            side['source_tight_bbox']=d['source_tight_bbox']
            side['locate_expand']=d.get('locate_expand')
        if frame is not None:
            offset = [ox, oy, ox, oy]
            side['feature_window'] = {
                **frame, 'source_sha256': source_sha,
                'classification_box_local': [v - p for v, p in zip(classification_window, offset)],
                'identity_box_local': [int(v - p) for v, p in zip(identity_box, offset)],
            }
        write_json(crops/(cid+'.json'),side)
    if source_sha and file_hash(ctx['source']) != source_sha:
        raise IntegrationError('SAMPLE_MISMATCH', '裁图期间绑定原图发生变化')
    if initial:
        write_json(out/'crops.json', {**result, 'engine':'yolo', 'count':count,
                                     'coordinate_system':'original_pixels'})
    else:
        write_json(out/'region_result.json',{**result,'engine':'glm_targeted' if targeted else 'yolo','count':count,
                                          'replacements':replacements,'coordinate_system':'original_pixels'})
    return 0


if __name__=='__main__':raise SystemExit(main())
