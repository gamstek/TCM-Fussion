"""固定检测侧库包装：局部检测/重新标定；根目录以-m启动，绝不import CLI。"""
import argparse
from pathlib import Path
from TCM_Fusion.lib.storage import read_json, write_json, IntegrationError


def main(argv=None):
    parser=argparse.ArgumentParser();parser.add_argument('--context',required=True)
    args=parser.parse_args(argv);ctx=read_json(args.context)
    from ls_common import imread_cn, TRUSTED_STATUSES
    from detect_core import lookup_px_per_um
    img=imread_cn(ctx['source'])
    if img is None:raise IntegrationError('IMAGE_INVALID','无法读取绑定原图')
    out=Path(ctx['output']);out.mkdir(parents=True,exist_ok=True)
    if ctx['tool'] == 'propose_regions':
        from TCM_Fusion.lib.recovery_regions import propose_regions
        write_json(out/'regions.json', {'regions': propose_regions(img)})
        return 0
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
    from TCM_Fusion.lib.storage import file_hash
    if targeted:
        from detect_core import detect_once
        from prompts import CLASSIFY_HINTS
        from ls_common import load_config, resolve_key
        x1, y1, x2, y2 = ctx['roi']
        if not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h and (x2-x1)*(y2-y1) < w*h):
            raise IntegrationError('ROI_INVALID', 'GLM 定向定位仅接受原图局部')
        label = next((name for name, fid in FEATURE_ID.items() if fid == ctx['feature_id']), None)
        if label not in CLASSIFY_HINTS:
            raise IntegrationError('ACTION_DENIED', '该特征没有受信形态描述')
        prompt = (f'只检查所给局部是否可定位 {label}。形态：{CLASSIFY_HINTS[label]}。'
                  '无法区分、目标截断或看不清时返回空 objects。最多返回一个完整目标的紧框。'
                  '坐标相对所给局部归一化到0至1000。'
                  f'仅输出JSON：{{"objects":[{{"label":"{label}","bbox_2d":[x1,y1,x2,y2],'
                  '"confidence":"high 或 medium","morphology":"可见判别特征"}]}')
        cfg = load_config()
        response, error = detect_once(img, cfg['vlm'], resolve_key(cfg), prompt, {label},
                                      region=ctx['roi'], max_side=1024, upscale=True)
        if error:
            raise IntegrationError('DETECTION_FAILED', 'GLM 定向定位请求失败')
        dets = []
        for det in response['dets'][:1]:
            left, top, right, bottom = map(round, det['tight_bbox'])
            if x1 < left < right < x2 and y1 < top < bottom < y2:
                dets.append({**det, 'tight_bbox': [left, top, right, bottom]})
        result = {'dets': dets, 'roi': ctx['roi'], 'feature_id': ctx['feature_id']}
    else:
        result = read_json(ctx['boxes'])
        if (result['source_sha256'] != file_hash(ctx['source']) or result['image_size'] != [w, h]
                or result['roi'] != ctx.get('roi')):
            raise IntegrationError('SAMPLE_MISMATCH', 'YOLO 框与绑定原图或局部范围不一致')
    detections = result['dets']
    if not targeted:
        from TCM_Fusion.lib.det_dataset import split_delivery
        detections, other = split_delivery(detections)
        write_json(out/'other_detections.json', {'dets':other})
    if initial:
        ctx['px_per_um'] = lookup_px_per_um(ctx['source'], img) if detections else None
    crops=out/'crops';crops.mkdir()
    from TCM_Fusion.lib.storage import digest
    count=0;replacements={};used_replacements=set();seen=list(ctx.get('existing_boxes',[]))
    for d in detections:
        # 已有同类区域不重复注入；复用框IoU原语，不把重复实例当新证据。
        overlaps=[b for b in seen if d['label']==b['label'] and iou_box(d['tight_bbox'],b['box'])>=0.5]
        replacement=next((b['id'] for b in overlaps if b.get('id')==ctx.get('replace_id')),None)
        if replacement in used_replacements:continue
        if overlaps and replacement is None:continue
        count+=1;fid=FEATURE_ID[d['label']];stem=Path(ctx['source']).stem
        suffix = '' if initial else ('_glm' if targeted else '_roi')+digest(ctx['roi'])[:12]
        cid=f'{fid}_{stem}{suffix}_{count:02d}'
        if replacement:
            replacements[cid]=replacement;used_replacements.add(replacement)
        seen.append({'id':cid,'label':d['label'],'box':d['tight_bbox']})
        box=d['tight_bbox'];ox,oy,ex,ey=loose_box(box,w,h)
        crop=img[oy:ey,ox:ex];ok,encoded=cv2.imencode('.png',crop)
        if not ok:raise IntegrationError('CROP_ENCODE','裁图编码失败')
        with (crops/(cid+'.png')).open('wb') as f:f.write(encoded.tobytes())
        side=contracts.crop_sidecar(cid,cid+'.png',ctx['source'],stem,fid,d['label'],'',
            ctx.get('px_per_um'),box,[box[0]-ox,box[1]-oy,box[2]-ox,box[3]-oy],
            [ox,oy,ex,ey],f'{crop.shape[1]}x{crop.shape[0]}',
            granularity=d.get('granularity') or contracts.GRANULARITY_DEFAULT)
        write_json(crops/(cid+'.json'),side)
    if initial:
        write_json(out/'crops.json', {**result, 'engine':'yolo', 'count':count,
                                     'coordinate_system':'original_pixels'})
    else:
        write_json(out/'region_result.json',{**result,'engine':'glm_targeted' if targeted else 'yolo','count':count,
                                          'replacements':replacements,'coordinate_system':'original_pixels'})
    return 0


if __name__=='__main__':raise SystemExit(main())
