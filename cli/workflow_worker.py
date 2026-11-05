"""固定检测侧库包装：局部检测/重新标定；根目录以-m启动，绝不import CLI。"""
import argparse
from pathlib import Path
from TCM_Fusion.lib.storage import read_json, write_json, IntegrationError


def main(argv=None):
    parser=argparse.ArgumentParser();parser.add_argument('--context',required=True)
    args=parser.parse_args(argv);ctx=read_json(args.context)
    from ls_common import imread_cn, load_config, TRUSTED_STATUSES
    from detect_core import lookup_px_per_um
    img=imread_cn(ctx['source'])
    if img is None:raise IntegrationError('IMAGE_INVALID','无法读取绑定原图')
    out=Path(ctx['output']);out.mkdir(parents=True,exist_ok=True)
    if ctx['tool']=='calibrate_fov':
        info=lookup_px_per_um(ctx['source'],img,full=True)
        if info.get('status') not in TRUSTED_STATUSES:
            raise IntegrationError('SCALE_UNTRUSTED','双引擎没有可信标定')
        write_json(out/'scale.json',{'px_per_um':float(info['px_per_um']),'status':info['status']})
        return 0
    if ctx['tool']!='detect_region':raise IntegrationError('TOOL_UNKNOWN','未知worker操作')
    from detect_core import detect_once, FEATURE_ID
    from prompts import PROMPT_VERSION
    from geom import loose_box, iou_box
    import cv2
    import contracts
    x1,y1,x2,y2=ctx['roi'];h,w=img.shape[:2]
    if not (0<=x1<x2<=w and 0<=y1<y2<=h) or (x2-x1)*(y2-y1)>=w*h:
        raise IntegrationError('ROI_INVALID','局部范围非法或扩大为整图')
    from ls_common import resolve_key
    cfg=load_config();key=resolve_key(cfg);cfg=cfg["vlm"]
    from prompts import build_prompt
    PROMPT=build_prompt(None)
    result,error=detect_once(img,cfg,key,PROMPT,FEATURE_ID,region=ctx['roi'],clamp='blocks')
    if error:raise IntegrationError('DETECTION_FAILED','局部检测失败')
    crops=out/'crops';crops.mkdir()
    from TCM_Fusion.lib.storage import digest
    count=0;replacements={};used_replacements=set();seen=list(ctx.get('existing_boxes',[]))
    for d in result['dets']:
        # 已有同类区域不重复注入；复用框IoU原语，不把重复实例当新证据。
        overlaps=[b for b in seen if d['label']==b['label'] and iou_box(d['tight_bbox'],b['box'])>=0.5]
        replacement=next((b['id'] for b in overlaps if b.get('id')==ctx.get('replace_id')),None)
        if replacement in used_replacements:continue
        if overlaps and replacement is None:continue
        count+=1;fid=FEATURE_ID[d['label']];stem=Path(ctx['source']).stem
        cid=f'{fid}_{stem}_roi{digest(ctx["roi"])[:12]}_{count:02d}'
        if replacement:
            replacements[cid]=replacement;used_replacements.add(replacement)
        seen.append({'id':cid,'label':d['label'],'box':d['tight_bbox']})
        box=d['tight_bbox'];ox,oy,ex,ey=loose_box(box,w,h)
        crop=img[oy:ey,ox:ex];ok,encoded=cv2.imencode('.png',crop)
        if not ok:raise IntegrationError('CROP_ENCODE','裁图编码失败')
        with (crops/(cid+'.png')).open('wb') as f:f.write(encoded.tobytes())
        side=contracts.crop_sidecar(cid,cid+'.png',ctx['source'],stem,fid,d['label'],'',
            ctx.get('px_per_um'),box,[box[0]-ox,box[1]-oy,box[2]-ox,box[3]-oy],
            [ox,oy,ex,ey],f'{crop.shape[1]}x{crop.shape[0]}')
        write_json(crops/(cid+'.json'),side)
    write_json(out/'region_result.json',{'count':count,'replacements':replacements,'roi':ctx['roi'],'coordinate_system':'original_pixels',
                                      'prompt_version':PROMPT_VERSION})
    return 0


if __name__=='__main__':raise SystemExit(main())
