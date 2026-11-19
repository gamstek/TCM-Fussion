"""分层QA受信评估：mask/crop→关联/诊断→用途资格；不改历史mask或原判据。"""
import json
import hashlib
from pathlib import Path
import cv2
import numpy as np
import contracts
import paths
import association_policy as assoc


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def policy():
    p=Path(paths.SEGMENTATION_CONFIG_DIR)/'quality_policy.json'
    return json.loads(p.read_text(encoding='utf-8'))


def morphology(feature):
    table=json.loads(Path(paths.TAXONOMY_PATH).read_text(encoding='utf-8')) if hasattr(paths,'TAXONOMY_PATH') else json.loads((Path(paths.PROJECT_ROOT)/'feature_taxonomy.json').read_text(encoding='utf-8'))
    return next((f.get('qa_shape','unknown') for f in table['features'] if f['feature_id']==feature),'unknown')


def assess(mask,crop,*,bindings,identity_verified=False,polarity_ambiguous=False,
           mapping_verified=True,stages=None,qa_policy=None):
    """纯评估。identity_verified只能来自调用方已验证的独立观察，不能由Agent赋值。"""
    cfg=qa_policy or policy();findings=[];metrics={};group=morphology(crop.feature_id)
    def finding(code,**detail):findings.append({'code':code,**detail})
    integrity=True
    if mask is None or mask.ndim!=2 or not np.isfinite(mask).all() or not np.isin(mask,[0,255]).all():
        integrity=False;finding('MASK_INVALID')
    if integrity:
        # crop像素是坐标权威，不能仅按mask自身形状自证尺寸。
        data=np.fromfile(crop.png,dtype=np.uint8);image=cv2.imdecode(data,cv2.IMREAD_UNCHANGED)
        if image is None or image.shape[:2]!=mask.shape or not mapping_verified:
            integrity=False;finding('COORDINATE_MAPPING_UNCONFIRMED')
    out=None if mask is None else mask.copy()
    if integrity:
        h,w=mask.shape;box=crop.tight_box_local
        if box is None or len(box)!=4 or not all(np.isfinite(x) for x in box):
            integrity=False;finding('COORDINATE_MAPPING_UNCONFIRMED')
        else:
            x1,y1,x2,y2=[int(round(x)) for x in box]
            if not (0<=x1<x2<=w and 0<=y1<y2<=h):
                integrity=False;finding('COORDINATE_MAPPING_UNCONFIRMED')
    if integrity:
        fg=(mask>0).astype(np.uint8);total=int(fg.sum());fraction=total/fg.size
        if total==0:finding('EMPTY_FOREGROUND')
        elif total<cfg['min_pixels']:finding('INSUFFICIENT_PIXELS',pixels=total)
        elif fraction<=cfg['low_fraction']:finding('LOW_FOREGROUND')
        if fraction>=cfg['high_fraction']:finding('HIGH_FOREGROUND')
        n,labels,stats,_=cv2.connectedComponentsWithStats(fg,8)
        tight=np.zeros((h,w),bool);tight[y1:y2,x1:x2]=True
        components=assoc.components_from_cv(labels,stats,tight_box=(x1,y1,x2,y2),
            roi=tight,keep_overlap=cfg['component_overlap'])
        decision=assoc.evaluate(assoc.AssociationContext(
            components=components,feature_id=crop.feature_id,shape=group,
            granularity=getattr(crop,'granularity',None) or contracts.GRANULARITY_DEFAULT,
            tight_box=(x1,y1,x2,y2)),assoc.load_policy())
        association_ok=decision.method!='ambiguous'
        if not association_ok:
            finding('TARGET_ASSOCIATION_AMBIGUOUS',candidates=len(components),
                    rule_trace=decision.trace)
        elif decision.method=='dominant_component':
            finding('DOMINANT_COMPONENT_SELECTED',severity='diagnostic',**decision.metrics)
        elif decision.method=='multipart':
            finding('MULTIPART_KEPT',severity='diagnostic')
        selected=np.isin(labels,decision.selected) if association_ok else fg.astype(bool)
        out=np.where(selected,255,0).astype(np.uint8)
        kept=int(selected.sum());kept_inside=int(selected[y1:y2,x1:x2].sum())
        outside=(kept-kept_inside)/max(1,kept)
        if outside>cfg['outside_fraction']:finding('POSSIBLE_MERGE')
        center_hit=bool(fg[(y1+y2)//2,(x1+x2)//2])
        if not center_hit:finding('CENTER_POINT_MISS',severity='diagnostic')
        occupancy=kept_inside/((x2-x1)*(y2-y1))
        if group in {'compact','unknown'} and occupancy<0.3:finding('LOW_COMPACT_OCCUPANCY')
        if group=='unknown':finding('UNKNOWN_FEATURE_SHAPE')
        touches=bool(np.any(selected[0]) or np.any(selected[-1]) or np.any(selected[:,0]) or np.any(selected[:,-1]))
        if touches:finding('POSSIBLE_TRUNCATION')
        if polarity_ambiguous:finding('POLARITY_AMBIGUOUS')
        if not identity_verified:finding('FEATURE_IDENTITY_UNCERTAIN')
        metrics.update(foreground_pixels=total,foreground_fraction=fraction,components=n-1,
                       associated_components=len(components),outside_before_clipping=outside,
                       center_hit=center_hit,occupancy=occupancy,shape_group=group,
                       association_method=decision.method,association_status=decision.status,
                       association_dropped_components=len(decision.dropped),
                       association_removed_pixels=total-kept,
                       association=dict(decision.metrics))
        changes=[]
        if stages:
            previous=None
            for name,value in stages.items():
                row={'stage':name,'pixels':int((value>0).sum()),'shape':list(value.shape)}
                if previous is not None and previous.shape==value.shape:
                    row['added']=int(((value>0)&(previous==0)).sum());row['removed']=int(((value==0)&(previous>0)).sum())
                    row['change_fraction']=(row['added']+row['removed'])/max(1,int((previous>0).sum()))
                    if row['change_fraction']>cfg['large_postprocess_fraction']:finding('LARGE_POSTPROCESS_CHANGE',stage=name)
                changes.append(row);previous=value
            metrics['postprocess']=changes
        else:finding('POSTPROCESS_UNAUDITED')
    codes=[f['code'] for f in findings]
    base_block=[f['code'] for f in findings if f.get('severity')!='diagnostic'
                and f['code'] not in {'CENTER_POINT_MISS','POSSIBLE_TRUNCATION'}]
    trusted=integrity and identity_verified and not base_block
    uses={u:{'allowed':False,'reasons':list(codes)} for u in contracts.QUALITY_USES}
    # 存在性只由已验证独立观察授予，不因review mask有前景自动转阳性。
    uses['existence']={'allowed':bool(integrity and identity_verified),'reasons':[] if integrity and identity_verified else codes}
    uses['morphology']={'allowed':bool(trusted),'reasons':[] if trusted else codes}
    calibrated=crop.px_per_um is not None and crop.px_per_um>0
    for u in ('length','area','width'):
        allowed=bool(trusted and calibrated and 'POSSIBLE_TRUNCATION' not in codes)
        if (u=='width' and not allowed and trusted and calibrated
                and integrity and decision.method=='dominant_component'
                and decision.allowed_uses.get('width')
                and cfg.get('local_width_on_truncation_validated')):
            # 宽度沿骨架测量，不依赖目标完整长度；但仍须显式通过
            # local_width_on_truncation_validated，默认 false，不静默放宽。
            allowed=True
        uses[u]={'allowed':allowed,'reasons':[] if allowed else codes+([] if calibrated else ['CALIBRATION_MISSING'])}
    if integrity and decision.method=='dominant_component':
        # 碎屑已分开、但长度/面积可能被截断或受丢弃块影响；宽度仍可用。
        for u in ('length','area'):
            if not decision.allowed_uses.get(u,True):
                uses[u]={'allowed':False,'reasons':codes+['DOMINANT_COMPONENT_NOT_QUALIFIED']}
    if not integrity:
        for u in uses:uses[u]['allowed']=False
    bindings=dict(bindings)
    for key,path in [('image',crop.source_image_path),('crop',crop.png),('crop_meta',crop.json_path)]:
        if Path(path).is_file():bindings[key]=sha(path)
        else:
            integrity=False;finding('SOURCE_UNREADABLE',source=key);bindings[key]='0'*64
    bindings.setdefault('mask',hashlib.sha256(out.tobytes()+str(out.shape).encode()).hexdigest() if out is not None else '0'*64)
    if not integrity:
        for use in uses.values():use.update(allowed=False,reasons=[f['code'] for f in findings])
    assessment={'schema':'quality-assessment/v1','integrity':integrity,'bindings':bindings,
        'findings':findings,'metrics':metrics,'uses':uses,'policy_version':cfg['version'],
        'association_policy_version':decision.policy_version if integrity else None,
        'association':({'method':decision.method,'status':decision.status,
                        'selected':list(decision.selected),'dropped':list(decision.dropped),
                        'metrics':dict(decision.metrics)} if integrity else None),
        'evaluator_version':'quality_eval/v1','risk_validated':False}
    return out,contracts.quality_assessment(assessment)


def replay(generation_dir,crop,out_dir,*,identity_verified=False):
    """已有生成图离线重放到新目录；原始、各后处理阶段与用途评估全部保留。"""
    from igm_segment import generated_to_mask,decode_generated,write_img
    source=Path(generation_dir);out=Path(out_dir);out.mkdir(parents=True,exist_ok=True)
    meta=json.loads((source/'meta.json').read_text(encoding='utf-8'))
    raw=(source/'gen_raw.png').read_bytes();gen,alpha,error=decode_generated(raw)
    if gen is None:raise contracts.ContractError('生成图不可解码')
    image=cv2.imdecode(np.fromfile(crop.png,dtype=np.uint8),cv2.IMREAD_UNCHANGED)
    stages={}
    mask,info=generated_to_mask(gen,alpha,image.shape[:2],meta['letterbox'],
        threshold=meta.get('threshold'),invert=meta.get('invert'),max_hole_ratio=meta.get('max_hole_ratio',0),
        box=crop.tight_box_local,audit_stages=stages)
    before=stages['pre_box']
    bindings={'generation':sha(source/'gen_raw.png'),'crop':sha(crop.png),'crop_meta':sha(crop.json_path)}
    mapped=(meta.get('image_hw')==list(image.shape[:2]) and meta.get('size_match') is True)
    associated,assessment=assess(before,crop,bindings=bindings,identity_verified=identity_verified,
        polarity_ambiguous=bool(meta.get('polarity_conflict')),mapping_verified=mapped,stages=stages)
    for name,value in stages.items():write_img(str(out/(name+'.png')),value,'.png')
    write_img(str(out/'associated_mask.png'),associated,'.png')
    assessment['bindings']['mask']=sha(out/'associated_mask.png')
    assessment['bindings']['generation_metadata']=sha(source/'meta.json')
    assessment['source_generation_dir']=str(source.resolve())
    contracts.quality_assessment(assessment)
    (out/'assessment.json').write_text(json.dumps(assessment,ensure_ascii=False,indent=2),encoding='utf-8')
    return assessment
