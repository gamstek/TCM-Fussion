"""受信用途适配：只消费当前证据库绑定评估；旧flags保留，原规则仅见合格投影。"""
import copy
import contracts
from .storage import file_hash,IntegrationError


def project(runtime):
    session=runtime.workflow;snap=session.store.current();rows=[]
    for entry in snap['records']:
        row=copy.deepcopy(entry['data']);row.pop('quality_assessment',None)
        scope=entry['scope'];seg=snap['artifacts'].get(scope,{})
        qa=session.assessment(seg)
        if qa is None:continue  # 旧工件无新评估，保持保守。
        crop=snap['artifacts'].get(seg.get('crop_id'),{})
        if not crop.get('valid'):continue
        m=contracts.CropMeta.from_json(crop['path'])
        if qa['bindings'].get('image')!=runtime.images[seg['image_id']]['sha256']:
            raise IntegrationError('QUALITY_SCOPE','原图身份与评估不符')
        if qa['bindings'].get('crop')!=file_hash(m.png) or qa['bindings'].get('crop_meta')!=file_hash(m.json_path):
            raise IntegrationError('QUALITY_SCOPE','工件用途资格失配')
        from pathlib import Path
        mask=Path(snap['artifacts'][seg['assessment_id']]['path']).parent/'associated_mask.png'
        if not mask.is_file() or file_hash(mask)!=qa['bindings'].get('mask'):
            raise IntegrationError('QUALITY_SCOPE','关联mask变化')
        if not qa['integrity'] or not qa['uses']['existence']['allowed']:continue
        # 仅投影，不修改原记录旗标。按量纲显式剥离不允许的量化和形状。
        pd=row.get('primary_dimension','');use='width' if 'width' in pd else 'area' if 'area' in pd else 'length'
        if not qa['uses'][use]['allowed'] or use=='area' or not any(x in pd for x in ('width','feret_max')):
            row['primary_value_um']=None;row['calibration_status']='unqualified'
        row['flags']=[f for f in contracts.parse_flags(row.get('flags')) if f!='igm_review']
        if not qa['uses']['morphology']['allowed']:
            from rule_engine import SHAPE_KEYS
            for key in SHAPE_KEYS:row.pop(key,None)
        for key in list(row):
            if any(s in key for s in ('feret','width','area','perimeter')) and key not in {'primary_dimension'}:
                field_use='width' if 'width' in key else 'area' if 'area' in key else 'length'
                if not qa['uses'][field_use]['allowed']:row[key]=None
        rows.append(row)
    return rows
