"""受信用途适配：只消费当前证据库绑定评估，把测量记录投影成规则引擎可用的行。

旧 flags 字段保留兼容，但只有通过合格质量投影的字段才向下游暴露；量纲不被许可时
清空数值并标记 unqualified，形态许可不足时剥离形状字段，绝不修改原记录本身。

主要定义：
- project: 遍历当前证据快照，校验图像/crop/mask 绑定与用途资格后输出合格测量行。

关联：
- 调用方 「lib/agent/planning_choices.py」、「lib/agent/semantic_execution.py」、「lib/application/workflow_entry.py」、「lib/workflow/workflow_runtime.py」。
- 直接依赖 「lib/runtime/storage.py」。
"""
import copy
import contracts
from TCM_Fusion.lib.runtime.storage import file_hash,IntegrationError


def dimension_use(field):
    """原用途投影的统一量纲分类；供语义目标检查复用，不改现有字段语义。"""
    return 'width' if 'width' in field else 'area' if 'area' in field else 'length'


def project(runtime, *, scope_id=None, measurement_id=None):
    """默认返回全案用途投影；实例级读取可限定分割来源与测量依赖，使用相同的资格检查。"""
    session=runtime.workflow;snap=session.store.current();rows=[]
    for entry in snap['records']:
        if ((scope_id is not None and entry['scope'] != scope_id) or
                (measurement_id is not None and measurement_id not in entry['depends_on'])):
            continue
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
        pd=row.get('primary_dimension','');use=dimension_use(pd)
        if not qa['uses'][use]['allowed'] or use=='area' or not any(x in pd for x in ('width','feret_max')):
            row['primary_value_um']=None;row['calibration_status']='unqualified'
        row['flags']=[f for f in contracts.parse_flags(row.get('flags')) if f!='igm_review']
        if not qa['uses']['morphology']['allowed']:
            from rule_engine import SHAPE_KEYS
            for key in SHAPE_KEYS:row.pop(key,None)
        for key in list(row):
            if any(s in key for s in ('feret','width','area','perimeter')) and key not in {'primary_dimension'}:
                field_use=dimension_use(key)
                if not qa['uses'][field_use]['allowed']:row[key]=None
        rows.append(row)
    return rows
