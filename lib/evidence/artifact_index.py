"""绑定样品下的只读 B 工件索引：契约检查与像素质量检查分开标记。

扫描只收集 crop/mask/measurement/measure_manifest 与已知状态文件，跳过符号链接与
越界路径；仅输出已知状态字段，不把日志、模型返回或目录药材名送入规划器。

主要定义：
- ArtifactIndex: 持有待扫描根目录，提供 scan 生成工件清单、计数、缺失阶段与索引摘要。

关联：
- 调用方 「lib/tools/tool_builtins.py」、「tests/workflow/test_workflow.py」。
- 直接依赖 「lib/runtime/storage.py」。
"""
from pathlib import Path
import contracts
from TCM_Fusion.lib.runtime.storage import file_hash, read_json, digest


class ArtifactIndex:
    def __init__(self, roots=()):
        self.roots = [Path(r).resolve() for r in roots]

    def scan(self):
        items = []
        for root in self.roots:
            if not root.is_dir():
                items.append({'kind':'root','status':'missing','id':digest(str(root))})
                continue
            for p in sorted(root.rglob('*')):
                if not p.is_file() or p.is_symlink() or not p.resolve().is_relative_to(root):
                    continue
                rel = p.relative_to(root).as_posix()
                kind = ('crop' if p.suffix=='.json' and p.parent.name=='crops' else
                        'mask' if p.suffix.lower()=='.png' and 'masks' in p.parts else
                        'measurement' if p.name.endswith('_measurements.jsonl') else
                        'measure_manifest' if p.name.endswith('__measure_manifest.csv') else
                        'status' if p.name in {'job.json','instances.json','meta.json','run_manifest.json','operations.json'} else None)
                if kind is None: continue
                item={'id':digest([str(root),rel]),'kind':kind,'relative_path':rel,
                      'sha256':file_hash(p),'status':'present','pixel_quality':'not_checked'}
                try:
                    if kind=='crop':
                        m=contracts.CropMeta.from_json(str(p))
                        item.update(feature_id=m.feature_id, image_stem=m.stem,
                                    scale_status='valid' if m.px_per_um else 'missing',
                                    status='present' if Path(m.png).is_file() else 'missing_pair')
                    elif kind=='measurement':
                        rows=list(contracts.iter_measurement_records(str(p)))
                        item.update(count=len(rows),status='present' if rows else 'empty')
                    elif kind=='measure_manifest':
                        import csv
                        with p.open(encoding='utf-8-sig',newline='') as f:
                            rows=[contracts.MeasureManifestRow.from_row(r) for r in csv.DictReader(f)]
                        item.update(count=len(rows),scale_status='valid' if rows and all(r.px_per_um for r in rows) else 'missing')
                        states=[r.igm_status for r in rows]
                        item['segmentation_status_counts']={s:states.count(s) for s in set(states)}
                    elif kind=='status':
                        value=read_json(p)
                        # 只输出已知状态，不把日志/模型返回/目录药材名送入规划器。
                        if isinstance(value,dict):
                            item.update({k:value[k] for k in ('status','error_code','stage') if k in value})
                            if p.name=='operations.json':
                                item['operations']=[{k:r[k] for k in ('tool','target_id','status','error_code') if k in r}
                                                    for r in value.get('entries',{}).values()]
                            if p.name=='instances.json':
                                states=[r.get('igm_status','unknown') for r in value.get('instances',[])]
                                item.update(instances_status_available=True,
                                            segmentation_status_counts={s:states.count(s) for s in set(states)},
                                            instance_count=len(states))
                except (ValueError,OSError,TypeError,KeyError,AttributeError,contracts.ContractError):
                    item['status']='invalid_contract'
                items.append(item)
        counts={k:sum(i['kind']==k for i in items) for k in ('crop','mask','measurement','measure_manifest')}
        return {'artifacts':items,'counts':counts,'missing_stages':[k for k,v in counts.items() if not v],
                'index_digest':digest(items),'quality_scope':'file_and_contract_only'}
