"""真实B阶段包装：固定入口子进程/具名库worker；单视野或单crop，不重跑整条B。"""
import shutil
from pathlib import Path
import contracts
import paths
from .controlled_process import run
from .storage import IntegrationError, read_json, write_json, file_hash, digest, project_path


class WorkflowBackend:
    @staticmethod
    def code_signature():
        root=Path(paths.PROJECT_ROOT)
        roots=[Path(paths.DETECTION_LIB_DIR),Path(paths.SEGMENTATION_LIB_DIR),Path(paths.COMMON_DIR),
               Path(paths.GEOM_DIR),Path(paths.JUDGE_SCRIPTS_DIR),root/'TCM_Fusion/lib',root/'TCM_Fusion/cli']
        files={p for folder in roots for p in folder.rglob('*.py')}
        files.update([root/'contracts.py',root/'feature_taxonomy.json',
                      Path(paths.SEGMENTATION_CONFIG_DIR)/'quality_policy.json',
                      Path(paths.SEGMENTATION_CONFIG_DIR)/'association_policy.json',
                      Path(paths.DETECTION_CLI_DIR)/'detect_sample.py',
                      Path(paths.SEGMENTATION_CLI_DIR)/'igm_segment_batch.py'])
        return digest([(p.relative_to(root).as_posix(),file_hash(p)) for p in sorted(files)])

    def execute(self, session, tool, target, work, context):
        def command(args, python=session.python, log='stage.log'):
            run([str(python),'-X','utf8',*map(str,args)],context_path=context,timeout=session.remaining_seconds(),log_path=work/log,cancel_event=session.cancel_event)
        if tool == 'roi_screen':
            out = work / 'screening'
            ctx = work / 'screen.json'
            write_json(ctx, {'tool': tool, 'source': str(target['source']),
                             'regions': str(target['regions']),
                             'feature_id': target.get('feature_id'),
                             'image_id': target['image_id'], 'output': str(out)})
            command(['-m', 'TCM_Fusion.cli.workflow_worker', '--context', ctx])
            return {'screen': str(out / 'screen.json')}
        if tool in {'propose_regions', 'locate_feature'}:
            out = work / 'detection'
            ctx = work / 'recovery.json'
            data = {'tool': tool, 'source': str(target['source']), 'output': str(out)}
            if tool == 'locate_feature':
                data.update(roi=target['roi'], feature_id=target['feature_id'],
                            px_per_um=target.get('px_per_um'),
                            locate_expand=session.config.get('locate_expand', 1.5),
                            existing_boxes=session.existing_boxes(target['image_id']))
            write_json(ctx, data)
            command(['-m', 'TCM_Fusion.cli.workflow_worker', '--context', ctx])
            if tool == 'propose_regions':
                return {'regions': str(out / 'regions.json')}
            return {'crops_dir': str(out / 'crops'), 'completion': str(out / 'region_result.json')}
        if tool in {'detect_fov_initial', 'detect_region'}:
            initial = tool == 'detect_fov_initial'
            out = work / 'detection'
            config = session.config['yolo']
            settings = {key: str(project_path(config[key]))
                        for key in ('weights', 'class_map', 'detector_config')}
            if 'confidence' in config:
                settings['confidence'] = config['confidence']
            source = str(target['path'] if initial else target['source'])
            ctx = work / 'yolo.json'
            roi = {} if initial else {'roi': target['roi']}
            write_json(ctx, {**settings, **roi,
                             'source': source, 'output': str(work / 'boxes.json'),
                             'signature': session.yolo_signature})
            command(['-m', 'TCM_Fusion.cli.yolo_worker', '--context', ctx], session.yolo_python, 'yolo.log')
            crop_context = {'tool': tool, 'source': source, 'output': str(out),
                            'boxes': str(work / 'boxes.json'), **roi}
            if not initial:
                crop_context.update(px_per_um=target.get('px_per_um'),
                                    existing_boxes=session.existing_boxes(target['image_id']),
                                    replace_id=target.get('replace_id'))
            ctx = work / 'crop_context.json'
            write_json(ctx, crop_context)
            command(['-m', 'TCM_Fusion.cli.workflow_worker', '--context', ctx])
            completion = out / ('crops.json' if initial else 'region_result.json')
            if not completion.is_file():
                raise IntegrationError('DETECTION_INCOMPLETE', '检测没有完成清单')
            return {'crops_dir': str(out / 'crops'), 'completion': str(completion)}
        if tool == 'calibrate_fov':
            out = work / 'detection'
            ctx = work / 'worker.json'
            write_json(ctx, {'tool': tool, 'source': target['source'], 'output': str(out)})
            command(['-m', 'TCM_Fusion.cli.workflow_worker', '--context', ctx])
            return {'scale': str(out / 'scale.json')}
        if tool=='classify_target':
            # 单crop类别复核：复用检测侧 classify_crop（提示词/解析/白名单只此一份），
            # 在子进程内跑，网络请求由 worker_bootstrap 的HTTP守卫按请求键记账。
            meta=contracts.CropMeta.from_json(target['path'])
            ctx={'tool':tool,'crop_id':meta.crop_id,'crop_json':str(target['path']),
                 'crop_png':str(meta.png),'source':meta.source_image_path,
                 'tight_box':meta.tight_box,'output':str(work/'classification')}
            if target.get('identity_required'):
                ctx['tight_box_local'] = meta.tight_box_local
            f=work/'worker.json';write_json(f,ctx)
            out=work/'classification'
            command(['-m','TCM_Fusion.cli.classify_worker','--context',f])
            if not (out/'classification.json').is_file():
                raise IntegrationError('CLASSIFY_INCOMPLETE','分类阶段没有产出清单')
            doc=read_json(out/'classification.json')
            # required crop 再做一次 YOLO 复核：GLM 标框/裁剪后把 tight crop 交给训练后的 YOLO。
            # 命中同类时按 YOLO:GLM=7:3 融合；未命中/YOLO 失败则完全保留 GLM 结论。
            if target.get('identity_required'):
                try:
                    yconfig=session.config['yolo']
                    settings={key:str(project_path(yconfig[key])) for key in
                              ('weights','class_map','detector_config')}
                    yctx=work/'yolo_crop.json'
                    write_json(yctx,{**settings,'source':str(meta.png),
                                     'output':str(work/'yolo_crop_boxes.json'),
                                     'signature':session.yolo_signature})
                    command(['-m','TCM_Fusion.cli.yolo_worker','--context',yctx],
                            session.yolo_python,'yolo_crop.log')
                    yolo=read_json(work/'yolo_crop_boxes.json')
                    target_label=meta.object_label
                    matches=[float(d.get('score') or 0) for d in yolo.get('dets',[])
                             if d.get('label')==target_label]
                    yscore=max(matches) if matches else None
                    glm_score={'high':0.9,'medium':0.7,'low':0.5}.get(
                        str(doc.get('confidence') or ''),0.5)
                    doc['glm_confidence']=doc.get('confidence')
                    doc['yolo_confidence']=yscore
                    doc['yolo_checked']=True
                    if yscore is not None:
                        conflict=bool(doc.get('label') and doc.get('label')!=target_label)
                        doc['yolo_conflict']=conflict
                        if not conflict:
                            fused=0.7*yscore+0.3*glm_score
                            doc['label']=target_label
                            doc['feature_id']=meta.feature_id
                            doc['reject']=''
                            doc['confidence_score']=round(fused,4)
                            doc['confidence_source']='yolo_glm_7_3'
                            doc['confidence']=('high' if fused>=0.75 else
                                               'medium' if fused>=0.55 else 'low')
                        else:
                            doc['confidence_source']='glm'
                    else:
                        doc['confidence_source']='glm'
                except Exception as exc:  # YOLO 不可用不拖垮 GLM 主结果
                    doc['yolo_checked']=False
                    doc['yolo_error']=str(exc)[:200]
                    doc['confidence_source']='glm'
                write_json(out/'classification.json',doc)
            return {'classification':str(out)}
        if tool=='segment_instance':
            src=Path(target['path']);meta=contracts.CropMeta.from_json(str(src))
            contracts.check_crops([meta]);inp=work/'crops';inp.mkdir()
            shutil.copy2(src,inp/src.name);shutil.copy2(meta.png,inp/Path(meta.png).name)
            out=work/'segmentation'
            command([Path(paths.SEGMENTATION_CLI_DIR)/'igm_segment_batch.py','--crops',inp,'--out-dir',out,
                     '--box-association', session.config.get('box_association','region_component')])
            doc=read_json(out/'instances.json')
            if not doc.get('instances'):raise IntegrationError('EMPTY_MASK','没有分割实例')
            return {'segmentation':str(out),'quality':'valid' if all(
                i.get('igm_status')=='ok' for i in doc['instances']) else 'review'}
        if tool=='measure_instances':
            # 局部阶段各自从inst01开始；先在新交付副本做确定性命名空间，
            # 使原measure输出的instance_id仍严格对应实际mask文件名。
            import csv
            plan = getattr(session, 'review_measure_plan', lambda target=None: None)(target)
            diagnostic = bool(plan)
            component_ids = set(plan.get('component_ids') or []) if plan else set()
            component_policy = plan.get('component_policy') if plan else 'as_delivered'
            source=Path(target['masks']);delivery=work/'delivery';delivery.mkdir()
            rows=[];offset=int(digest(target['id'])[:8],16)*10000
            for f in source.glob('*__measure_manifest.csv'):
                rows=[]
                with f.open(encoding='utf-8-sig',newline='') as stream:
                    for raw in csv.DictReader(stream):
                        row=contracts.MeasureManifestRow.from_row(raw)
                        if row.instance_id is None or not 0<=row.instance_id<10000:
                            raise IntegrationError('INSTANCE_ID','局部mask编号非法')
                        src=(source/row.inst_mask).resolve()
                        if not src.is_relative_to(source.resolve()) or not src.is_file():
                            raise IntegrationError('MASK_SCOPE','mask越出交付目录或不存在')
                        if diagnostic:
                            # Decision LLM 选择 split/diagnostic 后，只测最大的单一连通域；
                            # 原 review mask 保留，不改原证据，只在交付副本里做诊断隔离。
                            import cv2
                            import numpy as np
                            image=cv2.imread(str(src), cv2.IMREAD_GRAYSCALE)
                            if image is None:
                                raise IntegrationError('MASK_INVALID','诊断测量无法读取mask')
                            n,labels,stats,_=cv2.connectedComponentsWithStats((image>0).astype(np.uint8),8)
                            if n>1 and component_ids:
                                image=np.where(np.isin(labels,sorted(component_ids)),255,0).astype(np.uint8)
                            elif n>1 and component_policy in {'largest'}:
                                largest=1+int(np.argmax(stats[1:,cv2.CC_STAT_AREA]))
                                image=np.where(labels==largest,255,0).astype(np.uint8)
                            elif n>1 and component_policy in {'all','merge'}:
                                image=np.where(labels>0,255,0).astype(np.uint8)
                            elif n>1:
                                largest=1+int(np.argmax(stats[1:,cv2.CC_STAT_AREA]))
                                image=np.where(labels==largest,255,0).astype(np.uint8)
                            destination=delivery/f"{Path(row.source_image_path).stem}__{row.feature_id}__inst{offset+row.instance_id}.png"
                            ok,encoded=cv2.imencode('.png',image)
                            if not ok:raise IntegrationError('MASK_ENCODE','诊断mask编码失败')
                            destination.write_bytes(encoded.tobytes())
                            raw['igm_status']='ok'
                        if session.purpose_quality():
                            qa=session.assessment(target)
                            if not qa or not qa['integrity']:raise IntegrationError('QUALITY_SCOPE','无受信用途资格')
                            aid=target.get('assessment_id');assessment=session.store.current()['artifacts'][aid]
                            src=Path(assessment['path']).parent/'associated_mask.png'
                            if file_hash(src)!=qa['bindings']['mask']:raise IntegrationError('QUALITY_SCOPE','关联mask变化')
                            if rows:raise IntegrationError('QUALITY_MULTIPLE_INSTANCES','单目标资格不能用于多实例重复测量')
                        iid=offset+row.instance_id;name=f"{Path(row.source_image_path).stem}__{row.feature_id}__inst{iid}.png"
                        if not diagnostic:
                            shutil.copy2(src,delivery/name)
                        rows.append({**raw,'instance_id':iid,'inst_mask':name,'herb':''})
                with (delivery/f.name).open('w',encoding='utf-8-sig',newline='') as stream:
                    w=csv.DictWriter(stream,fieldnames=contracts.MANIFEST_COLUMNS);w.writeheader();w.writerows(rows)
            if (source/'crops').is_dir():shutil.copytree(source/'crops',delivery/'crops')
            out=work/'measurement';out.mkdir()
            command([Path(paths.GEOM_DIR)/'measure.py','--masks',delivery,
                     '--out-prefix',out/'result','--split-policy','all'])
            f=out/'result_measurements.jsonl'
            rows=list(contracts.iter_measurement_records(str(f)))
            if not rows:raise IntegrationError('EMPTY_MEASUREMENT','测量没有实例')
            return {'measurements':str(f),'delivery':str(delivery),'diagnostic_review':diagnostic,
                    'component_selection':(f"ids:{sorted(component_ids)}" if component_ids else component_policy)
                    if diagnostic else 'as_delivered'}
        raise IntegrationError('TOOL_UNIMPLEMENTED','没有真实阶段实现')
