"""真实B阶段包装：固定入口子进程/具名库worker；单视野或单crop，不重跑整条B。"""
import shutil
from pathlib import Path
import contracts
import paths
from .controlled_process import run
from .image_tiling import uniform_tiles
from .storage import IntegrationError, read_json, write_json, file_hash, digest, project_path


def _timeout(session):
    fn = getattr(session, 'call_timeout', None)
    return fn() if callable(fn) else session.remaining_seconds()


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
            run([str(python),'-X','utf8',*map(str,args)],context_path=context,timeout=_timeout(session),log_path=work/log,cancel_event=session.cancel_event)
        from .feature_profiles import hint_text, profile_for
        def feature_hint(feature_id):
            override = session.config.get('feature_profiles') or {}
            profile = profile_for(feature_id, override)
            return hint_text(profile)
        if tool == 'roi_screen_batch':
            from concurrent.futures import ThreadPoolExecutor
            items=list(target.get('items') or [])
            if not items:
                raise IntegrationError('SCREEN_INPUT','批量筛选没有输入视野')
            out=work/'screening_batch';out.mkdir(parents=True,exist_ok=True)
            workers=max(1,min(int(session.config.get('screen_concurrency',2)),len(items)))
            def one(index_item):
                index,item=index_item
                sub=out/f'{index:02d}';sub.mkdir()
                ctx=sub/'screen.json'
                write_json(ctx,{'tool':'roi_screen','source':str(item['source']),
                                'regions':str(item['regions']),'feature_id':item.get('feature_id'),
                                'image_id':item['image_id'],'output':str(sub/'result'),
                                'screen_max_rois':int(session.config.get('screen_max_rois',4)),
                                'feature_hint':feature_hint(item.get('feature_id'))})
                run([str(session.python),'-X','utf8','-m','TCM_Fusion.cli.workflow_worker',
                     '--context',str(ctx)],context_path=context,timeout=_timeout(session),
                    log_path=sub/'stage.log',cancel_event=session.cancel_event)
                return str(sub/'result'/'screen.json')
            with ThreadPoolExecutor(max_workers=workers) as pool:
                screens=list(pool.map(one,enumerate(items)))
            return {'screens':screens}
        if tool == 'roi_screen':
            out = work / 'screening'
            ctx = work / 'screen.json'
            write_json(ctx, {'tool': tool, 'source': str(target['source']),
                             'regions': str(target['regions']),
                             'feature_id': target.get('feature_id'),
                             'image_id': target['image_id'], 'output': str(out),
                             'screen_max_rois':int(session.config.get('screen_max_rois',4)),
                             'feature_hint': feature_hint(target.get('feature_id'))})
            command(['-m', 'TCM_Fusion.cli.workflow_worker', '--context', ctx])
            return {'screen': str(out / 'screen.json')}
        if tool in {'needle_scan', 'feature_scan'}:
            feature_id = target.get('feature_id')
            if tool == 'needle_scan':
                feature_id = feature_id or 'needle_crystal'
            elif not isinstance(feature_id, str) or not feature_id.strip():
                raise IntegrationError('FEATURE_SCAN_INPUT', 'feature_scan 必须显式提供 feature_id')
            from detect_core import FEATURE_ID
            from prompts import CLASSIFY_HINTS
            if not any(fid == feature_id and label in CLASSIFY_HINTS
                       for label, fid in FEATURE_ID.items()):
                raise IntegrationError('FEATURE_SCAN_INPUT', 'feature_scan 不支持该 feature_id')
            from concurrent.futures import ThreadPoolExecutor, as_completed
            from PIL import Image, ImageFilter, ImageStat
            source = Path(target['source'])
            prefix = 'feature_scan' if tool == 'feature_scan' else 'needle_scan'
            with Image.open(source) as handle:
                width, height = handle.size
            tile = max(256, min(2048, int(session.config.get(prefix + '_tile_size', 1024))))
            overlap = max(0.0, min(0.5, float(session.config.get(prefix + '_overlap', 0.15))))
            requested_tiles = int(target.get('max_tiles') or session.config.get(prefix + '_max_tiles', 4))
            boxes = uniform_tiles(width, height, tile=tile, overlap=overlap,
                                  max_tiles=max(1, requested_tiles))
            policy = str(session.config.get(prefix + '_tile_policy', 'prescreen'))
            if policy == 'prescreen' and len(boxes) > 1:
                def score(box):
                    try:
                        with Image.open(source) as image:
                            crop = image.crop(tuple(box)).convert('L')
                            crop.thumbnail((128, 128))
                            stat = ImageStat.Stat(crop)
                            edge = ImageStat.Stat(crop.filter(ImageFilter.FIND_EDGES)).mean[0]
                            return float(stat.stddev[0]) + 2.0 * float(edge)
                    except Exception:
                        return 0.0
                boxes = sorted(boxes, key=score, reverse=True)
            out = work / 'scan'
            out.mkdir(parents=True, exist_ok=True)
            workers = max(1, min(int(session.config.get(prefix + '_concurrency', 2)), len(boxes)))
            def one(index_box):
                index, box = index_box
                sub = out / f'tile_{index:02d}'
                sub.mkdir(parents=True, exist_ok=True)
                ctx = sub / 'locate.json'
                write_json(ctx, {'tool': 'locate_feature', 'source': str(source),
                                 'output': str(sub / 'detection'), 'roi': box,
                                 'feature_id': feature_id,
                                 'px_per_um': target.get('px_per_um'),
                                 'locate_expand': session.config.get('locate_expand', 1.5),
                                 'locate_min_box_side': session.config.get('locate_min_box_side', 90),
                                 'locate_min_roi_side': session.config.get('locate_min_roi_side', 256),
                                 'feature_hint': feature_hint(feature_id),
                                 'existing_boxes': []})
                run([str(session.python), '-X', 'utf8', '-m', 'TCM_Fusion.cli.workflow_worker',
                     '--context', str(ctx)], context_path=context,
                    timeout=_timeout(session), log_path=sub / 'stage.log',
                    cancel_event=session.cancel_event)
                return sub / 'detection'
            def has_crops(folder):
                return (folder / 'crops').is_dir() and any((folder / 'crops').glob('*.json'))
            errors = []
            if policy == 'prescreen' and boxes:
                # 先试扫最有纹理的 tile；无结果就停止，不再盲扫剩余 tile。
                try:
                    one((0, boxes[0]))
                except Exception as exc:
                    errors.append({'tile': 0, 'error': str(exc)[:200]})
                if not has_crops(out / 'tile_00' / 'detection'):
                    completion = out / 'scan_result.json'
                    write_json(completion, {'engine': 'glm_full_scan', 'count': 0,
                                            'tiles': boxes[:1], 'feature_id': feature_id,
                                            'pilot': True, 'stopped': 'pilot_empty',
                                            'errors': errors, 'coordinate_system': 'original_pixels'})
                    return {'crops_dir': str(out / 'crops'), 'completion': str(completion),
                            'tiles': boxes[:1], 'partial': bool(errors), 'errors': errors}
                remaining = list(enumerate(boxes))[1:]
            else:
                remaining = list(enumerate(boxes))
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(one, pair): pair[0] for pair in remaining}
                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as exc:
                        errors.append({'tile': futures[future], 'error': str(exc)[:200]})
            # 超时的 tile 也可能已经写出部分 detection；不要因为一个 tile 丢弃全部。
            dirs = sorted(p for p in out.glob('tile_*/detection') if p.is_dir())
            crops = out / 'crops'
            crops.mkdir(parents=True, exist_ok=True)
            count = 0
            for folder in dirs:
                source_crops = folder / 'crops'
                if not source_crops.is_dir():
                    continue
                for path in source_crops.iterdir():
                    if path.is_file():
                        shutil.copy2(path, crops / path.name)
                        if path.suffix == '.json':
                            count += 1
            completion = out / 'scan_result.json'
            write_json(completion, {'engine': 'glm_full_scan', 'count': count,
                                    'tiles': boxes, 'roi': None, 'feature_id': feature_id,
                                    'partial': bool(errors), 'errors': errors,
                                    'tile_policy': policy,
                                    'coordinate_system': 'original_pixels'})
            return {'crops_dir': str(crops), 'completion': str(completion), 'tiles': boxes,
                    'partial': bool(errors), 'errors': errors}
        if tool == 'hybrid_scan':
            from concurrent.futures import ThreadPoolExecutor, as_completed
            features = [str(x) for x in (target.get('features') or []) if str(x)]
            if not features:
                raise IntegrationError('HYBRID_INPUT', 'hybrid_scan 没有 feature')
            locate_cap = int(target.get('max_locate_calls') or 0)
            if locate_cap > 0:
                features = features[:max(1, locate_cap)]
                per_feature_tiles = max(1, locate_cap // max(1, len(features)))
            else:
                per_feature_tiles = None
            source = Path(target['source'])
            yolo_target = {'id': target['id'] + ':yolo', 'image_id': target['image_id'],
                           'path': str(source), 'source': str(source), 'parents': target.get('parents', [])}
            def run_yolo():
                return self.execute(session, 'detect_fov_initial', yolo_target,
                                    work / 'yolo', context)
            def run_feature(index_feature):
                index, fid = index_feature
                feature_target = {'id': target['id'] + ':feature:' + fid,
                                  'image_id': target['image_id'], 'source': str(source),
                                  'kind': 'feature_scan', 'feature_id': fid,
                                  'parents': target.get('parents', []),
                                  'px_per_um': target.get('px_per_um')}
                if per_feature_tiles:
                    feature_target['max_tiles'] = per_feature_tiles
                return fid, self.execute(session, 'feature_scan', feature_target,
                                         work / f'glm_{index:02d}', context)
            with ThreadPoolExecutor(max_workers=max(2, min(4, len(features) + 1))) as pool:
                yolo_future = pool.submit(run_yolo)
                feature_futures = {pool.submit(run_feature, pair): pair[1]
                                   for pair in enumerate(features)}
                yolo_result = None
                feature_results = []
                errors = []
                for future in as_completed([yolo_future, *feature_futures]):
                    try:
                        value = future.result()
                    except Exception as exc:
                        errors.append({'phase': 'yolo' if future is yolo_future else 'feature',
                                       'error': str(exc)[:200]})
                        continue
                    if future is yolo_future:
                        yolo_result = value
                    else:
                        feature_results.append(value)
            if not feature_results:
                raise IntegrationError('HYBRID_EMPTY', 'hybrid_scan 没有可用 feature 结果')
            merged = work / 'features'
            crops = merged / 'crops'
            crops.mkdir(parents=True, exist_ok=True)
            count = 0
            for _fid, result in feature_results:
                source_crops = Path(result['crops_dir'])
                if not source_crops.is_dir():
                    continue
                for path in source_crops.iterdir():
                    if path.is_file():
                        shutil.copy2(path, crops / path.name)
                        if path.suffix == '.json':
                            count += 1
            completion = merged / 'scan_result.json'
            write_json(completion, {'engine': 'glm_full_scan', 'count': count,
                                    'features': features, 'partial': bool(errors),
                                    'errors': errors, 'coordinate_system': 'original_pixels'})
            return {'detections': yolo_result or {},
                    'feature_scan': {'crops_dir': str(crops), 'completion': str(completion)},
                    'partial': bool(errors), 'errors': errors}
        if tool in {'propose_regions', 'locate_feature'}:
            out = work / 'detection'
            ctx = work / 'recovery.json'
            data = {'tool': tool, 'source': str(target['source']), 'output': str(out)}
            if tool == 'locate_feature':
                data.update(roi=target['roi'], feature_id=target['feature_id'],
                            px_per_um=target.get('px_per_um'),
                            locate_expand=session.config.get('locate_expand', 1.5),
                            locate_min_box_side=session.config.get('locate_min_box_side', 90),
                            locate_min_roi_side=session.config.get('locate_min_roi_side', 256),
                            feature_hint=feature_hint(target['feature_id']),
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
        if tool=='image_exemplar_audit':
            meta=contracts.CropMeta.from_json(target['path'])
            feature_id=meta.feature_id
            bank_path=project_path(session.config.get('image_exemplar_bank') or
                (Path(paths.PROJECT_ROOT)/'TCM_Fusion/config/feature_exemplar_bank.json'))
            bank=read_json(bank_path)
            entry=(bank.get('targets') or {}).get(feature_id) or {}
            positives=[x.get('crop') for x in (entry.get('positives') or []) if x.get('crop')]
            negatives=[x.get('crop') for x in (entry.get('negatives') or []) if x.get('crop')]
            if not positives or not negatives:
                raise IntegrationError('AUDIT_EXEMPLAR_MISSING',f'示例库没有 {feature_id} 的正例或负例')
            np=max(1,int(session.config.get('image_exemplar_positive',4)))
            nn=max(1,int(session.config.get('image_exemplar_negative',4)))
            out=work/'image_audit';ctx=work/'audit.json'
            write_json(ctx,{'tool':tool,'target':str(meta.png),'target_id':meta.crop_id,
                'target_label':feature_id,'feature_id':feature_id,
                'positives':positives[:np],'negatives':negatives[:nn],'output':str(out)})
            command(['-m','TCM_Fusion.cli.image_audit_worker','--context',ctx])
            if not (out/'audit.json').is_file():
                raise IntegrationError('AUDIT_INCOMPLETE','图片示例审计没有产出 audit.json')
            return {'audit':str(out/'audit.json'),'contact_sheet':str(out/'contact.jpg')}
        if tool=='classify_crops_batch':
            results=[];failures=[]
            for index,crop in enumerate(target.get('crops') or []):
                try:
                    result=self.execute(session,'classify_target',{**crop,'batch_cnn_only':True},
                                        work/f'classify_{index}',context)
                    results.append({'crop_id':crop['id'],'classification':result.get('classification'),
                                    'status':'done'})
                except Exception as exc:
                    failures.append({'crop_id':crop['id'],'status':'failed',
                                     'error_code':getattr(exc,'code','CLASSIFY_FAILED'),
                                     'error':str(exc)[:240]})
            if not results and not failures:
                raise IntegrationError('CLASSIFY_EMPTY','批量分类没有输入')
            return {'classifications':results,'failures':failures}
        if tool=='classify_target':
            # 单crop类别复核：CNN 批量初判，高置信直接落 classification；否则 GLM 兜底。
            meta=contracts.CropMeta.from_json(target['path'])
            cnn_info=None
            cnn_error=None
            if session.config.get('cnn_weights') and session.config.get('cnn_meta'):
                try:
                    cache=session.root/'cnn'/'predictions.json'
                    cnn_doc=read_json(cache) if cache.is_file() else None
                    rows={str(row.get('id')):row for row in ((cnn_doc or {}).get('rows') or [])}
                    if cnn_doc is None or target['id'] not in rows:
                        # cache 只含首次批次的 crop；后续新 crop 到来时失效重跑。
                        crops=[]
                        for tid,crop in session.targets().items():
                            if crop.get('kind')!='crop':
                                continue
                            m=contracts.CropMeta.from_json(crop['path'])
                            crops.append({'id':tid,'png':str(m.png)})
                        if not crops:
                            raise IntegrationError('CNN_EMPTY','CNN 没有待分类 crop')
                        weights=project_path(session.config.get('cnn_weights') or
                            '分类器设计与权重/cls_runs/abl_320_lb_e40_ls01_mix01_s20261031/best.pt')
                        cnn_meta=project_path(session.config.get('cnn_meta') or
                            '分类器设计与权重/cls_runs/abl_320_lb_e40_ls01_mix01_s20261031/model_meta.json')
                        if not weights.is_file() or not cnn_meta.is_file():
                            raise IntegrationError('CNN_ASSET_MISSING','CNN 权重或 meta 缺失')
                        cnn_output=session.root/'cnn'/'predictions.json'
                        cnn_context=work/'cnn_context.json'
                        write_json(cnn_context,{'crops':crops,'weights':str(weights),'meta':str(cnn_meta),
                            'output':str(cnn_output),'batch_size':int(session.config.get('cnn_batch_size',16))})
                        run([str(session.yolo_python),'-X','utf8','-m',
                             'TCM_Fusion.cli.cnn_classifier_worker','--context',str(cnn_context)],
                            context_path=context,timeout=_timeout(session),
                            log_path=work/'cnn.log',cancel_event=session.cancel_event)
                        if not cnn_output.is_file():
                            raise IntegrationError('CNN_INCOMPLETE','CNN 没有产出 predictions.json')
                        cnn_doc=read_json(cnn_output)
                        rows={str(row.get('id')):row for row in (cnn_doc.get('rows') or [])}
                    cnn_info=rows.get(target['id'])
                    threshold=float(session.config.get('cnn_threshold',.75))
                    entropy_limit=float(session.config.get('cnn_entropy_threshold',.8))
                    if cnn_info:
                        top1=str(cnn_info.get('top1') or '')
                        prob=float(cnn_info.get('top1_prob') or 0)
                        entropy=float(cnn_info.get('entropy') or 1)
                        other_fallback=bool(session.config.get('cnn_other_fallback',True))
                        direct=(top1 not in {'other',''} and prob>=threshold and entropy<=entropy_limit
                                and not (other_fallback and top1=='other'))
                        if direct:
                            labels={}
                            try:
                                tax=read_json(Path(paths.PROJECT_ROOT)/'feature_taxonomy.json')
                                labels={x.get('feature_id'):x.get('zh_name') for x in tax.get('features',[])}
                            except Exception:
                                pass
                            out=work/'classification';out.mkdir(parents=True,exist_ok=True)
                            doc={'label':labels.get(top1,top1),'raw_label':cnn_info.get('top1_class'),
                                'feature_id':top1,'second':'','confidence':'high',
                                'reason':'CNN 高置信初判','source':'cnn','focus':None,
                                'sent_hw':None,'granularity':meta.granularity,
                                'confidence_score':round(prob,6),'confidence_scale':'cnn_prob',
                                'crop_id':meta.crop_id,'crop_png':str(meta.png),
                                'glm_confidence':None,'yolo_confidence':None,'yolo_checked':False,
                                'confidence_source':'cnn','cnn_score':round(prob,6),
                                'cnn_entropy':round(entropy,6),'cnn_topk':cnn_info.get('topk') or []}
                            write_json(out/'classification.json',doc)
                            return {'classification':str(out)}
                except Exception as exc:
                    cnn_error=str(exc)[:240]
            if target.get('batch_cnn_only'):
                out=work/'classification'
                doc={'label':'待复核','reject':'pending_review','raw_label':'待复核',
                     'feature_id':'','second':'','confidence':'low',
                     'reason':'CNN低置信/other；批量动作不自动GLM',
                     'confidence_score':0.0,'confidence_source':'cnn',
                     'crop_id':meta.crop_id,'crop_png':str(meta.png),
                     'glm_confidence':None,'yolo_confidence':None,'yolo_checked':False,
                     'cnn_score':None,'cnn_entropy':None,'cnn_topk':[]}
                write_json(out/'classification.json',doc)
                return {'classification':str(out)}
            ctx={'tool':tool,'crop_id':meta.crop_id,'crop_json':str(target['path']),
                 'crop_png':str(meta.png),'source':meta.source_image_path,
                 'tight_box':meta.tight_box,'output':str(work/'classification'),
                 'feature_hint':feature_hint(meta.feature_id)}
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
                if cnn_info:
                    doc['cnn_score']=cnn_info.get('top1_prob')
                    doc['cnn_entropy']=cnn_info.get('entropy')
                    doc['cnn_topk']=cnn_info.get('topk') or []
                    doc['cnn_fallback_reason']='low_confidence_or_other'
                elif cnn_error:
                    doc['cnn_error']=cnn_error
                write_json(out/'classification.json',doc)
            return {'classification':str(out)}
        if tool=='segment_instance':
            src=Path(target['path']);meta=contracts.CropMeta.from_json(str(src))
            contracts.check_crops([meta]);inp=work/'crops';inp.mkdir()
            shutil.copy2(src,inp/src.name);shutil.copy2(meta.png,inp/Path(meta.png).name)
            out=work/'segmentation'
            command([Path(paths.SEGMENTATION_CLI_DIR)/'igm_segment_batch.py','--crops',inp,'--out-dir',out,
                     '--box-association', session.config.get('box_association','region_component'),
                     *(['--size', str(session.config['igm_size'])] if session.config.get('igm_size') else [])])
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
