"""原图/部分工件→工具候选→B局部执行→不可变证据快照；模型不提供尺寸。"""
import copy
import time
from pathlib import Path
import contracts
import paths
from .adapters import interpreter
from .evidence_store import EvidenceStore
from .ledger_lock import acquire, release
from .resource_budget import reserve
from .storage import IntegrationError, digest, file_hash, project_path, read_json, write_json, finite_number
from .workflow_backend import WorkflowBackend
from .service_errors import stage_error_kind


TOOLS=('detect_fov_initial','calibrate_fov','segment_instance','classify_target','measure_instances','detect_region',
       'propose_regions','roi_screen','roi_screen_batch','locate_feature','needle_scan','feature_scan','hybrid_scan','image_exemplar_audit')


class WorkflowSession:
    def __init__(self, runtime, root, config, *, allow_network=False, backend=None):
        self.runtime=runtime;self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True)
        import threading
        self.cancel_event=threading.Event()
        self.config=copy.deepcopy(config);self.sample_id=runtime.sample_id
        self.python=interpreter(config.get('python'));self.allow_network=allow_network
        self.yolo_signature = {}
        if config.get('detector', 'yolo') != 'yolo':
            raise IntegrationError('CONFIG_INVALID', '出框后端仅支持 YOLO')
        if not runtime.simulation:
            assets = Path(paths.PROJECT_ROOT) / 'TCM_Fusion' / 'yolo-results' / 'detection' / 'yolo_runs'
            yolo = {
                'weights': str(assets / 'v1_yolov10s_1024' / 'weights' / 'best.pt'),
                'class_map': str(assets / 'data_local.yaml'),
                'detector_config': str(Path(paths.PROJECT_ROOT) / 'TCM_Fusion' / 'config' / 'detector.json'),
                'python': str(Path(paths.VENV_PY).parent.parent / 'yolo' / 'Scripts' / 'python.exe'),
                **config.get('yolo', {}),
            }
            self.config['yolo'] = yolo
            for key in ('weights', 'class_map', 'detector_config'):
                path = project_path(yolo[key])
                if not path.is_file():
                    raise IntegrationError('YOLO_ASSET_MISSING', f'YOLO 文件缺失：{path}')
                self.yolo_signature[key] = file_hash(path)
            self.yolo_python = interpreter(yolo['python'])
            self.yolo_signature['python'] = file_hash(self.yolo_python)
            self.yolo_signature['taxonomy_config'] = runtime.config_identity
        self.code_signature=WorkflowBackend.code_signature()
        self.backend=backend or WorkflowBackend()
        if backend is not None and not runtime.simulation:
            raise IntegrationError('SIMULATION_SCOPE','真实运行禁止替身后端')
        self.identity=digest({'sample':self.sample_id,'images':sorted(runtime.images),
                              'config':config,'simulation':runtime.simulation,'vlm':runtime.config_identity,
                              'backend':self.code_signature,'python':file_hash(self.python),
                              **({'yolo': self.yolo_signature} if self.yolo_signature else {})})
        self.store=EvidenceStore(self.root/'evidence',self.identity)
        self.budget_path=self.root/'resources.json';self.operations=self.root/'operations.json'
        init_limit = config.get('max_initializations')
        if init_limit is None:
            init_limit = len(runtime.images) + int(config.get('initialization_reserve', 1))
        self.limits={'initialization':init_limit,
                     'recompute':config.get('max_recompute',8),'network':config.get('max_api_calls',16),
                     'planning':config.get('max_planning',6),'observation':config.get('max_observations',6),
                     'total':config.get('max_resource_units',48)}
        self.limits['finalization'] = int(bool(
            config.get('final_assessment_llm', True) and allow_network and not runtime.simulation
            and self.limits['network'] > 0 and self.limits['total'] >= 2))
        h=acquire(self.budget_path)
        try:
            if not self.budget_path.exists():write_json(self.budget_path,{'identity':self.identity,
                'deadline':time.time()+config.get('max_seconds',600),'used':{},'entries':{}})
            if read_json(self.budget_path)['identity']!=self.identity:raise IntegrationError('BUDGET_SCOPE','资源上下文不匹配')
        finally:release(h)
        h=acquire(self.operations)
        try:
            if not self.operations.exists():write_json(self.operations,{'entries':{}})
        finally:release(h)
        if not self.store.current()['artifacts']:
            images=[{'id':key,'kind':'image','path':str(v['path']),'sha256':v['sha256'],
                     'parents':[],'image_id':key} for key,v in runtime.images.items()]
            self.store.commit(self.store.current()['version'],images)

    def remaining_seconds(self):
        return max(0.1,read_json(self.budget_path)['deadline']-time.time())

    def close(self,reason):
        h=acquire(self.budget_path)
        try:
            value=read_json(self.budget_path);value.update(closed=True,stop_reason=reason);write_json(self.budget_path,value)
        finally:release(h)

    def diagnostic_quality(self):
        """仅联调允许review工件进入真实测量；不改原始QA/flags或最终规则。"""
        return self.config.get('quality_policy','strict')=='diagnostic_review'

    def purpose_quality(self):
        return self.config.get('quality_evaluation','legacy')=='purpose_v1'

    def locate_override(self, target):
        """GLM locate 高分框即使分类未通过，也允许进入诊断性 IGM/测量。"""
        score=target.get('det_score')
        try:
            score=float(score)
        except (TypeError, ValueError):
            return False
        return score >= float(self.config.get('glm_locate_threshold',0.8))

    def identity_pending(self):
        """最高 det_score 的 locate crop，尚未尝试 classify，且不适用高分绕过。"""
        candidates=[]
        done=read_json(self.operations).get('entries', {})
        interest=self.runtime.candidate_interest()
        active_features={c['feature'] for row in self.runtime.kb.candidates()
                         if row['id'] in interest['ids'] for c in row['criteria']}
        paused_features={c['feature'] for row in self.runtime.kb.candidates()
                         if row['id'] in set(interest.get('inactive_ids', []))
                         for c in row['criteria']}-active_features
        for tid,target in self.targets().items():
            if target.get('kind')!='crop' or not target.get('identity_required'):
                continue
            if self.locate_override(target):
                continue
            meta=contracts.CropMeta.from_json(target['path'])
            if meta.feature_id in paused_features:
                continue
            if self.store.current()['artifacts'].get('classification:'+tid,{}).get('valid'):
                continue
            key=self.operation_key('classify_target',target)
            row=done.get(key,{})
            if row.get('status') in {'done','failed','pending'}:
                continue
            try: score=float(target.get('det_score') or 0)
            except (TypeError, ValueError): score=0.0
            candidates.append((score,target))
        return max(candidates,key=lambda item:item[0])[1] if candidates else None

    def evaluate_segmentation(self,root,target,*,identity_verified=False):
        """在新目录重放原始生成图，保留旧工件；不从Agent参数接收资格。"""
        from quality_eval import replay
        meta=contracts.CropMeta.from_json(target['path']);root=Path(root)
        generation=root/'igm'/meta.crop_id
        if not (generation/'gen_raw.png').is_file():return None
        classification=None
        if not identity_verified:
            for key,artifact in self.store.current()['artifacts'].items():
                if not artifact['valid'] or artifact['kind']!='detection' or artifact.get('image_id')!=target['image_id']:continue
                doc=read_json(artifact['path'])
                for row in doc.get('crops',[]):
                    second=row.get('stage2') or {}
                    if row.get('crop_id')==meta.crop_id and row.get('feature_id')==meta.feature_id and (
                        (second.get('reject') is False and second.get('confidence')=='high') or (not second and row.get('confidence')=='high')):
                        identity_verified=True;classification=(key,artifact);break
        out=self.root/'quality'/digest([str(root),file_hash(meta.png),identity_verified,self.code_signature])[:16]
        if (out/'assessment.json').exists():return str(out/'assessment.json')
        value=replay(generation,meta,out,identity_verified=identity_verified)
        if classification:
            key,artifact=classification
            value['bindings']['identity_observation']=artifact['sha256']
            value['identity_artifact_id']=key;value['identity_basis']='trusted_detector_high_confidence_trial_not_calibrated'
            write_json(out/'assessment.json',value)
        return str(out/'assessment.json')

    def seed(self, root):
        """导入绑定原图的部分crop/mask；复制子集，不修改原B目录。"""
        import csv
        import shutil
        root=Path(root).resolve();marker=self.root/'imported.json'
        if marker.exists():return
        crops={};items=[];scale_values={};known={str(v['path'].resolve()):k for k,v in self.runtime.images.items()}
        for p in sorted(root.rglob('crops/*.json')):
            if p.is_symlink() or not p.resolve().is_relative_to(root):continue
            try:
                m=contracts.CropMeta.from_json(str(p))
                image=known.get(str(Path(m.source_image_path).resolve()))
                if not image or not Path(m.png).resolve().is_relative_to(root) or not Path(m.png).is_file():continue
                cid='crop:'+image+':'+m.crop_id
                previous=crops.get((image,m.crop_id))
                if previous:
                    old=previous[1]
                    if old.raw!=m.raw or file_hash(old.png)!=file_hash(m.png):
                        raise IntegrationError('CROP_IMPORT_CONFLICT','同一crop的重复侧车内容冲突，拒绝猜测来源')
                    continue  # 原B交付包含crops及masks/crops同源副本，只导入一次。
                pixel=self._art(cid+':pixels','crop_pixels',m.png,[image],image_id=image)
                items.extend([pixel,self._art(cid,'crop',p,[image,pixel['id']],image_id=image)])
                crops[(image,m.crop_id)]=(cid,m)
                if finite_number(m.px_per_um,minimum=1e-12):scale_values.setdefault(image,set()).add(m.px_per_um)
            except (OSError,ValueError,contracts.ContractError):continue
        for image in {key[0] for key in crops}:
            done=self.root/'imports'/('detected_'+image+'.json');write_json(done,{'source':'bound_original_crop'})
            items.append(self._art('detected:'+image,'detection',done,[image],image_id=image))
            values=scale_values.get(image,set())
            if len(values)==1:
                sf=self.root/'imports'/('scale_'+image+'.json');value=next(iter(values))
                write_json(sf,{'px_per_um':value,'source':'original_crop_contract'})
                items.append(self._art('scale:'+image,'scale',sf,[image],image_id=image,px_per_um=value))
        if items:self.store.commit(self.store.current()['version'],items)
        # 单crop隔离导入，避免一次measure工具暗中处理整个原B批次。
        for f in sorted(root.rglob('*__measure_manifest.csv')):
            if f.is_symlink() or not f.resolve().is_relative_to(root):continue
            try:
                with f.open(encoding='utf-8-sig',newline='') as stream:raws=list(csv.DictReader(stream))
                groups={}
                for raw in raws:
                    row=contracts.MeasureManifestRow.from_row(raw)
                    image=known.get(str(Path(row.source_image_path).resolve()))
                    pair=crops.get((image,row.crop_id))
                    scale=self.store.current()['artifacts'].get('scale:'+str(image),{})
                    allowed={'ok','review'} if self.diagnostic_quality() or self.purpose_quality() else {'ok'}
                    if not pair or not scale.get('valid') or row.px_per_um!=scale.get('px_per_um') or row.igm_status not in allowed:continue
                    src=(f.parent/row.inst_mask).resolve()
                    if src.is_file() and src.is_relative_to(root):groups.setdefault(pair[0],[]).append((raw,src,image,pair[1]))
                for cid,rows in groups.items():
                    out=self.root/'imports'/digest(cid);masks=out/'masks';masks.mkdir(parents=True,exist_ok=True)
                    copied=[]
                    for raw,src,image,m in rows:
                        shutil.copy2(src,masks/src.name);row={**raw,'inst_mask':src.name,'herb':''};copied.append(row)
                    with (masks/(m.stem+contracts.MANIFEST_SUFFIX)).open('w',encoding='utf-8-sig',newline='') as stream:
                        w=csv.DictWriter(stream,fieldnames=contracts.MANIFEST_COLUMNS);w.writeheader();w.writerows(copied)
                    cp=masks/'crops';cp.mkdir(exist_ok=True)
                    shutil.copy2(m.json_path,cp/Path(m.json_path).name);shutil.copy2(m.png,cp/Path(m.png).name)
                    quality='valid' if all(r.get('igm_status')=='ok' for r in copied) else 'review'
                    write_json(out/'instances.json',{'instances':[{'igm_status':r.get('igm_status')} for r in copied]})
                    result={'segmentation':str(out),'quality':quality}
                    if self.purpose_quality():
                        # 评估原生成产物，不用已裁剪mask的框内占比自证质量。
                        assessment=self.evaluate_segmentation(f.parent.parent,self.store.current()['artifacts'][cid])
                        if assessment:result['assessment']=assessment
                    self.accept('segment_instance',self.store.current()['artifacts'][cid],result,self.store.current()['version'])
            except (OSError,ValueError,KeyError,contracts.ContractError):continue
        write_json(marker,{'root_sha256':digest(str(root)),'scope':'bound_image_contracts_only'})

    def charge(self,pool,key,*,network=True):
        """预约资源；`network=False` 用于 worker 内已有受控HTTP守卫、网络由其请求键自记的场景。"""
        extra_pool = 'network' if network and pool in {'planning', 'observation', 'finalization'} else None
        if not reserve(self.budget_path,pool,key,self.limits,identity=self.identity,extra_pool=extra_pool):
            raise IntegrationError('RESOURCE_REPLAY','同资源操作已预约，禁止自动重发')
        return True

    def _art(self, identifier, kind, path, parents, **metadata):
        p=Path(path).resolve()
        return {'id':identifier,'kind':kind,'path':str(p),'sha256':file_hash(p),
                'parents':list(parents),**metadata}

    def blocked_tools(self):
        """任务内按工具熔断；locate/classify 允许有界重试后再熔断。"""
        limit = int(self.config.get('max_tool_failures', 2))
        retryable = {'locate_feature', 'classify_target', 'roi_screen', 'roi_screen_batch', 'needle_scan', 'feature_scan', 'detect_fov_initial'}
        failures = {}
        for row in read_json(self.operations)['entries'].values():
            if row.get('status') == 'failed' and stage_error_kind(row.get('error_code')) == 'service':
                failures.setdefault(row.get('tool'), []).append(row.get('error_code'))
        blocked = {}
        for tool, codes in failures.items():
            if tool in retryable and len(codes) < limit:
                continue
            blocked[tool] = codes[-1]
        return blocked

    def screen_rois(self, image):
        """返回 screen 结果中的 top ROI 索引；没有 screen 时不限制。"""
        for value in self.store.current()['artifacts'].values():
            if value.get('valid') and value.get('kind') == 'screen' and value.get('image_id') == image:
                try:
                    doc = read_json(value['path'])
                    rows = sorted(doc.get('rois') or [], key=lambda x: float(x.get('score') or 0), reverse=True)
                    top_k = max(1, int(self.config.get('screen_top_k', 2)))
                    return {int(row['index']) for row in rows[:top_k] if row.get('worth', True)}
                except (OSError, ValueError, KeyError, TypeError):
                    return None
        return None

    def targets(self):
        snap=self.store.current();arts=snap['artifacts'];targets={}
        for identifier,v in arts.items():
            if not v['valid']:continue
            if v['kind']=='image':
                targets[identifier]={**v,'source':v['path'],'image_id':identifier}
                if (self.config.get('needle_scan_enabled', False)
                        and arts.get('detected:'+identifier, {}).get('valid')):
                    targets['scan:'+identifier]={'id':'scan:'+identifier,'kind':'needle_scan',
                        'image_id':identifier,'source':v['path'],'parents':[identifier],
                        'px_per_um':arts.get('scale:'+identifier,{}).get('px_per_um')}
                if self.config.get('feature_scan_enabled', False):
                    requested=[str(x) for x in (self.config.get('feature_scan_features') or []) if str(x)]
                    if self.config.get('feature_scan_from_candidates', False):
                        try:
                            requested += self.recovery_features()
                        except Exception:
                            pass
                    features=list(dict.fromkeys(requested))[:max(1,int(self.config.get('feature_scan_max_features',2)))]
                    for feature_id in features:
                        key='feature_scan:'+identifier+':'+feature_id
                        targets[key]={'id':key,'kind':'feature_scan','feature_id':feature_id,
                            'image_id':identifier,'source':v['path'],'parents':[identifier],
                            'px_per_um':arts.get('scale:'+identifier,{}).get('px_per_um')}
                if self.config.get('hybrid_scan_enabled', False):
                    requested=[str(x) for x in (self.config.get('feature_scan_features') or []) if str(x)]
                    try:
                        requested += self.recovery_features()
                    except Exception:
                        pass
                    features=list(dict.fromkeys(requested))[:max(1,int(self.config.get('hybrid_scan_max_features',2)))]
                    if features:
                        targets['hybrid_scan:'+identifier]={'id':'hybrid_scan:'+identifier,
                            'kind':'hybrid_scan','features':features,'image_id':identifier,
                            'source':v['path'],'parents':[identifier],
                            'px_per_um':arts.get('scale:'+identifier,{}).get('px_per_um')}
            elif v['kind'] in {'crop','segmentation'}:targets[identifier]=v
        failed={r.get('target_id') for r in read_json(self.operations)['entries'].values()
                if r.get('tool')=='segment_instance' and r.get('status')=='failed'
                and stage_error_kind(r.get('error_code'))=='evidence'}
        failed |= {v.get('crop_id') for v in arts.values() if v['valid'] and v['kind']=='segmentation' and v.get('quality')=='review'}
        if self.purpose_quality():
            spatial=set()
            for seg in arts.values():
                if seg['valid'] and seg['kind']=='segmentation':
                    qa=self.assessment(seg)
                    if qa and any(f['code'] in {'TARGET_ASSOCIATION_AMBIGUOUS','POSSIBLE_MERGE','POSSIBLE_TRUNCATION'} for f in qa['findings']):spatial.add(seg.get('crop_id'))
            failed &= spatial  # 类别未知/中心点未命中本身不触发重新裁框。
        if 'segment_instance' in self.blocked_tools():failed=set()
        if self.diagnostic_quality():failed=set()  # 联调复用review，不为过门重复生图/重检。
        for cid in failed:
            crop=arts.get(cid,{})
            if not crop.get('valid'):continue
            m=contracts.CropMeta.from_json(crop['path'])
            if not m.loose_box:continue
            image=crop['image_id'];src=self.runtime.images[image]['path'];scale=arts.get('scale:'+image,{})
            targets['roi:auto:'+cid]={'id':'roi:auto:'+cid,'image_id':image,'source':str(src),
                'roi':[int(v) for v in m.loose_box],'parents':[cid,image],
                'px_per_um':scale.get('px_per_um'),'replace_id':cid}
        for i,region in enumerate(self.config.get('regions',[])):
            matches=[(k,v) for k,v in self.runtime.images.items() if region['image_stem'] in v['stems']]
            if len(matches)!=1:raise IntegrationError('ROI_SCOPE','ROI不能唯一绑定视野')
            key,v=matches[0];scale=arts.get('scale:'+key,{})
            targets['roi:'+str(i)]={'id':'roi:'+str(i),'image_id':key,'source':str(v['path']),
                'roi':region['box'],'parents':[key],'px_per_um':scale.get('px_per_um')}
        if self.config.get('targeted_recovery', False):
            gaps = self.recovery_features()
            screen_batch_items = []
            for item in arts.values():
                if not item['valid'] or item['kind'] != 'region_proposals':
                    continue
                image = item['image_id']
                if file_hash(item['path']) != item['sha256']:
                    raise IntegrationError('ARTIFACT_CHANGED', '区域提议已变化')
                screen_id = 'screen:' + image
                if screen_id not in arts and gaps:
                    targets[screen_id] = {'id': screen_id, 'kind': 'screen', 'image_id': image,
                                          'source': self.runtime.images[image]['path'],
                                          'regions': item['path'], 'parents': [image, item['id']],
                                          'feature_id': gaps[0]}
                    screen_batch_items.append({'image_id': image, 'source': self.runtime.images[image]['path'],
                                               'regions': item['path'], 'feature_id': gaps[0],
                                               'parents': [image, item['id']]})
                top_rois = self.screen_rois(image)
                for index, region in enumerate(read_json(item['path'])['regions']):
                    if top_rois is not None and index not in top_rois:
                        continue
                    identifier = f"roi:recovery:{image}:{index}"
                    region_target = {'id': identifier, 'image_id': image,
                                     'source': self.runtime.images[image]['path'],
                                     'roi': region['box'], 'line_score': region['line_score'],
                                     'parents': [image, item['id']],
                                     'px_per_um': arts.get('scale:'+image, {}).get('px_per_um')}
                    targets[identifier] = region_target
                    for feature in gaps:
                        key = f'locate:{identifier}:{feature}'
                        targets[key] = {**region_target, 'id': key, 'feature_id': feature}
            if len(screen_batch_items) > 1 and int(self.config.get('screen_concurrency', 1)) > 1:
                targets['screen:batch'] = {'id': 'screen:batch', 'kind': 'screen_batch',
                                           'image_id': screen_batch_items[0]['image_id'],
                                           'items': screen_batch_items,
                                           'parents': [x['image_id'] for x in screen_batch_items]}
        return targets

    def recovery_features(self):
        from detect_core import FEATURE_ID
        from prompts import CLASSIFY_HINTS
        from .candidate_policy import candidates, criteria_diff
        r = self.runtime
        interest = r.candidate_interest()
        supported = {fid for label, fid in FEATURE_ID.items() if label in CLASSIFY_HINTS}
        measured = {record.get('feature_id') or record.get('feature')
                    for record in self.store.verified_records()
                    if finite_number(record.get('primary_value_um'), minimum=1e-12)}
        return list(dict.fromkeys(c['feature'] for row in criteria_diff(r.kb, interest['ids'], {})
                                  for c in row['criteria'] if c['role'] == 'required'
                                  and c['feature'] in supported and c['feature'] not in measured))

    def dependencies(self, tool, target):
        arts=self.store.current()['artifacts'];ids=[target['id']] if target['id'] in arts else target['parents']
        if tool in {'segment_instance','detect_region','locate_feature'}:
            scale='scale:'+target['image_id']
            if scale in arts and arts[scale]['valid']:ids+=[scale]
        if tool == 'segment_instance' and target.get('identity_required') and not self.locate_override(target):
            ids.append('classification:'+target['id'])
        return sorted(set(ids))

    def _dependency_rows(self,tool,target,*,semantic=False):
        arts=self.store.current()['artifacts'];deps=[];keys=set()
        def visit(key):
            if key in keys:return
            value=arts.get(key)
            if not value or not value['valid']:raise IntegrationError('EVIDENCE_DEPENDENCY','动作依赖缺失或失效')
            keys.add(key)
            for parent in value.get('parents',[]):visit(parent)
        for key in self.dependencies(tool,target):visit(key)
        for key in sorted(keys):
            value=arts[key];content=value['sha256']
            if semantic and value['kind']=='quality_assessment':
                canonical=read_json(value['path']);canonical.pop('source_generation_dir',None)
                content=digest(canonical)
            if semantic and (value['kind']=='crop' or (value['kind']=='segmentation_file'
                            and Path(value['path']).parent.name=='crops' and value['path'].endswith('.json'))):
                m=contracts.CropMeta.from_json(value['path'])
                content=digest({'pixels':file_hash(m.png),'feature':m.feature_id,'ppu':m.px_per_um,
                    'tight':m.tight_box,'local':m.tight_box_local,'loose':m.loose_box,'granularity':m.granularity})
            deps.append((key,content))
        return deps

    def dependency_fingerprint(self,tool,target):
        return digest({'tool':tool,'target':target['id'],'roi':target.get('roi'),
                       'deps':self._dependency_rows(tool,target,semantic=True),
                       'model_config':self.runtime.config_identity,'backend':self.code_signature})

    def operation_key(self,tool,target):
        return digest({'tool':tool,'identity':self.identity,'target':target['id'],
            'roi':target.get('roi'),'deps':self._dependency_rows(tool,target)})

    def review_measure_allowed(self):
        """外部 review 决策选择 split/diagnostic 时，允许一次诊断测量绕过 review 门。"""
        return self.review_measure_plan() is not None

    def review_measure_plan(self, target=None):
        getter=getattr(self.runtime,'p0_focus',None)
        if getter is None:
            return None
        try:
            focus=getter()
        except Exception:
            return None
        if not focus or focus.get('status')!='crop_segmentation_review':
            return None
        snapshot=self.store.current()
        segmentation_id=focus.get('segmentation_id')
        segmentation=snapshot['artifacts'].get(segmentation_id,{})
        if not segmentation.get('valid'):
            return None
        if target is not None and (target.get('id') != segmentation_id or
                                   target.get('sha256') != segmentation.get('sha256')):
            return None
        decision=focus.get('review_decision') or {}
        repair=decision.get('repair')
        if repair not in {'split_component','diagnostic_measure'}:
            return None
        review=focus.get('review') or {}
        valid={int(c.get('id')) for c in (review.get('components') or []) if c.get('id') is not None}
        ids=[]
        for value in decision.get('component_ids') or []:
            try:
                cid=int(value)
            except (TypeError, ValueError):
                continue
            if cid in valid and cid not in ids:
                ids.append(cid)
        policy=str(decision.get('component_policy') or '')
        if policy not in {'largest','selected','all','merge'}:
            policy='selected' if ids else 'largest'
        if ids:
            policy='selected'
        return {'repair':repair,'component_ids':ids,'component_policy':policy,
                'target_id':segmentation_id,'target_sha256':segmentation['sha256'],
                'evidence_version':snapshot['version']}

    def propose(self, *, include_offline=False):
        snap=self.store.current();arts=snap['artifacts'];done=read_json(self.operations)['entries'];out=[]
        blocked=self.blocked_tools()
        recovery = self.config.get('targeted_recovery', False)
        gaps = self.recovery_features() if recovery else []
        review_plan = self.review_measure_plan()
        interest = self.runtime.candidate_interest()
        inactive = set(interest.get('inactive_ids', []))
        # A withdrawn hypothesis only closes its exclusive recovery branch. Shared
        # features and already committed evidence remain available to other hypotheses.
        active_features={c['feature'] for row in self.runtime.kb.candidates()
                         if row['id'] in interest['ids'] for c in row['criteria']}
        paused_features={c['feature'] for row in self.runtime.kb.candidates()
                         if row['id'] in inactive for c in row['criteria']}-active_features
        locate_attempts = sum(row['tool'] == 'locate_feature' for row in done.values())
        import re
        locate_pairs=set()
        for row in done.values():
            if row.get('tool')=='locate_feature' and row.get('status') in {'done','failed'}:
                hit=re.search(r'locate:roi:recovery:(img_[0-9a-f]{16,}):(\d+):',str(row.get('target_id') or ''))
                if hit: locate_pairs.add((hit.group(1),int(hit.group(2))))
        max_locate = self.config.get('max_locate_attempts')
        if max_locate is None:
            max_locate = max(4, int(self.config.get('locate_attempts_per_view', 2)) * len(self.runtime.images)
                             + int(self.config.get('locate_attempt_reserve', 2)))
        max_locate = int(max_locate)
        def offer(tool,t,priority):
            if tool in blocked:return
            if tool not in self.config.get('enabled_tools',TOOLS) and not include_offline:return
            if (tool not in {'measure_instances','propose_regions','detect_region'} and not self.allow_network and not self.runtime.simulation
                and not include_offline and self.config.get('execution')!='shadow'):return
            key=self.operation_key(tool,t)
            if key not in done:
                out.append(self.runtime.action(tool,{'target_id':t['id'],'evidence_version':snap['version']},
                    'workflow_repair',priority=priority))
        crops=[]
        target_map=self.targets()
        has_batch_screen='screen:batch' in target_map
        for tid,t in target_map.items():
            tool=None;image=t['image_id']
            origin = t if t.get('kind')=='crop' else arts.get(t.get('crop_id'), {})
            if origin.get('identity_required') and paused_features:
                meta=contracts.CropMeta.from_json(origin['path'])
                if meta.feature_id in paused_features:
                    continue
            if tid.startswith('locate:'):
                hit=re.search(r'locate:roi:recovery:(img_[0-9a-f]{16,}):(\d+):',tid)
                pair=(hit.group(1),int(hit.group(2))) if hit else None
                if locate_attempts < max_locate and pair not in locate_pairs:
                    tool = 'locate_feature'
            elif tid.startswith('roi:'):
                if 'detected:'+image in arts:tool='detect_region'
            elif t.get('kind') == 'screen_batch':
                tool = 'roi_screen_batch'
            elif t.get('kind') == 'needle_scan':
                tool = 'needle_scan'
            elif t.get('kind') == 'feature_scan':
                tool = 'feature_scan'
            elif t.get('kind') == 'hybrid_scan':
                tool = 'hybrid_scan'
            elif t.get('kind') == 'screen':
                if not has_batch_screen and not arts.get('screen:'+image, {}).get('valid'):
                    tool = 'roi_screen'
            elif t['kind']=='image':
                if 'detected:'+image not in arts:tool='detect_fov_initial'
                else:
                    if gaps:
                        offer('propose_regions', t, (1, 'propose_regions', tid))
                    if not arts.get('scale:'+image,{}).get('valid'):tool='calibrate_fov'
            elif t['kind']=='crop':
                crops.append(t)
                if (t.get('identity_required')
                        and not arts.get('classification:'+tid,{}).get('identity_verified')
                        and not self.locate_override(t)):
                    continue
                if self.config.get('image_exemplar_audit', False):
                    meta=contracts.CropMeta.from_json(t['path'])
                    if meta.feature_id:
                        offer('image_exemplar_audit', t, (2, 'image_exemplar_audit', tid))
                scale=arts.get('scale:'+image,{})
                seg=arts.get('seg:'+tid,{})
                review_measure=bool(review_plan and review_plan['target_id']=='seg:'+tid)
                usable=seg.get('quality')=='valid' or (self.diagnostic_quality() and seg.get('quality')=='review') or review_measure
                if self.locate_override(t) and seg.get('quality')=='review':
                    usable=True
                if self.purpose_quality() and seg.get('assessment_id'):usable=True
                if scale.get('valid') and (not seg.get('valid') or not usable):
                    tool='segment_instance'
            elif t['kind']=='segmentation' and (t.get('quality')=='valid' or
                    (self.diagnostic_quality() and t.get('quality')=='review') or
                    (review_plan and review_plan['target_id']==tid)):
                if not arts.get('measurement:'+tid,{}).get('valid'):tool='measure_instances'
            if t.get('kind')=='segmentation' and self.purpose_quality():
                assessment=self.assessment(t)
                if not assessment or not any(assessment['uses'][u]['allowed'] for u in ('length','width','area')):
                    tool=None
                elif not arts.get('measurement:'+tid,{}).get('valid'):tool='measure_instances'
            if tool:offer(tool,t,(0,tool,tid))
        # 复核类别是“修复”而非主路径：排在物理阶段之后，只在观察到的存在特征与crop类别相左时出现。
        pending = [t for t in crops if not arts.get('classification:'+t['id'], {}).get('valid')
                   and (t.get('identity_required') or self.label_conflict(t))
                   and not self.locate_override(t)]
        best = max(pending, key=lambda c: float(c.get('det_score') or 0), default=None)
        for t in crops:
            if t.get('identity_required') and self.locate_override(t):
                continue
            if arts.get('classification:'+t['id'],{}).get('valid'):continue
            if not t.get('identity_required') and not self.label_conflict(t):continue
            if t.get('identity_required') and best is not None and t['id'] != best['id']:
                continue
            offer('classify_target',t,(1,'classify_target',t['id']))
        return out

    def label_conflict(self,target):
        """判断该crop的类别是否与已观察到的存在特征相左。

        两个保守条件：① 没有任何存在性观察时不触发，否则每次检测都会白花一次分类调用；
        ② 已有可用掩码（quality=valid）时不触发，重判类别会让现有尺寸证据失效。
        """
        arts=self.store.current()['artifacts']
        if arts.get('seg:'+target['id'],{}).get('quality')=='valid':return False
        present=getattr(self.runtime,'present_features',None) or set()
        if not present:return False
        try:meta=contracts.CropMeta.from_json(target['path'])
        except (OSError,ValueError,contracts.ContractError):return False
        return bool(meta.feature_id) and meta.feature_id not in present

    def precondition(self,args):
        return args.get('target_id') in self.targets()

    def assessment(self,target):
        ref=target.get('assessment_id');item=self.store.current()['artifacts'].get(ref,{})
        if not item.get('valid'):return None
        if file_hash(item['path'])!=item['sha256']:raise IntegrationError('ARTIFACT_CHANGED','资格评估被改写')
        value=contracts.quality_assessment(read_json(item['path']))
        generation=Path(value['source_generation_dir'])
        for name,binding in [('gen_raw.png','generation'),('meta.json','generation_metadata')]:
            source=generation/name
            if not source.is_file() or file_hash(source)!=value['bindings'].get(binding):
                raise IntegrationError('QUALITY_SCOPE','生成图或变换元数据与评估不符')
        proof=value.get('identity_artifact_id')
        if proof:
            witness=self.store.current()['artifacts'].get(proof,{})
            if not witness.get('valid') or file_hash(witness['path'])!=value['bindings'].get('identity_observation'):
                raise IntegrationError('QUALITY_SCOPE','目标身份观察来源失效')
        mask=Path(item['path']).parent/'associated_mask.png'
        if not mask.is_file() or file_hash(mask)!=value['bindings'].get('mask'):
            raise IntegrationError('QUALITY_SCOPE','关联mask与用途评估不符')
        return value

    def coverage(self):
        """逐视野覆盖审计：detected / proposed / screened，以及 screen top-k 的 locate。"""
        import re
        images = list(self.runtime.images)
        detected, proposed, screened, located = set(), set(), set(), set()
        located_pairs = set()
        image_pattern = re.compile(r'img_[0-9a-f]{16,}')
        pair_pattern = re.compile(r'locate:roi:recovery:(img_[0-9a-f]{16,}):(\d+):')
        for row in read_json(self.operations).get('entries', {}).values():
            if row.get('status') not in {'done', 'failed'}:
                continue
            tool, target = row.get('tool'), str(row.get('target_id') or '')
            if tool == 'detect_fov_initial':
                if row.get('status') == 'done' and target in images:
                    detected.add(target)
            elif tool == 'propose_regions' and target in images:
                proposed.add(target)
            elif tool == 'roi_screen':
                hit = image_pattern.search(target)
                if hit:
                    screened.add(hit.group(0))
            elif tool == 'locate_feature':
                hit = image_pattern.search(target)
                if hit:
                    located.add(hit.group(0))
                pair = pair_pattern.search(target)
                if pair:
                    located_pairs.add((pair.group(1), int(pair.group(2))))
        for value in self.store.current()['artifacts'].values():
            if value.get('valid') and value.get('kind')=='screen' and value.get('image_id') in images:
                screened.add(value['image_id'])
        required_pairs = set()
        rows = []
        for image in images:
            top_rois = self.screen_rois(image)
            if top_rois is not None:
                required_pairs.update((image, int(index)) for index in top_rois)
            done_rois = {index for name, index in located_pairs if name == image}
            rows.append({'image_id': image, 'detected': image in detected,
                         'proposed': image in proposed, 'screened': image in screened,
                         'located': image in located,
                         'top_rois': sorted(top_rois) if top_rois is not None else None,
                         'located_rois': sorted(done_rois)})
        missing_pairs = sorted(required_pairs - located_pairs)
        missing_locate = sorted({image for image, _ in missing_pairs})
        located_all = (all(pair in located_pairs for pair in required_pairs)
                       if required_pairs else len(located) == len(images))
        return {
            'views': rows,
            'detected_all': len(detected) == len(images),
            'proposed_all': len(proposed) == len(images),
            'screened_all': len(screened) == len(images),
            'located_all': located_all,
            'required_locate_pairs': [{'image_id': image, 'index': index}
                                      for image, index in sorted(required_pairs)],
            'missing_locate_pairs': [{'image_id': image, 'index': index}
                                     for image, index in missing_pairs],
            'missing_propose': [x['image_id'] for x in rows if not x['proposed']],
            'missing_screen': [x['image_id'] for x in rows if not x['screened']],
            'missing_locate': missing_locate or [x['image_id'] for x in rows if not x['located']],
        }

    def segmentation_reviews(self):
        """把 IGM review 的原始原因整理成 planner 可读的结构化字段。

        这些字段只读，不改变证据版本；planner 据此决定重新定位、拆分量、诊断测量或放弃。
        """
        rows = []
        for v in self.store.current()['artifacts'].values():
            if not v.get('valid') or v.get('kind') != 'segmentation':
                continue
            try:
                root = Path(v['path'])
                doc = read_json(root)
            except (OSError, ValueError):
                continue
            for inst in doc.get('instances') or []:
                meta = {}
                igm_dir = root.parent / str(inst.get('igm_dir') or '')
                if igm_dir.is_dir():
                    try:
                        meta = read_json(igm_dir / 'meta.json')
                    except (OSError, ValueError):
                        meta = {}
                components = []
                mask_path = igm_dir / 'mask.png'
                if mask_path.is_file():
                    try:
                        import cv2
                        import numpy as np
                        arr = cv2.imdecode(np.fromfile(str(mask_path), dtype=np.uint8),
                                           cv2.IMREAD_GRAYSCALE)
                        if arr is not None:
                            n, labels, stats, centroids = cv2.connectedComponentsWithStats(
                                (arr > 0).astype(np.uint8), 8)
                            for cid in range(1, n):
                                x, y, w, h, area = [int(v) for v in stats[cid]]
                                comp = labels == cid
                                components.append({
                                    'id': cid, 'area_px': area,
                                    'area_ratio': round(area / float(arr.size), 6),
                                    'bbox': [x, y, w, h],
                                    'centroid': [round(float(centroids[cid][0]), 1),
                                                 round(float(centroids[cid][1]), 1)],
                                    'touches_border': bool(comp[0].any() or comp[-1].any()
                                                           or comp[:, 0].any() or comp[:, -1].any()),
                                })
                    except Exception:
                        components = []
                rows.append({
                    'crop_id': inst.get('crop_id'),
                    'feature_id': inst.get('feature_id'),
                    'quality': v.get('quality'),
                    'igm_status': inst.get('igm_status'),
                    'n_components': inst.get('n_components'),
                    'area_ratio': inst.get('igm_area_ratio'),
                    'fg_outside_box_ratio': inst.get('igm_fg_outside_box_ratio'),
                    'center_comp_ratio': inst.get('igm_center_comp_ratio'),
                    'touches_border': meta.get('touches_border', inst.get('igm_touches_border')),
                    'border_contact_ratio': meta.get('border_contact_ratio', inst.get('igm_border_contact_ratio')),
                    'review_reason_codes': list(meta.get('review_reason_codes') or []),
                    'components': components,
                    'mask': str(mask_path) if mask_path.is_file() else None,
                    'instances_json': str(root),
                    'overlay': str(igm_dir / 'overlay.jpg') if (igm_dir / 'overlay.jpg').is_file() else None,
                })
        return rows

    def confirm_identity(self,action,result):
        """独立合法存在性观察只能确认对应目标类别；重新评估资格，不编造尺寸。"""
        if read_json(self.budget_path).get('closed') or self.remaining_seconds()<=0.1:return
        if not self.purpose_quality() or action['tool']!='presence_check' or result.get('present') is not True or result.get('conf',0)<0.5:return
        snap=self.store.current()
        for sid,seg in list(snap['artifacts'].items()):
            if not seg['valid'] or seg['kind']!='segmentation' or seg.get('image_id')!=action['args']['image_id']:continue
            qa=self.assessment(seg)
            if not qa or qa['uses']['existence']['allowed']:continue
            crop=self.store.current()['artifacts'][seg['crop_id']];m=contracts.CropMeta.from_json(crop['path'])
            if m.feature_id!=action['args']['feature_id']:continue
            from quality_eval import replay
            out=self.root/'quality'/digest([sid,action['action_id'],result])[:16]
            value=replay(qa['source_generation_dir'],m,out,identity_verified=True)
            proof=out/'identity_observation.json';write_json(proof,{'action':action,'result':result})
            value['bindings']['identity_observation']=file_hash(proof);write_json(out/'assessment.json',value)
            aid=seg['assessment_id'];old=self.store.current()['artifacts'][aid]
            witness=self._art('identity:'+action['action_id'],'identity_observation',proof,[seg['image_id']],image_id=seg['image_id'])
            updated=self._art(aid,'quality_assessment',out/'assessment.json',old['parents']+[witness['id']],image_id=seg['image_id'])
            self.store.commit(self.store.current()['version'],[witness,updated,{k:v for k,v in seg.items() if k!='valid'}])

    def existing_boxes(self,image):
        rows=[]
        for v in self.store.current()['artifacts'].values():
            if v['valid'] and v['kind']=='crop' and v['image_id']==image:
                m=contracts.CropMeta.from_json(v['path'])
                rows.append({'id':v['id'],'label':m.object_label,'box':m.tight_box})
        return rows

    def execute(self,action):
        self.store.verified_records()  # 缓存命中也不能绕过当前工件完整性检查。
        if self.config.get('execution')=='shadow':
            raise IntegrationError('ACTION_DENIED','shadow只生成计划，禁止阶段执行')
        if WorkflowBackend.code_signature()!=self.code_signature:
            raise IntegrationError('CODE_CHANGED','工具/原B源码运行期间变化，禁止混用缓存和阶段产物')
        args=action['args'];tool=action['tool'];targets=self.targets()
        if tool not in TOOLS or args['target_id'] not in targets:raise IntegrationError('ACTION_DENIED','阶段或对象非法')
        target=targets[args['target_id']];key=self.operation_key(tool,target)
        h=acquire(self.operations)
        try:
            doc=read_json(self.operations)
            if key in doc['entries']:return {**doc['entries'][key],'cache_hit':True}
            if tool in self.blocked_tools():
                raise IntegrationError('TOOL_SERVICE_BLOCKED','该工具的模型服务在本任务已失败；不得通过换ROI或版本重发')
            if args['evidence_version']!=self.store.current()['version']:
                raise IntegrationError('EVIDENCE_STALE','阶段父版本失效')
            if action['action_id'] not in {a['action_id'] for a in self.propose()}:
                raise IntegrationError('ACTION_DENIED','阶段不在当前合法集合')
            if any(r['status']=='pending' for r in doc['entries'].values()):
                raise IntegrationError('WORKFLOW_BUSY','已有阶段正在执行')
            if tool not in {'measure_instances','propose_regions','detect_region'} and not self.runtime.simulation and not self.allow_network:
                raise IntegrationError('NETWORK_DISABLED','视觉重算须显式允许网络')
            # 类别复核是一次受控视觉观察：只占观察/网络预算，不挤占测量所需的重算额度；
            # 其网络请求由同一worker内的HTTP守卫按请求键记账，这里不重复预约。
            pool=('initialization' if tool=='detect_fov_initial' else
                  'observation' if tool in {'classify_target','image_exemplar_audit'} else 'recompute')
            self.charge(pool,key,network=tool!='classify_target')
            pending={'action_id':action['action_id'],'tool':tool,'target_id':target['id'],
                     'status':'pending','accepted':False,'repair_completed':False,'operation_key':key}
            doc['entries'][key]=pending;write_json(self.operations,doc)
        finally:release(h)
        # Windows默认路径长度有限；完整键仍在账本，目录短键冲突时拒绝覆盖。
        work=self.root/'ops'/key[:16]
        try:
            work.mkdir(parents=True,exist_ok=False)
            context=work/'budget_context.json';write_json(context,{'operation_key':key,'budget_path':str(self.budget_path),
                                                                'limits':self.limits,'identity':self.identity})
            fingerprint=self.dependency_fingerprint(tool,target)
            before_progress=self.progress_signature()
            result=self.backend.execute(self,tool,target,work,context)
            if self.remaining_seconds()<=0.1 or read_json(self.budget_path).get('closed'):
                raise IntegrationError('WORKFLOW_TIMEOUT','任务已截止，迟到工件只审计不提交')
            from .workflow_replay import bundle
            recorded=bundle(result,work)
            if self.dependency_fingerprint(tool,target)!=fingerprint:
                raise IntegrationError('EVIDENCE_STALE','执行期间相关依赖变化，结果只保留审计，不提交')
            version=self.accept(tool,target,result,self.store.current()['version'])
            record={**pending,'status':'done','accepted':True,
                    'repair_completed':self.progress_signature()!=before_progress and result.get('quality')!='review',
                    'evidence_version':version,'dependency_fingerprint':fingerprint,'bundle':recorded}
        except Exception as exc:
            code=getattr(exc,'code','WORKFLOW_STAGE_FAILED')
            record={**pending,'status':'failed','error_code':code,'error_kind':stage_error_kind(code)}
        h=acquire(self.operations)
        try:
            doc=read_json(self.operations);doc['entries'][key]=record;write_json(self.operations,doc)
        finally:release(h)
        return record

    def accept(self,tool,target,result,parent):
        image=target['image_id'];items=[];rows=[];replace=[]
        if tool=='hybrid_scan':
            detections=result.get('detections') or {}
            feature_scan=result.get('feature_scan') or {}
            for kind,bundle in (('detect',detections),('feature',feature_scan)):
                folder=Path(bundle.get('crops_dir') or '')
                if not folder.is_dir():
                    continue
                for m in contracts.load_crop_metas(str(folder)):
                    if file_hash(m.source_image_path)!=self.runtime.images[image]['sha256']:
                        raise IntegrationError('SAMPLE_MISMATCH','混合扫描产物不属于绑定原图')
                    cid='crop:'+image+':'+m.crop_id
                    image_item=self._art(cid+':pixels','crop_pixels',m.png,[image],image_id=image)
                    items.append(image_item)
                    metadata={'identity_required': True} if kind=='feature' else {}
                    if m.raw.get('det_score') is not None:
                        metadata['det_score']=m.raw.get('det_score')
                        metadata['det_confidence']=m.raw.get('det_confidence')
                    items.append(self._art(cid,'crop',m.json_path,[image,image_item['id']],
                                           image_id=image,**metadata))
                marker=('detected:'+image if kind=='detect' else 'scanned:'+image)
                items.append(self._art(marker,'detection',bundle.get('completion'),[image],
                                       image_id=image))
        elif tool == 'roi_screen_batch':
            for raw in result.get('screens') or []:
                path=Path(raw);doc=read_json(path)
                image_id=doc.get('image_id') or target.get('image_id')
                if image_id not in self.runtime.images or not isinstance(doc.get('rois'), list):
                    raise IntegrationError('SCREEN_INVALID','批量筛选结果与图像不匹配')
                items.append(self._art('screen:'+image_id,'screen',path,[image_id],image_id=image_id))
        elif tool == 'roi_screen':
            path = Path(result['screen'])
            doc = read_json(path)
            if doc.get('image_id') not in {None, image} or not isinstance(doc.get('rois'), list):
                raise IntegrationError('SCREEN_INVALID', 'ROI screen 结果与图像不匹配')
            items.append(self._art('screen:'+image, 'screen', path, target.get('parents', [image]),
                                   image_id=image))
        elif tool == 'propose_regions':
            items.append(self._art('regions:'+image, 'region_proposals', result['regions'],
                                   [image], image_id=image))
        elif tool in {'detect_fov_initial','detect_region','locate_feature','needle_scan','feature_scan'}:
            folder=Path(result['crops_dir']);metas=contracts.load_crop_metas(str(folder))
            completion=result['completion'];parents=[image]
            if tool in {'locate_feature','needle_scan','feature_scan'}:
                parents = target['parents']
            replacements=read_json(completion).get('replacements',{}) if tool=='detect_region' else {}
            for m in metas:
                if file_hash(m.source_image_path)!=self.runtime.images[image]['sha256']:
                    raise IntegrationError('SAMPLE_MISMATCH','检测产物不属于绑定原图')
                cid='crop:'+image+':'+m.crop_id
                if m.crop_id in replacements:
                    if replacements[m.crop_id]!=target.get('replace_id'):
                        raise IntegrationError('ROI_SCOPE','局部修复越过指定失败crop')
                    cid=replacements[m.crop_id]
                image_item=self._art(cid+':pixels','crop_pixels',m.png,parents,image_id=image)
                items.append(image_item)
                metadata={'identity_required': True} if tool in {'locate_feature','needle_scan','feature_scan'} else {}
                if m.raw.get('det_score') is not None:
                    metadata['det_score'] = m.raw.get('det_score')
                    metadata['det_confidence'] = m.raw.get('det_confidence')
                items.append(self._art(cid,'crop',m.json_path,[image,image_item['id']],image_id=image,
                                       **metadata))
            marker=('detected:'+image if tool=='detect_fov_initial' else
                    ('scanned:'+image if tool in {'needle_scan','feature_scan'} else 'detected:'+target['id']))
            items.append(self._art(marker,'detection',completion,parents,image_id=image))
            scales={m.px_per_um for m in metas if finite_number(m.px_per_um,minimum=1e-12)}
            if len(scales)==1 and not self.store.current()['artifacts'].get('scale:'+image,{}).get('valid'):
                scale_file=Path(completion).parent/'accepted_scale.json'
                write_json(scale_file,{'px_per_um':scales.pop(),'source':'trusted_detection_pipeline'})
                items.append(self._art('scale:'+image,'scale',scale_file,[image],image_id=image,
                                       px_per_um=read_json(scale_file)['px_per_um']))
        elif tool=='calibrate_fov':
            value=read_json(result['scale'])
            from ls_common import is_trusted_status
            if not is_trusted_status(value.get('status')) or not finite_number(value.get('px_per_um'),minimum=1e-12):
                raise IntegrationError('SCALE_UNTRUSTED','无可信检测侧标定')
            items.append(self._art('scale:'+image,'scale',result['scale'],[image],image_id=image,px_per_um=value['px_per_um']))
            # 在独立工作目录写新crop侧车；原B及旧sidecar只读。
            for cid,v in self.store.current()['artifacts'].items():
                if v['kind']=='crop' and v['valid'] and v['image_id']==image:
                    m=contracts.CropMeta.from_json(v['path']);side=copy.deepcopy(m.raw);side['px_per_um']=value['px_per_um']
                    side['png']=str(Path(m.png).resolve())
                    # 写侧仍由contracts唯一构造，不复制业务契约键表。
                    side=contracts.crop_sidecar(m.crop_id,str(Path(m.png).resolve()),m.source_image_path,m.stem,
                        m.feature_id,m.object_label,'',value['px_per_um'],m.tight_box,m.tight_box_local,m.loose_box,
                        m.crop_size,m.pad_info,m.granularity)
                    p=Path(result['scale']).parent/(m.crop_id+'.json');write_json(p,side)
                    items.append(self._art(cid,'crop',p,v['parents'],image_id=image,
                                           identity_required=v.get('identity_required', False)))
        elif tool=='segment_instance':
            root=Path(result['segmentation']);doc=read_json(root/'instances.json')
            if not doc.get('instances'):raise IntegrationError('EMPTY_MASK','无分割实例')
            parents=self.dependencies(tool,target);deps=list(parents)
            for p in sorted((root/'masks').rglob('*')):
                if p.is_file():
                    aid='segfile:'+target['id']+':'+p.relative_to(root).as_posix()
                    items.append(self._art(aid,'segmentation_file',p,parents,image_id=image));deps.append(aid)
            sid='seg:'+target['id']
            assessment=result.get('assessment')
            if self.purpose_quality() and not assessment:
                assessment=self.evaluate_segmentation(root,target)
            extra={}
            if assessment:
                value=contracts.quality_assessment(read_json(assessment))
                m=contracts.CropMeta.from_json(target['path'])
                if value['bindings'].get('crop')!=file_hash(m.png) or value['bindings'].get('crop_meta')!=file_hash(m.json_path):
                    raise IntegrationError('QUALITY_SCOPE','用途资格与当前crop不符')
                aid='quality:'+sid
                if value.get('identity_artifact_id'):parents=parents+[value['identity_artifact_id']]
                items.append(self._art(aid,'quality_assessment',assessment,parents,image_id=image))
                deps.append(aid);extra['assessment_id']=aid
            items.append(self._art(sid,'segmentation',root/'instances.json',deps,image_id=image,
                                   masks=str(root/'masks'),quality=result['quality'],crop_id=target['id'],
                                   locate_override=self.locate_override(target),**extra))
        elif tool=='image_exemplar_audit':
            root=Path(result['audit']);doc=read_json(root)
            if doc.get('schema') not in {'image-exemplar-audit/v1','image-exemplar-audit/v2'}:
                raise IntegrationError('AUDIT_OUTPUT_INVALID','图片示例审计 schema 不匹配')
            items.append(self._art('image_audit:'+target['id'],'image_audit',root,[target['id']],
                                   image_id=image, target_id=target['id'],
                                   contact_sheet=result.get('contact_sheet'),
                                   review_status='auto_unreviewed', non_blocking=True))
        elif tool=='classify_target':
            root=Path(result['classification']);doc=read_json(root/'classification.json')
            if doc.get('error'):raise IntegrationError('CLASSIFY_FAILED','分类阶段没有可采信结果')
            fid=str(doc.get('feature_id') or '').strip();label=str(doc.get('label') or '').strip()
            if not label:raise IntegrationError('CLASSIFY_EMPTY','分类结果缺类别')
            cid='classification:'+target['id']
            items.append(self._art(cid,'classification',root/'classification.json',[target['id']],
                                   image_id=image,crop_id=target.get('crop_id') or target['id'],
                                   feature_id=fid,label=label,
                                   glm_confidence=doc.get('glm_confidence'),
                                   yolo_confidence=doc.get('yolo_confidence'),
                                   confidence_source=doc.get('confidence_source'),
                                   confidence_score=doc.get('confidence_score')))
            meta=contracts.CropMeta.from_json(target['path'])
            if target.get('identity_required'):
                min_conf=self.config.get('identity_min_confidence','medium')
                allowed={'high'} if min_conf=='high' else (
                    {'high','medium'} if min_conf=='medium' else {'high','medium','low'})
                confidence=str(doc.get('confidence') or '')
                score=doc.get('confidence_score')
                try:
                    score=float(score)
                except (TypeError, ValueError):
                    score={'high':0.90,'medium':0.65,'low':0.30}.get(confidence.lower(),0.0)
                threshold=float(self.config.get('glm_identity_threshold',0.8))
                confident=score >= threshold and not doc.get('reject')
                items[-1]['identity_confidence']=confidence
                items[-1]['glm_confidence_score']=round(score,4)
                items[-1]['identity_verified'] = (fid == meta.feature_id
                    and not doc.get('reject') and (confidence in allowed or confident))
                items[-1]['identity_basis']=('glm_score>=%.2f' % threshold if confident
                                             else 'classifier_confidence')
                items[-1]['identity_review'] = bool(items[-1]['identity_verified'] and confidence != 'high')
            # 复核出的类别与crop不同：写一份**同像素、只改类别**的侧车，作为可重新分割的新目标。
            # 原crop与其掩码一律保留，失败/冲突仍可回溯；不是原地改写契约。
            if not target.get('identity_required') and fid and fid!=meta.feature_id:
                new_id=f"{meta.crop_id}__as_{fid}"
                side=contracts.crop_sidecar(new_id,str(Path(meta.png).resolve()),meta.source_image_path,
                    meta.stem,fid,label,'',meta.px_per_um,meta.tight_box,meta.tight_box_local,
                    meta.loose_box,meta.crop_size,meta.pad_info,meta.granularity)
                p=root/(new_id+'.json');write_json(p,side)
                items.append(self._art('crop:'+image+':'+new_id,'crop',p,[target['id'],cid],image_id=image))
        elif tool=='measure_instances':
            path=result['measurements'];parsed=list(contracts.iter_measurement_records(path))
            if not parsed:raise IntegrationError('EMPTY_MEASUREMENT','空测量不得提交为修复完成')
            mid='measurement:'+target['id'];parents=[target['id']]
            if result.get('delivery'):
                for p in sorted(Path(result['delivery']).rglob('*')):
                    if p.is_file():
                        aid='measurement_input:'+target['id']+':'+p.relative_to(result['delivery']).as_posix()
                        items.append(self._art(aid,'measurement_input',p,[target['id']],image_id=image));parents.append(aid)
            items.append(self._art(mid,'measurement',path,parents,image_id=image,
                                   diagnostic_review=bool(result.get('diagnostic_review')),
                                   component_selection=result.get('component_selection')))
            for rec in parsed:
                import math
                scale=self.store.current()['artifacts'].get('scale:'+image,{})
                if (rec.image_stem not in self.runtime.images[image]['stems'] or not scale.get('valid') or
                    not finite_number(rec.calibration_px_per_um,minimum=1e-12) or
                    not math.isclose(rec.calibration_px_per_um,scale['px_per_um'],rel_tol=1e-9)):
                    raise IntegrationError('MEASUREMENT_SCOPE','测量视野或尺度不合法')
                raw=copy.deepcopy(rec.raw);raw.pop('herb',None)
                raw.pop('quality_assessment',None)  # 外来字段不能赋予用途资格。
                if result.get('diagnostic_review'):
                    raw['diagnostic_review']=True
                    raw['component_selection']=result.get('component_selection','largest_component')
                    flags=list(raw.get('flags') or [])
                    if 'igm_review' not in flags:flags.append('igm_review')
                    if 'diagnostic_review' not in flags:flags.append('diagnostic_review')
                    raw['flags']=flags
                if target.get('locate_override'):
                    raw['locate_override']=True
                    flags=list(raw.get('flags') or [])
                    if 'locate_override' not in flags:flags.append('locate_override')
                    raw['flags']=flags
                iid=raw.get('instance_id')
                if type(iid) is not int or iid<0:raise IntegrationError('INSTANCE_ID','实例编号非法')
                rows.append({'scope':target['id'],'depends_on':[mid],'data':raw})
            replace=[target['id']]
        else:raise IntegrationError('TOOL_UNKNOWN','未知提交类型')
        return self.store.commit(parent,items,rows,replace)['version']

    def progress_signature(self):
        return digest([(k,v['sha256']) for k,v in sorted(self.store.current()['artifacts'].items())
                       if v['valid'] and v['kind'] in {'crop','scale','segmentation','measurement'}
                       and v.get('quality','valid')=='valid'])

    def unresolved(self):
        snap=self.store.current();arts=snap['artifacts']
        # 即使网络关闭、动作不被允许，也不能把未处理视野认作已完成。
        unqualified=any(v['valid'] and v['kind']=='segmentation' and v.get('quality')!='valid' and
            (not self.purpose_quality() or not self.assessment(v) or not any(u['allowed'] for u in self.assessment(v)['uses'].values())) for v in arts.values())
        return (not snap['records'] or any(a['tool'] not in {'propose_regions','locate_feature'}
                                           and not a['args']['target_id'].startswith('roi:recovery:')
                                           for a in self.propose(include_offline=True)) or unqualified or
                any(v['status'] in {'pending','failed'} for v in read_json(self.operations)['entries'].values()))
