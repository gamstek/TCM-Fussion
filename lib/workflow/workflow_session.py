"""绑定样品、输入、运行策略与阶段后端，维护可恢复的操作记录和时间线。由现有证据快照生成工具目标，执行前检查身份、标定、预算与重复请求。阶段返回后校验并提交新版本，后续测量和规则评分以该版本为依据。

主要定义：
- TOOLS: 工作流允许注册的工具名元组。
- WorkflowSession: 维护预算、工件存储与时间线，负责 targets/propose/execute/coverage/seed/close 等会话流程。
- 其余 `_timeline_*`、`_prior_feature_locate` 等方法：时间线、去重与测量满足性的内部辅助。

关联：
- 调用方 「lib/workflow/workflow_runtime.py」、「tests/application/test_multi_view.py」、「tests/workflow/test_retry_policy.py」。
- 直接依赖 「lib/decision/candidate_policy.py」、「lib/evidence/evidence_store.py」、「lib/evidence/instance_ledger.py」、「lib/evidence/stage_recording.py」、「lib/runtime/ledger_lock.py」、「lib/runtime/probe_budget.py」、「lib/runtime/resource_budget.py」、「lib/runtime/service_errors.py」。
"""
import copy
import time
from pathlib import Path
import contracts
import paths
from TCM_Fusion.lib.workflow.adapters import interpreter
from TCM_Fusion.lib.evidence.evidence_store import EvidenceStore
from TCM_Fusion.lib.runtime.ledger_lock import acquire, release
from TCM_Fusion.lib.runtime.resource_budget import close_budget, reserve
from TCM_Fusion.lib.runtime.storage import IntegrationError, digest, file_hash, project_path, read_json, write_json, finite_number
from TCM_Fusion.lib.workflow.workflow_backend import WorkflowBackend
from TCM_Fusion.lib.vision.box_quality import assess_meta
from TCM_Fusion.lib.runtime.service_errors import stage_error_kind
from TCM_Fusion.lib.evidence.instance_ledger import attempt as instance_attempt, project as project_instances


TOOLS=('detect_fov_initial','calibrate_fov','segment_instance','classify_target','classify_crops_batch','measure_instances','detect_region',
       'locate_feature','feature_scan','hybrid_scan','image_exemplar_audit')


class WorkflowSession:
    def __init__(self, runtime, root, config, *, allow_network=False, backend=None):
        self.runtime=runtime;self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True)
        import threading
        self.cancel_event=threading.Event()
        self.config=copy.deepcopy(config);self.sample_id=runtime.sample_id
        if self.config.get('agent_policy') == 'adaptive_v2':
            self.config.setdefault('physical_parallel',True)
            self.config.setdefault('max_parallel',4)
        self.python=interpreter(config.get('python'));self.allow_network=allow_network
        self.yolo_signature = {}
        self.cnn_signature = None
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
            if config.get('cnn_weights') or config.get('cnn_meta'):
                from TCM_Fusion.lib.vision.stage2_classifier import model_profile
                self.cnn_signature = model_profile(config)
        self.code_signature=WorkflowBackend.code_signature()
        self.backend=backend or WorkflowBackend()
        if backend is not None and not runtime.simulation:
            raise IntegrationError('SIMULATION_SCOPE','真实运行禁止替身后端')
        self.identity=digest({'sample':self.sample_id,'images':sorted(runtime.images),
                              'config':config,'simulation':runtime.simulation,'vlm':runtime.config_identity,
                              'backend':self.code_signature,'python':file_hash(self.python),
                              **({'yolo': self.yolo_signature} if self.yolo_signature else {}),
                              **({'stage2':self.cnn_signature} if self.cnn_signature else {})})
        self.store=EvidenceStore(self.root/'evidence',self.identity)
        self.budget_path=self.root/'resources.json';self.operations=self.root/'operations.json'
        self.timeline_path=self.root/'action_timeline.json'
        self._timeline={'schema':'action-timeline/v1','started_at':time.time(),'actions':{}}
        if self.timeline_path.exists():
            try:
                self._timeline=read_json(self.timeline_path)
            except (OSError,ValueError):
                pass
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

    def _timeline_key(self,action):
        tool=str(action.get('tool') or '')
        target=str(action.get('target_id') or (action.get('args') or {}).get('target_id') or '')
        return tool+'|'+target

    def _timeline_doc(self):
        if self.timeline_path.exists():
            try:
                return read_json(self.timeline_path)
            except (OSError,ValueError):
                pass
        return {'schema':'action-timeline/v1','started_at':time.time(),'actions':{}}

    def _timeline_with_lock(self, mutate):
        h=acquire(self.timeline_path)
        try:
            doc=self._timeline_doc()
            mutate(doc)
            write_json(self.timeline_path,doc)
            self._timeline=doc
        finally:
            release(h)

    def record_offers(self,actions,evidence_version):
        now=time.time();seen=set()
        def mutate(doc):
            for action in actions:
                key=self._timeline_key(action)
                if key=='|' or key in seen:continue
                seen.add(key)
                row=doc['actions'].setdefault(key,{})
                if not row:
                    row.update({'action_id':key,'tool':action.get('tool'),
                                'target_id':(action.get('args') or {}).get('target_id'),
                                'evidence_version':evidence_version,
                                'eligible_at':now,'eligibility_basis':'offered',
                                'first_offered_at':now,'last_offered_at':now,
                                'offer_count':1,'selected_at':None,'started_at':None,
                                'ended_at':None,'status':'offered','error_code':None,
                                'blocking_reasons':[]})
                else:
                    row['last_offered_at']=now
                    row['offer_count']=int(row.get('offer_count') or 0)+1
        self._timeline_with_lock(mutate)

    def _timeline_row(self,action,doc=None):
        doc=doc or self._timeline
        key=self._timeline_key(action)
        row=doc['actions'].get(key)
        if row:return row,key
        action_id=str(action.get('action_id') or '')
        row=doc['actions'].get(action_id)
        if row:return row,action_id
        return None,None

    def record_selected(self,action):
        def mutate(doc):
            row,key=self._timeline_row(action,doc)
            if not row:return
            now=time.time();row['selected_at']=row.get('selected_at') or action.get('selected_at') or now
            row['started_at']=now;row['status']='running';row['selected_action_id']=action.get('action_id')
        self._timeline_with_lock(mutate)

    def record_finished(self,action,status,error_code=None):
        def mutate(doc):
            row,key=self._timeline_row(action,doc)
            if not row:return
            now=time.time();row['ended_at']=now;row['status']=status;row['error_code']=error_code
            if row.get('selected_at'):row['offer_to_select_s']=round(row['selected_at']-row['first_offered_at'],3)
            if row.get('selected_at'):row['select_to_start_s']=round(row.get('started_at',row['selected_at'])-row['selected_at'],3)
            if row.get('started_at'):row['execution_s']=round(now-row['started_at'],3)
        self._timeline_with_lock(mutate)

    def remaining_seconds(self):
        return max(0.1,read_json(self.budget_path)['deadline']-time.time())

    def call_timeout(self):
        """Single worker timeout; never let one call consume the whole workflow."""
        cap = float(self.config.get('tool_timeout_seconds', 600))
        return max(1.0, min(self.remaining_seconds(), cap))

    def prepare_reserved_stage(self,key,epoch):
        from TCM_Fusion.lib.workflow.physical_schedule import prepare
        return prepare(self,key,epoch)

    def close(self,reason):
        handle = acquire(self.operations)
        try:
            close_budget(self.budget_path, reason)
            if self.config.get('physical_parallel'):
                from TCM_Fusion.lib.workflow.physical_schedule import fence_locked
                fence_locked(self, reason)
        finally:
            release(handle)

    def diagnostic_quality(self):
        """仅联调允许review工件进入真实测量；不改原始QA/flags或最终规则。"""
        return self.config.get('quality_policy','strict')=='diagnostic_review'

    def purpose_quality(self):
        return self.config.get('quality_evaluation','legacy')=='purpose_v1'

    def crop_segmentation_admission(self, target):
        """裁图纹理只作建议；低细节图须有同类别身份确认才能尝试分割。

        供工具提议与规划摘要共用，只判断裁图/身份；标定、预算和mask仍走原检查，
        不授予尺寸或裁决资格。
        low_detail统一作为提示，不提供按纹理一票否决分割的旧策略分支。
        """
        quality = target.get('box_quality')
        result = {'quality': quality, 'allowed': True, 'reason': 'existing_prerequisites',
                  'quality_advisory_only': False, 'grants_dimension_evidence': False}
        if not self.config.get('box_quality_enabled', True):
            return result
        if quality == 'blank':
            return {**result, 'allowed': False, 'reason': 'blank_crop'}
        if quality != 'low_detail':
            return result
        result.update(quality_advisory_only=True)
        classification = self.store.current()['artifacts'].get('classification:' + target['id'], {})
        if not classification.get('valid') or not classification.get('identity_verified'):
            return {**result, 'allowed': False, 'reason': 'identity_confirmation_required'}
        meta = contracts.CropMeta.from_json(target['path'])
        if classification.get('feature_id') != meta.feature_id:
            return {**result, 'allowed': False, 'reason': 'classification_feature_mismatch'}
        return {**result, 'reason': 'low_detail_advisory'}

    def identity_pending(self):
        """最高 det_score 的 locate crop，尚未尝试 classify。"""
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

    def charge(self,pool,key,*,network=True,probe=None):
        """预约资源；`network=False` 用于 worker 内已有受控HTTP守卫、网络由其请求键自记的场景。"""
        extra_pool = 'network' if network and pool in {'planning', 'observation', 'finalization'} else None
        if not reserve(self.budget_path,pool,key,self.limits,identity=self.identity,extra_pool=extra_pool,probe=probe):
            raise IntegrationError('RESOURCE_REPLAY','同资源操作已预约，禁止自动重发')
        return True

    def _art(self, identifier, kind, path, parents, **metadata):
        p=Path(path).resolve()
        return {'id':identifier,'kind':kind,'path':str(p),'sha256':file_hash(p),
                'parents':list(parents),**metadata}

    def _crop_metadata(self, meta, quality, *, kind, target):
        """One metadata builder for initial detect, hybrid and locate crops."""
        data={'identity_required': True} if kind=='feature' else {}
        data.update({'box_source':'yolo' if kind=='detect' else 'glm',
                     'box_quality':quality['status'],'box_quality_score':quality['score'],
                     'box_quality_flags':quality['flags'],'box_quality_metrics':quality['metrics'],
                     'box_quality_retry':bool(target.get('box_quality_retry', False))})
        if meta.raw.get('det_score') is not None:
            data['det_score']=meta.raw.get('det_score')
            data['det_confidence']=meta.raw.get('det_confidence')
        return data

    def blocked_tools(self):
        """任务内按工具熔断；locate/classify 允许有界重试后再熔断。"""
        limit = int(self.config.get('max_tool_failures', 2))
        retryable = {'locate_feature', 'classify_target', 'feature_scan', 'detect_fov_initial'}
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

    @staticmethod
    def _box_overlap_ratio(roi, box):
        if not roi or not box or len(roi) != 4 or len(box) != 4:
            return 0.0
        ix1,iy1=max(roi[0],box[0]),max(roi[1],box[1])
        ix2,iy2=min(roi[2],box[2]),min(roi[3],box[3])
        inter=max(0.0,ix2-ix1)*max(0.0,iy2-iy1)
        area=max(1.0,(box[2]-box[0])*(box[3]-box[1]))
        return inter/area

    def _existing_feature_box(self, image, feature, roi):
        for value in self.store.current()['artifacts'].values():
            if not value.get('valid') or value.get('kind')!='crop' or value.get('image_id')!=image:
                continue
            try:meta=contracts.CropMeta.from_json(value['path'])
            except (OSError,ValueError,contracts.ContractError):continue
            if meta.feature_id!=feature:continue
            if self._box_overlap_ratio(roi,meta.tight_box)>=.5:return True
        return False

    def _prior_feature_locate(self, image, feature):
        for row in read_json(self.operations)['entries'].values():
            if (row.get('tool')=='locate_feature' and row.get('status')=='done'
                    and row.get('feature_id')==feature and row.get('image_id')==image):
                return True
        return False

    def _feature_measurement_satisfied(self, feature_id):
        """A valid measurement inside an active candidate's required bound is enough."""
        if not self.config.get('suppress_repeated_measurement', False) or not feature_id:
            return False
        interest = set(self.runtime.candidate_interest().get('ids') or [])
        for candidate in self.runtime.kb.candidates():
            if candidate.get('id') not in interest:
                continue
            for criterion in candidate.get('criteria') or []:
                if criterion.get('feature') != feature_id or criterion.get('role') != 'required':
                    continue
                quant = criterion.get('quant') or {}
                if not finite_number(quant.get('min'), minimum=0) or not finite_number(quant.get('max'), minimum=0):
                    continue
                dim = quant.get('dim')
                for record in self.runtime.records:
                    if record.get('feature') != feature_id:
                        continue
                    value = record.get(dim)
                    if not finite_number(value, minimum=0):
                        if record.get('primary_dimension') == dim:
                            value = record.get('primary_value_um')
                    if finite_number(value, minimum=0) and quant['min'] <= value <= quant['max']:
                        return True
        return False

    def targets(self):
        snap=self.store.current();arts=snap['artifacts'];targets={}
        for identifier,v in arts.items():
            if not v['valid']:continue
            if v['kind']=='image':
                targets[identifier]={**v,'source':v['path'],'image_id':identifier}
            elif v['kind'] in {'crop','segmentation'}:targets[identifier]=v
        if self.config.get('classify_batch_enabled', True):
            for image in self.runtime.images:
                pending=[{**v,'id':cid} for cid,v in arts.items()
                         if v.get('valid') and v.get('kind')=='crop' and v.get('image_id')==image
                         and not (arts.get('classification:'+cid) or {}).get('valid')]
                if len(pending)>=2:
                    pending.sort(key=lambda c:(bool(c.get('identity_required')),
                                               float(c.get('det_score') or 0)),reverse=True)
                    limit=max(1,int(self.config.get('fast_classify_max_per_view',3)))
                    pending=pending[:limit]
                    key='classify_batch:'+image
                    targets[key]={'id':key,'kind':'classify_batch','image_id':image,
                                  'crops':pending,'parents':[image]}
        if self.config.get('fast_recovery_selection', 'legacy_bundle') == 'per_feature':
            targets.update(self.fast_recovery_targets(arts))
        else:
            # Compatibility: 旧配置仍保留每图一次 required 合并补扫。
            operations=read_json(self.operations)['entries']
            required=[]
            try:
                required=self.recovery_features()
            except Exception:
                required=[]
            present = set().union(*self._classified_features_by_image(arts).values())
            missing = [fid for fid in required if fid and fid not in present]
            for image in self.runtime.images:
                if not missing:
                    break
                if not arts.get('detected:'+image,{}).get('valid'):
                    continue
                image_crops=[(cid,a) for cid,a in arts.items()
                             if a.get('valid') and a.get('kind')=='crop' and a.get('image_id')==image]
                if any('classification:'+cid not in arts for cid,_ in image_crops):
                    continue
                fallback_id='hybrid_scan:'+image
                if any(str(row.get('target_id') or '')==fallback_id
                       for row in operations.values()):
                    continue
                features=list(dict.fromkeys(
                    [fid for fid in (self.config.get('feature_scan_features') or []) if fid in missing] + missing
                ))[:max(1,int(self.config.get('fast_fallback_max_features',1)))]
                if features:
                    targets[fallback_id]={'id':fallback_id,'kind':'hybrid_scan','features':features,
                        'image_id':image,'source':self.runtime.images[image]['path'],'parents':[image],
                        'max_locate_calls':int(self.config.get('fast_fallback_max_locate_calls',4)),
                        'px_per_um':arts.get('scale:'+image,{}).get('px_per_um')}
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
        # 框内容不可用：保留原 crop，但允许一次基于 loose_box 的重定位；不删除原证据。
        retry_done={r.get('target_id') for r in read_json(self.operations)['entries'].values()
                    if str(r.get('target_id') or '').startswith('locate:roi:boxq:')}
        for cid,crop in arts.items():
            if not self.config.get('box_quality_retry', True):
                break
            if not crop.get('valid') or crop.get('kind')!='crop':
                continue
            if crop.get('box_quality') not in {'blank','low_detail'}:
                continue
            if crop.get('box_quality_retry'):
                continue
            try:m=contracts.CropMeta.from_json(crop['path'])
            except (OSError,ValueError,contracts.ContractError):continue
            if not m.loose_box or not m.feature_id:continue
            image=crop.get('image_id');src=self.runtime.images[image]['path']
            if self._existing_feature_box(image,m.feature_id,m.loose_box):continue
            if self._prior_feature_locate(image,m.feature_id):continue
            key='locate:roi:boxq:'+image+':'+digest([cid])[:12]
            if key in retry_done:continue
            targets[key]={'id':key,'kind':'roi','image_id':image,'source':str(src),
                'roi':[int(v) for v in m.loose_box],'feature_id':m.feature_id,
                'parents':[cid,image],'px_per_um':arts.get('scale:'+image,{}).get('px_per_um'),
                'box_quality_retry':True}
        for i,region in enumerate(self.config.get('regions',[])):
            matches=[(k,v) for k,v in self.runtime.images.items() if region['image_stem'] in v['stems']]
            if len(matches)!=1:raise IntegrationError('ROI_SCOPE','ROI不能唯一绑定视野')
            key,v=matches[0];scale=arts.get('scale:'+key,{})
            targets['roi:'+str(i)]={'id':'roi:'+str(i),'image_id':key,'source':str(v['path']),
                'roi':region['box'],'parents':[key],'px_per_um':scale.get('px_per_um')}
        return targets

    def recovery_options(self):
        """Candidate-bound scan options, shared across roles; never rule evidence.

        Expose all supported options before the model chooses. Do not truncate by
        taxonomy/first-fold order or invent a default feature for a missing ID.
        """
        from detect_core import FEATURE_ID
        from prompts import CLASSIFY_HINTS
        from TCM_Fusion.lib.decision.candidate_policy import criteria_diff
        supported = {fid for label, fid in FEATURE_ID.items() if label in CLASSIFY_HINTS}
        interest = self.runtime.candidate_interest()
        roles = set(self.config.get('fast_recovery_roles', ['required', 'supporting']))
        options = {}
        for row in criteria_diff(self.runtime.kb, interest['ids'], {}):
            for criterion in row['criteria']:
                fid, role = criterion['feature'], criterion['role']
                if fid not in supported or role not in roles:
                    continue
                option = options.setdefault(fid, {'feature_id': fid, 'criteria': []})
                binding = {'candidate_id': row['candidate_id'], 'role': role}
                if binding not in option['criteria']:
                    option['criteria'].append(binding)
        return list(options.values())

    def fast_recovery_targets(self, arts):
        """Explicit single-feature alternatives with the existing per-view caps.

        The model selects which feature to scan. Offering an option spends no
        budget; pending, done and failed attempts reserve their requested tiles.
        These options do not satisfy coverage, create evidence or force a scan.
        """
        from TCM_Fusion.lib.runtime.probe_budget import enabled, available
        unified = enabled(self.runtime)
        options = self.recovery_options()
        present = set().union(*self._classified_features_by_image(arts).values())
        entries = list(read_json(self.operations)['entries'].values())
        feature_cap = max(1, int(self.config.get('fast_fallback_max_features', 2)))
        locate_cap = max(1, int(self.config.get('fast_fallback_max_locate_calls', 6)))
        tile_cap = max(1, int(self.config.get('feature_scan_max_tiles', 4)))
        supporting_total_cap=max(0,int(self.config.get('supporting_probe_total',0)))
        supporting_per_candidate=max(1,int(self.config.get('supporting_probe_per_candidate',2)))
        support_used={}
        support_total=0
        for row in entries:
            if row.get('tool')!='feature_scan':continue
            roles=[c for c in (row.get('criteria') or []) if c.get('role')=='supporting']
            if not roles:continue
            support_total+=1
            for c in roles:support_used[c.get('candidate_id')]=support_used.get(c.get('candidate_id'),0)+1
        global_support_features={row.get('feature_id') for row in entries if row.get('tool')=='feature_scan'
                                 and any(c.get('role')=='supporting' for c in (row.get('criteria') or []))}
        out = {}
        for image, info in self.runtime.images.items():
            if not arts.get('detected:' + image, {}).get('valid'):
                continue
            crops = [(cid, a) for cid, a in arts.items()
                     if a.get('valid') and a.get('kind') == 'crop' and a.get('image_id') == image]
            unresolved = [(cid, a) for cid, a in crops
                          if not arts.get('classification:' + cid, {}).get('valid')]
            if self.config.get('agent_policy')!='adaptive_v2' and any(a.get('identity_required') for _, a in unresolved):
                continue
            classified_in_view = sum(v.get('valid') and v.get('kind') == 'classification'
                                     and v.get('image_id') == image for v in arts.values())
            classify_cap = max(1, int(self.config.get('fast_classify_max_per_view', 3)))
            if self.config.get('agent_policy')!='adaptive_v2' and unresolved and classified_in_view < classify_cap:
                continue
            prefix = 'feature_scan:' + image + ':'
            attempts = [row for row in entries
                        if row.get('tool') == 'feature_scan'
                        and str(row.get('target_id') or '').startswith(prefix)]
            attempted = {row['target_id'] for row in attempts}
            used_tiles = sum(int(row.get('scan_tiles_reserved') or tile_cap) for row in attempts)
            remaining = locate_cap - used_tiles
            if len(attempted) >= feature_cap or remaining <= 0:
                continue
            for option in options:
                fid = option['feature_id']
                key = prefix + fid
                if fid in present or key in attempted:
                    continue
                roles={c.get('role') for c in option.get('criteria') or []}
                support_candidates={c.get('candidate_id') for c in option.get('criteria') or []
                                    if c.get('role')=='supporting'}
                is_support='supporting' in roles and 'required' not in roles
                if is_support and supporting_total_cap and not unified:
                    if fid in global_support_features or support_total>=supporting_total_cap:
                        continue
                    if any(support_used.get(cid,0)>=supporting_per_candidate for cid in support_candidates):
                        continue
                    support_total+=1
                    global_support_features.add(fid)
                    for cid in support_candidates:
                        support_used[cid]=support_used.get(cid,0)+1
                out[key] = {'id': key, 'kind': 'feature_scan', 'feature_id': fid,
                            'image_id': image, 'source': str(info['path']), 'parents': [image],
                            'px_per_um': arts.get('scale:' + image, {}).get('px_per_um'),
                            'max_tiles': min(tile_cap, remaining), 'fast_recovery': True,
                            'remaining_scan_features': feature_cap - len(attempted),
                            'remaining_scan_tiles': remaining,
                            'criteria': option['criteria']}
                if unified and not available(self.runtime, {'tool': 'feature_scan',
                        'args': {'target_id': key}}, out[key]):
                    del out[key]
        return out

    def recovery_features(self):
        from detect_core import FEATURE_ID
        from prompts import CLASSIFY_HINTS
        from TCM_Fusion.lib.decision.candidate_policy import candidates, criteria_diff
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
        if tool=='classify_crops_batch':
            ids += [c['id'] for c in target.get('crops') or []]
        if tool in {'segment_instance','detect_region','locate_feature'}:
            scale='scale:'+target['image_id']
            if scale in arts and arts[scale]['valid']:ids+=[scale]
        if tool == 'segment_instance' and target.get('identity_required'):
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

    def semantic_attempt(self, tool, target):
        """D1 内部尝试身份；不代替权限、预算或跨阶段 contracts 校验。"""
        parameters = {}
        if tool == 'measure_instances':
            plan = self.review_measure_plan(target) or {}
            parameters = {k: plan.get(k) for k in ('component_ids', 'component_policy')}
        return instance_attempt(tool, target, self.store.current(),
            {'backend': self.code_signature, 'config': self.runtime.config_identity,
             'workflow': digest(self.config), 'stage2': self.cnn_signature}, parameters)

    def _write_instance_index(self, operations):
        """调用者持 operations 锁；索引可从证据/执行账本重建，不是第二个证据头。"""
        write_json(self.root/'instance_ledger.json', project_instances(self.store.current(), operations))

    def worker_context(self, work, key):
        """普通及自动子阶段共用预算上下文；向后端传文件路径而非配置对象。"""
        context = work / 'budget_context.json'
        write_json(context, {'operation_key': key, 'budget_path': str(self.budget_path),
                             'limits': self.limits, 'identity': self.identity})
        return context

    def _auto_measure(self, target, work, parent):
        """保留原自动测量路径，同时登记其语义尝试；失败不可因换名重新执行。"""
        semantic = self.semantic_attempt('measure_instances', target)
        key = 'auto:' + semantic['semantic_key']
        handle = acquire(self.operations)
        try:
            doc = read_json(self.operations)
            if any(row.get('semantic_key') == semantic['semantic_key']
                   for row in doc['entries'].values()):
                return parent
            row = {**semantic, 'operation_key': key, 'tool': 'measure_instances',
                   'target_id': target['id'], 'status': 'pending', 'accepted': False,
                   'automatic': True}
            doc['entries'][key] = row; write_json(self.operations, doc)
            self._write_instance_index(doc)
        finally:
            release(handle)
        measure_started = time.monotonic()
        try:
            work.mkdir(parents=True, exist_ok=False)
            context = self.worker_context(work, key)
            result = self.backend.execute(self, 'measure_instances', target, work, context)
            if self.remaining_seconds() <= 0.1 or read_json(self.budget_path).get('closed'):
                raise IntegrationError('WORKFLOW_TIMEOUT', '自动测量迟到结果不提交')
            parent = self.accept('measure_instances', target, result, parent)
            from TCM_Fusion.lib.evidence.stage_recording import bundle
            row.update(status='done', accepted=True, evidence_version=parent, bundle=bundle(result, work))
        except Exception as exc:
            code = getattr(exc, 'code', 'WORKFLOW_STAGE_FAILED')
            row.update(status='failed', error_code=code, error_kind=stage_error_kind(code))
        row['latency_ms'] = round((time.monotonic()-measure_started)*1000,3)
        handle = acquire(self.operations)
        try:
            doc = read_json(self.operations); doc['entries'][key] = row
            write_json(self.operations, doc); self._write_instance_index(doc)
        finally:
            release(handle)
        return parent

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
        max_locate = self.config.get('max_locate_attempts')
        if max_locate is None:
            max_locate = max(4, int(self.config.get('locate_attempts_per_view', 2)) * len(self.runtime.images)
                             + int(self.config.get('locate_attempt_reserve', 2)))
        max_locate = int(max_locate)
        def offer(tool,t,priority):
            if tool in blocked:return
            if tool not in self.config.get('enabled_tools',TOOLS) and not include_offline:return
            if (tool not in {'measure_instances','detect_region'} and not self.allow_network and not self.runtime.simulation
                and not include_offline):return
            key=self.operation_key(tool,t)
            semantic = self.semantic_attempt(tool, t)
            if semantic and any(row.get('semantic_key') == semantic['semantic_key']
                                for row in done.values()):
                return
            if key not in done:
                action = self.runtime.action(tool,{'target_id':t['id'],'evidence_version':snap['version']},
                    'workflow_repair',priority=priority)
                if not any(a['action_id'] == action['action_id'] for a in out):
                    out.append(action)
        crops=[]
        audit_done=sum(1 for row in done.values()
                       if row.get('tool')=='image_exemplar_audit')
        audit_cap=max(1,int(self.config.get('image_exemplar_audit_max_crops',6)))
        target_map=self.targets()
        for tid,t in target_map.items():
            tool=None;image=t['image_id']
            origin = t if t.get('kind')=='crop' else arts.get(t.get('crop_id'), {})
            if (t.get('kind') == 'segmentation' and origin
                    and not self.crop_segmentation_admission(origin)['allowed']):
                continue
            if origin.get('identity_required') and paused_features:
                meta=contracts.CropMeta.from_json(origin['path'])
                if meta.feature_id in paused_features:
                    continue
            if tid.startswith('locate:'):
                if locate_attempts < max_locate:
                    tool = 'locate_feature'
            elif tid.startswith('roi:'):
                if 'detected:'+image in arts:tool='detect_region'
            elif t.get('kind') == 'feature_scan':
                tool = 'feature_scan'
            elif t.get('kind') == 'hybrid_scan':
                tool = 'hybrid_scan'
            elif t['kind']=='image':
                if 'detected:'+image not in arts:tool='detect_fov_initial'
                else:
                    if not arts.get('scale:'+image,{}).get('valid'):tool='calibrate_fov'
            elif t['kind']=='classify_batch':
                if any(not (arts.get('classification:'+c['id']) or {}).get('valid')
                       for c in t.get('crops') or []):
                    tool='classify_crops_batch'
            elif t['kind']=='crop':
                crops.append(t)
                classification=arts.get('classification:'+tid,{})
                # 标定重写侧车后，历史分类仍保留身份字段。只有当前有效版本可以授权分割。
                if not classification.get('valid'):
                    classification = {}
                fast_classify_cap=max(1,int(self.config.get('fast_classify_max_per_view',3)))
                classified_in_view=sum(1 for v in arts.values()
                    if v.get('valid') and v.get('kind')=='classification'
                    and v.get('image_id')==image)
                needs_classification = (not classification.get('valid')
                                        or classification.get('review_required'))
                if (needs_classification and not self._classify_attempted(t)
                        and (t.get('identity_required') or classification.get('review_required')
                             or classified_in_view < fast_classify_cap)):
                    # stage2/CNN 很快：所有候选都先识别，不因框质量或高分绕过而跳过分类证据；
                    # fast 模式对普通 YOLO crop 设置每视野上限，避免几十个低价值 crop 串行吃满预算。
                    offer('classify_target', t, (2, 'classify_all_crops', tid))
                if not self.crop_segmentation_admission(t)['allowed']:
                    continue
                if (t.get('identity_required')
                        and not classification.get('identity_verified')):
                    continue
                if self.config.get('image_exemplar_audit', False) and audit_done<audit_cap:
                    meta=contracts.CropMeta.from_json(t['path'])
                    score=0.0
                    try:score=float(t.get('det_score') or 0)
                    except (TypeError,ValueError):pass
                    skip_classified=bool(self.config.get('image_exemplar_audit_skip_classified',True)
                        and classification.get('identity_verified'))
                    if (meta.feature_id and score>=float(self.config.get('image_exemplar_audit_min_det_score',.8))
                            and not skip_classified):
                        offer('image_exemplar_audit', t, (2, 'image_exemplar_audit', tid))
                        audit_done+=1
                scale=arts.get('scale:'+image,{})
                seg=arts.get('seg:'+tid,{})
                review_measure=bool(review_plan and review_plan['target_id']=='seg:'+tid)
                usable=seg.get('quality')=='valid' or (self.diagnostic_quality() and seg.get('quality')=='review') or review_measure
                if self.purpose_quality() and seg.get('assessment_id'):usable=True
                if scale.get('valid') and (not seg.get('valid') or not usable):
                    if self._segment_reviewed(t, review_plan):
                        continue
                    try:meta=contracts.CropMeta.from_json(t['path'])
                    except (OSError,ValueError,contracts.ContractError):meta=None
                    if (meta and self._feature_measurement_satisfied(meta.feature_id)
                            and self.config.get('agent_policy') != 'adaptive_v2'):
                        continue
                    if self.config.get('auto_measure_after_segment', False):
                        t = {**t, 'measure_if': True}
                    tool='segment_instance'
            elif t['kind']=='segmentation' and (t.get('quality')=='valid' or
                    (self.diagnostic_quality() and t.get('quality')=='review') or
                    (review_plan and review_plan['target_id']==tid)):
                feature=None
                crop=arts.get(t.get('crop_id') or '')
                if crop:
                    try:feature=contracts.CropMeta.from_json(crop['path']).feature_id
                    except (OSError,ValueError,contracts.ContractError):feature=None
                if not arts.get('measurement:'+tid,{}).get('valid') and (
                        self.config.get('agent_policy') == 'adaptive_v2' or not self._feature_measurement_satisfied(feature)):
                    tool='measure_instances'
            if t.get('kind')=='segmentation' and self.purpose_quality():
                assessment=self.assessment(t)
                if not assessment or not any(assessment['uses'][u]['allowed'] for u in ('length','width','area')):
                    tool=None
                elif not arts.get('measurement:'+tid,{}).get('valid'):tool='measure_instances'
            if tool:offer(tool,t,(0,tool,tid))
        # 复核类别是“修复”而非主路径：排在物理阶段之后，只在观察到的存在特征与crop类别相左时出现。
        pending = [t for t in crops if not arts.get('classification:'+t['id'], {}).get('valid')
                   and not self._classify_attempted(t)
                   and (t.get('identity_required') or self.label_conflict(t))]
        best = max(pending, key=lambda c: float(c.get('det_score') or 0), default=None)
        for t in crops:
            if arts.get('classification:'+t['id'],{}).get('valid'):continue
            if self._classify_attempted(t):continue
            if not t.get('identity_required') and not self.label_conflict(t):continue
            if t.get('identity_required') and best is not None and t['id'] != best['id']:
                continue
            offer('classify_target',t,(1,'classify_target',t['id']))
        self.record_offers(out,snap['version'])
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

    def _classified_features_by_image(self, artifacts):
        features = {}
        for value in artifacts.values():
            image = value.get('image_id')
            feature = value.get('feature_id')
            if (value.get('valid') and value.get('kind') == 'classification'
                    and image in self.runtime.images and isinstance(feature, str) and feature.strip()):
                features.setdefault(image, set()).add(feature)
        return features

    def coverage(self):
        """逐视野覆盖审计：detected / located / useful；propose/screen 已随工具移除。"""
        import re
        images = list(self.runtime.images)
        detected, located = set(), set()
        artifacts = self.store.current()['artifacts']
        features_by_image = self._classified_features_by_image(artifacts)
        useful = set(features_by_image)
        image_pattern = re.compile(r'img_[0-9a-f]{16,}')
        for row in read_json(self.operations).get('entries', {}).values():
            if row.get('status') not in {'done', 'failed'}:
                continue
            tool, target = row.get('tool'), str(row.get('target_id') or '')
            if tool == 'detect_fov_initial':
                if row.get('status') == 'done' and target in images:
                    detected.add(target)
            elif tool == 'locate_feature':
                hit = image_pattern.search(target)
                if hit:
                    located.add(hit.group(0))
        try:
            required_features=set(self.recovery_features())
        except Exception:
            required_features=set()
        sample_missing = sorted(required_features - set().union(*features_by_image.values()))
        rows = []
        for image in images:
            rows.append({'image_id': image, 'detected': image in detected,
                         'proposed': False, 'screened': False,
                         'located': image in located, 'useful': image in useful,
                         'required_useful': bool(required_features & features_by_image.get(image, set())),
                         'missing_required': sorted(required_features - features_by_image.get(image, set())),
                         'top_rois': None, 'located_rois': []})
        located_all = len(located) == len(images)
        try:
            interest = set(self.runtime.candidate_interest().get('ids') or [])
        except Exception:
            interest = set()
        candidate_required = []
        if interest:
            for candidate in self.runtime.kb.candidates():
                if candidate['id'] not in interest:
                    continue
                for crit in candidate.get('criteria') or []:
                    if crit.get('role') != 'required':
                        continue
                    fid = crit.get('feature')
                    views = [image for image in images
                             if fid in features_by_image.get(image, set())]
                    state = 'DETECTED_PENDING' if views else 'NOT_DETECTED'
                    candidate_required.append({'candidate_id': candidate['id'],
                                               'feature_id': fid, 'state': state,
                                               'views': views,
                                               'provisional_weight': 0.5 if views else 0.0})
        missing_candidate = [x for x in candidate_required if x['state'] == 'NOT_DETECTED']
        pending_candidate = [x for x in candidate_required if x['state'] == 'DETECTED_PENDING']
        required_all = (not missing_candidate if interest else not sample_missing)
        missing_required = (missing_candidate if interest else
                            ([{'image_id': image, 'features': sample_missing}
                              for image in images if sample_missing]))
        return {
            'views': rows,
            'detected_all': len(detected) == len(images),
            'proposed_all': False,
            'screened_all': False,
            'located_all': located_all,
            'useful_all': len(useful) == len(images),
            'missing_useful': [x['image_id'] for x in rows if not x['useful']],
            'required_features': sorted(required_features),
            'candidate_required': candidate_required,
            'required_all': required_all,
            'required_verified_all': False,
            'missing_required': missing_required,
            'missing_required_views': [
                {'image_id': image,
                 'features': sorted(required_features - features_by_image.get(image, set()))}
                for image in images
                if required_features - features_by_image.get(image, set())],
            'pending_required': pending_candidate,
            'required_locate_pairs': [],
            'missing_locate_pairs': [],
            'missing_propose': [],
            'missing_screen': [],
            'missing_locate': [x['image_id'] for x in rows if not x['located']],
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
        if WorkflowBackend.code_signature()!=self.code_signature:
            raise IntegrationError('CODE_CHANGED','工具/原B源码运行期间变化，禁止混用缓存和阶段产物')
        args=action['args'];tool=action['tool'];targets=self.targets()
        if tool not in TOOLS or args['target_id'] not in targets:raise IntegrationError('ACTION_DENIED','阶段或对象非法')
        target=targets[args['target_id']];key=self.operation_key(tool,target)
        semantic = self.semantic_attempt(tool, target)
        h=acquire(self.operations)
        try:
            doc=read_json(self.operations)
            if key in doc['entries']:return {**doc['entries'][key],'cache_hit':True}
            if tool in self.blocked_tools():
                raise IntegrationError('TOOL_SERVICE_BLOCKED','该工具的模型服务在本任务已失败；不得通过换ROI或版本重发')
            if semantic:
                prior = next((row for row in doc['entries'].values()
                              if row.get('semantic_key') == semantic['semantic_key']), None)
                if prior:
                    return {**prior, 'cache_hit': True, 'semantic_cache_hit': True,
                            'requested_target_id': target['id']}
            if args['evidence_version']!=self.store.current()['version']:
                raise IntegrationError('EVIDENCE_STALE','阶段父版本失效')
            if action['action_id'] not in {a['action_id'] for a in self.propose()}:
                raise IntegrationError('ACTION_DENIED','阶段不在当前合法集合')
            pending_rows = [r for r in doc['entries'].values() if r['status']=='pending']
            if pending_rows:
                from TCM_Fusion.lib.workflow.physical_schedule import allows_single
                if not allows_single(pending_rows,tool,target) or len(pending_rows)>=self.config['max_parallel']:
                    raise IntegrationError('WORKFLOW_BUSY','已有阶段正在执行')
            if tool not in {'measure_instances','detect_region'} and not self.runtime.simulation and not self.allow_network:
                raise IntegrationError('NETWORK_DISABLED','视觉重算须显式允许网络')
            # 类别复核是一次受控视觉观察：只占观察/网络预算，不挤占测量所需的重算额度；
            # 其网络请求由同一worker内的HTTP守卫按请求键记账，这里不重复预约。
            pool=('initialization' if tool=='detect_fov_initial' else
                  'observation' if tool in {'classify_target','image_exemplar_audit'} else 'recompute')
            from TCM_Fusion.lib.runtime.probe_budget import describe
            self.charge(pool,key,network=tool!='classify_target',
                        probe=describe(self.runtime,action,target))
            pending={'action_id':action['action_id'],'tool':tool,'target_id':target['id'],
                     'status':'pending','accepted':False,'repair_completed':False,'operation_key':key}
            if semantic:
                pending.update(semantic)
            if target.get('feature_id') and 'feature_id' not in pending:
                pending.update(feature_id=target['feature_id'], image_id=target.get('image_id'))
            if target.get('fast_recovery'):
                pending.update(feature_id=target['feature_id'], image_id=target['image_id'],
                               scan_tiles_reserved=target['max_tiles'],
                               criteria=target.get('criteria') or [])
            doc['entries'][key]=pending;write_json(self.operations,doc)
            self._write_instance_index(doc)
        finally:release(h)
        self.record_selected({'tool':tool,'target_id':target['id'],'action_id':action.get('action_id')})
        # Windows默认路径长度有限；完整键仍在账本，目录短键冲突时拒绝覆盖。
        work=self.root/'ops'/key[:16]
        try:
            work.mkdir(parents=True,exist_ok=False)
            context = self.worker_context(work, key)
            fingerprint=self.dependency_fingerprint(tool,target)
            before_progress=self.progress_signature()
            result=self.backend.execute(self,tool,target,work,context)
            if self.remaining_seconds()<=0.1 or read_json(self.budget_path).get('closed'):
                raise IntegrationError('WORKFLOW_TIMEOUT','任务已截止，迟到工件只审计不提交')
            from TCM_Fusion.lib.evidence.stage_recording import bundle
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
            self._write_instance_index(doc)
        finally:release(h)
        self.record_finished({'tool':tool,'target_id':target['id'],'action_id':action.get('action_id')}, record.get('status','failed'), record.get('error_code'))
        return record

    def accept(self,tool,target,result,parent,*,receipt=None):
        image=target['image_id'];items=[];rows=[];replace=[]
        if tool=='hybrid_scan':
            detections=result.get('detections') or {}
            feature_scan=result.get('feature_scan') or {}
            yolo_boxes=[]
            def iou(a,b):
                if not a or not b or len(a)!=4 or len(b)!=4:return 0.0
                ax1,ay1,ax2,ay2=map(float,a);bx1,by1,bx2,by2=map(float,b)
                ix1,iy1=max(ax1,bx1),max(ay1,by1);ix2,iy2=min(ax2,bx2),min(ay2,by2)
                inter=max(0.0,ix2-ix1)*max(0.0,iy2-iy1)
                aa=max(0.0,ax2-ax1)*max(0.0,ay2-ay1);bb=max(0.0,bx2-bx1)*max(0.0,by2-by1)
                return inter/(aa+bb-inter+1e-9)
            for kind,bundle in (('detect',detections),('feature',feature_scan)):
                folder=Path(bundle.get('crops_dir') or '')
                if not folder.is_dir():
                    continue
                for m in contracts.load_crop_metas(str(folder)):
                    if file_hash(m.source_image_path)!=self.runtime.images[image]['sha256']:
                        raise IntegrationError('SAMPLE_MISMATCH','混合扫描产物不属于绑定原图')
                    if kind=='detect':
                        yolo_boxes.append((m.feature_id,m.tight_box))
                    elif any(fid==m.feature_id and iou(box,m.tight_box)>=0.5 for fid,box in yolo_boxes):
                        # 同一特征已被 YOLO 框覆盖：保留 YOLO，不重复提交 GLM 同框。
                        continue
                    quality=assess_meta(m)
                    cid='crop:'+image+':'+m.crop_id
                    if kind=='detect':
                        existing=self.store.current()['artifacts'].get(cid)
                        pixels=self.store.current()['artifacts'].get(cid+':pixels')
                        if (existing and pixels and existing.get('valid') and pixels.get('valid')
                                and existing.get('box_source')=='yolo'
                                and existing.get('box_quality_retry') is False
                                and existing.get('sha256')==file_hash(m.json_path)
                                and pixels.get('sha256')==file_hash(m.png)):
                            # 同一 YOLO crop（侧车与像素哈希都未变）已在初检提交：
                            # 复用有效工件，避免仅因 box_quality_retry 缺省/路径变化
                            # 让下游分类失效；真正换框/换像素时仍提交新工件。
                            continue
                    image_item=self._art(cid+':pixels','crop_pixels',m.png,[image],image_id=image)
                    items.append(image_item)
                    metadata=self._crop_metadata(m,quality,kind=kind,target=target)
                    items.append(self._art(cid,'crop',m.json_path,[image,image_item['id']],
                                           image_id=image,**metadata))
                marker=('detected:'+image if kind=='detect' else 'scanned:'+image)
                items.append(self._art(marker,'detection',bundle.get('completion'),[image],
                                       image_id=image))
        elif tool in {'detect_fov_initial','detect_region','locate_feature','feature_scan'}:
            folder=Path(result['crops_dir']);metas=contracts.load_crop_metas(str(folder))
            completion=result['completion'];parents=[image]
            if target.get('fast_recovery'):
                if (read_json(completion).get('feature_id') != target['feature_id']
                        or any(m.feature_id != target['feature_id'] for m in metas)):
                    raise IntegrationError('FEATURE_SCAN_SCOPE', '扫描产物与显式请求的微粒不一致')
            if tool in {'locate_feature','feature_scan'}:
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
                quality=assess_meta(m)
                metadata=self._crop_metadata(m,quality,
                    kind='feature' if tool in {'locate_feature','feature_scan'} else 'detect',
                    target=target)
                items.append(self._art(cid,'crop',m.json_path,[image,image_item['id']],image_id=image,
                                       **metadata))
            marker=('detected:'+image if tool=='detect_fov_initial' else
                    ('scanned:'+image if tool=='feature_scan' else 'detected:'+target['id']))
            if target.get('fast_recovery'):
                marker = 'scanned:' + image + ':' + target['feature_id']
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
                                   **extra))
        elif tool=='image_exemplar_audit':
            root=Path(result['audit']);doc=read_json(root)
            if doc.get('schema') not in {'image-exemplar-audit/v1','image-exemplar-audit/v2'}:
                raise IntegrationError('AUDIT_OUTPUT_INVALID','图片示例审计 schema 不匹配')
            items.append(self._art('image_audit:'+target['id'],'image_audit',root,[target['id']],
                                   image_id=image, target_id=target['id'],
                                   contact_sheet=result.get('contact_sheet'),
                                   review_status='auto_unreviewed', non_blocking=True))
        elif tool=='classify_crops_batch':
            parent=self.store.current()['version'];committed=0
            crops={c['id']:c for c in target.get('crops') or []}
            for item in result.get('classifications') or []:
                crop=crops.get(item.get('crop_id'))
                if not crop:continue
                try:
                    parent=self.accept('classify_target',crop,
                                       {'classification':item['classification']},parent)
                    committed+=1
                except IntegrationError:
                    continue
            if not committed:
                raise IntegrationError('CLASSIFY_EMPTY','批量分类没有可提交结果')
            return parent
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
                                   confidence_score=doc.get('confidence_score'),
                                   review_required=bool(doc.get('reject') == 'pending_review'),
                                   review_reason=doc.get('reason') if doc.get('reject') == 'pending_review' else None))
            meta=contracts.CropMeta.from_json(target['path'])
            # YOLO 初检 crop 的 identity_required=False，但分类结果同样可以是规则证据。
            # 只有“身份门要求”的 crop 才强制 fid==crop.feature_id；普通 YOLO crop 允许
            # 高置信分类改判，否则这些 classification 永远进不了 B 规则。
            if target.get('identity_required') or (fid and not doc.get('reject')):
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
                same_feature = (fid == meta.feature_id)
                items[-1]['identity_confidence']=confidence
                items[-1]['glm_confidence_score']=round(score,4)
                items[-1]['identity_verified'] = bool(
                    fid and not doc.get('reject')
                    and same_feature
                    and (confidence in allowed or confident))
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
                new_crop_id='crop:'+image+':'+new_id
                items.append(self._art(new_crop_id,'crop',p,[target['id'],cid],image_id=image))
                # 改判后的同像素 crop 仍需要自己的 classification，否则规则看不到新 feature。
                items.append(self._art('classification:'+new_crop_id,'classification',
                    root/'classification.json',[new_crop_id],image_id=image,crop_id=new_id,
                    feature_id=fid,label=label,glm_confidence=doc.get('glm_confidence'),
                    yolo_confidence=doc.get('yolo_confidence'),confidence_source=doc.get('confidence_source'),
                    confidence_score=doc.get('confidence_score'),identity_verified=True,
                    identity_basis='reclassified_crop'))
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
                iid=raw.get('instance_id')
                if type(iid) is not int or iid<0:raise IntegrationError('INSTANCE_ID','实例编号非法')
                rows.append({'scope':target['id'],'depends_on':[mid],'data':raw})
            replace=[target['id']]
        else:raise IntegrationError('TOOL_UNKNOWN','未知提交类型')
        if receipt:
            from TCM_Fusion.lib.workflow.physical_schedule import check_live
            version=self.store.commit(parent,items,rows,replace,receipt=receipt,
                                      before_publish=lambda:check_live(self))['version']
        else:
            version=self.store.commit(parent,items,rows,replace)['version']
        if (tool=='segment_instance' and (target.get('measure_if') or self.config.get('auto_measure_after_segment', False))
                and result.get('quality')=='valid'):
            scale=self.store.current()['artifacts'].get('scale:'+image,{})
            if scale.get('valid'):
                seg_target={'id':'seg:'+target['id'],'kind':'segmentation','image_id':image,
                            'masks':str(Path(result['segmentation'])/'masks'),
                            'path':str(Path(result['segmentation'])/'instances.json'),
                            'quality':'valid'}
                try:
                    version=self._auto_measure(seg_target,
                        Path(result['segmentation']).parent/'auto_measure',version)
                except Exception:
                    pass
        return version

    def _classify_attempted(self, target):
        """CNN pending_review 允许首次 GLM；GLM pending/done/failed 不自动重复。"""
        classification = self.store.current()['artifacts'].get('classification:'+target['id'], {})
        if classification.get('valid') and not classification.get('review_required'):
            return True
        semantic = self.semantic_attempt('classify_target', target)
        for row in read_json(self.operations).get('entries', {}).values():
            if row.get('tool') == 'classify_target' and (
                    row.get('target_id') == target['id'] and not row.get('semantic_key') or semantic and
                    row.get('semantic_key') == semantic['semantic_key']) \
                    and row.get('status') in {'done', 'failed', 'pending'}:
                return True
        return False

    @staticmethod
    def _box_iou(a, b):
        if not a or not b or len(a) != 4 or len(b) != 4:
            return 0.0
        ax1, ay1, ax2, ay2 = map(float, a)
        bx1, by1, bx2, by2 = map(float, b)
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0.0, ix2-ix1), max(0.0, iy2-iy1)
        inter = iw*ih
        union = max(0.0, (ax2-ax1)*(ay2-ay1)) + max(0.0, (bx2-bx1)*(by2-by1)) - inter
        return inter/union if union > 0 else 0.0

    @staticmethod
    def _crop_box(meta, target):
        return target.get('source_tight_bbox') or getattr(meta, 'source_tight_bbox', None) or meta.tight_box

    def _segment_reviewed(self, target, review_plan=None):
        """Block a repeated segment of the same semantic region after review/failure.

        A genuinely different crop (low IoU) or an explicit repair decision remains
        legal. This is deliberately semantic: new ROI filenames must not create an
        infinite review -> re-segment loop.
        """
        if review_plan is not None:
            return False
        rows = list(read_json(self.operations).get('entries', {}).values())
        target_meta = contracts.CropMeta.from_json(target['path'])
        target_src = str(target_meta.source_image_path)
        target_feature = target_meta.feature_id
        target_box = self._crop_box(target_meta, target)
        semantic = self.semantic_attempt('segment_instance', target)
        for row in rows:
            if row.get('tool') != 'segment_instance':
                continue
            bundle = row.get('bundle') or {}
            quality = bundle.get('quality')
            if not (row.get('status') == 'failed' or quality == 'review'):
                continue
            if row.get('semantic_key') and semantic:
                if row['semantic_key'] == semantic['semantic_key']:
                    return True
                # D2：高IoU只是关联候选。有效输入变化不能被旧的区域阈值一刀切阻断。
                continue
            root = Path(bundle.get('root') or '')
            matched_semantic = False
            if root.is_dir():
                for crop_json in root.glob('crops/*.json'):
                    try:
                        prior = contracts.CropMeta.from_json(str(crop_json))
                    except (OSError, ValueError, contracts.ContractError):
                        continue
                    if (str(prior.source_image_path) == target_src
                            and prior.feature_id == target_feature):
                        matched_semantic = True
                        if self._box_iou(self._crop_box(prior, {}), target_box) >= 0.85:
                            return True
            # Same target but a materially different crop/ROI is allowed; only
            # no-comparable-crop case remains conservatively blocked.
            if row.get('target_id') == target['id'] and not matched_semantic:
                return True
        return False

    def progress_signature(self, snapshot=None):
        snapshot = self.store.current() if snapshot is None else snapshot
        return digest([(k,v['sha256']) for k,v in sorted(snapshot['artifacts'].items())
                       if v['valid'] and v['kind'] in {'crop','scale','segmentation','measurement'}
                       and v.get('quality','valid')=='valid'])

    def unresolved(self):
        snap=self.store.current();arts=snap['artifacts']
        # 即使网络关闭、动作不被允许，也不能把未处理视野认作已完成。
        unqualified=any(v['valid'] and v['kind']=='segmentation' and v.get('quality')!='valid' and
            (not self.purpose_quality() or not self.assessment(v) or not any(u['allowed'] for u in self.assessment(v)['uses'].values())) for v in arts.values())
        return (not snap['records'] or any(a['tool'] not in {'locate_feature'}
                                           and not a['args']['target_id'].startswith('roi:recovery:')
                                           for a in self.propose(include_offline=True)) or unqualified or
                any(v['status'] in {'pending','failed'} for v in read_json(self.operations)['entries'].values()))
