"""以原图、A 候选与工作流配置创建调查运行时。将工具注册表绑定到 WorkflowSession，管理候选兴趣和假设状态，向 Agent 提供合法动作、证据摘要、候选比较和只读详情。原规则评分由 EvidenceRuntime 提供。

主要定义：
- HYPOTHESIS_UPDATES: 候选假设状态（proposed/investigating/paused/withdrawn）的 JSON Schema。
- WorkflowRuntime: 继承EvidenceRuntime，绑定 WorkflowSession 与工具注册，维护候选兴趣和假设文档。
- from_task: 由任务配置和原图构造运行时，并校验 A 预测与图像绑定一致。
- `_hypothesis_document` 系列：读写 investigation.json、合并假设状态的内部工具。

关联：
- 调用方 「lib/application/workflow_entry.py」、「lib/mcp/mcp_bridge.py」、「lib/offline/workflow_replay.py」、「tests/agent/test_p0_priority.py」、「tests/application/test_lifecycle.py」、「tests/decision/test_candidate_audit.py」。
- 直接依赖 「lib/decision/candidate_audit.py」、「lib/decision/candidate_policy.py」、「lib/decision/comparison_problems.py」、「lib/decision/decision_llm.py」、「lib/evidence/evidence_projection.py」、「lib/evidence/instance_ledger.py」、「lib/evidence/provisional_evidence.py」、「lib/evidence/purpose_evidence.py」。
"""
from pathlib import Path
import contracts
from TCM_Fusion.lib.evidence.provisional_evidence import ADMITTED, TIER
import paths
from TCM_Fusion.lib.workflow.evidence_runtime import EvidenceRuntime
from TCM_Fusion.lib.workflow.workflow_session import WorkflowSession, TOOLS
from TCM_Fusion.lib.tools.tool_registry import ToolSpec, object_schema, validate_arguments
from TCM_Fusion.lib.runtime.storage import (
    IntegrationError, project_path, file_hash, digest, read_json, reusable_reads, write_json,
)


HYPOTHESIS_UPDATES = {'type':'array','minItems':1,'maxItems':3,'items':{
    'type':'object','additionalProperties':False,
    'required':['candidate_id','status','reason'],
    'properties':{'candidate_id':{'type':'string','minLength':1},
                  'status':{'enum':['proposed','investigating','paused','withdrawn']},
                  'reason':{'type':'string','minLength':1,'maxLength':480}}}}


class WorkflowRuntime(EvidenceRuntime):
    def __init__(self, images, *, root, workflow_config, sample_id, a=None, aliases=None,
                 allow_network=False, simulation=False, backend=None, source_run=None):
        self.workflow=None
        self._observed_features=set()
        self.allow_network=allow_network
        self.decision_llm=None
        self._review_decisions={}
        self._admission_decisions={}
        config_file=Path(paths.DETECTION_CONFIG_DIR)/'vlm_config.json'
        identity=file_hash(config_file) if config_file.exists() else 'unconfigured'
        super().__init__([],images,sample_id=sample_id,a=a,aliases=aliases,simulation=simulation,
                         config_identity=identity,candidate_policy=workflow_config.get('candidate_policy',{'top_k':3}))
        self.workflow=WorkflowSession(self,root,workflow_config,allow_network=allow_network,backend=backend)
        if source_run:self.workflow.seed(source_run)
        self.artifact_roots=[self.workflow.root]+([Path(source_run)] if source_run else [])
        # 供工作流判断“观察到存在的特征”与crop类别是否相左；只保存布尔结论，不保存原始观察。
        self.present_features=set()
        self.tools.register_precondition('workflow_target',lambda r,a,s:r.workflow.precondition(a))
        self.tools.register_proposer('workflow_actions',lambda r,s:r.workflow.propose())
        for tool in TOOLS:
            self.tools.register_handler(tool,lambda r,a,s:r.workflow.execute(s['action']))
            self.tools.register(ToolSpec(tool,'recompute','1',object_schema({'target_id':'string','evidence_version':'string'}),
                {'type':'object','required':['accepted','status','action_id'],
                 'properties':{'accepted':{'type':'boolean'},'status':{'enum':['pending','done','failed']},
                               'action_id':{'type':'string'}}},('workflow_repair',),'recompute',True,
                tool,'workflow_target',proposal_id='workflow_actions',
                default_allow=tool in workflow_config.get('enabled_tools',TOOLS)))

    def score(self, observations):
        # 支持性特征只要被 GLM 审查过（无论 present/absent）就记入 observed，
        # 避免 supporting_review 门在同一特征上无限循环。
        if self.workflow is not None:self.records=self.workflow.store.verified_records()
        if self.workflow is not None and self.workflow.purpose_quality():
            from TCM_Fusion.lib.evidence.purpose_evidence import project
            original=self.records;self.records=project(self)
            try:scored=super().score(observations)
            finally:self.records=original
        else:
            scored=super().score(observations)
        self.present_features={fid for fid,c in scored['cards'].items() if c.present}
        self._observed_fovs={(row['image_id'], row['feature_id'])
                             for row in scored.get('observation_states', [])
                             if row['state'] in {'present', 'absent'}}
        self._observed_features={fid for _, fid in self._observed_fovs}
        return scored

    def candidate_interest(self):
        interest = super().candidate_interest()
        workflow = getattr(self, 'workflow', None)
        if workflow is None:
            return interest
        doc = self._hypothesis_document()
        states = doc.get('hypotheses', {})
        inactive = [cid for cid, row in states.items() if row['status'] in {'paused','withdrawn'}]
        nominated = doc.get('ids', [])
        active = [cid for cid in nominated if cid not in inactive]
        ids = [cid for cid in dict.fromkeys(interest['ids']+active) if cid not in inactive]
        return {**interest, 'ids': ids, 'a_topk_ids': interest['ids'],
                'investigation_ids': active, 'inactive_ids': inactive,
                'hypotheses': [{
                    'candidate_id':cid, 'status':states.get(cid,{}).get('status','proposed'),
                    'reason':states.get(cid,{}).get('reason','A候选先验，仅供提出假设'),
                    'is_evidence':False} for cid in dict.fromkeys(interest['ids']+nominated)],
                'investigation_is_evidence': False}

    def _hypothesis_document(self):
        """Read task-bound routing state, including the older additive-only format."""
        workflow = self.workflow
        path = workflow.root/'investigation.json'
        if not path.exists():
            return {'identity':workflow.identity,'kb_sha256':self.kb_sha256,
                    'ids':[],'hypotheses':{},'history':[],'is_evidence':False}
        doc = read_json(path)
        known = {row['id'] for row in self.kb.candidates()}
        if doc.get('identity') != workflow.identity or doc.get('kb_sha256') != self.kb_sha256:
            raise IntegrationError('INVESTIGATION_SCOPE', '调查范围与任务/知识库不符')
        ids = doc.get('ids', [])
        if not isinstance(ids, list) or len(ids) > len(known) or any(not isinstance(x, str) or x not in known for x in ids):
            raise IntegrationError('INVESTIGATION_SCOPE', '调查候选非法')
        states = doc.get('hypotheses', {cid:{'status':'investigating','reason':doc.get('reason','旧调查记录')} for cid in ids})
        if not isinstance(states,dict) or set(states)-set(ids):
            raise IntegrationError('INVESTIGATION_SCOPE','假设状态不匹配')
        for row in states.values():
            if not isinstance(row,dict) or row.get('status') not in {'proposed','investigating','paused','withdrawn'}:
                raise IntegrationError('INVESTIGATION_SCOPE','假设状态非法')
        return {**doc,'hypotheses':states}

    def read_revision(self, state):
        observations = sorted([
            [o.get('action',{}).get('action_id',''), bool(o.get('accepted')), o.get('result',{})]
            for o in state.get('observations',[])], key=lambda row:row[0])
        return digest({'rule':state['evidence_digest'],
                       'artifacts':self.workflow.store.current()['version'],
                       'operations':read_json(self.workflow.operations),
                       'hypotheses':{cid:row['status'] for cid,row in
                                     self._hypothesis_document()['hypotheses'].items()},
                       'observations':observations})

    def investigate(self, ids, reason, evidence_digest):
        """Compatibility for earlier planners; nomination never grants priority."""
        if (not isinstance(ids, list) or not 1 <= len(ids) <= 3 or
                any(not isinstance(x, str) for x in ids) or
                not isinstance(reason, str) or not reason.strip()):
            raise IntegrationError('INVESTIGATION_SCOPE', '扩展候选需要1至3个知识库ID和具体理由')
        self.update_hypotheses([{'candidate_id':cid,'status':'investigating','reason':reason}
                                for cid in ids], evidence_digest)
        return self.candidate_interest().get('investigation_ids', [])

    def update_hypotheses(self, updates, evidence_digest, *, commit=True, decision_source='runtime'):
        """Model-owned routing lifecycle. Never delete observations or re-score truth."""
        validate_arguments(HYPOTHESIS_UPDATES, updates)
        known = {row['id'] for row in self.kb.candidates()}
        ids = [row['candidate_id'] for row in updates]
        if len(set(ids)) != len(ids) or any(cid not in known for cid in ids):
            raise IntegrationError('INVESTIGATION_SCOPE','未知或重复假设ID')
        if any(not row['reason'].strip() for row in updates):
            raise IntegrationError('INVESTIGATION_SCOPE','假设变更必须有理由')
        workflow = self.workflow
        if workflow.remaining_seconds() <= .1 or read_json(workflow.budget_path).get('closed'):
            raise IntegrationError('WORKFLOW_CLOSED','结束后不能变更调查范围')
        doc = self._hypothesis_document()
        combined = list(dict.fromkeys(doc['ids']+ids))
        a_ids = super().candidate_interest()['ids']
        if len(set(combined)-set(a_ids)) > 6:
            raise IntegrationError('INVESTIGATION_SCOPE', '本任务额外调查候选上限为6')
        before = {cid for cid in a_ids+doc['ids']
                  if doc['hypotheses'].get(cid,{}).get('status') not in {'paused','withdrawn'}}
        states = {**doc['hypotheses']}
        for row in updates:
            states[row['candidate_id']]={
                'status':row['status'],'reason':row['reason'],
                'evidence_digest':evidence_digest,'decision_source':decision_source}
        after = {cid for cid in a_ids+combined if states.get(cid,{}).get('status') not in {'paused','withdrawn'}}
        change = {'updates':updates,'evidence_digest':evidence_digest,
                  'scope_changed':before!=after,'is_evidence':False,
                  'decision_source':decision_source,'simulation':self.simulation}
        if commit:
            write_json(workflow.root/'investigation.json', {
                **doc,'schema':'investigation/v2','ids':combined,'hypotheses':states,
                'history':doc.get('history',[])+[change],'is_evidence':False})
        return change

    def _decision_client(self):
        if not getattr(self, 'allow_network', False):
            return None
        workflow = getattr(self, 'workflow', None)
        if workflow is not None and (workflow.remaining_seconds() <= .1 or
                                    read_json(workflow.budget_path).get('closed')):
            return None
        if self.decision_llm is None:
            try:
                from TCM_Fusion.lib.decision.decision_llm import DecisionLLM
                self.decision_llm=DecisionLLM()
                self.decision_llm.workflow=workflow
            except Exception:
                return None
        return self.decision_llm

    def review_decision(self, focus):
        review=focus.get('review') or {}
        key=digest(review)
        if not hasattr(self, '_review_decisions'):
            self._review_decisions={}
        if key not in self._review_decisions:
            llm=self._decision_client()
            if llm is None:
                value={'available':False,'repair':'abandon',
                       'reason':'decision LLM unavailable; default abandon','fallback':True}
            else:
                value=llm.review(review,{'candidate_id':focus.get('candidate_id'),
                                         'feature_id':focus.get('feature_id')})
            self._review_decisions[key]=value
        return self._review_decisions[key]

    def admission_decision(self, decision):
        if decision.get('needs_review') or decision.get('verdict') == 'insufficient_evidence':
            return {'available':False,'decision':'verify','fallback':True,
                    'reason':'rule evidence requires review; no model admission requested',
                    'decision_source':'runtime_precondition'}
        ranking=decision.get('ranking') or []
        top=ranking[0] if ranking else {}
        key=digest({'top':top.get('id'),'checks':top.get('checks')})
        if key in self._admission_decisions:
            return self._admission_decisions[key]
        try:
            from TCM_Fusion.lib.decision.candidate_policy import candidates as a_candidates
            a_ids=list(a_candidates(self.a,self.kb,self.aliases,self.candidate_policy).get('ids') or [])
        except Exception:
            a_ids=[]
        candidate={'id':top.get('id'),'display':top.get('display'),
                   'score':top.get('rule_score',top.get('confidence')),
                   'rule_score':top.get('rule_score',top.get('confidence')),'fused_score':top.get('fused_score'),
                   'confidence':top.get('confidence'),
                   'support_count':top.get('support_count'),'checks':top.get('checks') or []}
        llm=self._decision_client()
        if llm is None:
            checks=candidate['checks']
            specific=[c for c in checks if c.get('present') and c.get('role')=='required'
                      and c.get('feature_id') not in {'epidermis_cell','secretory_tissue'}]
            value={'available':False,
                   'decision':'verify',
                   'reason':'decision LLM unavailable; fail closed for confirmation',
                   'fallback':True}
        else:
            value=llm.admission(candidate,{'a_candidate_ids':a_ids,
                                           'checks':candidate['checks'],
                                           'outside_a':candidate['id'] not in a_ids},a_ids)
        self._admission_decisions[key]=value
        return value

    def p0_focus(self):
        """An unfilled criterion is advisory in v2; it is never an execution duty."""
        focus = WorkflowRuntime._legacy_p0_focus(self)
        if focus and self.workflow.config.get('agent_policy')=='adaptive_v2':
            focus={**focus,'mandatory':False,'advisory_only':True,
                   'directive':'这是未解决的判据缺口，不是强制路线。可核验、暂停、撤销或切换假设；'
                               '若选择尺寸采信，仍须满足真实身份、标定与质量前置条件。'}
        return focus

    def _legacy_p0_focus(self):
        """A 前列候选的 required P0 特征缺口；用于强制先做定向检索。"""
        workflow = self.workflow
        if workflow is None or not workflow.config.get('p0_priority', True):
            return None
        if not workflow.config.get('targeted_recovery', False):
            return None
        from TCM_Fusion.lib.decision.candidate_policy import candidates as a_candidates
        from detect_core import FEATURE_ID
        try:
            interest = (self.candidate_interest() if hasattr(self, 'candidate_interest') else
                        a_candidates(self.a, self.kb, self.aliases, self.candidate_policy))
        except IntegrationError:
            return None
        rows = {row['id']: row for row in self.kb.candidates()}
        snap = workflow.store.current()
        arts = snap.get('artifacts') or {}
        crop_features = set()
        crops = []
        for art_id, art in arts.items():
            if not art.get('valid') or art.get('kind') != 'crop':
                continue
            try:
                meta = contracts.CropMeta.from_json(art['path'])
            except (OSError, ValueError, contracts.ContractError):
                continue
            crop_features.add(meta.feature_id)
            crops.append((art_id, meta, art))
        path = Path(workflow.operations)
        entries = read_json(path).get('entries', {}) if path.is_file() else {}
        locate_count = sum(1 for v in entries.values()
                           if v.get('tool') == 'locate_feature'
                           and v.get('status') in {'done', 'failed'})
        attempted = locate_count >= 1
        present = getattr(self, 'present_features', set()) or set()
        for cid in interest.get('ids') or []:
            row = rows.get(cid)
            if not row:
                continue
            for crit in row.get('criteria') or []:
                if crit.get('role') != 'required':
                    continue
                fid = crit.get('feature')
                if self.kb.tier(fid) != 'P0':
                    continue
                if fid in present:
                    continue
                pending = []
                for art_id, meta, art in crops:
                    if meta.feature_id != fid or not art.get('identity_required'):
                        continue
                    cls = arts.get('classification:' + art_id)
                    if cls and cls.get('valid'):
                        continue
                    if any(op.get('tool') == 'classify_target'
                           and op.get('target_id') == art_id
                           and op.get('status') == 'failed' for op in entries.values()):
                        continue
                    pending.append((float(art.get('det_score') or 0), art_id, meta))
                if pending:
                    score, art_id, meta = max(pending, key=lambda row: (row[0], row[1]))
                    return {'candidate_id': cid, 'candidate_display': row.get('display') or cid,
                            'feature_id': fid, 'feature_zh': next(
                                (label for label, value in FEATURE_ID.items() if value == fid), fid),
                            'tier': 'P0', 'quant': crit.get('quant'), 'morph': crit.get('morph'),
                            'status': 'crop_pending_classification', 'crop_id': meta.crop_id,
                            'crop_artifact_id': art_id, 'det_score': score, 'attempted': False,
                            'directive': '先 classify_target 最高分 P0 crop；未完成前不得继续 locate 或查 P1/C。'}
                verified_crop = False
                for art_id, meta, art in crops:
                    if meta.feature_id != fid:
                        continue
                    cls = arts.get('classification:' + art_id)
                    if not (cls and cls.get('identity_verified')):
                        continue
                    verified_crop = True
                    seg = arts.get('seg:' + art_id)
                    meas = arts.get('measurement:seg:' + art_id)
                    if meas and meas.get('valid'):
                        continue
                    if seg and seg.get('valid') and seg.get('quality') == 'valid':
                        measure_failed = any(
                            op.get('tool') == 'measure_instances'
                            and op.get('target_id') == 'seg:' + art_id
                            and op.get('status') == 'failed' for op in entries.values())
                        if measure_failed:
                            continue
                        return {'candidate_id': cid, 'candidate_display': row.get('display') or cid,
                                'feature_id': fid, 'feature_zh': next(
                                    (label for label, value in FEATURE_ID.items() if value == fid), fid),
                                'tier': 'P0', 'quant': crit.get('quant'), 'morph': crit.get('morph'),
                                'status': 'crop_segmented_unmeasured', 'crop_id': meta.crop_id,
                                'attempted': False,
                                'directive': 'P0 crop 已完成有效分割；先 measure_instances，再查 P1/C。'}
                    if seg and seg.get('valid') and seg.get('quality') != 'valid':
                        measured=arts.get('measurement:seg:'+art_id)
                        if measured and measured.get('valid') and measured.get('diagnostic_review'):
                            continue
                        reviews = getattr(workflow, 'segmentation_reviews', None)
                        if reviews is None:
                            continue
                        review = next((r for r in reviews()
                                       if r.get('crop_id') == meta.crop_id), {})
                        focus={'candidate_id': cid, 'candidate_display': row.get('display') or cid,
                               'feature_id': fid, 'feature_zh': next(
                                   (label for label, value in FEATURE_ID.items() if value == fid), fid),
                               'tier': 'P0', 'quant': crit.get('quant'), 'morph': crit.get('morph'),
                               'status': 'crop_segmentation_review', 'crop_id': meta.crop_id,
                               'segmentation_id': 'seg:'+art_id,
                               'review': review, 'attempted': False,
                               'directive': '先 read_quality 读取 review_reason_codes/n_components/fg_outside_box_ratio；'
                                            '由 planner 判断重新定位、拆分分量、诊断测量或放弃，不得直接跳到别的候选。'}
                        decider=(getattr(self, 'review_decision', None)
                                 if workflow.config.get('agent_policy')!='adaptive_v2' else None)
                        focus['review_decision']=(decider(focus) if decider else
                                                  {'available':False,'repair':'abandon',
                                                   'reason':'review decision unavailable','fallback':True})
                        return focus
                    seg_failed = any(
                        op.get('tool') == 'segment_instance'
                        and op.get('target_id') == art_id
                        and op.get('status') == 'failed' for op in entries.values())
                    if seg_failed:
                        continue
                    return {'candidate_id': cid, 'candidate_display': row.get('display') or cid,
                            'feature_id': fid, 'feature_zh': next(
                                (label for label, value in FEATURE_ID.items() if value == fid), fid),
                            'tier': 'P0', 'quant': crit.get('quant'), 'morph': crit.get('morph'),
                            'status': 'crop_found_unsegmented', 'crop_id': meta.crop_id,
                            'attempted': False,
                            'directive': 'P0 crop 已 identity 通过；先 segment_instance、再 measure_instances，完成前不得查 P1/C。'}
                if verified_crop:
                    continue
                zh = next((label for label, value in FEATURE_ID.items() if value == fid), fid)
                return {'candidate_id': cid, 'candidate_display': row.get('display') or cid,
                        'feature_id': fid, 'feature_zh': zh, 'tier': 'P0',
                        'quant': crit.get('quant'), 'morph': crit.get('morph'),
                        'status': 'no_crop' if fid not in crop_features else 'no_measurement',
                        'attempted': attempted,
                        'directive': '先执行 locate_feature（或分块检测）；完成或失败前不得查 P1/C。'}
        # P0 物理证据完成后，按判据库的 supporting 列表补一轮 GLM 存在性审查。
        # 不把 supporting 写死为 epidermis_cell：哪个候选/哪条判据需要，就查哪条。
        for cid in interest.get('ids') or []:
            row = rows.get(cid)
            if not row:
                continue
            for crit in row.get('criteria') or []:
                if crit.get('role') != 'supporting':
                    continue
                fid = crit.get('feature')
                checked = getattr(self, '_observed_fovs', set())
                observed_all = bool(getattr(self, 'images', {})) and all(
                    (image, fid) in checked for image in self.images)
                if not fid or fid in present or observed_all:
                    continue
                zh = next((label for label, value in FEATURE_ID.items() if value == fid), fid)
                return {'candidate_id': cid, 'candidate_display': row.get('display') or cid,
                        'feature_id': fid, 'feature_zh': zh, 'tier': 'supporting',
                        'quant': crit.get('quant'), 'morph': crit.get('morph'),
                        'status': 'supporting_review', 'attempted': False,
                        'directive': 'P0 物理证据已完成；先对该 supporting 特征做一次 GLM 存在性审查，'
                                     '记录 present/absent 后再结束。'}
        return None

    @reusable_reads()
    def legal_actions(self,state):
        base=super().legal_actions(state)
        if base and base[0]['mandatory'] and self.workflow.config.get('agent_policy') != 'adaptive_v2':return base
        repair=[a for a in base if self.tools.get(a['tool']).budget_class=='recompute']
        rest=[a for a in base if self.tools.get(a['tool']).budget_class!='recompute']
        actions=repair+rest if self.records or self.workflow.diagnostic_quality() or self.workflow.config.get('agent_policy') in {'adaptive_v1','adaptive_v2'} else repair
        if self.workflow.config.get('agent_policy')=='adaptive_v2':
            return actions  # Candidate gaps cannot suppress independent lawful alternatives.
        focus=self.p0_focus()
        if focus and focus.get('status')=='crop_pending_classification' and not focus['attempted']:
            target=focus.get('crop_artifact_id')
            filtered=[a for a in actions
                      if a['tool'] not in {'presence_check','locate_feature'}
                      and (a['tool']!='classify_target'
                           or (a.get('args') or {}).get('target_id')==target)]
            if filtered:return filtered
        if (focus and focus.get('status') in {'crop_found_unsegmented','crop_segmented_unmeasured'}
                and not focus['attempted']):
            filtered=[a for a in actions
                      if a['tool'] not in {'presence_check','locate_feature'}]
            if filtered:return filtered
        if focus and focus.get('status')=='crop_segmentation_review' and not focus['attempted']:
            allowed={'read_quality','read_artifact_status','list_artifacts','query_criteria','list_features'}
            repair=(focus.get('review_decision') or {}).get('repair')
            if repair=='relocate':
                allowed.update({'locate_feature'})
            if repair in {'split_component','diagnostic_measure'}:
                allowed.add('measure_instances')
            filtered=[a for a in actions if a['tool'] in allowed]
            if filtered:return filtered
        if focus and focus.get('status')=='supporting_review' and not focus['attempted']:
            fid=focus.get('feature_id')
            filtered=[a for a in actions if a['tool']=='presence_check'
                      and (a.get('args') or {}).get('feature_id')==fid]
            if filtered:return filtered
        if focus and not focus['attempted']:
            filtered=[a for a in actions if a['tool']!='presence_check']
            if filtered:return filtered
        return actions

    def validate_action(self,action):
        if self.tools.get(action.get('tool')).budget_class=='recompute':
            return self.tools.validate(action,self)
        return super().validate_action(action)

    def valid_classifications(self):
        """Classifications in the current valid evidence snapshot, not history."""
        rows = []
        artifacts = self.workflow.store.current()['artifacts']
        for value in artifacts.values():
            if not value.get('valid') or value.get('kind') != 'classification':
                continue
            try:
                doc = read_json(value['path'])
            except (OSError, ValueError):
                continue
            crop = artifacts.get((value.get('parents') or [''])[0], {})
            alias = crop.get('reclassified_from') or value.get('identity_basis') == 'reclassified_crop'
            visual = doc.get('visual_identity')
            rows.append({'crop_id': value.get('crop_id') if alias else (doc.get('crop_id') or value.get('crop_id')),
                         'classification_id': value['id'], 'image_id': value.get('image_id'),
                         'tool': value.get('classification_tool'),
                         'feature_id': str(doc.get('feature_id') or value.get('feature_id') or ''),
                         'label': doc.get('label'), 'reject': doc.get('reject'),
                         'confidence': str(doc.get('confidence') or value.get('identity_confidence') or '').lower(),
                         'confidence_source': doc.get('confidence_source'),
                         'glm_confidence': doc.get('glm_confidence'),
                         'yolo_confidence': doc.get('yolo_confidence'),
                         'reason': str(doc.get('reason') or ''),
                         'confidence_score': doc.get('confidence_score'),
                         'identity_verified': bool(value.get('identity_verified')),
                         **({'visual_identity': visual} if visual else {})})
        return rows

    def provisional_state(self, decision):
        from TCM_Fusion.lib.evidence.provisional_evidence import build
        valid = self.valid_classifications()
        return build(decision, {'valid_classifications': valid})

    def required_states(self, provisional):
        """Candidate-level required states derived from the same provisional rows."""
        out = []
        gaps = {}
        for row in provisional.get('ranking') or []:
            states = []
            for state in row.get('criterion_states') or []:
                if state.get('role') != 'required':
                    continue
                states.append({'feature_id': state.get('feature_id'),
                               'state': state.get('state'),
                               'provisional_weight': state.get('weight') if state.get('state') == 'VERIFIED_MATCH' else
                                   TIER.get(state.get('state'), 0.0),
                               'instance_ids': state.get('instance_ids') or [],
                               'formal_state': state.get('formal_state')})
                if state.get('state') not in ADMITTED:
                    gaps.setdefault(state.get('feature_id'), []).append(row.get('candidate_id'))
            out.append({'candidate_id': row.get('candidate_id'),
                        'display': row.get('display'),
                        'required_states': states,
                        'pending': [s['feature_id'] for s in states
                                    if s['state'] in {'DETECTED_PENDING', 'PARTIAL_MATCH', 'CONFLICT', 'UNKNOWN'}]})
        discriminating = [{'feature_id': fid, 'candidate_ids': sorted(set(cids))}
                          for fid, cids in gaps.items()]
        return out, discriminating

    def verification_opportunities(self, actions, budget=None, *, decision=None):
        """Advisory only: existing lawful tools for unresolved candidate dimensions.

        This read neither grants tool eligibility nor upgrades formal evidence.
        All candidates in the active scope are considered, not just A top1.
        """
        snapshot = self.workflow.store.current()
        artifacts = snapshot['artifacts']
        targets = self.workflow.targets()
        active = set(self.candidate_interest()['ids'])
        candidates = [row for row in self.kb.candidates() if row['id'] in active]
        pending_required = {row['id'] for row in (decision or {}).get('ranking', [])
                            if (row.get('score_components') or {}).get('required_complete') is False}
        checks = {(row['id'], check['feature_id']): check
                  for row in (decision or {}).get('ranking', [])
                  for check in row.get('checks', [])}
        opportunities = []
        for action in actions:
            tool = action['tool']
            if tool not in {'segment_instance', 'measure_instances'}:
                continue
            target_id = action.get('args', {}).get('target_id')
            target = targets.get(target_id)
            if not target:
                continue
            crop = artifacts.get(target.get('crop_id'), {}) if tool == 'measure_instances' else target
            if not crop.get('valid'):
                continue
            try:
                meta = contracts.CropMeta.from_json(crop['path'])
            except (OSError, ValueError, contracts.ContractError):
                continue
            criteria = []
            for row in candidates:
                for criterion in row['criteria']:
                    if criterion['feature'] != meta.feature_id or not isinstance(criterion.get('quant'), dict):
                        continue
                    check = checks.get((row['id'], meta.feature_id), {})
                    if check.get('present') and check.get('state') in {'IN_RANGE', 'EXTREME', 'OUT_OF_RANGE'}:
                        continue  # an already-resolved dimension does not justify repeat measurement
                    criteria.append({'candidate_id': row['id'], 'role': criterion['role'],
                                     'quant': criterion['quant'], 'morph': criterion.get('morph'),
                                     'state': check.get('state', 'UNKNOWN')})
            if not criteria or all(c['role'] == 'supporting' and c['candidate_id'] in pending_required
                                   for c in criteria):
                continue
            steps = 2 if tool == 'segment_instance' else 1
            network = int(tool == 'segment_instance' and not self.simulation)
            if budget is not None and (budget.get('recompute', 0) < steps
                                      or budget.get('total', 0) < steps + network
                                      or budget.get('network', 0) < network):
                continue
            classification = artifacts.get('classification:' + crop['id'], {})
            identity = bool(classification.get('valid') and classification.get('identity_verified')
                            and classification.get('feature_id') == meta.feature_id)
            opportunities.append({'action_id': action['action_id'], 'tool': tool,
                                  'target_id': target_id, 'feature_id': meta.feature_id,
                                  'image_id': crop['image_id'], 'criteria': criteria,
                                  'identity_verified': identity,
                                  'identity_basis': 'classification' if identity else 'existing_runtime_admission',
                                  'evidence_version': snapshot['version'],
                                  'required_recompute_steps': steps,
                                  'next_tool': 'measure_instances' if tool == 'segment_instance' else None,
                                  'advisory_only': True, 'changes_formal_eligibility': False})
        return sorted(opportunities, key=lambda row: (
            row['tool'] != 'measure_instances',
            not any(c['role'] == 'required' for c in row['criteria']),
            not row['identity_verified'], row['target_id']))

    def supporting_states(self, provisional, observation_states=None):
        """Candidate-level supporting rows; advisory for search breadth, not evidence."""
        checks={}
        for obs in observation_states or []:
            fid=obs.get('feature_id')
            checks.setdefault(fid,[]).append(obs)
        out=[]
        for row in provisional.get('ranking') or []:
            for state in row.get('criterion_states') or []:
                if state.get('role') != 'supporting':
                    continue
                fid=state.get('feature_id');observed=checks.get(fid) or []
                if any(o.get('state')=='present' for o in observed):check_state='checked_present'
                elif any(o.get('state')=='absent' for o in observed):check_state='checked_negative'
                elif observed:check_state='unreadable_or_failed'
                else:check_state='unchecked'
                out.append({'candidate_id': row.get('candidate_id'),
                            'feature_id': fid,
                            'state': state.get('state'),
                            'formal_state': state.get('formal_state'),
                            'instance_ids': state.get('instance_ids') or [],
                            'check_state': check_state,
                            'checks': len(observed),
                            'present': state.get('state') in ADMITTED})
        return out

    def decision_view(self, state, *, scored=None):
        """D2 只读视图：按当前头、观察、候选及规则状态失效，不另请模型审查。"""
        from TCM_Fusion.lib.evidence.evidence_projection import build as eligibility
        from TCM_Fusion.lib.decision.comparison_problems import build as comparison
        from TCM_Fusion.lib.evidence.instance_ledger import project as instances
        snapshot = self.workflow.store.current()
        observations = state.get('observations', [])
        scored = scored or self.score(observations)
        actions = state.get('actions')
        if actions is None:
            # 自主路径的合法集合是本地检查；MCP只读请求也需得到真实的动作可达性。
            actions = self.legal_actions(state) if self.workflow.config.get('agent_policy') == 'adaptive_v2' else []
        operations = read_json(self.workflow.operations)
        key = digest({'head': snapshot['version'], 'observations': observations,
                      'interest': self.candidate_interest(), 'decision': scored['decision'],
                      'operations': operations, 'actions': actions})
        cached = getattr(self, '_decision_view_cache', None)
        if cached and cached[0] == key:
            return cached[1]
        view = eligibility(self, scored, observations)
        result = {'eligibility': view, 'comparison': comparison(self, scored, view, actions),
                  'instances': instances(snapshot, operations)}
        from TCM_Fusion.lib.decision.candidate_audit import build as audit_candidates
        result['candidate_audit']=audit_candidates(self,scored,observations,
                                                  eligibility=view,actions=actions)
        from TCM_Fusion.lib.decision.comparison_problems import investigation_focus
        result['investigation_focus']=investigation_focus(self,scored,result['candidate_audit'])
        self._decision_view_cache = (key, result)
        return result

    @reusable_reads()
    def summary(self,state, *, scored=None):
        scored=scored or self.score(state.get('observations', []))
        result=super().summary(state, scored=scored)
        provisional=self.provisional_state(scored['decision'])
        required, gaps=self.required_states(provisional)
        focus=self.p0_focus()
        result['p0_focus']=focus
        result['hypothesis_catalog']=[{
            'id': row['id'], 'display': row.get('display', row['id']),
            'required': [{'feature': c['feature'], 'tier': self.kb.tier(c['feature'])}
                         for c in row['criteria'] if c['role']=='required']}
            for row in self.kb.candidates()]
        result['workflow']={'evidence_version':self.workflow.store.current()['version'],
                            'resources':read_json(self.workflow.budget_path)['used'],
                            'operations':list(read_json(self.workflow.operations)['entries'].values()),
                            'blocked_tools':self.workflow.blocked_tools(),
                            'p0_focus':focus,
                            'unresolved':self.workflow.unresolved()}
        # Current trusted calibration must not be hidden behind an old read_scale
        # snapshot: calibration can change without changing rule cards or verdicts.
        result['workflow']['calibration']=self.tools.read(self,'read_scale',{},state)
        result['workflow']['quality_assessments']=[self.workflow.assessment(v) for v in
            self.workflow.store.current()['artifacts'].values() if v['valid'] and v['kind']=='segmentation' and v.get('assessment_id')]
        result['workflow']['segmentation_reviews']=self.workflow.segmentation_reviews()
        result['workflow']['coverage']=self.workflow.coverage()
        pending=self.workflow.identity_pending()
        if pending:
            meta=contracts.CropMeta.from_json(pending['path'])
            result['workflow']['identity_pending']={'target_id':pending['id'],
                'feature_id':meta.feature_id,'det_score':pending.get('det_score')}
        else:
            result['workflow']['identity_pending']=None
        # 类别复核结论单独列出：规划器据此决定是否重分割/重检，规则分不受其直接改写。
        result['workflow']['classifications']=[{'crop_id':v.get('crop_id'),'feature_id':v.get('feature_id'),
                                                 'label':v.get('label'), 'reject':v.get('reject'),
                                                 'requested_feature_id':v.get('requested_feature_id'),
                                                 'identity_verified':v.get('identity_verified')}
            for v in self.workflow.store.current()['artifacts'].values()
            if v['valid'] and v['kind']=='classification']
        # 复用既有分类摘要传递质量建议，避免另加付费审查或整份质量历史。
        artifacts = self.workflow.store.current()['artifacts']
        for row in result['workflow']['classifications']:
            crop = artifacts.get(row['crop_id'], {})
            if crop.get('box_quality') in {'blank', 'low_detail'}:
                row['segmentation_admission'] = self.workflow.crop_segmentation_admission(crop)
        result['workflow']['provisional']=provisional
        result['workflow']['required_states']=required
        result['workflow']['supporting_states']=self.supporting_states(provisional, result.get('observation_states'))
        result['workflow']['supporting_gaps']=[row for row in result['workflow']['supporting_states']
                                               if not row['present']]
        result['workflow']['discriminating_gaps']=gaps
        from TCM_Fusion.lib.evidence.evidence_projection import compact as compact_eligibility
        from TCM_Fusion.lib.decision.comparison_problems import compact as compact_comparison
        view = self.decision_view(state, scored=scored)
        from TCM_Fusion.lib.decision.candidate_audit import compact as compact_audit
        result['workflow']['candidate_audit']=compact_audit(view['candidate_audit'])
        result['workflow']['investigation_focus']=view['investigation_focus']
        result['workflow']['eligibility_projection'] = compact_eligibility(view['eligibility'])
        result['workflow']['comparison_problems'] = compact_comparison(view['comparison'])
        associations = view['instances']['associations']
        result['workflow']['instance_associations'] = {'total': len(associations),
            'rows': associations[:8], 'detail_access': 'read_evidence_detail(kind=instances)',
            'qualification_effect': 'none'}
        if self.workflow.config.get('fast_recovery_selection') == 'per_feature':
            result['workflow']['recovery_options']=self.workflow.recovery_options()
            result['workflow']['recovery_policy']=(
                '每个 feature_scan 动作只查 target_id 中显式指定的一类微粒；'
                'required/supporting 共用定位方法，criteria 给出关联候选。'
                '选项不是强制任务；未检出不是阴性，supporting 不替代 required。')
        return result


def from_task(config, a, directory):
    agent=config['agent'];images=[]
    for value in agent['image_roots']:
        p=project_path(value)
        if not p.exists():raise IntegrationError('INPUT_MISSING','指定Agent原图路径不存在')
        images.extend([p] if p.is_file() else sorted(x for x in p.rglob('*') if x.is_file() and x.suffix.lower() in {
            '.png','.jpg','.jpeg','.tif','.tiff','.bmp','.webp'}))
    if not images:raise IntegrationError('AGENT_INPUT_MISSING','工作流Agent需要绑定原图')
    if a.get('view_predictions'):
        bound = {(p.resolve(), file_hash(p)) for p in images}
        predicted = {(project_path(v['image']).resolve(), v.get('input_sha256'))
                     for v in a['view_predictions']}
        if predicted != bound or len(predicted) != len(a['view_predictions']):
            raise IntegrationError('SAMPLE_MISMATCH','多视野A结果与本次Agent原图不一致')
    if a.get('signature',{}).get('image') and a['signature']['image'] not in {file_hash(p) for p in images}:
        raise IntegrationError('SAMPLE_MISMATCH','A不属于本次Agent原图')
    source=(config['b'].get('run_dir') if config['b'].get('mode')=='existing' else
            config['b'].get('input') if config['b'].get('input_mode') in {'crops','masks'} else None)
    if config['b'].get('mode')=='recorded':raise IntegrationError('CONFIG_INVALID','真实workflow禁止录制B冒充部分工件')
    if source and not project_path(source).is_dir():
        raise IntegrationError('B_PARTIAL_MISSING','指定B部分工件目录不存在')
    options={**agent['workflow'],'candidate_policy':agent.get('candidate_policy',{'top_k':3})}
    return WorkflowRuntime(images,root=Path(directory)/'workflow',workflow_config=options,
                           sample_id=config.get('sample_id','sample'),a=a,aliases=config.get('aliases'),
                           allow_network=config.get('allow_network',False),source_run=project_path(source) if source else None)
