"""D2 候选区分问题：共享物理查询，分别解释判据，不生成强制调查剧本。

只消费 KB、原规则检查、当前合法动作与资源状态。WorkflowRuntime 汇入 ReAct 上下文；
完整结果由只读工具分页读取。相同特征但形态问题、用途或视野不同不会强行合并。
evidence_paths关联直接/发现入口；prior_summary及investigation_focus保留逐折先验主解释与竞争支持。
它们只描述调查可达性与解释边界，不调度工具、不修改证据资格。

主要定义：
- range_relation。未知参考/不同量纲不能被解释为区间互斥；不改变原B数值规则。
- canonical_presence_hint。通用身份问题来自唯一taxonomy；候选形态留在区分问题，不透传药材名。
- presence_key。所有角色共用的查询身份；实际下发的形态问题不同即不是同一查询。
- action_question。跨工具的物理调查身份；候选是消费者，扫描→crop身份核验不是换问题。
- evidence_paths。只关联本轮真实动作，不虚构未来crop；初检是发现入口而不是特征阳性。
- prior_summary。逐视野逐折首选计数仅供解释先验分歧，不平均概率、不作为独立显微证据。
- investigation_focus。主解释与竞争支持分开显示；弱阳性不能让未调查的主解释从视窗中消失。
- build。返回问题而非工具序列；预计收益只作定性说明，不另设未校准评分权重。
- compact。默认只发少量问题摘要；保留完整候选并集和按需读取入口。

关联：
- 调用方 「lib/agent/planning_choices.py」、「lib/agent/react_graph.py」、「lib/agent/reflection_controller.py」、「lib/decision/candidate_audit.py」、「lib/workflow/evidence_runtime.py」、「lib/workflow/workflow_runtime.py」。
- 直接依赖 「lib/runtime/storage.py」、「lib/vision/feature_profiles.py」。
"""
from TCM_Fusion.lib.runtime.storage import digest, read_json


def range_relation(references):
    """未知参考/不同量纲不能被解释为区间互斥；不改变原B数值规则。"""
    if len(references) < 2 or any(not isinstance(q, dict) for q in references):
        return 'not_comparable'
    if len({q.get('dim') for q in references}) != 1 or any(
            not isinstance(q.get(k), (int, float)) for q in references for k in ('min', 'max')):
        return 'not_comparable'
    return 'overlap' if max(q['min'] for q in references) <= min(q['max'] for q in references) else 'disjoint'


def canonical_presence_hint(runtime, feature_id):
    """通用身份问题来自唯一taxonomy；候选形态留在区分问题，不透传药材名。"""
    cache = getattr(runtime, '_presence_hint_cache', None)
    if cache is None:
        from TCM_Fusion.lib.vision.feature_profiles import default_profiles
        cache = default_profiles(); runtime._presence_hint_cache = cache
    profile = cache.get(feature_id, {})
    tier = runtime.kb.tier(feature_id)
    name = runtime.kb.raw.get('feature_vocab', {}).get(tier, {}).get(feature_id, feature_id)
    return '仅核验这类显微特征是否可见，不判药材或物理尺寸。' + str(profile.get('positive') or name)


def presence_key(runtime, action):
    """所有角色共用的查询身份；实际下发的形态问题不同即不是同一查询。"""
    if action.get('tool') != 'presence_check':
        return None
    args = action.get('args') or {}
    return digest({'source': runtime.images.get(args.get('image_id'), {}).get('sha256'),
                   'feature': args.get('feature_id'), 'question': args.get('morph_hint', '').strip(),
                   'scope': 'bound_fov', 'tool': 'presence_check',
                   'version': runtime.tools.get('presence_check').version,
                   'model': runtime.config_identity, 'kb': runtime.kb_sha256})


def action_question(runtime, action, *, targets=None):
    """跨工具的物理调查身份；候选是消费者，扫描→crop身份核验不是换问题。

    同源同类身份问题共用ID；实例/方法/输入仍由原执行键区分，不改变权限或去重键。
    专门形态提示单列语义范围，不能把不同观察问题合并为一次证据。
    """
    args = action.get('args') or {}
    if targets is None:
        workflow = getattr(runtime, 'workflow', None)
        targets = workflow.targets() if workflow is not None else {}
    target = targets.get(args.get('target_id'), {})
    image = args.get('image_id') or target.get('image_id')
    if not image and args.get('target_id') in runtime.images:
        image = args['target_id']
    feature = _action_feature(action, targets)
    hint = str(args.get('morph_hint') or '').strip()
    variant = hint if hint and (not feature or hint != canonical_presence_hint(runtime, feature)) else 'canonical'
    purpose = ('geometry' if action.get('tool') == 'measure_instances' else
               'identity' if feature else action.get('tool'))
    scope = {'source': runtime.images.get(image, {}).get('sha256'),
             'feature': feature, 'purpose': purpose, 'variant': variant}
    return {'question_id': 'physical-question:' + digest(scope)[:24],
            'feature_id': feature, 'image_id': image, 'purpose': purpose}


def _action_feature(action, targets):
    args = action.get('args') or {}
    if args.get('feature_id'):
        return args['feature_id']
    target = targets.get(args.get('target_id'), {})
    if target.get('feature_id'):
        return target['feature_id']
    crop = target if target.get('kind') == 'crop' else targets.get(target.get('crop_id'), {})
    if crop.get('kind') == 'crop':
        import contracts
        return contracts.CropMeta.from_json(crop['path']).feature_id
    return None


def evidence_paths(runtime, actions, feature_ids):
    """只关联本轮真实动作，不虚构未来crop；初检是发现入口而不是特征阳性。"""
    targets = runtime.workflow.targets()
    paths = {fid: [] for fid in feature_ids}
    for action in actions:
        tool = action['tool']; args = action.get('args') or {}
        fid = _action_feature(action, targets)
        image = args.get('image_id') or targets.get(args.get('target_id'), {}).get('image_id')
        if args.get('target_id') in runtime.images:
            image = args['target_id']
        affected = [fid] if fid in paths else []
        discovery = tool == 'detect_fov_initial' and image in runtime.images
        if discovery:
            affected = list(paths)
        if affected and runtime.tools.get(tool).layer == 'read':
            continue
        for feature in affected:
            pool = ('initialization' if discovery else 'observation' if tool == 'classify_target'
                    else runtime.tools.get(tool).budget_class)
            paths[feature].append({'action_id': action['action_id'], 'tool': tool,
                'image_id': image, 'kind': 'discovery_prerequisite' if discovery else 'feature_step',
                'budget_class': pool, 'is_evidence': False,
                'limit': '可能发现实例；空框不证明不存在；后继逐项校验' if discovery else
                         '只授权本步骤；身份/形态/标定/量纲资格分别核验'})
    return paths


def prior_summary(interest):
    """逐视野逐折首选计数仅供解释先验分歧，不平均概率、不作为独立显微证据。"""
    views = interest.get('views') or [{'image_stem':'single','fold_top_k':interest.get('fold_top_k',[])}]
    counts = {}; unmapped = {}; total = 0
    for view in views:
        for fold in view.get('fold_top_k', []):
            rows = fold.get('ranking', [])
            if not rows:
                continue
            total += 1; row = rows[0]; cid = row.get('candidate_id')
            table, key = (counts, cid) if cid else (unmapped, row.get('label'))
            table[key] = table.get(key, 0) + 1
    maximum = max(counts.values(), default=0)
    return {'a_leads': [cid for cid,n in counts.items() if n == maximum],
            'fold_top1_counts':counts, 'unmapped_fold_top1':unmapped, 'fold_observations':total,
            'meaning':'相关训练折的先验摘要，不是统计独立投票或证据，不强制调查顺序'}


def investigation_focus(runtime, scored, audit):
    """主解释与竞争支持分开显示；弱阳性不能让未调查的主解释从视窗中消失。"""
    prior = prior_summary(runtime.candidate_interest())
    leads = set(prior['a_leads'])
    gaps = {r['candidate_id']:[x['feature_id'] for x in r['required'] if not x['assessed']]
            for r in audit['candidates'] if r['candidate_id'] in leads}
    positives = {r['candidate_id']:[x['feature_id'] for x in r['required'] if x['satisfies_criterion']]
                 for r in audit['candidates'] if r['candidate_id'] not in leads}
    return {**prior, 'lead_required_gaps':gaps,
            'competing_support':{cid:fids for cid,fids in positives.items() if fids},
            'current_rule_lead':next((r['id'] for r in scored['decision'].get('ranking',[])), None),
            'interpretation':'竞争支持不自动成为主解释反证；先验领先者也可被合法反证推翻。',
            'scope':'状态提示；模型可选任意合法调查或review，不要求先查A首选'}


def build(runtime, scored, eligibility, actions):
    """返回问题而非工具序列；预计收益只作定性说明，不另设未校准评分权重。"""
    interest = runtime.candidate_interest()
    active = list(interest.get('ids', []))
    kb = {c['id']: c for c in runtime.kb.candidates()}
    targets = runtime.workflow.targets()
    related = {}
    for action in actions:
        fid = _action_feature(action, targets)
        if fid:
            related.setdefault(fid, []).append(action)
    paths = evidence_paths(runtime, actions,
        {c['feature'] for cid in active if cid in kb for c in kb[cid]['criteria']})
    candidates = {r['id']: r for r in scored['decision'].get('ranking', [])}
    groups = {}; boundaries = []; physical_facts = {}
    for cid in active:
        candidate = kb.get(cid)
        if not candidate or not candidate.get('criteria'):
            boundaries.append({'candidate_id': cid, 'capability': 'kb_incomplete',
                               'repeat_will_help': False, 'legal_action_ids': []})
            continue
        if candidate.get('kind') == 'group':
            boundaries.append({'candidate_id': cid, 'capability': 'kb_indistinguishable',
                               'scope': 'within_group_members_only', 'members': candidate.get('members', []),
                               'reason': runtime.kb.groups[cid].get('reason'),
                               'repeat_will_help': False, 'legal_action_ids': []})
        for criterion in candidate['criteria']:
            fid = criterion['feature']; quant = criterion.get('quant') or {}
            signature = {'feature_id': fid, 'purpose': 'measure' if quant else 'observe',
                         'dimension': quant.get('dim'), 'morphology_question': criterion.get('morph') or ''}
            qid = 'question:' + digest(signature)[:24]
            question = groups.setdefault(qid, {'problem_id': qid, **signature,
                'consumers': [], 'evidence_refs': [], 'legal_action_ids': []})
            check = next((c for c in candidates.get(cid, {}).get('checks', [])
                          if c.get('feature_id') == fid and c.get('role') == criterion.get('role')), {})
            fact_ref = None
            if check.get('instance_refs'):
                physical = {'feature_id': fid, 'instance_refs': check['instance_refs'],
                            'stats': check.get('stats')}
                fact_ref = 'physical:' + digest(physical)[:24]
                physical_facts[fact_ref] = physical
            question['consumers'].append({'candidate_id': cid, 'role': criterion.get('role'),
                'quant': criterion.get('quant'), 'state': check.get('state', 'unknown'),
                'fact_ref': fact_ref, 'q_state': check.get('q_state'),
                'reference_state': 'available' if quant else 'not_provided',
                'capability_gap': 'knowledge_reference' if check.get('state') == 'NO_REF' else None,
                'remeasure_can_supply_reference': False,
                'is_exclusion': False})
            for row in eligibility['projections']:
                if row['candidate_id'] == cid:
                    question['evidence_refs'].extend(g['ref'] for g in row['gaps'] if g['feature_id'] == fid)
    book = read_json(runtime.workflow.budget_path)
    def affordable(action):
        tool=action['tool']; pool=runtime.tools.get(tool).budget_class
        if tool=='classify_target':pool='observation'
        if tool=='detect_fov_initial':pool='initialization'
        return (not book.get('closed') and
                book.get('used',{}).get(pool,0)<runtime.workflow.limits.get(pool,0) and
                book.get('used',{}).get('total',0)<runtime.workflow.limits.get('total',0))
    for question in groups.values():
        relevant = [a for a in related.get(question['feature_id'], [])
                    if a['tool'] != 'presence_check' or
                    (a.get('args') or {}).get('morph_hint', '') == question['morphology_question']]
        entries = paths.get(question['feature_id'], [])
        question['entry_paths'] = entries
        question['legal_action_ids'] = sorted({p['action_id'] for p in entries})
        question['current_states'] = {c['candidate_id']:c['state'] for c in question['consumers']}
        question['required_by'] = [c['candidate_id'] for c in question['consumers'] if c['role']=='required']
        question['supporting_by'] = [c['candidate_id'] for c in question['consumers'] if c['role']!='required']
        question['evidence_refs'] = sorted(set(question['evidence_refs']))
        states = {c['state'] for c in question['consumers']}
        accepted_states = {'IN_RANGE', 'EXTREME', 'OUT_OF_RANGE'}
        if question['purpose'] == 'observe':
            accepted_states.add('NO_REF')
        answered = bool(states) and states <= accepted_states
        if answered:
            capability = 'satisfied_by_current_rule_checks'
        elif book.get('closed') or relevant and not any(affordable(a) for a in relevant):
            capability = 'budget_blocked'
        elif relevant:
            capability = 'supported'
        elif entries:
            entry_ids = {p['action_id'] for p in entries}
            capability = ('reachable_via_prerequisite' if any(affordable(a) for a in actions
                          if a['action_id'] in entry_ids) else 'budget_blocked')
        elif not runtime.images:
            capability = 'insufficient_input'
        elif (question['purpose'] == 'measure' and not runtime.tools.get('measure_instances').default_allow
              or not any(s.default_allow for s in runtime.tools.specs.values() if s.layer != 'read')):
            capability = 'unsupported_tool'
        else:
            capability = 'missing_prerequisite'
        question.update(capability=capability, expected_impact='compare candidate criteria using one physical observation',
            costs=sorted({p['budget_class'] for p in entries}),
            future_requirements=['实例身份','形态/原判据','合格mask与标定/量纲'] if question['purpose']=='measure'
                                else ['当前视野有效观察','候选形态判据单独解释'],
            evidence_limits='特征阳性仅支持相应判据；不自动排除其他药材；后继不存在则回交ReAct',
            may_select_or_review=True, is_evidence=False)
        # 同特征不同形态问题仍分别保留；参考比较跨消费者，缺参考不伪造反证。
        references = [c.get('quant') for cid in active if cid in kb for c in kb[cid]['criteria']
            if c['feature'] == question['feature_id'] and (not c.get('quant') or
                not question['dimension'] or c['quant'].get('dim') == question['dimension'])]
        question['range_relation'] = range_relation(references)
        question['overlapping_ranges'] = question['range_relation'] == 'overlap'
        question['interpretation_note'] = ('overlap cannot by itself distinguish candidates'
            if question['overlapping_ranges'] else 'unobserved or failed is not candidate absence')
    # 通用存在性是一个物理查询；各候选的形态/量纲要求仍保留为独立解释问题。
    shared_queries = []
    for fid, feature_actions in related.items():
        seen = set()
        for action in feature_actions:
            if action['tool'] != 'presence_check':
                continue
            key = presence_key(runtime, action)
            if key in seen:
                continue
            seen.add(key)
            consumers = sorted({c['candidate_id'] for q in groups.values() if q['feature_id']==fid
                                for c in q['consumers']})
            shared_queries.append({'problem_id': 'presence:'+key[:24], 'feature_id': fid,
                'image_id': action['args']['image_id'], 'question_group': 'feature_presence_only',
                'candidate_ids': consumers, 'legal_action_ids': [action['action_id']],
                'does_not_verify': ['candidate_morphology', 'dimensions', 'herb_identity']})
    for label in interest.get('unmapped', []):
        boundaries.append({'candidate_id': label if isinstance(label,str) else label.get('label'),
                           'capability':'kb_incomplete','repeat_will_help':False,'legal_action_ids':[]})
    return {'schema': 'comparison-problems/d2', 'active_candidates': active,
            'prior_summary':prior_summary(interest),
            'a_topk_union': interest.get('a_topk_ids', active),
            'inactive_candidates': interest.get('inactive_ids', []),
            'questions': list(groups.values()), 'shared_queries': shared_queries, 'boundaries': boundaries,
            'physical_facts': physical_facts,
            'score_semantics': {'rule_score': 'b-rule/2，required基础分与独立佐证加分',
                'required_confidence': '候选必需判据达成度，用于计算基础分与佐证加分上限',
                'NO_REF': '缺KB参考，不等于未测量或超范围；重测不能生成参考；不排除消费者',
                'supporting': '候选全部required核验通过后，独立supporting才可加分；已有观察可供其他候选核对'},
            'evidence_version': eligibility['evidence_version'],
            'candidate_reactivation_allowed': True}


def compact(view, limit=8):
    """默认只发少量问题摘要；保留完整候选并集和按需读取入口。"""
    leads = set((view.get('prior_summary') or {}).get('a_leads', []))
    questions = sorted(view['questions'], key=lambda q: (not any(c['candidate_id'] in leads
                       and c['role']=='required' for c in q['consumers']),
                       q['capability'] not in {'supported','reachable_via_prerequisite'},
                       len({c['candidate_id'] for c in q['consumers']}) < 2, q['problem_id']))
    return {'active_candidates': view['active_candidates'], 'a_topk_union': view['a_topk_union'],
            'shared_queries': view['shared_queries'][:limit], 'shared_queries_total': len(view['shared_queries']),
            'questions': [{**{k: q[k] for k in ('problem_id','feature_id','purpose','dimension',
                'capability','overlapping_ranges','range_relation','interpretation_note','costs','legal_action_ids',
                'entry_paths','current_states','required_by','supporting_by','future_requirements','evidence_limits')},
                'consumers':q['consumers'][:8], 'consumers_total':len(q['consumers'])}
                for q in questions[:limit]], 'boundaries': view['boundaries'],
            'physical_facts': {key: fact for key, fact in view.get('physical_facts', {}).items()
                if any(c.get('fact_ref') == key for q in questions[:limit] for c in q['consumers'])},
            'score_semantics': view.get('score_semantics', {}),
            'total': len(questions), 'omitted': max(0, len(questions)-limit),
            'detail_access': 'read_evidence_detail(kind=comparison)'}
