"""D3任务内反思控制器：实际反馈→补丁建议→校验应用→下一动作后验。

由react_graph持有；只写独立策略审计，不写证据、判据或模型权重。
模型在下一次必要规划中可接受/修改/拒绝建议，不为反思新增模型请求。
"""
from .storage import digest, read_json, write_json, IntegrationError
from .tool_registry import validate_arguments


PATCH_SCHEMA = {'type': 'object', 'additionalProperties': False,
    'required': ['reflection_id', 'change_condition', 'avoid_action_keys',
                 'activate_candidates', 'recommended_exit'], 'properties': {
        'reflection_id': {'type': 'string'},
        'change_condition': {'enum': ['different_input', 'different_method', 'different_view', 'switch_question', 'stop']},
        'avoid_action_keys': {'type': 'array', 'maxItems': 4, 'uniqueItems': True, 'items': {'type': 'string'}},
        'activate_candidates': {'type': 'array', 'maxItems': 3, 'uniqueItems': True, 'items': {'type': 'string'}},
        'recommended_exit': {'enum': ['continue', 'review']}}}

RESPONSE_SCHEMA = {'type': 'object', 'additionalProperties': False,
    'required': ['issue_id', 'revision', 'disposition', 'reason'], 'properties': {
        'issue_id': {'type': 'string'}, 'revision': {'type': 'integer', 'minimum': 1},
        'disposition': {'enum': ['apply', 'maintain', 'defer', 'reject']},
        'reason': {'type': 'string', 'minLength': 1, 'maxLength': 240},
        'strategy': {'type': 'object', 'additionalProperties': False,
            'required': ['problem_id', 'change_condition', 'expected', 'stop_when'], 'properties': {
                'problem_id': {'type': 'string'},
                'change_condition': PATCH_SCHEMA['properties']['change_condition'],
                'expected': {'enum': ['capability', 'qualified_evidence', 'resolve_gap', 'stop']},
                'stop_when': {'enum': ['no_progress', 'condition_failed', 'gap_resolved', 'review_now']}}}}}


class ReflectionController:
    def __init__(self, runtime):
        self.runtime = runtime
        self.path = runtime.workflow.root / 'reflection_controller.json'
        self.rows, self.cards, self.applications = [], [], []
        self.rejections = []
        self.pending, self.active = None, None
        self.avoided = set()
        self.issues, self.responses, self.exposures = {}, [], []
        self.goal_failures = []
        self.identity_trigger_at = None
        self.last_strategy = None

    def key(self, action):
        workflow = self.runtime.workflow
        identifier = (action.get('args') or {}).get('target_id')
        target = workflow.targets().get(identifier) if identifier else None
        if target:
            bound = workflow.semantic_attempt(action['tool'], target)
            if bound:
                return bound['semantic_key']
        args = {k: v for k, v in (action.get('args') or {}).items() if k != 'evidence_version'}
        return digest({'tool': action['tool'], 'args': args})

    def save(self):
        write_json(self.path, {'schema': 'task-reflection-controller/d3', 'scope': 'task_policy_only',
            'evidence_effect': 'none', 'rows': self.rows, 'cards': self.cards,
            'applications': self.applications, 'rejections': self.rejections,
            'issues': self.issues, 'responses': self.responses, 'exposures': self.exposures,
            'goal_failures': self.goal_failures,
            'pending': self.pending, 'avoided': sorted(self.avoided)})

    def binding(self, action):
        """在执行前冻结问题及输入身份；分类重定类不能抹掉原调查来源。"""
        args = action.get('args') or {}
        target = self.runtime.workflow.targets().get(args.get('target_id'), {}) if args.get('target_id') else {}
        from .comparison_problems import action_question
        question = action_question(self.runtime, action)
        image = question['image_id']
        from .instance_ledger import describe
        bound = describe(target, self.runtime.workflow.store.current()) if target.get('kind') in {'crop','segmentation'} else None
        input_key = digest([bound['source_image_sha256'], bound['crop_sha256'], bound['geometry']]) if bound else digest(
            self.runtime.images.get(image, {}).get('sha256'))
        return {'question_id':question['question_id'],'image_id':image,'input_fingerprint':input_key}

    def record(self, action, *, progress, feedback, evidence_version, action_key=None, candidate_state=None, binding=None):
        key = action_key or self.key(action)
        frozen = binding or self.binding(action)
        image = frozen['image_id']; input_key = frozen['input_fingerprint']
        axes = feedback.get('progress_axes') or {}
        if not axes:
            # 并行子动作不继承join全局收益；仅从该子动作受信反馈恢复自己的进展。
            accepted = bool(feedback.get('accepted'))
            axes = {'evidence': bool(accepted and any(feedback.get(k) for k in (
                        'new_features', 'new_counterevidence', 'resolved_criteria',
                        'candidate_support_changes', 'qualified_counter_changes'))),
                    'capability': bool(accepted and feedback.get('capability_progress')),
                    'diagnostic': not accepted}
        row = {'key': key, 'tool': action['tool'], 'problem_id': feedback.get('problem_id', key),
               'question_id': frozen['question_id'], 'progress_axes': axes,
               'progress': bool(progress), 'accepted': feedback.get('accepted', False),
               'failure_code': feedback.get('failure_code'), 'latency_ms': feedback.get('latency_ms', 0),
               'objective': feedback.get('objective'), 'resolved_criteria': feedback.get('resolved_criteria', []),
               'candidate_support_changes':feedback.get('candidate_support_changes',{}),
               'qualified_counter_changes':feedback.get('qualified_counter_changes',[]),
               'primary_required_progress':feedback.get('primary_required_progress',False),
               'discrimination_progress':feedback.get('discrimination_progress','not_established'),
               'evidence_version': evidence_version, 'image_id': image, 'input_fingerprint': input_key,
               'candidate_signature': digest([{k:r.get(k) for k in ('id','confidence','support_count','checks')}
                   for r in candidate_state.get('ranking', [])[:2]]) if candidate_state is not None else None}
        if self.active is not None:
            baseline = self.active['baseline']
            condition = self.active['change_condition']
            changed = {'different_input': input_key != baseline.get('input_fingerprint'),
                'different_method': action['tool'] != baseline['tool'],
                'different_view': bool(image and baseline.get('image_id') and image != baseline['image_id']),
                'switch_question': row['question_id'] != baseline.get('question_id'), 'stop': False}[condition]
            assessment = {'reflection_id': self.active['reflection_id'], 'applied': True,
                'next_action_key': key, 'repeated_avoided_key': key in self.active['avoid_action_keys'],
                'change_condition': condition, 'condition_observed': changed,
                'new_information': bool(progress), 'effective': bool(progress and changed),
                'progress_axes': row['progress_axes'],
                'evidence_added': row['progress_axes'].get('evidence', bool(progress)),
                'policy_delta': self.active.get('policy_delta'),
                'recommended_exit': 'continue' if progress and changed else 'review',
                'evidence_version': evidence_version, 'causal_improvement_validated': False}
            keep_goal = False
            self.applications.append(assessment)
            if self.active.get('strategy'):
                strategy = self.active['strategy']
                qualified = bool(row['candidate_support_changes'] or row['qualified_counter_changes'])
                gap_resolved = bool(row['resolved_criteria'])
                capability = bool(row['progress_axes'].get('capability'))
                expected = {'capability': capability, 'qualified_evidence': qualified,
                            'resolve_gap': gap_resolved, 'stop': False}[strategy['expected']]
                assessment.update(strategy=strategy, qualified_fact_added=qualified,
                    goal_gap_resolved=gap_resolved, expected_outcome_observed=expected,
                    effective=bool(changed and expected),
                    evidence_added=qualified, attribution='temporal_followup_not_causal_proof')
                keep_goal = bool(self.active.get('goal_plan_id') and not expected and capability
                                 and not row['failure_code'])
                assessment['posterior_pending'] = keep_goal
            if self.active.get('track_goal_return'):
                self.active['condition_observed'] |= changed
                self.active['qualified_fact_added'] |= qualified
                assessment.update(posterior_pending=True, effective=False)
                keep_goal = True
            if not keep_goal:
                self.active = None
        self.rows.append(row)
        self.save()

    def propose(self, state, threshold=2):
        if self.identity_trigger_at == len(self.rows):
            return self.pending
        if not self.rows:
            return self.pending
        last = self.rows[-1]
        capability_advance = any(r['progress_axes'].get('capability') or r['progress_axes'].get('evidence')
                                 for r in self.rows[-4:])
        stagnant = bool(not capability_advance and len(self.rows) >= 4 and last.get('candidate_signature') and
                        len({r.get('candidate_signature') for r in self.rows[-4:]}) == 1)
        explicit_failure = bool(last['failure_code'] or not last['accepted'])
        if last['progress'] and not stagnant and not explicit_failure:
            had_pending = self.pending is not None
            self.pending = None
            if had_pending:
                self.save()
            return None
        failed = [r for r in self.rows if not r['progress']]
        repeated = sum(r['key'] == last['key'] for r in self.rows) >= 2
        same_problem = sum(r['question_id'] == last['question_id'] for r in failed) >= 2
        if not (explicit_failure or state.get('no_gain', 0) >= threshold or repeated or same_problem or stagnant):
            return self.pending
        error = last['failure_code'] or ''
        kind = ('service' if error.startswith(('UPSTREAM', 'TOOL_DEADLINE')) else
                'segmentation' if last['tool'] == 'segment_instance' else
                'identity' if last['tool'] == 'classify_target' else
                'candidate_stagnation' if stagnant else 'no_gain')
        issue_id = 'issue:' + digest([last['question_id'], kind, error])[:24]
        # 记录数、文件名和候选排名不属于问题的新事实。
        fingerprint = digest([last['input_fingerprint'], last['key'], last['accepted'], error,
                              last['progress_axes'], last['resolved_criteria']])
        return self._offer_issue(issue_id, fingerprint, kind, last, failed,
                                 self.rows[-3:] if stagnant else failed[-3:])

    def record_identity_failure(self, failure, return_reason):
        """目标失败独立入账；分类工具已返回不代表所需身份已通过。"""
        action = {'tool': 'classify_target', 'args': {'target_id': failure['crop_id']}}
        bound = self.binding(action)
        baseline = {**bound, 'key': self.key(action), 'tool': action['tool'],
                    'failure_code': failure['outcome'], 'evidence_version': failure['evidence_version'],
                    'progress_axes': {'evidence': False, 'capability': False, 'diagnostic': True}}
        identity = {**failure, 'return_reason': return_reason}
        issue_id = 'issue:' + digest(['goal_identity', failure['instance_ref'], failure['expected_feature']])[:24]
        # 证据版本或动作计数前进不会单独造成同一个失败重开。
        fingerprint = digest([failure[k] for k in ('bound_input', 'outcome', 'reported_feature')])
        self.goal_failures.append({'issue_id': issue_id, **identity})
        self.identity_trigger_at = len(self.rows)
        card = self._offer_issue(issue_id, fingerprint, 'goal_identity', baseline, [], [], identity=identity)
        self.save()
        return card

    def _offer_issue(self, issue_id, fingerprint, kind, last, failed, facts, *, identity=None):
        issue = self.issues.get(issue_id)
        if issue and issue['fingerprint'] == fingerprint:
            if self.pending and self.pending.get('issue_id') == issue_id:
                return self.pending
            self.pending = None
            self.save()
            return None
        revision = issue['revision'] + 1 if issue else 1
        self.issues[issue_id] = {'revision': revision, 'fingerprint': fingerprint}
        avoid = list(dict.fromkeys(r['key'] for r in failed[-4:]))
        card = {'reflection_id': 'reflection:' + digest([issue_id, revision])[:24], 'scope': 'task',
            'issue_id': issue_id, 'revision': revision,
            'failure_type': kind, 'affected_candidates': self.runtime.candidate_interest()['ids'],
            'evidence_refs': [last['evidence_version']], 'trigger_fingerprint': fingerprint,
            'facts': facts,
            'baseline': last,
            'change_condition': 'different_input' if kind == 'segmentation' else 'switch_question',
            'proposed_patch': {'avoid_action_keys': avoid, 'activate_candidates': [], 'recommended_exit': 'review'},
            'expected_gain': 'resolve a different gap or supply verifiably changed input',
            'interpretation':'候选支持增加不等于排除其他候选；主解释未核验与真正反证分开处理',
            'stop_condition': 'no valid change or repeated no progress within existing budgets',
            'evidence_effect': 'none'}
        card['proposed_patch'].update(reflection_id=card['reflection_id'],
                                      change_condition=card['change_condition'])
        if identity is not None:
            card['identity_failure'] = identity
        self.pending = card
        self.cards.append(card)
        self.save()
        return card

    def respond(self, value, selected, context, *, goal=None):
        """同一次正常规划的可选策略响应；不选动作、不扩权、不发模型请求。"""
        card = self.pending
        if not card and value is None:
            return None
        if isinstance(value, str):
            return self._respond_short(value, selected, context, goal)
        if value is None:
            row = {'issue_id': card['issue_id'], 'revision': card['revision'],
                   'disposition': 'not_acknowledged', 'extra_planning_calls': 0}
        else:
            validate_arguments(RESPONSE_SCHEMA, value)
            if not card or (value['issue_id'], value['revision']) != (card['issue_id'], card['revision']):
                raise IntegrationError('REFLECTION_STALE', '响应须绑定本轮问题及版本')
            row = dict(value, extra_planning_calls=0)
            if value['disposition'] == 'apply':
                strategy = value.get('strategy')
                if not strategy:
                    raise IntegrationError('REFLECTION_STRATEGY_REQUIRED', 'apply须声明可检验策略')
                questions = (((context.get('evidence') or {}).get('workflow') or {}).get(
                    'comparison_problems') or {}).get('questions', [])
                question = next((q for q in questions if q['problem_id'] == strategy['problem_id']), None)
                stopping = selected.get('tool') in {'finish', 'request_review'}
                if strategy['expected'] == 'stop':
                    if (not stopping or strategy['change_condition'] != 'stop' or
                            not question and strategy['problem_id'] != card['issue_id']):
                        raise IntegrationError('REFLECTION_ACTION_CONFLICT', '停止策略须对应本轮停止')
                elif not question or selected['action_id'] not in question.get('legal_action_ids', []):
                    raise IntegrationError('REFLECTION_SCOPE', '策略须引用本轮可见问题及其真实入口')
                # NO_REF参考缺失无法用图像重测修复，但不禁止其他消费者合法测量。
                applicability = 'current_evidence_gap'
                if question and all(c.get('capability_gap') == 'knowledge_reference'
                                    for c in question.get('consumers', [])):
                    applicability = 'knowledge_reference_cannot_be_generated_by_remeasurement'
                delta = {'strategy_target_changed': strategy != self.last_strategy,
                         'changes_current_menu_or_scope': False, 'action_order_effect': 'model_selected_only',
                         'applicability': applicability}
                self.active = {'reflection_id': card['reflection_id'], 'change_condition': strategy['change_condition'],
                    'avoid_action_keys': [], 'recommended_exit': 'review' if stopping else 'continue',
                    'baseline': card['baseline'], 'strategy': strategy, 'policy_delta': delta,
                    'goal_plan_id': goal.get('plan_id') if goal else None}
                self.last_strategy = strategy
                self.applications.append({'reflection_id': card['reflection_id'], 'issue_id': card['issue_id'],
                    'revision': card['revision'], 'applied': True, 'posterior_pending': True,
                    'strategy': strategy, 'policy_delta': delta, 'evidence_effect': 'none'})
            if value['disposition'] in {'apply', 'maintain', 'reject'}:
                self.pending = None
        self.responses.append(row)
        self.save()
        return row

    def _respond_short(self, disposition, selected, context, goal):
        card = self.pending
        offered = (context.get('task_reflection') or {}).get('pending') or {}
        if not card or (offered.get('issue_id'), offered.get('revision')) != (card['issue_id'], card['revision']):
            raise IntegrationError('REFLECTION_STALE', '简短响应须绑定本轮已展示的反思')
        if disposition not in {'apply', 'maintain', 'defer', 'reject'}:
            raise IntegrationError('REFLECTION_SCOPE', '未知反思处置')
        row = {'issue_id': card['issue_id'], 'revision': card['revision'], 'disposition': disposition,
               'selected_action_id': selected['action_id'], 'extra_planning_calls': 0}
        if disposition == 'apply':
            if not any(a['action_id'] == selected['action_id'] for a in context.get('actions', [])):
                raise IntegrationError('REFLECTION_SCOPE', '改策须使用当轮菜单已经展示的动作')
            baseline = card['baseline']
            if selected['tool'] in {'finish', 'request_review'}:
                condition = 'stop'
            elif self.runtime.tools.get(selected['tool']).layer == 'read':
                raise IntegrationError('REFLECTION_NO_TRANSITION', '仅阅读时可维持或推迟策略')
            else:
                actual = self.binding(selected)
                changed = (
                    ('different_view', actual['image_id'] and baseline['image_id'] and
                     actual['image_id'] != baseline['image_id']),
                    ('switch_question', actual['question_id'] != baseline['question_id']),
                    ('different_input', actual['input_fingerprint'] != baseline['input_fingerprint']),
                    ('different_method', selected['tool'] != baseline['tool']),
                )
                condition = next((name for name, yes in changed if yes), None)
                if not condition:
                    raise IntegrationError('REFLECTION_NO_TRANSITION', '当前选择未改变已失败的条件')
            row['transition'] = condition
            delta = {'action_order_effect': 'model_selected_only', 'changes_current_menu_or_scope': False,
                     'declared_action_transition': condition, 'selected_action_id': selected['action_id']}
            strategy = {'change_condition': condition, 'expected': 'stop' if condition == 'stop' else 'qualified_evidence'}
            self.active = {'reflection_id': card['reflection_id'], 'baseline': baseline,
                           'change_condition': condition, 'avoid_action_keys': [], 'strategy': strategy,
                           'recommended_exit': 'review' if condition == 'stop' else 'continue',
                           'policy_delta': delta, 'goal_plan_id': goal.get('plan_id') if goal else None,
                           'track_goal_return': bool(goal), 'condition_observed': False, 'qualified_fact_added': False}
            self.applications.append({'reflection_id': card['reflection_id'], **row, 'applied': True,
                                      'posterior_pending': True, 'policy_delta': delta, 'evidence_effect': 'none'})
        if disposition != 'defer':
            self.pending = None
        self.responses.append(row)
        self.save()
        return row

    def close_goal(self, reason):
        """语义目标回交时关闭未兑现的意图，不归因给下一轮无关动作。"""
        if self.active and self.active.get('goal_plan_id'):
            tracked = self.active.get('track_goal_return', False)
            satisfied = reason == 'local_goal_satisfied'
            qualified = bool(tracked and self.active['qualified_fact_added'])
            condition = bool(tracked and self.active['condition_observed'])
            self.applications.append({'reflection_id': self.active['reflection_id'], 'applied': True,
                'strategy': self.active['strategy'], 'goal_termination': reason, 'posterior_pending': False,
                'condition_observed': condition, 'effective': condition and satisfied and qualified,
                'qualified_fact_added': qualified, 'goal_gap_resolved': tracked and satisfied,
                'expected_outcome_observed': satisfied and qualified, 'policy_delta': self.active['policy_delta'],
                'causal_improvement_validated': False})
            self.active = None
            self.save()

    def offered(self, request_number, context):
        card = (context.get('task_reflection') or {}).get('pending')
        if card:
            self.exposures.append({'request_number': request_number, 'issue_id': card['issue_id'],
                'revision': card['revision'], 'reflection_id': card['reflection_id'],
                'fact_refs': card.get('evidence_refs', []), 'exposure': 'submitted_to_planner'})
            self.save()

    def apply(self, patch, selected=None, *, legal_actions=None):
        validate_arguments(PATCH_SCHEMA, patch)
        card = self.pending
        if not card or patch['reflection_id'] != card['reflection_id']:
            raise IntegrationError('REFLECTION_STALE', '反思补丁未绑定当前反馈')
        if not set(patch['avoid_action_keys']) <= set(card['proposed_patch']['avoid_action_keys']):
            raise IntegrationError('REFLECTION_SCOPE', '补丁不能凭空屏蔽其他动作')
        known = {r['id'] for r in self.runtime.kb.candidates()}
        if not set(patch['activate_candidates']) <= known:
            raise IntegrationError('REFLECTION_SCOPE', '未知候选')
        if selected and self.key(selected) in patch['avoid_action_keys']:
            raise IntegrationError('REFLECTION_ACTION_CONFLICT', '当前动作与所提补丁冲突')
        known_states = {r['candidate_id']: r.get('status') for r in
                        self.runtime.candidate_interest().get('hypotheses', [])}
        new_candidates = [cid for cid in patch['activate_candidates']
                          if known_states.get(cid) != 'investigating']
        newly_avoided = set(patch['avoid_action_keys']) - self.avoided
        available = {self.key(a) for a in legal_actions} if legal_actions is not None else None
        effective_avoid = sorted(newly_avoided if available is None else newly_avoided & available)
        policy_delta = {'new_candidate_activations': new_candidates,
                        'new_legal_action_exclusions': effective_avoid,
                        'legal_menu_checked': available is not None,
                        'changes_current_menu_or_scope': bool(new_candidates or effective_avoid),
                        'exit_recommendation': patch['recommended_exit']}
        self.active = {**patch, 'baseline': card['baseline'], 'policy_delta': policy_delta}
        self.avoided.update(patch['avoid_action_keys'])
        self.pending = None
        self.applications.append({'reflection_id': patch['reflection_id'], 'applied': True,
                                  'posterior_pending': True, 'evidence_effect': 'none', 'patch': patch,
                                  'policy_delta': policy_delta})
        self.save()
        return patch

    def reject(self, patch, error):
        """拒绝原因反馈到下一正常规划；只保存已知ID/计数/摘要，不扩权或重问。"""
        value = patch if isinstance(patch, dict) else {}
        allowed = (self.pending or {}).get('proposed_patch', {}).get('avoid_action_keys', [])
        keys = value.get('avoid_action_keys')
        keys = [k for k in keys if isinstance(k, str)] if isinstance(keys, list) else []
        candidates = value.get('activate_candidates')
        candidates = [c for c in candidates if isinstance(c, str)] if isinstance(candidates, list) else []
        known = {r['id'] for r in self.runtime.kb.candidates()}
        cause = error.__cause__
        path = [part for part in getattr(cause, 'absolute_path', [])
                if isinstance(part, int) or part in PATCH_SCHEMA['properties']][:3]
        validator = getattr(cause, 'validator', None)
        schema_error = {'validator': validator if validator in {
            'required', 'type', 'enum', 'additionalProperties', 'maxItems', 'uniqueItems'} else None,
            'field_path': path,
            'missing_fields': [key for key in PATCH_SCHEMA['required'] if key not in value]}
        row = {'reason': error.code, 'patch_sha256': digest(patch),
               'schema_error': schema_error,
               'matches_current_reflection': bool(self.pending and
                   value.get('reflection_id') == self.pending['reflection_id']),
               'unknown_action_keys_count': len(set(keys) - set(allowed)),
               'unknown_candidates_count': len(set(candidates) - known),
               'selected_action_preserved': True, 'extra_planning_calls': 0,
               'guidance': 'omit optional patch or copy current proposed_patch; avoid keys are exact semantic keys, not action short IDs'}
        self.rejections.append(row)
        self.save()
        return row

    def guard(self, action):
        # 只守护同输入同方法的精确键；不是禁止再次调查整个特征/候选。
        return not self.avoided or self.key(action) not in self.avoided

    def compact(self):
        card = self.pending
        if card:
            baseline = card['baseline']
            summary = {k: card[k] for k in ('reflection_id','issue_id','revision','failure_type','evidence_refs')}
            if 'identity_failure' in card:
                failure = card['identity_failure']
                summary['identity_failure'] = {k: failure[k] for k in (
                    'semantic_task', 'instance_ref', 'outcome', 'expected_feature', 'reported_feature')}
            else:
                summary.update({k: card[k] for k in ('change_condition','proposed_patch','stop_condition','evidence_effect')})
                summary['last_feedback'] = {k: baseline[k] for k in ('tool','question_id','failure_code',
                    'progress_axes','candidate_support_changes','qualified_counter_changes')}
            card = summary
        posterior = self.applications[-1:]
        posterior = [{k:v for k,v in r.items() if k not in {'patch','baseline','next_action_key'}} for r in posterior]
        return {'pending': card, 'last_posterior': posterior,
                'last_response': self.responses[-1:], 'last_rejection': self.rejections[-1:]}

    def finalize(self, reason):
        if self.active is not None:
            self.applications.append({'reflection_id': self.active['reflection_id'],
                'applied': True, 'posterior_pending': False, 'termination': reason,
                'condition_observed': self.active['change_condition'] == 'stop',
                'effective': False, 'evidence_added': False, 'evidence_effect': 'none',
                'exit_recommendation_followed': self.active['recommended_exit'] == 'review'
                    and reason in {'planner_request_review','planner_finish'},
                'declared_exit_condition_consistent': self.active['change_condition'] == 'stop',
                'policy_delta': self.active.get('policy_delta'),
                'causal_improvement_validated': False})
            if self.active.get('strategy'):
                self.applications[-1].update(strategy=self.active['strategy'], qualified_fact_added=False,
                    goal_gap_resolved=False, expected_outcome_observed=(
                        self.active['strategy']['expected'] == 'stop' and reason in {
                            'planner_request_review','planner_finish'}))
            self.active = None
            self.save()
