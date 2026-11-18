"""D3送模视图：压缩重复字段并测量实际JSON大小，不改变合法动作或证据资格。

消费react_graph已生成的当前上下文；完整历史仍在账本，详情通过注册只读工具获取。
focus只是显示窗口，五折候选并集和全知识库ID始终保留；不参与工具筛选或decide。

主要定义：
- size。
- profile。
- deduplicate_notes。相同长说明一份正文、多处引用；不压缩工具参数、资格、五折榜或显式详情。
- compact。返回新视图、纯指纹cursor及大小审计；不修改调用方原始对象。

关联：
- 调用方 「lib/agent/react_graph.py」、「tests/agent/test_reflection_strategy.py」、「tests/integration/test_agent_efficiency.py」、「tests/integration/test_d3.py」。
- 直接依赖 「lib/runtime/storage.py」。
"""
import copy
import json
from collections import Counter

from TCM_Fusion.lib.runtime.storage import digest


def size(value):
    return len(json.dumps(value, ensure_ascii=False, separators=(',', ':')).encode('utf-8'))


def profile(value):
    return {'bytes': size(value), 'fields': {k: size(v) for k, v in value.items()},
            'sha256': digest(value), 'unit': 'utf8_json_bytes_not_tokens'}


def deduplicate_notes(value):
    """相同长说明一份正文、多处引用；不压缩工具参数、资格、五折榜或显式详情。"""
    fields = {'interpretation', 'scope', 'limit', 'evidence_limits', 'incremental_value',
              'stop_condition', 'expected_resolution', 'limits', 'guidance', 'note',
              'cost_note', 'interpretation_note'}
    protected = {'args', 'eligibility_projection', 'candidate_interest', 'read_results'}
    counts = Counter()
    def visit(node, replace=False):
        if isinstance(node, list):
            for row in node: visit(row, replace)
        elif isinstance(node, dict):
            for key, item in list(node.items()):
                if key in protected:
                    continue
                if key in fields and isinstance(item, str) and len(item.encode('utf-8')) >= 80:
                    if replace and counts[item] > 1:
                        node.pop(key); node[key + '_note_ref'] = aliases[item]
                    elif not replace:
                        counts[item] += 1
                else:
                    visit(item, replace)
    visit(value)
    aliases = {text: 'N'+str(i) for i,text in enumerate((t for t,n in counts.items() if n>1), 1)}
    visit(value, True)
    if aliases:
        value['context_notes'] = {ref:text for text,ref in aliases.items()}


def compact(context, previous=None):
    """返回新视图、纯指纹cursor及大小审计；不修改调用方原始对象。"""
    before = profile(context)
    out = copy.deepcopy(context)
    evidence = out.get('evidence') or {}
    workflow = evidence.get('workflow') or {}
    mirrors = {field for field in ('supporting_gaps','coverage','identity_pending','p0_focus')
               if field in out and field in workflow and out[field] == workflow[field]}
    interest = evidence.get('candidate_interest') or {}
    focus = []
    sources = (list((workflow.get('investigation_focus') or {}).get('a_leads', [])) +
               [r['candidate_id'] for r in interest.get('hypotheses', [])
                if r.get('status') == 'investigating'] +
               [r['id'] for r in evidence.get('decision', {}).get('ranking', [])
                if r.get('confidence', 0) > 0 or r.get('support_count', 0) > 0] +
               list(interest.get('ids', [])) +
               [r['id'] for r in evidence.get('decision', {}).get('ranking', [])])
    for cid in sources:
        if cid not in focus and len(focus) < 6:
            focus.append(cid)
    evidence.pop('skills', None)  # 顶层已给skills的ID/purpose。
    catalog = evidence.pop('hypothesis_catalog', [])
    if catalog:
        evidence['hypothesis_catalog_index'] = {
            'ids': [r['id'] for r in catalog], 'total': len(catalog),
            'detail_access': 'read_evidence_detail(kind=candidate_catalog) / query_criteria'}
    criteria = evidence.get('criteria_diff', [])
    if criteria:
        evidence['criteria_diff_total'] = len(criteria)
        evidence['criteria_diff'] = [r for r in criteria if r.get('candidate_id') in focus]
    provisional = workflow.get('provisional')
    if isinstance(provisional, dict):
        ranking = provisional.get('ranking', [])
        provisional['ranking'] = [r for r in ranking if r.get('candidate_id') in focus]
        provisional.update(total=len(ranking), omitted=len(ranking)-len(provisional['ranking']),
                           detail_access='read_evidence_detail(kind=provisional)')
    for field in ('required_states', 'supporting_states', 'supporting_gaps', 'discriminating_gaps'):
        rows = workflow.get(field)
        if isinstance(rows, list):
            workflow[field] = [r for r in rows if not r.get('candidate_id') or r['candidate_id'] in focus]
            workflow[field + '_total'] = len(rows)
    # 同一判据状态只传一份；完整provisional/当前详情仍可按需读取。
    canonical = {}
    for candidate in (workflow.get('provisional') or {}).get('ranking', []):
        for state in candidate.pop('criterion_states', []):
            row = {k: v for k, v in state.items() if k in {
                'feature_id','role','tier','state','formal_state','instance_ids','quant_note','pending'}}
            row['candidate_id'] = candidate['candidate_id']
            canonical[(row['candidate_id'], row['feature_id'])] = row
        candidate.pop('pending_verification', None)  # canonical.state/pending/instance_ids保留。
    for state in workflow.get('supporting_states', []):
        key = (state['candidate_id'], state['feature_id'])
        row = canonical.setdefault(key, {'candidate_id':key[0], 'feature_id':key[1], 'role':'supporting'})
        row.update({k: v for k, v in state.items() if k not in {'candidate_id','feature_id'}})
    if canonical:
        workflow['criterion_states'] = list(canonical.values())
        for field in ('supporting_states', 'supporting_gaps'):
            workflow[field] = [{k: r[k] for k in ('candidate_id','feature_id')} for r in workflow.get(field, [])]
        workflow['state_detail_access'] = 'criterion_states; full: read_evidence_detail(kind=provisional|comparison)'
    # 顶层与workflow的镜像不再随每轮重发。
    for field in ('supporting_gaps', 'coverage', 'identity_pending', 'p0_focus'):
        if field in mirrors:
            out.pop(field)
    if provisional:
        for field in ('top_candidate','formal_top_candidate'):
            value = provisional.get(field)
            if isinstance(value, dict):
                provisional[field] = {k:value[k] for k in ('candidate_id','display','provisional_score','formal_score') if k in value}
    evidence['workflow'] = workflow
    evidence['candidate_window'] = {'ids': focus, 'selection_effect': 'none',
        'all_a_candidates': interest.get('a_topk_ids', interest.get('ids', [])),
        'reactivation_allowed': True, 'detail_access': 'query_criteria / read_evidence_detail'}
    for field in ('actions', 'parallel_observations'):
        out[field] = [{k: a[k] for k in ('action_id', 'tool', 'args', 'mandatory',
                                       'target_id', 'feature_id', 'image_id') if k in a}
                      for a in out.get(field, [])]
    objectives = out.get('action_objectives', {})
    if objectives:
        first = next(iter(objectives.values()))
        out['objective_contract'] = {k: first[k] for k in ('expected_resolution', 'stop_condition') if k in first}
        out['action_objectives'] = {key: {k: row[k] for k in ('problem_id', 'objective', 'question_id',
                                      'incremental_value', 'recent_latency_seconds') if k in row}
                                    for key, row in objectives.items()}
    for row in out.get('previous_observations', []):
        action = row.get('action') or {}
        row['action'] = {k: action[k] for k in ('action_id', 'tool', 'args') if k in action}
    for row in out.get('shared_observations', []):
        # 只去掉索引中重复保存的原始响应；state、scope和来源键保留。
        if isinstance(row, dict):
            row.pop('raw_response', None)
    # 一个问题的当前状态来自consumers；相同保护契约不逐问题重复。
    comparison = workflow.get('comparison_problems') or {}
    contracts = {}
    path_details = {}
    for question in comparison.get('questions', []):
        if question.get('current_states') == {c['candidate_id']:c.get('state') for c in question.get('consumers', [])}:
            question.pop('current_states')
        purpose = question.get('purpose')
        if 'future_requirements' in question:
            contracts[purpose] = question.pop('future_requirements')
            question['requirements_ref'] = purpose
        paths = question.get('entry_paths', [])
        for entry in paths:
            path_details[entry['action_id']] = entry
        question['entry_paths'] = [{k:entry[k] for k in ('action_id','tool','kind') if k in entry}
                                   for entry in paths]
        # 角色、状态、量程和事实引用保留；固定保护只说明一次。
        question.pop('required_by', None)
        question.pop('supporting_by', None)
        for consumer in question.get('consumers', []):
            for key in ('fact_ref', 'capability_gap'):
                if consumer.get(key) is None: consumer.pop(key, None)
            consumer.pop('remeasure_can_supply_reference', None)
            consumer.pop('is_exclusion', None)
    if contracts:
        comparison['requirements_by_purpose'] = contracts
    if path_details:
        comparison['entry_path_details'] = list(path_details.values())
    if comparison:
        comparison['interpretation_contract'] = (
            'consumers携带各候选角色/判据；物理事实由fact_ref共享。所有入口仅授权当前步骤，'
            '不授阳性或后继资格；NO_REF不能靠重测补参考；局部未见/unknown均不排除标本。')
    deduplicate_notes(out)
    evidence_keys = ('decision', 'observation_states', 'cards', 'candidate_interest')
    cursor = {k: digest(evidence.get(k)) for k in evidence_keys}
    cursor['evidence_version'] = workflow.get('evidence_version')
    out['round_delta'] = {'changed': [k for k in cursor if (previous or {}).get(k) != cursor[k]],
                          'baseline_available': bool(previous), 'current_state_is_authoritative': True}
    # 历史大型只读结果只发摘要；完整当前页仍可通过同一详情工具重读。
    for row in out.get('read_results', []):
        value = row.get('result')
        if row.get('stale') and isinstance(value, dict) and size(value) > 6000:
            row['result'] = {k: value[k] for k in ('kind', 'offset', 'limit', 'total', 'evidence_version') if k in value}
            row['result'].update(summary_only=True, sha256=digest(value),
                detail_access='repeat the same registered read; stale results are not current evidence')
    return out, cursor, {'before': before, 'after': profile(out), 'focus_ids': focus,
                         'workflow_fields': {k:size(v) for k,v in workflow.items()},
                         'changes_tools_or_qualification': False}
