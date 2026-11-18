"""读取已完成 case 的输入清单、轨迹、账本和证据版本，先冻结历史基线，
再在临时 EvidenceStore 中执行固定工件的原序与跨视野初检乱序提交。仅支持能够
逐项还原的追加型提交，遇到改写或缺件直接报错。报告写入独立新目录。

主要定义：run 接收正式运行目录列表和新输出目录；baseline 提取观测，
replay 分别验证原始提交版本和改变到达顺序后的最终有效事实。

关联：cli.offline 的 schedule-replay；复用 EvidenceStore 的工件校验、提交和
实例约束，离线对拍不创建规划、网络请求或新的模型裁决。
"""
from collections import Counter
from itertools import combinations
from pathlib import Path
import re
from tempfile import TemporaryDirectory

from TCM_Fusion.lib.evidence.evidence_store import EvidenceStore
from TCM_Fusion.lib.runtime.storage import (
    IntegrationError, digest, fingerprint, read_json, write_json,
)


def require(condition, message):
    if not condition:
        raise IntegrationError('SCHEDULE_REPLAY', message)


def baseline(run):
    source_digest = fingerprint(run.parent.parent)
    job = read_json(run / 'job.json')
    trace = read_json(run / 'agent_trace.json')
    output = read_json(run / 'output.json')
    final = read_json(run / 'final_assessment.json')
    packet = read_json(run / 'final_evidence.json')
    resources = read_json(run / 'workflow/resources.json')
    head = read_json(run / 'workflow/evidence/head.json')['version']
    require(job['status'] == 'completed' and trace['simulation'] is False and
            resources['closed'], '基线需要已经关闭并完成的真实任务')
    require(head == trace['evidence_version'] == packet['evidence_version'], '终态证据版本不一致')
    events = trace['events']
    plans = [e for e in events if e['phase'] == 'planning_timing']
    require(len(plans) == trace['planning_requests'], '规划计时覆盖不完整')
    timeline = read_json(run / 'workflow/action_timeline.json')['actions'].values()
    http = Counter(row['status'] for row in resources['entries'].values()
                   if row['pool'] == 'network' and row.get('operation_key'))
    audit = final['candidate_audit']
    states = {role: dict(Counter(item['state'] for row in audit['candidates']
                                 for item in row[role])) for role in ('required', 'supporting')}
    views = read_json(run.parent.parent / 'input_proof.json')['views']
    return {
        'run': run.name, 'case': run.parent.parent.name, 'source_run': str(run),
        'source_tree_sha256': source_digest,
        'input_views': [{k: view[k] for k in ('view', 'sha256')} for view in views],
        'config_sha256': job['config_sha256'], 'code_sha256': job['code_signature']['current'],
        'semantic_execution': read_json(run / 'fusion.json')['workflow_options']['semantic_execution'],
        'evidence_version': head, 'elapsed_seconds': job['elapsed_seconds'],
        'planning_calls': len(plans), 'planning_seconds': round(sum(e['elapsed_seconds'] for e in plans), 3),
        'parallel_joins': sum(e['phase'] == 'parallel_join' for e in events),
        'observation_requests': trace['observation_requests'], 'worker_http_reservations': dict(http),
        'initial_detection_timings': [{k: row.get(k) for k in ('target_id', 'status', 'offer_to_select_s',
                                                                'execution_s')} for row in timeline
                                     if row['tool'] == 'detect_fov_initial' and row.get('selected_at')],
        'first_discriminating_evidence_seconds': None,
        'first_discriminating_evidence_note': '历史记录缺少绑定合格区分事实的统一时间戳',
        'candidate': (output['top_candidate'] or {}).get('id'), 'status': output['status'],
        'review_required': output['review_required'], 'review_reasons': final['review_reasons'],
        'criterion_states': states, 'unmapped': len(audit['unmapped']),
    }


def history(run, version):
    """读取并校验由冻结头可达的链，拒绝越界工件、摘要不符和不可还原的历史。"""
    chain = []
    seen = set()
    while version is not None:
        require(re.fullmatch(r'[0-9a-f]{64}', version) and version not in seen, '非法证据版本')
        seen.add(version)
        snapshot = read_json(run / 'workflow/evidence/versions' / (version + '.json'))
        content = {k: v for k, v in snapshot.items() if k != 'version'}
        require(version == snapshot['version'] == digest(content), '录制快照摘要不匹配')
        chain.append(snapshot)
        version = snapshot['parent']
    chain.reverse()
    require(chain and not chain[0]['records'] and not chain[0]['artifacts'], '缺少空初始快照')
    return chain


def additions(before, after, case):
    old, new = before['artifacts'], after['artifacts']
    require(old.keys() <= new.keys() and all(old[k] == new[k] for k in old),
            '该录制包含替换或失效传播，追加型回放不适用')
    n = len(before['records'])
    require(after['records'][:n] == before['records'], '测量记录发生替换，不能当作追加')
    items = [a for k, a in new.items() if k not in old]
    for a in items:
        require(a['valid'] and Path(a['path']).resolve().is_relative_to(case),
                '工件失效或位于源 case 之外')
    return items, after['records'][n:]


def replay(run, version, scratch_root):
    chain = history(run, version)
    steps = {}
    for before, after in zip(chain, chain[1:]):
        steps[after['version']] = additions(before, after, run.parent.parent)
    order = list(steps)
    ops = read_json(run / 'workflow/operations.json')['entries'].values()
    targets = {op['evidence_version']: op['target_id'] for op in ops
               if op['tool'] == 'detect_fov_initial' and op['status'] == 'done' and op['accepted']}
    require(targets.keys() <= steps.keys(), '初检提交不在冻结版本链')
    for key, target in targets.items():
        items, records = steps[key]
        ids = {a['id'] for a in items}
        external = {p for a in items for p in a.get('parents', [])} - ids
        require(external == {target} and not records and
                all(a.get('image_id') == target for a in items), '初检包含额外依赖或跨视野写入')

    def execute(sequence, serial):
        with TemporaryDirectory(prefix='replay-', dir=scratch_root) as temporary:
            store = EvidenceStore(Path(temporary), chain[0]['identity'])
            current = store.current()
            for step in sequence:
                items, records = steps[step]
                current = store.commit(current['version'], items, records)
                if serial:
                    require(current['version'] == step, '原序重建未复现录制版本')
            store.verified_records()
            for key in ('identity', 'artifacts', 'records'):
                require(current[key] == chain[-1][key], '乱序后终态不一致：' + key)
            return current['version']

    original = execute(order, True)
    pairs = []
    for a, b in combinations([step for step in order if step in targets], 2):
        if targets[a] == targets[b]:
            continue
        changed = order.copy()
        changed.remove(b)
        changed.insert(changed.index(a), b)
        rebuilt = execute(changed, False)
        pairs.append({'first_arrival': targets[b], 'second_arrival': targets[a],
                      'original_commits': [a, b], 'replayed_head': rebuilt})
    return {'original_head': original, 'original_commits': len(order), 'reordered_pairs': pairs,
            'final_artifacts': len(chain[-1]['artifacts']), 'measurement_rows': len(chain[-1]['records']),
            'final_facts_sha256': digest([chain[-1]['artifacts'], chain[-1]['records']])}


def run(sources, out):
    sources = [Path(p).resolve() for p in sources]
    out = Path(out).resolve()
    require(sources and len(set(sources)) == len(sources), '需要互不重复的实跑目录')
    for source in sources:
        require(source.parent.name == 'runs' and source.is_dir(), '请传入 case/runs 下的正式运行目录')
        require(not out.is_relative_to(source.parent.parent), '禁止写入原 case')
    out.mkdir(parents=True, exist_ok=False)
    baselines = [baseline(source) for source in sources]
    write_json(out / 'baseline.json', {'schema': 'schedule-baseline/v1', 'cases': baselines})
    results = []
    for source, row in zip(sources, baselines):
        result = replay(source, row['evidence_version'], out)
        results.append({'case': row['case'], **result})
    for source, row, result in zip(sources, baselines, results):
        after = fingerprint(source.parent.parent)
        require(after == row['source_tree_sha256'], '回放期间源 case 内容变化')
        result['source_tree_sha256_after'] = after
    write_json(out / 'replay.json', {'schema': 'schedule-replay/v1', 'cases': results,
               'passed': True, 'new_model_calls': 0, 'delivery_recomputed': False,
               'scope': '追加型证据提交；原序逐版本与跨视野初检乱序终态对拍'})
