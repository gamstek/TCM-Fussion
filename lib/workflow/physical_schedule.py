"""为同一样品的跨视野物理动作组冻结输入、预算和写入范围。worker 只准备私有工件。
协调者逐支校验并发布；关闭、重复回执和发布进程中断均以磁盘账本与证据祖先为准。

主要定义：
- reserve_group 绑定本轮合法动作，prepare 按票据执行一次，publish 处理单支回执。
- fence_locked 在关闭或接管运行时对账已发布版本，隔离其余未提交结果。

关联：
- ReAct 选择动作，LocalToolbox / MCP 调用 prepare，WorkflowSession.accept 校验正式交付契约。
"""
import copy
import time
from contextlib import contextmanager
from TCM_Fusion.lib.evidence.stage_recording import bundle
from TCM_Fusion.lib.runtime.ledger_lock import acquire, release
from TCM_Fusion.lib.runtime.resource_budget import reserve, revoke_operation
from TCM_Fusion.lib.runtime.service_errors import stage_error_kind
from TCM_Fusion.lib.runtime.storage import IntegrationError, digest, read_json, write_json
from TCM_Fusion.lib.workflow.workflow_backend import WorkflowBackend


POOLS = {'detect_fov_initial': 'initialization', 'classify_target': 'observation'}


def epoch_timings(operations):
    """按物理组汇总准备阶段的实测区间。缺时间的分支单独计数。"""
    groups = {}
    for row in operations.values():
        if row.get('epoch'):
            groups.setdefault(row['epoch'], []).append(row)
    summaries = []
    for epoch, rows in groups.items():
        intervals = sorted((r['started_at'], r['returned_at']) for r in rows
                           if isinstance(r.get('started_at'), (int, float))
                           and isinstance(r.get('returned_at'), (int, float))
                           and r['returned_at'] >= r['started_at'])
        union = total = 0.0
        until = None
        for start, end in intervals:
            total += end - start
            union += max(0, end - max(start, until)) if until is not None else end - start
            until = end if until is None else max(until, end)
        summaries.append({'epoch': epoch, 'branches': len(rows), 'timed_branches': len(intervals),
                          'accepted': sum(r.get('accepted') is True for r in rows),
                          'branch_seconds': round(total, 6), 'union_seconds': round(union, 6),
                          'overlapping_branch_seconds': round(total - union, 6)})
    return summaries


@contextmanager
def locked(session):
    handle = acquire(session.operations)
    try:
        yield read_json(session.operations)
    finally:
        release(handle)


def enabled(session):
    return (session.config.get('agent_policy') == 'adaptive_v2'
            and session.config['physical_parallel'] is True
            and 2 <= session.config['max_parallel'] <= 4)


def allows_single(pending, tool, target):
    """慢支只写私有目录时，允许另一视野已选中的单实例工作立即启动。"""
    return (bool(pending) and all(r.get('epoch') for r in pending)
            and tool == 'classify_target'
            and all(r['image_id'] != target.get('image_id') for r in pending))


def _scope(snapshot, image):
    return digest({k:v for k,v in snapshot['artifacts'].items() if v.get('image_id') == image})


def check_live(session):
    budget = read_json(session.budget_path)
    if budget.get('closed') or session.cancel_event.is_set():
        raise IntegrationError('WORKFLOW_CLOSED', '任务已经关闭')
    if time.time() >= budget['deadline']:
        raise IntegrationError('WORKFLOW_TIMEOUT', '任务已截止')


def reserve_pair(session, actions):
    if len(actions) != 2:
        raise IntegrationError('PAIR_DENIED','双动作接口需要两个动作')
    return reserve_group(session,actions)


def reserve_group(session, actions):
    tools = {a['tool'] for a in actions}
    if (not enabled(session) or not 2 <= len(actions) <= session.config['max_parallel']
            or len(tools)!=1 or not tools <= POOLS.keys()):
        raise IntegrationError('PAIR_DENIED', '物理并发宽度或同类阶段约束未满足')
    with locked(session) as book:
        check_live(session)
        session.store.verified_records()
        snap = session.store.current()
        legal = {a['action_id']: a for a in session.propose()}
        targets = session.targets()
        rows = []
        for a in actions:
            canonical = legal.get(a['action_id'])
            if not canonical or any(a.get(k) != canonical[k] for k in ('tool','skill','args')):
                raise IntegrationError('PAIR_STALE', '动作已失效或参数不匹配')
            session.runtime.tools.validate(a, session.runtime)
            tool = a['tool']
            target = targets[a['args']['target_id']]
            image = target['image_id']
            key = session.operation_key(tool,target)
            rows.append({**(session.semantic_attempt(tool,target) or {}),
                'operation_key':key, 'action_id':a['action_id'], 'tool':tool, 'target_id':target['id'],
                'image_id':image, 'write_set':[image], 'scope_sha':_scope(snap,image),
                'plan_revision':snap['version'], 'read_set':session._dependency_rows(tool,target),
                'dependency_fingerprint':session.dependency_fingerprint(tool,target),
                'status':'pending', 'phase':'reserved', 'accepted':False, 'repair_completed':False,
                'selected_at':time.time(), 'target':target})
        instances = [row['semantic_key'] for row in rows if row.get('semantic_key')]
        if len({r['image_id'] for r in rows}) != len(rows) or len(set(instances)) != len(instances):
            raise IntegrationError('PAIR_OVERLAP', '组内动作触及同一视野或同一实例')
        if any(r['status']=='pending' for r in book['entries'].values()):
            raise IntegrationError('WORKFLOW_BUSY','已有预约')
        keys = [r['operation_key'] for r in rows]
        epoch = digest([session.identity, snap['version'], sorted(keys)])
        for row in rows:
            row['epoch'] = epoch
            book['entries'][row['operation_key']] = row
        # 在允许 worker 看见预约前，先持锁持久化意图。中断后只对账，不自动重发。
        write_json(session.operations,book)
        try:
            if not reserve(session.budget_path, POOLS[rows[0]['tool']], keys[0], session.limits,
                           identity=session.identity, additional_keys=keys[1:]):
                raise IntegrationError('RESOURCE_REPLAY','该阶段已预约')
        except Exception:
            for key in keys:
                del book['entries'][key]
            write_json(session.operations,book)
            raise
        return [(key,epoch) for key in keys]


def _row(book, key, epoch):
    row = book['entries'].get(key)
    if not row or row.get('epoch') != epoch or not epoch:
        raise IntegrationError('PAIR_TICKET_INVALID','分支票据无效')
    return row


def _failed(row, code):
    row.update(status='failed', accepted=False, error_code=code, error_kind=stage_error_kind(code),
               ended_at=time.time())


class _BranchCancellation:
    """受控子进程轮询此标志；MCP 进程也能观察到协调者已持久化的分支撤销。"""
    def __init__(self,session,key):
        self.session = session
        self.key = key
        self.next_poll = 0
        self.revoked = False

    def is_set(self):
        if time.monotonic() >= self.next_poll and not self.revoked:
            budget = read_json(self.session.budget_path)
            self.revoked = budget.get('closed') or budget['entries'][self.key].get('revoked',False)
            self.next_poll = time.monotonic()+0.25
        return self.revoked or self.session.cancel_event.is_set()


def _check(session, row):
    check_live(session)
    snap = session.store.current()
    if (_scope(snap,row['image_id']) != row['scope_sha']
            or digest(session._dependency_rows(row['tool'],row['target'])) != digest(row['read_set'])
            or session.dependency_fingerprint(row['tool'],row['target']) != row['dependency_fingerprint']):
        raise IntegrationError('EVIDENCE_STALE','绑定输入或写入视野已改变')
    if WorkflowBackend.code_signature() != session.code_signature:
        raise IntegrationError('CODE_CHANGED','执行代码已改变')
    session.store.verified_records()
    return snap['version']


def prepare(session, key, epoch):
    with locked(session) as book:
        row = _row(book,key,epoch)
        if row['status'] != 'pending' or row['phase'] != 'reserved':
            return {'phase':row['phase'], 'status':row['status']}
        if key not in read_json(session.budget_path)['entries']:
            raise IntegrationError('RESOURCE_REPLAY','缺少物理阶段预约')
        _check(session,row)
        row.update(phase='running', started_at=time.time())
        write_json(session.operations,book)
    work = session.root/'ops'/key[:16]
    try:
        session.record_selected(row)
        work.mkdir(parents=True,exist_ok=False)
        worker = copy.copy(session)
        worker.cancel_event = _BranchCancellation(session,key)
        result = session.backend.execute(worker,row['tool'],row['target'],work,session.worker_context(work,key))
        recorded = bundle(result,work)
        write_json(work/'prepared.json', {'result':result, 'bundle':recorded})
        update = {'phase':'ready','prepared_sha':digest([result,recorded])}
    except Exception as exc:
        update = {'phase':'error','stage_error':getattr(exc,'code','WORKFLOW_STAGE_FAILED')}
    with locked(session) as book:
        row = _row(book,key,epoch)
        if row['status'] == 'pending':
            row.update(**update, returned_at=time.time())
        else:
            row['late_return_at'] = time.time()
        write_json(session.operations,book)
        return {'phase':row['phase'], 'status':row['status']}


def _recover(session,row):
    if row['phase'] != 'publishing':
        return False
    version = session.store.published_version(row['operation_key'],row['plan_revision'])
    if version:
        snapshot = read_json(session.store.root/'versions'/(version+'.json'))
        row.update(status='done', accepted=True, evidence_version=version, phase='published',
                   repair_completed=session.progress_signature(snapshot)!=row['progress_before'], ended_at=time.time())
    return bool(version)


def publish(session,key,epoch,*,transport_error=None):
    with locked(session) as book:
        row = _row(book,key,epoch)
        if row['status'] != 'pending':
            return dict(row)
        if row['phase'] in {'reserved','running'} and not transport_error:
            raise IntegrationError('PAIR_NOT_READY','分支仍在执行，保留原预约')
        if not _recover(session,row):
            try:
                parent = _check(session,row)
                if row['phase'] not in {'ready','publishing'}:
                    raise IntegrationError(row.get('stage_error') or transport_error or 'PAIR_NOT_READY','缺少有效回执')
                work = session.root/'ops'/key[:16]
                payload = read_json(work/'prepared.json')
                result = payload['result']
                recorded = bundle(result,work)
                if recorded != payload['bundle'] or digest([result,recorded]) != row['prepared_sha']:
                    raise IntegrationError('ARTIFACT_CHANGED','私有回执或文件发生变化')
                before = session.progress_signature()
                row.update(phase='publishing', bundle=recorded, progress_before=before)
                write_json(session.operations,book)
                version = session.accept(row['tool'],row['target'],result,parent,receipt=key)
                row.update(status='done',phase='published',accepted=True,ended_at=time.time(),
                           evidence_version=version,repair_completed=session.progress_signature()!=before)
            except Exception as exc:
                if not _recover(session,row):
                    revoke_operation(session.budget_path,key)
                    _failed(row,getattr(exc,'code','WORKFLOW_STAGE_FAILED'))
        write_json(session.operations,book)
        session._write_instance_index(book)
        session.record_finished(row,row['status'],row.get('error_code'))
        return dict(row)


def fence_locked(session,reason):
    book = read_json(session.operations)
    changed = False
    for row in book['entries'].values():
        if row['status'] != 'pending':
            continue
        if row.get('epoch'):
            if not _recover(session,row):
                revoke_operation(session.budget_path,row['operation_key'])
                _failed(row,'BRANCH_REVOKED')
                row.update(phase='revoked', revoke_reason=reason)
        else:
            # 没有提交回执，保留预约和未知远端结果，禁止自动重发。
            _failed(row, 'WORKFLOW_RESULT_UNKNOWN')
            row.update(revoke_reason=reason)
        if not row.get('automatic'):
            session.record_finished(row,row['status'],row.get('error_code'))
        changed = True
    if changed:
        write_json(session.operations,book)
        session._write_instance_index(book)
