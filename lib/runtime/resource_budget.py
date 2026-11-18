"""以文件锁保护共享资源账本。调用前原子登记操作键、预算和保留额度，失败及未知计费仍留在账本。受控 worker 接入相同的预留与终态记账接口。

主要定义：
- reserve: 在账本锁内校验样品、任务关闭、截止时间与额度后登记 pending 预约，并为收尾研判保留额度。
- finish_http: 保留第一次 HTTP 结果元数据，不读取或记录响应正文、凭据和 URL 参数。
- close_budget: 在同一事务内关闭任务，并将无终态的 worker HTTP 预约标为结果未知。
- operation_failure: 按 operation_key 查出已失败的预算条目。
- install_http_guard: 仅在专用 worker 内替换 urlopen，拒绝同一请求重放与隐式重定向。

关联：
- 调用方 「lib/runtime/controlled_process.py」、「lib/runtime/worker_bootstrap/sitecustomize.py」、「lib/workflow/workflow_session.py」、「tests/runtime/test_probe_budget.py」、「tests/runtime/test_service_failures.py」、「tests/workflow/test_workflow.py」。
- 直接依赖 「lib/runtime/ledger_lock.py」、「lib/runtime/probe_budget.py」、「lib/runtime/storage.py」。
"""
import time
from TCM_Fusion.lib.runtime.ledger_lock import acquire, release
from TCM_Fusion.lib.runtime.storage import IntegrationError, read_json, write_json, digest


_UNKNOWN_ON_CLOSE = 'WORKFLOW_CLOSED_WITHOUT_HTTP_RESULT'


def reserve(path, pool, key, limits, *, identity=None, metadata=None, extra_pool=None, probe=None):
    handle=acquire(path)
    try:
        value=read_json(path)
        if identity is not None and value['identity']!=identity:
            raise IntegrationError('BUDGET_SCOPE','预算样品不匹配')
        if value.get('closed'):raise IntegrationError('WORKFLOW_CLOSED','任务已结束，禁止新调用或迟到提交')
        if time.time()>value['deadline']:raise IntegrationError('WORKFLOW_TIMEOUT','任务时间预算耗尽')
        reservations = [(pool, key)]
        if extra_pool:
            reservations.append((extra_pool, 'api:' + key))
        if any(k in value['entries'] for _, k in reservations):return False
        if probe is not None:
            from TCM_Fusion.lib.runtime.probe_budget import check_reservation
            code = check_reservation(value, probe)
            if code:raise IntegrationError(code, 'supporting探查重复或额度耗尽')
        held = max(0, limits.get('finalization',0)-value['used'].get('finalization',0))
        if pool != 'finalization' and held:
            network_after = value['used'].get('network',0)+sum(p=='network' for p,_ in reservations)
            if (network_after > limits['network']-held or
                    sum(value['used'].values())+len(reservations) > limits['total']-2*held):
                raise IntegrationError('RESOURCE_BUDGET','保留收尾研判额度；停止追加调查')
        if (any(value['used'].get(p,0)>=limits[p] for p, _ in reservations)
                or sum(value['used'].values()) + len(reservations) > limits['total']):
            raise IntegrationError('RESOURCE_BUDGET','资源预算耗尽')
        for p, k in reservations:
            value['used'][p]=value['used'].get(p,0)+1
            value['entries'][k]={'pool':p,'status':'pending',**(metadata or {})}
        if probe is not None:
            value.setdefault('supporting_probes', {})[probe['key']] = {
                **probe, 'reservation_owner': key, 'pool': pool}
        write_json(path,value)
        return True
    finally:release(handle)


def finish_http(path,key,*,status,http_status=None,error_type=None):
    """保留第一次HTTP结果元数据；不读取或记录响应正文、凭据和URL参数。"""
    handle=acquire(path)
    try:
        value=read_json(path);entry=value['entries'][key]
        if entry['status'] == 'unknown' and entry.get('terminal_reason') == _UNKNOWN_ON_CLOSE:
            # 允许迟到的结果补齐计费审计，任务关闭状态和证据资格由会话控制。
            entry['result_received_after_close'] = True
        elif entry['status'] != 'pending':
            return
        entry.update(status=status,http_status=http_status,error_type=error_type)
        write_json(path,value)
    finally:release(handle)


def close_budget(path, reason):
    """封闭账本并结束悬空 HTTP 预约，保留原扣费、去重键和已知结果。"""
    handle = acquire(path)
    try:
        value = read_json(path)
        for entry in value['entries'].values():
            if entry.get('pool') == 'network' and entry.get('operation_key') and entry.get('status') == 'pending':
                entry.update(status='unknown', terminal_reason=_UNKNOWN_ON_CLOSE)
        value.update(closed=True, stop_reason=reason)
        write_json(path, value)
    finally:
        release(handle)


def operation_failure(context_path):
    ctx=read_json(context_path);value=read_json(ctx['budget_path'])
    return next((r for r in value['entries'].values()
                 if r.get('operation_key')==ctx['operation_key'] and r.get('status')=='failed'),None)


def install_http_guard(context_path):
    """仅在专用worker及其子进程安装，不改变原B或全局用户环境。"""
    import urllib.request
    ctx=read_json(context_path);original=urllib.request.urlopen
    def guarded(request,*args,**kwargs):
        url=request.full_url if hasattr(request,'full_url') else str(request)
        body=getattr(request,'data',None)
        key=digest([ctx['operation_key'],url,body.hex() if body else None])
        if not reserve(ctx['budget_path'],'network',key,ctx['limits'],identity=ctx['identity'],
                       metadata={'operation_key':ctx['operation_key']}):
            raise IntegrationError('NETWORK_REPLAY_DENIED','同一请求已尝试；禁止自动重发未知调用')
        try:
            response=original(request,*args,**kwargs)
        except Exception as exc:
            from urllib.error import HTTPError
            finish_http(ctx['budget_path'],key,status='failed',
                        http_status=exc.code if isinstance(exc,HTTPError) else None,error_type=type(exc).__name__)
            raise
        status=getattr(response,'status',None)
        finish_http(ctx['budget_path'],key,status='response_received',
                    http_status=status if isinstance(status,int) else None)
        return response
    urllib.request.urlopen=guarded
    def no_redirect(*args,**kwargs):
        raise IntegrationError('NETWORK_REDIRECT_DENIED','受控worker禁止隐式重定向')
    urllib.request.HTTPRedirectHandler.redirect_request=no_redirect
