"""子进程共享重算/网络预算；先预约再调用，失败和未知结果不退款不自动重发。"""
import time
from .ledger_lock import acquire, release
from .storage import IntegrationError, read_json, write_json, digest


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
            from .probe_budget import check_reservation
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
        entry.update(status=status,http_status=http_status,error_type=error_type)
        write_json(path,value)
    finally:release(handle)


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
