"""规划/服务错误的最小诊断分类；不记录响应正文或密钥，也不替换真实模型。

主要定义：
- EVIDENCE_REPAIR_ERRORS: 可视为证据修复类错误的代码集合。
- planner_retryable: 只允许只读规划请求在断连时重试一次，不扩展为工具重试。
- stage_error_kind: 把错误码归类为服务/配置/其他阶段类型。
- service_error: 从异常中提取最小错误码与类别，不泄露原始响应。

关联：
- TCM_Fusion.lib.react_graph: 用 planner_retryable/service_error/stage_error_kind 处理规划与工具失败。
- TCM_Fusion.lib.workflow_session: stage_error_kind 标记阶段失败类型。
- TCM_Fusion.lib.final_assessment: 归类终止原因。
- TCM_Fusion.lib.hard_calls: 文本调用失败时复用 service_error。
"""
import json
from http.client import RemoteDisconnected
from urllib.error import HTTPError


EVIDENCE_REPAIR_ERRORS=frozenset({'EMPTY_MASK','MASK_INVALID','MASK_QUALITY_FAILED','MASK_OUT_OF_BOUNDS'})


def planner_retryable(exc):
    """Only a disconnected read-only planner request may be retried once.

    This is not a tool retry policy: an image-service failure stays a service
    error, and auth/credit/configuration failures must not issue another call.
    """
    return (getattr(exc, 'code', None) == 'UPSTREAM_REMOTE_DISCONNECTED'
            or isinstance(exc, RemoteDisconnected))


def stage_error_kind(code):
    """服务/请求失败不由图像重检修复；只有明确的证据质量错误可自动产生修复ROI。"""
    code=code if isinstance(code,str) else ''
    if code.startswith('UPSTREAM_HTTP_') or code in {
        'UPSTREAM_NETWORK_FAILED','UPSTREAM_REMOTE_DISCONNECTED','CONFIG_MISSING','CONFIG_INVALID','CREDENTIALS_MISSING',
        'CREDENTIAL_AMBIGUOUS','CONFIG_MODEL_MISSING','WORKFLOW_TIMEOUT',
        'VLM_CREDIT_EXHAUSTED','VLM_AUTH_FAILED','VLM_REQUEST_REJECTED'}:
        return 'service'
    if code in EVIDENCE_REPAIR_ERRORS:return 'evidence'
    return 'unknown'


def service_error(exc):
    result={'error_type':type(exc).__name__}
    if not isinstance(exc,HTTPError):return result,None
    result['http_status']=exc.code
    try:
        body=json.loads(exc.read(8192).decode('utf-8'))
        error=body.get('error',{}) if isinstance(body,dict) else {}
        code=str(error.get('code','')) if isinstance(error,dict) else ''
        if code.isdigit() and len(code)<=12:result['vendor_code']=code
    except (ValueError,UnicodeError,OSError):pass
    if exc.code==429 and result.get('vendor_code')=='1113':reason='VLM_CREDIT_EXHAUSTED'
    elif exc.code in {401,403}:reason='VLM_AUTH_FAILED'
    elif exc.code in {400,404,422}:reason='VLM_REQUEST_REJECTED'
    else:reason=None
    return result,reason
