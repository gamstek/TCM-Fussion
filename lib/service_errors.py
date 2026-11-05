"""规划服务错误的最小诊断；不记录响应正文或密钥，不替换真实模型。"""
import json
from urllib.error import HTTPError


EVIDENCE_REPAIR_ERRORS=frozenset({'EMPTY_MASK','MASK_INVALID','MASK_QUALITY_FAILED','MASK_OUT_OF_BOUNDS'})


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
