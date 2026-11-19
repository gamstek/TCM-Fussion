"""为视觉编码与模型请求提供可终止的子进程执行通道，临时载荷用完即删。

请求通过独立 worker 子进程完成，使用文件 IPC 且不继承 MCP 协议管道；
超时即整树终止，失败不自动重试（尤其不重发付费请求），响应正文不保留。

主要定义：
- encode_image: 经 worker 子进程把本地图像编码为模型可用的 data URL。
- post: 经 worker 子进程发送模型请求，返回响应正文或抛出结构化上游错误。
- worker: worker 子进程入口，负责图像编码或转发模型 HTTP 请求并写结果文件。
- _run_worker: 私有辅助，负责建临时目录、起子进程、按截止时间终止并读取结果。

关联：
- 调用方 「cli/api_worker.py」、「lib/agent/react_graph.py」、「lib/decision/decision_llm.py」、「lib/decision/final_assessment.py」、「lib/tools/agent_tools.py」、「tests/integration/test_adaptive_quality.py」。
- 直接依赖 「lib/runtime/controlled_process.py」、「lib/runtime/service_errors.py」、「lib/runtime/storage.py」。
"""
import base64
import os
import sys
import io
import json
import tempfile
import subprocess
import time
from pathlib import Path
from http.client import RemoteDisconnected
from urllib.error import HTTPError, URLError
import paths
from TCM_Fusion.lib.runtime.storage import output_root,write_json,read_json,IntegrationError
from TCM_Fusion.lib.runtime.controlled_process import kill_tree


def _run_worker(request, seconds, timeout_code, *, diagnostics=None):
    if seconds<=0:raise IntegrationError('WORKFLOW_TIMEOUT','提交前时间预算耗尽')
    root=output_root()/'request_work';root.mkdir(parents=True,exist_ok=True)
    started = time.monotonic()
    with tempfile.TemporaryDirectory(dir=root,prefix='request_') as temp:
        p=Path(temp);write_json(p/'input.json',request)
        command=[sys.executable,'-X','utf8','-m','TCM_Fusion.cli.api_worker','--input',str(p/'input.json'),'--output',str(p/'output.json')]
        # Worker uses files for IPC and must not inherit the MCP protocol pipe.
        child=subprocess.Popen(command,cwd=paths.PROJECT_ROOT,env=paths.python_environment(),stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,
            start_new_session=os.name!='nt',creationflags=(subprocess.CREATE_NEW_PROCESS_GROUP|subprocess.CREATE_NO_WINDOW) if os.name=='nt' else 0)
        try:
            try:child.wait(timeout=seconds)
            except BaseException as exc:
                kill_tree(child)
                if isinstance(exc,subprocess.TimeoutExpired):
                    raise IntegrationError(timeout_code, '受控子进程超过截止时间，已终止') from exc
                raise
        finally:
            headers = p/'http_headers.json'
            if diagnostics is not None and headers.is_file():
                diagnostics.update(read_json(headers))
            if diagnostics is not None:
                diagnostics['worker_elapsed_seconds'] = round(time.monotonic()-started,6)
        if child.returncode or not (p/'output.json').is_file():raise IntegrationError('API_WORKER_FAILED','请求worker未产生合法结果')
        return read_json(p/'output.json')


def encode_image(path, seconds):
    result = _run_worker({'image_path': str(path)}, seconds, 'IMAGE_ENCODE_TIMEOUT')
    if 'image_url' not in result:
        raise IntegrationError('IMAGE_ENCODE_FAILED', '图像编码子进程未返回可用载荷')
    return result['image_url']


def post(payload,seconds, *, diagnostics=None):
    result = _run_worker({'payload': payload, 'timeout': seconds}, seconds, 'WORKFLOW_TIMEOUT',
                         diagnostics=diagnostics)
    if diagnostics is not None:
        for key in ('http_status', 'vendor_code', 'error_type', 'transport_error_type', 'errno', 'winerror'):
            if key in result:
                diagnostics[key] = result[key]
        if 'http_timing' in result:
            diagnostics['http_timing'] = result['http_timing']
    if 'body' in result:return result['body']
    if result.get('http_status', 0) >= 400:
        code=result.get('vendor_code','')
        raise HTTPError('',result['http_status'],'upstream request failed',{},io.BytesIO(json.dumps({'error':{'code':code}}).encode()))
    raise IntegrationError(result.get('error_code','UPSTREAM_NETWORK_FAILED'),'模型调用失败；不保留响应正文')


def _observation_image_url(path):
    """用 Pillow 准备观察载荷，避开 cv2/NumPy 的原生模块加载路径。"""
    from PIL import Image, ImageOps
    from agent_loop import IMAGE_MAX_SIDE

    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source)
        longest = max(image.size)
        mime = {'JPEG': 'image/jpeg', 'PNG': 'image/png'}.get(source.format)
        if mime and longest <= IMAGE_MAX_SIDE:
            return f'data:{mime};base64,' + base64.b64encode(Path(path).read_bytes()).decode('ascii')
        image = image.convert('RGB')
        if longest > IMAGE_MAX_SIDE:
            scale = IMAGE_MAX_SIDE / longest
            size = tuple(max(1, int(length * scale)) for length in image.size)
            image = image.resize(size, Image.Resampling.BOX)
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG', quality=92)
        return 'data:image/jpeg;base64,' + base64.b64encode(buffer.getvalue()).decode('ascii')


def worker(input_path,output_path):
    request = read_json(input_path)
    if 'image_path' in request:
        try:
            result = {'image_url': _observation_image_url(request['image_path'])}
        except Exception:
            result = {'error_code': 'IMAGE_ENCODE_FAILED'}
        write_json(output_path, result)
        return
    from adjudicator import AdjudicatorClient
    from TCM_Fusion.lib.runtime.service_errors import service_error
    class WorkerClient(AdjudicatorClient):
        def _http_headers_received(self, status):
            super()._http_headers_received(status)
            self.headers_seconds = time.monotonic()-http_started
            write_json(Path(output_path).with_name('http_headers.json'),
                       {'stage': 'http_body_read', 'http_status': status,
                        'send_to_headers_seconds':round(self.headers_seconds,6)})

    client=WorkerClient(Path(paths.DETECTION_CONFIG_DIR)/'vlm_config.json')
    client.timeout_s=min(client.timeout_s,request['timeout'])
    http_started = time.monotonic()
    try:result={'body':client._post(request['payload'])}
    except Exception as exc:
        result,_=service_error(exc)
        cause = exc.reason if isinstance(exc, URLError) and isinstance(exc.reason, Exception) else exc
        result['transport_error_type'] = type(cause).__name__
        for key in ('errno', 'winerror'):
            number = getattr(cause, key, None)
            if type(number) is int:
                result[key] = number
        if isinstance(cause, TimeoutError):
            result['error_code'] = 'UPSTREAM_HTTP_TIMEOUT'
        elif isinstance(cause, RemoteDisconnected):
            result['error_code'] = 'UPSTREAM_REMOTE_DISCONNECTED'
        elif isinstance(exc, (json.JSONDecodeError, UnicodeDecodeError)):
            result['error_code'] = 'UPSTREAM_HTTP_BODY_INVALID'
        else:
            result['error_code'] = 'UPSTREAM_NETWORK_FAILED'
    if hasattr(client, 'last_http_status'):
        result['http_status'] = client.last_http_status
    http_total = time.monotonic()-http_started
    result['http_timing'] = {'send_to_completion_seconds':round(http_total,6),
        'send_to_headers_seconds':round(client.headers_seconds,6) if hasattr(client,'headers_seconds') else None,
        'headers_to_completion_seconds':round(http_total-client.headers_seconds,6) if hasattr(client,'headers_seconds') else None,
        'scope':'HTTP call and parse; send-to-headers includes connect/upload/server wait'}
    write_json(output_path,result)
