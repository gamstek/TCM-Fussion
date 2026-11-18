"""接收当前合法的观察或重算动作，检查样品绑定和预算，在调用前登记去重项并在结束时记账。LiveBackend 校验视觉响应和可选定位，LocalToolbox 统一这些行为供本地和 MCP 调用。显式模拟上下文可以注入替身。

主要定义：
- LiveBackend。
- LocalToolbox。

关联：
- 调用方 「lib/application/workflow_entry.py」、「lib/mcp/mcp_bridge.py」、「lib/offline/reflection.py」、「lib/offline/workflow_replay.py」、「tests/agent/test_investigation_control.py」、「tests/agent/test_planning_choices.py」。
- 直接依赖 「lib/runtime/hard_calls.py」、「lib/runtime/ledger_lock.py」、「lib/runtime/probe_budget.py」、「lib/runtime/storage.py」、「lib/tools/simulation.py」、「lib/vision/presence_localization.py」。
"""
from __future__ import annotations

import asyncio
import copy
import json
import re
import time
import threading
from urllib.error import HTTPError
from pathlib import Path

import paths

from TCM_Fusion.lib.runtime.storage import IntegrationError, digest, read_json, write_json
from TCM_Fusion.lib.tools.simulation import FixtureBackend


class LiveBackend:
    IMAGE_ENCODE_MAX_SECONDS = 20

    def __init__(self):
        from agent_loop import VLMClient
        class StrictClient(VLMClient):
            def presence_check(client, feature_zh, morph_hint, image_path):
                value = super().presence_check(feature_zh, morph_hint, image_path)
                if value.get('ok') and client.presence_proposal is not None:
                    value['localization_proposal'] = client.presence_proposal
                return value

            def _encode_image(client, image_path):
                return client.encoded_image

            # 原presence_check把字符串"false"转为True；在原解析之前严格拦截。
            def _post(client, payload):
                from TCM_Fusion.lib.runtime.hard_calls import post
                context = getattr(client, 'observation_context', {})
                if client.expected_tool == 'presence_check' and context.get('model_input_size'):
                    width, height = context['model_input_size']
                    payload = copy.deepcopy(payload)
                    payload['messages'][0]['content'].append({'type': 'text', 'text':
                        f'本次发送的图像宽{width}、高{height}像素。定位为可选线索，'
                        '不要为了输出框而猜测。有把握指出疑似目标位置时，可在同一JSON中增加 '
                        '"box":{"bbox":[x1,y1,x2,y2],"coordinate_space":"model_input_pixels"}；'
                        '坐标必须基于实际收到的这张图，从左上角起按像素计。'
                        '无法定位则省略box；定位不能替代present/conf/reason，也不要生成尺寸或标定。'})
                if client.expected_tool == "scene_check" and client.scene_reasoning_effort:
                    payload = {**payload, "reasoning_effort": client.scene_reasoning_effort}
                client.diagnostics["stage"] = "http_request"
                try:
                    body = post(payload, client.timeout_s, diagnostics=client.diagnostics)
                except HTTPError as exc:
                    client.failure = IntegrationError(f"UPSTREAM_HTTP_{exc.code}", "视觉请求 HTTP 错误")
                    raise client.failure from exc
                except IntegrationError as exc:
                    code = "UPSTREAM_HTTP_TIMEOUT" if exc.code == "WORKFLOW_TIMEOUT" else exc.code
                    client.failure = IntegrationError(code, "视觉请求失败")
                    raise client.failure from exc
                client.diagnostics["stage"] = "model_output"
                usage = body.get("usage") if isinstance(body, dict) else None
                if isinstance(usage, dict):
                    client.last_usage = {
                        key: usage[key] for key in ("prompt_tokens", "completion_tokens", "total_tokens")
                        if type(usage.get(key)) is int and usage[key] >= 0
                    }
                try:
                    choice = body["choices"][0]
                    finish = choice.get("finish_reason")
                    if isinstance(finish, str) and re.fullmatch(r"[a-zA-Z0-9_-]{1,32}", finish):
                        client.diagnostics["finish_reason"] = finish
                    if finish == "length":
                        raise IntegrationError("MODEL_OUTPUT_TRUNCATED", "视觉模型输出达到长度上限")
                    text = choice["message"]["content"]
                    if not isinstance(text, str) or not text.strip():
                        raise IntegrationError("MODEL_OUTPUT_EMPTY", "视觉模型正文为空")
                    match = re.search(r"\{.*\}", text, re.S)
                    if not match:
                        raise IntegrationError("MODEL_JSON_INVALID", "视觉模型正文缺少 JSON 对象")
                    obj = json.loads(match.group(0))
                    client.diagnostics["stage"] = "field_validation"
                    from TCM_Fusion.lib.runtime.storage import finite_number
                    if not finite_number(obj.get("conf"), maximum=1):
                        raise IntegrationError("MODEL_FIELD_INVALID", "conf 缺失或越界")
                    if client.expected_tool == "presence_check" and not isinstance(obj.get("present"), bool):
                        raise IntegrationError("MODEL_FIELD_INVALID", "present 须为布尔值")
                    if client.expected_tool == 'presence_check':
                        from TCM_Fusion.lib.vision.presence_localization import sanitize_proposal
                        client.presence_proposal = sanitize_proposal(obj.get('box'))
                    return body
                except json.JSONDecodeError:
                    client.failure = IntegrationError("MODEL_JSON_INVALID", "视觉模型 JSON 无法解析")
                except (KeyError, IndexError, TypeError, AttributeError):
                    client.failure = IntegrationError("MODEL_ENVELOPE_INVALID", "视觉模型响应结构异常")
                except IntegrationError as exc:
                    client.failure = exc
                raise client.failure
        config_path = Path(paths.DETECTION_CONFIG_DIR)/"vlm_config.json"
        self.client = StrictClient(config_path)
        config = read_json(config_path)
        self.client.scene_reasoning_effort = config.get("vlm", config).get("reasoning_effort")
        if not self.client.api_key:
            raise IntegrationError("CONFIG_MISSING", "缺VLM凭据")
        self.client.last_usage = None
        self.client.presence_proposal = None

    def call(self, runtime, action, path):
        from TCM_Fusion.lib.runtime.hard_calls import encode_image

        self.client.last_usage = None
        self.client.expected_tool = action["tool"]
        self.client.failure = None
        self.client.presence_proposal = None
        self.client.observation_context = {}
        self.client.diagnostics = {"stage": "image_encode"}
        timeout = self.client.timeout_s
        deadline = time.monotonic() + timeout
        self.client.encoded_image = encode_image(path, min(timeout, self.IMAGE_ENCODE_MAX_SECONDS))
        try:
            if action['tool'] == 'presence_check':
                from TCM_Fusion.lib.vision.presence_localization import observation_context
                self.client.observation_context = observation_context(path, self.client.encoded_image)
            for attempt in (1, 2):
                self.client.timeout_s = deadline - time.monotonic()
                if self.client.timeout_s <= 0:
                    raise IntegrationError("WORKFLOW_TIMEOUT", "视觉请求预算已耗尽")
                self.client.diagnostics["attempts"] = attempt
                try:
                    value = runtime.tools.invoke(runtime, action, {"backend": self, "path": path})
                    return value, self.client.last_usage
                except IntegrationError as exc:
                    failure = (self.client.failure or exc) if exc.code == "TOOL_OUTPUT_INVALID" else exc
                    workflow = getattr(runtime, "workflow", None)
                    if (attempt != 1 or action["tool"] != "scene_check" or workflow is None
                            or failure.code != "UPSTREAM_REMOTE_DISCONNECTED"
                            or deadline - time.monotonic() < 5):
                        raise failure
                    try:
                        workflow.charge("network", "scene-retry:" + action["action_id"])
                    except IntegrationError:
                        self.client.diagnostics["retry_skipped"] = "budget"
                        raise failure
                    self.client.diagnostics = {"previous_error_code": failure.code,
                                               "previous_transport_error_type": self.client.diagnostics.get("transport_error_type")}
                    self.client.failure = None
                    self.client.last_usage = None
        finally:
            self.client.timeout_s = timeout
            del self.client.encoded_image


class LocalToolbox:
    def __init__(self, runtime, *, ledger_path, max_requests=6, allow_network=False, backend=None):
        if not isinstance(max_requests, int) or isinstance(max_requests, bool) or not 0 <= max_requests <= 100:
            raise IntegrationError("CONFIG_INVALID", "工具预算须为0～100整数")
        if backend is not None and not runtime.simulation:
            raise IntegrationError("SIMULATION_SCOPE", "真实观察禁止注入替身后端")
        self.runtime = runtime
        self.ledger_path = Path(ledger_path)
        self.max_requests = max_requests
        self.allow_network = allow_network
        self.backend = backend
        self.lock = asyncio.Lock()
        self.identity = digest({"sample": runtime.sample_id, "config": runtime.config_identity,
                                "kb": runtime.kb_sha256, "images": sorted(runtime.images),
                                "records": runtime.records, "simulation": runtime.simulation})
        if getattr(runtime, "workflow", None) is not None:
            self.identity = runtime.workflow.identity
        self.ledger = {"identity": self.identity, "requests": 0, "entries": {}}
        if self.ledger_path.exists():
            self.ledger = read_json(self.ledger_path)
            if self.ledger.get("identity") != self.identity:
                raise IntegrationError("LEDGER_MISMATCH", "工具账本与样品不匹配")

    async def recompute(self, action):
        if getattr(self.runtime, "workflow", None) is None:
            raise IntegrationError("ACTION_DENIED", "没有工作流上下文")
        task=asyncio.create_task(asyncio.to_thread(self.runtime.tools.invoke, self.runtime, action))
        try:return await asyncio.shield(task)
        except asyncio.CancelledError:
            self.runtime.workflow.cancel_event.set()
            await asyncio.shield(task)
            raise

    async def read(self, name, args, state):
        return self.runtime.tools.read(self.runtime, name, args, state)


    async def observe(self, action):
        from TCM_Fusion.lib.runtime.ledger_lock import acquire, release
        async with self.lock:
            handle = await asyncio.to_thread(acquire, self.ledger_path)
            try:
                if self.ledger_path.exists():
                    self.ledger = read_json(self.ledger_path)
                    if self.ledger.get("identity") != self.identity:
                        raise IntegrationError("LEDGER_MISMATCH", "工具账本与样品不匹配")
                pending=self._prepare_observation(action)
            finally:
                release(handle)
        if pending.get('cache_hit'):return pending
        try:
            record=await self._run_observation(action,pending)
        except asyncio.CancelledError:
            record = {**pending, 'accepted': False, 'status': 'cancelled',
                      'error_code': 'OBSERVATION_CANCELLED',
                      'result': {'ok': False, 'error_code': 'OBSERVATION_CANCELLED'}}
            await asyncio.shield(self._persist_observation(action, record))
            raise
        except Exception as exc:
            code = 'INPUT_CHANGED' if isinstance(exc, OSError) else getattr(exc, 'code', 'OBSERVATION_FAILED')
            record = {**pending, 'accepted': False, 'status': 'failed', 'error_code': code,
                      'result': {'ok': False, 'error_code': code}}
        await self._persist_observation(action, record)
        return record

    async def _persist_observation(self, action, record):
        """Write terminal accounting even after close, never accept late evidence."""
        from TCM_Fusion.lib.runtime.ledger_lock import acquire, release
        async with self.lock:
            handle=await asyncio.to_thread(acquire,self.ledger_path)
            try:
                self.ledger=read_json(self.ledger_path)
                workflow = getattr(self.runtime, 'workflow', None)
                budget_handle = await asyncio.to_thread(acquire, workflow.budget_path) if workflow is not None else None
                try:
                    self._invalidate_closed_observation(record, workflow)
                    try:self.runtime.validate_action(action)
                    except (IntegrationError, OSError):
                        record.update(accepted=False, status='failed', error_code='INPUT_CHANGED',
                                      result={'ok': False, 'error_code': 'INPUT_CHANGED'})
                        if 'presence_localization' in record:
                            record['presence_localization'].update(state='unknown', localization='invalid',
                                                                   ineligible_reason='INPUT_CHANGED')
                    # Terminal accounting is allowed after close; late evidence is not.
                    self.ledger['entries'][action['action_id']]=record
                    write_json(self.ledger_path,self.ledger)
                finally:
                    if budget_handle is not None:release(budget_handle)
            finally:release(handle)

    def _prepare_observation(self, action):
        path = self.runtime.validate_action(action)
        key = self.runtime.action(action["tool"], action["args"], action["skill"])["action_id"]
        if key != action.get("action_id"):
            raise IntegrationError("ACTION_DENIED", "动作标识被篡改")
        if key in self.ledger["entries"]:
            return {**self.ledger["entries"][key], "cache_hit": True, "new_requests": 0}
        history = list(self.ledger["entries"].values())
        state = {"observations": [r for r in history if "action" in r],
                 "attempted": list(self.ledger["entries"])}
        legal = {a["action_id"] for a in self.runtime.legal_actions(state)}
        if key not in legal:
            raise IntegrationError("ACTION_DENIED", "动作不在当前合法集合；强制场景先行")
        if self.ledger["requests"] >= self.max_requests:
            raise IntegrationError("TOOL_BUDGET", "观察服务器预算耗尽")
        if not self.runtime.simulation and not self.allow_network:
            raise IntegrationError("NETWORK_DISABLED", "真实观察需要allow_network=true")
        # 调用前写pending。崩溃/重启不自动重发，失败结果也入去重账本。
        workflow = getattr(self.runtime, "workflow", None)
        if workflow is not None:
            from TCM_Fusion.lib.runtime.probe_budget import describe
            workflow.charge("observation", "observation:"+key, probe=describe(self.runtime, action))
        pending = {"action_id": key, "accepted": False, "result": {"ok": False},
                   "action": copy.deepcopy(action),
                   "status": "pending", "cache_hit": False, "new_requests": 1}
        self.ledger["requests"] += 1
        self.ledger["entries"][key] = pending
        write_json(self.ledger_path, self.ledger)
        return pending

    async def _run_observation(self,action,pending):
        workflow=getattr(self.runtime,'workflow',None)
        stop_code = self._observation_stop_code(workflow)
        if stop_code:
            return {**pending, 'accepted': False, 'status': 'failed', 'new_requests': 0,
                    'error_code': stop_code, 'result': {'ok': False, 'error_code': stop_code}}
        path=self.runtime.validate_action(action)
        backend = None
        try:
            backend=self.backend or (FixtureBackend() if self.runtime.simulation else LiveBackend())
            if workflow is not None and hasattr(backend,'client'):
                # 时间预算默认 10 分钟；不再让客户端配置里的短 timeout 先截断长观察。
                cap = workflow.config.get("max_observation_seconds",
                                          workflow.config.get("max_call_seconds", 600))
                backend.client.timeout_s = min(workflow.remaining_seconds(), float(cap))
            result, usage = await asyncio.to_thread(backend.call, self.runtime, action, path)
            # 失败reason可能含HTTP body/凭据；只保留结构化成功观察。
            accepted = self.runtime.validate_observation(action, result)
            record = {**pending, "status": "done", "accepted": accepted,
                      "result": result if accepted else {"ok": False, "error": "INVALID_OR_FAILED_OBSERVATION"},
                      "usage": usage}
        except Exception as exc:
            record = {**pending, "status": "failed", "error_type": type(exc).__name__}
            record['error_code']=getattr(exc,'code','OBSERVATION_FAILED')
            record['result'] = {"ok": False, "error_code": record['error_code']}
        client = getattr(backend, "client", None)
        if isinstance(getattr(client, "diagnostics", None), dict):
            record["diagnostics"] = client.diagnostics
            record["usage"] = client.last_usage
            record["new_requests"] = client.diagnostics.get("attempts", record["new_requests"])
        try:
            self.runtime.validate_action(action)  # 观察期间改图，结果不可用
        except (IntegrationError, OSError):
            record.update(accepted=False, result={"ok": False, "error": "INPUT_CHANGED"})
        self._invalidate_closed_observation(record, workflow)
        if action['tool'] == 'presence_check' and record.get('accepted'):
            from TCM_Fusion.lib.vision.presence_localization import assess_presence
            image = self.runtime.images[action['args']['image_id']]
            context = getattr(client, 'observation_context', {})
            cancelled = threading.Event()
            try:
                record['presence_localization'] = await asyncio.to_thread(
                    assess_presence, record['result'], path=path,
                    expected_sha256=image['sha256'], feature_id=action['args']['feature_id'],
                    context=context, output_dir=self.ledger_path.parent / 'presence_localizations' / action['action_id'],
                    existing=self._presence_spatial_context(action),
                    can_publish=lambda: not cancelled.is_set() and self._observation_stop_code(workflow) is None)
            except asyncio.CancelledError:
                cancelled.set()
                raise
            except Exception:
                # Optional diagnostics must not erase a valid legacy P1/C observation.
                record['presence_localization'] = {'policy_version': 'presence_localization_shadow_v1',
                    'mode': 'shadow', 'state': 'unknown', 'findings': ['LOCALIZATION_DIAGNOSTIC_FAILED'],
                    'legacy_presence_unchanged': True}
            try:
                self.runtime.validate_action(action)
            except (IntegrationError, OSError):
                record.update(accepted=False, result={'ok': False, 'error': 'INPUT_CHANGED'})
                record['presence_localization'].update(state='unknown', localization='invalid')
        self._invalidate_closed_observation(record, workflow)
        return record

    @staticmethod
    def _observation_stop_code(workflow):
        if workflow is None:
            return None
        if workflow.cancel_event.is_set():
            return 'WORKFLOW_CANCELLED'
        budget = read_json(workflow.budget_path)
        if budget.get('closed'):
            return 'WORKFLOW_CLOSED'
        if time.time() >= budget['deadline']:
            return 'WORKFLOW_TIMEOUT'
        return None

    def _invalidate_closed_observation(self, record, workflow):
        code = self._observation_stop_code(workflow)
        if code:
            record.update(accepted=False, status='late', error_code=code,
                          result={'ok': False, 'error_code': code})
            if 'presence_localization' in record:
                record['presence_localization'].update(state='unknown', localization='invalid',
                                                       ineligible_reason=code)

    def _presence_spatial_context(self, action):
        """Read current formal crop geometry; overlap never deletes/approves a hint."""
        workflow = getattr(self.runtime, 'workflow', None)
        if workflow is None:
            return []
        import contracts
        rows = []
        for identifier, artifact in workflow.store.current()['artifacts'].items():
            if (not artifact.get('valid') or artifact.get('kind') != 'crop'
                    or artifact.get('image_id') != action['args']['image_id']):
                continue
            try:
                meta = contracts.CropMeta.from_json(artifact['path'])
                rows.append({'id': identifier, 'box': meta.tight_box, 'feature_id': meta.feature_id})
            except (OSError, ValueError, KeyError, TypeError):
                continue
        return rows
