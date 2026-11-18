"""真实服务故障的离线反例：余额/鉴权失败停机、首个 HTTP 失败留痕、断连重试上限。
该组用例由项目根 selftest 发现。涉及模型回复的用例使用可控输入，涉及进程的用例以各断言规定的执行边界为准。

主要定义：
- ServiceFailureTests（错误解析）: service_error 保留厂商码但丢弃响应正文，5xx 不映射为余额错误。
- ServiceFailureTests（信用与鉴权）: 429 或 VLM_AUTH_FAILED 不触发付费工具回退，只要求复核。
- ServiceFailureTests（断连重试）: 首次 UPSTREAM_REMOTE_DISCONNECTED 在预算内重试一次，持续断连两次后停止。
- ServiceFailureTests（HTTP 守卫）: 首个失败被持久化并提前终止 worker，成功响应原样交给原解析器。
- http_error / budget: 构造带假正文的 HTTPError 与临时预算上下文，验证不泄露敏感串。

关联：
- 直接依赖 「lib/agent/react_graph.py」、「lib/runtime/controlled_process.py」、「lib/runtime/resource_budget.py」、「lib/runtime/service_errors.py」、「lib/runtime/storage.py」、「lib/tools/agent_tools.py」、「lib/workflow/workflow_runtime.py」、「tests/support/workflow_fixtures.py」。
- 统一执行见 tests/README.md。
"""
import asyncio
import io
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from urllib.error import HTTPError
import urllib.request

from TCM_Fusion.lib.tools.agent_tools import FixtureBackend,LocalToolbox
from TCM_Fusion.lib.runtime.controlled_process import run
from TCM_Fusion.lib.runtime.resource_budget import install_http_guard,operation_failure
from TCM_Fusion.lib.runtime.service_errors import service_error
from TCM_Fusion.lib.runtime.storage import IntegrationError,read_json,write_json
from TCM_Fusion.lib.workflow.workflow_runtime import WorkflowRuntime
from TCM_Fusion.lib.agent.react_graph import run_react
from TCM_Fusion.tests.support.workflow_fixtures import StageFixture


def http_error(status=429):
    body={'error':{'code':'1113' if status==429 else '500','message':'UNIT_TEST_SECRET_NOT_REAL'}}
    return HTTPError('https://invalid.example',status,'test',None,io.BytesIO(json.dumps(body).encode()))


class ServiceFailureTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)

    def budget(self):
        budget=self.root/'budget.json';ctx=self.root/'context.json'
        write_json(budget,{'identity':'test','deadline':time.time()+60,'used':{},'entries':{}})
        write_json(ctx,{'identity':'test','budget_path':str(budget),'operation_key':'op',
                        'limits':{'network':2,'total':2}})
        return budget,ctx

    def test_service_error_retains_codes_without_response_body(self):
        result,reason=service_error(http_error())
        self.assertEqual(reason,'VLM_CREDIT_EXHAUSTED');self.assertEqual(result['vendor_code'],'1113')
        self.assertNotIn('UNIT_TEST_SECRET',json.dumps(result))
        self.assertIsNone(service_error(http_error(500))[1])

    def runtime(self, **config):
        image=self.root/'fov.png';image.write_bytes(b'explicit-simulation')
        backend=StageFixture()
        runtime=WorkflowRuntime([image],root=self.root/'workflow',workflow_config=config,sample_id='test',
                                simulation=True,backend=backend)
        return runtime,backend

    def test_credit_error_does_not_fallback_to_more_paid_tools(self):
        class Planner:
            def choose(self,context):raise http_error()
        runtime,backend=self.runtime()
        toolbox=LocalToolbox(runtime,ledger_path=self.root/'obs.json',backend=FixtureBackend())
        with patch('urllib.request.urlopen') as remote:
            trace=asyncio.run(run_react(runtime,toolbox,planner=Planner(),planner_mode='llm'))
        remote.assert_not_called();self.assertEqual(backend.calls,[])
        self.assertEqual(trace['stop_reason'],'VLM_CREDIT_EXHAUSTED');self.assertTrue(trace['needs_review'])
        self.assertEqual(trace['events'][-1]['service_error']['http_status'],429)

    def test_transient_planner_error_preserves_bounded_fallback(self):
        class Planner:
            def choose(self,context):raise http_error(500)
        runtime,backend=self.runtime()
        toolbox=LocalToolbox(runtime,ledger_path=self.root/'obs.json',backend=FixtureBackend())
        trace=asyncio.run(run_react(runtime,toolbox,planner=Planner(),planner_mode='llm',max_steps=2))
        self.assertIn('detect_fov_initial',backend.calls);self.assertTrue(trace['planner_degraded'])

    def run_autonomous(self, planner, **config):
        runtime,backend=self.runtime(agent_policy='adaptive_v2', auto_prepare_stage=False, **config)
        toolbox=LocalToolbox(runtime,ledger_path=self.root/'obs.json',backend=FixtureBackend())
        with patch('urllib.request.urlopen') as remote:
            trace=asyncio.run(run_react(runtime,toolbox,planner=planner,planner_mode='llm',max_steps=1))
        remote.assert_not_called()
        return runtime,backend,trace

    def test_disconnected_planner_retries_once_then_keeps_model_choice(self):
        class Planner:
            calls=0
            def choose(self,context):
                self.calls+=1
                if self.calls==1:
                    raise IntegrationError('UPSTREAM_REMOTE_DISCONNECTED','fixture disconnect')
                action=next(a for a in context['actions'] if a['tool']=='detect_fov_initial')
                return {'action_id':action['action_id'],'decision_summary':'按当前上下文执行初检'}, {'total_tokens':7}
        planner=Planner()
        runtime,backend,trace=self.run_autonomous(planner)
        self.assertEqual(planner.calls,2)
        self.assertEqual(backend.calls,['detect_fov_initial'])
        self.assertFalse(trace['planner_degraded'])
        self.assertEqual(trace['planning_requests'],2)
        self.assertEqual(read_json(runtime.workflow.budget_path)['used']['planning'],2)
        errors=[e for e in trace['events'] if e['phase']=='planning_error']
        self.assertEqual(len(errors),1);self.assertTrue(errors[0]['retry_scheduled'])
        self.assertIsNone(errors[0]['usage'])
        self.assertEqual(next(e for e in trace['events'] if e['phase']=='thought')['decision_source'],
                         'injected_planner')  # Explicit fixture provenance, never impersonate a live model.

    def test_persistent_disconnect_stops_after_two_charged_requests(self):
        planner=SimpleNamespace(choose=unittest.mock.Mock(side_effect=
            IntegrationError('UPSTREAM_REMOTE_DISCONNECTED','fixture disconnect')))
        runtime,backend,trace=self.run_autonomous(planner)
        self.assertEqual(planner.choose.call_count,2);self.assertEqual(backend.calls,[])
        self.assertEqual(trace['stop_reason'],'runtime_request_review')
        self.assertEqual(trace['termination_reason'],'UPSTREAM_REMOTE_DISCONNECTED')
        self.assertEqual(read_json(runtime.workflow.budget_path)['used']['network'],2)
        self.assertEqual([e['retry_scheduled'] for e in trace['events']
                          if e['phase']=='planning_error'],[True,False])

    def test_disconnect_without_planning_budget_is_not_retried(self):
        planner=SimpleNamespace(choose=unittest.mock.Mock(side_effect=
            IntegrationError('UPSTREAM_REMOTE_DISCONNECTED','fixture disconnect')))
        _,backend,trace=self.run_autonomous(planner,max_planning=1)
        self.assertEqual(planner.choose.call_count,1);self.assertEqual(backend.calls,[])
        self.assertEqual(trace['termination_reason'],'UPSTREAM_REMOTE_DISCONNECTED')

    def test_autonomous_credit_error_never_retries_or_executes_paid_fallback(self):
        planner=SimpleNamespace(choose=unittest.mock.Mock(side_effect=http_error()))
        _,backend,trace=self.run_autonomous(planner)
        self.assertEqual(planner.choose.call_count,1);self.assertEqual(backend.calls,[])
        self.assertEqual(trace['termination_reason'],'VLM_CREDIT_EXHAUSTED')

    def test_typed_auth_failure_does_not_execute_initial_detection_fallback(self):
        planner=SimpleNamespace(choose=unittest.mock.Mock(side_effect=
            IntegrationError('VLM_AUTH_FAILED','fixture auth failure')))
        _,backend,trace=self.run_autonomous(planner)
        self.assertEqual(planner.choose.call_count,1);self.assertEqual(backend.calls,[])
        self.assertEqual(trace['termination_reason'],'VLM_AUTH_FAILED')

    def test_first_http_failure_is_persisted_and_stops_worker(self):
        budget,ctx=self.budget();original=urllib.request.urlopen;redirect=urllib.request.HTTPRedirectHandler.redirect_request
        try:
            with patch('urllib.request.urlopen',side_effect=http_error()) as remote:
                install_http_guard(ctx)
                with self.assertRaises(HTTPError):urllib.request.urlopen('https://invalid.example')
                with self.assertRaises(IntegrationError):urllib.request.urlopen('https://invalid.example')
                self.assertEqual(remote.call_count,1)
        finally:urllib.request.urlopen=original;urllib.request.HTTPRedirectHandler.redirect_request=redirect
        self.assertEqual(operation_failure(ctx)['http_status'],429)
        started=time.monotonic()
        with self.assertRaises(IntegrationError) as caught:
            run([sys.executable,'-c','import time; time.sleep(10)'],context_path=ctx,
                timeout=15,log_path=self.root/'worker.log')
        self.assertEqual(caught.exception.code,'UPSTREAM_HTTP_429')
        self.assertLess(time.monotonic()-started,5)
        self.assertNotIn('UNIT_TEST_SECRET',json.dumps(read_json(budget)))

    def test_success_guard_preserves_response_for_original_parser(self):
        budget,ctx=self.budget();original=urllib.request.urlopen;redirect=urllib.request.HTTPRedirectHandler.redirect_request
        response=SimpleNamespace(status=200,read=lambda:b'original-body')
        try:
            with patch('urllib.request.urlopen',return_value=response):
                install_http_guard(ctx)
                value=urllib.request.urlopen('https://invalid.example')
                self.assertIs(value,response);self.assertEqual(value.read(),b'original-body')
        finally:urllib.request.urlopen=original;urllib.request.HTTPRedirectHandler.redirect_request=redirect
        self.assertIsNone(operation_failure(ctx))
        self.assertEqual(next(iter(read_json(budget)['entries'].values()))['http_status'],200)


    def test_invalid_worker_context_rejected_before_spawn(self):
        from TCM_Fusion.lib.runtime.controlled_process import run
        with patch('TCM_Fusion.lib.runtime.controlled_process.subprocess.Popen') as spawn:
            for value in [{},self.root/'missing_context.json']:
                with self.subTest(value=value),self.assertRaises(IntegrationError) as caught:
                    run(['unused'],context_path=value,timeout=1,log_path=self.root/'invalid.log')
                self.assertEqual(caught.exception.code,'BUDGET_CONTEXT_INVALID')
            spawn.assert_not_called()

    def test_local_supervisor_failure_is_not_reported_as_timeout(self):
        from TCM_Fusion.lib.runtime.controlled_process import run
        from unittest.mock import Mock
        _,context=self.budget();worker=Mock();worker.poll.return_value=None
        with patch('TCM_Fusion.lib.runtime.resource_budget.operation_failure',side_effect=[None,RuntimeError('fixture')]), \
             patch('TCM_Fusion.lib.runtime.controlled_process.subprocess.Popen',return_value=worker), \
             patch('TCM_Fusion.lib.runtime.controlled_process.kill_tree') as cleanup:
            with self.assertRaises(IntegrationError) as caught:
                run(['unused'],context_path=context,timeout=1,log_path=self.root/'supervisor.log')
        self.assertEqual(caught.exception.code,'WORKER_CONTROL_FAILED');cleanup.assert_called_once_with(worker)
