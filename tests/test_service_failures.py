"""真实服务故障的离线反例：余额错误停机、首个HTTP失败留痕、阻止空检测被采信。"""
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

from TCM_Fusion.lib.agent_tools import FixtureBackend,LocalToolbox
from TCM_Fusion.lib.controlled_process import run
from TCM_Fusion.lib.resource_budget import install_http_guard,operation_failure
from TCM_Fusion.lib.service_errors import service_error
from TCM_Fusion.lib.storage import IntegrationError,read_json,write_json
from TCM_Fusion.lib.workflow_runtime import WorkflowRuntime
from TCM_Fusion.lib.react_graph import run_react
from TCM_Fusion.tests.test_workflow import StageFixture


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

    def runtime(self):
        image=self.root/'fov.png';image.write_bytes(b'explicit-simulation')
        backend=StageFixture()
        runtime=WorkflowRuntime([image],root=self.root/'workflow',workflow_config={},sample_id='test',
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
