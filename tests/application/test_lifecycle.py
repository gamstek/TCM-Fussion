"""验证正式运行入口的终态和模拟边界。

使用临时账本和显式故障注入，覆盖初始签名、工作流退出以及活动 MCP 上下文。
该组用例由项目根 selftest 发现。涉及模型回复的用例使用可控输入，涉及进程的用例以各断言规定的执行边界为准。

主要定义：
- LifecycleTests 执行真实的应用层错误处理与工具箱绑定。

关联：
- 直接依赖 「lib/application/pipeline.py」、「lib/application/workflow_entry.py」、「lib/mcp/mcp_bridge.py」、「lib/runtime/storage.py」、「lib/tools/agent_tools.py」、「lib/workflow/evidence_runtime.py」、「lib/workflow/workflow_backend.py」、「lib/workflow/workflow_runtime.py」。
- 统一执行见 tests/README.md。
"""
import asyncio
import time
import unittest
from unittest.mock import patch

from TCM_Fusion.lib.application.pipeline import run_pipeline
from TCM_Fusion.lib.application.workflow_entry import execute_workflow
from TCM_Fusion.lib.mcp.mcp_bridge import load_bound_context
from TCM_Fusion.lib.runtime.storage import IntegrationError, read_json, write_json
from TCM_Fusion.lib.tools.agent_tools import FixtureBackend, LocalToolbox
from TCM_Fusion.lib.workflow.workflow_backend import WorkflowBackend
from TCM_Fusion.lib.workflow.evidence_runtime import EvidenceRuntime
from TCM_Fusion.lib.workflow import workflow_runtime
from TCM_Fusion.tests.support.workflow_fixtures import WorkflowFixture


class LifecycleTests(WorkflowFixture, unittest.TestCase):
    def config(self):
        return {
            'a': {'mode': 'unavailable'},
            'b': {'mode': 'run'},
            'agent': {
                'mode': 'active', 'planner': 'heuristic', 'image_roots': [str(self.image)],
                'workflow': {},
            },
        }

    def test_startup_signature_error_leaves_failed_job(self):
        with patch.object(WorkflowBackend, 'code_signature', side_effect=OSError('signature read failed')):
            with self.assertRaises(IntegrationError):
                run_pipeline(self.config(), base=self.root)
        job, = (self.root / 'runs').glob('*/job.json')
        self.assertEqual(read_json(job)['status'], 'failed')

    def test_abnormal_workflow_exit_closes_budget_and_signals_cancel(self):
        runtime = self.runtime()
        failure = IntegrationError('PLANNING_FAILED', 'injected failure')
        with patch('TCM_Fusion.lib.application.workflow_entry.from_task', return_value=runtime):
            with patch('TCM_Fusion.lib.application.workflow_entry.run_react', side_effect=failure):
                with self.assertRaises(IntegrationError) as ctx:
                    asyncio.run(execute_workflow(
                        self.config(), {}, self.root, routing={}, start=time.perf_counter(),
                        job={}, status_path=self.root/'job.json',
                    ))
        self.assertIs(ctx.exception, failure)
        self.assertTrue(runtime.workflow.cancel_event.is_set())
        ledger = read_json(runtime.workflow.budget_path)
        self.assertTrue(ledger['closed'])
        self.assertEqual(ledger['stop_reason'], 'PLANNING_FAILED')

    def test_non_simulation_rejects_backend_injection_even_with_network_enabled(self):
        runtime = EvidenceRuntime([], [self.image], sample_id='real')
        with self.assertRaises(IntegrationError) as ctx:
            LocalToolbox(runtime, ledger_path=self.root/'ledger.json',
                         allow_network=True, backend=FixtureBackend())
        self.assertEqual(ctx.exception.code, 'SIMULATION_SCOPE')
        self.assertFalse((self.root/'ledger.json').exists())

    def test_workflow_exports_audit_for_the_finished_evidence_version(self):
        config = self.config()
        config['agent']['workflow']['feature_audit'] = True
        runtime = self.runtime()
        with patch('TCM_Fusion.lib.application.workflow_entry.from_task', return_value=runtime):
            asyncio.run(execute_workflow(
                config, {}, self.root, routing={}, start=time.perf_counter(),
                job={}, status_path=self.root/'job.json',
            ))
        audit = read_json(self.root/'feature_audit/feature_audit.json')
        output = read_json(self.root/'output.json')
        self.assertEqual(audit, output['feature_audit'])
        self.assertEqual(audit['evidence_version'], runtime.workflow.store.current()['version'])

    def test_mcp_workflow_context_uses_task_ledger(self):
        runtime = self.runtime()
        context_path = self.root/'context.json'
        a_path = self.root/'a_prediction.json'
        write_json(a_path, {})
        write_json(context_path, {'workflow_config': self.config(), 'directory': str(self.root),
                                  'a_prediction': str(a_path)})
        with patch.object(workflow_runtime, 'from_task', return_value=runtime):
            _, toolbox = load_bound_context(context_path)
        self.assertEqual(toolbox.ledger_path, self.root/'agent_tool_ledger.json')
