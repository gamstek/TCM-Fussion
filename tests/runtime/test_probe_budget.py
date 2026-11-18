"""支持性探测预算与 presence 上下文压缩的离线回归，不调用任何模型 API。
该组用例由项目根 selftest 发现。涉及模型回复的用例使用可控输入，涉及进程的用例以各断言规定的执行边界为准。

主要定义：
- ProbeBudgetTests（零消耗提议）: 反复生成 presence 动作不会改动预算账本。
- ProbeBudgetTests（上下文压缩）: compact_round_context 去掉 artifacts/transform 与原始 metrics，保留空间重叠摘要且不改原对象。
- ProbeBudgetTests（共享探测）: 同图同特征的探测在不同候选间共享 key 与配额，第二次预约返回 PROBE_ALREADY_ATTEMPTED。
- ProbeBudgetTests（并发原子性）: 两个竞争调用者只能有一个成功预约，账本只记录一次。
- ProbeBudgetTests（共享配额）: presence_check 与 feature_scan 共享 supporting 配额，用尽后不再提供扫描。

关联：
- 直接依赖 「lib/agent/react_graph.py」、「lib/evidence/search_ledger.py」、「lib/runtime/probe_budget.py」、「lib/runtime/resource_budget.py」、「lib/runtime/storage.py」、「lib/tools/agent_tools.py」、「lib/workflow/workflow_runtime.py」、「tests/support/workflow_fixtures.py」。
- 统一执行见 tests/README.md。
"""
import tempfile
import unittest
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from TCM_Fusion.lib.tools.agent_tools import FixtureBackend, LocalToolbox
from TCM_Fusion.lib.runtime.probe_budget import describe, check_reservation
from TCM_Fusion.lib.evidence.search_ledger import render_shared
from TCM_Fusion.lib.agent.react_graph import compact_round_context
from TCM_Fusion.lib.runtime.resource_budget import reserve
from TCM_Fusion.lib.runtime.storage import IntegrationError, read_json, write_json
from TCM_Fusion.lib.workflow.workflow_runtime import WorkflowRuntime
from TCM_Fusion.tests.support.workflow_fixtures import StageFixture


class ProbeBudgetTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.image = self.root / 'fov.png'; self.image.write_bytes(b'explicit-offline-image')
        self.a = {'status': 'ok', 'ranking': [{'label': '白芍', 'probability': .6},
                                            {'label': 'G1', 'probability': .4}]}
        self.runtime = WorkflowRuntime(
            [self.image], root=self.root / 'workflow', sample_id='probe-budget-test',
            a=self.a, simulation=True, backend=StageFixture(), workflow_config={
                'agent_policy': 'adaptive_v2', 'supporting_probe_total': 6,
                'supporting_probe_per_candidate': 2, 'max_observations': 10,
                'max_planning': 10, 'max_api_calls': 40, 'max_resource_units': 100})
        self.backend = FixtureBackend()
        self.toolbox = LocalToolbox(self.runtime, ledger_path=self.root / 'observations.json',
                                    max_requests=10, backend=self.backend)

    def presence_actions(self):
        return [a for a in self.runtime.legal_actions({'observations': [], 'attempted': []})
                if a['tool'] == 'presence_check' and describe(self.runtime, a)]

    async def test_proposals_do_not_spend_budget(self):
        before = read_json(self.runtime.workflow.budget_path)
        first, second = self.presence_actions(), self.presence_actions()
        self.assertEqual(first, second)
        self.assertEqual(before, read_json(self.runtime.workflow.budget_path))

    async def test_presence_shadow_is_compacted_before_model_context(self):
        self.runtime.workflow.config['context_compact'] = True
        shadow = {
            'policy_version': 'presence_localization_shadow_v1',
            'localization': 'valid', 'quality': 'usable', 'identity': 'pending',
            'state': 'present_pending', 'findings': ['small_box'],
            'usage_eligibility': {'presence': False, 'length': False},
            'source_tight_bbox': [1, 2, 30, 40], 'crop_bbox': [0, 0, 50, 60],
            'quality_diagnostic': {'metrics': {'large': 'x' * 5000}},
            'artifacts': {'target_png': '/very/long/path/target.png'},
            'transform': {'scale_xy': [2, 2], 'padding_xy': [0, 0]},
            'spatial_agreement': [
                {'artifact_id': 'crop-1', 'feature_id': 'cluster_crystal',
                 'iou': 0.8, 'possible_duplicate': True,
                 'possible_identity_conflict': False, 'identity_inference': False}],
        }
        context = {'previous_observations': [{'presence_localization': shadow}]}
        compacted = compact_round_context(context, self.runtime.workflow)
        value = compacted['previous_observations'][0]['presence_localization']
        self.assertTrue(value['compacted'])
        self.assertEqual(value['spatial_agreement']['count'], 1)
        self.assertTrue(value['spatial_agreement']['possible_duplicate'])
        self.assertNotIn('artifacts', value)
        self.assertNotIn('transform', value)
        self.assertEqual(value['quality_diagnostic']['summary'], 'details_in_diagnostic_json')
        self.assertIn('artifacts', shadow)
        self.assertEqual(len(shadow['quality_diagnostic']['metrics']['large']), 5000)

    async def test_same_image_feature_presence_is_shared_across_candidates(self):
        actions = self.presence_actions()
        first = next(a for a in actions if a['args'].get('feature_id'))
        second = {**first, 'args': {**first['args'], 'morph_hint': 'different candidate morphology'}}
        first_probe = describe(self.runtime, first)
        second_probe = describe(self.runtime, second)
        self.assertIsNotNone(second_probe)
        self.assertEqual(first_probe['key'], second_probe['key'])
        self.assertEqual(first_probe['candidate_ids'], second_probe['candidate_ids'])
        reserve(self.runtime.workflow.budget_path, 'observation', 'shared-first',
                self.runtime.workflow.limits, extra_pool='network', probe=first_probe)
        book = read_json(self.runtime.workflow.budget_path)
        self.assertEqual(check_reservation(book, second_probe), 'PROBE_ALREADY_ATTEMPTED')
        state = {'observations': [{'action': first,
            'accepted': True, 'result': {'ok': True, 'present': True, 'conf': .9}}]}
        shared = render_shared(self.runtime, state)
        self.assertTrue(shared)
        self.assertEqual(shared[0]['status'], 'shareable')
        self.assertEqual(shared[0]['candidate_ids'], first_probe['candidate_ids'])

    async def test_shared_reservation_is_atomic_across_competing_callers(self):
        first = describe(self.runtime, self.presence_actions()[0])
        first['total_limit'] = 1
        second = {**first, 'key': 'different-question'}
        def charge(index):
            try:
                reserve(self.runtime.workflow.budget_path, 'observation', 'atomic-' + str(index),
                        self.runtime.workflow.limits, extra_pool='network',
                        probe=[first, second][index])
                return 'OK'
            except IntegrationError as exc:
                return exc.code
        with ThreadPoolExecutor(max_workers=2) as pool:
            codes = list(pool.map(charge, [0, 1]))
        self.assertEqual(codes.count('OK'), 1)
        book = read_json(self.runtime.workflow.budget_path)
        self.assertEqual(len(book['supporting_probes']), 1)
        self.assertEqual(book['used']['observation'], 1)
        self.assertEqual(book['used']['network'], 1)

    def scan_runtime(self):
        class EmptyScan(StageFixture):
            def execute(stage, session, tool, target, work, context):
                if tool in {'detect_fov_initial', 'feature_scan'}:
                    folder = work / 'output/crops'; folder.mkdir(parents=True)
                    completion = folder.parent / 'completion.json'
                    write_json(completion, {'count': 0, 'feature_id': target.get('feature_id')})
                    return {'crops_dir': str(folder), 'completion': str(completion)}
                return super().execute(session, tool, target, work, context)
        return WorkflowRuntime(
            [self.image], root=self.root / 'scan-workflow', sample_id='scan-budget',
            a=self.a, simulation=True, backend=EmptyScan(), workflow_config={
                'agent_policy': 'adaptive_v2', 'supporting_probe_total': 1,
                'supporting_probe_per_candidate': 2, 'fast_yolo_stage2': True,
                'fast_recovery_selection': 'per_feature', 'fast_fallback_max_features': 4,
                'fast_fallback_max_locate_calls': 16, 'feature_scan_enabled': True,
                'max_api_calls': 40, 'max_resource_units': 100})

    async def test_presence_and_feature_scan_share_quota(self):
        runtime = self.scan_runtime()
        tools = LocalToolbox(runtime, ledger_path=self.root / 'shared-quota.json', backend=FixtureBackend())
        presence = next(a for a in runtime.legal_actions({'observations': []})
                        if a['tool'] == 'presence_check' and describe(runtime, a))
        result = await tools.observe(presence)
        self.assertTrue(result['accepted'])
        offers = runtime.legal_actions({'observations': [result], 'attempted': [presence['action_id']]})
        self.assertFalse(any(a['tool'] == 'feature_scan' and describe(runtime, a) for a in offers))
        self.assertEqual(len(read_json(runtime.workflow.budget_path)['supporting_probes']), 1)
