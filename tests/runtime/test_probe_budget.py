"""支持性探测预算与 presence 上下文压缩的离线回归，不调用任何模型 API。
该组用例由项目根 selftest 发现。涉及模型回复的用例使用可控输入，涉及进程的用例以各断言规定的执行边界为准。

主要定义：
- ProbeBudgetTests（零消耗提议）: 反复生成 presence 动作不会改动预算账本。
- ProbeBudgetTests（上下文压缩）: compact_round_context 去掉 artifacts/transform 与原始 metrics，保留空间重叠摘要且不改原对象。
- ProbeBudgetTests（共享探测）: 同图同特征的探测在不同候选间共享 key 与配额，第二次预约返回 PROBE_ALREADY_ATTEMPTED。
- ProbeBudgetTests（并发原子性）: 两个竞争调用者只能有一个成功预约，账本只记录一次。
- ProbeBudgetTests（分池与旧配置）: 真实送模菜单、共享发起者和单次预留对拍。

关联：
- 直接依赖 「lib/agent/react_graph.py」、「lib/evidence/search_ledger.py」、「lib/runtime/probe_budget.py」、「lib/runtime/resource_budget.py」、「lib/runtime/storage.py」、「lib/tools/agent_tools.py」、「lib/workflow/workflow_runtime.py」、「tests/support/workflow_fixtures.py」。
- 统一执行见 tests/README.md。
"""
import json
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from TCM_Fusion.lib.tools.agent_tools import FixtureBackend, LocalToolbox
from TCM_Fusion.lib.runtime.probe_budget import describe, check_reservation
from TCM_Fusion.lib.evidence.search_ledger import render_shared
from TCM_Fusion.lib.decision.candidate_audit import investigation_opportunities, protect_investigation_budget
from TCM_Fusion.lib.agent.react_graph import LLMPlanner, compact_action_menu, compact_round_context, run_react
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

    def scan_runtime(self, name='scan-workflow', **config):
        class EmptyScan(StageFixture):
            def execute(stage, session, tool, target, work, context):
                if tool == 'feature_scan':
                    folder = work / 'output/crops'; folder.mkdir(parents=True)
                    completion = folder.parent / 'completion.json'
                    write_json(completion, {'count': 0, 'feature_id': target.get('feature_id')})
                    return {'crops_dir': str(folder), 'completion': str(completion)}
                return super().execute(session, tool, target, work, context)
        return WorkflowRuntime(
            [self.image], root=self.root / name, sample_id='scan-budget',
            a=self.a, simulation=True, backend=EmptyScan(), workflow_config={
                'agent_policy': 'adaptive_v2', 'supporting_probe_total': 1,
                'supporting_probe_per_candidate': 2, 'fast_yolo_stage2': True,
                'fast_recovery_selection': 'per_feature', 'fast_fallback_max_features': 4,
                'fast_fallback_max_locate_calls': 16, 'feature_scan_enabled': True,
                'max_api_calls': 40, 'max_resource_units': 100, **config})

    def test_disabled_feature_scan_is_absent_from_menu(self):
        runtime = self.scan_runtime(feature_scan_enabled=False)
        self.initialize(runtime)
        self.assertFalse(any(a['tool'] == 'feature_scan' for a in runtime.workflow.propose()))

    def initialize(self, runtime):
        action = next(a for a in runtime.workflow.propose() if a['tool'] == 'detect_fov_initial')
        self.assertEqual(runtime.workflow.execute(action)['status'], 'done')

    def scan_actions(self, runtime):
        return {a['args']['target_id'].rsplit(':', 1)[-1]: a for a in runtime.workflow.propose()
                if a['tool'] == 'feature_scan'}

    async def test_presence_and_feature_scan_share_quota(self):
        runtime = self.scan_runtime()
        self.initialize(runtime)
        self.assertIn('starch_grain', self.scan_actions(runtime))
        tools = LocalToolbox(runtime, ledger_path=self.root / 'shared-quota.json', backend=FixtureBackend())
        presence = next(a for a in runtime.legal_actions({'observations': []})
                        if a['tool'] == 'presence_check' and describe(runtime, a))
        result = await tools.observe(presence)
        self.assertTrue(result['accepted'])
        offers = runtime.legal_actions({'observations': [result], 'attempted': [presence['action_id']]})
        self.assertFalse(any(a['tool'] == 'feature_scan' and describe(runtime, a) for a in offers))
        self.assertEqual(len(read_json(runtime.workflow.budget_path)['supporting_probes']), 1)

    async def test_split_scan_keeps_other_candidate_and_charges_one_request(self):
        runtime = self.scan_runtime(supporting_probe_total=6, supporting_probe_scan_total=2)
        self.initialize(runtime)
        for feature, payer in [('starch_grain', 'G1'), ('vessel', '白芍')]:
            action = self.scan_actions(runtime)[feature]
            result = runtime.workflow.execute(action)
            self.assertEqual(result['status'], 'done')
            probe = next(row for row in read_json(runtime.workflow.budget_path)['supporting_probes'].values()
                         if row['feature_id'] == feature)
            self.assertEqual(probe['candidate_ids'], ['G1', '白芍'])
            self.assertEqual(probe['charged_candidate_ids'], [payer])
        book = read_json(runtime.workflow.budget_path)
        self.assertEqual(len(book['supporting_probes']), 2)
        self.assertEqual(book['used']['recompute'], 2)
        self.assertNotIn('starch_grain', self.scan_actions(runtime))
        self.assertNotIn('vessel', self.scan_actions(runtime))
        self.assertIn('cluster_crystal', self.scan_actions(runtime))  # required is a separate route

    async def test_weak_probe_cannot_spend_last_required_path(self):
        runtime = self.scan_runtime(supporting_probe_total=6, supporting_probe_scan_total=2)
        self.initialize(runtime)
        state = {'observations': [], 'attempted': []}
        actions = runtime.legal_actions(state)
        view = runtime.decision_view(state | {'actions': actions})
        opportunities = investigation_opportunities(runtime, view['candidate_audit'],
            view['investigation_focus'], actions, runtime.workflow.limits)
        required = next(row for row in opportunities if row['role'] == 'required')
        remaining = runtime.workflow.limits | {'network': required['cost']['network']}
        kept, deferred = protect_investigation_budget(runtime, actions, opportunities, remaining)
        self.assertIn(required['action_id'], {a['action_id'] for a in kept})
        self.assertTrue(any(r['tool'] == 'presence_check' and r['feature_id'] == 'cork_cell'
                            for r in deferred))
        runtime.workflow.config['protect_investigation_budget'] = False
        self.assertEqual(protect_investigation_budget(runtime, actions, opportunities, remaining),
                         (actions, []))

    async def test_parallel_supporting_observations_share_remaining_slots(self):
        for slots in (1, 2):
            with self.subTest(slots=slots):
                name = f'paired-probes-{slots}'
                runtime = self.scan_runtime(name, supporting_probe_total=2 + slots,
                                            supporting_probe_scan_total=2,
                                            protect_investigation_budget=False)
                backend = FixtureBackend()
                tools = LocalToolbox(runtime, ledger_path=self.root / (name + '.json'), backend=backend)
                calls = []

                class Planner:
                    def choose(self, context):
                        calls.append(context)
                        if len(calls) > 1:
                            return {'action_id': 'review'}, None
                        questions = {r['feature_id']: r['question_id']
                                     for r in context['execution_options']['questions']}
                        return {'decision': {'kind': 'observe_questions',
                            'question_ids': [questions['fiber'], questions['cork_cell']]}}, None

                trace = await run_react(runtime, tools, planner=Planner(), planner_mode='scripted',
                                        max_steps=8)
                self.assertEqual(trace['planning_requests'], 2)
                self.assertEqual(len(backend.calls), slots)
                self.assertEqual(len(trace['observations']), slots)
                self.assertTrue(all(o['accepted'] for o in trace['observations']))
                probes = read_json(runtime.workflow.budget_path)['supporting_probes'].values()
                self.assertEqual({p['feature_id'] for p in probes},
                                 {'fiber'} if slots == 1 else {'fiber', 'cork_cell'})
                rejected = [e for e in trace['events'] if e['phase'] == 'parallel_question_rejected']
                if slots == 1:
                    self.assertEqual(len(rejected), 1)
                    self.assertEqual(rejected[0]['reason'], 'SUPPORTING_POOL_BUDGET')
                    self.assertEqual(rejected[0]['new_planning_requests'], 0)
                else:
                    self.assertFalse(rejected)

    async def test_split_preserves_both_p0_supporting_actions_in_serialized_model_menu(self):
        comparison = {}
        scan_features = {'starch_grain', 'vessel'}
        for name, config in [('legacy', {}), ('split', {'supporting_probe_scan_total': 2})]:
            runtime = self.scan_runtime(name, supporting_probe_total=6, context_compact=True, **config)
            planner = LLMPlanner.__new__(LLMPlanner)
            planner.client = SimpleNamespace(model='offline-fixture', timeout_s=1)
            planner.config = {}
            menus, request_bytes = [], []
            def post(payload, timeout):
                content = payload['messages'][1]['content']
                request_bytes.append(len(content.encode('utf-8')))
                ctx = json.loads(content)
                menus.append(ctx)
                actions = ctx['actions']
                turn = len(menus)
                if turn == 1:
                    action = next(a for a in actions if a['tool'] == 'detect_fov_initial')
                elif turn in (2, 3):
                    feature = 'cork_cell' if turn == 2 else 'fiber'
                    action = next(a for a in actions if a['tool'] == 'presence_check'
                                  and a['args']['feature_id'] == feature)
                else:
                    action = next(a for a in actions if a['tool'] == 'request_review')
                reply = {'decision': {'kind': 'action', 'action_id': action['action_id']},
                         'decision_summary': '离线固定输入，用于比较工具准入。'}
                return {'choices': [{'message': {'content': json.dumps(reply)}}]}
            tools = LocalToolbox(runtime, ledger_path=self.root / (name + '-observations.json'),
                                 backend=FixtureBackend())
            started = time.monotonic()
            with patch('TCM_Fusion.lib.runtime.hard_calls.post', side_effect=post) as transport, patch(
                 'TCM_Fusion.lib.agent.react_graph.model_action_menu',
                 side_effect=lambda runtime, actions, limit, **kwargs: compact_action_menu(actions, limit, **kwargs)):
                trace = await run_react(runtime, tools, planner=planner, planner_mode='scripted', max_steps=8)
            seconds = time.monotonic() - started
            self.assertEqual(transport.call_count, 4)
            self.assertEqual(trace['observation_requests'], 2)
            self.assertEqual(trace['stop_reason'], 'planner_request_review')
            self.assertTrue(all(o['accepted'] and not o['result']['present'] for o in trace['observations']))
            self.assertFalse(runtime.workflow.store.verified_records())
            audit = trace['candidate_audit']
            self.assertEqual(audit['candidate_ids'], ['白芍', 'G1'])
            scans = {a['args']['target_id'].rsplit(':', 1)[-1] for a in menus[-1]['actions']
                     if a['tool'] == 'feature_scan'}
            self.assertIn('cluster_crystal', scans)
            self.assertEqual(scans & scan_features, scan_features if name == 'split' else set())
            offer = [event for event in trace['events'] if event['phase'] == 'planner_offer'][-1]
            self.assertTrue(offer['actual_llm_short_ids'])
            self.assertEqual(offer['menu']['actions'], menus[-1]['actions'])
            opportunities = offer['menu']['investigation_opportunities']
            if name == 'split':
                self.assertTrue(scan_features <= {r['feature_id'] for r in opportunities
                                                  if r['role'] == 'supporting'})
                self.assertTrue(any(r['role'] == 'required' and r['feature_id'] == 'cluster_crystal'
                                    for r in opportunities))
                self.assertTrue(any(e['phase'] == 'review_with_open_opportunities' for e in trace['events']))
            else:
                self.assertFalse(opportunities)
            self.assertTrue(all(event.get('usage') is None for event in trace['events']
                                if event['phase'] == 'thought'))
            comparison[name] = {'planning_requests': trace['planning_requests'], 'seconds': seconds,
                'sent_scan_features': sorted(scans), 'required': {r['candidate_id']: r['required'] for r in audit['candidates']},
                'supporting': {r['candidate_id']: r['supporting'] for r in audit['candidates']},
                'new_positive_criterion_links': sum(state == 'supported' for row in audit['candidates']
                    for role in ('required', 'supporting') for state in row[role].values()),
                'new_admitted_measurements': 0, 'last_context_bytes': request_bytes[-1],
                'sum_context_bytes': sum(request_bytes), 'usage': None}
        self.assertEqual(comparison['legacy']['required'], comparison['split']['required'])
        self.assertEqual(comparison['legacy']['supporting'], comparison['split']['supporting'])
        self.comparison = comparison
