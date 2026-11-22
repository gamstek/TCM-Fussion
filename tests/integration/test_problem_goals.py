"""用真实 ReAct、证据库和预算，验证尚无实例的鉴别问题在同一个语义 DAG 中完成或回交。

主要定义
- VesselStages 产生可控的补扫、身份、掩码和几何测量工件。
- ProblemGoalTests 对拍实例目标与问题目标，并通过真实送模序列化检查授权和回执。

关联
- selftest 自动发现。HTTP 与视觉后端为离线替身，使用 tests/support/workflow_fixtures 的运行入口。
"""
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PIL import Image
import contracts
from TCM_Fusion.lib.agent.planning_choices import attach, decode
from TCM_Fusion.lib.agent.react_graph import LLMPlanner
from TCM_Fusion.lib.agent.semantic_execution import compile_problem, next_action
from TCM_Fusion.lib.runtime.storage import IntegrationError, read_json, write_json
from TCM_Fusion.lib.workflow.workflow_runtime import WorkflowRuntime
from TCM_Fusion.tests.support.workflow_fixtures import D3Fixture, StageFixture


class VesselStages(StageFixture):
    def __init__(self, *, count=1, partial=False, unknown_coverage=False, wrong_dimension=False,
                 granularity='instance', **kwargs):
        super().__init__(**kwargs)
        self.count = count
        self.partial = partial
        self.unknown_coverage = unknown_coverage
        self.wrong_dimension = wrong_dimension
        self.granularity = granularity

    def execute(self, session, tool, target, work, context):
        if tool == 'feature_scan':
            self.calls.append(tool)
            if tool == self.fail:
                raise IntegrationError('SIMULATED_FAILURE', '离线补扫故障')
            folder = work / 'scan/crops'
            folder.mkdir(parents=True)
            source = session.runtime.images[target['image_id']]['path']
            for i in range(self.count):
                name = f'vessel-{i}'
                x = 15 + i * 55
                pixels = Image.frombytes('L', (48, 48), bytes((j * 13) % 251 for j in range(48*48)))
                pixels.save(folder / (name + '.png'))
                meta = contracts.crop_sidecar(name, name + '.png', str(source), source.stem,
                    'vessel', '导管', '', 2, [x, 15, x+48, 63], [0, 0, 48, 48], [x, 15, x+48, 63], '48x48')
                write_json(folder / (name + '.json'), meta)
            completion = folder.parent / 'completion.json'
            result = {'count': self.count, 'feature_id': 'vessel', 'partial': self.partial}
            if not self.unknown_coverage:
                result['tiles'] = [[0, 0, 128, 128]]
            write_json(completion, result)
            return {'crops_dir': str(folder), 'completion': str(completion)}
        result = super().execute(session, tool, target, work, context)
        if tool == 'classify_target':
            path = Path(result['classification']) / 'classification.json'
            write_json(path, {**read_json(path), 'granularity': self.granularity})
        if tool == 'detect_fov_initial':
            side = Path(result['crops_dir']) / 'crop.json'
            meta = read_json(side)
            source = session.runtime.images[target['image_id']]['path']
            meta.update(source_image_path=str(source), stem=source.stem)
            write_json(side, meta)
        if tool == 'measure_instances':
            path = Path(result['measurements'])
            row = json.loads(path.read_text(encoding='utf-8'))
            dim = 'feret_max_um' if self.wrong_dimension else 'median_width_um'
            row.update(feature='vessel', primary_dimension=dim, primary_value_um=19.5, mean_width_um=21)
            path.write_text(json.dumps(row) + '\n', encoding='utf-8')
        return result


class ProblemGoalTests(D3Fixture, unittest.TestCase):
    def prepared(self, name, backend=None, images=None, **config):
        runtime = WorkflowRuntime(images or [self.image], root=self.root/name, sample_id=name,
            a={'status': 'ok', 'ranking': [{'label': '白芍', 'probability': .8}]},
            simulation=True, backend=backend or VesselStages(), workflow_config={
                'agent_policy': 'adaptive_v2', 'semantic_execution': True, 'problem_goal_execution': True,
                'fast_recovery_selection': 'per_feature', 'protect_investigation_budget': False,
                'supporting_probe_total': 8, 'supporting_probe_scan_total': 3, 'context_compact': True,
                'max_planning': 12, 'max_resource_units': 150, 'max_recompute': 12, **config})
        result = runtime.workflow.execute(self.action(runtime, 'detect_fov_initial'))
        self.assertEqual(result['status'], 'done')
        return runtime

    def select_problem(self, runtime, *, expected_entry='feature_scan'):
        seen = []

        class Planner:
            def choose(_, ctx):
                seen.append(ctx)
                if len(seen) == 1:
                    option = next(p for p in ctx['execution_options']['problem_goals']
                                  if p['feature_id'] == 'vessel' and p['entry_tool'] == expected_entry)
                    return {'decision': {'kind': 'problem_goal', 'goal_id': option['choice_id']}}, None
                return {'action_id': 'review'}, None

        trace = self.run_agent(runtime, Planner())
        self.assertEqual(trace['planning_requests'], 2, trace['stop_reason'])
        event = next(e for e in trace['events'] if e['phase'] == 'semantic_return')
        self.assertEqual(seen[1]['last_problem_result'], event['problem_result'])
        return event

    def test_serialized_choice_and_paired_instance_baseline(self):
        r = self.prepared('question', auto_measure_after_segment=True)
        menus = []
        planner = LLMPlanner.__new__(LLMPlanner)
        planner.client = SimpleNamespace(model='offline-fixture', timeout_s=1)
        planner.config = {}

        def post(payload, timeout):
            menu = json.loads(payload['messages'][1]['content'])
            menus.append(menu)
            if len(menus) == 1:
                offer = next(o for o in menu['execution_options']['problem_goals'] if o['feature_id'] == 'vessel')
                self.assertRegex(offer['choice_id'], r'^P[0-9]+$')
                self.assertRegex(offer['entry'], r'^A[0-9]+$')
                choice = {'kind': 'problem_goal', 'goal_id': offer['choice_id']}
            else:
                choice = {'kind': 'action', 'action_id': next(a['action_id'] for a in menu['actions']
                                                             if a['tool'] == 'request_review')}
            return {'choices': [{'message': {'content': json.dumps({'decision': choice})}}]}
        with patch('TCM_Fusion.lib.runtime.hard_calls.post', side_effect=post):
            trace = self.run_agent(r, planner)
        self.assertEqual(trace['planning_requests'], 2, trace['stop_reason'])
        plan = read_json(r.workflow.root / 'semantic_plan.json')
        self.assertEqual(plan['return_reason'], 'local_goal_satisfied', plan)
        self.assertEqual(plan['facts'][0]['value_um'], 19.5)
        self.assertTrue(all(n['status'] == 'succeeded' for n in plan['nodes']))
        self.assertEqual(plan['nodes'][1]['depends_on'], ['resolve_instance'])
        self.assertEqual(menus[1]['last_problem_result']['instance_ref'], plan['goal']['instance_ref'])

        old = self.prepared('instance', problem_goal_execution=False, auto_measure_after_segment=True)
        turns = []
        class InstancePlanner:
            def choose(_, ctx):
                turns.append(ctx)
                self.assertNotIn('problem_goals', ctx['execution_options'])
                if len(turns) == 1:
                    action = next(a for a in ctx['actions'] if a['tool'] == 'feature_scan'
                                  and a['args']['target_id'].endswith(':vessel'))
                    return {'action_id': action['action_id']}, None
                if len(turns) == 2:
                    g = next(g for g in ctx['execution_options']['goals'] if g['feature_id'] == 'vessel'
                             and g['goal']['semantic_task'] == 'measure_width')
                    return {'decision': {'kind': 'instance_goal', 'goal_id': g['goal_id']}}, None
                return {'action_id': 'review'}, None
        baseline = self.run_agent(old, InstancePlanner())
        self.assertEqual(baseline['planning_requests'], 3)
        self.assertEqual(old.workflow.backend.calls, r.workflow.backend.calls)
        self.assertEqual(r.workflow.backend.calls, ['detect_fov_initial', 'feature_scan', 'classify_target',
                                                   'segment_instance', 'measure_instances'])
        self.assertEqual(old.workflow.store.verified_records(), r.workflow.store.verified_records())

    def test_existing_measurement_reused_without_tool_or_new_evidence(self):
        r = self.prepared('reuse')
        scan = next(a for a in r.workflow.propose() if a['tool'] == 'feature_scan'
                    and a['args']['target_id'].endswith(':vessel'))
        self.assertTrue(r.workflow.execute(scan)['accepted'])
        for tool in ('classify_target', 'segment_instance', 'measure_instances'):
            action = next(a for a in r.workflow.propose() if a['tool'] == tool and 'vessel-0' in a['args']['target_id'])
            self.assertTrue(r.workflow.execute(action)['accepted'])
        before = list(r.workflow.backend.calls), r.workflow.store.current()['version']
        result = self.select_problem(r, expected_entry='reuse')
        self.assertEqual(result['problem_result']['facts'][0]['value_um'], 19.5)
        self.assertEqual(result['plan']['issued'], {})
        self.assertEqual((r.workflow.backend.calls, r.workflow.store.current()['version']), before)

    def test_empty_partial_failure_or_multiple_return_without_classification(self):
        for name, stage, reason in [
                ('empty', VesselStages(count=0), 'scan_empty'),
                ('partial', VesselStages(partial=True), 'scan_partial'),
                ('unknown', VesselStages(count=0, unknown_coverage=True), 'scan_coverage_unknown'),
                ('failed', VesselStages(fail='feature_scan'), 'SIMULATED_FAILURE'),
                ('multiple', VesselStages(count=2), 'ambiguous_instances')]:
            with self.subTest(name=name):
                r = self.prepared(name, stage)
                event = self.select_problem(r)
                self.assertEqual(event['reason'], reason)
                self.assertFalse(event['problem_result']['facts'])
                self.assertEqual(stage.calls, ['detect_fov_initial', 'feature_scan'])

    def test_identity_conflict_or_wrong_primary_does_not_produce_evidence(self):
        for name, stage in [('fiber', VesselStages(classification_feature='fiber')),
                            ('secondary_width', VesselStages(wrong_dimension=True))]:
            with self.subTest(name=name):
                r = self.prepared(name, stage)
                event = self.select_problem(r)
                self.assertEqual(event['plan']['status'], 'returned_to_react')
                self.assertEqual(event['problem_result']['facts'], [])
                if name == 'fiber':
                    self.assertNotIn('segment_instance', stage.calls)
                else:
                    self.assertEqual(event['reason'], 'requested_dimension_unavailable')

    def test_scope_cancellation_stale_and_mixed_choices(self):
        r = self.prepared('stale')
        state = self.state()
        ctx = attach(r, {'actions': r.workflow.propose()}, can_pair=False)
        offer = next(p for p in ctx['execution_options']['problem_goals'] if p['feature_id'] == 'vessel')
        plan = compile_problem(offer)
        with self.assertRaises(IntegrationError):
            decode({'decision': {'kind': 'problem_goal', 'goal_id': offer['choice_id']},
                    'parallel_question_id': 'Q1'}, ctx)
        with self.assertRaises(IntegrationError):
            decode({'decision': {'kind': 'problem_goal', 'goal_id': 'not-offered'}}, ctx)
        r.workflow.cancel_event.set()
        self.assertEqual(next_action(r, plan, state), (None, 'task_cancelled'))
        r.workflow.cancel_event.clear()
        r.workflow.store.commit(offer['version'], [])
        self.assertEqual(next_action(r, plan, state), (None, 'evidence_changed_before_dispatch'))
        self.assertEqual(r.workflow.backend.calls, ['detect_fov_initial'])

    def test_paid_measurement_requires_a_beneficiary_of_this_problem(self):
        runtime = self.prepared('measurement-gain')
        for tool in ('feature_scan', 'classify_target'):
            step = next(a for a in runtime.workflow.propose() if a['tool'] == tool
                        and 'vessel' in a['args']['target_id'])
            self.assertTrue(runtime.workflow.execute(step)['accepted'])
        entry = next(a for a in runtime.workflow.propose() if a['tool'] == 'segment_instance'
                     and 'vessel' in a['args']['target_id'])
        bindings = {entry['action_id']: 'vessel-group'}
        visible = {'actions': [entry]}
        initial = attach(runtime, visible, can_pair=False)['execution_options']['problem_goals'][0]

        def group_for(candidate):
            return {'dimensions': {initial['dimension']: {'n_measured': 2, 'worth_measuring': True,
                    'consumers': [{'candidate_id': candidate, 'reasons': ['resolve_split_instances']}]}}}

        groups = {'vessel-group': group_for(initial['consumers'][0])}
        allowed = attach(runtime, {**visible, 'measurement_groups': groups}, can_pair=False,
                         measurement_actions=bindings)['execution_options']['problem_goals']
        offer = next(p for p in allowed if p['problem_id'] == initial['problem_id'])
        groups = {'vessel-group': group_for('outside-this-problem')}
        filtered = attach(runtime, {**visible, 'measurement_groups': groups}, can_pair=False,
                          measurement_actions=bindings)
        self.assertFalse(filtered['execution_options'].get('problem_goals'))
        before = list(runtime.workflow.backend.calls)
        with patch('TCM_Fusion.lib.agent.measurement_gain.measurement_menu',
                   return_value=([entry], groups, bindings)):
            self.assertEqual(next_action(runtime, compile_problem(offer), self.state()),
                             (None, 'measurement_value_changed'))
        self.assertEqual(runtime.workflow.backend.calls, before)

    def test_existing_candidate_binds_without_another_scan(self):
        r = self.prepared('existing')
        scan = next(a for a in r.workflow.propose() if a['tool'] == 'feature_scan'
                    and a['args']['target_id'].endswith(':vessel'))
        self.assertTrue(r.workflow.execute(scan)['accepted'])
        event = self.select_problem(r, expected_entry='classify_target')
        self.assertEqual(set(event['plan']['issued']), {'classify_target', 'segment_instance', 'measure_instances'})
        self.assertEqual(event['reason'], 'local_goal_satisfied')
        self.assertEqual(r.workflow.backend.calls.count('feature_scan'), 1)

    def test_other_view_cannot_bind_or_satisfy_empty_view(self):
        other = self.root/'other.png'
        other.write_bytes(b'second-offline-view')
        r = self.prepared('two', images=[self.image, other], supporting_probe_scan_per_candidate=2)
        self.assertTrue(r.workflow.execute(self.action(r, 'detect_fov_initial'))['accepted'])
        ids = list(r.images)
        first = next(a for a in r.workflow.propose() if a['tool'] == 'feature_scan'
                     and a['args']['target_id'] == 'feature_scan:' + ids[0] + ':vessel')
        self.assertTrue(r.workflow.execute(first)['accepted'])
        r.workflow.backend.count = 0
        result = self.select_problem(r)
        self.assertEqual(result['problem_result']['image_id'], ids[1])
        self.assertEqual(result['reason'], 'scan_empty')
        self.assertNotIn('classify_target', r.workflow.backend.calls)

    def test_step_cap_precedes_automatic_measurement(self):
        r = self.prepared('step-cap', auto_measure_after_segment=True)
        class Planner:
            def choose(_, ctx):
                p = next(p for p in ctx['execution_options']['problem_goals'] if p['feature_id'] == 'vessel')
                return {'decision': {'kind': 'problem_goal', 'goal_id': p['choice_id']}}, None
        trace = self.run_agent(r, Planner(), max_steps=3)
        self.assertEqual(trace['stop_reason'], 'max_steps')
        self.assertEqual(r.workflow.backend.calls, ['detect_fov_initial', 'feature_scan', 'classify_target', 'segment_instance'])
        self.assertEqual(read_json(r.workflow.root / 'semantic_plan.json')['return_reason'], 'max_steps')

    def test_purpose_policy_needs_width_qualification_even_with_valid_segmentation(self):
        r = self.prepared('width-qa', auto_measure_after_segment=True, quality_evaluation='purpose_v1')
        event = self.select_problem(r)
        self.assertEqual(event['reason'], 'dimension_not_qualified')
        self.assertFalse(event['problem_result']['facts'])
        self.assertIn('segment_instance', r.workflow.backend.calls)
        self.assertNotIn('measure_instances', r.workflow.backend.calls)

    def test_verified_vessel_bundle_does_not_measure_an_individual_width(self):
        r = self.prepared('bundle', VesselStages(granularity='bundle'))
        event = self.select_problem(r)
        self.assertEqual(event['reason'], 'granularity_not_qualified')
        self.assertNotIn('segment_instance', r.workflow.backend.calls)

    def test_instance_and_direct_routes_respect_reviewed_granularity(self):
        from TCM_Fusion.lib.agent.semantic_execution import compile_goal
        for grain in ('instance', 'bundle'):
            with self.subTest(grain=grain):
                r = self.prepared(grain, VesselStages(granularity=grain), auto_measure_after_segment=True)
                scan = next(a for a in r.workflow.propose() if a['tool'] == 'feature_scan'
                            and a['args']['target_id'].endswith(':vessel'))
                self.assertTrue(r.workflow.execute(scan)['accepted'])
                classify = next(a for a in r.workflow.propose() if a['tool'] == 'classify_target'
                                and 'vessel-0' in a['args']['target_id'])
                offers = attach(r, {'actions': r.workflow.propose()}, can_pair=False)
                goal = next(g['goal'] for g in offers['execution_options']['goals']
                            if g['feature_id'] == 'vessel' and g['goal']['semantic_task'] == 'measure_width')
                plan = compile_goal(r, goal, classify)
                self.assertTrue(r.workflow.execute(classify)['accepted'])
                if grain == 'bundle':
                    self.assertEqual(next_action(r, plan, self.state())[1], 'granularity_not_qualified')
                segment = next(a for a in r.workflow.propose() if a['tool'] == 'segment_instance'
                               and 'vessel-0' in a['args']['target_id'])
                self.assertTrue(r.workflow.execute(segment)['accepted'])
                record, = r.workflow.store.verified_records()
                self.assertEqual('coarse_quant' in record['flags'], grain == 'bundle')
                self.assertEqual(bool(r.score([])['cards']['vessel'].dims), grain == 'instance')

    def test_late_bundle_classification_revokes_previously_measured_dimension(self):
        r = self.prepared('late-bundle', StageFixture(), auto_measure_after_segment=True)
        workflow = r.workflow
        crop = next(v for v in workflow.targets().values() if v['kind'] == 'crop')
        segment = workflow.backend.execute(workflow, 'segment_instance', crop, self.root/'early-seg', None)
        workflow.accept('segment_instance', crop, segment, workflow.store.current()['version'])
        previous, = workflow.store.verified_records()
        self.assertNotIn('coarse_quant', previous.get('flags', []))
        identity = self.root/'late-classification'
        write_json(identity/'classification.json', {'feature_id':'single_crystal', 'label':'方晶',
                   'confidence':'high', 'granularity':'clump'})
        workflow.accept('classify_target', crop, {'classification':str(identity)},
                        workflow.store.current()['version'])
        revised, = workflow.store.verified_records()
        self.assertIn('coarse_quant', revised['flags'])
        self.assertFalse(r.score([])['cards']['single_crystal'].dims)

    def test_scope_rechecks_source_and_calibration(self):
        r = self.prepared('read-set')
        ctx = attach(r, {'actions': r.workflow.propose()}, can_pair=False)
        offer = next(p for p in ctx['execution_options']['problem_goals'] if p['feature_id'] == 'vessel')
        plan = compile_problem(offer)
        original = self.image.read_bytes()
        self.image.write_bytes(b'other-source')
        self.assertEqual(next_action(r, plan, self.state()), (None, 'source_changed'))
        self.image.write_bytes(original)
        scale = r.workflow.store.current()['artifacts']['scale:' + offer['image_id']]
        write_json(scale['path'], {'px_per_um': 3})
        self.assertEqual(next_action(r, plan, self.state()), (None, 'calibration_changed'))

    def test_issue_menu_accounts_for_each_scan_tile(self):
        r = self.prepared('tiles', feature_scan_max_tiles=4)
        actions = r.workflow.propose()
        self.assertTrue(any(a['tool'] == 'feature_scan' and a['args']['target_id'].endswith(':vessel') for a in actions))
        for network, offered in [(3, False), (4, True)]:
            ctx = attach(r, {'actions': actions, 'remaining_budget': {'network': network, 'total': 5, 'recompute': 1}},
                         can_pair=False)
            self.assertEqual(any(o['feature_id'] == 'vessel' for o in ctx['execution_options'].get('problem_goals', [])), offered)
