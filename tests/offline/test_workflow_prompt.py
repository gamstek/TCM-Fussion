"""Offline LangMem/prompt-release workflow test, excluded from main-chain tests."""
import asyncio
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from TCM_Fusion.lib.agent_tools import LocalToolbox, FixtureBackend
from TCM_Fusion.lib.react_graph import run_react
from TCM_Fusion.lib.storage import digest, read_json
from TCM_Fusion.lib.workflow_runtime import WorkflowRuntime
from TCM_Fusion.tests.test_workflow import StageFixture


class WorkflowPromptTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.image = self.root / 'fov.png'
        self.image.write_bytes(b'simulation-fov')

    def runtime(self, backend=None, **config):
        return WorkflowRuntime(
            [self.image], root=self.root / 'task', sample_id='sample', a={},
            workflow_config={'max_recompute': 8, **config}, simulation=True,
            backend=backend or StageFixture())

    def test_workflow_prompt_dual_replay_and_release(self):
        from TCM_Fusion.lib import prompt_optimization as P
        r = self.runtime(max_planning=30, max_resource_units=200)
        tools = LocalToolbox(r, ledger_path=self.root / 'recorded_obs.json',
                             backend=FixtureBackend(), max_requests=100)
        trace = asyncio.run(run_react(r, tools, planner_mode='heuristic',
                                      max_requests=100, max_steps=100))
        report = {'route': 'workflow_agent', 'simulation': True, 'a': {}, 'sample_id': 'sample',
                  'workflow_options': r.workflow.config, 'agent': {
                      'operations': read_json(r.workflow.operations)['entries'],
                      'observations': trace['observations']},
                  'b': {'verdict': {**trace['decision'], 'source_images': [str(self.image)]}}}
        candidate = {'schema': 'prompt-candidate/v1', 'generator': 'unit-test-only',
                     'instruction': 'choose without extra read',
                     'baseline': P.DEFAULT_INSTRUCTION, 'guard_sha256': digest(P.GUARD),
                     'training_groups': ['train'], 'training_sha256': 'fixture',
                     'profile': P.profile(), 'simulation': True}
        expected = {'verdict': trace['decision']['verdict'],
                    'conclusion_id': (trace['decision'].get('conclusion') or {}).get('id')}
        dataset = {'split': 'development', 'mode': 'workflow', 'simulation': True, 'cases': [{
            'group_id': 'development', 'report': report, 'report_sha256': digest(report),
            'expected': expected}]}

        class Planner:
            def __init__(self, instruction):
                self.extra = instruction == P.DEFAULT_INSTRUCTION

            def choose(self, context):
                selected = context['actions'][0]
                if self.extra:
                    selected = next(a for a in context['actions'] if a['tool'] == 'read_evidence')
                    self.extra = False
                return {'action_id': selected['action_id']}, None

        with patch('TCM_Fusion.lib.workflow_prompt_validation.output_root', return_value=self.root), \
             patch('TCM_Fusion.lib.prompt_optimization.output_root', return_value=self.root):
            validation = asyncio.run(P.validate(candidate, dataset, planner_factory=Planner))
            self.assertTrue(validation['passed'])
            self.assertEqual(validation['new_image_api_calls'], 0)
            version = asyncio.run(P.publish(candidate, validation, planner_factory=Planner))
            self.assertTrue(P.load_release(version, r)['simulation'])
            truth_data = copy.deepcopy(dataset)
            truth_data['mode'] = 'workflow_truth'
            truth_data['cases'][0]['reference'] = {
                'label': 'independent_fixture_truth', 'label_source': 'test_fixture',
                'label_scope': 'sample_identity'}
            truth_validation = asyncio.run(P.validate(candidate, truth_data, planner_factory=Planner))
            self.assertEqual(truth_validation['metric'], 'paired_truth_with_bounded_cost')
            self.assertIn('outcome', truth_validation['rows'][0]['candidate'])
            bad = copy.deepcopy(dataset)
            ops = bad['cases'][0]['report']['agent']['operations']
            for key in list(ops):
                if ops[key]['tool'] == 'segment_instance':
                    del ops[key]
            bad['cases'][0]['report_sha256'] = digest(bad['cases'][0]['report'])
            self.assertFalse(asyncio.run(P.validate(candidate, bad, planner_factory=Planner))['passed'])
