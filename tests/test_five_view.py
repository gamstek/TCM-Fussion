"""Five-view interface contracts; model calls are explicit mocks, never live tests."""
import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import paths
from TCM_Fusion.lib import five_view as F
from TCM_Fusion.lib.storage import read_json, write_json, IntegrationError


class FiveViewTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root / 'source'
        self.source.mkdir()
        self.proof = {'herb': 'REFERENCE_ONLY', 'feature_id': 'TRUTH_ONLY', 'views': []}
        for i in range(5):
            p = self.source / f'original_{i}.jpg'
            p.write_bytes(f'explicit-test-placeholder-{i}'.encode())
            self.proof['views'].append({'view': p.name, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()})
        self.save()
        self.options = {'output_root': self.root / 'outputs', 'a_python': paths.VENV_PY}
        self.template = read_json(Path(paths.PROJECT_ROOT) / 'TCM_Fusion/config/five_view.json')

    def save(self):
        write_json(self.source / 'input_proof.json', self.proof)

    def test_hash_missing_and_duplicate_rejected_before_case_creation(self):
        self.proof['views'][0]['sha256'] = '0' * 64
        self.save()
        with self.assertRaises(IntegrationError):
            F.run_five_view(self.source, **self.options, dry_run=True)
        self.assertFalse(self.options['output_root'].exists())
        self.proof['views'][0] = copy.deepcopy(self.proof['views'][1])
        self.save()
        with self.assertRaises(IntegrationError) as error:
            F.inspect_sources(self.source)
        self.assertEqual(error.exception.code, 'FIVE_VIEW_DUPLICATE')

    def test_traversal_and_wrong_count_rejected(self):
        outside = self.root / 'outside.jpg'
        outside.write_bytes(b'outside')
        self.proof['views'][0]['path'] = '../outside.jpg'
        self.save()
        with self.assertRaises(IntegrationError) as error:
            F.inspect_sources(self.source)
        self.assertEqual(error.exception.code, 'FIVE_VIEW_SCOPE')
        self.proof['views'].pop()
        self.save()
        with self.assertRaises(IntegrationError):
            F.inspect_sources(self.source)

    def test_a_python_required_before_case_creation(self):
        with self.assertRaises(IntegrationError) as error:
            F.run_five_view(self.source, output_root=self.root / 'outputs', dry_run=True)
        self.assertEqual(error.exception.code, 'FIVE_VIEW_A_PYTHON_MISSING')
        self.assertFalse((self.root / 'outputs').exists())

    def test_dry_run_copies_only_sources_fresh_cache_no_labels_or_model(self):
        (self.source / 'a_cache').mkdir()
        (self.source / 'a_cache/old.json').write_text('old')
        with patch.object(F, 'doctor', return_value={'ready': True}), \
                patch.object(F, 'run_pipeline') as pipeline, patch('urllib.request.urlopen') as network:
            base = F.run_five_view(self.source, **self.options, dry_run=True)
        pipeline.assert_not_called()
        network.assert_not_called()
        cfg = read_json(base / 'config.json')
        self.assertNotIn('REFERENCE_ONLY', json.dumps(cfg))
        self.assertNotIn('TRUTH_ONLY', json.dumps(cfg))
        self.assertFalse((base / 'a_cache').exists())
        self.assertFalse((base / 'runs').exists())
        self.assertEqual(len(cfg['a']['images']), 5)
        self.assertEqual(cfg['a']['images'], cfg['agent']['image_roots'])
        self.assertEqual(cfg['agent']['transport'], 'mcp')
        self.assertNotIn('routing', cfg)
        self.assertEqual(read_json(base / 'interface_status.json')['status'], 'prepared')

    def test_live_calls_existing_pipeline_with_fresh_case_and_writes_pointer(self):
        def run(cfg, *, base):
            self.assertFalse((base / 'a_cache').exists())
            self.assertFalse((base / 'runs').exists())
            result = base / 'runs/fake-run'
            result.mkdir(parents=True)
            return result
        with patch.object(F, 'doctor', return_value={'ready': True}), patch.object(F, 'run_pipeline', side_effect=run) as call:
            result = F.run_five_view(self.source, **self.options)
        self.assertEqual(call.call_count, 1)
        base = result.parent.parent
        self.assertEqual(read_json(base / 'run_location.json')['directory'], str(result))
        self.assertEqual(read_json(base / 'interface_status.json')['status'], 'completed')

    def test_doctor_failure_is_terminal_and_persisted(self):
        with patch.object(F, 'doctor', return_value={'ready': False}), patch.object(F, 'run_pipeline') as call:
            with self.assertRaises(IntegrationError) as error:
                F.run_five_view(self.source, **self.options)
        call.assert_not_called()
        self.assertEqual(error.exception.code, 'FIVE_VIEW_NOT_READY')
        base = next(self.options['output_root'].iterdir())
        self.assertEqual(read_json(base / 'interface_status.json')['stage'], 'doctor')
        self.assertEqual(read_json(base / 'interface_status.json')['status'], 'failed')

    def test_pipeline_failure_never_claims_completion(self):
        with patch.object(F, 'doctor', return_value={'ready': True}), \
                patch.object(F, 'run_pipeline', side_effect=IntegrationError('TEST_FAILURE', 'fixture')):
            with self.assertRaises(IntegrationError):
                F.run_five_view(self.source, **self.options)
        base = next(self.options['output_root'].iterdir())
        self.assertFalse((base / 'run_location.json').exists())
        self.assertEqual(read_json(base / 'interface_status.json')['error_code'], 'TEST_FAILURE')

    def test_template_rejects_hidden_simulation_or_routing(self):
        images = [str(self.source / row['view']) for row in self.proof['views']]
        for field, value in [('routing', {'mode': 'shadow'}), ('allow_recorded', True)]:
            cfg = copy.deepcopy(self.template)
            cfg[field] = value
            with self.assertRaises(IntegrationError):
                F.bind_config(cfg, images, sample_id='test')

    def test_reference_policy_field_parity_and_explicit_defaults(self):
        ref = read_json(Path(paths.PROJECT_ROOT) / 'TCM_Fusion/运行结果/case_9cc59640/config.json')
        old = ref['agent']['workflow']
        new = self.template['agent']['workflow']
        for field, value in old.items():
            if field != 'p0_priority':
                self.assertEqual(new[field], value, field)
        from TCM_Fusion.lib.workflow_session import TOOLS
        self.assertEqual(set(new['enabled_tools']), set(TOOLS))
        self.assertNotIn('p0_priority', new)
        self.assertEqual(new['glm_identity_threshold'], .8)
        self.assertEqual(new['glm_locate_threshold'], .8)
        self.assertEqual(new['locate_expand'], 1.5)
        self.assertNotIn('python', self.template['a'])
        self.assertNotIn('case_9cc59640', json.dumps(self.template))
