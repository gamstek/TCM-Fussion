"""多图接口契约回归：源图校验、case 目录隔离、dry-run 与真实 pipeline 指针，模型调用均为显式替身。
该组用例由项目根 selftest 发现。涉及模型回复的用例使用可控输入，涉及进程的用例以各断言规定的执行边界为准。

主要定义：
- MultiViewTests（源图校验）: SHA 缺失/重复、路径穿越、数量不符在创建 case 前被拒绝。
- MultiViewTests（数量与解释器）: 图片数量不固定为五张；a_python 缺失时给出明确错误码。
- MultiViewTests（case 命名）: 验证北京时间命名、同秒隔离、元数据和旧覆盖参数边界。
- MultiViewTests（dry-run）: 只复制源图，不建缓存、不跑 pipeline，配置中不出现参考标签。
- MultiViewTests（真实执行）: 调用现有 pipeline、写 run_location 与 interface_status。
- MultiViewTests（失败路径）: doctor 失败与 pipeline 异常都持久化终态且不宣称完成。
- MultiViewTests（模板与参考对齐）: 拒绝隐藏 routing/allow_recorded，并核对历史参考模板的字段取值。

关联：
- 直接依赖 「lib/application/multi_view.py」、「lib/runtime/storage.py」、「lib/workflow/workflow_session.py」。
- 统一执行见 tests/README.md。
"""
import copy
import hashlib
import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import paths
from TCM_Fusion.lib.application import multi_view as F
from TCM_Fusion.lib.runtime.storage import read_json, write_json, IntegrationError


class MultiViewTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root / 'source'
        self.source.mkdir()
        self.proof = {'herb': 'REFERENCE_ONLY', 'feature_id': 'TRUTH_ONLY', 'views': []}
        for i in range(3):
            p = self.source / f'original_{i}.jpg'
            p.write_bytes(f'explicit-test-placeholder-{i}'.encode())
            self.proof['views'].append({'view': p.name, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()})
        self.save()
        self.options = {'output_root': self.root / 'outputs', 'a_python': paths.VENV_PY}
        self.template = read_json(Path(paths.PROJECT_ROOT) / 'config/multi_view.json')

    def save(self):
        write_json(self.source / 'input_proof.json', self.proof)

    def test_hash_missing_and_duplicate_rejected_before_case_creation(self):
        self.proof['views'][0]['sha256'] = '0' * 64
        self.save()
        with self.assertRaises(IntegrationError):
            F.run_multi_view(self.source, **self.options, dry_run=True)
        self.assertFalse(self.options['output_root'].exists())
        self.proof['views'][0] = copy.deepcopy(self.proof['views'][1])
        self.save()
        with self.assertRaises(IntegrationError) as error:
            F.inspect_sources(self.source)
        self.assertEqual(error.exception.code, 'MULTI_VIEW_DUPLICATE')

    def test_traversal_and_wrong_count_rejected(self):
        outside = self.root / 'outside.jpg'
        outside.write_bytes(b'outside')
        self.proof['views'][0]['path'] = '../outside.jpg'
        self.save()
        with self.assertRaises(IntegrationError) as error:
            F.inspect_sources(self.source)
        self.assertEqual(error.exception.code, 'MULTI_VIEW_SCOPE')
        self.proof['views'] = []
        self.save()
        with self.assertRaises(IntegrationError):
            F.inspect_sources(self.source)

    def test_arbitrary_count_is_not_fixed_to_five(self):
        for i in range(3, 7):
            p = self.source / f'original_{i}.jpg'
            p.write_bytes(f'explicit-test-placeholder-{i}'.encode())
            self.proof['views'].append({'view': p.name,
                                        'sha256': hashlib.sha256(p.read_bytes()).hexdigest()})
        self.save()
        bindings = F.inspect_sources(self.source)
        self.assertEqual(len(bindings), 7)
        images = [row['source'] for row in bindings]
        cfg = F.bind_config(self.template, images, sample_id='test', a_python=paths.VENV_PY)
        self.assertEqual(len(cfg['a']['images']), 7)
        self.assertEqual(len(cfg['agent']['image_roots']), 7)

    def test_a_python_required_before_case_creation(self):
        with self.assertRaises(IntegrationError) as error:
            F.run_multi_view(self.source, output_root=self.root / 'outputs', dry_run=True)
        self.assertEqual(error.exception.code, 'MULTI_VIEW_A_PYTHON_MISSING')
        self.assertFalse((self.root / 'outputs').exists())

    def test_case_name_rejects_os_invalid_and_reserved_names(self):
        invalid = ['', '   ', '.', '..', 'a<b', 'a>b', 'a"b', 'a|b', 'a?b', 'a*b',
                   'a:b', 'a\\b', 'a/b', 'a\x00b', 'a\nb', ' name', 'name ',
                   'name.', 'CON', 'con.txt', 'COM1', 'LPT9', 'a' * 101]
        for name in invalid:
            with self.subTest(name=name):
                with self.assertRaises(IntegrationError) as error:
                    F.run_multi_view(self.source, **self.options, dry_run=True, case_name=name)
                self.assertEqual(error.exception.code, 'MULTI_VIEW_NAME')
        self.assertFalse(self.options['output_root'].exists())

    def test_case_name_accepts_portable_names(self):
        with patch.object(F, 'doctor', return_value={'ready': True}):
            base = F.run_multi_view(self.source, **self.options, dry_run=True,
                                    case_name='龙胆测试_v1')
        self.assertEqual(base.name, '龙胆测试_v1')
        self.assertTrue((base / 'interface_status.json').is_file())

    def test_directory_tags_are_validated_and_separate_from_sample_id(self):
        for options in ({'subject': '../outside'}, {'purpose': '两个__字段'},
                        {'case_name': 'legacy', 'subject': 'S01'}):
            with self.subTest(options=options), self.assertRaises(IntegrationError) as error:
                F.run_multi_view(self.source, **self.options, **options, dry_run=True)
            self.assertEqual(error.exception.code, 'MULTI_VIEW_NAME')
        self.assertFalse(self.options['output_root'].exists())
        with patch.object(F, 'doctor', return_value={'ready': True}):
            base = F.run_multi_view(self.source, **self.options, dry_run=True,
                                    subject='样品S01', purpose='链路验收')
        sample_id = read_json(base / 'config.json')['sample_id']
        self.assertRegex(base.name, r'^模拟样品S01_\d{1,2}月\d{1,2}日\d{2}时\d{2}分\d{2}秒$')
        self.assertRegex(sample_id, r'^case_[a-f0-9]{8}$')
        self.assertNotIn(sample_id[5:], base.name)
        metadata = read_json(base / 'interface_status.json')
        self.assertEqual((metadata['subject'], metadata['purpose'], metadata['view_count']),
                         ('样品S01', '链路验收', 3))
        self.assertEqual(metadata['kind'], '模拟')
        self.assertTrue(metadata['started_at'].endswith('+08:00'))

    def test_same_second_keeps_separate_cases(self):
        instant = datetime(2026, 11, 19, 14, 30, 25, tzinfo=F._RUN_TIMEZONE)
        with patch.object(F, 'datetime') as clock, patch.object(F, 'doctor', return_value={'ready': True}):
            clock.now.return_value = instant
            first = F.run_multi_view(self.source, **self.options, dry_run=True, subject='龙胆')
            second = F.run_multi_view(self.source, **self.options, dry_run=True, subject='龙胆')
        self.assertEqual(first.name, '模拟龙胆_11月19日14时30分25秒')
        self.assertEqual(second.name, first.name + '-02')
        self.assertNotEqual(read_json(first / 'config.json')['sample_id'],
                            read_json(second / 'config.json')['sample_id'])

    def test_standard_output_root_rejects_exact_override(self):
        with patch.object(paths, 'RUN_RESULTS', str(self.options['output_root'])):
            with self.assertRaises(IntegrationError) as error:
                F.run_multi_view(self.source, **self.options, dry_run=True, case_name='旧目录名')
        self.assertEqual(error.exception.code, 'MULTI_VIEW_NAME')
        self.assertFalse(self.options['output_root'].exists())

    def test_dry_run_copies_only_sources_fresh_cache_no_labels_or_model(self):
        (self.source / 'a_cache').mkdir()
        (self.source / 'a_cache/old.json').write_text('old')
        with patch.object(F, 'doctor', return_value={'ready': True}), \
                patch.object(F, 'run_pipeline') as pipeline, patch('urllib.request.urlopen') as network:
            base = F.run_multi_view(self.source, **self.options, dry_run=True)
        pipeline.assert_not_called()
        network.assert_not_called()
        self.assertRegex(base.name, r'^模拟未标注_\d{1,2}月\d{1,2}日\d{2}时\d{2}分\d{2}秒$')
        cfg = read_json(base / 'config.json')
        self.assertNotIn('REFERENCE_ONLY', json.dumps(cfg))
        self.assertNotIn('TRUTH_ONLY', json.dumps(cfg))
        self.assertFalse((base / 'a_cache').exists())
        self.assertFalse((base / 'runs').exists())
        self.assertEqual(len(cfg['a']['images']), len(self.proof['views']))
        self.assertEqual(cfg['a']['images'], cfg['agent']['image_roots'])
        self.assertEqual(cfg['agent']['transport'], 'mcp')
        workflow = cfg['agent']['workflow']
        self.assertIs(workflow['semantic_execution'], True)
        self.assertIs(workflow['problem_goal_execution'], True)
        self.assertIs(workflow['physical_parallel'], True)
        self.assertEqual(workflow['max_parallel'], 4)
        self.assertNotIn('routing', cfg)
        self.assertEqual(read_json(base / 'interface_status.json')['status'], 'prepared')

    def test_live_calls_existing_pipeline_with_fresh_case_and_writes_pointer(self):
        def run(cfg, *, base):
            self.assertEqual(cfg, read_json(base / 'config.json'))
            self.assertIs(cfg['agent']['workflow']['semantic_execution'], True)
            self.assertIs(cfg['agent']['workflow']['problem_goal_execution'], True)
            self.assertIs(cfg['agent']['workflow']['physical_parallel'], True)
            self.assertFalse((base / 'a_cache').exists())
            self.assertFalse((base / 'runs').exists())
            result = base / 'runs/fake-run'
            result.mkdir(parents=True)
            return result
        with patch.object(F, 'doctor', return_value={'ready': True}), patch.object(F, 'run_pipeline', side_effect=run) as call:
            result = F.run_multi_view(self.source, **self.options)
        self.assertEqual(call.call_count, 1)
        base = result.parent.parent
        self.assertRegex(base.name, r'^实测未标注_\d{1,2}月\d{1,2}日\d{2}时\d{2}分\d{2}秒$')
        self.assertEqual(read_json(base / 'run_location.json')['directory'], str(result))
        self.assertEqual(read_json(base / 'interface_status.json')['status'], 'completed')

    def test_doctor_failure_is_terminal_and_persisted(self):
        with patch.object(F, 'doctor', return_value={'ready': False}), patch.object(F, 'run_pipeline') as call:
            with self.assertRaises(IntegrationError) as error:
                F.run_multi_view(self.source, **self.options)
        call.assert_not_called()
        self.assertEqual(error.exception.code, 'MULTI_VIEW_NOT_READY')
        base = next(self.options['output_root'].iterdir())
        self.assertEqual(read_json(base / 'interface_status.json')['stage'], 'doctor')
        self.assertEqual(read_json(base / 'interface_status.json')['status'], 'failed')

    def test_pipeline_failure_never_claims_completion(self):
        with patch.object(F, 'doctor', return_value={'ready': True}), \
                patch.object(F, 'run_pipeline', side_effect=IntegrationError('TEST_FAILURE', 'fixture')):
            with self.assertRaises(IntegrationError):
                F.run_multi_view(self.source, **self.options)
        base = next(self.options['output_root'].iterdir())
        self.assertFalse((base / 'run_location.json').exists())
        self.assertEqual(read_json(base / 'interface_status.json')['error_code'], 'TEST_FAILURE')

    def test_template_rejects_hidden_simulation_or_routing(self):
        images = [str(self.source / row['view']) for row in self.proof['views']]
        for field, value in [('routing', {'mode': 'shadow'}), ('allow_recorded', True)]:
            cfg = copy.deepcopy(self.template)
            cfg[field] = value
            with self.assertRaises(IntegrationError):
                F.bind_config(cfg, images, sample_id='test', a_python=paths.VENV_PY)

    def test_legacy_template_semantic_default_and_independent_overrides(self):
        images = [str(self.source / row['view']) for row in self.proof['views']]
        for semantic, physical in ((None, True), (False, True), (True, False), (False, False)):
            with self.subTest(semantic=semantic, physical=physical):
                template = copy.deepcopy(self.template)
                workflow = template['agent']['workflow']
                del workflow['problem_goal_execution']
                workflow['physical_parallel'] = physical
                if semantic is None:
                    del workflow['semantic_execution']
                else:
                    workflow['semantic_execution'] = semantic
                bound = F.bind_config(template, images, sample_id='test', a_python=paths.VENV_PY)
                effective = bound['agent']['workflow']
                self.assertIs(effective['semantic_execution'], True if semantic is None else semantic)
                self.assertIs(effective['physical_parallel'], physical)
                self.assertEqual('semantic_execution' in workflow, semantic is not None)

    def test_problem_goal_can_be_disabled_and_requires_the_semantic_executor(self):
        images = [str(self.source / row['view']) for row in self.proof['views']]
        template = copy.deepcopy(self.template)
        workflow = template['agent']['workflow']

        def bound_workflow():
            return F.bind_config(template, images, sample_id='test', a_python=paths.VENV_PY)['agent']['workflow']

        workflow['problem_goal_execution'] = False
        self.assertIs(bound_workflow()['problem_goal_execution'], False)
        del workflow['problem_goal_execution']
        self.assertNotIn('problem_goal_execution', bound_workflow())
        workflow.update(problem_goal_execution=True, semantic_execution=False)
        with self.assertRaisesRegex(IntegrationError, '语义 DAG'):
            bound_workflow()

    def test_reference_policy_field_parity_and_explicit_defaults(self):
        ref_path = Path(paths.PROJECT_ROOT) / '运行结果/case_9cc59640/config.json'
        if not ref_path.is_file():
            self.skipTest('historical reference case is not present in this workspace')
        ref = read_json(ref_path)
        old = ref['agent']['workflow']
        new = self.template['agent']['workflow']
        for field, value in old.items():
            if field not in {'p0_priority', 'max_initializations', 'max_locate_attempts', 'semantic_execution',
                             'max_planning', 'max_observations', 'max_api_calls',
                             'max_resource_units'}:
                self.assertEqual(new[field], value, field)
        self.assertEqual(new['max_planning'], 60)
        self.assertEqual(new['max_observations'], 40)
        self.assertEqual(new['max_api_calls'], 200)
        self.assertEqual(new['max_resource_units'], 400)
        self.assertEqual(self.template['agent']['max_steps'], 60)
        self.assertNotIn('max_initializations', new)
        self.assertNotIn('max_locate_attempts', new)
        self.assertEqual(new['initialization_reserve'], 1)
        self.assertEqual(new['locate_attempts_per_view'], 2)
        from TCM_Fusion.lib.workflow.workflow_session import TOOLS
        self.assertEqual(set(new['enabled_tools']), set(TOOLS))
        self.assertNotIn('p0_priority', new)
        self.assertEqual(new['glm_identity_threshold'], .8)
        self.assertEqual(new['locate_expand'], 1.5)
        self.assertNotIn('needle_audit', new)
        self.assertIn('针晶', new['feature_profiles']['needle_crystal']['positive'])
        self.assertNotIn('needle_positive_hint', new)
        self.assertNotIn('needle_negative_hint', new)
        self.assertNotIn('python', self.template['a'])
        self.assertNotIn('case_9cc59640', json.dumps(self.template))
