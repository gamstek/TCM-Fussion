"""本地配置检查与启动前失败回归；替身只覆盖检查分支，绝不发起模型请求。

主要定义：
- ReadinessTests（配置解析）: 缺失、非法 JSON 与 BOM 编码分别给出 CONFIG_MISSING/CONFIG_INVALID/CONFIG_ENCODING_UNSUPPORTED。
- ReadinessTests（凭据与端点）: 模板无 key 不可 ready；文件与环境凭据冲突、含用户信息的端点被拒绝且不回显密钥。
- ReadinessTests（离线放行）: heuristic 本地规划不要求凭据即可 ready。
- ReadinessTests（启动拦截）: preflight 失败时不构造 runtime/MCP，并持久化 agent_preflight.json。

关联：
- TCM_Fusion.lib.runtime_readiness: 被测的 profile_checks 与 planner_preflight。
- TCM_Fusion.lib.workflow_entry.execute_workflow: 启动前检查与失败拦截路径。
- paths.DETECTION_CONFIG_DIR: 被临时目录替换以隔离真实配置。
- 位于 tests/ 下，随 selftest 递归发现。
"""
import asyncio
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import paths
from TCM_Fusion.lib.runtime_readiness import profile_checks,planner_preflight
from TCM_Fusion.lib.storage import write_json,read_json,IntegrationError
from TCM_Fusion.lib.workflow_entry import execute_workflow


class ReadinessTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.path_patch=patch.object(paths,'DETECTION_CONFIG_DIR',str(self.root));self.path_patch.start()
        self.addCleanup(self.path_patch.stop)
        self.config={'agent':{'mode':'active','planner':'llm','workflow':{'execution':'active'}}}

    def profile(self,**extra):
        values={'base_url':'https://invalid.example/v1','model':'unit-test','api_key':'','api_key_env':'TEST_FUSION_KEY',**extra}
        write_json(self.root/'vlm_config.json',{'vlm':values,'igm':values})

    def test_missing_and_malformed_config_are_explicit(self):
        self.assertEqual(profile_checks()[0]['code'],'CONFIG_MISSING')
        (self.root/'vlm_config.json').write_text('{bad','utf-8')
        self.assertEqual(profile_checks()[0]['code'],'CONFIG_INVALID')

    def test_template_without_key_is_not_ready(self):
        self.profile()
        with patch.dict(os.environ,{},clear=True):
            value=planner_preflight(self.config)
        self.assertFalse(value['ready'])
        self.assertIn('CREDENTIALS_MISSING',[r['code'] for r in value['checks']])

    def test_bom_is_rejected_before_legacy_json_reader_fails(self):
        (self.root/'vlm_config.json').write_bytes(b'\xef\xbb\xbf{}')
        self.assertEqual(profile_checks()[0]['code'],'CONFIG_ENCODING_UNSUPPORTED')

    def test_ready_check_does_not_call_network_or_expose_key(self):
        self.profile()
        secret='UNIT_TEST_SECRET_NOT_A_REAL_KEY'
        with patch.dict(os.environ,{'TEST_FUSION_KEY':secret},clear=True),patch('urllib.request.urlopen') as remote:
            checks=profile_checks(('vlm','igm'))
        self.assertTrue(all(r['ok'] for r in checks));remote.assert_not_called()
        self.assertNotIn(secret,json.dumps(checks))

    def test_conflicting_credentials_and_invalid_endpoint_are_denied(self):
        self.profile(api_key='FILE_FAKE_KEY',base_url='https://user:pass@invalid.example/v1')
        with patch.dict(os.environ,{'TEST_FUSION_KEY':'DIFFERENT_FAKE_KEY'},clear=True):
            checks=profile_checks()
        codes=[r['code'] for r in checks]
        self.assertIn('CREDENTIAL_AMBIGUOUS',codes);self.assertIn('CONFIG_ENDPOINT_INVALID',codes)
        self.assertNotIn('DIFFERENT_FAKE_KEY',json.dumps(checks))

    def test_offline_cpu_does_not_require_credentials(self):
        self.config['agent']['planner']='heuristic'
        self.assertTrue(planner_preflight(self.config)['ready'])

    def test_failed_preflight_never_starts_runtime_or_mcp(self):
        directory=self.root/'run';directory.mkdir()
        with patch('TCM_Fusion.lib.workflow_entry.from_task') as runtime:
            with self.assertRaises(IntegrationError) as failure:
                asyncio.run(execute_workflow(self.config,{},directory,routing={},start=time.perf_counter(),
                                            job={'status':'running'},status_path=directory/'job.json'))
        self.assertEqual(failure.exception.code,'CONFIG_MISSING');runtime.assert_not_called()
        self.assertFalse(read_json(directory/'agent_preflight.json')['ready'])
