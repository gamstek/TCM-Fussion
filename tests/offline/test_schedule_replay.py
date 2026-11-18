"""用独立临时目录检查调度回放的拒绝条件，避免把输入篡改或不可还原的
版本变化报告为乱序通过。成功路径由 10 号方案登记的真实工件对拍覆盖。

主要定义：ScheduleReplayTests 检查源目录保护、版本摘要、越界和替换。

关联：lib.offline.schedule_replay，由统一 selftest 收集，不调用视觉服务。
"""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from TCM_Fusion.lib.offline.schedule_replay import additions, history, run
from TCM_Fusion.lib.runtime.storage import IntegrationError, write_json


class ScheduleReplayTests(unittest.TestCase):
    def test_source_case_and_existing_output_are_protected(self):
        with TemporaryDirectory() as tmp:
            source = Path(tmp) / 'case/runs/fusion_1'
            source.mkdir(parents=True)
            with self.assertRaises(IntegrationError):
                run([source], source.parent.parent / 'replay')
            with self.assertRaises(FileExistsError):
                run([source], Path(tmp))

    def test_tampered_version_is_rejected(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            fake = 'a' * 64
            write_json(root / 'workflow/evidence/versions' / (fake + '.json'),
                       {'version': fake, 'parent': None, 'artifacts': {}, 'records': []})
            with self.assertRaisesRegex(IntegrationError, '摘要'):
                history(root, fake)

    def test_outside_case_and_replacements_are_rejected(self):
        empty = {'artifacts': {}, 'records': []}
        with TemporaryDirectory() as tmp:
            case = Path(tmp) / 'case'
            outside = {'id': 'crop', 'valid': True, 'path': str(case.parent / 'other')}
            with self.assertRaisesRegex(IntegrationError, '之外'):
                additions(empty, {'artifacts': {'crop': outside}, 'records': []}, case)
            with self.assertRaisesRegex(IntegrationError, '替换'):
                additions({'artifacts': {'crop': outside}, 'records': []}, empty, case)
            with self.assertRaisesRegex(IntegrationError, '测量'):
                additions({'artifacts': {}, 'records': [{'scope': 'instance'}]}, empty, case)
