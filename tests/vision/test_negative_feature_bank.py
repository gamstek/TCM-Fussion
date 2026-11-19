"""负例特征库结构回归：按 P0 测试错误归因构建，只作提示与审计，不替代召回。
该组用例由项目根 selftest 发现。涉及模型回复的用例使用可控输入，涉及进程的用例以各断言规定的执行边界为准。

主要定义：
- NegativeFeatureBankTests（schema）: 校验 schema 版本与目标集合覆盖针晶及主要组织特征。
- NegativeFeatureBankTests（针晶硬负例）: 针晶硬负例限定为组织类特征，淀粉粒排除、簇晶记为软负例。
- NegativeFeatureBankTests（特征条目）: 每个特征都具备 morphology、evidence 与 priority，且 gt 大于零。
- NegativeFeatureBankTests（P0 混淆）: 记录 fiber 的 gt、stone_cell 的 fp 与 secretory_tissue 的 miss，并保持淀粉粒无 negative_for。

关联：
- 统一执行见 tests/README.md。
"""
import json
import unittest
from pathlib import Path

import paths


class NegativeFeatureBankTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path = Path(paths.PROJECT_ROOT) / 'config/negative_feature_bank.json'
        cls.bank = json.loads(cls.path.read_text(encoding='utf-8'))

    def test_schema_and_target_conditioned_targets(self):
        self.assertEqual(self.bank['schema'], 'negative-feature-bank/v1')
        self.assertIn('needle_crystal', self.bank['targets'])
        for target in ('fiber', 'epidermis_cell', 'vessel', 'stone_cell', 'secretory_tissue'):
            self.assertIn(target, self.bank['targets'])

    def test_needle_hard_negatives_are_tissue_cluster(self):
        target = self.bank['targets']['needle_crystal']
        self.assertEqual(set(target['hard_negatives']),
                         {'fiber', 'epidermis_cell', 'vessel', 'stone_cell', 'secretory_tissue'})
        self.assertIn('starch_grain', target['excluded'])
        self.assertIn('cluster_crystal', target['soft_negatives'])

    def test_features_have_morphology_and_evidence(self):
        for name, row in self.bank['features'].items():
            self.assertTrue(row['morphology'], name)
            self.assertTrue(row['evidence'], name)
            self.assertIn('priority', row, name)
            self.assertGreater(row['evidence'].get('gt', 0), 0, name)

    def test_p0_confusion_cluster_is_documented(self):
        feature = self.bank['features']
        self.assertGreater(feature['fiber']['evidence']['gt'], 0)
        self.assertGreater(feature['stone_cell']['evidence']['fp'], 0)
        self.assertGreater(feature['secretory_tissue']['evidence']['miss'], 0)
        self.assertEqual(feature['starch_grain']['negative_for'], [])


if __name__ == '__main__':
    unittest.main()
