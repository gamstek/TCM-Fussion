"""特征 profile：taxonomy 默认全集 + 针对特征覆盖；不把针晶负例施加给其他特征。"""
import unittest

from TCM_Fusion.lib.feature_profiles import build_profiles, hint_text, profile_for


class FeatureProfileTests(unittest.TestCase):
    def test_taxonomy_is_default_for_all_features(self):
        profiles = build_profiles()
        self.assertIn('needle_crystal', profiles)
        self.assertIn('fiber', profiles)
        self.assertIn('pollen_grain', profiles)
        self.assertTrue(profiles['fiber']['positive'])
        self.assertEqual(profiles['fiber']['source'], 'taxonomy')

    def test_needle_override_does_not_leak_to_fiber(self):
        profiles = build_profiles({'needle_crystal': {
            'positive': '针晶正例',
            'negative': ['纤维', '表皮细胞'],
        }})
        self.assertIn('针晶正例', profiles['needle_crystal']['positive'])
        self.assertIn('纤维', profiles['needle_crystal']['negative'])
        self.assertNotIn('针晶正例', profiles['fiber']['positive'])
        self.assertEqual(profiles['fiber']['negative'], [])

    def test_unknown_feature_has_no_profile(self):
        self.assertEqual(profile_for('not_a_taxonomy_feature'), {})
        self.assertEqual(hint_text({}), '')

    def test_hint_text_contains_positive_and_negative(self):
        text = hint_text(profile_for('needle_crystal', {
            'needle_crystal': {'positive': '单根', 'negative': ['纤维']}}))
        self.assertIn('单根', text)
        self.assertIn('纤维', text)


if __name__ == '__main__':
    unittest.main()
