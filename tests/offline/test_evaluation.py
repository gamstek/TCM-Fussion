"""Offline evaluation/reporting tests; not part of the live five-view chain."""
import unittest

from TCM_Fusion.lib.action_feedback import reflection_card
from TCM_Fusion.lib.ablation_evaluation import compare, wilson
from TCM_Fusion.lib.outcome_evaluation import evaluate as outcome
from TCM_Fusion.lib.storage import IntegrationError


class EvaluationTests(unittest.TestCase):
    def test_true_reference_is_not_exposed_by_task_feedback(self):
        trace={'decision':{'conclusion':{'id':'correct'},'verdict':'confirmed'},'needs_review':False,'events':[]}
        ref={'label':'correct','label_source':'expert','group_id':'g','split':'training','label_scope':'sample_identity'}
        value=outcome(trace,ref,a_label='wrong')
        self.assertTrue(value['corrected_a']);self.assertNotIn('label',value)
        self.assertFalse(reflection_card([])['uses_reference_label'])
        trace['needs_review']=True
        value=outcome(trace,ref,a_label='correct')
        self.assertTrue(value['abstained']);self.assertFalse(value['broke_correct_a'])

    def test_image_label_is_not_silently_sample_identity_truth(self):
        trace={'decision':{'conclusion':{'id':'x'}},'needs_review':False}
        result=outcome(trace,{'label':'x','label_source':'expert','group_id':'g','split':'test','label_scope':'image_class'})
        self.assertFalse(result['scope_compatible']);self.assertIsNone(result['correct'])

    def test_zero_errors_does_not_imply_zero_risk(self):
        self.assertGreater(wilson(0,5)[1],0)
        with self.assertRaises(IntegrationError):
            compare({'split':'test','cases':[]})
