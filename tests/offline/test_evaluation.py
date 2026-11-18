"""离线评估与报告回归，不属于线上五视图链路。
该组用例由项目根 selftest 发现。涉及模型回复的用例使用可控输入，涉及进程的用例以各断言规定的执行边界为准。

主要定义：
- EvaluationTests.test_true_reference_is_not_exposed_by_task_feedback: 真实参考标签不进入任务反馈，`reflection_card` 不标注使用参考标签，review 视为弃答。
- test_image_label_is_not_silently_sample_identity_truth: `image_class` 标签不得被静默当作样本身份真值。
- test_zero_errors_does_not_imply_zero_risk: Wilson 上界在零错误时仍大于 0；空用例 `compare` 直接报错。

关联：
- 直接依赖 「lib/agent/action_feedback.py」、「lib/offline/ablation_evaluation.py」、「lib/offline/outcome_evaluation.py」、「lib/runtime/storage.py」。
- 统一执行见 tests/README.md。
"""
import unittest

from TCM_Fusion.lib.agent.action_feedback import reflection_card
from TCM_Fusion.lib.offline.ablation_evaluation import compare, wilson
from TCM_Fusion.lib.offline.outcome_evaluation import evaluate as outcome
from TCM_Fusion.lib.runtime.storage import IntegrationError


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
