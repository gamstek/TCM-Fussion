# -*- coding: utf-8 -*-
"""P0 优先缺口判定的离线回归：只读工作流状态推断下一焦点，不联网、不执行阶段。

主要定义：
- P0PriorityTest: 验证 WorkflowRuntime.p0_focus 对缺 crop、已观察 supporting、待分类、待分割、待测量、分割复核和已尝试等状态的判定。
- _Workflow / _Store: 提供最小工作流外观（p0/recovery 开关、locate 上限、操作条目与 artifact 快照）。
- _a_dragon: 构造龙胆 Top-1 的五折 A 结果，触发针晶 P0 焦点。

关联：
- TCM_Fusion.lib.workflow_runtime: 被测的 p0_focus 优先级、身份门与测量门。
- rule_engine.KnowledgeBase: 从 herb_criteria.json 读取候选判据。
- TCM_Fusion.lib.storage.write_json: 构造临时 operations.json。
- 位于 tests/ 下，随 selftest 递归发现。
"""
import tempfile
import unittest
from pathlib import Path

import paths
from rule_engine import KnowledgeBase

from TCM_Fusion.lib.storage import write_json
from TCM_Fusion.lib.workflow_runtime import WorkflowRuntime


class _Store:
    def __init__(self, artifacts=None):
        self.artifacts = artifacts or {}

    def current(self):
        return {"artifacts": self.artifacts}


class _Workflow:
    def __init__(self, operations, *, p0=True, recovery=True, max_locate=4):
        self.config = {"p0_priority": p0, "targeted_recovery": recovery,
                       "max_locate_attempts": max_locate}
        self.store = _Store()
        self.operations = operations

    def segmentation_reviews(self):
        return []


class _Runtime:
    pass


def _a_dragon():
    views = []
    for i in range(1, 6):
        views.append({"status": "ok", "image": f"D:/case/view_0{i}.jpg", "input_sha256": f"sha{i}",
                      "fold_rankings": [[{"label": "龙胆", "probability": 0.99},
                                         {"label": "桔梗", "probability": 0.01}]] * 5})
    return {"status": "ok", "ranking": [], "view_predictions": views}


class P0PriorityTest(unittest.TestCase):
    def _runtime(self, root, *, p0=True, recovery=True, present=None, entries=None, max_locate=4):
        ops = Path(root) / "operations.json"
        write_json(ops, {"entries": entries or {}})
        runtime = _Runtime()
        runtime.workflow = _Workflow(ops, p0=p0, recovery=recovery, max_locate=max_locate)
        runtime.kb = KnowledgeBase(Path(paths.JUDGE_DIR) / "data/herb_criteria.json")
        runtime.a = _a_dragon()
        runtime.aliases = {}
        runtime.candidate_policy = {"top_k": 3}
        runtime.present_features = set(present or [])
        return runtime

    def test_dragon_needle_is_p0_focus(self):
        with tempfile.TemporaryDirectory() as root:
            focus = WorkflowRuntime.p0_focus(self._runtime(root))
            self.assertIsNotNone(focus)
            self.assertEqual(focus["candidate_id"], "龙胆")
            self.assertEqual(focus["feature_id"], "needle_crystal")
            self.assertEqual(focus["status"], "no_crop")
            self.assertFalse(focus["attempted"])

    def test_present_feature_then_supporting_review(self):
        with tempfile.TemporaryDirectory() as root:
            runtime = self._runtime(root, present=["needle_crystal"])
            focus = WorkflowRuntime.p0_focus(runtime)
            self.assertIsNotNone(focus)
            self.assertEqual(focus["status"], "supporting_review")
            self.assertEqual(focus["feature_id"], "epidermis_cell")

    def _runtime_with_crop(self, root, *, seg_quality=None, measured=False, identity=True,
                           det_score=None, identity_required=False, classify=True):
        crop_json = Path(root) / "needle_crop.json"
        write_json(crop_json, {"feature_id": "needle_crystal", "det_score": det_score})
        artifacts = {"crop:img:needle_crop": {"valid": True, "kind": "crop",
                                              "path": str(crop_json),
                                              "det_score": det_score,
                                              "identity_required": identity_required}}
        if classify:
            artifacts["classification:crop:img:needle_crop"] = {
                "valid": True, "kind": "classification", "identity_verified": identity}
        if seg_quality:
            artifacts["seg:crop:img:needle_crop"] = {
                "valid": True, "kind": "segmentation", "quality": seg_quality}
        if measured:
            artifacts["measurement:seg:crop:img:needle_crop"] = {
                "valid": True, "kind": "measurement"}
        runtime = self._runtime(root)
        runtime.workflow.store.artifacts = artifacts
        return runtime

    def test_verified_crop_requires_segmentation(self):
        with tempfile.TemporaryDirectory() as root:
            focus = WorkflowRuntime.p0_focus(self._runtime_with_crop(root))
            self.assertIsNotNone(focus)
            self.assertEqual(focus["status"], "crop_found_unsegmented")
            self.assertEqual(focus["crop_id"], "needle_crop")
            self.assertFalse(focus["attempted"])

    def test_pending_high_score_crop_requires_classification(self):
        with tempfile.TemporaryDirectory() as root:
            focus = WorkflowRuntime.p0_focus(self._runtime_with_crop(
                root, det_score=0.92, identity_required=True, classify=False))
            self.assertIsNotNone(focus)
            self.assertEqual(focus["status"], "crop_pending_classification")
            self.assertEqual(focus["crop_id"], "needle_crop")
            self.assertEqual(focus["det_score"], 0.92)
            self.assertFalse(focus["attempted"])

    def test_valid_segmentation_requires_measurement(self):
        with tempfile.TemporaryDirectory() as root:
            focus = WorkflowRuntime.p0_focus(self._runtime_with_crop(root, seg_quality="valid"))
            self.assertIsNotNone(focus)
            self.assertEqual(focus["status"], "crop_segmented_unmeasured")
            self.assertFalse(focus["attempted"])

    def test_review_segmentation_opens_review_gate(self):
        with tempfile.TemporaryDirectory() as root:
            review_focus = WorkflowRuntime.p0_focus(
                self._runtime_with_crop(root, seg_quality="review"))
            self.assertIsNotNone(review_focus)
            self.assertEqual(review_focus["status"], "crop_segmentation_review")
            self.assertFalse(review_focus["attempted"])
            measured_focus = WorkflowRuntime.p0_focus(
                self._runtime_with_crop(root, seg_quality="valid", measured=True))
            self.assertNotEqual((measured_focus or {}).get("status"), "crop_segmentation_review")

    def test_first_locate_attempt_marks_p0_attempted(self):
        with tempfile.TemporaryDirectory() as root:
            entries = {"l1": {"tool": "locate_feature", "status": "done"}}
            runtime = self._runtime(root, entries=entries)
            focus = WorkflowRuntime.p0_focus(runtime)
            self.assertTrue(focus["attempted"])

    def test_disabled_flags_skip_gate(self):
        with tempfile.TemporaryDirectory() as root:
            self.assertIsNone(WorkflowRuntime.p0_focus(self._runtime(root, p0=False)))
            self.assertIsNone(WorkflowRuntime.p0_focus(self._runtime(root, recovery=False)))


if __name__ == "__main__":
    unittest.main()
