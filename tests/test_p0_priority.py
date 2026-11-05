# -*- coding: utf-8 -*-
"""P0 优先缺口的离线回归；不联网、不执行阶段。"""
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

    def test_present_feature_clears_focus(self):
        with tempfile.TemporaryDirectory() as root:
            runtime = self._runtime(root, present=["needle_crystal"])
            self.assertIsNone(WorkflowRuntime.p0_focus(runtime))

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

    def test_review_segmentation_or_measurement_releases_gate(self):
        with tempfile.TemporaryDirectory() as root:
            self.assertIsNone(WorkflowRuntime.p0_focus(
                self._runtime_with_crop(root, seg_quality="review")))
            self.assertIsNone(WorkflowRuntime.p0_focus(
                self._runtime_with_crop(root, seg_quality="valid", measured=True)))

    def test_all_grid_locate_attempts_required(self):
        with tempfile.TemporaryDirectory() as root:
            bundle = Path(root) / "op"
            (bundle / "detection").mkdir(parents=True)
            write_json(bundle / "detection/regions.json",
                       {"regions": [{"box": [0, 0, 10, 10]}, {"box": [5, 5, 15, 15]}]})
            entries = {
                "p": {"tool": "propose_regions", "status": "done",
                      "bundle": {"root": str(bundle),
                                 "files": {"detection/regions.json": "hash"}}},
                "l1": {"tool": "locate_feature", "status": "done"},
            }
            runtime = self._runtime(root, entries=entries, max_locate=2)
            focus = WorkflowRuntime.p0_focus(runtime)
            self.assertFalse(focus["attempted"])
            entries["l2"] = {"tool": "locate_feature", "status": "failed"}
            write_json(Path(root) / "operations.json", {"entries": entries})
            focus = WorkflowRuntime.p0_focus(runtime)
            self.assertTrue(focus["attempted"])

    def test_disabled_flags_skip_gate(self):
        with tempfile.TemporaryDirectory() as root:
            self.assertIsNone(WorkflowRuntime.p0_focus(self._runtime(root, p0=False)))
            self.assertIsNone(WorkflowRuntime.p0_focus(self._runtime(root, recovery=False)))


if __name__ == "__main__":
    unittest.main()
