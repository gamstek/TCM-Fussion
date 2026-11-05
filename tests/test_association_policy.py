"""连通域关联策略：规则表、主导连通域、粒度守卫与保守回退。"""
import json
import shutil
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

import association_policy as ap
import igm_segment as ig
import quality_eval as qe
import paths


def _comp(identifier, area, *, center=False, border=False, tight=1.0):
    return ap.Component(id=identifier, area=area, bbox=(0, 0, 1, 1),
                        inside_ratio=1.0, tight_overlap=tight,
                        center_hit=center, touches_border=border)


def _ctx(components, *, feature="fiber", shape="elongated", granularity="instance"):
    return ap.AssociationContext(components=tuple(components), feature_id=feature,
                                 shape=shape, granularity=granularity,
                                 tight_box=(0, 0, 100, 100))


class AssociationPolicyTest(unittest.TestCase):
    def setUp(self):
        self.policy = ap.load_policy()

    def test_dominant_keeps_main_and_records_dropped(self):
        decision = ap.evaluate(_ctx([
            _comp(1, 266655, center=True, border=True),
            _comp(2, 7977),
        ]), self.policy)
        self.assertEqual(decision.method, "dominant_component")
        self.assertEqual(decision.selected, (1,))
        self.assertEqual(decision.dropped, (2,))
        self.assertEqual(decision.status, "ok")
        self.assertTrue(decision.metrics["width_only"])
        self.assertAlmostEqual(decision.metrics["largest_share"], 0.97095, places=4)
        self.assertTrue(decision.allowed_uses["width"])
        self.assertFalse(decision.allowed_uses["length"])

    def test_comparable_components_stay_ambiguous(self):
        decision = ap.evaluate(_ctx([_comp(1, 60, center=True), _comp(2, 40)]), self.policy)
        self.assertEqual(decision.method, "ambiguous")
        self.assertEqual(decision.status, "review")
        self.assertFalse(decision.allowed_uses["width"])

    def test_bundle_granularity_never_takes_largest(self):
        decision = ap.evaluate(_ctx([_comp(1, 80, center=True), _comp(2, 20)],
                                    granularity="bundle"), self.policy)
        self.assertEqual(decision.method, "multipart")
        self.assertEqual(decision.selected, (1, 2))
        self.assertEqual(decision.status, "review")

    def test_unknown_feature_keeps_conservative_fallback(self):
        decision = ap.evaluate(_ctx([_comp(1, 80, center=True), _comp(2, 20)],
                                    feature=None, shape="unknown"), self.policy)
        self.assertEqual(decision.method, "ambiguous")
        self.assertEqual(decision.status, "review")

    def test_other_elongated_feature_is_not_eligible(self):
        decision = ap.evaluate(_ctx([_comp(1, 97, center=True), _comp(2, 3)],
                                    feature="crystal_fiber", shape="elongated"), self.policy)
        self.assertEqual(decision.method, "ambiguous")
        self.assertEqual(decision.status, "review")

    def test_dropped_component_touching_border_disqualifies(self):
        decision = ap.evaluate(_ctx([_comp(1, 97, center=True), _comp(2, 3, border=True)]),
                               self.policy)
        self.assertEqual(decision.method, "ambiguous")

    def test_mask_level_dominant_removes_extra_component(self):
        mask = np.zeros((100, 100), np.uint8)
        mask[5:95, 40:60] = 255       # 主体
        mask[2:8, 10:18] = 255        # 孤立碎屑
        out, info = ig.apply_box_prior(
            mask, [5, 0, 95, 99], association="region_component",
            feature_id="fiber", granularity="instance", association_policy=self.policy)
        self.assertEqual(info["association_method"], "dominant_component")
        self.assertEqual(info["association_dropped_components"], 1)
        self.assertEqual(cv2.connectedComponents((out > 0).astype(np.uint8), 8)[0] - 1, 1)

    def test_mask_level_without_feature_keeps_old_ambiguous_behavior(self):
        mask = np.zeros((100, 100), np.uint8)
        mask[5:95, 40:60] = 255
        mask[2:8, 10:18] = 255
        out, info = ig.apply_box_prior(
            mask, [5, 0, 95, 99], association="region_component",
            feature_id=None, granularity="instance", association_policy=self.policy)
        self.assertEqual(info["association_method"], "ambiguous")
        self.assertEqual(cv2.connectedComponents((out > 0).astype(np.uint8), 8)[0] - 1, 2)

    def test_quality_eval_allows_width_but_not_length_for_dominant(self):
        base = Path(paths.PROJECT_ROOT) / "results" / "ui_out" / "fusion" / "verification"
        base.mkdir(parents=True, exist_ok=True)
        folder = base / ("qa_" + uuid.uuid4().hex[:8])
        folder.mkdir(parents=True, exist_ok=False)
        try:
            image = np.zeros((100, 100, 3), np.uint8)
            cv2.imwrite(str(folder / "crop.png"), image)
            (folder / "crop.json").write_text("{}", encoding="utf-8")
            crop = SimpleNamespace(
                feature_id="fiber", granularity="instance", px_per_um=5.0,
                png=str(folder / "crop.png"), json_path=str(folder / "crop.json"),
                source_image_path=str(folder / "crop.png"), tight_box_local=[5, 0, 95, 99])
            mask = np.zeros((100, 100), np.uint8)
            mask[5:95, 40:60] = 255
            mask[2:8, 10:18] = 255
            out, assessment = qe.assess(mask, crop, bindings={}, identity_verified=True,
                                        stages={"pre_box": mask, "final": mask})
            self.assertEqual(assessment["association"]["method"], "dominant_component")
            self.assertTrue(assessment["uses"]["width"]["allowed"])
            self.assertFalse(assessment["uses"]["length"]["allowed"])
            self.assertFalse(assessment["uses"]["area"]["allowed"])
            self.assertEqual(cv2.connectedComponents((out > 0).astype(np.uint8), 8)[0] - 1, 1)
        finally:
            shutil.rmtree(folder, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
