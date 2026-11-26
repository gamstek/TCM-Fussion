"""验证比例尺单位的像素复读和标定入口。

文字框替身覆盖空间、数值、单位与置信度约束。视觉环境另用真实 RapidOCR 复现误读。
关联：common/scalebar_duo.py、detect_core.lookup_px_per_um。
"""
import unittest
from importlib.util import find_spec
from unittest.mock import patch

import cv2
import numpy as np

import detect_core
import scalebar_duo as scale


def caption_image(text):
    image = np.full((110, 360, 3), 255, np.uint8)
    cv2.putText(image, text, (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3, cv2.LINE_AA)
    return image


def bar_image(text):
    image = np.full((480, 840, 3), 255, np.uint8)
    cv2.putText(image, text, (510, 464), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.rectangle(image, (530, 388), (729, 390), (0, 0, 0), -1)
    for x in (530, 729):
        cv2.rectangle(image, (x, 380), (x, 398), (0, 0, 0), -1)
    return image


def text_box(text, bounds=(25, 20, 65, 50), conf=.96):
    x1, y1, x2, y2 = bounds
    return [[[x1, y1], [x2, y1], [x2, y2], [x1, y2]], text, conf]


class ScaleCaptionTests(unittest.TestCase):
    def setUp(self):
        self.ocr = self.enterContext(patch.object(scale, "_get_ocr")).return_value
        self.ocr.text_score = .5
        self.image = caption_image("100 um")

    def test_value_must_be_bound_to_a_micron_unit(self):
        examples = {
            "100 um": 100, "20μm": 20, "20 µm": 20, "长度：20:00um": 20,
            "图 7 长度：12.50 μm": 12.5, "长度２０．００㎛": 20, "20 μ m。": 20,
            "100": None, "100 mm": None, "100 nm": None, "100 um/px": None,
            "100 μm²": None, "100 µmol": None, "-12.50um": None, "10:50um": None,
            "1001um": None, "0.25um": None,
        }
        for raw, expected in examples.items():
            with self.subTest(raw=raw):
                self.ocr.return_value = [text_box(raw)], None
                self.assertEqual(scale.ocr_read_um(self.image, [20, 15, 100, 20]), expected)

    def test_reopens_only_the_original_pixels_of_the_number_and_unit(self):
        self.ocr.side_effect = [
            ([text_box("wn", (70, 24, 105, 51)), text_box("100")], None),
            ([["100 um", .94]], None),
        ]
        labels = scale._read_um_labels(self.image)
        self.assertEqual(labels, [dict(um=100., bbox=[25, 20, 105, 51], text="100 um", conf=.94)])
        reread = self.ocr.call_args_list[1]
        np.testing.assert_array_equal(reread.args[0], self.image[20:52, 25:106])
        self.assertEqual(reread.kwargs, dict(use_det=False, use_cls=False))

    def test_blurred_single_box_also_requires_an_explicit_rereading(self):
        for answer, expected in [("100 um", 100), ("100 wn", None)]:
            with self.subTest(answer=answer):
                self.ocr.side_effect = [([text_box("100 wn")], None), ([[answer, .97]], None)]
                self.assertEqual(scale.ocr_read_um(self.image, [20, 15, 100, 20]), expected)

    def test_rereading_must_confirm_both_number_and_unit(self):
        for line in ([["100 wn", .99]], [["100 mm", .99]], [["50 um", .99]],
                     [["100 um", .25]], None):
            with self.subTest(line=line):
                self.ocr.side_effect = [
                    ([text_box("100"), text_box("wn", (72, 20, 135, 50))], None), (line, None)]
                self.assertIsNone(scale.ocr_read_um(self.image, [20, 15, 100, 20]))

    def test_unrelated_boxes_and_other_units_do_not_trigger_rereading(self):
        for unit, bbox in (("wn", (70, 60, 95, 90)), ("um", (160, 20, 190, 50)),
                           ("wn", (5, 20, 20, 50)), ("mm", (70, 20, 95, 50)),
                           ("nm", (70, 20, 95, 50))):
            with self.subTest(unit=unit, bbox=bbox):
                self.ocr.reset_mock()
                self.ocr.return_value = [text_box("100"), text_box(unit, bbox)], None
                self.assertEqual(scale._read_um_labels(self.image), [])
                self.ocr.assert_called_once()

    def test_resized_fallback_uses_original_bar_coordinates(self):
        self.ocr.return_value = [
            text_box("50 um", (0, 0, 80, 20), .99),
            text_box("100", (550, 340, 600, 365), .99),
            text_box("20 μm", (610, 400, 670, 415), .89),
        ], None
        image = np.zeros((1000, 1600, 3), np.uint8)
        self.assertEqual(scale.ocr_read_um(image, [1200, 790, 1400, 795], max_side=800), 20)

    def test_missing_unit_cannot_reach_downstream_even_with_corroborating_pixels(self):
        self.enterContext(patch.object(detect_core, "load_config", return_value={"vlm": {}}))
        self.enterContext(patch.object(scale, "resolve_key", return_value=""))
        self.enterContext(patch("ls_common.family_bar_px", return_value=200))
        self.enterContext(patch("ls_common.confirmed_bar_px", return_value={200}))
        self.enterContext(patch.object(scale, "engine_b", return_value={"bar_px": 200, "bar_um": 100}))
        for raw, expected in (("100", ""), ("100 mm", ""), ("100 nm", ""), ("100 um", 2.0),
                              ("20 um", "")):
            with self.subTest(raw=raw):
                self.ocr.return_value = [text_box(raw, (512, 410, 752, 465))], None
                self.assertEqual(detect_core.lookup_px_per_um("test.png", img=bar_image(raw)), expected)


@unittest.skipUnless(find_spec("rapidocr_onnxruntime"), "真实 OCR 用例在 vision 环境运行")
class RapidOCRScaleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from rapidocr_onnxruntime import RapidOCR
        cls.enterClassContext(patch.object(
            scale, "_OCR", RapidOCR(intra_op_num_threads=2, inter_op_num_threads=1)))

    def test_reported_caption_with_no_bar(self):
        image = caption_image("100 um")
        labels = scale._read_um_labels(image)
        self.assertEqual([row["um"] for row in labels], [100])
        result = scale.engine_a(image)
        self.assertEqual((result["bar_um"], result["bar_px"]), (100, 0))
        row = scale.cross_validate("caption", "", result, {"bar_px": 0}, .1)
        self.assertEqual(row["px_per_um"], "")

    def test_units_and_missing_unit_on_a_measurable_bar(self):
        self.enterContext(patch.object(scale, "resolve_key", return_value=""))
        self.enterContext(patch.object(scale, "engine_b", return_value={"error": "offline"}))
        for text, unit, ratio in (("100 um", 100, 2.0), ("50 um", 50, 4.0), ("20.00um", 20, 10.0),
                                  ("100 mm", None, None), ("100 nm", None, None),
                                  ("100 wn", None, None), ("100", None, None)):
            with self.subTest(text=text):
                result = scale.measure_img_px_per_um(bar_image(text), "bar", cfg={"vlm": {}})
                self.assertEqual(result["bar_px_cv"], 200)
                self.assertEqual((result["bar_um"], result["px_per_um"]), (unit, ratio))
                self.assertEqual(result["status"], "OK-medium" if ratio else "manual")
