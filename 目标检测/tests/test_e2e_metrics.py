# -*- coding: utf-8 -*-
"""test_e2e_metrics.py — 两阶段端到端验收的离线断言（不加载模型、不读真实图）。

**为什么存在**：`e2e_eval` 的三个数字（定位召回 / 命中框分类准确率 / 端到端召回）
只有"框匹配后**再**比类别"这一种口径。若把类错也算进定位命中、或把 e2e 精确率
写成分类准确率，数值看起来都正常但含义全错。这里用最小合成例子钉住口径，并验证
`match_pairs` 与 `box_recall` 历史使用的 `match_image` 等价（两个入口数字可比的前提）。

跑法（项目根）: python 目标检测/tests/test_e2e_metrics.py
"""
import importlib.util
import os

import cls_net
import paths
from box_metrics import match_image, match_pairs

CLI_DIR = os.path.join(paths.DETECTION_DIR, "cli")
_spec = importlib.util.spec_from_file_location("e2e_eval", os.path.join(CLI_DIR, "e2e_eval.py"))
e2e = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(e2e)

# ① match_pairs 与 match_image 等价（一对一、贪心、IoU 阈值）
_aboxes = [{"label": "a", "box": [0, 0, 10, 10]},
           {"label": "b", "box": [20, 20, 30, 30]}]
_preds = [{"box": [0, 0, 10, 10], "score": 0.9, "label": "a"},
          {"box": [20, 20, 30, 30], "score": 0.8, "label": "x"},
          {"box": [50, 50, 60, 60], "score": 0.7, "label": "b"}]
_pairs, _unused = match_pairs(_aboxes, _preds, 0.5)
assert [(gi, j) for gi, j, _ in _pairs] == [(0, 0), (1, 1)], _pairs
assert _unused == [2], _unused
_hits, _unused2, _ious = match_image(_aboxes, _preds, 0.5)
assert _hits == {0, 1} and _unused2 == {2}, (_hits, _unused2)
assert [round(v, 4) for v in _ious] == [1.0, 1.0], _ious
# 一个预测只能被一个 GT 占用：两个 GT 抢同一个预测时，第二个必须落空
_p2, _u2 = match_pairs([{"box": [0, 0, 10, 10]}, {"box": [0, 0, 10, 10]}],
                       [{"box": [0, 0, 10, 10], "score": 1.0, "label": "a"}], 0.5)
assert len(_p2) == 1 and _u2 == [], (_p2, _u2)

# ② 端到端口径：框命中但类错 → 计入 loc_tp，不计入 cls_tp
_ann = {"k": [{"label": "a", "box": [0, 0, 10, 10]},
              {"label": "b", "box": [20, 20, 30, 30]}]}
_r = e2e.evaluate(_ann, {"k": [
    {"box": [0, 0, 10, 10], "score": 0.9, "label": "a"},
    {"box": [20, 20, 30, 30], "score": 0.8, "label": "a"}]}, conf=0.5, iou_thr=0.5)
assert _r["loc_tp"] == 2 and _r["cls_tp"] == 1, _r
assert _r["recall_loc"] == 1.0 and _r["precision_loc"] == 1.0, _r
assert _r["cls_acc_on_matched"] == 0.5, _r
assert _r["recall_e2e"] == 0.5 and _r["precision_e2e"] == 0.5, _r
assert _r["confusions"] == {"b→a": 1}, _r["confusions"]
assert _r["per_class"]["b"]["loc_recall"] == 1.0
assert _r["per_class"]["b"]["cls_recall"] == 0.0
assert _r["per_class"]["b"]["cls_acc_on_matched"] == 0.0

# ③ conf 过滤：低于阈值的框不参与任何指标
_r2 = e2e.evaluate(_ann, {"k": [
    {"box": [0, 0, 10, 10], "score": 0.9, "label": "a"},
    {"box": [20, 20, 30, 30], "score": 0.8, "label": "a"}]}, conf=0.85, iou_thr=0.5)
assert _r2["n_pred"] == 1 and _r2["loc_tp"] == 1 and _r2["cls_tp"] == 1, _r2
assert _r2["recall_loc"] == 0.5 and _r2["recall_e2e"] == 0.5, _r2

# ④ 类别表：目录名 → 中文名（推理侧翻译的唯一来源）
_names = cls_net.load_class_names()
assert _names.get("05_starch_grain") == "淀粉粒", _names.get("05_starch_grain")
assert _names.get("13_other") == "其他", _names.get("13_other")
assert len(_names) == 14, len(_names)

print("E2E-METRICS TEST: ALL ASSERTIONS PASS")
