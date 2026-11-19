# -*- coding: utf-8 -*-
"""test_cls_train.py — stage 2 分类器训练入口的离线断言（不训练、不联网）。

**为什么存在**：`train_cls.py` 是入口（不进 import 注入点），训练本身跑一次要几分钟，
不能当回归。这里只钉住三件事——数据扫描的失败模式、类别权重口径、混淆矩阵指标算法——
这三处算错会让报告看起来"正常"但数值是错的。

跑法（项目根）:
  python 目标检测/tests/test_cls_train.py
"""
import importlib.util
import os
import shutil
import sys
import tempfile

import paths

CLI_DIR = os.path.join(paths.DETECTION_DIR, "cli")
_spec = importlib.util.spec_from_file_location("train_cls", os.path.join(CLI_DIR, "train_cls.py"))
tc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tc)
import cls_net  # noqa: E402  变换的唯一实现（训练/推理共用）


def _make_data(root, extra_class=False):
    """造 3 split × 2 类（可加第 3 类只放 train，用来验证"类别目录不一致即报错"）。"""
    for split in ("train", "val", "test"):
        for cls in ("00_a", "01_b"):
            d = os.path.join(root, split, cls)
            os.makedirs(d)
            for i in range(2):
                with open(os.path.join(d, "{}.jpg".format(i)), "wb") as f:
                    f.write(b"\xff\xd8\xff\xd9")
    if extra_class:
        d = os.path.join(root, "train", "02_c")
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "0.jpg"), "wb") as f:
            f.write(b"\xff\xd8\xff\xd9")


# ① 数据扫描：正常数据 → 逐 split 逐类计数；三 split 类别不一致 → 必须报错
_tmp = tempfile.mkdtemp(prefix="cls_train_")
try:
    _make_data(_tmp)
    _counts, _names = tc.scan_splits(_tmp)
    assert _names == ["00_a", "01_b"], _names
    assert _counts["train"] == {"00_a": 2, "01_b": 2}, _counts["train"]
    assert _counts["test"] == {"00_a": 2, "01_b": 2}, _counts["test"]
    assert tc.scan_splits(_tmp, limit=1)[0]["train"] == {"00_a": 1, "01_b": 1}

    _bad = tempfile.mkdtemp(prefix="cls_train_bad_")
    try:
        _make_data(_bad, extra_class=True)
        try:
            tc.scan_splits(_bad)
            raise AssertionError("三 split 类别目录不一致时应报错")
        except ValueError as e:
            assert "类别目录" in str(e), e
    finally:
        shutil.rmtree(_bad, ignore_errors=True)
finally:
    shutil.rmtree(_tmp, ignore_errors=True)

# ② 类别权重：sqrt 压缩但方向正确（样本少的类权重更大），并按均值归一
_w = tc.class_weights([100, 100, 100, 10], "sqrt")
assert abs(sum(_w) / len(_w) - 1.0) < 1e-9, _w
assert _w[3] > _w[0] > 0, _w
assert abs(_w[0] - _w[1]) < 1e-12, _w
assert tc.class_weights([3, 1], "none") == [1.0, 1.0]
assert tc.class_weights([100, 10], "inv")[1] > tc.class_weights([100, 10], "sqrt")[1]

# ③ 指标：混淆矩阵对角线 → 准确率 1.0 / macro-F1 1.0；单类判错 → 该类推回率为 0
_names = ["a", "b"]
_m = tc.macro_scores([[3, 0], [0, 5]], _names)
assert _m["n"] == 8 and _m["accuracy"] == 1.0 and _m["macro_f1"] == 1.0, _m
assert [c["support"] for c in _m["per_class"]] == [3, 5], _m
_m = tc.macro_scores([[0, 3], [0, 5]], _names)
assert _m["accuracy"] == round(5 / 8, 4), _m
_a = _m["per_class"][0]
assert _a["recall"] == 0.0 and _a["precision"] == 0.0 and _a["f1"] == 0.0, _a
assert _m["macro_f1"] == round((0.0 + round(2 * 0.625 * 1.0 / 1.625, 4)) / 2, 4), _m
# 空 support 的类不参与 macro（避免全 0 类稀释指标）
_m = tc.macro_scores([[2, 0], [0, 0]], ["a", "b"])
assert _m["macro_f1"] == 1.0 and _m["per_class"][1]["support"] == 0, _m

# ④ 变换：三种输入口径都产出固定尺寸张量，评估侧无随机成分
import torch  # noqa: E402

_tr, _ev, _mean, _std = cls_net.build_transforms(64)
from PIL import Image  # noqa: E402

_img = Image.new("RGB", (200, 120), (128, 128, 128))
assert tuple(_tr(_img).shape) == (3, 64, 64), _tr(_img).shape
_x1, _x2 = _ev(_img), _ev(_img)
assert tuple(_x1.shape) == (3, 64, 64) and torch.equal(_x1, _x2)
assert len(_mean) == 3 and len(_std) == 3
for _mode in ("squash", "letterbox"):
    _t, _e, _, _ = cls_net.build_transforms(64, _mode)
    assert tuple(_t(_img).shape) == (3, 64, 64), (_mode, _t(_img).shape)
    _y1, _y2 = _e(_img), _e(_img)
    assert torch.equal(_y1, _y2), _mode

# ⑤ letterbox 保留长条目标的完整长度：5:1 的图缩入 64×64 后，非填充区域也应是 5:1
_long = Image.new("RGB", (400, 80), (200, 200, 200))
_padded = cls_net.Letterbox(64)(_long)
assert _padded.size == (64, 64), _padded.size
_px = _padded.load()
assert _px[32, 0][0] == 128, "上下应为填充灰"          # 顶部边缘是填充
assert _px[32, 32][0] == 200, "中心应是原图内容"        # 中心是原图

# ⑥ mixup：alpha<=0 为恒等；alpha>0 时形状不变、lam∈[0,1]、两个标签都来自原批次
_x = torch.randn(8, 3, 4, 4)
_y = torch.arange(8) % 3
_xa, _ya, _yb, _lam = tc.mixup_batch(_x, _y, 0)
assert _lam == 1.0 and torch.equal(_xa, _x) and torch.equal(_ya, _y), "alpha=0 应恒等"
_xm, _ya, _yb, _lam = tc.mixup_batch(_x, _y, 0.1)
assert _xm.shape == _x.shape and 0.0 <= _lam <= 1.0
assert torch.equal(_ya, _y) and set(_yb.tolist()) <= set(_y.tolist())

print("CLS-TRAIN TEST: ALL ASSERTIONS PASS")
