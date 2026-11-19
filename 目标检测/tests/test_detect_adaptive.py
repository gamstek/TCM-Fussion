# -*- coding: utf-8 -*-
"""test_detect_adaptive.py — 离线验证 detect_adaptive 的行为（不联网）

2026-09-30 定：**整图 1 次；分块兜底已整条删除**（含空检 2×2 重检）。理由见
`detect_core.detect_adaptive` 的 docstring：分块会切过目标、会往"模型本来认为没东西"
的图上硬塞框（实测淀粉粒 14→36、菌丝 1→6），且与 v7.7 那条"宁可少框、不要制造
无依据的证据"直接冲突。

用桩替换 call_vlm / encode_jpeg_b64，验证：
  ① 整图非空 → 只调 1 次，坐标正确映射回原图
  ② **整图为空 → 仍然只调 1 次**（不再有任何分块重检），且返回空列表
  ③ 目录先验是**显式 opt-in**：传了才注入，未传 / feature 为空都不注入
  ④ 多区域检测仍可**显式**调用（detect_tiles(regions=...)），但 detect_tiles 不传
     regions 时默认整图 —— 不留"不传就悄悄分块"的可达性陷阱
  ⑤~⑧ 多轮自洽采样（`detect_consensus`，V1 只加票不过滤）：
     ⑤ n=1 与 detect_adaptive **逐框一致**且只调 1 次（默认路径不经任何新逻辑）
     ⑥ 4 轮同一个框 + 1 轮另一个框 → 2 簇、票数 4/1，**1 票的框仍保留**
     ⑦ 聚簇按类别：同位置同类合 1 簇（票 2）；同位置异类分 2 簇
     ⑧ 单轮失败只跳过该轮：N=3 挂 1 轮 → 仍出框，`rollout_n` 如实记 2（不冒充满轮），
        且**该轮不进 `raw`**（原始框只落成功轮）
  ⑩ 逐轮原始框（`raw`，V3 的数据源）：与聚簇结果同源、**不带 `votes`**（是聚簇前的形状），
     每轮一条且 `rollout` 编号与成功轮一致 —— 缺了它事后既做不了忽略标签的再聚簇、
     也换不了 thr（2026-10-08 的 V1 因此只能重跑）
  ⑨ 重构对拍：`nms_merge` 改为取 `cluster_dets` 的簇代表后，**链式重叠**（A~B、B~C，
     A≁C）下仍只留 A、C —— 与原贪心实现逐框一致
  ⑪~⑬ stage 2 分类接口（`classify_crop`，两阶段检测的第二步）：小 crop 被**放大**到长边
     768；清单外的类别名**不静默映射**（label=None 且留 raw_label）；「都不是」→ feature_id
     为空；几何先验 ctx 进提示词；`impl="model"` 抛 NotImplementedError（预留钩子）
"""
import os
import threading
from importlib import import_module

import numpy as np

DET_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(DET_DIR)

_core = import_module("detect_core")
# 桩替换目标说明：detect_core 现在是 `from vlm_api import call_vlm`（2026-09-29 P1-3 起不再经
# 入口脚本 vlm_prelabel 转发），故补丁打在 **被测量模块自己的名字** `_core.call_vlm` 上。

BOX_JSON = ('{"objects": [{"label": "草酸钙簇晶", "bbox_2d": [0, 0, 100, 100],'
            ' "confidence": "high", "morphology": "星状"}]}')

calls = []
sent_sizes = []


def make_stub(empty):
    """empty=True 时恒返回空 objects（整图空检）；False 时恒返回一个框。"""
    def stub(cfg_vlm, api_key, b64, prompt=None):
        calls.append(prompt)
        return ('{"objects": []}' if empty else BOX_JSON), None
    return stub


def stub_encode(img, max_side=None, quality=90):
    """桩：与 vlm_api.encode_jpeg_b64 同签名，返回 (b64, scale)。"""
    sent_sizes.append(img.shape[:2])
    return "STUB", 1.0


_seq_n = {"n": 0}
_seq_lock = threading.Lock()


def make_seq_stub(payloads):
    """按**调用次序**依次返回 payloads 的桩（线程安全，超出则复用最后一项）。

    项可以是 JSON 字符串，也可以是 `(None, "错误")` 元组表示该轮失败。
    用于多轮采样的"4 轮同意 + 1 轮异见"这类**结果可确定**的场景：哪条线程拿到哪一项
    不确定，但"恰好 4 次 A / 1 次 B"这个多重集是确定的。
    """
    def stub(cfg_vlm, api_key, b64, prompt=None):
        calls.append(prompt)
        with _seq_lock:
            i = _seq_n["n"]
            _seq_n["n"] += 1
        p = payloads[min(i, len(payloads) - 1)]
        return (p, None) if isinstance(p, str) else p
    return stub


_orig_call, _orig_encode = _core.call_vlm, _core.encode_jpeg_b64
_core.call_vlm, _core.encode_jpeg_b64 = make_stub(True), stub_encode
try:
    img = np.full((1536, 2048, 3), 200, np.uint8)
    args = ({}, "k", ("大黄草酸钙簇晶", "大黄", "草酸钙簇晶"))

    # ① 整图非空 → 1 次调用，坐标映射正确（0~1000 → 原图像素，clamp="blocks" 口径）
    calls.clear(), sent_sizes.clear()
    _core.call_vlm = make_stub(False)
    dets = _core.detect_adaptive(img, *args)
    assert len(calls) == 1, f"整图非空应只调 1 次，实际 {len(calls)}"
    assert len(dets) == 1, dets
    assert dets[0]["tight_bbox"] == [0.0, 0.0, 204.8, 153.6], dets[0]
    assert sent_sizes == [(1152, 1536)], sent_sizes      # 整图 2048x1536 → 降采样一次

    # ② 整图为空 → **仍然只调 1 次**（分块兜底已删除），返回空
    calls.clear(), sent_sizes.clear()
    _core.call_vlm = make_stub(True)
    dets = _core.detect_adaptive(img, *args)
    assert len(calls) == 1, f"整图空检不得再触发分块重检，实际调了 {len(calls)} 次"
    assert dets == [], dets
    assert sent_sizes == [(1152, 1536)], sent_sizes

    # ③ 目录先验 opt-in：显式传三元组才注入；不传、或 feature 为空（送检目录恒为药材名）都不注入
    _core.call_vlm = make_stub(False)
    calls.clear()
    _core.detect_adaptive(img, {}, "k", ("大黄草酸钙簇晶", "大黄", "草酸钙簇晶"))
    assert "【本图先验" in calls[0], calls[0][-600:]
    calls.clear()
    _core.detect_adaptive(img, {}, "k")
    assert "【本图先验" not in calls[0], calls[0][-600:]
    calls.clear()
    _core.detect_adaptive(img, {}, "k", ("虎杖", "虎杖", None))
    assert "【本图先验" not in calls[0], calls[0][-600:]

    # ④ 多区域仍可显式调用；不传 regions 时默认整图（不得悄悄分块）
    calls.clear()
    _core.detect_tiles(img, {}, "k")
    assert len(calls) == 1, f"不传 regions 应默认整图 1 区域，实际 {len(calls)}"
    calls.clear()
    _core.detect_tiles(img, {}, "k", regions=_core.tiles_of(2048, 1536)[:-1])
    assert len(calls) == 4, f"显式传 4 个区域应调 4 次，实际 {len(calls)}"

    # ⑤~⑧ 多轮自洽采样（detect_consensus / V1 只加票不过滤）
    BOX_FAR = ('{"objects": [{"label": "草酸钙簇晶", "bbox_2d": [800, 800, 900, 900],'
               ' "confidence": "medium", "morphology": "远处"}]}')
    BOX_OTHER = ('{"objects": [{"label": "导管", "bbox_2d": [0, 0, 100, 100],'
                 ' "confidence": "high", "morphology": "管状"}]}')

    # ⑤ n=1 与 detect_adaptive 逐框一致，且只调 1 次（默认路径不经任何新逻辑）
    _core.call_vlm = make_stub(False)
    calls.clear()
    d_ad = _core.detect_adaptive(img, {}, "k")
    assert len(calls) == 1, len(calls)
    calls.clear()
    d_c1, raw1 = _core.detect_consensus(img, {}, "k", n=1)
    assert [(d["label"], d["tight_bbox"]) for d in d_c1] == \
           [(d["label"], d["tight_bbox"]) for d in d_ad], (d_ad, d_c1)
    assert len(calls) == 1, f"n=1 应只调 1 次，实际 {len(calls)}"
    assert d_c1[0]["votes"] == 1 and d_c1[0]["rollout_n"] == 1, d_c1[0]
    # ⑩ 原始框：同源、不带 votes（聚簇前形状）、rollout 编号正确
    assert len(raw1) == 1 and raw1[0]["rollout"] == 1, raw1
    o0 = raw1[0]["objects"][0]
    assert o0["tight_bbox"] == d_c1[0]["tight_bbox"] and "votes" not in o0, o0
    assert set(o0) == {"label", "tight_bbox", "confidence", "score", "morphology"}, o0

    # ⑥ 4 轮同框 + 1 轮异框 → 2 簇、票数 4/1；**1 票的框仍保留**（不过滤）
    _seq_n["n"] = 0
    calls.clear()
    _core.call_vlm = make_seq_stub([BOX_JSON, BOX_JSON, BOX_JSON, BOX_JSON, BOX_FAR])
    d5, raw5 = _core.detect_consensus(img, {}, "k", n=5)
    assert len(calls) == 5, f"n=5 应调 5 次，实际 {len(calls)}"
    assert len(d5) == 2, d5
    assert sorted(d["votes"] for d in d5) == [1, 4], d5
    assert all(d["rollout_n"] == 5 for d in d5), d5
    assert [r["rollout"] for r in raw5] == [1, 2, 3, 4, 5], raw5
    assert sum(len(r["objects"]) for r in raw5) == 5, raw5      # 5 轮各 1 框，全在 raw 里

    # ⑦ 聚簇按类别：同位置同类 → 1 簇票 2；同位置异类 → 2 簇
    _seq_n["n"] = 0
    _core.call_vlm = make_seq_stub([BOX_JSON, BOX_JSON])
    d2, _ = _core.detect_consensus(img, {}, "k", n=2)
    assert len(d2) == 1 and d2[0]["votes"] == 2, d2
    _seq_n["n"] = 0
    _core.call_vlm = make_seq_stub([BOX_JSON, BOX_OTHER])
    d2b, _ = _core.detect_consensus(img, {}, "k", n=2)
    assert len(d2b) == 2 and {d["label"] for d in d2b} == {"草酸钙簇晶", "导管"}, d2b
    assert all(d["votes"] == 1 for d in d2b), d2b

    # ⑧ 单轮失败只跳过该轮；rollout_n 如实记成功轮数（不冒充满轮），失败轮不进 raw
    # ⚠️ 只断言**多重集性质**，不断言具体编号：`make_seq_stub` 按"调用到达顺序"发 payload，
    #    并发下失败轮可能落在任意一个索引上（2026-10-08 实测：断言 [1,3] 时三次跑里挂两次）。
    _seq_n["n"] = 0
    _core.call_vlm = make_seq_stub([BOX_JSON, (None, "HTTP 429: 速率限制"), BOX_JSON])
    d3, raw3 = _core.detect_consensus(img, {}, "k", n=3)
    assert len(d3) == 1 and d3[0]["votes"] == 2 and d3[0]["rollout_n"] == 2, d3
    assert len(raw3) == 2 and len({r["rollout"] for r in raw3}) == 2 \
        and all(r["rollout"] in (1, 2, 3) for r in raw3), raw3

    # ⑨ 重构对拍：链式重叠（A~B、B~C 均 ≥0.5，A~C = 0.25）下 nms_merge 仍只留 A、C
    def K(lab, box, sc):
        return {"label": lab, "tight_bbox": list(box), "score": sc}

    chain = [K("导管", (0, 0, 100, 100), 0.92), K("导管", (30, 0, 130, 100), 0.65),
             K("导管", (60, 0, 160, 100), 0.65)]
    kept = _core.nms_merge(chain, thr=0.5)
    assert [d["tight_bbox"] for d in kept] == [[0, 0, 100, 100], [60, 0, 160, 100]], kept
    cl = _core.cluster_dets(chain, thr=0.5)
    assert sorted(len(c["members"]) for c in cl) == [1, 2], cl

    # ⑭ 两阶段 stage 1 接口：prompt 覆盖 + 无标签对象按兜底标签收下（default_label）
    from prompts import ENTITY_LABEL, PROMPT_S1

    NO_LABEL_JSON = ('{"objects": [{"bbox_2d": [0, 0, 100, 100], "confidence": "high",'
                     ' "appearance": "细长"}]}')
    _core.call_vlm = make_stub(False)
    calls.clear()
    _core.call_vlm = make_seq_stub([NO_LABEL_JSON])
    _seq_n["n"] = 0
    d_s1 = _core.detect_adaptive(img, {}, "k", None, prompt=PROMPT_S1,
                                 whitelist={ENTITY_LABEL}, default_label=ENTITY_LABEL)
    assert "粒度" in calls[0], calls[0][:200]          # prompt 覆盖生效（默认 PROMPT 里没有"粒度"）
    assert len(d_s1) == 1 and d_s1[0]["label"] == ENTITY_LABEL, d_s1
    assert d_s1[0]["tight_bbox"] == [0.0, 0.0, 204.8, 153.6], d_s1[0]
    # 不给 default_label 时，无标签对象被白名单丢掉 —— 这正是 S1 走不通默认路径的原因，
    # 也反证了默认路径行为未变
    _seq_n["n"] = 0
    calls.clear()
    _core.call_vlm = make_seq_stub([NO_LABEL_JSON])
    assert _core.detect_adaptive(img, {}, "k") == [], "无 label 的对象应被白名单丢掉"

    # ⑪~⑬ stage 2 分类接口（classify_crop）：形状 / 非法答案 / 放大 / 预留钩子
    _cc = import_module("classify_crop")
    from prompts import NOT_FEATURE_LABEL, UNREADABLE_LABEL

    def cls_stub(label):
        def stub(cfg_vlm, api_key, b64, prompt=None):
            calls.append(prompt)
            return ('{"label": "%s", "second": "", "confidence": "high", "reason": "桩"}'
                    % label), None
        return stub

    small = np.zeros((60, 40, 3), np.uint8)          # 小 crop：应被**放大**到长边 768
    _orig_cvlm = _cc.call_vlm
    try:
        _cc.call_vlm = cls_stub("草酸钙针晶")
        r = _cc.classify_crop(small, {}, "k")
        assert r["label"] == "草酸钙针晶" and r["feature_id"] == "needle_crystal", r
        assert r["source"] == "vlm" and r["sent_hw"] == [768, 512], r    # 放大生效
        _cc.call_vlm = cls_stub(NOT_FEATURE_LABEL)
        r2 = _cc.classify_crop(small, {}, "k")
        assert r2["label"] == NOT_FEATURE_LABEL and r2["feature_id"] == "", r2
        _cc.call_vlm = cls_stub("外星晶体")           # 清单外的名字：**不静默映射**
        r3 = _cc.classify_crop(small, {}, "k")
        assert r3["label"] is None and r3["raw_label"] == "外星晶体" \
            and r3["feature_id"] == "", r3
        _cc.call_vlm = cls_stub("草酸钙针晶")
        r4 = _cc.classify_crop(small, {}, "k", ctx="候选框长宽比 8.2，细长条形")
        assert "候选框长宽比 8.2" in calls[-1], calls[-1][-300:]
        # 两个拒绝类必须分开（2026-10-08 修正）：标签与给代码用的稳定枚举都要对
        _cc.call_vlm = cls_stub(UNREADABLE_LABEL)
        r5 = _cc.classify_crop(small, {}, "k")
        assert r5["label"] == UNREADABLE_LABEL and r5["reject"] == "unreadable" \
            and r5["feature_id"] == "", r5
        _cc.call_vlm = cls_stub(NOT_FEATURE_LABEL)
        assert _cc.classify_crop(small, {}, "k")["reject"] == "not_feature", "拒绝类枚举"
        # 本地清晰度门槛：**默认关**（校准后降级，见 classify_crop 注释）；
        # 开时纯黑图被拦下且**不调 API**、source=gate、标签为「图像不可判」
        n0 = len(calls)
        _cc.call_vlm = cls_stub("草酸钙针晶")
        blank = np.zeros((200, 200, 3), np.uint8)
        r6 = _cc.classify_crop(blank, {}, "k", focus_gate=True)
        assert r6["source"] == "gate" and r6["label"] == UNREADABLE_LABEL \
            and len(calls) == n0, (r6, len(calls) - n0)
        r7 = _cc.classify_crop(blank, {}, "k")
        assert r7["source"] == "vlm" and len(calls) == n0 + 1, r7
        try:
            _cc.classify_crop(small, {}, "k", impl="model")
            raise AssertionError("impl='model' 缺 weights 应抛 ValueError")
        except ValueError:
            pass
    finally:
        _cc.call_vlm = _orig_cvlm
finally:
    _core.call_vlm, _core.encode_jpeg_b64 = _orig_call, _orig_encode

print("DETECT-ADAPTIVE TEST: ALL ASSERTIONS PASS")
