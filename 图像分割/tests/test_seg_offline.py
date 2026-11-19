# -*- coding: utf-8 -*-
"""test_seg_offline.py — segmentation/ 的离线回归入口（不联网、不耗 IGM 额度）。

三条职责（P4-2，2026-09-29；③ 系当日端到端跑到后补）：

① **聚合**：进程内依次跑两个生产自检（`igm_segment` 55 / `igm_segment_batch` 12），
   把两条命令收敛成一条，供根 README 的回归基准引用 —— 之前它们各自独立，缺一条没人会发现。
   （`build_seg_crops` 原为第三项，其 6 条断言随该脚本于 2026-09-29 归档 —— 它是旧裁剪链的
   CLI，与 `seg_manifest.csv` 构成闭环且无活链消费者。`igm_segment_batch` 住在 `cli/`、
   不进 import 注入点，故按文件路径加载；`igm_segment` 仍走 import。）

② **补缺口**：LS 分割审核的标注解析原语此前**零离线测试**，
   而它正是 UI 画笔审核链（`ui/seg_stage`、`ui/detect_stage`）复用的关键一步：
   LS 的 annotation → 二值 mask（2026-09-29 P4-9 起这些原语在 `segmentation/lib/ls_review.py`）。
   这里用合成 annotation 锁定它声称支持的三种载体
   （polygon 的三种写下法 / RLE 的 dict 与逗号串 / 多结果并集），特别是
   `rle_to_mask` 的**列优先（Fortran order）朝向** —— 转置错误不会报错，
   只会把 mask 静默旋转，是这类代码最难发现的一类回归。

③ **路径派生落点**：`igm_segment` 移入 `lib/` 后仍按旧深度推导路径（当日端到端跑到才发现）——
   本体表找不到、结果目录错落一层，而 `--selftest` 走本地合成路径、从不读本体表，
   故两条生产自检全绿也拦不住。见下方的 `run_path_derivation_tests`。

**已知未覆盖**：`_ls_brush_decode` 的首选通路依赖 venv 里的
`label_studio_converter.brush.decode_rle`，其返回值随该库版本变化，故只做
"不抛异常 + 形状正确"的冒烟，不锁逐像素；真实像素级校验需要真 LS 标注（不在本测试范围）。

用法（项目根起，用项目 venv 的解释器）：
  python -X utf8 标注体系\\segmentation\\tests\\test_seg_offline.py
"""
import os
import sys
from importlib import import_module

import numpy as np

import paths  # noqa: E402 路径唯一权威（③ 用它断言派生路径的落点）

SEG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_fail = []
_n_pass = 0


def check(name, cond, extra=""):
    global _n_pass
    if cond:
        _n_pass += 1
        print(f"  [PASS] {name}")
    else:
        _fail.append(f"{name} {extra}")
        print(f"  [FAIL] {name} {extra}")


# ---------------------------------------------------------------- ① 生产自检聚合
def _load_cli(name):
    """按**文件路径**加载 CLI 入口。

    P4-4 起 `cli/` 不进 import 注入点（"只做入口"的目录不该被 import），故不能 import_module；
    这与 detection/tests 下加载 `vlm_prelabel` / `import_to_labelstudio` 的做法一致。
    """
    import importlib.util
    p = os.path.join(SEG_DIR, "cli", name + ".py")
    spec = importlib.util.spec_from_file_location(name, p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_production_selftests():
    """进程内跑两个生产 --selftest。各自打印自己的 [PASS]/[FAIL]，此处只汇总返回码。"""
    print("① 生产自检聚合（igm_segment / igm_segment_batch）")
    bad = []
    rc = import_module("igm_segment").selftest()
    print(f"  → igm_segment.selftest() 返回 {rc}")
    if rc != 0:
        bad.append("igm_segment")
    # igm_segment_batch 住在 cli/（不进注入点），按文件路径加载
    rc = _load_cli("igm_segment_batch").selftest()
    print(f"  → igm_segment_batch.selftest() 返回 {rc}")
    if rc != 0:
        bad.append("igm_segment_batch")
    if bad:
        _fail.append(f"生产自检未通过: {bad}")
    return bad


# ---------------------------------------------------------------- ② 标注解析原语
W, H = 20, 10
QUAD = [[1, 1], [9, 1], [9, 8], [1, 8]]          # 左半区矩形，四顶点与像素边界错开一格
# rle_to_mask 用**列优先**：h=4/w=4 的"左半边"在列优先下是前 8 个像素
RLE_LEFT_HALF = [0, 8, 8]


def run_annotation_parser_tests():
    """LS 分割审核的 annotation → mask 解析（`lib/ls_review.py`）。"""
    print("② 标注解析原语（lib/ls_review，此前零覆盖）")
    exp = import_module("ls_review")

    def ann(*results):
        return {"result": list(results)}

    def res(**over):
        base = {"type": "polygonlabels", "original_width": W, "original_height": H,
                "value": {"points": QUAD}}
        base.update(over)
        return base

    # --- polygon 的三种写下法必须给出**同一个** mask（旧代码把 path 的 str/list 两形态
    #     与新版 points 列表分三处处理，任一处写错都会静默漏掉一部分标注）
    m_list = exp.brush_annotation_to_mask(ann(res()))
    m_shape = exp.brush_annotation_to_mask(ann(res(
        type="rectanglelabels", value={"shape": "polygon", "points": {"path": QUAD}})))
    m_str = exp.brush_annotation_to_mask(ann(res(
        type="rectanglelabels",
        value={"shape": "polygon", "points": {"path": "1,1 9,1 9,8 1,8"}})))
    m_fmt = exp.brush_annotation_to_mask(ann(res(
        type="rectanglelabels", value={"format": "polygon", "points": QUAD})))

    check("polygon: points 列表可解析", m_list is not None and m_list.shape == (H, W),
          f"{None if m_list is None else m_list.shape}")
    check("polygon: 三种写法 mask 逐像素一致",
          all(m is not None and np.array_equal(m, m_list)
              for m in (m_shape, m_str, m_fmt)),
          f"一致={[None if m is None else bool(np.array_equal(m, m_list)) for m in (m_shape, m_str, m_fmt)]}")
    check("polygon: 值域为 0/255", set(np.unique(m_list)) <= {0, 255}, f"{np.unique(m_list)}")
    # 内部点在前景、外部点在背景（形状与位置都对，不只是"非空"）
    check("polygon: 框内点为前景、框外点为背景",
          m_list[4, 4] == 255 and m_list[4, 15] == 0 and m_list[0, 0] == 0,
          f"in={m_list[4, 4]} out={m_list[4, 15]} corner={m_list[0, 0]}")

    # --- RLE 两种载体（dict.counts / 逗号串）走 rle_to_mask，落点必须一致
    m_rle_dict = exp.brush_annotation_to_mask(ann({
        "type": "brushlabels", "original_width": 4, "original_height": 4,
        "value": {"format": "rle", "rle": {"counts": RLE_LEFT_HALF}}}))
    m_rle_str = exp.brush_annotation_to_mask(ann({
        "type": "brushlabels", "original_width": 4, "original_height": 4,
        "value": {"format": "brushRLE", "rle": "0,8,8"}}))
    check("RLE: dict.counts 与逗号串结果一致",
          m_rle_dict is not None and m_rle_str is not None
          and np.array_equal(m_rle_dict, m_rle_str),
          f"{None if m_rle_dict is None else 'ok'} {None if m_rle_str is None else 'ok'}")
    check("RLE: 列优先朝向正确（左半边而非上半边）",
          m_rle_dict is not None and (m_rle_dict[:, :2] == 255).all()
          and (m_rle_dict[:, 2:] == 0).all(),
          f"{None if m_rle_dict is None else m_rle_dict.tolist()}")
    # 直接锁原语：转置错了这里就会红（值域是 0/1，不是 0/255）
    raw = exp.rle_to_mask(RLE_LEFT_HALF, (4, 4))
    check("RLE: rle_to_mask 直接调用朝向正确",
          raw.shape == (4, 4) and (raw[:, :2] > 0).all() and raw[:, 2:].sum() == 0,
          f"{raw.tolist()}")

    # --- 多结果合并：两个分离的方块 → 两处都是前景
    m_union = exp.brush_annotation_to_mask(ann(
        res(value={"points": [[1, 1], [5, 1], [5, 5], [1, 5]]}),
        res(value={"points": [[10, 4], [18, 4], [18, 8], [10, 8]]})))
    check("并集: 两条 result 的标注都保留",
          m_union is not None and m_union[3, 3] == 255 and m_union[6, 12] == 255
          and m_union[0, 0] == 0,
          f"{None if m_union is None else 'shape ok'}")

    # --- 边界：无 result / 缺尺寸 → None（不能静默产出空 mask 冒充成功）
    check("边界: 无 result 返回 None", exp.brush_annotation_to_mask({"result": []}) is None)
    check("边界: 缺 original_size 返回 None",
          exp.brush_annotation_to_mask(ann({
              "type": "polygonlabels", "value": {"points": QUAD}})) is None)
    check("边界: 顶点不足 3 个不算多边形",
          exp.brush_annotation_to_mask(ann(res(value={"points": [[1, 1], [5, 5]]}))) is None)

    # --- smoke：brush RLE 列表通路（首选解依赖 venv 里的 label_studio_converter，
    #     返回值随版本变化 → 只验"不抛异常且形状正确或诚实返回 None"）
    m_brush = exp.brush_annotation_to_mask(ann({
        "type": "brushlabels", "original_width": 4, "original_height": 4,
        "value": {"format": "brushRLE", "rle": [0, 0, 2, 255, 2, 0]}}))
    check("brush: RLE 列表通路不抛异常且形状正确或返回 None",
          m_brush is None or m_brush.shape == (4, 4),
          f"{None if m_brush is None else m_brush.shape}")


# ---------------------------------------------------------------- ③ 派生路径的落点
def run_path_derivation_tests():
    """锁"模块移入 `lib/` 后仍按旧深度算路径"这类回归（2026-09-29 端到端跑才暴露）。

    真实现象：`igm_segment` 移入 `lib/` 后，`TAXONOMY_PATH` 仍按"文件在 `segmentation/` 下"
    推导 → 打开 `segmentation/feature_taxonomy.json` 抛 FileNotFoundError，IGM 阶段整段失败；
    `RESULTS_SEG`/`IGM_DIR` 同时错落一层。两条生产自检**都没拦住**，因为 `--selftest` 走本地
    合成路径、从不查本体表。故这里花几毫秒把两个常量钉死。
    """
    print("③ 派生路径落点（移入 lib/ 后不得按旧深度算）")
    seg = import_module("igm_segment")

    check("TAXONOMY_PATH 指向 项目根/feature_taxonomy.json",
          os.path.isfile(seg.TAXONOMY_PATH)
          and os.path.normcase(os.path.dirname(os.path.abspath(seg.TAXONOMY_PATH)))
          == os.path.normcase(os.path.abspath(paths.PROJECT_ROOT)),
          seg.TAXONOMY_PATH)
    try:
        desc = seg.load_feature_desc("cluster_crystal")
    except BaseException as e:      # 旧实现此处 open() 抛 FileNotFoundError / 或 sys.exit
        desc = "<%s: %s>" % (type(e).__name__, e)
    check("load_feature_desc 真读到簇晶描述（非空）",
          isinstance(desc, str) and len(desc.strip()) > 10, repr(desc)[:70])
    check("RESULTS_SEG/IGM_DIR 落在 项目根/results/segmentation 下",
          os.path.normcase(os.path.abspath(seg.IGM_DIR)).startswith(
              os.path.normcase(os.path.abspath(paths.SEG_RESULTS)) + os.sep),
          "%s | %s" % (seg.RESULTS_SEG, seg.IGM_DIR))


def run_box_association_tests():
    """中心空洞、细长结构与误关联：只改QA，不改变mask像素；原B默认保留。"""
    import cv2
    from unittest.mock import patch
    from igm_segment import generated_to_mask, apply_box_prior
    box = [20, 20, 100, 100]
    canvas = {"top_ratio": 0, "left_ratio": 0, "cw_ratio": 1, "ch_ratio": 1}

    def qa(mask, mode="region_component", target=box):
        return generated_to_mask(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), None, mask.shape,
                                 canvas, invert=False, clean=False, box=target, box_association=mode)

    bend = np.zeros((120, 120), np.uint8)
    cv2.polylines(bend, [np.array([[30, 30], [30, 90], [90, 90]])], False, 255, 7)
    old, oi = qa(bend, "center_point")
    new, ni = qa(bend)
    check("区域关联: L形中心背景不再单独拒绝", oi["status"] == "ok" and ni["status"] == "ok"
          and ni["center_comp_ratio"] == 0 and ni["association_candidates"] == 1, str(ni))
    check("区域关联: 改QA不改mask像素", np.array_equal(old, new))

    thin = np.zeros_like(bend)
    cv2.line(thin, (28, 28), (92, 92), 255, 3)
    _, ti = qa(thin)
    check("区域关联: 细长结构不受30%前景占框下限误拦", ti["status"] == "ok"
          and ti["center_comp_ratio"] < 0.3 and ti["associated_bbox_ratio"] >= 0.3, str(ti))

    u = np.zeros_like(bend)
    cv2.polylines(u, [np.array([[25, 25], [25, 95], [95, 95], [95, 25]])], False, 255, 3)
    _, ui = qa(u)
    check("区域关联: 中心区域也空时唯一框内域可回退", ui["status"] == "ok"
          and not ui["center_region_hit"] and ui["association_method"] == "unique_box_component", str(ui))

    tiny = np.zeros_like(bend);cv2.circle(tiny, (60, 60), 4, 255, -1)
    _, si = qa(tiny)
    check("区域关联: 中心小斑点仍拒绝", si["status"] == "review"
          and "TARGET_TOO_SMALL" in si["review_reason_codes"], str(si))
    multiple = np.zeros_like(bend)
    cv2.circle(multiple, (45, 60), 12, 255, -1);cv2.circle(multiple, (80, 60), 12, 255, -1)
    mm, mi = qa(multiple)
    check("区域关联: 多个候选保留歧义而非取最大", mi["status"] == "review"
          and mi["association_candidates"] == 2 and cv2.connectedComponents(mm)[0] == 3)
    touch = np.zeros_like(bend);cv2.rectangle(touch, (35, 0), (85, 85), 255, -1)
    _, ei = qa(touch, target=[30, 0, 90, 90])
    check("区域关联: 疑似截断不放行完整尺寸", ei["status"] == "review"
          and "POSSIBLE_TRUNCATION" in ei["review_reason_codes"])
    _, zi = qa(np.zeros_like(bend))
    check("区域关联: 空mask仍为empty", zi["status"] == "empty")
    oversized = np.zeros_like(bend);cv2.rectangle(oversized, (15, 15), (105, 105), 255, -1)
    _, bi = qa(oversized)
    check("区域关联: 大块外溢仍review", bi["status"] == "review"
          and "ASSOCIATED_COMPONENT_TOO_LARGE" in bi["review_reason_codes"])
    outsider = bend.copy();cv2.rectangle(outsider, (0, 0), (8, 8), 255, -1)
    _, xi = apply_box_prior(outsider, box, association="region_component")
    check("区域关联: 框外邻粒仍删除", xi["dropped_components"] == 1 and xi["association_candidates"] == 1)
    batch = _load_cli("igm_segment_batch")
    captured = []
    with patch.object(batch.ig, "main", side_effect=lambda: captured.extend(sys.argv) or 1):
        batch.run_igm({"png": "unused.png", "feature_id": "fiber"}, "unused", box_association="region_component")
    check("区域关联: 批入口真实透传单图参数", "--box-association" in captured
          and captured[captured.index("--box-association") + 1] == "region_component")


def main():
    bad = run_production_selftests()
    run_annotation_parser_tests()
    run_path_derivation_tests()
    run_box_association_tests()

    n_own = _n_pass + len(_fail)
    print(f"\n新增断言: {_n_pass}/{n_own}")
    if bad or _fail:
        print(f"SEG-OFFLINE TEST: FAIL（生产自检未过 {len(bad)} 个，新增断言失败 {len(_fail)} 条）")
        for f in _fail:
            print(f"  - {f}")
        return 1
    print("SEG-OFFLINE TEST: ALL ASSERTIONS PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
