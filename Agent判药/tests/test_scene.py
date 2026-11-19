# -*- coding: utf-8 -*-
"""test_scene — 场景证据（三层模型第 2/3 层）单测。

覆盖：
  1) scene_attributes.build_scene：每视野计数、丰度档位、质心共线性
     （rows / scattered / insufficient）、尺寸分布；
  2) rule_engine.scene_fit + decide(scene)：平分排序变化，conf / support_count /
     verdict / needs_review 不变（铁律）；scene=None 回归锚点（与旧排序一致）；
  3) scene_attributes.merge_vlm_scene：VLM 与几何并存记录、VLM 优先参与拟合；
  4) agent_loop.augment_and_rescore 平分自动触发场景判读（FakeVLM，不发网络），
     含失败静默降级路径。
直接运行： python tests/test_scene.py
"""
import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "scripts"))

import scene_attributes as sa  # noqa: E402
from agent_loop import augment_and_rescore  # noqa: E402
from rule_engine import (  # noqa: E402
    SCENE_CONTRA_FACTOR, SCENE_MATCH_FACTOR,
    KnowledgeBase, _scene_location_factor, decide, pool_evidence, scene_fit)

KB = KnowledgeBase(HERE.parent / "data" / "herb_criteria.json")
PASS = 0


def check(name, cond):
    global PASS
    if not cond:
        raise AssertionError(f"用例失败: {name}")
    PASS += 1
    print(f"  [PASS] {name}")


def rec(stem, iid, value=55.0, center=None, half=25, feature="cluster_crystal", extra=None):
    """构造一条测量记录（契约对齐 evidence_pool/rule_engine）。"""
    r = {"image_stem": stem, "instance_id": iid, "feature": feature,
         "calibration_status": "ok", "calibration_px_per_um": 11.7,
         "primary_dimension": "feret_max_um", "primary_value_um": value,
         "feret_max_um": value, "conf": 0.9, "flags": [],
         "defect_count": 5, "solidity": 0.8}
    if center is not None:
        r.update({"bbox_x": center[0] - half, "bbox_y": center[1] - half,
                  "bbox_w_px": 2 * half, "bbox_h_px": 2 * half})
    if extra:
        r.update(extra)
    return r


# ---------------------------------------------------------------- 1) build_scene
def test_build_scene():
    print("[1] scene_attributes.build_scene 几何提取")
    # 共线 → rows
    recs = [rec("1", i, center=(100 * i, 100 * i)) for i in range(1, 5)]
    sc = sa.build_scene(recs)
    check("共线4点: verdict=rows", sc["arrangement"][0]["verdict"] == "rows")
    check("共线4点: rms_ratio≤0.25", sc["arrangement"][0]["rms_ratio"] is not None
          and sc["arrangement"][0]["rms_ratio"] <= sa.ROW_RMS_RATIO)
    check("每视野计数", sc["per_fov_counts"] == {"1": 4})
    check("丰度档位: 中位4个/视野 → 众多", sc["abundance_obs"] == "众多")
    check("尺寸分布", sc["size_stats"] == {"min": 55.0, "max": 55.0, "median": 55.0})
    check("来源标记 geometry", sc["sources"] == {"arrangement": "geometry",
                                                 "abundance": "geometry"})

    # 散乱 → scattered
    recs = [rec("1", 1, center=(100, 100)), rec("1", 2, center=(260, 180)),
            rec("1", 3, center=(140, 320)), rec("1", 4, center=(320, 260))]
    sc = sa.build_scene(recs)
    check("散乱4点: verdict=scattered", sc["arrangement"][0]["verdict"] == "scattered")

    # 实例数不足 → insufficient
    recs = [rec("1", 1, center=(100, 100)), rec("1", 2, center=(200, 200))]
    sc = sa.build_scene(recs)
    check("2点: verdict=insufficient", sc["arrangement"][0]["verdict"] == "insufficient"
          and sc["arrangement"][0]["rms_ratio"] is None)

    # 丰度档位映射
    check("丰度: 1个/视野 → 可见",
          sa.build_scene([rec("1", 1), rec("2", 1), rec("3", 1)])["abundance_obs"] == "可见")
    check("丰度: 12个/视野 → 极多",
          sa.build_scene([rec("1", i) for i in range(1, 13)])["abundance_obs"] == "极多")
    check("空记录: abundance_obs=None", sa.build_scene([])["abundance_obs"] is None)


# ------------------------------------------- 2) scene_fit + decide 平分排序铁律
def _tie_records():
    """虎杖样品案例复刻：3 视野 × 1 实例（48.2/54.7/59.9μm），三方 conf 0.60 平分。"""
    vals = [59.9, 48.2, 54.7]
    return [rec(str(i), 1, value=v) for i, v in enumerate(vals, 1)]


def _rows_scene():
    """3 视野 × 3 共线实例（值 52/55/58）→ rows + 丰度'众多'。"""
    recs = []
    for si, stem in enumerate(("A", "B", "C")):
        for i, v in enumerate((52.0, 55.0, 58.0), 1):
            recs.append(rec(stem, i, value=v,
                            center=(100 * i + 50 * si, 100 * i + 50 * si)))
    return sa.build_scene(recs)


def test_scene_fit_and_decide():
    print("[2] rule_engine.scene_fit + decide(scene) 平分排序")
    recs = _tie_records()
    cards = pool_evidence(recs)

    # 回归锚点：scene=None → 与旧行为逐字段一致（rank0=虎杖，无 scene_fit 键）
    d0 = decide(KB, cards)
    check("无scene: rank0=虎杖", d0["ranking"][0]["id"] == "虎杖")
    check("无scene: 三方 conf 均 0.60",
          all(r["confidence"] == 0.6 for r in d0["ranking"][:3]
              if r["id"] in ("虎杖", "马齿苋", "G1")))
    check("无scene: 不写 scene_fit 键", all("scene_fit" not in r for r in d0["ranking"]))
    check("无scene: verdict=insufficient_evidence + needs_review",
          d0["verdict"] == "insufficient_evidence" and d0.get("needs_review"))

    # 场景证据：rows + 丰度'众多' → G1（morph 含'排列成行'）升至 rank0
    scene = _rows_scene()
    check("rows scene 构造成功", all(a["verdict"] == "rows" for a in scene["arrangement"]))
    d = decide(KB, cards, scene=scene)
    check("有scene: rank0=G1（排列成行 + 丰度相符）", d["ranking"][0]["id"] == "G1")
    fits = {r["id"]: r.get("scene_fit") for r in d["ranking"]}
    check("有scene: G1 fit=+2", fits["G1"] == 2)
    check("有scene: 马齿苋 fit=+1", fits["马齿苋"] == 1)
    check("有scene: 虎杖 fit=0（丰度'极多'高于观测一档不计分）", fits["虎杖"] == 0)
    # 铁律：裁决语义不变
    check("铁律: verdict 不变", d["verdict"] == d0["verdict"] == "insufficient_evidence")
    check("铁律: needs_review 不变", bool(d.get("needs_review")))
    check("铁律: support_feature_count 不变", d["support_feature_count"] == 1)
    check("铁律: 三方 conf 仍 0.60",
          all(next(r for r in d["ranking"] if r["id"] == i)["confidence"] == 0.6
              for i in ("虎杖", "马齿苋", "G1")))
    check("notes: 追加场景排序说明",
          any("场景证据排序" in n for n in d["notes"]))
    check("notes: 说明不改置信度与门槛",
          any("不改变置信度与证据门槛" in n for n in d["notes"]))

    # scene_fit 直测：散在 morph 的候选遇 rows 被罚（KB 内"散在"表述判据）
    cand_白术 = {"criteria": KB.herbs["白术"]["criteria"]}  # 针晶'存在于薄壁细胞中'无散在词
    f0, _ = scene_fit(cand_白术, scene, ["needle_crystal"])
    check("scene_fit: 无结晶判据的候选 fit=0", f0 == 0)
    # VLM 覆盖后的 scene：abundance_obs 被覆盖为'极多' → 虎杖丰度维转 +1
    merged = sa.merge_vlm_scene(scene, [{"fov": "A", "crystal_location": "intracellular",
                                         "arrangement": "rows", "abundance": "very_many",
                                         "reason": "fake"}])
    f_hu, r_hu = scene_fit({"criteria": KB.herbs["虎杖"]["criteria"]}, merged,
                           ["cluster_crystal"])
    check("VLM覆盖: 虎杖丰度维转 +1（观测'极多'≥声明'极多'）", f_hu == 1
          and any("丰度" in x for x in r_hu))

    # 宿主语境乘数（第 2 层评分通道）：_scene_location_factor 直测
    crit_intra = {"morph": "薄壁细胞中含草酸钙簇晶"}
    crit_free = {"morph": "草酸钙簇晶散在"}
    crit_none = {"morph": "草酸钙簇晶极多，较大"}   # 虎杖式：正文无细胞/散在表述
    sc_intra = {"vlm": {"crystal_location": "intracellular", "crystal_location_conf": 0.9}}
    sc_free = {"vlm": {"crystal_location": "free", "crystal_location_conf": 0.9}}
    check("乘数: intracellular ×『细胞内含』→ 1.10",
          _scene_location_factor(crit_intra, sc_intra) == SCENE_MATCH_FACTOR)
    check("乘数: intracellular ×『散在』→ 0.60（相悖）",
          _scene_location_factor(crit_free, sc_intra) == SCENE_CONTRA_FACTOR)
    check("乘数: intracellular × 无表述（虎杖式）→ 1.0",
          _scene_location_factor(crit_none, sc_intra) == 1.0)
    check("乘数: free ×『散在』→ 1.10",
          _scene_location_factor(crit_free, sc_free) == SCENE_MATCH_FACTOR)
    check("乘数: free ×『细胞内含』→ 0.60（相悖）",
          _scene_location_factor(crit_intra, sc_free) == SCENE_CONTRA_FACTOR)
    check("乘数: VLM 置信度 0.5 < 门限 0.70 → 1.0（只记录不评分）",
          _scene_location_factor(
              crit_intra, {"vlm": {"crystal_location": "intracellular",
                                   "crystal_location_conf": 0.5}}) == 1.0)
    check("乘数: mixed → 1.0",
          _scene_location_factor(
              crit_intra, {"vlm": {"crystal_location": "mixed",
                                   "crystal_location_conf": 0.9}}) == 1.0)
    check("乘数: 纯几何 scene（无 vlm 块）→ 1.0",
          _scene_location_factor(crit_intra, {"arrangement": []}) == 1.0)

    # 评分通道端到端：白芍样本 + VLM 相悖判读 → conf 被乘数压低、D8b 放行被拦下
    recs_bs = [rec(str(i), 1, value=v) for i, v in enumerate([12.5, 13.0, 12.7], 1)]
    cards_bs = pool_evidence(recs_bs)
    d_base = decide(KB, cards_bs)
    check("评分通道: 基线白芍 conf 1.00（D8b 放行 confirmed）",
          bool(d_base["conclusion"]) and d_base["conclusion"]["id"] == "白芍"
          and d_base["verdict"] == "confirmed")
    d_contra = decide(KB, cards_bs, scene=sc_free)
    r_bs = next(r for r in d_contra["ranking"] if r["id"] == "白芍")
    check("评分通道: 相悖乘数把白芍 conf 压到 0.60", abs(r_bs["confidence"] - 0.6) < 1e-6)
    check("评分通道: 白芍判据 scene_factor=0.6", r_bs["checks"][0]["scene_factor"] == 0.6)
    check("评分通道: D8b 放行被拦下（conf < 0.75）→ 证据不足",
          d_contra["verdict"] == "insufficient_evidence")
    check("铁律: support_count 不变（仍 1 类，不因场景证据虚增）",
          d_contra["support_feature_count"] == 1)
    d_match = decide(KB, cards_bs, scene=sc_intra)
    r_bsm = next(r for r in d_match["ranking"] if r["id"] == "白芍")
    check("评分通道: 相符乘数不越界（conf 仍 ≤1.00）", r_bsm["confidence"] <= 1.0)
    check("评分通道: scene_factor=1.1 记录", r_bsm["checks"][0]["scene_factor"] == 1.1)
    check("评分通道: 无 scene 时 scene_factor=None（回归锚点）",
          d_base["ranking"][0]["checks"][0]["scene_factor"] is None
          or d_base["ranking"] and all(
              c.get("scene_factor") is None or c.get("scene_factor") == 1.0
              for r in d_base["ranking"] for c in r["checks"]))


# ------------------------------------------------------------ 3) merge_vlm_scene
def test_merge_vlm_scene():
    print("[3] scene_attributes.merge_vlm_scene VLM/几何并存")
    geo = sa.build_scene([rec("1", i, center=(100 * i, 100 * i)) for i in range(1, 4)]
                         + [rec("2", 1, value=50.0, center=(80, 90)),
                            rec("2", 2, value=52.0, center=(160, 88))])
    answers = [
        {"fov": "1", "crystal_location": "intracellular", "arrangement": "rows",
         "abundance": "very_many", "reason": "晶体位于细胞内"},
        {"fov": "2", "crystal_location": "free", "arrangement": "scattered",
         "abundance": "many", "reason": "r2"},
    ]
    m = sa.merge_vlm_scene(geo, answers)
    by_fov = {a["fov"]: a for a in m["arrangement"]}
    check("并存: 几何 verdict 保留", by_fov["1"]["verdict"] == "rows"
          and by_fov["2"]["verdict"] == "insufficient")
    check("并存: VLM 判读写入 vlm_verdict", by_fov["1"].get("vlm_verdict") == "rows"
          and by_fov["2"].get("vlm_verdict") == "scattered")
    check("来源: arrangement→vlm", m["sources"]["arrangement"] == "vlm")
    check("来源: abundance→vlm", m["sources"]["abundance"] == "vlm")
    check("丰度: VLM 覆盖为'极多'，几何观测保留", m["abundance_obs"] == "极多"
          and m.get("abundance_obs_geometry") == "众多")
    check("宿主语境: 多数票汇总", m["vlm"]["crystal_location"] in ("intracellular", "free"))
    check("vlm 块: n_calls=2", m["vlm"]["n_calls"] == 2)
    check("入参不被修改", geo["sources"]["abundance"] == "geometry"
          and "vlm" not in geo)
    # unknown 判读不改来源
    m2 = sa.merge_vlm_scene(sa.empty_scene(),
                            [{"fov": "9", "crystal_location": "unknown",
                              "arrangement": "unknown", "abundance": "unknown"}])
    check("全 unknown: 来源保持 geometry", m2["sources"] == {"arrangement": "geometry",
                                                             "abundance": "geometry"})


# --------------------------------- 4) augment_and_rescore 平分触发场景判读（FakeVLM）
class FakeSceneVLM:
    """场景判读 Fake（不发网络）。fail=True 模拟调用失败。"""

    model = "fake-vlm"

    def __init__(self, fail=False):
        self.fail = fail
        self.scene_calls = []

    def scene_check(self, image_path):          # 2026-09-23: 不再接收候选 morph 提示
        self.scene_calls.append(str(image_path))
        if self.fail:
            return {"crystal_location": "unknown", "crystal_location_raw": "unknown",
                    "cell_wall_visible": False, "guarded": False,
                    "arrangement": "unknown", "abundance": "unknown", "conf": 0.0,
                    "reason": "vlm_error: boom", "ok": False}
        return {"crystal_location": "intracellular", "crystal_location_raw": "intracellular",
                "cell_wall_visible": True, "guarded": False,
                "arrangement": "rows",
                "abundance": "very_many", "conf": 0.9, "reason": "fake", "ok": True}

    def presence_check(self, *a, **k):  # 场景触发路径不应调用
        raise AssertionError("presence_check 不应被调用")


def test_augment_scene_trigger():
    print("[4] agent_loop.augment_and_rescore 平分触发场景判读")
    recs = _tie_records()
    cards = pool_evidence(recs)
    d0 = decide(KB, cards)
    check("前置: d0 平分需复核", d0.get("needs_review") is True)

    fake = FakeSceneVLM()
    image_index = {"1": Path("1.jpg"), "2": Path("2.jpg"), "3": Path("3.jpg")}
    d, notes, meta, scene = augment_and_rescore(
        KB, cards, d0, fake, image_index, recs, max_calls=6)
    check("触发: 每视野一次场景判读（3 次）", meta["scene_vlm_calls"] == 3
          and meta["vlm_calls"] == 3)
    check("触发: mode=rule+vlm", meta["mode"] == "rule+vlm")
    check("解耦: 场景判读只传图像路径，不再带候选 morph 提示",
          fake.scene_calls == ["1.jpg", "2.jpg", "3.jpg"])
    check("归并: scene 带 vlm 块", scene is not None and scene.get("vlm", {}).get("n_calls") == 3)
    check("归并: 判读置信度取最小（0.9 ≥ 评分门限 0.70）",
          scene["vlm"].get("crystal_location_conf") == 0.9)
    check("归并: 来源切换为 vlm", scene["sources"]["arrangement"] == "vlm")
    check("重评: notes 记录场景判读与重跑说明",
          any("VLM 场景判读" in n for n in notes)
          and any("并入 VLM 场景证据" in n for n in notes))
    check("铁律: 重评后裁决语义不变",
          d["verdict"] == "insufficient_evidence" and d.get("needs_review") is True
          and d["support_feature_count"] == 1)

    # 失败静默降级
    fake_bad = FakeSceneVLM(fail=True)
    d_b, notes_b, meta_b, scene_b = augment_and_rescore(
        KB, cards, decide(KB, cards), fake_bad, image_index, recs, max_calls=6)
    check("降级: 失败标记 degraded", meta_b["degraded"] is True)
    check("降级: scene 保持几何来源（未并入任何 VLM 判读）",
          scene_b is None or scene_b["sources"]["arrangement"] == "geometry")
    check("降级: notes 记录失败", any("场景判读失败" in n for n in notes_b))

    # 兜底: scene=None 时用空场景归并
    fake2 = FakeSceneVLM()
    _, _, meta2, scene2 = augment_and_rescore(
        KB, cards, decide(KB, cards), fake2, image_index, recs, max_calls=6, scene=None)
    check("兜底: scene=None 时构造空场景并归并", scene2 is not None
          and scene2.get("vlm", {}).get("n_calls") == 3 and meta2["scene_vlm_calls"] == 3)

    # 强制触发（2026-09-22 用户决议）：confirmed 且未标 needs_review 也必须做场景判读
    recs_dh = ([rec("1", i, value=v) for i, v in
                enumerate([120, 132, 115, 128, 126, 118, 130, 138], 1)]
               + [rec("1", i, value=v, feature="starch_grain",
                      extra={"roundness": 0.8, "dt_peak_count": 4})
                  for i, v in enumerate([18, 22, 15, 30, 12, 25], 11)]
               + [rec("1", 17, value=120, feature="vessel",
                      extra={"aspect_ratio": 4.5, "tortuosity": 1.6})])
    cards_dh = pool_evidence(recs_dh)
    d_dh = decide(KB, cards_dh)
    check("强制触发前置: 大黄 confirmed 且未标 needs_review",
          d_dh["verdict"] == "confirmed" and not d_dh.get("needs_review"))
    fake3 = FakeSceneVLM()
    d_dh2, notes_dh2, meta_dh2, scene_dh2 = augment_and_rescore(
        KB, cards_dh, d_dh, fake3, {"1": Path("1.jpg")}, recs_dh, max_calls=6)
    check("强制触发: confirmed 样本仍发起场景判读", meta_dh2["scene_vlm_calls"] == 1)
    check("强制触发: 场景已并入并重评", scene_dh2 is not None
          and scene_dh2.get("vlm", {}).get("n_calls") == 1
          and any("并入 VLM 场景证据" in n for n in notes_dh2))
    check("强制触发: 裁决仍 confirmed（大黄 morph 无细胞表述，乘数 1.0）",
          d_dh2["verdict"] == "confirmed")

    # 无结晶检出: 不发起场景判读（scene_check 只对结晶类特征有意义）
    recs_min = [rec("1", 1, value=90, feature="mineral_feature")]
    fake4 = FakeSceneVLM()
    _, _, meta_min, scene_min = augment_and_rescore(
        KB, pool_evidence(recs_min), decide(KB, pool_evidence(recs_min)),
        fake4, {"1": Path("1.jpg")}, recs_min, max_calls=6)
    check("无结晶检出: 不发起场景判读", meta_min["scene_vlm_calls"] == 0
          and fake4.scene_calls == [] and scene_min is None)


# ---------------- 5) scene_check 提示词解耦 + 视觉前提守卫（真 VLMClient，不发网络）
def test_scene_prompt_guard():
    print("[5] VLMClient.scene_check 提示词解耦 + cell_wall_visible 守卫")
    import agent_loop
    check("解耦: agent_loop 不再保留候选形态提示函数 _scene_hint",
          not hasattr(agent_loop, "_scene_hint"))
    c = object.__new__(agent_loop.VLMClient)      # 绕过配置加载
    c.model, c.temperature, c.max_tokens = "fake-model", 0.1, 512
    captured = {}

    def fake_post(payload):
        captured["prompt"] = payload["messages"][0]["content"][1]["text"]
        return {"choices": [{"message": {"content": captured["reply"]}}]}

    orig = agent_loop._image_data_url
    agent_loop._image_data_url = lambda p: "data:image/jpeg;base64,AAAA"
    c._post = fake_post
    try:
        # (a) 未看见细胞结构 + 模型仍答 intracellular → 改判 free（不设"无法判断"）
        captured["reply"] = ('{"cell_wall_visible": false, "crystal_location": "intracellular",'
                             ' "arrangement": "scattered", "abundance": "few", "conf": 0.8,'
                             ' "reason": "结晶位于细胞内"}')
        res = c.scene_check(Path("x.jpg"))
        check("二值: cell_wall_visible=false + intracellular → 改判 free",
              res["crystal_location"] == "free" and res["guarded"] is True)
        check("二值: 原判留痕 crystal_location_raw",
              res["crystal_location_raw"] == "intracellular")
        check("二值: 不出现 unknown 出口",
              res["crystal_location"] in ("intracellular", "free"))
        check("二值: 改判不丢排列/丰度",
              res["arrangement"] == "scattered" and res["abundance"] == "few")
        prompt = captured["prompt"]
        check("提示词: 先问视觉前提 cell_wall_visible", "cell_wall_visible" in prompt)
        check("提示词: 位置判读明确二值、不给'无法判断'选项",
              "二值" in prompt and "不答'无法判断'" in prompt)
        check("提示词: 删除候选药材口径提示行", "候选药材形态口径提示" not in prompt)
        check("提示词: 不含候选名与候选 morph 原文",
              all(x not in prompt for x in ("马齿苋", "赤芍", "牡丹皮", "虎杖",
                                            "存在于薄壁细胞中", "排列成行")))
        # 措辞口径回归锚点（2026-09-24 A/B 选定）：只保留"删偏置 + 点名簇晶"两处改动。
        # 收紧型改法（判定规则/举证责任/排除清单/面积阈值）实测全部更差
        # （细胞内 15/18 → 6/18 → 1/18，Fisher p=0.0059/<0.0001），不得在无复测的情况下回潮。
        check("措辞: 删除单向否证暗示（看不见就答 false / 通常有薄壁细胞）",
              "看不见就答 false" not in prompt and "通常有薄壁细胞" not in prompt)
        check("措辞: 保留点名区分线索（簇晶自身不是细胞）",
              "草酸钙簇晶自身" in prompt and "多角星状或花瓣状" in prompt)
        check("措辞: 不引入收紧型约束（判定规则/举证责任/面积阈值/点名清单）",
              all(x not in prompt for x in ("判定规则", "举证责任", "必须", "cell_evidence",
                                            "占画面相当大的一块区域", "淀粉粒")))

        # (b) 看见细胞结构 + intracellular → 保留（守卫不关闭宿主语境通道）
        captured["reply"] = ('{"cell_wall_visible": true, "crystal_location": "intracellular",'
                             ' "arrangement": "scattered", "abundance": "few", "conf": 0.9,'
                             ' "reason": "细胞壁清晰"}')
        res = c.scene_check(Path("x.jpg"))
        check("二值: cell_wall_visible=true + intracellular → 保留",
              res["crystal_location"] == "intracellular" and res["guarded"] is False)

        # (c) 字段缺失（旧模型/解析异常）→ 正向主张无据 → 改判 free
        captured["reply"] = '{"crystal_location": "intracellular", "conf": 0.9, "reason": "r"}'
        res = c.scene_check(Path("x.jpg"))
        check("二值: cell_wall_visible 缺失 + intracellular → 改判 free",
              res["crystal_location"] == "free" and res["guarded"] is True)

        # (d) 模型答出二值之外的词（unknown/mixed/乱值）→ 一律按"不位于细胞内"计
        for raw in ("unknown", "mixed", "也许"):
            captured["reply"] = ('{"cell_wall_visible": true, "crystal_location": "%s",'
                                 ' "arrangement": "scattered", "abundance": "few",'
                                 ' "conf": 0.9, "reason": "r"}' % raw)
            res = c.scene_check(Path("x.jpg"))
            check(f"二值: 原答 {raw!r} → 归 free（留痕）",
                  res["crystal_location"] == "free" and res["crystal_location_raw"] == raw
                  and res["guarded"] is True)

        # (e) 模型直接答 free → 不改判
        captured["reply"] = ('{"cell_wall_visible": false, "crystal_location": "free",'
                             ' "arrangement": "scattered", "abundance": "few", "conf": 0.9,'
                             ' "reason": "r"}')
        res = c.scene_check(Path("x.jpg"))
        check("二值: 原答 free → 不改判",
              res["crystal_location"] == "free" and res["guarded"] is False)
    finally:
        agent_loop._image_data_url = orig


# ------------------------- 6) 报告口径：票数分布 + 逐视野改判留痕 + 改判的评分后果
def test_report_scene_section():
    print("[6] report.write_report 场景证据段（二值留痕 / 票数分布 / 改判后的排名）")
    import tempfile
    from report import write_report
    cards = pool_evidence(_tie_records())
    # 复刻实样本（虎杖草酸钙簇晶·送检样品A）改判后的形态：
    # 模型对三视野均答 unknown 且未见细胞结构 → scene_check 一律判 free 并留痕
    scene = sa.merge_vlm_scene(
        sa.build_scene(_tie_records()),
        [{"fov": "1", "crystal_location": "free", "crystal_location_raw": "unknown",
          "cell_wall_visible": False, "guarded": True, "arrangement": "scattered",
          "abundance": "few", "conf": 0.8, "reason": "画面中未见明显的细胞壁轮廓、细胞腔"},
         {"fov": "2", "crystal_location": "free", "crystal_location_raw": "intracellular",
          "cell_wall_visible": False, "guarded": True, "arrangement": "scattered",
          "abundance": "many", "conf": 0.8, "reason": "结晶位于薄壁细胞内"},
         {"fov": "3", "crystal_location": "free", "crystal_location_raw": "free",
          "cell_wall_visible": False, "guarded": False, "arrangement": "scattered",
          "abundance": "few", "conf": 0.8, "reason": "结晶游离散在"}])
    d = decide(KB, cards, scene=scene)
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "report.md"
        write_report(td, "二值用例", d, n_images=3, vlm_notes=["[agent] fake"], scene=scene)
        md = p.read_text(encoding="utf-8")
    check("报告: 宿主语境行给出票数分布（短标签，不重复释义）",
          "3 视野判读：不位于细胞内 3" in md and "宿主语境" in md)
    check("报告: 位置用语为二值口径『不位于细胞内（游离散在）』",
          "结晶位置 不位于细胞内（游离散在）" in md)
    check("报告: 已按有界乘数作用于置信度（free 达门限即生效）",
          "已按有界乘数作用于置信度" in md)
    check("报告: 不再出现易误读的『3 次视野判读』口径", "3 次视野判读" not in md)
    check("报告: 逐视野改判留痕分两类（矛盾 / 非二值）",
          md.count("原判与视觉前提矛盾（cell_wall_visible=false，未见细胞结构） → 按不位于细胞内计") == 1
          and md.count("原答非二值（unknown） → 按不位于细胞内计") == 1)
    check("报告: 未改判的 free 视野不写改判语",
          "视野3：位置 不位于细胞内（游离散在）（依据：结晶游离散在）" in md)
    check("报告: 逐视野保留原始依据原文",
          "结晶位于薄壁细胞内" in md)
    check("报告: 局限声明写入二值口径与举证要求",
          "位于细胞内" in md and "cell_wall_visible=true" in md
          and "画面中未见细胞结构即判不位于细胞内" in md)
    # 评分计算过程节（2026-09-28 新增）：逐项得分 + 置信度推导 + top5 汇总
    check("报告: 评分计算过程节存在", "评分计算过程" in md and "S(h) 逐项展开" in md)
    check("报告: 逐判据行含公式每项（w/q/morph/m/rel/得分）",
          "判据得分 w·q·morph·m·rel" in md and "进置信度" in md and "宿主乘数" in md)
    check("报告: 置信度推导（分子/分母/封顶/乘数）", "封顶前置信度" in md and "宿主语境乘数" in md)
    check("报告: 候选汇总表（含独立证据类数）", "| 候选 | S(h)" in md and "独立证据类数" in md)
    # 改判的评分后果：free 对 morph 含"细胞内含"的候选取相悖 ×0.60
    g1 = next(r for r in d["ranking"] if r["id"] == "G1")
    hu = next(r for r in d["ranking"] if r["id"] == "虎杖")
    check("评分后果: 赤芍/牡丹皮（morph 含薄壁细胞）被 ×0.60 压低至 0.36",
          abs(g1["confidence"] - 0.36) < 1e-6
          and g1["checks"][0]["scene_factor"] == SCENE_CONTRA_FACTOR)
    check("评分后果: 虎杖（morph 无细胞表述）乘数 1.0、conf 不受影响",
          hu["checks"][0]["scene_factor"] == 1.0 and abs(hu["confidence"] - 0.6) < 1e-6)
    check("评分后果: 假 top1（赤芍/牡丹皮）已让出首位",
          d["ranking"][0]["id"] != "G1")


# --------------- 7) 乘数施加位置：封顶不吸收惩罚（2026-09-23 修正的回归锚点）
def test_scene_factor_after_cap():
    print("[7] 宿主语境乘数在置信度封顶之后施加")
    cards = pool_evidence(_tie_records())
    sc_free = {"vlm": {"crystal_location": "free", "crystal_location_conf": 0.9}}
    sc_intra = {"vlm": {"crystal_location": "intracellular", "crystal_location_conf": 0.9}}
    d0 = decide(KB, cards)
    d_free = decide(KB, cards, scene=sc_free)
    d_intra = decide(KB, cards, scene=sc_intra)
    pick = lambda d, i: next(r for r in d["ranking"] if r["id"] == i)

    # 马齿苋簇晶 quant 7~108μm（span 101 ≥ WIDE_SPAN_THRESHOLD）→ 恒被 DOM cap 压到 0.60；
    # 修正前惩罚并入贡献分，conf = min(0.6, 0.60) = 0.60，与无 scene 完全同值（惩罚消失）
    mz0, mzf = pick(d0, "马齿苋"), pick(d_free, "马齿苋")
    check("封顶候选: 无 scene 时 conf=DOM_MAX_FACTOR 0.60", abs(mz0["confidence"] - 0.6) < 1e-6)
    check("封顶候选: free 相悖乘数在封顶后生效（0.60→0.36）",
          abs(mzf["confidence"] - 0.36) < 1e-6)
    check("封顶候选: 候选级净效力 scene_factor_eff=0.6",
          mzf.get("scene_factor_eff") == SCENE_CONTRA_FACTOR)
    check("封顶候选: 贡献分不含乘数（score 与无 scene 一致）",
          abs(mzf["score"] - mz0["score"]) < 1e-6)
    check("封顶候选: 逐判据 scene_factor 仍逐条记录",
          mzf["checks"][0]["scene_factor"] == SCENE_CONTRA_FACTOR)

    # 相符乘数同样不再被封顶吃掉（马齿苋同为封顶候选）
    mzi = pick(d_intra, "马齿苋")
    check("封顶候选: intracellular 相符乘数在封顶后生效（0.60→0.66）",
          abs(mzi["confidence"] - 0.66) < 1e-6 and mzi.get("scene_factor_eff") == SCENE_MATCH_FACTOR)
    # 虎杖 morph「草酸钙簇晶极多，较大」无细胞内含/散在表述 → 乘数恒 1.0（不受任何影响）
    hz0, hzi = pick(d0, "虎杖"), pick(d_intra, "虎杖")
    check("无相关 morph 表述者: scene_factor_eff 与逐判据乘数均 1.0",
          hz0.get("scene_factor_eff") == 1.0 and hzi.get("scene_factor_eff") == 1.0
          and hzi["checks"][0]["scene_factor"] == 1.0
          and abs(hzi["confidence"] - 0.6) < 1e-6)

    # 本样本裁决方向：free 后虎杖独占首位，两个 morph 含"薄壁细胞"的候选取 0.36
    check("排名: free 后虎杖独占首位",
          d_free["ranking"][0]["id"] == "虎杖" and abs(d_free["ranking"][0]["confidence"] - 0.6) < 1e-6)
    check("排名: 赤芍/牡丹皮与马齿苋同被压到 0.36",
          abs(pick(d_free, "G1")["confidence"] - 0.36) < 1e-6
          and abs(mzf["confidence"] - 0.36) < 1e-6)
    check("排名: 铁律不变（verdict/证据类数不受乘数影响）",
          d_free["verdict"] == d0["verdict"] == "insufficient_evidence"
          and d_free["support_feature_count"] == d0["support_feature_count"] == 1)
    check("排名: 无 scene 锚点不变（仍在 0.60 并列）",
          all(abs(pick(d0, i)["confidence"] - 0.6) < 1e-6 for i in ("虎杖", "马齿苋", "G1")))


# ------------------------- 8) 评分计算过程：逐项得分可复算（2026-09-28 新增）
def test_score_breakdown():
    print("[8] 评分计算过程（S(h) 逐项展开 + 置信度复算 + 报告/verdict 落盘）")
    import tempfile
    from report import write_report
    # 大黄式双特征（簇晶 + 淀粉粒）：required 有量化判据 → 分母非 0
    recs = [rec("1", 1, value=55.0, feature="cluster_crystal"),
            rec("1", 2, value=15.0, feature="starch_grain")]
    d = decide(KB, pool_evidence(recs))
    top = next(r for r in d["ranking"] if r.get("conf_denom"))
    # 逐判据：公式各项齐备，且 w·q·morph·m·rel 复算 == 记录的判据得分
    for c in top["checks"]:
        check(f"breakdown: {c['feature_id']} 含公式各项字段",
              all(k in c for k in ("weight", "q_state", "m_abundance", "counts_toward_conf")))
        if c["score_contribution"] > 0:
            prod = (c["weight"] * c["q_state"] * c["morph_plausibility"]
                    * c["m_abundance"] * c["reliability"])
            check(f"breakdown: {c['feature_id']} w·q·morph·m·rel = 判据得分",
                  abs(prod - c["score_contribution"]) < 1e-3)
        else:
            check(f"breakdown: {c['feature_id']} 无贡献项得分为 0（q=0 或 OUT_OF_RANGE）",
                  c["score_contribution"] == 0.0)
    # 候选级：分子/分母 → 封顶前置信度；无 scene 时最终值不超过它（封顶只会压低）
    num, den, raw = top["conf_required_num"], top["conf_denom"], top["conf_raw"]
    check("breakdown: 分母 = Σ required w（>0）", den > 0)
    check("breakdown: 分子/分母 = 封顶前置信度", abs(num / den - raw) < 1e-3)
    check("breakdown: 无 scene 时最终 conf ≤ 封顶前 conf（封顶/乘数只压不抬）",
          top["confidence"] <= raw + 1e-9)
    check("breakdown: cap 标志为布尔（供报告与审计读）",
          all(isinstance(r.get("cap_dom_wide"), bool)
              and isinstance(r.get("cap_non_official"), bool) for r in d["ranking"]))
    # 报告与 verdict.json 均落盘该节
    with tempfile.TemporaryDirectory() as td:
        import report as rp
        write_report(td, "评分工", d, n_images=2)
        rp.write_verdict(Path(td), "评分工", d)
        md = (Path(td) / "report.md").read_text(encoding="utf-8")
        vj = json.loads((Path(td) / "verdict.json").read_text(encoding="utf-8"))
    check("报告: 输出评分计算过程节（含候选标注）",
          "评分计算过程" in md
          and ("结论候选" in md or "排名第 1 的候选" in md))
    check("verdict.json: score_breakdown 落盘（含公式逐项）",
          vj.get("score_breakdown") and vj["score_breakdown"]["terms"]
          and vj["score_breakdown"]["candidate"]["conf_denom"] > 0)


def main():
    test_build_scene()
    test_scene_fit_and_decide()
    test_merge_vlm_scene()
    test_augment_scene_trigger()
    test_scene_prompt_guard()
    test_report_scene_section()
    test_scene_factor_after_cap()
    test_score_breakdown()
    print(f"\n== 场景证据单测全部通过：{PASS} 项断言 ==")


if __name__ == "__main__":
    main()
