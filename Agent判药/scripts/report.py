# -*- coding: utf-8 -*-
"""report — 判药结果落盘：verdict.json（机器可读）+ report.md（人读报告）。

报告骨架（设计方案 §6.2）：
  1 结论卡  2 判据核验表  2·b 评分计算过程（S(h) 逐项展开）  2·c 场景证据与测量分布
  3 证据图链（3·b 源视野图）  4 局限声明  5 Agent 补充  6 影子仲裁
"""
import json
import shutil
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

from rule_engine import DOM_MAX_FACTOR, NONOFF_MAX_CONF   # 封顶取值单一来源（评分计算过程展示）

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

VERDICT_ZH = {
    "confirmed": "鉴别成立",
    "probable": "倾向成立（需复核）",
    "group_level": "鉴别至组（组内显微不可分）",
    "insufficient_evidence": "证据不足，转人工",
}
STATE_ZH = {
    "IN_RANGE": "✓ 范围相符", "EXTREME": "△ 落入'个别可达'区间", "NO_REF": "✓ 检出（无量化判据）",
    "UNCAL": "△ 检出但未标定", "OUT_OF_RANGE": "✗ 超出范围", "NOT_DETECTED": "— 未检出",
}


def write_verdict(out_dir, sample_id, decision, extra=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    doc = {
        "schema": "agent-judgement/v1",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "sample_id": sample_id,
        "verdict": decision["verdict"],
        "verdict_zh": VERDICT_ZH.get(decision["verdict"], decision["verdict"]),
        "conclusion": decision.get("conclusion"),
        "alternatives": decision.get("alternatives", []),
        "margin": decision.get("margin"),
        "support_feature_count": decision.get("support_feature_count"),  # D8 证据数量门槛输入
        "ranking_top5": [
            {"id": r["id"], "display": r["display"], "confidence": r["confidence"]}
            for r in decision.get("ranking", [])[:5]
        ],
        "criteria_check": _full_checks(decision),
        "score_breakdown": _score_breakdown(decision),   # 评分计算过程（逐项，2026-09-28 新增）
        "notes": decision.get("notes", []),
        "needs_review": decision.get("needs_review", False),
        "agent": (extra or {}).get("agent", {"mode": "rule_only", "vlm_calls": 0, "degraded": False}),
        "evidence_files": (extra or {}).get("evidence_files", []),
        "source_images": (extra or {}).get("source_images", []),
        # 场景证据（三层模型第 2/3 层）：仅备选排序/复核参考，不改裁决语义
        "scene_attributes": (extra or {}).get("scene_attributes", None),
        "shadow_bounded_adjudicator": (extra or {}).get("shadow", None),  # 影子仲裁（方案D，不改 verdict）
    }
    path = out_dir / "verdict.json"
    path.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _full_checks(decision):
    """找到 top1（结论候选）的完整核验明细（无结论候选时返回空）。"""
    if not decision.get("conclusion"):
        return []
    top_id = decision["conclusion"]["id"]
    for r in decision.get("ranking", []):
        if r["id"] == top_id:
            return r["checks"]
    return []


def _score_breakdown(decision):
    """top1 的评分计算过程（机器可读镜像，供审计复算）。

    top1 = 结论候选；无结论时取排名第 1 的候选。逐判据的公式各项取 checks 的同名字段，
    候选级取 conf_required_num / conf_denom / conf_raw / cap_* / scene_factor_eff。
    """
    ranking = decision.get("ranking") or []
    if not ranking:
        return None
    top_id = (decision.get("conclusion") or {}).get("id")
    top = next((r for r in ranking if r["id"] == top_id), None) if top_id else ranking[0]
    if top is None:
        return None
    term_keys = ("feature_id", "role", "tier", "weight", "state", "q_state", "morph_plausibility",
                 "abundance", "m_abundance", "reliability", "score_contribution", "penalty",
                 "counts_toward_conf", "scene_factor")
    cand_keys = ("id", "display", "score", "sup_net", "conf_required_num", "conf_denom",
                 "conf_raw", "cap_dom_wide", "cap_non_official", "scene_factor_eff",
                 "confidence", "support_count")
    return {
        "formula": "S(h)=Σ w(tier)·q(state)·morph·m(abundance)·rel − 罚项；"
                   "C(h)=min(1, max(0,S_required)/Σ_required w) × s(scene)，封顶先于乘数",
        "note": "supporting 净贡献不进置信度（只作用于证据广度与并列排序）；"
                "required 未检出也占分母，另有缺失罚",
        "top1_id": top["id"],
        "candidate": {k: top.get(k) for k in cand_keys},
        "terms": [{k: c.get(k) for k in term_keys} for c in top.get("checks", [])],
        "penalties": {"absent_required": top.get("absent_required") or [],
                      "wrong_present": top.get("wrong_present") or []},
        "ranking_top5": [{k: r.get(k) for k in cand_keys} for r in ranking[:5]],
    }


def fmt_stats(stats):
    if not stats:
        return "—"
    return (f"median {stats['median']:.1f}μm（IQR {stats['p25']:.1f}~{stats['p75']:.1f}，"
            f"n={stats['n_measured']}" + (f"，另 {stats['n_uncal']} 实例未标定" if stats['n_uncal'] else "") + "）")


def _copy_evidence(evidence_files, ev_dir):
    """把叠加图复制进报告目录 `证据图/`（报告自包含），返回 {文件名: 相对路径}。

    叠加图由几何 measure.py 产出，契约文件名 `<stem>__inst<id>_测量.jpg`。
    """
    m = {}
    if not evidence_files:
        return m
    ev_dir.mkdir(parents=True, exist_ok=True)
    for p in sorted({str(x) for x in evidence_files}):
        p = Path(p)
        if not p.exists():
            continue
        dst = ev_dir / p.name
        try:
            shutil.copyfile(p, dst)        # 覆盖旧副本：重跑必须刷新证据图
        except OSError:
            continue
        m[p.name] = f"证据图/{p.name}"
    return m


def _row_imgs(ev_map, refs):
    """按判据核验表的实例引用（`img1#1` 形式）解析该行证据图：markdown 图片，横排并列。"""
    out = []
    for r in refs or []:
        if "#" not in r:
            continue
        stem, iid = r.split("#", 1)
        iid = (iid.split()[0] if iid else None)
        if not iid:
            continue
        fn = f"{stem}__inst{iid}_测量.jpg"
        if fn in ev_map:
            out.append(f"![{fn}]({ev_map[fn]})")
    return " ".join(out)


ARR_ZH = {"rows": "排列成行", "scattered": "散乱分布", "insufficient": "实例数不足，未判读"}
LOC_ZH = {"intracellular": "位于细胞内", "free": "不位于细胞内（游离散在）",
          "mixed": "细胞内外均有", "unknown": "无法判断"}
# 票数分布用短标签：括号内的释义在同一行重复出现会成为噪声
LOC_ZH_VOTE = {"intracellular": "位于细胞内", "free": "不位于细胞内",
               "mixed": "细胞内外均有", "unknown": "无法判断"}


def _num(v):
    """报告数值格式：None → '—'；整数值去小数；其余保留有效位（不引入虚假精度）。"""
    if v is None:
        return "—"
    if isinstance(v, float) and v == int(v):
        return f"{int(v)}"
    return f"{v:.4f}".rstrip("0").rstrip(".") if isinstance(v, float) else str(v)


def _append_score_breakdown(lines, decision, feature_zh):
    """§2·b 评分计算过程：逐判据展开公式每一项，并复算置信度（每个数字可人工复算）。

    数据全部取自 rule_engine.evaluate_candidate：逐判据的 w / q_state / morph_plausibility /
    m_abundance / reliability / score_contribution / penalty / counts_toward_conf / scene_factor，
    以及候选级的 conf_required_num（分子）/ conf_denom（分母）/ conf_raw / cap_* / scene_factor_eff。
    """
    ranking = decision.get("ranking") or []
    if not ranking:
        return
    lines += ["", "## 二·b、评分计算过程（S(h) 逐项展开）", ""]
    lines.append("- 公式（设计方案 §5.2）：`S(h) = Σ w(级别)·q(核验)·morph(形态)·m(丰度)·rel(质量) − 罚项`；"
                 "`C(h) = min(1, max(0, S_required) / Σ_required w)`，最后乘宿主语境乘数 `s(scene)`。")
    lines.append("- 取值口径：`w` = 通道权重（P0 1.0 / P1 0.5 / C 0.3）；`q` = 量化核验五态得分"
                 "（IN_RANGE 1.0 / EXTREME 0.7 / NO_REF 0.6 / UNCAL 0.5 / OUT_OF_RANGE 0.0，未检出 0）；"
                 "`morph` = 形态签名相符度；`m` = 判据声明丰度的乘数（未声明 1.0）；`rel` = 检出质量"
                 "（conf_factor × quality_factor）。supporting 判据净贡献**不进置信度**（只用于证据广度与"
                 "并列排序）；OUT_OF_RANGE 的判据贡献记 0 并按 `0.5·w·rel` 罚分；"
                 "封顶与场景乘数在置信度算出**之后**依次施加。")
    top = ranking[0]
    kind_txt = "结论候选" if decision.get("conclusion") else "排名第 1 的候选"
    lines += ["", f"逐项得分（{kind_txt}：{top['display']}）：", ""]
    lines.append("| 判据特征 | 角色 | w（级别） | q（核验） | morph（形态） | m（丰度） | rel（质量） "
                 "| 判据得分 w·q·morph·m·rel | 进置信度 | 宿主乘数 | 罚项 |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for c in top.get("checks", []):
        fid = c["feature_id"]
        zh = (feature_zh or {}).get(fid) or ""
        name = f"{fid} / {zh}" if zh else fid
        morph = "—" if c.get("morph_plausibility") is None else _num(c["morph_plausibility"])
        scene_f = "—" if c.get("scene_factor") is None else _num(c["scene_factor"])
        lines.append(
            f"| {name} | {c['role']} | {_num(c.get('weight'))}（{c['tier']}） "
            f"| {_num(c.get('q_state'))}（{STATE_ZH.get(c['state'], c['state'])}） "
            f"| {morph} | {_num(c.get('m_abundance'))}（{c.get('abundance') or '未声明'}） "
            f"| {_num(c.get('reliability'))} | **{_num(c.get('score_contribution'))}** "
            f"| {'✓ 分子/分母' if c.get('counts_toward_conf') else '✗ 仅排序'} "
            f"| {scene_f} | {_num(c.get('penalty'))} |")

    if top.get("absent_required") or top.get("wrong_present"):
        lines += ["", f"罚项（{top['display']}）："]
        for a in top.get("absent_required") or []:
            lines.append(f"- required 缺失罚 −{_num(a['penalty'])}：{a['feature_id']}（{a['note']}）")
        for w_ in top.get("wrong_present") or []:
            lines.append(f"- 错误特征出现罚 −{_num(w_['penalty'])}：{w_['feature_id']}"
                         f"（{w_['note']}，{w_['n_instances']} 实例）")

    lines += ["", f"置信度推导（{top['display']}）："]
    lines.append(f"- 判据贡献合计 S(h) = **{_num(top.get('score'))}**（已含罚项与 supporting 判据贡献）")
    lines.append(f"- 剔除 supporting 净贡献 {_num(top.get('sup_net'))} → "
                 f"分子 S_required = {_num(top.get('conf_required_num'))}"
                 "（supporting 的贡献与罚项都不进置信度，只作证据广度与并列排序；分子为负时按 0 计）")
    lines.append(f"- 分母 Σ_required w = {_num(top.get('conf_denom'))}"
                 "（required 未检出也占分母，另有缺失罚压低比值）")
    lines.append(f"- 封顶前置信度 = 分子 / 分母 = **{_num(top.get('conf_raw'))}**")
    caps = []
    if top.get("cap_dom_wide"):
        caps.append(f"宽区间判据主导，封顶 ≤ {DOM_MAX_FACTOR:g}")
    if top.get("cap_non_official"):
        caps.append(f"判据依据非药典正文，封顶 ≤ {NONOFF_MAX_CONF:g}")
    lines.append(f"- 封顶：{'；'.join(caps) if caps else '未触发'}")
    lines.append(f"- 宿主语境乘数 s(scene) = {_num(top.get('scene_factor_eff'))}"
                 f" → **最终置信度 = {_num(top.get('confidence'))}**")
    if not decision.get("conclusion"):
        lines.append("- 注：本样本无结论候选（裁决带/D8 证据数量门槛未达标），以上为排名第 1 的候选；"
                     "门槛判定见「一、结论」的 notes。")

    lines += ["", "| 候选 | S(h) | 分子 | 分母 | 封顶前 conf | 封顶 | s(scene) | 置信度 | 独立证据类数 |"]
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for r in ranking[:5]:
        caps_r = []
        if r.get("cap_dom_wide"):
            caps_r.append("宽区间主导")
        if r.get("cap_non_official"):
            caps_r.append("非药典依据")
        lines.append(f"| {r['display']} | {_num(r.get('score'))} | {_num(r.get('conf_required_num'))} "
                     f"| {_num(r.get('conf_denom'))} | {_num(r.get('conf_raw'))} "
                     f"| {'、'.join(caps_r) or '—'} | {_num(r.get('scene_factor_eff'))} "
                     f"| **{_num(r.get('confidence'))}** | {r.get('support_count', 0)} |")


def igm_review_refs(records):
    """挑出 IGM mask 状态非 ok 的实例（flags 带 igm_review）→ [(stem, instance_id)]。

    flags 兼容两种载体：JSONL 数组 / CSV 分号连接串（见 measure.py 的 CSV 契约）。
    """
    out = []
    for r in records or []:
        fl = r.get("flags") or []
        if isinstance(fl, str):
            fl = [x for x in fl.split(";") if x]
        if "igm_review" in fl:
            out.append((str(r.get("image_stem", "?")), str(r.get("instance_id", "?"))))
    return out


def write_report(out_dir, sample_id, decision, n_images=0, vlm_notes=None,
                 evidence_files=None, source_images=None, feature_zh=None, shadow=None,
                 scene=None, review_refs=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    v = decision["verdict"]
    lines = [f"# 显微鉴别报告 · {sample_id}", ""]
    lines.append(f"> 生成时间：{datetime.now().isoformat(timespec='seconds')}｜"
                 f"视野图 {n_images} 张｜推理模式：{('rule_only' if not vlm_notes else 'rule+agent')}")

    # 1 结论卡
    lines += ["", "## 一、结论", ""]
    if decision.get("conclusion"):
        c = decision["conclusion"]
        lines.append(f"- **{VERDICT_ZH.get(v, v)}：{c['display']}**（置信度 {c['confidence']:.2f}）")
        if decision.get("alternatives"):
            alts = "；".join(f"{a['display']} {a['confidence']:.2f}" for a in decision["alternatives"])
            lines.append(f"- 备选：{alts}")
    else:
        lines.append(f"- **{VERDICT_ZH.get(v, v)}**")
        top = decision.get("ranking_top5") or decision.get("ranking") or []
        if top:
            lines.append("- 候选排名（供人工复核参考）：" +
                         "；".join(f"{r['display']} {r['confidence']:.2f}" for r in top[:5]))
    for n in decision.get("notes", []):
        lines.append(f"- ⚠ {n}")
    if decision.get("needs_review"):
        lines.append("- ⚠ 本样本标记人工复核（needs_review）")

    # 2 判据核验表（证据图内嵌于表格列，原图尺寸）
    lines += ["", "## 二、判据核验（对照《中国药典》判据四元组·量化+形态）", ""]
    checks = _full_checks(decision)
    ev_dir = out_dir / "证据图"
    ev_map = _copy_evidence(evidence_files, ev_dir)
    if checks:
        lines.append("| 判据特征 | 级别 | 核验 | 实测 vs 药典 | 形态 | 证据图 |")
        lines.append("|---|---|---|---|---|---|")
        for c in checks:
            qtxt = fmt_stats(c.get("stats"))
            if c["quant_note"]:
                qtxt += "｜" + c["quant_note"]
            morph = f"签名相符度 {c['morph_plausibility']}" if c.get("morph_plausibility") is not None else "—"
            imgs = _row_imgs(ev_map, c.get("instance_refs")) or "—"
            fid = c["feature_id"]
            zh = (feature_zh or {}).get(fid) or ""
            name = f"{fid} / {zh}" if zh else fid
            lines.append(f"| {name}（{c['role']}） | {c['tier']} | {STATE_ZH.get(c['state'], c['state'])} "
                         f"| {qtxt} | {morph} | {imgs} |")
    else:
        lines.append("（无结论候选，无核验表）")

    # 2·b 评分计算过程（S(h) 逐项展开，2026-09-28 新增）：让每个数字都能被人工复算
    _append_score_breakdown(lines, decision, feature_zh)

    # 2·c 场景证据与测量分布（三层模型第 2/3 层）：复核参考；宿主语境以有界乘数参与评分
    if scene and (not decision.get("conclusion") or decision.get("needs_review")):
        lines += ["", "## 二·c、场景证据与测量分布（供人工复核参考）", ""]
        ss = scene.get("size_stats")
        if ss:
            lines.append(f"- 实测分布（feret_max）：min {ss['min']}μm / median {ss['median']}μm"
                         f" / max {ss['max']}μm，共 {scene.get('n_total', 0)} 实例。")
        pfc = scene.get("per_fov_counts") or {}
        if pfc:
            counts_desc = "、".join(f"视野{k}：{v} 个" for k, v in pfc.items())
            ab = scene.get("abundance_obs")
            src = (scene.get("sources") or {}).get("abundance", "geometry")
            src_zh = "VLM 判读" if src == "vlm" else "几何计数"
            lines.append(f"- 每视野检出数：{counts_desc}（丰度档位估计：{ab or '—'}，来源 {src_zh}，"
                         "受检测召回影响，仅供参考）。")
        arr = scene.get("arrangement") or []
        if arr:
            desc = "；".join(
                f"视野{a['fov']}(n={a['n']})：{ARR_ZH.get(a.get('verdict'), a.get('verdict'))}"
                + (f"（VLM 判读：{ARR_ZH.get(a.get('vlm_verdict'), a.get('vlm_verdict'))}）"
                   if a.get("vlm_verdict") else "")
                for a in arr)
            lines.append(f"- 排列判读：{desc}。")
        vlm_block = scene.get("vlm")
        if vlm_block:
            loc_conf = vlm_block.get("crystal_location_conf")
            conf_txt = (f"；保守置信度 {loc_conf:.2f}，取有效判读最小值"
                        if isinstance(loc_conf, (int, float)) else "")
            gate_txt = ""
            if (isinstance(loc_conf, (int, float)) and loc_conf >= 0.70
                    and vlm_block.get("crystal_location") in ("intracellular", "free")):
                gate_txt = "→ 已按有界乘数作用于置信度（乘数记录见 verdict.json 的 scene_factor_eff）"
            # 票数分布（含 unknown 票）：只写"n 次视野判读"会让人误读为 n 票一致
            votes = Counter(str(a.get("crystal_location") or "unknown")
                            for a in (vlm_block.get("per_fov") or {}).values()
                            if isinstance(a, dict))
            vote_txt = "：" + " / ".join(f"{LOC_ZH_VOTE.get(k, k)} {v}" for k, v in
                                         sorted(votes.items(), key=lambda kv: (-kv[1], kv[0])))
            lines.append(f"- VLM 宿主语境：结晶位置 "
                         f"{LOC_ZH.get(vlm_block.get('crystal_location'), vlm_block.get('crystal_location'))}"
                         f"（{vlm_block.get('n_calls', 0)} 视野判读{vote_txt}{conf_txt}）{gate_txt}。")
            # 逐视野判读原文（含视觉前提 cell_wall_visible 与降级留痕），供复核核对票数来源
            pfov = vlm_block.get("per_fov") or {}
            for fov, a in sorted(pfov.items(),
                                 key=lambda kv: ((0, int(kv[0])) if str(kv[0]).isdigit()
                                                 else (1, 0))):
                if not isinstance(a, dict):
                    continue
                loc = a.get("crystal_location")
                vis_txt = ""
                if a.get("guarded"):
                    raw = a.get("crystal_location_raw")
                    why = ("原判与视觉前提矛盾（cell_wall_visible=false，未见细胞结构）"
                           if raw == "intracellular" else f"原答非二值（{raw}）")
                    vis_txt = f"；{why} → 按不位于细胞内计"
                elif a.get("cell_wall_visible"):
                    vis_txt = "；cell_wall_visible=true（画面可见细胞结构）"
                lines.append(f"  - 视野{fov}：位置 {LOC_ZH.get(loc, loc)}{vis_txt}"
                             f"（依据：{a.get('reason') or '—'}）")
        # 测量分布 vs 各候选区间表（top-5 候选的量化判据）
        rows = []
        for r in (decision.get("ranking") or [])[:5]:
            for c in r.get("checks", []):
                qb = c.get("quant_bounds")
                st = c.get("stats")
                if not qb or not c.get("present") or not st:
                    continue
                lo, hi = qb.get("min"), qb.get("max")
                if lo is None or hi is None:
                    continue
                med = st["median"]
                rows.append((r["display"], qb.get("dim"), lo, hi, med, lo <= med <= hi,
                             min(abs(med - lo), abs(hi - med))))
        if rows:
            lines.append("")
            lines.append("| 候选 | 量纲 | 判据区间 | 实测 median | median 在区间内 | 距边缘最近 |")
            lines.append("|---|---|---|---|---|---|")
            for disp, dim, lo, hi, med, inside, dist in rows:
                lines.append(f"| {disp} | {dim} | {lo:g}~{hi:g}μm | {med:.1f}μm | "
                             f"{'✓' if inside else '✗'} | {dist:.1f}μm |")
        # 打破平分方向提示：平分候选的判据区间边缘
        ranking = decision.get("ranking") or []
        top_conf = ranking[0].get("confidence") if ranking else None
        tips, seen_disp = [], set()
        for r in ranking:
            if top_conf is None or r.get("confidence") != top_conf:
                continue
            if r["display"] in seen_disp:
                continue
            seen_disp.add(r["display"])
            for c in r.get("checks", []):
                qb = c.get("quant_bounds")
                if (qb and c.get("role") == "required"
                        and qb.get("min") is not None and qb.get("max") is not None):
                    tips.append(f"若再取视野测得 <{qb['min']:g}μm 或 >{qb['max']:g}μm 的新实例，"
                                f"可排除 {r['display']}（判据区间 {qb['min']:g}~{qb['max']:g}μm）")
        if tips:
            lines.append("")
            for t in tips:
                lines.append(f"- 打破平分方向：{t}（供人工复核参考）。")

    # 3 证据图链：叠加图已复制进 证据图/ 并内嵌于 §二 判据核验表的「证据图」列
    lines += ["", "## 三、证据图链", ""]
    if checks:
        lines.append("- 证据叠加图（轮廓+最大长径+主判尺寸）已复制至本报告目录 `证据图/`，并内嵌于上方"
                     "「判据核验表」的『证据图』列；完整清单见 `verdict.json.evidence_files`（只引用不额外复制）。")
    elif ev_map:
        lines.append("叠加图已复制至本报告目录 `证据图/`（本样本无结论候选，未落入核验表，以下全量展示）：")
        for name in sorted(ev_map):
            lines.append(f"![{name}]({ev_map[name]})")
    else:
        lines.append("无可引用实例。")

    # 3b 源视野图：复制原始显微图进报告目录（报告自包含），markdown 内嵌
    src = [p for p in (source_images or []) if Path(p).exists()]
    seen_names = set()
    if src:
        img_dir = out_dir / "源图"
        img_dir.mkdir(exist_ok=True)
        lines += ["", "## 三·b、源视野图（原始显微图）", ""]
        lines.append(f"判定依据的原始视野，共 {len(src)} 张（复制至报告目录 `源图/`，报告自包含）。")
        for p in sorted(set(src)):
            dst = img_dir / Path(p).name
            try:
                shutil.copyfile(p, dst)
            except OSError as e:
                lines.append(f"- `{Path(p).name}`（复制失败：{e}）")
                continue
            if dst.name in seen_names:
                lines.append(f"- `{dst.name}`（已存在同名字图，略）")
                continue
            seen_names.add(dst.name)
            lines.append(f"![{dst.name}](源图/{dst.name})")
    elif n_images:
        lines += ["", "## 三·b、源视野图", ""]
        lines.append("- 未提供 `--images` 或未匹配到源图，未内嵌原始视野图。")

    # 4 局限声明
    lines += ["", "## 四、局限声明", ""]
    lines.append("- 尺寸结论仅来自几何测量链（mask 真值 + 比例尺标定）；偏光/化学显色判据未参与自动裁决。")
    uncal = any(c.get("state") == "UNCAL" for c in checks) if checks else False
    if uncal:
        lines.append("- 存在检出但未标定的实例，其尺寸未进入统计（见核验表）。")
    if review_refs:
        # IGM 是**现役分割通路**，其生成是随机的；mask 可能极性反判或与框外物体粘连（见
        # igm_segment.py 的 review 判据）。这类实例在判药侧已被整条丢弃（rule_engine.pool_evidence），
        # 故此处点名是为了交代"哪些证据被排除了"，避免读者以为它们参与了上表统计。
        refs = "、".join(f"`{s}__inst{i}`" for s, i in review_refs)
        lines.append(f"- ⚠ 以下实例的 IGM mask 状态非 ok（极性/目标框先验异常），**已整条排除在证据之外**"
                     f"（未计入上表统计、未参与结论）：{refs}。"
                     "请人工复核对应 mask（`证据图/` 里可看轮廓）；若确需其参与判读，须先修正 mask。")
    lines.append("- P1/P2 特征（纤维、分泌组织、毛茸等）在检测类别扩展前依赖图像级/VLM 弱证据，结论权重相应降低。")
    lines.append("- 场景证据中，排列/丰度仅用于备选排序与复核参考；宿主语境（结晶位置二值判读："
                 "位于细胞内 / 不位于细胞内－游离散在）以有界乘数（相符 ×1.10 / 相悖 ×0.60）"
                 "在置信度封顶**之后**作用于 candidate.confidence"
                 "（逐判据乘数见 checks.scene_factor、候选级净效力见 scene_factor_eff）；"
                 "判读置信度 ≥0.70 时生效，且'位于细胞内'须以画面可见细胞结构"
                 "（cell_wall_visible=true）为证——画面中未见细胞结构即判不位于细胞内"
                 "（明场未染色细胞壁常不可见，防以此类药材学常识替代观察）；"
                 "均不计入 D8 证据类数，丰度档位受检测召回影响仅供参考。")
    lines.append("- 组级结论为药典事实（组内显微特征相同），如需细分建议结合性状/理化鉴别。")

    # 5 Agent 补充
    if vlm_notes:
        lines += ["", "## 五、Agent（VLM）补充判读", ""]
        for n in vlm_notes:
            lines.append(f"- {n}")

    # 6 影子仲裁（方案D，仅记录，不改 verdict）
    if shadow and shadow.get("applied") and shadow.get("adjudications"):
        lines += ["", "## 六、影子仲裁建议（方案D，仅参考，不改结论）", ""]
        lines.append(f"- 触发：{shadow['trigger']['verdict']}，"
                     f"top1={shadow['trigger']['top1']} vs top2={shadow['trigger']['top2']}，"
                     f"分差 {shadow['trigger']['margin']:.3f} < {shadow['trigger']['band_threshold']}")
        for ad in shadow["adjudications"]:
            status = {"ok": "有效", "error": "调用失败", "invalid_reference": "引用不一致，作废"}.get(
                ad["status"], ad["status"])
            if ad["status"] == "ok":
                lines.append(f"- 建议：{ad.get('choice_zh')}（choice={ad.get('choice')}）；理由：{ad.get('reason')}")
                if ad.get("cited"):
                    cit = "、".join(f"{c.get('candidate')}:{c.get('feature')}"
                                    for c in ad["cited"])
                    lines.append(f"- 引用判据：{cit}（已通过一致性校验）")
            else:
                lines.append(f"- 无效/降级：{status}（{ad.get('detail')}）")
        lines.append(f"- {shadow.get('disclaimer', '')}")

    path = out_dir / "report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
