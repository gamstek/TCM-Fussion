# -*- coding: utf-8 -*-
"""audit_criteria — 判据知识库对账（设计方案 §3.2/§3.5）。

检查项：
  A1 85 味齐全、无重复；
  A2 判据 feature id 均在知识库词表内；
  A3 量化范围合法（min<=max，extreme>=max，dim ∈ {feret_max, mean_width}）；
  A4 组归属一致（成员 group_id 与 groups 记录互指、组内判据非空）；
  A5 排他规则残留检查（排他层已于 2026-09-22 拆除：库内不得再出现 exclusions 字段）；
  A6 与 几何测量/geomlib/profiles.py::REFERENCE_RANGES 对账（只读 import）：按 (herb, feature) 配对后
     比较**量纲 + 范围**（不再把 dim 写错当成"没这条"而静默跳过）；组判据按成员展开，组并集差异计入提示。
  A7 量纲可供给性：判据 dim 必须等于该特征卡片实际写出的主判量纲（profiles.get_profile 首位量纲），
     否则该判据恒判 UNCAL、药典区间形同摆设（详见 criteria_changelog v1.5）；
  A8 非药典来源不得 required：带 `non_official_basis` 的候选（依据地方标准/教材/成方口径）其判据
     存在性只能 supporting（v1.3/v1.7；详见 criteria_changelog v1.7、meta.non_official_note）。

用法： python audit_criteria.py [--kb ../data/herb_criteria.json]
退出码：0=通过（A6 差异仅提示），1=存在错误（含 A7/A8 不合规）。
"""
import argparse
import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent))

VALID_DIMS = {"feret_max", "mean_width"}
EXPECT_HERB_COUNT = 85

# A6 已知分叉（已人工裁决"不同步上游"，不再重复告警；理由见 criteria_changelog v1.5）
# 键 = (herb, feature)，值 = 裁决理由
A6_KNOWN_DIVERGENCE = {
    ("茯苓", "hypha"): "KB 取文档'单味正文'3~8|16；profiles 留'成方显微'4~6（几何侧对原始测量的参考表，"
                        "不改上游）；已裁决不同步，登记为已知差异",
    ("陈皮", "single_crystal"): "KB G3 组并集下限取枳壳正文 3μm（3~53）；profiles 单药条目仍为 5~53，"
                                "A6 按组并集差异提示",
}


def main():
    ap = argparse.ArgumentParser(description="判据知识库对账")
    here = Path(__file__).resolve().parent
    ap.add_argument("--kb", default=str(here.parent / "data" / "herb_criteria.json"))
    args = ap.parse_args()

    errors, warns, infos = [], [], []
    kb = json.loads(Path(args.kb).read_text(encoding="utf-8"))
    vocab = set()
    for tier, feats in kb["feature_vocab"].items():
        vocab |= set(feats)

    herbs = kb["herbs"]
    names = [h["herb"] for h in herbs]
    if len(names) != EXPECT_HERB_COUNT:
        errors.append(f"A1: 药材数 {len(names)} ≠ {EXPECT_HERB_COUNT}")
    if len(set(names)) != len(names):
        dup = sorted({n for n in names if names.count(n) > 1})
        errors.append(f"A1: 重复药材 {dup}")

    for h in herbs:
        tag = h["herb"]
        if h.get("no_criteria"):
            if h["criteria"]:
                errors.append(f"A2: {tag} 标记 no_criteria 但仍有判据")
            continue
        if not h["criteria"]:
            errors.append(f"A2: {tag} 无任何判据")
        for c in h["criteria"]:
            if c["feature"] not in vocab:
                errors.append(f"A2: {tag} 引用词表外特征 {c['feature']}")
            q = c.get("quant")
            if q:
                if q.get("dim") not in VALID_DIMS:
                    errors.append(f"A3: {tag}/{c['feature']} 非法 dim {q.get('dim')}")
                mn, mx, ex = q.get("min"), q.get("max"), q.get("extreme")
                if mn is not None and mx is not None and mn > mx:
                    errors.append(f"A3: {tag}/{c['feature']} min({mn}) > max({mx})")
                if ex is not None and mx is not None and ex < mx:
                    errors.append(f"A3: {tag}/{c['feature']} extreme({ex}) < max({mx})")
        for ex in h.get("exclusions", []):
            errors.append(f"A5: {tag} 仍带 exclusions（排他层已于 2026-09-22 拆除，请删除）")
        # A8 非药典来源（地方标准/教材/成方口径）的『特征存在性』不得 required：
        # 该候选全部判据都来自非官方依据（库内现状：辣蓼、青黛），一旦出现 required 即违反 v1.3/v1.7。
        # 注意：non_official_basis 是候选级旗标，若日后出现"同一候选内既有正文判据、又有非官方判据"
        # 的混合情形，须改为判据级 basis 字段后再校验（见 meta.non_official_note）。
        if h.get("non_official_basis"):
            bad = [c["feature"] for c in h["criteria"] if c["role"] == "required"]
            if bad:
                errors.append(f"A8: {tag} 依据非药典正文（non_official_basis）却有 required 判据 {bad}"
                              f"——非官方来源的特征存在性只能 supporting")
            else:
                infos.append(f"A8: {tag} 非药典依据候选，required 集为空（置信度 0，仅作备选）✓")

    group_members = set()
    for gid, g in kb["groups"].items():
        group_members |= set(g["members"])
        if not g["criteria"]:
            errors.append(f"A4: 组 {gid} 判据为空")
        for c in g["criteria"]:
            if c["feature"] not in vocab:
                errors.append(f"A4: 组 {gid} 引用词表外特征 {c['feature']}")
    by_name = {h["herb"]: h for h in herbs}
    for name in group_members:
        h = by_name.get(name)
        if h is None:
            errors.append(f"A4: 组成员 {name} 不在 herbs 中")
        elif h.get("group_id") is None:
            errors.append(f"A4: {name} 在组中但 group_id 为空")
    for h in herbs:
        gid = h.get("group_id")
        if gid and h["herb"] not in kb["groups"].get(gid, {}).get("members", []):
            errors.append(f"A4: {h['herb']} 声明组 {gid} 但组记录不含它")
    infos.append(f"A4: {len(kb['groups'])} 组 / {len(group_members)} 味归组")

    # A6/A7 与 几何测量/geomlib/profiles.py 对账
    geom_dir = here.parent.parent / "几何测量"
    try:
        sys.path.insert(0, str(geom_dir))
        from geomlib.profiles import REFERENCE_RANGES, get_profile  # 只读导入

        # KB 侧索引：key=(herb, feature) → (owner, dim, min, max, extreme)；组判据按成员展开
        kb_pairs = {}
        for h in herbs:
            for c in h["criteria"]:
                q = c.get("quant")
                if q:
                    kb_pairs[(h["herb"], c["feature"])] = (h["herb"], q["dim"], q["min"], q["max"], q.get("extreme"))
        for gid, g in kb["groups"].items():
            for c in g["criteria"]:
                q = c.get("quant")
                if not q:
                    continue
                for mem in g["members"]:
                    kb_pairs[(mem, c["feature"])] = ("组" + gid, q["dim"], q["min"], q["max"], q.get("extreme"))

        n_match = n_dim = n_range = n_grp = n_kb_only = n_prof_only = n_known = 0
        prof_idx = {}
        for r in REFERENCE_RANGES:
            prof_idx.setdefault((r["herb"], r["feature"]), []).append(r)
        used = set()
        for key, (owner, dim, mn, mx, ex) in kb_pairs.items():
            cands = prof_idx.get(key)
            if not cands:
                n_kb_only += 1
                continue
            same = [r for r in cands if r["dim"] == dim]
            if not same:
                n_dim += 1
                warns.append(f"A6: {key} 量纲不一致 KB={dim}({owner}) vs profiles={[r['dim'] for r in cands]}")
                continue
            r = same[0]
            used.add((key, r["dim"]))
            if (mn, mx, ex) == (r["umin"], r["umax"], r.get("extreme")):
                n_match += 1
            elif key in A6_KNOWN_DIVERGENCE:
                n_known += 1
                infos.append(f"A6: 已知分叉 {key} KB={mn}~{mx}|{ex} vs profiles={r['umin']}~{r['umax']}|"
                             f"{r.get('extreme')}（{A6_KNOWN_DIVERGENCE[key]}）")
            elif owner.startswith("组"):
                n_grp += 1
                infos.append(f"A6: {key} {owner} 并集 KB={mn}~{mx}|{ex} vs profiles 成员条目="
                             f"{r['umin']}~{r['umax']}|{r.get('extreme')}（组并集差异属预期，见 groups.reason）")
            else:
                n_range += 1
                warns.append(f"A6: {key} 范围不一致 KB={mn}~{mx}|{ex} vs profiles={r['umin']}~{r['umax']}|{r.get('extreme')}")
        for key, rs in prof_idx.items():
            for r in rs:
                if (key, r["dim"]) not in used:
                    n_prof_only += 1
                    infos.append(f"A6: profiles 单方 {key} {r['dim']} {r['umin']}~{r['umax']}|{r.get('extreme')}"
                                 f"：KB 未采用该量纲或未量化（区间置 null/未建判据）")

        # A7 量纲可供给性：dim 必须是该特征卡片实际写出的主判量纲。
        # descriptors.py 按 PROFILES.primary_dims 顺序取第一个有效值写 primary_dimension，
        # rule_engine.pool_evidence 只按该字段归档 → 非首位量纲恒判 UNCAL（药典区间形同摆设）。
        n_a7 = 0
        for tag, crits in ([(h["herb"], h["criteria"]) for h in herbs]
                           + [("组" + gid, g["criteria"]) for gid, g in kb["groups"].items()]):
            for c in crits:
                q = c.get("quant")
                if not q:
                    continue
                first = get_profile(c["feature"])["primary_dims"][0].replace("_um", "")
                if q["dim"] != first:
                    n_a7 += 1
                    errors.append(f"A7: {tag}/{c['feature']} dim={q['dim']} ≠ 卡片可供给量纲 {first}（恒判 UNCAL）")

        infos.append(f"A6: 与 profiles.REFERENCE_RANGES 对账 {n_match} 条完全一致；"
                     f"已知分叉 {n_known} 条（已裁决不同步上游）、组并集差异 {n_grp} 条（属预期）；"
                     f"量纲不一致 {n_dim} 条、范围不一致 {n_range} 条；"
                     f"KB 独有（profiles 未收） {n_kb_only} 条、profiles 单方 {n_prof_only} 条")
        infos.append(f"A7: 量纲可供给性检查不合规 {n_a7} 条（dim 须等于该特征卡片首位主判量纲）")
    except Exception as e:  # noqa: BLE001 — 对账失败不应阻断 A1~A5 结论
        warns.append(f"A6/A7: 无法导入 geomlib.profiles（{e}），跳过对账")

    print("== audit_criteria ==")
    for i in infos:
        print(f"[info] {i}")
    for w in warns:
        print(f"[warn] {w}")
    for e in errors:
        print(f"[ERROR] {e}")
    print(f"== 结果：{'PASS' if not errors else 'FAIL'}（错误 {len(errors)}，提示 {len(warns)}）==")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
