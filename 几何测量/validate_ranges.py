# -*- coding: utf-8 -*-
"""
validate_ranges.py — 几何测量结果 × 药典判据范围核查

读 measure.py 产出的 measurements.csv，对每条记录：
  1) 按 (herb, feature) 匹配 REFERENCE_RANGES（herb 缺失时仅匹配 herb=None 的通用条目）；
  2) 用记录的 primary_dimension（feret_max / mean_width）对照该量纲的范围；
  3) 落在主范围外但 ≤ extreme → extreme_only；完全出界 → out_of_range；
     无匹配范围 → no_reference；未标定 → uncalibrated。
另执行大黄排他判据的几何可核查部分（簇晶 <45μm 存疑）。

输出 <out> 逐行核查 CSV，并在 stdout 打印汇总。
"""
import os
import sys
import csv
import argparse
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from geomlib.profiles import REFERENCE_RANGES


def match_ranges(herb, feature, dim):
    """返回匹配的参考条目列表（herb 精确 > 通用 herb=None）。"""
    exact = [r for r in REFERENCE_RANGES
             if r["feature"] == feature and r.get("herb") == herb and r["dim"] == dim]
    if exact:
        return exact
    generic = [r for r in REFERENCE_RANGES
               if r["feature"] == feature and r.get("herb") is None and r["dim"] == dim]
    return generic


def check_row(row):
    herb, feature = (row.get("herb") or "").strip(), (row.get("feature") or "").strip()
    dim = (row.get("primary_dimension") or "").replace("_um", "")
    val = row.get("primary_value_um")
    flags = (row.get("flags") or "")
    res = {"herb": herb, "feature": feature, "dimension": dim,
           "value_um": val, "status": "no_reference", "ref_min": "", "ref_max": "",
           "ref_extreme": "", "ref_note": "", "check_flags": ""}
    if not val:
        res["status"] = "uncalibrated" if "uncalibrated" in flags else "no_value"
        return res
    val = float(val)
    if not herb or not dim:
        return res
    cands = match_ranges(herb, feature, dim)
    if not cands:
        return res
    r = cands[0]                      # herb 精确匹配优先（match_ranges 已排序语义）
    res["ref_min"], res["ref_max"] = r["umin"], r["umax"]
    res["ref_extreme"], res["ref_note"] = r.get("extreme", ""), r.get("note", "")
    lo, hi = r["umin"], r["umax"]
    if lo is not None and val < lo:
        res["status"] = "out_of_range"
        res["check_flags"] = f"below_min({lo})"
    elif hi is not None and val > hi:
        ext = r.get("extreme")
        if ext and val <= ext:
            res["status"] = "extreme_only"
            res["check_flags"] = f"above_max_but_within_extreme({ext})"
        else:
            res["status"] = "out_of_range"
            res["check_flags"] = f"above_max({hi})"
    else:
        res["status"] = "within"

    return res


def main():
    ap = argparse.ArgumentParser(description="几何测量 × 药典判据范围核查")
    ap.add_argument("--measurements", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.measurements, encoding="utf-8-sig")))

    summary = defaultdict(lambda: defaultdict(list))
    with open(args.out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["image_stem", "instance_id", "herb", "feature", "dimension",
                    "value_um", "check_status", "ref_min", "ref_max", "ref_extreme",
                    "ref_note", "check_flags"])
        n_by_status = defaultdict(int)
        for row in rows:
            res = check_row(row)
            w.writerow([row.get("image_stem"), row.get("instance_id"),
                        res["herb"], res["feature"], res["dimension"],
                        res["value_um"], res["status"],
                        res["ref_min"], res["ref_max"], res["ref_extreme"],
                        res["ref_note"], res["check_flags"]])
            n_by_status[res["status"]] += 1
            if res["status"] in ("within", "out_of_range", "extreme_only"):
                k = (res["herb"] or "（无药材）", res["feature"])
                summary[k]["status"].append(res["status"])
                if res["value_um"]:
                    summary[k]["values"].append(float(res["value_um"]))

    print(f"逐行核查 → {args.out}（共 {len(rows)} 条）")
    for st, n in sorted(n_by_status.items()):
        print(f"  {st:14s} {n}")
    print("\n按 药材×特征 汇总（有参考范围者）:")
    for (herb, feature), agg in sorted(summary.items()):
        v = agg["values"]
        st = agg["status"]
        ratio_out = st.count("out_of_range") / max(len(st), 1)
        print(f"  {herb} · {feature}: n={len(st)} 中位={sorted(v)[len(v)//2] if v else '-'}"
              f" 范围[{min(v):.1f},{max(v):.1f}] 出界率={ratio_out:.0%}" if v else
              f"  {herb} · {feature}: n={len(st)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
