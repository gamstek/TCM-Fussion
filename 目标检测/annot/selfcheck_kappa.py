# -*- coding: utf-8 -*-
"""
selfcheck_kappa.py — 单人自检一致性（Kappa ≥0.7 才可量产，见方案 §六 验收）
单人审核的自检方案：从已审核完成的项目抽 10% 图，导入"自检(盲标)"项目重新盲标；
隔周（或至少隔一天）标注后跑 --report 对比两轮框与类别的一致性。

用法:
  python annot\selfcheck_kappa.py --source-project-id 12 --create   # 抽样建自检任务
  python annot\selfcheck_kappa.py --source-project-id 12 --selfcheck-project-id 13 --report
"""
import argparse
import json
import os
import random
import sys
from collections import defaultdict


from geom import iou_box  # noqa: E402 框 IoU 原语（原为本地第 3 份副本，2026-09-29 P4-1a 上收）
from ls_common import (DET_CONFIG, DET_RESULTS, ensure_dirs, load_config, ls_api)  # noqa: E402
from ls_tasks import fetch_tasks, rects_of  # noqa: E402 LS 任务/标注读写（P4-3 上收）

STATE_PATH = os.path.join(DET_RESULTS, "selfcheck_state.json")


def create(cfg, src_pid):
    tasks = fetch_tasks(cfg, src_pid)
    done = [t for t in tasks
            if t.get("annotations") and not t["annotations"][-1].get("was_cancelled")]
    print(f"源项目已审核完成 {len(done)}/{len(tasks)}")
    n = max(3, round(len(done) * cfg["labelstudio"].get("selfcheck_ratio", 0.1)))
    random.Random(cfg["pilot"].get("seed", 42) + 1).shuffle(done)
    sample = done[:n]

    with open(os.path.join(DET_CONFIG, "labeling_config.xml"), encoding="utf-8") as f:
        label_config = f.read()
    r = ls_api(cfg, "POST", "/api/projects", json={
        "title": cfg["labelstudio"]["selfcheck_title"],
        "label_config": label_config,
        "expert_instruction": "自检盲标：不看第一轮结果，独立重标本图。只画你有把握的框，拿不准就标[疑似待定]。",
    })
    sc_pid = r.json()["id"]

    # 复用已上传的 LS 图片 URL，直接以任务 JSON 导入（不传图、不带入第一轮结果，保证盲标）
    new_tasks = []
    for t in sample:
        new_tasks.append({
            "data": {"image": t["data"]["image"],
                     "original_filename": t["data"].get("original_filename", ""),
                     "herb": t["data"].get("herb", ""),
                     "group_key": t["data"].get("group_key", ""),
                     "source_task_id": t["id"]},
        })

    r = ls_api(cfg, "POST", f"/api/projects/{sc_pid}/import", json=new_tasks)
    n_imported = r.json().get("task_count", len(new_tasks))
    state = {"source_project": src_pid, "selfcheck_project": sc_pid,
             "pairs": [{"source_task_id": t["data"]["source_task_id"]} for t in new_tasks]}
    with open(STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    print(f"自检项目 id={sc_pid}，导入 {n_imported} 张盲标任务")
    print(f"注意: 第一轮结果保存在源项目 id={src_pid}（state: {STATE_PATH}）")


def report(cfg, src_pid, sc_pid):
    src_tasks = {t["id"]: t for t in fetch_tasks(cfg, src_pid)}
    sc_tasks = fetch_tasks(cfg, sc_pid)

    pairs = []  # (第一轮rects, 第二轮rects, filename)
    for t in sc_tasks:
        anns = [a for a in t.get("annotations", []) if not a.get("was_cancelled")]
        if not anns or not t["data"].get("source_task_id"):
            continue
        src_t = src_tasks.get(t["data"]["source_task_id"])
        if not src_t or not src_t.get("annotations"):
            continue
        W = anns[-1]["result"][0].get("original_width", 2048) if anns[-1]["result"] else 2048
        H = anns[-1]["result"][0].get("original_height", 1536) if anns[-1]["result"] else 1536
        r1 = rects_of(src_t["annotations"][-1].get("result", []), W, H)
        r2 = rects_of(anns[-1].get("result", []), W, H)
        pairs.append((r1, r2, t["data"].get("original_filename", t["id"])))

    if not pairs:
        print("自检项目还没有完成的盲标，先去标注再跑 --report。")
        return

    # 贪心 IoU 匹配（阈值0.5）
    ious, matched_label_pairs = [], []
    n_unmatched = 0
    for r1s, r2s, name in pairs:
        used = set()
        for a in r1s:
            best, bi = 0.0, None
            for j, b in enumerate(r2s):
                if j in used:
                    continue
                v = iou_box(a["box"], b["box"])
                if v > best:
                    best, bi = v, j
            if best >= 0.5 and bi is not None:
                used.add(bi)
                ious.append(best)
                matched_label_pairs.append((a["label"], r2s[bi]["label"], name))
            else:
                n_unmatched += 1
        n_unmatched += len(r2s) - len(used)

    # Cohen's Kappa（匹配对上类别一致性的标定）
    n = len(matched_label_pairs)
    if n:
        labels = sorted({l for p in matched_label_pairs for l in (p[0], p[1])})
        idx = {l: i for i, l in enumerate(labels)}
        mat = defaultdict(int)
        for a, b, _ in matched_label_pairs:
            mat[(idx[a], idx[b])] += 1
        po = sum(v for (i, j), v in mat.items() if i == j) / n
        row = [sum(mat[(i, j)] for j in range(len(labels))) / n for i in range(len(labels))]
        col = [sum(mat[(i, j)] for i in range(len(labels))) / n for j in range(len(labels))]
        pe = sum(r * c for r, c in zip(row, col))
        kappa = (po - pe) / (1 - pe) if pe < 1 else 1.0
    else:
        po, kappa = 0, 0

    mean_iou = sum(ious) / len(ious) if ious else 0
    print("=" * 60)
    print(f"自检样本 {len(pairs)} 图 | 第一轮框 {sum(len(a) for a,_,_ in pairs)} | "
          f"第二轮框 {sum(len(b) for _,b,_ in pairs)}")
    print(f"IoU≥0.5 匹配 {len(ious)} 对，未匹配 {n_unmatched} 框")
    print(f"平均 IoU = {mean_iou:.3f}")
    print(f"类别观察一致率 po = {po:.3f}   Cohen's Kappa = {kappa:.3f}")
    verdict = "PASS ✔ 可进入量产" if (kappa >= 0.7 and mean_iou >= 0.7) else \
              "FAIL ✘ 先修标注规范再量产（方案验收线 Kappa≥0.7）"
    print(f"验收判定: {verdict}")
    # 混淆明细
    if n:
        print("\n类别混淆（第一轮→第二轮）:")
        for (i, j), v in sorted(mat.items(), key=lambda kv: -kv[1]):
            if i != j:
                print(f"  {labels[i]} → {labels[j]}: {v}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-project-id", type=int, required=True)
    ap.add_argument("--selfcheck-project-id", type=int, default=0)
    ap.add_argument("--create", action="store_true")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    ensure_dirs()
    cfg = load_config()

    if args.create:
        create(cfg, args.source_project_id)
    elif args.report:
        pid = args.selfcheck_project_id
        if not pid and os.path.exists(STATE_PATH):
            with open(STATE_PATH, encoding="utf-8") as f:
                pid = json.load(f)["selfcheck_project"]
        if not pid:
            print("请提供 --selfcheck-project-id 或先运行 --create", file=sys.stderr)
            sys.exit(1)
        report(cfg, args.source_project_id, pid)
    else:
        print("指定 --create 或 --report", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
