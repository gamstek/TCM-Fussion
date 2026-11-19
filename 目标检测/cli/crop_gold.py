# -*- coding: utf-8 -*-
"""crop_gold.py — crop 级盲标 gold set：抽样 → 出标注表 → 算准确率（两阶段 stage 2 的评测入口）。

**为什么需要它**：`det_eval.py` 评的是**框**（定位召回/精确），而两阶段把"命名"挪到了 stage 2，
它的准确率**必须用 crop 级真值**才测得出。此前只能用"stage 2 与 stage 1 的自洽率"代替，那是
循环论证（两边的标签都不是真值）。这是整条检测链当前**最大的证据缺口**（见
`docs/检测两阶段_无类别提框+分类_方案与实验设计.md` §5）。

设计要点（**盲标纪律**，破了就白做）：
  · 导出图的文件名一律是 `001.jpg`…**不含** feature_id/stem/herb，顺序**打乱**（带 seed，可复现）
    ——否则标注者能从文件名与排序里猜出模型的判断与类别。
  · `标注表.csv` **只有编号与空格**（外加一列 `原图路径`，见下），不出现任何模型标签；
    模型答案另存 `_key.json`（报告用，**标注者不要看**）。
  · CSV 用 `utf-8-sig`（Excel 直接双击不开乱码）。

⚠️ **`原图路径` 这一列的代价（用户 2026-10-08 明确要求加入，用于对照原图上下文）**：
  路径里含**药材名**，而药材名是**很强的类别先验**（半夏→针晶、白芍→簇晶）。加了它之后，
  报告出的准确率是"**带药材先验**的分类准确率"，**不再是"仅凭形态"的口径**——两者含义不同，
  `--report` 会把这一点写进输出，别把该数字当"两阶段能否靠形态分开易混类对"的证据。
  （生产链本来就知道药材名，所以就"整链实际效果"而言这个口径反而更贴切；但它回答不了
  "本征不可分"那个问题。若要后者，需去掉该列重跑盲标。）

三个动作（互不耦合）：
  `--build`  按**分层**抽样（stage 2 给出的类别各抽若干，另加一小撮被拒的作对照）→ 出图 + 表 + key
  `--report` 读回标注表 → 整体准确率 / 混淆矩阵 / **易混类对专项** / 拒绝对不对
  `--rerun`  用**当前** prompt 重跑同一批 crop → 「旧 vs 新」配对对比（**改 prompt 后必跑**）
  `--selftest` 离线合成自检（不联网、不落盘）

用法:
  python crop_gold.py --build --n 60 --seed 42 --out-dir 流水线数据/盲标_crop分类
  python crop_gold.py --report --dir 流水线数据/盲标_crop分类
  python crop_gold.py --rerun  --dir 流水线数据/盲标_crop分类 --repeat 5   # 复测（K>1 才报噪声地板）
"""
import argparse
import collections
import csv
import json
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

from ls_common import canon_path, imread_cn, load_config, resolve_key
from prompts import (CLASSIFY_VERSION, NOT_FEATURE_LABEL, REFUSE_LABELS,
                     UNREADABLE_LABEL, VALID_LABELS)

CROPS_ROOT = os.path.join("流水线数据", "detect_v712_full")
STAGE2_RESULTS = os.path.join("流水线数据", "stage2_probe", "results_v2.json")
# 已知"本征易混"的类对（见方案文档 §3.4）：它们的准确率单独看，才能回答
# "错误是集中在不可分的类对上，还是散在别处"
CONFUSABLE = [("纤维", "草酸钙针晶"), ("纤维", "导管"), ("草酸钙针晶", "导管"),
              ("淀粉粒", "草酸钙簇晶"), ("石细胞", "分泌组织")]
LABELS_FOR_HUMAN = list(VALID_LABELS) + list(REFUSE_LABELS)


# ---------------------------------------------------------------- 抽样
def load_pool(results_path, crops_root):
    """从 stage 2 结果里取池子：`{code: {...}}`，每项带源 crop 路径。

    只收**结果文件与盘上 png 都能对上**的项（缺图的静默跳过——盘上 crop 是历史产物）。
    """
    with open(results_path, encoding="utf-8") as f:
        res = json.load(f)["results"]
    pool = []
    for key, v in res.items():
        if "error" in v:
            continue
        herb, crop_id = key.rsplit("/", 1)
        png = os.path.join(crops_root, herb, "crops", crop_id + ".png")
        if not os.path.isfile(png):
            continue
        # 原图路径从 crop 侧车 json 取（契约里就有），过 canon_path —— 存量路径可能是迁移前旧根
        src = ""
        side = os.path.join(crops_root, herb, "crops", crop_id + ".json")
        if os.path.isfile(side):
            try:
                src = canon_path(json.load(open(side, encoding="utf-8"))
                                 .get("source_image_path") or "")
            except Exception:
                src = ""
        pool.append({"crop_id": crop_id, "herb": herb, "png": png,
                     "source_image_path": src if os.path.isfile(src) else "",
                     "stage1_label": v.get("stage1_label"), "stage2_label": v.get("label"),
                     "reject": v.get("reject", ""), "focus": v.get("focus")})
    return pool


def stratify(pool, n_classified, n_rejected, seed):
    """分层抽样：stage 2 给出类别的按类别尽量均分；被拒的另抽（作"拒绝对不对"的对照）。

    **必须打乱输出顺序**：stratify 的结果按类别成组，直接导出会让标注者从前 5 张就看出规律。
    """
    rng = random.Random(seed)
    by_cls = collections.defaultdict(list)
    for it in pool:
        if it["stage2_label"] in VALID_LABELS:
            by_cls[it["stage2_label"]].append(it)
    rejected = [it for it in pool if it["stage2_label"] in REFUSE_LABELS or it["reject"]]

    picked = []
    classes = sorted(by_cls)
    if classes:
        per = max(1, n_classified // len(classes))     # 先均分，再按需补
        for c in classes:
            picked += rng.sample(by_cls[c], min(per, len(by_cls[c])))
    if len(picked) < n_classified:
        rest = [it for c in classes for it in by_cls[c] if it not in picked]
        rng.shuffle(rest)
        picked += rest[:n_classified - len(picked)]
    picked = picked[:n_classified]
    for it in picked:
        it["group"] = "classified"
    rej = rng.sample(rejected, min(n_rejected, len(rejected)))
    for it in rej:
        it["group"] = "rejected"
    out = picked + rej
    rng.shuffle(out)                                   # ← 打乱：标注者看不出分组与顺序规律
    return out


def build(args):
    pool = load_pool(args.results, args.crops_root)
    picked = stratify(pool, args.n, args.n_rejected, args.seed)
    img_dir, sheet_dir = os.path.join(args.out_dir, "images"), os.path.join(args.out_dir, "sheets")
    for d in (img_dir, sheet_dir):
        os.makedirs(d, exist_ok=True)
    key, rows, small = {}, [], []
    for i, it in enumerate(picked, 1):
        code = f"{i:03d}"
        img = imread_cn(it["png"])
        if img is None:
            continue
        cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tofile(
            os.path.join(img_dir, code + ".jpg"))
        key[code] = {k: it[k] for k in ("crop_id", "herb", "stage1_label",
                                        "stage2_label", "reject", "group",
                                        "source_image_path")}
        rows.append({"编号": code, "图片文件": f"images/{code}.jpg",
                     "原图路径": it["source_image_path"] or "(侧车缺失)",
                     "你的判断(填类别名)": "", "备注(可选)": ""})
        small.append((code, img))

    for s in range(0, len(small), 12):                 # 接触图：每张 12 格，便于快速导航
        sheet = make_sheet(small[s:s + 12])
        cv2.imencode(".jpg", sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tofile(
            os.path.join(sheet_dir, f"{s // 12 + 1:02d}.jpg"))

    with open(os.path.join(args.out_dir, "标注表.csv"), "w", encoding="utf-8-sig",
              newline="") as f:
        w = csv.DictWriter(f, fieldnames=["编号", "图片文件", "原图路径",
                                         "你的判断(填类别名)", "备注(可选)"])
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(args.out_dir, "_key.json"), "w", encoding="utf-8") as f:
        json.dump({"seed": args.seed, "results": args.results, "key": key},
                  f, ensure_ascii=False, indent=1)
    with open(os.path.join(args.out_dir, "说明.txt"), "w", encoding="utf-8") as f:
        f.write(
            f"crop 分类盲标（{len(rows)} 张，seed={args.seed}）\n"
            "─────────────────────────────────────────────\n"
            "1. 打开 images/ 下的 001.jpg …，逐张判断**这个实体最像哪一类**，"
            "把类别名原样填进 标注表.csv 的『你的判断』列。\n"
            "2. 可选类别（请逐字复制，不要自创名字）：\n   "
            + "、".join(LABELS_FOR_HUMAN) + "\n"
            "3. 看不清形态 → 填「图像不可判」；看清了但不是特征实体（气泡/杂质/组织背景）→ 填「不是特征」。\n"
            "4. sheets/ 下是每 12 张的缩略接触图，只用于导航；**判定请以 images/ 原尺寸为准**。\n"
            "5. **文件名与顺序已打乱，与模型判断无关**——请完全按自己看到的填。\n"
            "6. 拿不准的可以只填『你的判断』并在备注写疑点，不要跳过编号。\n"
            "7. 『原图路径』列便于对照原图上下文（同视野里还有别的东西时可参考）。\n"
            "   ⚠️ 该路径含**药材名**，而药材名是强类别先验——所以本次准确率是"
            "**带药材先验**的口径，不是『仅凭形态』；两者含义不同，报告里会标明。\n")
    print(f"抽样 {len(rows)} 张（classified {sum(1 for k in key.values() if k['group'] == 'classified')}"
          f" / rejected {sum(1 for k in key.values() if k['group'] == 'rejected')}）")
    print(f"  图 → {img_dir}｜接触图 → {sheet_dir}｜表 → 标注表.csv｜答案 → _key.json（勿给标注者）")
    print(f"  各类分布: {dict(collections.Counter(v['stage2_label'] for v in key.values()))}")


def make_sheet(items, cell=380, cols=4):
    """12 张拼一张接触图，每格上方印编号（判定仍以原尺寸为准）。"""
    rows = (len(items) + cols - 1) // cols
    sheet = 255 * np.ones((rows * (cell + 28), cols * cell, 3), "uint8")
    for i, (code, img) in enumerate(items):
        h, w = img.shape[:2]
        s = min(cell / w, cell / h)
        th = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
        r, c = divmod(i, cols)
        y0, x0 = r * (cell + 28) + 28, c * cell
        sheet[y0:y0 + th.shape[0], x0:x0 + th.shape[1]] = th
        cv2.putText(sheet, code, (x0 + 4, y0 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 200), 2)
    return sheet


# ---------------------------------------------------------------- 报告
def evaluate(rows, key):
    """纯函数：`rows=[{编号,判断}]` × `key` → 指标字典（离线、可自检）。"""
    cl = {"n": 0, "ok": 0, "conf": collections.Counter(), "blank": 0, "illegal": []}
    rj = {"n": 0, "agree_unreadable": 0, "agree_notfeat": 0, "called_feature": 0, "other": 0}
    yes_like = collections.Counter()
    pair = collections.Counter()
    pair_ok = collections.Counter()
    for r in rows:
        code = str(r["编号"]).strip()
        if code not in key:
            continue
        k = key[code]
        ans = (r.get("你的判断(填类别名)") or r.get("你的判断") or "").strip()
        if k["group"] == "classified":
            cl["n"] += 1
            if not ans:
                cl["blank"] += 1
                continue
            if ans not in LABELS_FOR_HUMAN:
                cl["illegal"].append((code, ans))
                continue
            if ans == k["stage2_label"]:
                cl["ok"] += 1
            else:
                cl["conf"][(k["stage2_label"], ans)] += 1
            a, b = k["stage2_label"], ans
            for x, y in CONFUSABLE:
                if {x, y} == {a, b} or (a == b and a in (x, y)):
                    pair[(x, y)] += 1
                    if a == b:
                        pair_ok[(x, y)] += 1
                    break
        else:
            rj["n"] += 1
            yes_like[ans] += 1
            if ans == "图像不可判":
                rj["agree_unreadable"] += 1
            elif ans == "不是特征":
                rj["agree_notfeat"] += 1
            elif ans in VALID_LABELS:
                rj["called_feature"] += 1
            else:
                rj["other"] += 1
    return {"classified": cl, "rejected": rj, "pairs": dict(pair), "pairs_ok": dict(pair_ok),
            "rejected_answers": dict(yes_like)}


def jsonable(rep):
    """把 evaluate() 的结果转成可 json 落盘的结构。

    混淆/易混对用 `(a, b)` **元组**作键——json 的键只能是 str/int/float/bool/None，
    元组键会让 `json.dump` 直接抛 `TypeError`（`default=` 只作用于**值**、救不了键）。
    落盘前统一转成 `"a→b"` / `"a↔b"` 文本。
    """
    out = dict(rep)
    cl = dict(rep["classified"])          # 浅拷贝：别把原 rep 的 conf 键改掉
    cl["conf"] = {f"{a}→{b}": c for (a, b), c in rep["classified"]["conf"].items()}
    out["classified"] = cl
    out["pairs"] = {f"{a}↔{b}": v for (a, b), v in rep["pairs"].items()}
    out["pairs_ok"] = {f"{a}↔{b}": v for (a, b), v in rep["pairs_ok"].items()}
    return out


def report(args):
    with open(os.path.join(args.dir, "_key.json"), encoding="utf-8") as f:
        key = json.load(f)["key"]
    with open(os.path.join(args.dir, "标注表.csv"), encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    rep = evaluate(rows, key)
    cl, rj = rep["classified"], rep["rejected"]
    print("口径提示：标注表含『原图路径』（含药材名），故下列准确率是【带药材先验】的——"
          "\n  生产链本来就知道药材名，用于评估整链效果合适；但它**不能**用来回答"
          "『两阶段能否仅凭形态分开易混类对』。")
    print(f"\n分类组 {cl['n']} 张：准确率 {cl['ok']}/{cl['n'] - cl['blank']}"
          f" = {100.0 * cl['ok'] / max(1, cl['n'] - cl['blank']):.0f}%"
          f"（空 {cl['blank']}、非法答案 {len(cl['illegal'])}）")
    if cl["illegal"]:
        # **必须点名**：非法答案会被 evaluate 直接从分母里剔除，不打印出来标注者根本不知道
        # 自己哪一行被丢了（本轮就踩到：053 填了词表外的「草酸钙柱晶」）。
        print("  ⚠️ 非法答案（不在可选类别里，已从分母剔除）："
              + "、".join(f"{c}={a}" for c, a in cl["illegal"]))
    if cl["conf"]:
        print("  混淆（stage2 → 人工）top10：")
        for (a, b), c in cl["conf"].most_common(10):
            print(f"    {a:<8} → {b:<8} {c}")
    if rep["pairs"]:
        print("  易混类对专项（该对上的判定数 / 其中判对数）：")
        for k, v in rep["pairs"].items():
            print(f"    {k[0]}↔{k[1]}: {v} / {rep['pairs_ok'].get(k, 0)}")
    print(f"\n被拒组 {rj['n']} 张（模型判『不是特征/图像不可判』，看人工是否同意）："
          f" 同意不可判 {rj['agree_unreadable']}｜同意不是特征 {rj['agree_notfeat']}"
          f"｜人工反而给出特征类 {rj['called_feature']}｜其他 {rj['other']}")
    print(f"  人工答案分布: {rep['rejected_answers']}")
    out = os.path.join(args.dir, "report.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(jsonable(rep), f, ensure_ascii=False, indent=1)
    print(f"\n→ {out}")


# ---------------------------------------------------------------- 复测
def _rerun_acc(rep):
    cl = rep["classified"]
    return cl["ok"], cl["n"] - cl["blank"]


def _majority(labels):
    """K 次重复的多数票 → `(多数标签, 票数, 是否并列)`；并列取**出现次序最早**者（可复现）。"""
    cnt = collections.Counter(labels)
    top = max(cnt.values())
    maj = next(lab for lab in labels if cnt[lab] == top)
    return maj, top, sum(1 for v in cnt.values() if v == top) > 1


def rerun(args):
    """用**当前** prompt 重跑同一批 crop → 「旧 vs 新」配对对比；`--repeat K>1` **重复采样降方差**。

    为什么要它（评估纪律，见 docs/检测两阶段_… §5）：`--report` 只读烘焙进 `_key.json` 的旧
    stage2 标签，改 prompt 后它报出来的还是旧数；拿"历史基线"比又会混进模型逐次波动。

    为什么要 `--repeat`（2026-10-09 实测教训，代价惨痛）：判据改动复测出 47%→32%，但补做的
    **同刻对照臂（同一份 prompt 重跑）也只有 37%**——"同一 prompt 自己重跑"就翻转了 22/60 的
    crop、净差 6 张，**比处理效应（3 张）还大**。单轮 n=60 的数字根本分辨不了改动优劣。
    故本模式对每张 crop 采样 K 次，报 **K 轮准确率的均值/极差**（≈噪声地板）与 **多数票准确率**
    （降方差的点估计），并把逐轮结果与一致度落盘，供定位"不稳定样本"。

    产物 `_rerun_key.json`（多数票标签 + 逐轮明细 + 一致度）＋ `report_rerun.json`（旧/新指标）；
    原 `_key.json` 与 `标注表.csv` **只读不改**，基线可随时复算。
    """
    from classify_crop import classify_crop      # 迟导入：--selftest 保持离线、不拉起检测链

    repeat = max(1, int(args.repeat))
    with open(os.path.join(args.dir, "_key.json"), encoding="utf-8") as f:
        key_old = json.load(f)["key"]
    with open(os.path.join(args.dir, "标注表.csv"), encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    cfg = load_config()
    api_key = resolve_key(cfg)
    if not api_key:
        sys.exit("[FAIL] 未配置 VLM API key（VLM_API_KEY 环境变量或 vlm_config.json）")

    items = sorted(key_old.items())
    imgs = {c: imread_cn(os.path.join(args.crops_root, k["herb"], "crops",
                                      k["crop_id"] + ".png")) for c, k in items}

    def one(task):                                # 摊平成 K×N 个任务，让并发吃满
        code, i = task
        if imgs[code] is None:
            return code, i, {"label": None, "raw_label": "(crop 缺失)", "reject": ""}
        return code, i, classify_crop(imgs[code], cfg["vlm"], api_key)

    runs_of = collections.defaultdict(dict)
    with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as ex:
        tasks = [(c, i) for c, _ in items for i in range(repeat)]
        for code, i, r in ex.map(one, tasks):
            runs_of[code][i] = r

    def lab_of(r):
        # 清单外答案（label=None）落 raw_label：**不静默映射**到"最像"的类，如实计入错误
        lab = r.get("label")
        return lab if lab is not None else (r.get("raw_label") or "")

    key_new, detail, n_err = json.loads(json.dumps(key_old)), {}, 0
    for code, k in key_new.items():
        batch = [runs_of[code].get(i, {}) for i in range(repeat)]
        if any(r.get("error") for r in batch):
            n_err += 1
        labs = [lab_of(r) for r in batch]
        maj, votes, tied = _majority(labs)
        k["stage2_label"] = maj
        k["reject"] = ("not_feature" if maj == NOT_FEATURE_LABEL else
                       "unreadable" if maj == UNREADABLE_LABEL else "")
        detail[code] = {"runs": labs, "majority": maj, "votes": votes, "tied": tied}

    human = {str(r["编号"]).strip(): (r.get("你的判断(填类别名)") or "").strip() for r in rows}
    cls = [c for c, k in key_new.items() if k.get("group") == "classified" and c in human]
    run_acc = [sum(1 for c in cls if detail[c]["runs"][i] == human[c]) for i in range(repeat)]

    rep_old, rep_new = evaluate(rows, key_old), evaluate(rows, key_new)
    ok_old, n_cls = _rerun_acc(rep_old)
    ok_new, _ = _rerun_acc(rep_new)

    fixed, broke = [], []
    for code, k in key_new.items():
        if code not in cls:
            continue
        ans, o, n = human[code], key_old[code]["stage2_label"], k["stage2_label"]
        if o != ans and n == ans:
            fixed.append(f"{code} {k['herb']} {o}→{n}")
        elif o == ans and n != ans:
            broke.append(f"{code} {k['herb']} {o}→{n}")

    print(f"复测（分类组 {n_cls} 张；当前 prompt = {CLASSIFY_VERSION}；每张重复 {repeat} 次）")
    if repeat > 1:
        print(f"  逐轮命中 {run_acc} → 均值 {sum(run_acc) / repeat:.1f}/{n_cls}，"
              f"极差 {max(run_acc) - min(run_acc)} 张  ← **这就是噪声地板，先看它再看下面的差值**")
    print(f"  {'多数票' if repeat > 1 else '本轮'}命中 {ok_new}/{n_cls} = "
          f"{100.0 * ok_new / max(1, n_cls):.0f}%"
          f"   （旧基线 {ok_old}/{n_cls} = {100.0 * ok_old / max(1, n_cls):.0f}%）")
    if n_err:
        print(f"  ⚠️ {n_err} 张出现调用失败（按空标签计入错误，未静默剔出分母）")
    if repeat > 1:
        unstable = [f"{c} {key_new[c]['herb']} {detail[c]['votes']}/{repeat}"
                    for c in sorted(detail) if detail[c]["votes"] < repeat]
        print(f"  不稳定样本（票数 < {repeat}）{len(unstable)}/{len(detail)}："
              f"{'；'.join(unstable) or '无'}")
        tied = [c for c in sorted(detail) if detail[c]["tied"]]
        if tied:
            print(f"  ⚠️ 最高票并列（须复核，不算「稳定」）{len(tied)}：{'；'.join(tied)}")
    print(f"  改对 {len(fixed)}：{'；'.join(fixed) or '无'}")
    print(f"  改坏 {len(broke)}：{'；'.join(broke) or '无'}")
    if rep_new["classified"]["conf"]:
        print("  新混淆（stage2 → 人工）top10：")
        for (a, b), c in rep_new["classified"]["conf"].most_common(10):
            print(f"    {a:<8} → {b:<8} {c}")
    still = sum(1 for k in key_new.values()
                if k.get("group") == "rejected" and k["stage2_label"] in REFUSE_LABELS)
    print(f"  被拒组：新预测仍判拒绝 {still}/8｜（人工答案分布 {rep_new['rejected_answers']}）")

    out_key = os.path.join(args.dir, "_rerun_key.json")
    out_rep = os.path.join(args.dir, "report_rerun.json")
    with open(out_key, "w", encoding="utf-8") as f:
        json.dump({"rerun_of": "_key.json", "classify_version": CLASSIFY_VERSION,
                   "repeat": repeat, "key": key_new, "detail": detail}, f,
                  ensure_ascii=False, indent=1)
    with open(out_rep, "w", encoding="utf-8") as f:
        json.dump({"old": jsonable(rep_old), "new": jsonable(rep_new),
                   "repeat": repeat, "run_acc": run_acc}, f, ensure_ascii=False, indent=1)
    print(f"\n→ {out_rep}｜{out_key}")


# ---------------------------------------------------------------- 自检
def selftest():
    checks = []

    def ck(name, cond):
        checks.append((name, bool(cond)))

    pool = []
    for lab, cnt in (("淀粉粒", 10), ("纤维", 10), ("导管", 5), ("草酸钙针晶", 5)):
        pool += [{"crop_id": f"{lab}_{i}", "herb": "X", "png": f"{lab}_{i}.png",
                  "stage1_label": lab, "stage2_label": lab, "reject": "", "focus": 1.0}
                 for i in range(cnt)]
    pool.append({"crop_id": "r1", "herb": "X", "png": "r1.png", "stage1_label": "淀粉粒",
                 "stage2_label": "图像不可判", "reject": "unreadable", "focus": 0.1})
    picked = stratify(pool, 12, 1, 42)
    ck("抽样数 = 13", len(picked) == 13)
    ck("被拒组各 1", sum(1 for p in picked if p["group"] == "rejected") == 1)
    ck("分层：4 类各被抽到", len({p["stage2_label"] for p in picked
                              if p["group"] == "classified"}) == 4)
    ck("顺序被打乱（不是按类别成组）",
       [p["stage2_label"] for p in picked].count(picked[0]["stage2_label"]) < 13)
    ck("同 seed 可复现",
       [p["crop_id"] for p in stratify(pool, 12, 1, 42)] == [p["crop_id"] for p in picked])
    ck("换 seed 会变",
       [p["crop_id"] for p in stratify(pool, 12, 1, 7)] != [p["crop_id"] for p in picked])

    key = {"001": {"group": "classified", "stage2_label": "淀粉粒"},
           "002": {"group": "classified", "stage2_label": "纤维"},
           "003": {"group": "classified", "stage2_label": "纤维"},
           "004": {"group": "classified", "stage2_label": "导管"},
           "005": {"group": "rejected", "stage2_label": "图像不可判"},
           "006": {"group": "rejected", "stage2_label": "不是特征"}}
    rows = [{"编号": "001", "你的判断(填类别名)": "淀粉粒"},
            {"编号": "002", "你的判断(填类别名)": "草酸钙针晶"},   # 易混：纤维↔针晶
            {"编号": "003", "你的判断(填类别名)": "纤维"},          # 对，且属易混对
            {"编号": "004", "你的判断(填类别名)": ""},              # 空
            {"编号": "005", "你的判断(填类别名)": "图像不可判"},
            {"编号": "006", "你的判断(填类别名)": "草酸钙簇晶"}]     # 人工反而给出特征类
    r = evaluate(rows, key)
    ck("分类组 n=4 / 答对 2", (r["classified"]["n"], r["classified"]["ok"]) == (4, 2))
    ck("空判计 1", r["classified"]["blank"] == 1)
    ck("混淆 纤维→草酸钙针晶 记 1",
       r["classified"]["conf"][("纤维", "草酸钙针晶")] == 1)
    ck("易混对 纤维↔针晶 判 2 次对 1 次",
       r["pairs"][("纤维", "草酸钙针晶")] == 2
       and r["pairs_ok"][("纤维", "草酸钙针晶")] == 1)
    ck("被拒组：同意不可判 1 / 人工给特征类 1",
       (r["rejected"]["agree_unreadable"], r["rejected"]["called_feature"]) == (1, 1))

    rows2 = [{"编号": "001", "你的判断(填类别名)": "外星晶体"}]
    r2 = evaluate(rows2, key)
    ck("非法答案不计入正确率", r2["classified"]["ok"] == 0
       and len(r2["classified"]["illegal"]) == 1)
    ck("报告可落盘（元组键已转文本）",
       bool(json.dumps(jsonable(r), ensure_ascii=False)))

    npass = sum(1 for _, v in checks if v)
    for name, v in checks:
        print(f"  [{'PASS' if v else 'FAIL'}] {name}")
    print(f"\ncrop_gold selftest: {npass}/{len(checks)} PASS")
    return npass == len(checks)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--rerun", action="store_true",
                    help="用当前 prompt 重跑同一批 crop 并出「旧 vs 新」对比（改 prompt 后必跑）")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--dir", default=os.path.join("流水线数据", "盲标_crop分类"))
    ap.add_argument("--n", type=int, default=60, help="分类组抽样张数（按 stage2 类别分层）")
    ap.add_argument("--n-rejected", type=int, default=8, help="被拒组张数（对照：拒绝对不对）")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--results", default=STAGE2_RESULTS)
    ap.add_argument("--crops-root", default=CROPS_ROOT)
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--workers", type=int, default=4, help="--rerun 的并发数（≤8）")
    ap.add_argument("--repeat", type=int, default=1,
                    help="--rerun 每张 crop 重复采样次数（K>1 才报得出噪声地板与多数票）")
    args = ap.parse_args(argv)
    if args.selftest:
        return 0 if selftest() else 1
    if args.build:
        if args.out_dir:
            args.dir = args.out_dir
        build(args)
        return 0
    if args.report:
        report(args)
        return 0
    if args.rerun:
        rerun(args)
        return 0
    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
