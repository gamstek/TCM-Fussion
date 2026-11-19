# -*- coding: utf-8 -*-
"""
det_eval.py — 检测质量评测：gold set 抽样 + 指标报告（P0-1）

为什么需要它（2026-09-24 检测准确率分析结论）：
  此前 7 轮 prompt 迭代（v4 零框漏检 → v5/v6 补漏 → v7.1/v7.2）的取舍全靠"个案目检"，
  因为主试点 417 任务 **人工标注 0 条**（LS 库核验），漏检/误检率无法测得。
  本脚本给出唯一可复现的口径：固定 gold set（人工**盲标全量框**，不带任何 AI 预标注）
  + 每次改 prompt / 换模型后重跑 `--report`，用同一批图对比同一组数字。

动作（互不耦合，可单独跑）：
  ① --build-gold   按类分层抽 N 张 → det_eval/gold_set.json（带 seed，可复现）
      └ --import-ls 追加：建「检测评测-gold(盲标)」项目并导入，**不带 predictions**
        （关键：带预标注就是自证。人工必须独立全量框出所有实例）
      └ --gold-from-p0：原 blind gold 不在本机时，从 P0 split_report.gold_eval +
        单类导出 labels 重建 60 张 gold + 人工框（非盲标，仅供 YOLO 臂；会写 provenance 警告）
  ② --report       读 gold_set + 人工标注（ann-json / LS 导出 JSON / 活 LS 项目）→ 与预测比对
      └ --engine vlm|yolo：vlm 读 prelabels；yolo 按 detector.json 现跑 gold 图
      └ --pred-dir：读离线 YOLO 预测（每图一份 DetBox JSON，不跑推理）
      └ --p3：主指标剔除「其他」+ 逐类 AP50/AP50-95、混淆矩阵、P0 六类门槛、
              其他类单独、跨药材子集、原始 score 的 ECE/Brier 诊断
  ③ --selftest     离线合成真值自检（不联网、不落盘）

指标口径（对应分析报告里的缺陷清单）：
  - recall_loc  ：定位召回（类别无关，IoU≥阈值）——只考"找到没找到"
  - recall_e2e  ：端到端召回（类别也对）——定位与识别误差分开看
  - precision   ：以 **预测框的类别** 归属计算，答错类别的框同时拉低该类精确度
  - IoU 分布    ：中位/IQR/最小 + 严格档（0.75，测量级贴合）
  - 覆盖率偏差  ：**带符号** area_人工/area_预测 的中位（>1 = 系统性框小了 → 测量偏低）
                  贴边框的 pair 面积不可信，单独剔除后另算 clean 档
  - 贴边率/巨型框：坐标被夹紧、面积>50% 画面的框——下游会当完整实体测径，需显式标记
  - 框数误差    ：每图 |n_pred - n_human| —— 补漏/过分割的双向回归检测

用法:
  python det_eval.py --build-gold --n 60              # 抽样（只出清单）
  python det_eval.py --build-gold --n 60 --import-ls  # 抽样 + 建盲标项目（需 LS 在跑）
  python det_eval.py --report --project-id 31         # 人工标完后出报告
  python det_eval.py --report --export-json ls_export.json
  python det_eval.py --gold-from-p0 --out-dir results/detection/det_eval/p3
  python det_eval.py --report --engine yolo --p3 \
      --gold-set results/detection/det_eval/p3/gold_set.json \
      --ann-json results/detection/det_eval/p3/gold_ann.json \
      --out-dir results/detection/det_eval/p3
  python det_eval.py --selftest                       # 离线自检
"""
import argparse
import json
import math
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict


from contracts import PrelabelRecord  # noqa: E402 第 ⑤ 段契约（prelabel 记录；C2a 起 index 值即此类型）
import detector_yolo  # noqa: E402 P3：YOLO 预测（与 detect_sample 共用同一适配器/配置解析）
from geom import iou_box  # noqa: E402 框 IoU 原语（不再从 05 脚本借）
from ls_common import (DET_CONFIG, DET_RESULTS, PRELABEL_DIR, PROJECT_ROOT, canon_path,  # noqa: E402
                       ensure_dirs, imread_cn, load_config, load_prelabel_index, ls_api, norm_path)
from ls_tasks import (ann_from_tasks, ensure_local_storage, fetch_tasks,  # noqa: E402
                      import_in_chunks)

EVAL_DIR = os.path.join(DET_RESULTS, "det_eval")
GOLD_PATH = os.path.join(EVAL_DIR, "gold_set.json")
GOLD_ANN_PATH = os.path.join(EVAL_DIR, "gold_ann.json")
LABEL_CONFIG_PATH = os.path.join(DET_CONFIG, "labeling_config.xml")
GOLD_PROJECT_TITLE = "检测评测-gold(盲标)"
P0_SPLIT_REPORT = os.path.join(PROJECT_ROOT, "dataset", "检测训练集_v1", "split_report.json")
P0_LABEL_ROOT = os.path.join(PROJECT_ROOT, "dataset", "目标检测标注")

IOU_THR = 0.5        # 主档：定位/召回判定
STRICT_IOU = 0.75    # 严格档：贴合到"可当测量用"的程度
BIG_BOX_RATIO = 0.50  # 单框占画面 >50% → 巨型框（crop 等于原图）
COVERAGE_TOL = 0.01   # 覆盖率偏差"视为相等"的容差：LS 百分比存 3 位小数，回算会有 ~1e-6
                      # 相对误差；不设容差时"人工更大/预测更大"两栏会被浮点噪声灌满。

# P3 验收（方案 §6.5）：主门槛之外，P0 六类逐类同样过 recall_e2e/precision 线；
# 「其他」类单独报告，不进主门槛（它没有 IGM 特征描述，是识别类而非交付类）。
P0_CLASSES = ("草酸钙簇晶", "草酸钙方晶", "草酸钙针晶", "导管", "石细胞", "淀粉粒")
P0_CLASS_GATE = 0.85
OTHER_LABEL = "其他"
CONFUSABLE_PAIRS = (("纤维", "草酸钙针晶"), ("淀粉粒", "草酸钙簇晶"), ("石细胞", "分泌组织"))
AP_IOU_THRS = tuple(round(0.5 + 0.05 * i, 2) for i in range(10))   # AP50 与 AP50-95

# 验收线（改 prompt/换模型后按此判定）
GATE_RECALL_E2E = 0.85
GATE_PRECISION = 0.85
GATE_MEDIAN_IOU = 0.85
GATE_COVERAGE_DEV = 0.10   # |覆盖率偏差中位 - 1| ≤ 0.10


def _median(vals):
    """标准中位（偶数个取中间两值均值）；空返回 None。"""
    s = sorted(vals)
    n = len(s)
    if not n:
        return None
    return (s[(n - 1) // 2] + s[n // 2]) / 2.0


# ---------------------------------------------------------------- ① 分层抽样
def allocate(counts, n, min_n, max_n):
    """按类按比例分配名额（最大余数 + 上/下限钳制），返回 {类: 名额}，总和 ≤ n。

    下限保证小类也进 gold（否则 分泌组织/菌丝 这类只有 2~3 张的永远没样本可评）。
    """
    keys = sorted(counts)
    total = sum(counts.values())
    if total <= n:
        return dict(counts)
    q = {k: max(min_n, min(max_n, int(round(n * counts[k] / total)))) for k in keys}
    q = {k: min(q[k], counts[k]) for k in keys}
    order = sorted(keys, key=lambda k: (-q[k], k))
    guard = 0
    while sum(q.values()) > n and guard < 10 * len(keys):
        guard += 1
        for k in order:
            if sum(q.values()) <= n:
                break
            if q[k] > min(min_n, counts[k]):
                q[k] -= 1
    i = 0
    while sum(q.values()) < n and i < 10 * len(keys):
        k = order[i % len(order)]
        if q[k] < min(max_n, counts[k]):
            q[k] += 1
        i += 1
    return q


def build_gold(n, seed, min_per_class, max_per_class, dry_run=False):
    idx = load_prelabel_index()
    if not idx:
        sys.exit(f"[FAIL] {PRELABEL_DIR} 下没有 prelabel，请先跑 vlm_prelabel.py")
    groups, stale = defaultdict(list), []
    for p, pre in idx.items():
        # 源图已改名/删除的 prelabel 不能进 gold：LS 里图片会 500，人工无从标注。
        # 2026-09-24 实测 422 份中有 6 份属于此列（大黄导管/大黄淀粉粒 的 4/5/6、7/8/9）。
        if not os.path.isfile(pre.image_path):
            stale.append((pre.feature_label_folder or "?", pre.image_path))
            continue
        groups[pre.feature_label_folder or "?"].append(p)
    if stale:
        print(f"⚠️ 跳过 {len(stale)} 份源图缺失的 prelabel：")
        for f, p in stale:
            print(f"    {f:<10} ← {p}")
    counts = {k: len(v) for k, v in groups.items()}
    quota = allocate(counts, n, min_per_class, max_per_class)

    rng = random.Random(seed)
    items = []
    print(f"gold 抽样：池 {sum(counts.values())} 张 / {len(counts)} 类 → 目标 {n} 张（seed={seed}）")
    for k in sorted(groups):
        pool = sorted(groups[k])          # 先排序保证跨机一致，再抽样
        picked = rng.sample(pool, min(quota.get(k, 0), len(pool)))
        print(f"  {k:<12} 池 {counts[k]:>4} 张 → 抽 {len(picked):>3}"
              f"（现有框均 {sum(len(idx[p].objects) for p in picked)/max(1,len(picked)):.1f}）")
        for p in picked:
            pre = idx[p]
            items.append({"image_path": pre.image_path, "feature_label": k,
                          "herb": pre.herb, "group_key": pre.group_key,
                          "stem": os.path.splitext(os.path.basename(pre.image_path))[0],
                          "n_prelabel_boxes": len(pre.objects),
                          "ls_task_id": None})
    out = {"created_at": time.strftime("%Y-%m-%d %H:%M:%S"), "seed": seed, "n": len(items),
           "source": "prelabels", "purpose": "人工盲标全量框（不得带 AI 预标注，否则评测自证）",
           "skipped_stale": len(stale), "pool": len(idx) - len(stale), "items": items}
    if dry_run:
        print(f"\n[DRY-RUN] 不落盘；将写 {len(items)} 项 → {GOLD_PATH}")
        return out
    os.makedirs(EVAL_DIR, exist_ok=True)
    with open(GOLD_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\ngold set 已落盘：{len(items)} 项 → {GOLD_PATH}")
    print("下一步：--import-ls 建盲标项目，或人工在任意工具里标完再用 --report")
    return out


# ---------------------------------------------------------------- ①b 盲标项目导入
def fetch_tasks_light(cfg, pid):
    """分页拉任务（默认字段即含 id/data；不需要 fields=all）。"""
    out, page = [], 1
    while True:
        j = ls_api(cfg, "GET", "/api/tasks",
                   params={"project": pid, "page": page, "page_size": 100}).json()
        batch = j["tasks"] if isinstance(j, dict) and "tasks" in j else j
        if not batch:
            break
        out.extend(batch)
        if len(batch) < 100:
            break
        page += 1
    return out


def find_project_by_title(cfg, title):
    """按标题找已有项目（分页遍历）；没有返回 None。"""
    page = 1
    while True:
        j = ls_api(cfg, "GET", "/api/projects",
                   params={"page": page, "page_size": 100}).json()
        items = j.get("results") if isinstance(j, dict) else j
        if not items:
            return None
        for p in items:
            if p.get("title") == title:
                return p["id"]
        if len(items) < 100:
            return None
        page += 1


def import_to_ls(gold, dry_run=False):
    """建「检测评测-gold(盲标)」项目并导入 —— **不带 predictions**（盲标，防评测自证）。

    幂等：同名项目已存在时复用；若该项目已有**人工标注**则拒绝覆盖并退出
    （60 张盲标是长耗时人工投入，绝不能被重跑脚本抹掉）；无标注则清空旧任务重建。
    """
    from ls_common import criteria_hint_for, ensure_tif_proxy, view_file_url
    cfg = load_config()
    tasks = []
    for it in gold["items"]:
        p = it["image_path"]
        if not dry_run:
            ensure_tif_proxy(p)           # tif 入 LS 前必须先落 jpg 代理（dry-run 不落盘）
        tasks.append({"data": {
            "image": view_file_url(p),
            "original_filename": os.path.basename(p),
            "image_path": p,              # 评测回链主键（导出 JSON 也能对上）
            "herb": it.get("herb", ""),
            "folder_feature": it.get("feature_label", ""),
            "group_key": it.get("group_key", ""),
            "criteria_hint": criteria_hint_for(it.get("herb", ""), it.get("feature_label", ""), ""),
            "eval_note": "gold 盲标：独立全量框出所有可见特征实例，不要参考任何 AI 结果",
        }})
    if dry_run:
        print(f"[DRY-RUN] 将建项目「{GOLD_PROJECT_TITLE}」并导入 {len(tasks)} 个盲标任务"
              f"（不带 predictions）")
        if tasks:
            print(f"[DRY-RUN] 示例 data: {json.dumps(tasks[0]['data'], ensure_ascii=False)[:220]}")
        return
    with open(LABEL_CONFIG_PATH, encoding="utf-8") as f:
        label_config = f.read()
    pid = find_project_by_title(cfg, GOLD_PROJECT_TITLE)
    if pid:
        existing = fetch_tasks_light(cfg, pid)
        n_ann = sum(len(t.get("annotations") or []) for t in existing)
        if n_ann:
            sys.exit(f"[FAIL] 项目 {pid}「{GOLD_PROJECT_TITLE}」已有 {n_ann} 条人工标注，"
                     f"拒绝覆盖。请在 LS 里改名或删除该项目后再重跑。")
        for t in existing:
            ls_api(cfg, "DELETE", f"/api/tasks/{t['id']}/")
        print(f"复用已有项目 id={pid}（清空 {len(existing)} 个旧任务后重建，无人工标注损失）")
    else:
        pid = ls_api(cfg, "POST", "/api/projects", json={
            "title": GOLD_PROJECT_TITLE, "label_config": label_config,
            "expert_instruction": "gold 盲标：不看任何 AI 结果，独立把全图所有可辨识特征实例"
                                  "逐一框出（全量，勿抽样）；框画法紧贴实体外接矩形不得切断实体。"
                                  "确实一个都没有才提交空标注。",
        }).json()["id"]
        print(f"已建盲标项目 id={pid}「{GOLD_PROJECT_TITLE}」")
    ensure_local_storage(cfg, pid)
    n = import_in_chunks(cfg, pid, tasks)
    url2id = {t["data"]["image"]: t["id"] for t in fetch_tasks_light(cfg, pid)}
    hit = 0
    for it, t in zip(gold["items"], tasks):
        tid = url2id.get(t["data"]["image"])
        if tid:
            it["ls_task_id"] = tid
            hit += 1
    gold["ls_project_id"] = pid
    with open(GOLD_PATH, "w", encoding="utf-8") as f:
        json.dump(gold, f, ensure_ascii=False, indent=2)
    print(f"导入 {n} 个盲标任务，回链 task_id {hit}/{len(tasks)} → {GOLD_PATH}")
    print(f"标注入口: {cfg['labelstudio']['url']}/projects/{pid}/data")


# ---------------------------------------------------------------- ①b P3 数据源
def read_yolo_labels(path, W, H, label):
    """单类 YOLO 导出集的 `labels/<stem>.txt` → `[{"label", "box"}]`（原图像素）。"""
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 5:
                continue
            cx, cy, w, h = [float(v) for v in parts[1:5]]
            box = [(cx - w / 2) * W, (cy - h / 2) * H,
                   (cx + w / 2) * W, (cy + h / 2) * H]
            box = [min(max(v, 0.0), float(W if i % 2 == 0 else H))
                   for i, v in enumerate(box)]
            if box[2] - box[0] < 1 or box[3] - box[1] < 1:
                continue
            out.append({"label": label, "box": [round(v, 1) for v in box]})
    return out


def build_gold_from_p0(split_report=P0_SPLIT_REPORT, label_root=P0_LABEL_ROOT,
                       out_dir=EVAL_DIR, dry_run=False, overwrite=False):
    """从 P0 `split_report.json` 的 `gold_eval` 记录重建 60 张 gold 与人工框。

    **为什么需要**：原 `det_eval/gold_set.json` 与盲标 LS 项目不在本机（都是 Windows 侧
    results/ 产物）；P0 已把这 60 张按 md5 整帧预留、不参与训练，且单类导出集里保留了
    人工框。故可从「gold_eval 记录 + 单类 labels」重建同一批图/框（框数应与
    `gold_eval_count.boxes` 对账）。产物写 `gold_set.json` + `gold_ann.json`。

    ⚠ 这批框来自 P0 标注语料，**不是**原「gold(盲标)」项目的独立盲标结果；用于 YOLO
    臂评测没有训练泄漏（这 60 张未参与训练），但不能冒充 VLM 的同图盲标金标准。
    """
    with open(split_report, encoding="utf-8") as f:
        report = json.load(f)
    records = [r for r in report.get("images", []) if r.get("split") == "gold_eval"]
    if not records:
        raise ValueError("split_report 里没有 gold_eval 记录：{}".format(split_report))
    items, ann = [], {}
    n_boxes, missing = 0, []
    for r in records:
        boxes, image_path = [], ""
        for s in r.get("sources") or []:
            ip = os.path.join(label_root, s["dir"], "images", s["name"])
            lp = os.path.join(label_root, s["dir"], "labels",
                              os.path.splitext(s["name"])[0] + ".txt")
            if not os.path.isfile(ip) or not os.path.isfile(lp):
                missing.append((s["dir"], s["name"]))
                continue
            img = imread_cn(ip)
            if img is None:
                missing.append((s["dir"], s["name"] + "（读图失败）"))
                continue
            H, W = img.shape[:2]
            if not image_path:
                image_path = ip
            boxes.extend(read_yolo_labels(lp, W, H, s["dir"]))
        if not image_path:
            continue
        # 合并重复来源时同一框可能出现两次：按 (label, 坐标四舍五入) 去重。
        uniq = {}
        for b in boxes:
            uniq[(b["label"], tuple(round(v, 1) for v in b["box"]))] = b
        boxes = list(uniq.values())
        n_boxes += len(boxes)
        image_path = os.path.abspath(image_path)
        items.append({
            "image_path": image_path,
            "feature_label": (r["sources"][0]["dir"] if len(r.get("sources") or []) == 1
                              else "multi"),
            "herb": r.get("herb", ""),
            "group_key": r.get("group_key", ""),
            "stem": os.path.splitext(os.path.basename(image_path))[0],
            "cross_herb": bool(r.get("cross_herb")),
            "n_prelabel_boxes": len(boxes),
            "source_dirs": [s["dir"] for s in r.get("sources") or []],
            "ls_task_id": None,
        })
        ann[norm_path(image_path)] = boxes
    expected = (report.get("gold_eval_count") or {}).get("boxes")
    if expected is not None and expected != n_boxes:
        raise ValueError("重建框数 {} 与 split_report.gold_eval_count.boxes={} 不一致：{}".format(
            n_boxes, expected, missing[:5]))
    gold = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source": "P0 split_report.gold_eval + dataset/目标检测标注 单类 labels",
        "purpose": "YOLO 臂 gold 评测（这 60 张未参与训练）；非原 blind gold 项目",
        "provenance_warning": "人工框来自 P0 标注语料，不是独立盲标；VLM 同图盲标对比不可用它",
        "n": len(items), "items": items,
    }
    print("重建 gold：{} 张 / {} 框（split_report 期望 {} 框）；缺源 {} 项".format(
        len(items), n_boxes, expected, len(missing)))
    for miss in missing[:5]:
        print(f"  缺: {miss}")
    if dry_run:
        print("[DRY-RUN] 不落盘")
        return gold, ann
    os.makedirs(out_dir, exist_ok=True)
    gp = os.path.join(out_dir, "gold_set.json")
    ap = os.path.join(out_dir, "gold_ann.json")
    if os.path.exists(gp) and not overwrite:
        raise ValueError("gold_set.json 已存在：{}（原 blind gold 不能被无声覆盖；"
                         "确认要重建请加 --force）".format(gp))
    with open(gp, "w", encoding="utf-8") as f:
        json.dump(gold, f, ensure_ascii=False, indent=2)
    with open(ap, "w", encoding="utf-8") as f:
        json.dump({"note": gold["provenance_warning"], "items": ann}, f,
                  ensure_ascii=False, indent=2)
    print("gold set → {}\n人工框 → {}".format(gp, ap))
    return gold, ann


def load_ann_json(path):
    """`gold_ann.json`（`{norm_path: [{label, box}]}` 或 `{items: {...}}`）→ ann_by_path。"""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    items = data.get("items", data) if isinstance(data, dict) else data
    out = {}
    if isinstance(items, dict):
        for k, v in items.items():
            boxes = v.get("boxes", v.get("objects", v)) if isinstance(v, dict) else v
            out[norm_path(canon_path(k))] = [
                {"label": b["label"], "box": [float(x) for x in b.get("box", b.get("tight_bbox"))]}
                for b in boxes
            ]
    else:
        for it in items:
            p = canon_path(it.get("image_path", ""))
            boxes = it.get("boxes", it.get("objects")) or []
            out[norm_path(p)] = [
                {"label": b["label"], "box": [float(x) for x in b.get("box", b.get("tight_bbox"))]}
                for b in boxes
            ]
    return out


def load_pred_dir(path):
    """YOLO 预测目录/单文件 → `{norm_path: PrelabelRecord}`。

    每份预测是一个 JSON：`{"image_path", "original_size", "objects":[{label,tight_bbox,
    score,...}]}`（`contracts.DetBox` 字段）；也接受单文件列表、单条记录或
    `{image_path: {original_size, objects}}` 映射。缺 `original_size` 时现读原图。
    """
    files = ([path] if os.path.isfile(path) else
             sorted(os.path.join(path, f) for f in os.listdir(path) if f.endswith(".json")))
    idx = {}
    for fp in files:
        with open(fp, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            recs = data
        elif isinstance(data, dict) and ("objects" in data or "image_path" in data):
            recs = [data]
        elif isinstance(data, dict):
            recs = [{"image_path": k, **({"objects": v} if isinstance(v, list) else v)}
                    for k, v in data.items()]
        else:
            continue
        for r in recs:
            pre = PrelabelRecord.from_dict(r)
            p = canon_path(pre.image_path)
            if not p:
                continue
            if not pre.original_size:
                img = imread_cn(p)
                if img is None:
                    print(f"[warn] 预测缺 original_size 且读图失败：{p}")
                    continue
                pre.original_size = [img.shape[1], img.shape[0]]
            pre.image_path = p
            idx[norm_path(p)] = pre
    return idx


def yolo_predictions(gold, weights=None, config_path=None):
    """用 `detector.json` 的权重/阈值对 gold 图逐张跑 YOLO → `{norm_path: PrelabelRecord}`。"""
    settings = detector_yolo.load_settings(load_config(), config_path, weights)
    idx = {}
    print("YOLO 推理：{} 张（imgsz={} conf={} iou={} tile={}）".format(
        len(gold["items"]), settings["imgsz"], settings["conf"], settings["iou"],
        settings["tile"]))
    for i, it in enumerate(gold["items"], 1):
        p = canon_path(it["image_path"])
        img = imread_cn(p)
        if img is None:
            print(f"  [{i}/{len(gold['items'])}] FAIL 读图：{p}")
            continue
        dets = detector_yolo.detect(
            img, weights=settings["weights"], imgsz=settings["imgsz"],
            conf=settings["conf"], iou=settings["iou"], device=settings["device"],
            max_det=settings["max_det"], tile=settings["tile"],
            overlap=settings["overlap"], classes=settings["classes"])
        idx[norm_path(p)] = PrelabelRecord.from_dict({
            "image_path": p, "original_size": [img.shape[1], img.shape[0]],
            "objects": dets, "prompt_version": "yolo",
            "model": os.path.basename(settings["weights"]),
        })
        if i % 10 == 0 or i == len(gold["items"]):
            print(f"  [{i}/{len(gold['items'])}] {os.path.basename(p)} → {len(dets)} 框")
    return idx, settings


# ---------------------------------------------------------------- ①d 置信度校准（P3 §6.5 第 4 条）
CALIBRATION_MIN_N = 20          # 少于此样本量的类不拟合，T=1（避免过拟合）
CALIBRATION_T_GRID = tuple(round(0.25 + 0.05 * i, 2) for i in range(196))  # 0.25..10.0


def _clip_score(s):
    return min(max(float(s), 1e-6), 1.0 - 1e-6)


def _logit(p):
    p = _clip_score(p)
    return math.log(p / (1.0 - p))


def _sigmoid(z):
    if z >= 0:
        e = math.exp(-z)
        return 1.0 / (1.0 + e)
    e = math.exp(z)
    return e / (1.0 + e)


def _ece_from_obs(obs, temperature=1.0):
    """逐类 ECE（10 个等宽置信桶）；`temperature>1` 时先把 score 做温度缩放。"""
    if not obs:
        return None
    n = len(obs)
    bins = defaultdict(lambda: [0, 0.0, 0])
    for s, y in obs:
        p = _sigmoid(_logit(s) / temperature) if temperature != 1.0 else _clip_score(s)
        b = min(9, max(0, int(p * 10)))
        bins[b][0] += 1
        bins[b][1] += p
        bins[b][2] += y
    return round(sum(cnt / n * abs(correct / cnt - psum / cnt)
                     for cnt, psum, correct in bins.values()), 4)


def fit_temperature(obs):
    """按类拟合温度 T（最小化 BCE）；样本不足 `CALIBRATION_MIN_N` 返回 1.0。"""
    if len(obs) < CALIBRATION_MIN_N:
        return 1.0
    best_t, best_loss = 1.0, None
    for t in CALIBRATION_T_GRID:
        loss = 0.0
        for s, y in obs:
            p = _sigmoid(_logit(s) / t)
            p = min(max(p, 1e-12), 1.0 - 1e-12)
            loss -= y * math.log(p) + (1 - y) * math.log(1 - p)
        loss /= len(obs)
        if best_loss is None or loss < best_loss:
            best_t, best_loss = t, loss
    return best_t


def read_multi_labels(path, W, H, classes):
    """多类 YOLO 标签 → `[{"label": 中文, "box": 像素}]`（class id 按训练顺序）。"""
    out = []
    if not os.path.isfile(path):
        return out
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 5:
                continue
            cid = int(float(parts[0]))
            if cid < 0 or cid >= len(classes):
                continue
            label = classes[cid]["zh"]
            cx, cy, w, h = [float(v) for v in parts[1:5]]
            box = [(cx - w / 2) * W, (cy - h / 2) * H,
                   (cx + w / 2) * W, (cy + h / 2) * H]
            box = [min(max(v, 0.0), float(W if i % 2 == 0 else H))
                   for i, v in enumerate(box)]
            if box[2] - box[0] < 1 or box[3] - box[1] < 1:
                continue
            out.append({"label": label, "box": [round(v, 1) for v in box]})
    return out


def collect_calibration_obs(settings, split="val", limit=0, iou_thr=IOU_THR):
    """在 `dataset/检测训练集_v1/<split>` 上跑 YOLO，收集逐预测类别的 (score, 是否正确)。

    正确 = 与同类 GT 在 IoU≥`iou_thr` 匹配（每 GT 只用一次）。**这是校准集，不是评测集**；
    val 同时参与早停选模，故校准有轻微乐观偏差，报告里要写清。
    """
    root = os.path.join(PROJECT_ROOT, "dataset", "检测训练集_v1")
    img_dir, lbl_dir = os.path.join(root, "images", split), os.path.join(root, "labels", split)
    files = sorted(f for f in os.listdir(img_dir)
                   if f.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")))
    if limit:
        files = files[:limit]
    obs = defaultdict(list)
    print("校准观测：{} 张 {} 图（imgsz={} conf={}）".format(
        len(files), split, settings["imgsz"], settings["conf"]))
    for i, fn in enumerate(files, 1):
        p = os.path.join(img_dir, fn)
        img = imread_cn(p)
        if img is None:
            continue
        H, W = img.shape[:2]
        gt = read_multi_labels(os.path.join(lbl_dir, os.path.splitext(fn)[0] + ".txt"),
                               W, H, settings["classes"])
        dets = detector_yolo.detect(
            img, weights=settings["weights"], imgsz=settings["imgsz"],
            conf=settings["conf"], iou=settings["iou"], device=settings["device"],
            max_det=settings["max_det"], tile=settings["tile"],
            overlap=settings["overlap"], classes=settings["classes"])
        gt_by_label = defaultdict(list)
        for g in gt:
            gt_by_label[g["label"]].append(g["box"])
        used = {k: [False] * len(v) for k, v in gt_by_label.items()}
        for d in sorted(dets, key=lambda x: -float(x.get("score") or 0.0)):
            best, bi = 0.0, None
            for j, gb in enumerate(gt_by_label.get(d["label"], [])):
                if used[d["label"]][j]:
                    continue
                v = iou_box(d["tight_bbox"], gb)
                if v >= iou_thr and v > best:
                    best, bi = v, j
            if bi is not None:
                used[d["label"]][bi] = True
            obs[d["label"]].append((float(d.get("score") or 0.0), 1 if bi is not None else 0))
        if i % 100 == 0 or i == len(files):
            print("  [{}/{}] {} → 累计 {} 类观测".format(i, len(files), fn, len(obs)))
    return dict(obs), len(files)


def build_calibration(obs, split, n_images, iou_thr=IOU_THR):
    classes = {}
    for cls, o in sorted(obs.items()):
        t = fit_temperature(o)
        classes[cls] = {
            "T": round(t, 4), "n": len(o),
            "ece_before": _ece_from_obs(o, 1.0),
            "ece_after": _ece_from_obs(o, t),
            "mean_score": round(sum(s for s, _ in o) / len(o), 4) if o else None,
        }
    return {"split": split, "n_images": n_images, "iou": iou_thr,
            "min_n": CALIBRATION_MIN_N,
            "note": "温度缩放按类拟合 BCE；val 同时参与早停选模，校准有轻微乐观偏差",
            "classes": classes}


def apply_temperature(pred_index, calibration):
    """把逐类温度 T 应用到预测 score（原地改记录；只给报告/诊断，不改框）。"""
    table = (calibration or {}).get("classes") or {}
    for rec in pred_index.values():
        for b in rec.objects:
            t = float((table.get(b.label) or {}).get("T", 1.0))
            if b.score is not None and t != 1.0:
                b.score = round(_sigmoid(_logit(b.score) / t), 4)


# ---------------------------------------------------------------- ①c AP / P3 汇总
def _ap_from_arrays(tp, fp, n_gt):
    """单类单 IoU 阈值的 101 点插值 AP（COCO 口径；纯函数，便于自检）。"""
    if n_gt <= 0:
        return None
    prec, rec = [], []
    t = f = 0
    for a, b in zip(tp, fp):
        t += a
        f += b
        prec.append(t / (t + f) if (t + f) else 0.0)
        rec.append(t / n_gt)
    for i in range(len(prec) - 1, 0, -1):
        prec[i - 1] = max(prec[i - 1], prec[i])
    total = 0.0
    for k in range(101):
        r = k / 100.0
        p = 0.0
        for i, ri in enumerate(rec):
            if ri >= r:
                p = prec[i]
                break
        total += p
    return total / 101.0


def average_precision(gold, pred_index, ann_by_path, iou_thrs=AP_IOU_THRS):
    """逐类 AP50 / AP50-95（预测按 score 降序贪心匹配同类 GT；无 GT 的类返回 None）。"""
    by_cls = defaultdict(lambda: {"gt": [], "pred": []})
    for it in gold["items"]:
        p = norm_path(it["image_path"])
        rec = pred_index.get(p)
        gt = ann_by_path.get(p)
        if rec is None or gt is None:
            continue
        for o in rec.objects:
            if o.label and o.tight_bbox:
                by_cls[o.label]["pred"].append(
                    (float(o.score if o.score is not None else 0.0), p,
                     [float(v) for v in o.tight_bbox]))
        for a in gt:
            by_cls[a["label"]]["gt"].append((p, [float(v) for v in a["box"]]))
    out = {}
    for cls, d in sorted(by_cls.items()):
        n_gt, preds = len(d["gt"]), d["pred"]
        per_thr = []
        for thr in iou_thrs:
            gt_by_img = defaultdict(list)
            for p, box in d["gt"]:
                gt_by_img[p].append(box)
            used = {p: [False] * len(v) for p, v in gt_by_img.items()}
            tp, fp = [], []
            for score, p, box in sorted(preds, key=lambda x: -x[0]):
                best, bi = 0.0, None
                for j, gb in enumerate(gt_by_img.get(p, [])):
                    if used[p][j]:
                        continue
                    v = iou_box(box, gb)
                    if v >= thr and v > best:
                        best, bi = v, j
                if bi is not None:
                    used[p][bi] = True
                    tp.append(1)
                    fp.append(0)
                else:
                    tp.append(0)
                    fp.append(1)
            per_thr.append(_ap_from_arrays(tp, fp, n_gt))
        vals = [v for v in per_thr if v is not None]
        out[cls] = {
            "n_gt": n_gt, "n_pred": len(preds),
            "ap50": round(per_thr[0], 4) if per_thr[0] is not None else None,
            "ap50_95": round(sum(vals) / len(vals), 4) if vals else None,
        }
    return out


def _score_stats(score_obs):
    """逐类原始 score 的均值 / ECE / Brier（10 个等宽置信桶；仅诊断，不做校准）。"""
    out = {}
    for cls, obs in sorted(score_obs.items()):
        if not obs:
            continue
        n = len(obs)
        mean = sum(s for s, _ in obs) / n
        bins = defaultdict(lambda: [0, 0.0, 0])
        for s, c in obs:
            b = min(9, max(0, int(s * 10)))
            bins[b][0] += 1
            bins[b][1] += s
            bins[b][2] += c
        ece = sum(cnt / n * abs(correct / cnt - ssum / cnt)
                  for cnt, ssum, correct in bins.values())
        brier = sum((s - c) ** 2 for s, c in obs) / n
        out[cls] = {"n": n, "mean_score": round(mean, 4),
                    "ece": round(ece, 4), "brier": round(brier, 4)}
    return out


def p3_extras(gold, pred_index, ann_by_path, main_rep, main_exclude):
    """P3 报告增量：AP、混淆矩阵/易混对、P0 六类门槛、其他类单独、跨药材子集。"""
    labels = sorted({a["label"] for v in ann_by_path.values() for a in v} |
                    {o.label for rec in pred_index.values() for o in rec.objects})
    ap = average_precision(gold, pred_index, ann_by_path)
    full = evaluate(gold["items"], pred_index, ann_by_path, exclude=())
    matrix = {a: {b: 0 for b in labels} for a in labels}
    for key, n in (full.get("label_confusion") or {}).items():
        a, b = key.split("→", 1)
        matrix.setdefault(a, {x: 0 for x in labels})[b] = n
    conf = {}
    for a, b in CONFUSABLE_PAIRS:
        conf[f"{a}↔{b}"] = {
            f"{a}→{b}": int((full.get("label_confusion") or {}).get(f"{a}→{b}", 0)),
            f"{b}→{a}": int((full.get("label_confusion") or {}).get(f"{b}→{a}", 0)),
        }
    other = None
    if OTHER_LABEL in labels:
        only_other = set(labels) - {OTHER_LABEL}
        r = evaluate(gold["items"], pred_index, ann_by_path, exclude=only_other)
        other = {"n_human": r["n_human"], "n_pred": r["n_pred"],
                 "recall_e2e": r["recall_e2e"], "precision": r["precision"],
                 "per_class": r["per_class"].get(OTHER_LABEL), "note": "不进主门槛"}
    cross_items = [it for it in gold["items"] if it.get("cross_herb")]
    cross = None
    if cross_items:
        r = evaluate(cross_items, pred_index, ann_by_path, exclude=main_exclude)
        cross = {"n_gold": len(cross_items), "n_human": r["n_human"], "n_pred": r["n_pred"],
                 "recall_e2e": r["recall_e2e"], "precision": r["precision"]}
    else:
        cross = {"n_gold": 0, "note": "gold 中无 cross_herb 记录"}
    p0 = []
    for cls in P0_CLASSES:
        v = main_rep["per_class"].get(cls) or {}
        rec, prec = v.get("recall_e2e"), v.get("precision")
        good = (rec is not None and prec is not None
                and rec >= P0_CLASS_GATE and prec >= P0_CLASS_GATE)
        p0.append({
            "class": cls, "recall_e2e": rec, "precision": prec,
            "n_human": v.get("n_human", 0), "n_pred": v.get("n_pred", 0),
            "pass": bool(good),
            "line": f"P0 {cls}  召回 {rec if rec is not None else '无数据'}"
                    f" / 精确 {prec if prec is not None else '无数据'}"
                    f" {'✔' if good else '✘'}（线 ≥{P0_CLASS_GATE}/{P0_CLASS_GATE}）",
        })
    return {
        "ap": ap,
        "confusion_matrix": matrix,
        "confusable_pairs": conf,
        "p0_gates": p0,
        "other": other,
        "cross_herb": cross,
        "main_excludes_other": True,
        "calibration": {
            "status": "raw_score",
            "note": "YOLO score 未校准；下表是逐类原始 score 的 ECE/Brier 诊断，"
                    "温度/等距校准需另用 val 集拟合后再接入判药证据",
            "per_class": full.get("score_stats"),
        },
        "limitations": [
            "VLM 臂未在本机重跑（无 prelabels/VLM_API_KEY/盲标 LS）；参考臂用同一 60 张 gold 的"
            "既有 2026-09-24 VLM 报告（0.661/0.312），见 reference",
            "「其他」类主指标剔除、单独报告；hybrid 仍是方案 §6.4 的可选实验档，未接线",
            "人工框来自 P0 标注语料（gold_eval 预留图），不是原 blind gold 项目",
            "三条权重臂是不同模型/增强，不是同模型多种子；单 seed 的均值/极差纪律未满足。"
            "但本结论是绝对门槛（VLM 与 YOLO 都未过），不作处理效应显著性声明",
            "YOLO 未过检测门槛，未进入下游证据；温度/等距校准按 §6.5 第 4 条待模型过线后再做",
            "AP 基于 detector.json conf 阈值的预测；低分真阳被截断，"
            "与 P1 test eval_report（Ultralytics val 的 conf≈0.001）不可直接横比",
        ],
    }


# ---------------------------------------------------------------- ② 评测核心
def evaluate(items, pre_index, ann_by_path, iou_thr=IOU_THR, exclude=()):
    """纯函数：gold 项 × prelabel × 人工标注 → 指标字典（不落盘、不联网，便于自检）。

    exclude：要剔除的类别名集合。**两侧同时剔除**——只在"标注单元口径一致"的类上
    才能测模型质量；若某类人工按"整团一框"、模型按"逐颗粒"（实测淀粉粒 71 vs 14），
    其召回/精确反映的是口径冲突而非模型误差，需用本参数分开看。
    """
    pairs = []          # (人工框, 预测框, IoU)
    missed = []         # 人工框无对应预测 → 漏检
    extra = []          # 预测框无对应对应人工 → 误检/无关框
    pending = []        # 未标注的 gold 项
    per_cls = defaultdict(lambda: {"n_human": 0, "n_pred": 0, "hit_loc": 0,
                                   "hit_e2e": 0, "n_match": 0})
    n_human = n_pred = 0
    box_err = []
    clip_pred = clip_human = big_pred = 0
    clf_conf = Counter()   # 匹配对上的类别混淆（人工 → 预测）
    score_obs = defaultdict(list)   # 预测类别的 (score, 是否正确)，用于置信度诊断

    for it in items:
        p = norm_path(it["image_path"])
        pre = pre_index.get(p)
        ann = ann_by_path.get(p)
        if pre is None:
            pending.append((it["image_path"], "无 prelabel"))
            continue
        if ann is None:
            pending.append((it["image_path"], "未标注"))
            continue
        W, H = pre.original_size
        IA = float(W) * H
        P = [{"label": b.label, "box": [float(v) for v in b.tight_bbox],
              "score": float(b.score) if b.score is not None else 0.0}
             for b in pre.objects if b.label not in exclude]
        ann = [a for a in ann if a["label"] not in exclude]
        n_pred += len(P)
        n_human += len(ann)
        box_err.append(abs(len(P) - len(ann)))
        for pb in P:
            x1, y1, x2, y2 = pb["box"]
            at = (x1 <= 0.5 or y1 <= 0.5 or x2 >= W - 0.5 or y2 >= H - 0.5)
            if at:
                clip_pred += 1
                pb["clipped"] = True
            if (x2 - x1) * (y2 - y1) / IA > BIG_BOX_RATIO:
                big_pred += 1
            per_cls[pb["label"]]["n_pred"] += 1
        for a in ann:
            per_cls[a["label"]]["n_human"] += 1
            x1, y1, x2, y2 = a["box"]
            if x1 <= 0.5 or y1 <= 0.5 or x2 >= W - 0.5 or y2 >= H - 0.5:
                clip_human += 1
                a["clipped"] = True

        used = set()
        for a in ann:
            best, bi = 0.0, None
            for j, pb in enumerate(P):
                if j in used:
                    continue
                v = iou_box(a["box"], pb["box"])
                if v > best:
                    best, bi = v, j
            if best >= iou_thr and bi is not None:
                used.add(bi)
                pb = P[bi]
                per_cls[a["label"]]["hit_loc"] += 1
                per_cls[pb["label"]]["n_match"] += 1
                correct = a["label"] == pb["label"]
                if correct:
                    per_cls[a["label"]]["hit_e2e"] += 1
                else:
                    clf_conf[(a["label"], pb["label"])] += 1
                score_obs[pb["label"]].append((pb["score"], 1 if correct else 0))
                pairs.append({"path": it["image_path"], "a": a, "p": pb, "iou": best,
                              "clipped": bool(a.get("clipped") or pb.get("clipped"))})
            else:
                missed.append({"path": it["image_path"], "label": a["label"], "best_iou": best})
        for j, pb in enumerate(P):
            if j not in used:
                x1, y1, x2, y2 = pb["box"]
                extra.append({"path": it["image_path"], "label": pb["label"],
                              "score": pb["score"],
                              "area_ratio": (x2 - x1) * (y2 - y1) / IA})
                score_obs[pb["label"]].append((pb["score"], 0))

    # ---- 聚合
    n_match = len(pairs)
    hit_e2e = sum(1 for pr in pairs if pr["a"]["label"] == pr["p"]["label"])
    ious = sorted(pr["iou"] for pr in pairs)
    strict = sum(1 for v in ious if v >= STRICT_IOU)

    def pct(v):
        if not v:
            return None
        return {"mean": round(sum(v) / len(v), 4), "median": round(_median(v), 4),
                "p25": round(v[len(v) // 4], 4), "p75": round(v[len(v) * 3 // 4], 4),
                "min": round(v[0], 4), "max": round(v[-1], 4)}

    def _ratio(pr):
        aw = pr["a"]["box"][2] - pr["a"]["box"][0]
        ah = pr["a"]["box"][3] - pr["a"]["box"][1]
        pw = pr["p"]["box"][2] - pr["p"]["box"][0]
        ph = pr["p"]["box"][3] - pr["p"]["box"][1]
        return {"area_ratio": (aw * ah) / max(1e-6, pw * ph),
                "dw_rel": (aw - pw) / max(1e-6, pw), "dh_rel": (ah - ph) / max(1e-6, ph)}

    cov = [_ratio(pr) for pr in pairs]
    cov_clean = [_ratio(pr) for pr in pairs if not pr["clipped"]]
    cov_diff = [c for c in cov if abs(c["area_ratio"] - 1.0) > COVERAGE_TOL]
    med = lambda v: round(_median(v), 4) if v else None
    cls = {}
    for k, v in per_cls.items():
        cls[k] = {
            "n_human": v["n_human"], "n_pred": v["n_pred"], "n_match": v["n_match"],
            "recall_loc": round(v["hit_loc"] / v["n_human"], 4) if v["n_human"] else None,
            "recall_e2e": round(v["hit_e2e"] / v["n_human"], 4) if v["n_human"] else None,
            "precision": round(v["hit_e2e"] / v["n_pred"], 4) if v["n_pred"] else None,
        }
    # 框数比异常 → 疑似"标注单元口径不一致"（人工整团一框 vs 模型逐颗粒）。
    # 实测 2026-09-24 淀粉粒 71:14 即属此列，其精确/召回不能当作模型误差读。
    unit_mis = []
    for k, v in cls.items():
        if v["n_human"] >= 3 and v["n_pred"] >= 2 * v["n_human"]:
            unit_mis.append({"class": k, "n_pred": v["n_pred"], "n_human": v["n_human"],
                             "ratio": round(v["n_pred"] / v["n_human"], 2),
                             "hint": "预测远多于人工：或过检，或人工按整团一框"})
        elif v["n_pred"] >= 3 and v["n_human"] >= 2 * v["n_pred"]:
            unit_mis.append({"class": k, "n_pred": v["n_pred"], "n_human": v["n_human"],
                             "ratio": round(v["n_pred"] / v["n_human"], 2),
                             "hint": "人工远多于预测：或漏检，或模型按整团一框"})
    # 漏检分档：best_iou=0 是"完全没框到"，0<iou<thr 是"框了但位置偏"（定位误差）
    miss_offset = sum(1 for m in missed if m["best_iou"] > 0)
    return {
        "iou_thr": iou_thr,
        "n_gold": len(items), "n_scored": len(items) - len(pending),
        "n_human": n_human, "n_pred": n_pred, "n_match": n_match,
        "recall_loc": round(n_match / n_human, 4) if n_human else None,
        "recall_e2e": round(hit_e2e / n_human, 4) if n_human else None,
        "precision": round(hit_e2e / n_pred, 4) if n_pred else None,
        "missed_offset": miss_offset, "missed_no_box": len(missed) - miss_offset,
        "unit_mismatch": unit_mis,
        "strict_iou_hits": strict,
        "strict_recall_loc": round(strict / n_human, 4) if n_human else None,
        "iou": pct(ious),
        "coverage": {"n": len(cov), "n_clean": len(cov_clean),
                     "area_ratio_median": med([c["area_ratio"] for c in cov]),
                     "area_ratio_median_clean": med([c["area_ratio"] for c in cov_clean]),
                     "n_diff": len(cov_diff),
                     "area_ratio_median_diff": med([c["area_ratio"] for c in cov_diff]),
                     "dw_rel_median": med([c["dw_rel"] for c in cov]),
                     "dh_rel_median": med([c["dh_rel"] for c in cov]),
                     "n_human_larger": sum(1 for c in cov if c["area_ratio"] > 1 + COVERAGE_TOL),
                     "n_pred_larger": sum(1 for c in cov if c["area_ratio"] < 1 - COVERAGE_TOL),
                     "n_equal": sum(1 for c in cov if abs(c["area_ratio"] - 1) <= COVERAGE_TOL)},
        "clip": {"pred": clip_pred, "human": clip_human,
                 "pred_rate": round(clip_pred / n_pred, 4) if n_pred else None,
                 "pairs_affected": sum(1 for pr in pairs if pr["clipped"])},
        "big_pred_boxes": big_pred,
        "box_count_err": {"mean_abs": round(sum(box_err) / len(box_err), 3) if box_err else None,
                          "total_abs": sum(box_err)},
        # 置信度诊断（不是校准）：按预测类别统计原始 score 的 ECE/Brier。
        "score_stats": _score_stats(score_obs),
        "per_class": dict(sorted(cls.items())),
        "label_confusion": {f"{a}→{b}": c for (a, b), c in clf_conf.most_common()},
        "missed": sorted(missed, key=lambda x: -x["best_iou"]),
        "extra": sorted(extra, key=lambda x: -x["area_ratio"]),
        "pending": pending,
    }


GATES = [
    ("端到端召回", "recall_e2e", GATE_RECALL_E2E, ">="),
    ("精确", "precision", GATE_PRECISION, ">="),
    ("匹配 IoU 中位", "iou.median", GATE_MEDIAN_IOU, ">="),
]


def verdict_of(rep):
    """验收判定：返回 (是否全过, 逐条文本)。覆盖率偏差用双侧窗口判。"""
    lines, ok_all = [], True
    for name, key, gate, op in GATES:
        v = rep
        for part in key.split("."):
            v = (v or {}).get(part) if isinstance(v, dict) else None
        if v is None:
            lines.append(f"  {name:<14} 无数据")
            ok_all = False
            continue
        good = v >= gate
        ok_all &= good
        lines.append(f"  {name:<14} {v:.3f}  {'✔' if good else '✘'}（线 {op}{gate}）")
    cm = rep["coverage"]["area_ratio_median_clean"]
    if cm is None:
        lines.append("  覆盖率偏差中位   无数据")
        ok_all = False
    else:
        good = abs(cm - 1.0) <= GATE_COVERAGE_DEV
        ok_all &= good
        biased = "框偏小（测量偏低）" if cm > 1.0 else "框偏大"
        lines.append(f"  覆盖率偏差中位   {cm:.3f}  {'✔' if good else '✘'}"
                     f"（线 |x-1|≤{GATE_COVERAGE_DEV}；{'' if good else biased}）")
    # P3：P0 六类逐类门槛（仅 P3 模式有 p0_gates；其他类不进主门槛）。
    for g in rep.get("p0_gates") or []:
        ok_all &= bool(g.get("pass"))
        lines.append("  " + g.get("line", g.get("class", "P0")))
    return ok_all, lines


# ---------------------------------------------------------------- ② 标注源
def report(gold, tasks, cfg_note="", out_dir=EVAL_DIR, exclude=(), p3=False):
    """VLM 路径：prelabels 作预测 + LS 任务/导出 JSON 作人工标注。"""
    pre_index = load_prelabel_index()
    ann, unmatched, blank = ann_from_tasks(tasks, gold, pre_index)
    pv = Counter((pre_index[norm_path(it["image_path"])].prompt_version or "?")
                 for it in gold["items"] if norm_path(it["image_path"]) in pre_index)
    return _finalize(gold, pre_index, ann, cfg_note, out_dir, exclude, {
        "annotated": len(ann), "blank": blank, "unmatched": unmatched,
        "prompt_versions": dict(pv), "engine": "vlm", "p3": p3,
    })


def report_pred(gold, pred_index, ann, cfg_note="", out_dir=EVAL_DIR, exclude=(),
                meta=None):
    """已拿到预测索引（YOLO 现跑 / --pred-dir）时的统一报告出口。"""
    return _finalize(gold, pred_index, ann, cfg_note, out_dir, exclude, meta or {})


def _project_rel(path):
    """报告里的路径统一写项目相对路径；项目外路径保持原样（不产生 `..`）。"""
    p = os.path.normpath(str(path))
    if not os.path.isabs(p):
        return p.replace(os.sep, "/")
    try:
        rel = os.path.relpath(p, PROJECT_ROOT)
    except ValueError:          # Windows 跨盘符
        return p.replace(os.sep, "/")
    return p.replace(os.sep, "/") if rel.startswith("..") else rel.replace(os.sep, "/")


def _finalize(gold, pred_index, ann, cfg_note, out_dir, exclude, meta):
    rep = evaluate(gold["items"], pred_index, ann, exclude=exclude)
    # 报告是给人看/入库的产物：明细里的路径改记项目相对路径（匹配仍在内存用绝对路径）。
    for key in ("missed", "extra"):
        for item in rep[key]:
            item["path"] = _project_rel(item["path"])
    rep["pending"] = [(_project_rel(p), why) for p, why in rep["pending"]]
    det = rep.get("detector") or {}
    for key in ("weights", "path"):
        if det.get(key):
            det[key] = _project_rel(det[key])
    rep["excluded_classes"] = sorted(exclude)
    rep["annotated"] = meta.get("annotated", len(ann))
    rep["blank_annotations"] = meta.get("blank", 0)
    rep["unmatched_tasks"] = meta.get("unmatched", [])
    rep["prompt_versions"] = meta.get("prompt_versions", {})
    rep["engine"] = meta.get("engine", "vlm")
    rep["detector"] = meta.get("detector", {})
    rep["reference"] = meta.get("reference")
    rep["note"] = cfg_note
    if meta.get("p3"):
        # 主表已剔除「其他」；P3 增量把 AP/混淆矩阵/其他类/跨药材/P0 门槛补齐。
        rep.update(p3_extras(gold, pred_index, ann, rep, exclude))
    cal = meta.get("calibration")
    if cal:
        rep.setdefault("calibration", {})
        rep["calibration"].update({
            "status": "temperature_scaled",
            "note": "已按类温度缩放（{} 拟合；val 同时参与早停，校准有轻微乐观偏差）；"
                    "下表 per_class 是 gold 上校准后的 ECE/Brier".format(cal.get("split", "?")),
            "fit_split": cal.get("split"),
            "temperature": cal.get("classes"),
        })
        rep["limitations"] = [x for x in rep.get("limitations", [])
                              if not x.startswith("YOLO 未过检测门槛")]
        rep["limitations"].append(
            "已做按类温度缩放（val 拟合，见 temperature）；模型未过检测门槛，校准仅作诊断、未接入判药")
    ok, gate_lines = verdict_of(rep)
    rep["pass"] = ok
    rep["decision"] = {
        "switch_default_engine": bool(ok),
        "basis": "P3 主门槛（recall_e2e/precision/median IoU/coverage）+ P0 六类门槛",
    }
    rep["gate_lines"] = [l.strip() for l in gate_lines]

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "report.json"), "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)

    # ---- 控制台摘要
    print("=" * 68)
    print(f"gold {rep['n_gold']} 张 | 已标注 {rep['annotated']}"
          f"（空标注 {rep['blank_annotations']}）| 可评 {rep['n_scored']} 张 | "
          f"未匹配任务 {len(rep['unmatched_tasks'])}")
    if rep["excluded_classes"]:
        print(f"已剔除类别（不参与指标）：{rep['excluded_classes']}")
    print(f"引擎 {rep['engine']} | prelabel/预测版本 {rep['prompt_versions']}"
          f" | IoU 阈值 {rep['iou_thr']}")
    print(f"人工框 {rep['n_human']} | 预测框 {rep['n_pred']} | 匹配 {rep['n_match']}")
    print(f"定位召回 {rep['recall_loc']} | 端到端召回 {rep['recall_e2e']} | 精确 {rep['precision']}")
    print(f"漏检分档：完全无框 {rep['missed_no_box']} | 框了但位置偏(IoU<{rep['iou_thr']}) "
          f"{rep['missed_offset']}")
    if rep["iou"]:
        print(f"IoU 中位 {rep['iou']['median']}（P25 {rep['iou']['p25']} / 最小 {rep['iou']['min']}）"
              f" | 严格档 0.75 命中 {rep['strict_iou_hits']}（召回 {rep['strict_recall_loc']}）")
    cv = rep["coverage"]
    print(f"覆盖率偏差 中位 {cv['area_ratio_median']}（剔除贴边 {cv['n_clean']} 对后 "
          f"{cv['area_ratio_median_clean']}）")
    print(f"  人工更大 {cv['n_human_larger']} / 预测更大 {cv['n_pred_larger']} / "
          f"基本相等 {cv['n_equal']}（容差 {COVERAGE_TOL:.0%}）；"
          f"仅看在 {cv['n_diff']} 对确有差异的：中位 {cv['area_ratio_median_diff']}"
          f"（>1 = 框小了）")
    print(f"贴边率 预测 {rep['clip']['pred_rate']}（{rep['clip']['pred']} 框）"
          f" | 巨型框(>50%画面) {rep['big_pred_boxes']} | 框数误差 {rep['box_count_err']}")
    print("\n逐类:")
    print(f"  {'类别':<12}{'人工':>5}{'预测':>5}{'匹配':>5}{'定位召回':>9}{'端到端召回':>11}{'精确':>7}")
    for k, v in rep["per_class"].items():
        f = lambda x: "  -  " if x is None else f"{x:.2f}"
        print(f"  {k:<12}{v['n_human']:>5}{v['n_pred']:>5}{v['n_match']:>5}"
              f"{f(v['recall_loc']):>9}{f(v['recall_e2e']):>11}{f(v['precision']):>7}")
    if rep["label_confusion"]:
        print(f"\n类别混淆（人工→预测）: {rep['label_confusion']}")
    if rep["unit_mismatch"]:
        print("\n⚠️ 框数比异常（疑似标注单元口径不一致，其精确/召回不可当模型误差读）:")
        for u in rep["unit_mismatch"]:
            print(f"  {u['class']}: 预测 {u['n_pred']} / 人工 {u['n_human']} = {u['ratio']}"
                  f"  —— {u['hint']}")
    if rep.get("ap"):
        print("\n逐类 AP（含「其他」）:")
        for k, v in rep["ap"].items():
            f = lambda x: "  -  " if x is None else f"{x:.3f}"
            print(f"  {k:<12} GT {v['n_gt']:>3} / Pred {v['n_pred']:>3}"
                  f" | AP50 {f(v['ap50'])} | AP50-95 {f(v['ap50_95'])}")
    if rep.get("other"):
        print(f"\n其他类（不进主门槛）: {rep['other']}")
    if rep.get("cross_herb"):
        print(f"跨药材子集: {rep['cross_herb']}")
    print("\n验收判定:", "PASS ✔" if ok else "FAIL ✘")
    print("默认引擎决策:", "可切换" if rep["decision"]["switch_default_engine"] else "不切换")
    for l in gate_lines:
        print(l)
    print(f"\n漏检（人工有框、预测无）{len(rep['missed'])} 例 | "
          f"多出框（预测无对应）{len(rep['extra'])} 例")
    for m in rep["missed"][:8]:
        print(f"  [漏] {os.path.basename(m['path'])} {m['label']} 最佳IoU={m['best_iou']:.3f}")
    for e in rep["extra"][:8]:
        print(f"  [多] {os.path.basename(e['path'])} {e['label']} 面积比={e['area_ratio']*100:.2f}%")
    print(f"\n报告 → {os.path.join(out_dir, 'report.json')}")
    return rep


def write_md(rep, out_dir=EVAL_DIR):
    """人读版报告（每次改 prompt/换模型后重跑，用同一批图对比同一组数字）。"""
    L = []
    L.append("# 检测评测报告（gold set）\n")
    L.append(f"- 生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}")
    L.append(f"- gold 图数：{rep['n_gold']}｜已标注 {rep['annotated']}（空标注 {rep['blank_annotations']}）"
             f"｜可评 {rep['n_scored']}｜未匹配任务 {len(rep['unmatched_tasks'])}")
    L.append(f"- 引擎：{rep.get('engine', 'vlm')}｜detector：{rep.get('detector', {})}")
    L.append(f"- 预测版本：{rep['prompt_versions']}｜IoU 阈值 {rep['iou_thr']}")
    if rep.get("note"):
        L.append(f"- 口径说明：{rep['note']}")
    if rep["excluded_classes"]:
        L.append(f"- 已剔除类别（不参与指标）：{'、'.join(rep['excluded_classes'])}")
    L.append(f"- 人工框 {rep['n_human']}｜预测框 {rep['n_pred']}｜匹配 {rep['n_match']}")
    ref = rep.get("reference")
    if ref:
        L.append(f"- 参考臂：{ref.get('source', '')}｜召回 {ref.get('recall_e2e')}"
                 f"｜精确 {ref.get('precision')}｜IoU 中位 {ref.get('iou_median')}"
                 f"｜覆盖率偏差 {ref.get('coverage_median_clean')}")
    L.append("")
    L.append(f"**验收判定：{'PASS ✔' if rep['pass'] else 'FAIL ✘'}**\n")
    dec = rep.get("decision") or {}
    L.append(f"- 默认引擎决策：{'可切换' if dec.get('switch_default_engine') else '不切换'}"
             f"（依据 {dec.get('basis', 'P3 门槛')}）\n")
    L.append("| 验收项 | 实测 | 判定 |")
    L.append("|---|---|---|")
    for l in rep["gate_lines"]:
        parts = re.split(r"\s{2,}", l.strip(), maxsplit=1)
        if len(parts) == 2:
            cell = parts[1].replace("|", "\\|")
            L.append(f"| {parts[0]} | {cell} | |")
    L.append("\n## 总体指标\n")
    L.append(f"- 定位召回（IoU≥{rep['iou_thr']}，类别无关）：{rep['recall_loc']}")
    L.append(f"- 端到端召回（类别也对）：{rep['recall_e2e']}")
    L.append(f"- 精确（按预测框类别归属）：{rep['precision']}")
    L.append(f"- 漏检分档：完全无框（模型没框到）{rep['missed_no_box']}｜"
             f"框了但位置偏（IoU<{rep['iou_thr']}）{rep['missed_offset']}")
    if rep["iou"]:
        L.append(f"- 匹配 IoU：中位 {rep['iou']['median']}／P25 {rep['iou']['p25']}／"
                 f"最小 {rep['iou']['min']}／最大 {rep['iou']['max']}")
        L.append(f"- 严格档 IoU≥0.75 命中 {rep['strict_iou_hits']}"
                 f"（定位召回 {rep['strict_recall_loc']}）")
    cv = rep["coverage"]
    L.append(f"- 覆盖率偏差（人工面积/预测面积，>1 即预测框偏小）：中位 {cv['area_ratio_median']}；"
             f"剔除贴边对后中位 {cv['area_ratio_median_clean']}（clean 样本 {cv['n_clean']}）")
    L.append(f"  - Δw 相对中位 {cv['dw_rel_median']}／Δh 相对中位 {cv['dh_rel_median']}；"
             f"人工更大 {cv['n_human_larger']} 对 / 预测更大 {cv['n_pred_larger']} 对 / "
             f"基本相等 {cv['n_equal']} 对（容差 {COVERAGE_TOL:.0%}）")
    L.append(f"  - 只看确有差异的 {cv['n_diff']} 对：面积比中位 {cv['area_ratio_median_diff']}"
             f"（>1 = 预测框偏小）")
    L.append(f"- 贴边（坐标被夹紧）：预测 {rep['clip']['pred']} 框"
             f"（率 {rep['clip']['pred_rate']}）／人工 {rep['clip']['human']} 框；"
             f"受影响匹配对 {rep['clip']['pairs_affected']}")
    L.append(f"- 巨型框（>50% 画面）：{rep['big_pred_boxes']}")
    L.append(f"- 框数误差：每图均值 {rep['box_count_err']['mean_abs']}"
             f"／合计 {rep['box_count_err']['total_abs']}\n")
    L.append("## 逐类\n")
    L.append("| 类别 | 人工框 | 预测框 | 匹配 | 定位召回 | 端到端召回 | 精确 |")
    L.append("|---|---|---|---|---|---|---|")
    for k, v in rep["per_class"].items():
        f = lambda x: "—" if x is None else f"{x:.2f}"
        L.append(f"| {k} | {v['n_human']} | {v['n_pred']} | {v['n_match']} | "
                 f"{f(v['recall_loc'])} | {f(v['recall_e2e'])} | {f(v['precision'])} |")
    if rep["label_confusion"]:
        L.append(f"\n类别混淆（人工→预测）：{rep['label_confusion']}\n")
    if rep.get("p0_gates"):
        L.append("\n## P0 六类逐类门槛（P3）\n")
        L.append("| 类别 | 人工框 | 预测框 | 端到端召回 | 精确 | 判定 |")
        L.append("|---|---|---|---|---|---|")
        for g in rep["p0_gates"]:
            f = lambda x: "—" if x is None else f"{x:.2f}"
            L.append(f"| {g['class']} | {g['n_human']} | {g['n_pred']} | "
                     f"{f(g['recall_e2e'])} | {f(g['precision'])} | "
                     f"{'PASS' if g['pass'] else 'FAIL'} |")
    if rep.get("ap"):
        L.append("\n## 逐类 AP50 / AP50-95（含「其他」类）\n")
        L.append("| 类别 | GT | Pred | AP50 | AP50-95 |")
        L.append("|---|---|---|---|---|")
        for k, v in rep["ap"].items():
            f = lambda x: "—" if x is None else f"{x:.3f}"
            L.append(f"| {k} | {v['n_gt']} | {v['n_pred']} | {f(v['ap50'])} | "
                     f"{f(v['ap50_95'])} |")
    if rep.get("confusion_matrix"):
        labels = list(rep["confusion_matrix"])
        L.append("\n## 混淆矩阵（IoU≥0.5 匹配对；行=人工，列=预测）\n")
        L.append("| 人工＼预测 | " + " | ".join(labels) + " |")
        L.append("|---|" + "---|" * len(labels))
        for a in labels:
            vals = [str(rep["confusion_matrix"].get(a, {}).get(b, 0)) for b in labels]
            L.append(f"| {a} | " + " | ".join(vals) + " |")
    if rep.get("confusable_pairs"):
        L.append("\n## 易混类对专项\n")
        for k, v in rep["confusable_pairs"].items():
            L.append(f"- {k}: {v}")
    if rep.get("other"):
        o = rep["other"]
        L.append("\n## 「其他」类（单独报告，不进主门槛）\n")
        L.append(f"- 人工 {o['n_human']} / 预测 {o['n_pred']}｜端到端召回 {o['recall_e2e']}"
                 f"｜精确 {o['precision']}")
    if rep.get("cross_herb"):
        c = rep["cross_herb"]
        L.append("\n## 跨药材 hold-out（gold 子集）\n")
        if c.get("n_gold"):
            L.append(f"- 图 {c['n_gold']}｜人工 {c['n_human']} / 预测 {c['n_pred']}"
                     f"｜端到端召回 {c['recall_e2e']}｜精确 {c['precision']}")
        else:
            L.append(f"- {c.get('note', '无 cross_herb 记录')}")
    if rep.get("calibration"):
        cal_status = rep["calibration"].get("status", "raw_score")
        title = "置信度诊断（温度缩放后）" if cal_status == "temperature_scaled" \
            else "置信度诊断（原始 score，未校准）"
        L.append(f"\n## {title}\n")
        L.append(rep["calibration"].get("note", ""))
        L.append("")
        L.append("| 类别 | n | mean score | ECE | Brier |")
        L.append("|---|---|---|---|---|")
        for k, v in (rep["calibration"].get("per_class") or {}).items():
            L.append(f"| {k} | {v['n']} | {v['mean_score']} | {v['ece']} | {v['brier']} |")
        temps = rep["calibration"].get("temperature") or {}
        if temps:
            L.append("\n温度缩放参数（{} 拟合）\n".format(
                rep["calibration"].get("fit_split", "?")))
            L.append("| 类别 | n | T | 拟合集 ECE(before→after) |")
            L.append("|---|---|---|---|")
            for k, v in temps.items():
                L.append(f"| {k} | {v['n']} | {v['T']} | "
                         f"{v['ece_before']} → {v['ece_after']} |")
    if rep.get("limitations"):
        L.append("\n## P3 未完成 / 口径限制\n")
        for x in rep["limitations"]:
            L.append(f"- {x}")
    if rep["unit_mismatch"]:
        L.append("\n## ⚠️ 框数比异常（疑似标注单元口径不一致）\n")
        L.append("以下类别预测/人工框数比 ≥2：精确/召回不可当模型误差读，需先统一标注单元口径"
                 "（人工整团一框 vs 模型逐颗粒）。\n")
        for u in rep["unit_mismatch"]:
            L.append(f"- **{u['class']}**：预测 {u['n_pred']} / 人工 {u['n_human']} = "
                     f"{u['ratio']} —— {u['hint']}")
        L.append("")
    for title, key, fmt in (("漏检（人工有框、预测无对应）", "missed",
                             lambda m: f"最佳IoU={m['best_iou']:.3f}"),
                            ("多出框（预测无对应人工框）", "extra",
                             lambda e: f"面积比={e['area_ratio']*100:.2f}%")):
        L.append(f"## {title}（{len(rep[key])} 例）\n")
        for x in rep[key][:40]:
            L.append(f"- `{os.path.basename(x['path'])}` {x['label']} {fmt(x)}")
        if len(rep[key]) > 40:
            L.append(f"- …另 {len(rep[key]) - 40} 例见 report.json")
        L.append("")
    if rep["pending"]:
        L.append(f"## 未纳入评测（{len(rep['pending'])} 张）\n")
        for p, why in rep["pending"][:40]:
            L.append(f"- `{os.path.basename(p)}`：{why}")
        L.append("")
    path = os.path.join(out_dir, "report.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    return path


# ---------------------------------------------------------------- ③ 离线自检
def selftest():
    """合成真值验证全部指标口径（不联网、不落盘）。"""
    checks = []

    def ck(name, cond):
        checks.append((name, bool(cond)))

    def mk_pre(W, H, objs, pv="vX"):
        # C2a 起 evaluate / ann_from_tasks 吃契约记录（= load_prelabel_index 的值类型）
        return PrelabelRecord.from_dict({"original_size": [W, H], "objects": objs,
                                         "prompt_version": pv})

    def O(lab, b):
        return {"label": lab, "tight_bbox": b}

    def A(lab, b):
        return {"label": lab, "box": b}

    items = [{"image_path": "C:/a.png", "feature_label": "草酸钙簇晶"},
             {"image_path": "C:/b.png", "feature_label": "石细胞"},
             {"image_path": "C:/c.png", "feature_label": "导管"}]
    pre = {norm_path("C:/a.png"): mk_pre(1000, 1000, [O("草酸钙簇晶", [100, 100, 300, 300]),
                                                O("导管", [400, 100, 880, 200]),
                                                O("淀粉粒", [600, 600, 700, 700])]),
           norm_path("C:/b.png"): mk_pre(1000, 1000, [O("石细胞", [0, 0, 150, 150]),
                                                O("草酸钙簇晶", [500, 500, 520, 520])])}
    ann = {norm_path("C:/a.png"): [A("草酸钙簇晶", [100, 100, 300, 300]),
                             A("导管", [400, 100, 900, 200])],
           norm_path("C:/b.png"): [A("石细胞", [0, 0, 200, 200])]}
    r = evaluate(items, pre, ann)

    ck("样本口径：gold 3 / 可评 2 / 未标注 1", r["n_gold"] == 3 and r["n_scored"] == 2
       and len(r["pending"]) == 1)
    ck("计数：人工 3 / 预测 5 / 匹配 3", (r["n_human"], r["n_pred"], r["n_match"]) == (3, 5, 3))
    ck("定位召回 = 1.0", r["recall_loc"] == 1.0)
    ck("端到端召回 = 1.0", r["recall_e2e"] == 1.0)
    ck("精确 = 3/5 = 0.6", abs(r["precision"] - 0.6) < 1e-9)
    ck("漏检 0 例", len(r["missed"]) == 0)
    ck("多出框 2 例（淀粉粒 + 孤立小框）", len(r["extra"]) == 2)
    ck("IoU 中位 0.96 / 最小 0.5625",
       abs(r["iou"]["median"] - 0.96) < 1e-4 and abs(r["iou"]["min"] - 0.5625) < 1e-4)
    ck("严格档 0.75：命中 2（0.5625 那一对不达标）",
       r["strict_iou_hits"] == 2 and abs(r["strict_recall_loc"] - 2 / 3) < 1e-3)
    ck("覆盖率偏差中位（含贴边，3 对）= 1.0417",
       abs(r["coverage"]["area_ratio_median"] - 1.0417) < 1e-3)
    ck("覆盖率偏差 clean（剔 1 对贴边，2 对）= 1.0208",
       r["coverage"]["n_clean"] == 2 and abs(r["coverage"]["area_ratio_median_clean"] - 1.0208) < 1e-3)
    ck("under-cover 方向：人工更大 2 对（另 1 对面积相等）/ 预测更大 0 对",
       r["coverage"]["n_human_larger"] == 2 and r["coverage"]["n_pred_larger"] == 0
       and r["coverage"]["n_equal"] == 1)
    ck("仅看差异对（2 对）面积比中位 = (1.0417+1.7778)/2 = 1.4097",
       r["coverage"]["n_diff"] == 2
       and abs(r["coverage"]["area_ratio_median_diff"] - 1.4097) < 1e-3)
    ck("贴边：预测 1 / 人工 1 / 受影响对 1",
       r["clip"]["pred"] == 1 and r["clip"]["human"] == 1 and r["clip"]["pairs_affected"] == 1)
    ck("巨型框：0（最大 5% 画面）", r["big_pred_boxes"] == 0)
    ck("框数误差：a 图 |3-2|=1、b 图 |2-1|=1 → 均值 1.0",
       r["box_count_err"]["total_abs"] == 2 and abs(r["box_count_err"]["mean_abs"] - 1.0) < 1e-9)
    ck("逐类精确：淀粉粒 0/1 = 0.0（纯误检类）", r["per_class"]["淀粉粒"]["precision"] == 0.0)
    ck("逐类召回：石细胞 1/1；导管 1/1",
       r["per_class"]["石细胞"]["recall_e2e"] == 1.0 and r["per_class"]["导管"]["recall_loc"] == 1.0)
    ck("空标注 → 计为已标注且 0 框（不计入 pending）",
       evaluate(items[:1], {norm_path("C:/a.png"): pre[norm_path("C:/a.png")]},
                {norm_path("C:/a.png"): []})["n_human"] == 0)

    # 巨型框（95%×95% 画面，与人工框几乎不重叠 → 同时是漏检+多出框）
    pre2 = {norm_path("C:/a.png"): mk_pre(1000, 1000, [O("导管", [0, 0, 950, 950])])}
    ann2 = {norm_path("C:/a.png"): [A("草酸钙簇晶", [500, 500, 600, 600])]}
    r2 = evaluate(items[:1], pre2, ann2)
    ck("巨型框计 1（95%×95% 画面）", r2["big_pred_boxes"] == 1)
    ck("巨型框贴边计 1", r2["clip"]["pred"] == 1)
    ck("巨型框与人工框不重叠 → 1 漏检 + 1 多出框",
       len(r2["missed"]) == 1 and len(r2["extra"]) == 1)

    # 类别混淆（IoU 达标但类别答错）：定位召回满分、端到端召回 0
    pre3 = {norm_path("C:/a.png"): mk_pre(1000, 1000, [O("导管", [0, 0, 400, 400])])}
    ann3 = {norm_path("C:/a.png"): [A("草酸钙簇晶", [10, 10, 300, 300])]}
    r3 = evaluate(items[:1], pre3, ann3)
    ck("类别混淆 草酸钙簇晶→导管 记 1", r3["label_confusion"] == {"草酸钙簇晶→导管": 1})
    ck("召回归因：定位未漏但端到端漏（recall_loc=1 / e2e=0）",
       r3["recall_loc"] == 1.0 and r3["recall_e2e"] == 0.0)
    ck("错答的框把精确度算到预测类别（导管 0/1）", r3["per_class"]["导管"]["precision"] == 0.0)
    ck("错答同时拉低人工类别的端到端召回（簇晶 0/1）",
       r3["per_class"]["草酸钙簇晶"]["recall_e2e"] == 0.0)

    # 剔除类别：两侧同时剔除（口径不一致的类不参与指标）
    r4 = evaluate(items, pre, ann, exclude={"淀粉粒"})
    ck("剔除类别：预测 5→4（淀粉粒被剔）、人工 3、多出框 2→1",
       (r4["n_pred"], r4["n_human"], len(r4["extra"])) == (4, 3, 1))
    ck("剔除类别：该类的 per_class 不再出现", "淀粉粒" not in r4["per_class"])

    # 框数比异常（标注单元口径不一致）自动诊断
    pre5 = {norm_path("C:/a.png"): mk_pre(1000, 1000, [O("淀粉粒", [0, 0, 100, 100]),
                                                 O("淀粉粒", [200, 200, 300, 300]),
                                                 O("淀粉粒", [400, 400, 500, 500]),
                                                 O("淀粉粒", [600, 600, 700, 700]),
                                                 O("淀粉粒", [100, 600, 200, 700]),
                                                 O("淀粉粒", [300, 100, 400, 200])])}
    ann5 = {norm_path("C:/a.png"): [A("淀粉粒", [0, 0, 100, 100]),
                              A("淀粉粒", [200, 200, 300, 300]),
                              A("淀粉粒", [400, 400, 500, 500])]}
    r5 = evaluate(items[:1], pre5, ann5)
    ck("框数比异常：6 预测 / 3 人工 = 2.0 触发告警",
       len(r5["unit_mismatch"]) == 1 and r5["unit_mismatch"][0]["class"] == "淀粉粒"
       and r5["unit_mismatch"][0]["ratio"] == 2.0)
    ck("框数比正常（3:3）不告警", evaluate(items[:1], pre5, ann5,
                                          exclude={"淀粉粒"})["unit_mismatch"] == [])

    # 漏检分档：完全没框到 vs 框了但位置偏
    pre6 = {norm_path("C:/a.png"): mk_pre(1000, 1000, [O("导管", [0, 0, 100, 100])])}
    ann6 = {norm_path("C:/a.png"): [A("导管", [500, 500, 600, 600])]}
    r6 = evaluate(items[:1], pre6, ann6)
    ck("漏检分档：完全无重叠 → missed_no_box 1 / missed_offset 0",
       r6["missed_no_box"] == 1 and r6["missed_offset"] == 0)
    ck("漏检分档：有重叠但 IoU<阈值 → missed_offset 1 / missed_no_box 0",
       r2["missed_offset"] == 1 and r2["missed_no_box"] == 0)

    # 判定线
    ok, lines = verdict_of({"recall_e2e": 0.9, "precision": 0.9, "iou": {"median": 0.9},
                            "coverage": {"area_ratio_median_clean": 1.05}})
    ck("验收线：全过 → PASS", ok is True and len(lines) == 4)
    ok2, _ = verdict_of({"recall_e2e": 0.9, "precision": 0.9, "iou": {"median": 0.9},
                         "coverage": {"area_ratio_median_clean": 1.30}})
    ck("验收线：覆盖率偏差 1.30 超窗 → FAIL", ok2 is False)
    ok3, l3 = verdict_of({"recall_e2e": None, "precision": 0.9, "iou": {"median": 0.9},
                          "coverage": {"area_ratio_median_clean": None}})
    ck("验收线：缺数据 → FAIL 且提示无数据", ok3 is False and any("无数据" in x for x in l3))

    # 分层配额
    q = allocate({"a": 171, "b": 101, "c": 68, "d": 45, "e": 24, "f": 8, "g": 3, "h": 2},
                 60, 3, 12)
    ck("配额：总和 ≤ 目标 60 且各类不超过池", sum(q.values()) <= 60
       and all(q[k] <= v for k, v in {"a": 171, "b": 101, "c": 68, "d": 45, "e": 24,
                                      "f": 8, "g": 3, "h": 2}.items()))
    ck("配额：小类有下限（菌丝 2 张全取、方晶 ≥3）", q["h"] == 2 and q["f"] >= 3)
    ck("配额：大类不超上限（簇晶 ≤12）", q["a"] <= 12)
    ck("配额：池小于目标时全取", allocate({"x": 4, "y": 3}, 60, 3, 12) == {"x": 4, "y": 3})

    # P3：逐类 AP（101 点插值，AP50 / AP50-95）
    items_ap = [{"image_path": "C:/ap.png", "feature_label": "导管"}]
    pre_ap = {norm_path("C:/ap.png"): mk_pre(1000, 1000, [
        {"label": "导管", "tight_bbox": [100, 100, 300, 300], "score": 0.9}], pv="yolo")}
    ann_ap = {norm_path("C:/ap.png"): [A("导管", [100, 100, 300, 300])]}
    r_ap = average_precision({"items": items_ap}, pre_ap, ann_ap)
    ck("AP：完美匹配 AP50=1 / AP50-95=1",
       r_ap["导管"]["ap50"] == 1.0 and r_ap["导管"]["ap50_95"] == 1.0)
    # IoU=0.6 时只有 0.50/0.55/0.60 三个阈值算 TP → AP50-95=0.3
    pre_ap2 = {norm_path("C:/ap.png"): mk_pre(1000, 1000, [
        {"label": "导管", "tight_bbox": [100, 100, 220, 300], "score": 0.9}], pv="yolo")}
    ann_ap2 = {norm_path("C:/ap.png"): [A("导管", [100, 100, 300, 300])]}
    r_ap2 = average_precision({"items": items_ap}, pre_ap2, ann_ap2)
    ck("AP：IoU≈0.6 → AP50=1 / AP50-95≈0.3",
       r_ap2["导管"]["ap50"] == 1.0 and abs(r_ap2["导管"]["ap50_95"] - 0.3) < 0.02)
    # 1 TP（高分）+ 1 FP（低分），2 个 GT → 101 点插值 AP50=51/101≈0.505
    pre_ap3 = {norm_path("C:/ap.png"): mk_pre(1000, 1000, [
        {"label": "导管", "tight_bbox": [100, 100, 300, 300], "score": 0.9},
        {"label": "导管", "tight_bbox": [700, 700, 800, 800], "score": 0.3}], pv="yolo")}
    ann_ap3 = {norm_path("C:/ap.png"): [A("导管", [100, 100, 300, 300]),
                                        A("导管", [500, 500, 700, 700])]}
    r_ap3 = average_precision({"items": items_ap}, pre_ap3, ann_ap3)
    ck("AP：1 TP + 1 FP（2 GT）→ AP50≈0.505",
       abs(r_ap3["导管"]["ap50"] - 51 / 101) < 1e-4)
    ck("AP：无 GT 的类别 AP=None（不把纯误检类刷成 0）",
       average_precision({"items": items_ap}, pre_ap3,
                         {norm_path("C:/ap.png"): []}).get("导管", {}).get("ap50") is None)
    # P3 增量与总判定：P0 六类里只有导管有数据且全对，其余无数据 → 整体 FAIL
    main_p3 = evaluate(items_ap, pre_ap, ann_ap, exclude={"其他"})
    p3x = p3_extras({"items": items_ap}, pre_ap, ann_ap, main_p3, {"其他"})
    main_p3.update(p3x)
    ok_p3, lines_p3 = verdict_of(main_p3)
    ck("P3：P0 六类门槛进入总判定（仅导管有数据 → 整体 FAIL）", not ok_p3)
    ck("P3：无「其他」预测时 other=None，且 AP/P0 表都生成",
       p3x["other"] is None and "导管" in p3x["ap"]
       and any(g["class"] == "导管" and g["pass"] for g in p3x["p0_gates"]))
    ck("P3：P0 门槛行可进 gate_lines", any("P0 导管" in l for l in lines_p3))

    # 置信度校准：温度缩放原语
    ck("校准：sigmoid(logit(p)) 恒等", abs(_sigmoid(_logit(0.37)) - 0.37) < 1e-9)
    ck("校准：全对且 score=0.9 的 ECE=0.1（10 个观测）",
       abs(_ece_from_obs([(0.9, 1)] * 10) - 0.1) < 1e-9)
    ck("校准：恒错的高分预测拟合出 T>1（降低置信）",
       fit_temperature([(0.9, 0)] * CALIBRATION_MIN_N) > 1.0)

    # 报告路径口径：项目内绝对路径 → 相对项目根；项目外保持原样（不写 `..`）
    inside = os.path.join(PROJECT_ROOT, "dataset", "a.jpg")
    outside = os.path.join(os.path.dirname(PROJECT_ROOT), "outside.jpg")
    ck("报告路径：项目内绝对路径 → 相对路径", _project_rel(inside) == "dataset/a.jpg")
    ck("报告路径：项目外绝对路径保持原样",
       _project_rel(outside) == outside.replace(os.sep, "/"))

    npass = sum(1 for _, v in checks if v)
    for name, v in checks:
        print(f"  [{'PASS' if v else 'FAIL'}] {name}")
    print(f"\n{PROMPT_TAG} offline selftest: {npass}/{len(checks)} PASS")
    return npass == len(checks)


PROMPT_TAG = "det_eval"


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-gold", action="store_true", help="按类分层抽样建 gold set")
    ap.add_argument("--import-ls", action="store_true", help="建盲标项目并导入（配 --build-gold）")
    ap.add_argument("--gold-from-p0", action="store_true",
                    help="从 P0 split_report 的 gold_eval + 单类 labels 重建 gold/人工框"
                         "（原 blind gold 不在本机时的 P3 回退；产物非盲标，见 report 注记）")
    ap.add_argument("--force", action="store_true", help="允许 --gold-from-p0 覆盖已有 gold_set.json")
    ap.add_argument("--split-report", default=P0_SPLIT_REPORT)
    ap.add_argument("--label-root", default=P0_LABEL_ROOT)
    ap.add_argument("--gold-set", default=GOLD_PATH, help="gold_set.json（默认 det_eval/gold_set.json）")
    ap.add_argument("--report", action="store_true", help="读 gold 项与人工标注出评测报告")
    ap.add_argument("--ann-json", default="",
                    help="人工框 JSON（gold_ann.json；与 --project-id/--export-json 三选一）")
    ap.add_argument("--pred-dir", default="",
                    help="YOLO 预测目录/单文件（每图一份 DetBox JSON；不跑推理）")
    ap.add_argument("--engine", choices=["vlm", "yolo"], default="vlm",
                    help="预测引擎：vlm=读 prelabels；yolo=按 detector.json 现跑 gold 图")
    ap.add_argument("--weights", default="", help="YOLO 权重（覆盖 detector.json）")
    ap.add_argument("--detector-config", default="", help="YOLO 推理配置（覆盖 detector.json 路径）")
    ap.add_argument("--p3", action="store_true",
                    help="P3 报告增量：主指标剔除「其他」+ 逐类 AP/混淆矩阵/P0 六类门槛/"
                         "其他类单独/跨药材/score 诊断")
    ap.add_argument("--reference-json", default="",
                    help="参考臂指标摘要（如既有 VLM gold 报告；不重跑，仅并排展示）")
    ap.add_argument("--calibrate-scores", action="store_true",
                    help="P3 §6.5 第 4 条：在 calibration-split 上跑 YOLO，逐类拟合温度缩放")
    ap.add_argument("--calibration-split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--calibration-limit", type=int, default=0, help="只取前 N 张做校准（0=全量）")
    ap.add_argument("--calibration-out", default=os.path.join(EVAL_DIR, "calibration.json"))
    ap.add_argument("--calibration-json", default="",
                    help="报告时应用已拟合的温度缩放（并把校准信息写进报告）")
    ap.add_argument("--project-id", type=int, default=0, help="标注所在的 LS 项目 id")
    ap.add_argument("--export-json", default="", help="LS 导出的 JSON（离线评测，任务数组或 {tasks:[…]}）")
    ap.add_argument("--exclude-classes", default="",
                    help='报告时剔除的类别名（逗号分隔）。用于排除"标注单元口径不一致"的类，'
                         '如 "淀粉粒,分泌组织"——两侧同时剔除，只看口径一致的类')
    ap.add_argument("--n", type=int, default=60, help="gold 目标张数（默认 60）")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-per-class", type=int, default=3)
    ap.add_argument("--max-per-class", type=int, default=12)
    ap.add_argument("--out-dir", default=EVAL_DIR)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--selftest", action="store_true", help="离线合成真值自检")
    args = ap.parse_args()

    if args.selftest:
        sys.exit(0 if selftest() else 1)

    ensure_dirs()
    os.makedirs(args.out_dir, exist_ok=True)

    if args.build_gold:
        gold = build_gold(args.n, args.seed, args.min_per_class, args.max_per_class,
                          dry_run=args.dry_run)
        if args.import_ls:
            import_to_ls(gold, dry_run=args.dry_run)
        return

    if args.gold_from_p0:
        try:
            build_gold_from_p0(args.split_report, args.label_root, args.out_dir,
                               dry_run=args.dry_run, overwrite=args.force)
        except (OSError, ValueError) as e:
            sys.exit(f"[FAIL] 重建 gold 失败：{e}")
        return

    if args.calibrate_scores:
        try:
            settings = detector_yolo.load_settings(load_config())
            obs, n = collect_calibration_obs(settings, args.calibration_split,
                                             args.calibration_limit)
        except (OSError, ValueError, RuntimeError) as e:
            sys.exit(f"[FAIL] 校准失败：{e}")
        cal = build_calibration(obs, args.calibration_split, n)
        if args.dry_run:
            print("[DRY-RUN] 不落盘：{}".format(cal))
            return
        os.makedirs(os.path.dirname(os.path.abspath(args.calibration_out)), exist_ok=True)
        with open(args.calibration_out, "w", encoding="utf-8") as f:
            json.dump(cal, f, ensure_ascii=False, indent=2)
        print("校准 → {}".format(args.calibration_out))
        for cls, v in cal["classes"].items():
            print(f"  {cls:<12} n={v['n']:>5} T={v['T']:<6} "
                  f"ECE {v['ece_before']} → {v['ece_after']}")
        return

    if args.report:
        if not os.path.exists(args.gold_set):
            sys.exit(f"[FAIL] 没有 gold set：{args.gold_set}（先跑 --build-gold 或 --gold-from-p0）")
        with open(args.gold_set, encoding="utf-8") as f:
            gold = json.load(f)
        gold["items"] = [dict(it, image_path=canon_path(it["image_path"])) for it in gold["items"]]
        exclude = {s.strip() for s in args.exclude_classes.split(",") if s.strip()}
        if args.p3:
            exclude.add(OTHER_LABEL)

        # 预测源：pred-dir（离线）> yolo（现跑）> vlm（prelabels）
        settings = None
        if args.pred_dir:
            pred_index = load_pred_dir(args.pred_dir)
            engine = "pred-dir"
            detector = {"engine": "pred-dir", "path": args.pred_dir}
        elif args.engine == "yolo":
            try:
                pred_index, settings = yolo_predictions(gold, args.weights or None,
                                                        args.detector_config or None)
            except (OSError, ValueError, RuntimeError) as e:
                sys.exit(f"[FAIL] YOLO 推理失败：{e}")
            engine = "yolo"
            detector = {"engine": "yolo", "weights": settings["weights"],
                        "imgsz": settings["imgsz"], "conf": settings["conf"],
                        "iou": settings["iou"], "tile": settings["tile"],
                        "overlap": settings["overlap"]}
        else:
            pred_index = load_prelabel_index()
            engine = "vlm"
            detector = {"engine": "vlm"}
        if not pred_index:
            sys.exit("[FAIL] 没有预测：vlm 需 prelabels，yolo/pred-dir 需可读预测")

        # 人工标注源：ann-json（离线重建）> LS 导出 JSON > 活 LS 项目
        ann_note = ""
        if args.ann_json:
            ann = load_ann_json(args.ann_json)
            unmatched, blank = [], 0
            ann_note = f"标注源: {os.path.basename(args.ann_json)}"
        elif args.export_json or args.project_id:
            if args.export_json:
                with open(args.export_json, encoding="utf-8") as f:
                    j = json.load(f)
                tasks = j["tasks"] if isinstance(j, dict) and "tasks" in j else j
                ann_note = f"标注源: 导出 JSON {os.path.basename(args.export_json)}"
            else:
                tasks = fetch_tasks(load_config(), args.project_id)
                ann_note = f"标注源: LS 项目 {args.project_id}"
            ann, unmatched, blank = ann_from_tasks(tasks, gold, pred_index)
        else:
            sys.exit("[FAIL] 请给 --ann-json / --export-json / --project-id 之一")
        pv = Counter((pred_index[norm_path(it["image_path"])].prompt_version or "?")
                     for it in gold["items"]
                     if norm_path(it["image_path"]) in pred_index)
        reference = None
        if args.reference_json:
            with open(args.reference_json, encoding="utf-8") as f:
                reference = json.load(f)
        calibration = None
        if args.calibration_json:
            with open(args.calibration_json, encoding="utf-8") as f:
                calibration = json.load(f)
            apply_temperature(pred_index, calibration)
            detector["calibration"] = args.calibration_json
        meta = {"annotated": len(ann), "blank": blank, "unmatched": unmatched,
                "prompt_versions": dict(pv), "engine": engine, "detector": detector,
                "p3": args.p3, "reference": reference, "calibration": calibration}
        note = "{}；预测源: {}（{} 张）".format(ann_note, engine, len(pred_index))
        if args.p3:
            note += "；主指标剔除「其他」类，其他类单独报告"
        rep = report_pred(gold, pred_index, ann, note, args.out_dir, exclude=exclude, meta=meta)
        print("md 报告 →", write_md(rep, args.out_dir))
        return

    ap.print_help()


if __name__ == "__main__":
    main()