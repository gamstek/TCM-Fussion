"""reflect_detect.py — 检测提示词的「案例→候选→盲评」自演化闭环（离线，运行期不改提示）。

为什么存在（2026-11-04，用户愿景）：模型把纤维判成导管时，靠人反复手改提示词既慢又不可复现。
本入口把这件事变成闭环，**人只负责判定对错，不负责写提示词**：

  ① --record   记一条案例：图片（可选框）+ 人工判定 + 当时的模型判定
  ② --propose  让模型读「当前提示词 + 失败案例（含图）」自己写一版候选提示词
  ③ --evaluate 用候选提示词重跑同一批案例，与人工判定比对，出混淆矩阵与配对差
  ④ --accept   把通过评测的候选落盘留档（**不自动改 prompts.py**，上线仍需人工确认）

纪律（与 crop_gold/det_eval 一致）：
  · 评测集固定、可复现；改完必须重跑同一批，不能只看个案；
  · 候选只在离线评测里生效，生产提示词仍是 prompts.py 的常量；
  · 案例少（<30）时候选极易过拟合，报告会显式标注样本量；
  · 不联网的 --selftest 覆盖匹配/汇总/案例校验，回归可离线跑。

用法（项目根）：
  python 目标检测\\cli\\reflect_detect.py --record --image <图> --label 纤维 --model-label 导管 --box 675,0,1206,1410
  python 目标检测\\cli\\reflect_detect.py --list
  python 目标检测\\cli\\reflect_detect.py --evaluate --name baseline
  python 目标检测\\cli\\reflect_detect.py --propose --out candidates/c1.txt
  python 目标检测\\cli\\reflect_detect.py --evaluate --name c1 --prompt-file candidates/c1.txt
  python 目标检测\\cli\\reflect_detect.py --selftest
"""
import argparse
import collections
import csv
import hashlib
import json
import math
import os
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path

import paths
from geom import iou_box

SCHEMA = "detect-reflect-case/v1"
EVAL_SCHEMA = "detect-reflect-eval/v1"
# 案例库文件名；--cases 可切换。legacy 诊断集与正式分组集必须分开，不能混算。
CASE_FILE = "cases.jsonl"


def case_path():
    path = Path(CASE_FILE)
    return path if path.is_absolute() else root_dir() / path


def root_dir(base=None):
    return Path(base) if base else Path(paths.DET_RESULTS) / "reflect_detect"


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _box(value):
    if value in (None, "", []):
        return None
    if isinstance(value, str):
        parts = [p for p in value.replace("，", ",").split(",") if p.strip()]
        value = [float(p) for p in parts]
    if len(value) != 4:
        raise ValueError("box 需要 4 个数字：x1,y1,x2,y2")
    x1, y1, x2, y2 = [float(v) for v in value]
    return [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)]


def validate_case(case):
    """案例只认这些字段；人工判定与图片不能空。返回规范化副本或抛 ValueError。"""
    if not isinstance(case, dict):
        raise ValueError("案例必须是对象")
    image = str(case.get("image") or "").strip()
    human = str(case.get("human_label") or "").strip()
    if not image or not human:
        raise ValueError("案例缺 image 或 human_label")
    model_label = str(case.get("model_label") or "").strip()
    split = str(case.get("split") or "dev").strip().lower()
    if split not in {"dev", "test"}:
        raise ValueError("split 只能是 dev/test")
    return {"schema": SCHEMA, "image": image, "image_sha256": str(case.get("image_sha256") or ""),
            "box": _box(case.get("box")), "human_label": human, "model_label": model_label,
            "split": split, "note": str(case.get("note") or ""),
            "recorded_at": str(case.get("recorded_at") or "")}


def case_key(case):
    """同图同框视为同一条案例；框为空时按整图算。框先归一化，字符串栏位也一致。"""
    try:
        box = _box(case.get("box")) or []
    except (TypeError, ValueError):
        box = []
    return hashlib.sha256(json.dumps([case.get("image_sha256") or case["image"], box],
                                     ensure_ascii=False).encode("utf-8")).hexdigest()


def load_cases(base=None, split=None):
    path = case_path() if base is None else Path(base) / "cases.jsonl"
    if not path.is_file():
        return []
    rows, seen = [], set()
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if not line:
                continue
            try:
                case = validate_case(json.loads(line))
            except (ValueError, json.JSONDecodeError):
                continue
            if split and split != "all" and case["split"] != split:
                continue
            key = case_key(case)
            if key in seen:
                continue
            seen.add(key)
            rows.append({**case, "case_id": key[:12]})
    return rows


def append_case(case):
    path = case_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(case, ensure_ascii=False) + "\n")
    return path


def command_import_manifest(args):
    """按已审计的分组草案导入案例；草案里的 split/box/真值原样保留，不重新抽样。"""
    source = Path(args.import_manifest)
    if not source.is_file():
        print(f"[FAIL] 分组草案不存在：{source}")
        return 2
    doc = json.loads(source.read_text(encoding="utf-8"))
    rows = doc.get("rows") or []
    if not rows:
        print("[FAIL] 草案没有 rows")
        return 2
    existing = {case_key(c) for c in load_cases()}
    added = skipped = 0
    for row in rows:
        try:
            case = validate_case({"image": row["image"], "image_sha256": row.get("sha256") or "",
                                  "box": row.get("box"), "human_label": row["human_label"],
                                  "split": row["split"],
                                  "note": f"manifest {source.name}; group={row.get('group_key', '')}; "
                                          f"herb={row.get('herb', '')}"})
        except (KeyError, ValueError) as exc:
            print(f"[SKIP] 草案行非法：{exc}")
            skipped += 1
            continue
        key = case_key(case)
        if key in existing:
            skipped += 1
            continue
        append_case(case)
        existing.add(key)
        added += 1
    print(f"[OK] 从 {source.name} 导入 {added} 条（跳过 {skipped}）→ {case_path()}")
    return 0


def match_detection(dets, case):
    """在同图检测结果里找与案例框 IoU 最大的框；案例没有框时取最高分框。"""
    rows = [d for d in dets if d.get("object_label")]
    if not rows:
        return None, None
    box = case.get("box")
    if box:
        best = max(rows, key=lambda d: iou_box(d.get("tight_box_source"), box))
        return best, iou_box(best.get("tight_box_source"), box)
    return max(rows, key=lambda d: d.get("score") or 0), None


def score_cases(cases, detections):
    """`detections`: {image 路径: [crops.json 行]} → 逐条比对与汇总。"""
    rows = []
    pairs = collections.Counter()
    for case in cases:
        det, iou = match_detection(detections.get(case["image"], []), case)
        predicted = (det or {}).get("object_label") or ""
        if not predicted:
            bucket = "miss"
        elif iou is None:
            bucket = "unlocalized"
        elif iou < 0.7:
            bucket = "localization_error"
        else:
            bucket = "correct" if predicted == case["human_label"] else "classification_error"
        ok = bucket == "correct"
        rows.append({"case_id": case.get("case_id") or case_key(case)[:12],
                     "image": case["image"], "split": case.get("split", "dev"),
                     "human_label": case["human_label"],
                     "predicted": predicted, "iou": iou, "ok": ok, "error_bucket": bucket})
        pairs[(case["human_label"], predicted or "未检出")] += 1
    ok = sum(1 for r in rows if r["ok"])
    return {"n": len(rows), "correct": ok, "accuracy": round(ok / len(rows), 4) if rows else 0.0,
            "by_split": _by_split(rows),
            "confusion": {f"{human}→{pred}": n for (human, pred), n in sorted(pairs.items())},
            "rows": rows}


def _label_fields(votes, human_label):
    """只看标签是否一致的稳定口径；框抖动另由 median_iou 表示。"""
    labels = [row["predicted"] for row in votes]
    ious = [row["iou"] for row in votes if row.get("iou") is not None]
    label_stable = len(set(labels)) == 1
    median_iou = statistics.median(ious) if ious else None
    if not label_stable:
        bucket = "label_unstable"
    elif not labels[0]:
        bucket = "miss"
    elif median_iou is None or median_iou < 0.7:
        bucket = "localization_error"
    elif labels[0] != human_label:
        bucket = "classification_error"
    else:
        bucket = "correct"
    return {"label_stable": label_stable, "median_iou": median_iou, "label_bucket": bucket}


def enrich_rows(rows):
    """旧报告没有标签口径字段时，从 votes 现场补算，避免为口径升级重跑整批评测。"""
    out = []
    for row in rows:
        if row.get("label_bucket"):
            out.append(row)
            continue
        votes = row.get("votes") or []
        out.append({**row, "label_bucket": "unknown"} if not votes
                   else {**row, **_label_fields(votes, row.get("human_label", ""))})
    return out


def summarize_repeats(rounds):
    """按案例汇总逐轮结果。

    同时保留两种稳定口径：
      · `stable`/`error_bucket`：标签与定位资格都一致（旧口径，框抖动会判不稳定）；
      · `label_stable`/`label_bucket`：只要求标签一致，IoU 另用中位数衡量
        （分诊反射候选时应使用后者——框抖动不等于类别判断不稳）。
    """
    rows = []
    for votes in zip(*(round_result["rows"] for round_result in rounds), strict=True):
        first = votes[0]
        if any(row["case_id"] != first["case_id"] for row in votes):
            raise ValueError("重复轮次案例顺序不一致")
        counts = collections.Counter((row["predicted"], row["error_bucket"]) for row in votes)
        (label, bucket), count = counts.most_common(1)[0]
        stable = count == len(votes)
        fields = _label_fields(votes, first["human_label"])
        rows.append({**first, "predicted": label if count > len(votes) / 2 else "",
                     "iou": first["iou"] if len(votes) == 1 else None,
                     "error_bucket": bucket if stable else "unstable",
                     **fields,
                     "ok": stable and bucket == "correct",
                     "stable": stable, "votes": list(votes)})
    return rows


def _by_split(rows):
    out = {}
    for split in sorted({r["split"] for r in rows}):
        group = [r for r in rows if r["split"] == split]
        hit = sum(1 for r in group if r["ok"])
        out[split] = {"n": len(group), "correct": hit, "accuracy": round(hit / len(group), 4)}
    return out


def run_detection(images, out_dir, prompt_file=None, python=None, timeout=1800, temperature=None):
    """以子进程跑 detect_sample；解释器需具备 cv2+RapidOCR（默认 .venv/vision）。"""
    bad = [str(p) for p in images if "," in str(p)]
    if bad:
        raise RuntimeError(f"图片路径含逗号，无法作为 --images 列表传递（会拆错图）：{bad[:3]}")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cmd = [str(python or default_python()), "-X", "utf8",
           str(Path(paths.DETECTION_CLI_DIR) / "detect_sample.py"),
           "--images", ",".join(str(p) for p in images),
           "--sample", "reflect_eval", "--herb", "unknown", "--out-dir", str(out)]
    if prompt_file:
        cmd += ["--prompt-file", str(prompt_file)]
    if temperature is not None:
        cmd += ["--temperature", str(temperature)]
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(paths.MODULE_DIRS),
               PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
    done = subprocess.run(cmd, cwd=paths.PROJECT_ROOT, env=env, capture_output=True,
                          text=True, encoding="utf-8", errors="replace", timeout=timeout)
    if done.returncode:
        raise RuntimeError(f"检测子进程退出码 {done.returncode}")
    if any("] FAIL" in line for line in done.stdout.splitlines()):
        raise RuntimeError("检测轮次存在读取、请求或解析失败，不能计为漏检")
    crops = out / "crops.json"
    if not crops.is_file():
        raise RuntimeError("检测子进程没有产出 crops.json")
    rows = json.loads(crops.read_text(encoding="utf-8")).get("crops", [])
    grouped = collections.defaultdict(list)
    for row in rows:
        grouped[str(Path(row.get("source_image") or ""))].append(row)
    return grouped


def default_python():
    candidate = Path(paths.PROJECT_ROOT) / ".venv" / "vision" / "Scripts" / "python.exe"
    return candidate if candidate.is_file() else Path(sys.executable)


def command_import(args):
    """从 YOLO 格式特征目录导入案例：classes.txt 给真值，labels/*.txt 给框，attributes.csv 给溯源。"""
    base = Path(args.import_dataset)
    images_dir, labels_dir = base / "images", base / "labels"
    if not images_dir.is_dir() or not labels_dir.is_dir():
        print(f"[FAIL] 数据集目录缺 images/ 或 labels/：{base}")
        return 2
    classes_file = base / "classes.txt"
    if not classes_file.is_file():
        print(f"[FAIL] 缺 classes.txt：{base}")
        return 2
    labels = [line.strip() for line in classes_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not labels:
        print("[FAIL] classes.txt 为空")
        return 2
    human_zh = labels[0]
    herbs = {}
    attr = base / "attributes.csv"
    if attr.is_file():
        with attr.open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                herbs[row.get("image", "")] = row.get("herb", "")
    files = [p for p in sorted(images_dir.iterdir()) if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    if args.seed is not None:
        random.Random(args.seed).shuffle(files)
    existing = {case_key(c) for c in load_cases()}
    from PIL import Image
    added = skipped = 0
    for image in files:
        if args.limit and added >= args.limit:
            break
        label_file = labels_dir / (image.stem + ".txt")
        if not label_file.is_file():
            skipped += 1
            continue
        rows = [line.split() for line in label_file.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not rows:
            skipped += 1
            continue
        try:
            _cls, cx, cy, w, h = [float(v) for v in rows[0][:5]]
            width, height = Image.open(image).size
        except (ValueError, OSError):
            skipped += 1
            continue
        box = [(cx - w / 2) * width, (cy - h / 2) * height,
               (cx + w / 2) * width, (cy + h / 2) * height]
        case = validate_case({"image": str(image.resolve()), "image_sha256": _sha256(image),
                              "box": box, "human_label": human_zh, "split": args.split,
                              "note": f"导入自 {base.name}；herb={herbs.get(image.name, '')}",
                              "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S")})
        key = case_key(case)
        if key in existing:
            skipped += 1
            continue
        append_case(case)
        existing.add(key)
        added += 1
    print(f"[OK] 从 {base.name} 导入 {added} 条（跳过 {skipped}），split={args.split}，真值={human_zh}")
    return 0


def command_record(args):
    image = Path(args.image).expanduser()
    if not image.is_file():
        print(f"[FAIL] 图片不存在：{image}")
        return 2
    case = validate_case({"image": str(image.resolve()), "image_sha256": _sha256(image),
                          "box": args.box, "human_label": args.label,
                          "model_label": args.model_label, "split": args.split,
                          "note": args.note,
                          "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S")})
    key = case_key(case)
    for old in load_cases():
        if case_key(old) == key:
            print(f"[SKIP] 已有同图同框案例：{old['case_id']}（人工={old['human_label']}，模型={old['model_label']}）")
            return 0
    path = append_case(case)
    print(f"[OK] 已记录案例 {key[:12]} → {path}")
    return 0


def command_list(args):
    cases = load_cases(split=getattr(args, "split", "all") or "all")
    if not cases:
        print("案例库为空。先用 --record / --import-dataset 记录人工判定。")
        return 0
    counter = collections.Counter(c["human_label"] for c in cases)
    splits = collections.Counter(c["split"] for c in cases)
    print(f"案例 {len(cases)} 条；split={dict(splits)}；人工判定分布：{dict(counter)}")
    for case in cases:
        model = case.get("model_label") or "-"
        flag = "✓" if model == case["human_label"] else "✗"
        print(f"  {case['case_id']}  {case['split']:<4} {flag} 人工={case['human_label']:<6} "
              f"模型={model:<6} {Path(case['image']).name}")
    return 0


def _bucket_counts(rows):
    counts = collections.Counter()
    for row in rows:
        bucket = row.get("error_bucket") or ("correct" if row.get("ok") else "unknown")
        counts[(bucket, bool(row.get("stable", True)))] += 1
    return counts


def command_triage(args):
    """读 K 次重复报告，分别按“标签一致”和“标签+定位都一致”两种口径分诊。"""
    path = Path(args.report)
    if not path.is_file():
        print(f"[FAIL] 报告不存在：{path}")
        return 2
    report = json.loads(path.read_text(encoding="utf-8"))
    rows = enrich_rows(report.get("rows") or [])
    if not rows:
        print("[FAIL] 报告没有逐例结果")
        return 2
    counts = _bucket_counts(rows)
    stable_correct = counts[("correct", True)]
    print(f"报告：{path.name} | repeat={report.get('repeat')} | n={len(rows)} | "
          f"标签+定位都稳定且正确 {stable_correct}/{len(rows)}（{stable_correct / len(rows):.1%}）")
    for (bucket, stable), n in sorted(counts.items()):
        print(f"  {bucket:22s} {'稳定' if stable else '不稳定'}: {n}")
    label_counts = collections.Counter(row.get("label_bucket") or "unknown" for row in rows)
    print("\n按“标签是否一致”分诊（反射应看这个口径；框抖动不算类别不稳）：")
    for bucket, n in sorted(label_counts.items()):
        print(f"  {bucket:22s}: {n}")
    stable_errors = [r for r in rows if r.get("label_bucket") == "classification_error"]
    other_stable = [r for r in rows if r.get("label_bucket") in {"localization_error", "miss"}]
    unstable = [r for r in rows if r.get("label_bucket") == "label_unstable"]
    print(f"\n标签稳定的分类错：{len(stable_errors)} 条（只有这类能进 reflection）")
    for row in stable_errors:
        print(f"  {row['case_id']}  {Path(row['image']).name}  人工={row['human_label']} "
              f"预测={row.get('predicted') or '未检出'} 中位IoU={row.get('median_iou')}")
    if other_stable:
        print(f"标签稳定、但定位/漏检：{len(other_stable)} 条（应改定位/召回策略，不应调提示词）")
        for row in other_stable:
            print(f"  {row['case_id']}  {Path(row['image']).name}  {row.get('label_bucket')}")
    if unstable:
        print(f"标签不稳定：{len(unstable)} 条（应走重复投票/复核，不应调提示词）")
    threshold = int(args.min_stable_errors or 0)
    eligible = len(stable_errors) >= threshold
    print(f"\n反射资格：{'可用' if eligible else '不足'}（阈值 {threshold} 条标签稳定分类错）")
    if any(row.get("split") == "test" for row in rows):
        print("[注意] 报告含 test 案例；不应据此生成候选。")
        eligible = False
    if (report.get("repeat") or 1) < 3:
        print("[注意] 该报告不是 K≥3；单轮结果不能用于分诊。")
        eligible = False
    return 0 if eligible else 3


def _exact_mcnemar(better, worse):
    """配对二分类的精确双侧检验；样本小时比卡方可靠。"""
    n = better + worse
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(better, worse) + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def command_compare(args):
    """同一批案例上比较基线与候选：配对增减、错误桶迁移、精确 McNemar。"""
    paths = {"baseline": Path(args.baseline_report), "candidate": Path(args.candidate_report)}
    data = {}
    for name, path in paths.items():
        if not path.is_file():
            print(f"[FAIL] {name} 报告不存在：{path}")
            return 2
        data[name] = json.loads(path.read_text(encoding="utf-8"))
    left = {row["case_id"]: row for row in data["baseline"].get("rows", [])}
    right = {row["case_id"]: row for row in data["candidate"].get("rows", [])}
    common = sorted(set(left) & set(right))
    if not common:
        print("[FAIL] 两份报告没有共同案例，不能做配对比较")
        return 2
    improve = [cid for cid in common if not left[cid].get("ok") and right[cid].get("ok")]
    worsen = [cid for cid in common if left[cid].get("ok") and not right[cid].get("ok")]
    both = [cid for cid in common if left[cid].get("ok") and right[cid].get("ok")]
    neither = [cid for cid in common if not left[cid].get("ok") and not right[cid].get("ok")]
    transitions = collections.Counter(
        (left[cid].get("error_bucket") or "unknown", right[cid].get("error_bucket") or "unknown")
        for cid in common)
    p = _exact_mcnemar(len(improve), len(worsen))
    print(f"配对比较：{len(common)} 条共同案例（baseline vs candidate）")
    print(f"  改善 {len(improve)} | 变差 {len(worsen)} | 都对 {len(both)} | 都错 {len(neither)}")
    print(f"  精确 McNemar 双侧 p = {p:.4f}")
    for cid in improve[:10]:
        print(f"  [改善] {cid} {Path(left[cid]['image']).name} "
              f"{left[cid].get('error_bucket')} → {right[cid].get('error_bucket')}")
    for cid in worsen[:10]:
        print(f"  [变差] {cid} {Path(left[cid]['image']).name} "
              f"{left[cid].get('error_bucket')} → {right[cid].get('error_bucket')}")
    for (a, b), n in sorted(transitions.items()):
        if a != b:
            print(f"  迁移 {a} → {b}: {n}")
    out = Path(args.candidate_report).parent / "comparison.json"
    out.write_text(json.dumps({"schema": "detect-reflect-compare/v1", "n": len(common),
                               "improved": improve, "worsened": worsen, "both_correct": both,
                               "both_wrong": neither, "mcnemar_p": p,
                               "transitions": {f"{a}->{b}": n for (a, b), n in transitions.items()},
                               "baseline": str(paths["baseline"].resolve()),
                               "candidate": str(paths["candidate"].resolve())},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  对比报告：{out}")
    return 0 if len(improve) > len(worsen) else 3


def command_evaluate(args):
    cases = load_cases(split=getattr(args, "split", "all") or "all")
    if not cases:
        print("[FAIL] 案例库为空，先 --record / --import-dataset")
        return 2
    if args.prompt_file and not Path(args.prompt_file).is_file():
        print(f"[FAIL] 候选提示词不存在：{args.prompt_file}")
        return 2
    from prompts import PROMPT, PROMPT_VERSION
    if any(not Path(c["image"]).is_file() or _sha256(c["image"]) != c["image_sha256"] for c in cases):
        print("[FAIL] 案例图像缺失或哈希不符")
        return 2
    run_dir = root_dir() / "eval" / f"{args.name}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=False)
    prompt = Path(args.prompt_file).read_text(encoding="utf-8") if args.prompt_file else PROMPT
    (run_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
    (run_dir / "cases_snapshot.json").write_text(json.dumps(cases, ensure_ascii=False, indent=2), encoding="utf-8")
    started = time.time()
    rounds = []
    try:
        for i in range(args.repeat):
            print(f"重复轮次 {i+1}/{args.repeat}", flush=True)
            directory = run_dir / ("detection" if args.repeat == 1 else f"round_{i+1:02d}")
            grouped = run_detection(list(dict.fromkeys(c["image"] for c in cases)), directory,
                                    prompt_file=run_dir / "prompt.txt", python=args.python,
                                    timeout=args.timeout, temperature=args.temperature)
            round_result = score_cases(cases, grouped)
            rounds.append(round_result)
            (directory / "scores.json").write_text(json.dumps(round_result, ensure_ascii=False, indent=2), encoding="utf-8")
    except (RuntimeError, subprocess.TimeoutExpired) as exc:
        message = str(exc) if isinstance(exc, RuntimeError) else "检测轮次超时"
        (run_dir / "failure.json").write_text(json.dumps({"completed_rounds": len(rounds), "error": message},
                                                        ensure_ascii=False), encoding="utf-8")
        print(f"[FAIL] {message}")
        return 1
    rows = summarize_repeats(rounds)
    n, correct = len(rows), sum(row["ok"] for row in rows)
    result = {"rows": rows, "n": n, "correct": correct, "accuracy": round(correct / n, 4),
              "by_split": _by_split(rows), "confusion": dict(collections.Counter(
                  f"{row['human_label']}→{row['predicted'] or row['error_bucket']}" for row in rows))}
    report = {"schema": EVAL_SCHEMA, "name": args.name, "split": args.split,
              "prompt_file": str(Path(args.prompt_file).resolve()) if args.prompt_file else "builtin",
              "n": result["n"], "correct": result["correct"], "accuracy": result["accuracy"],
              "by_split": result["by_split"],
              "confusion": result["confusion"], "rows": result["rows"],
              "elapsed_s": round(time.time() - started, 2), "repeat": args.repeat,
              "temperature": args.temperature,
              "prompt_version": PROMPT_VERSION if not args.prompt_file else None,
              "prompt_sha256": _sha256(run_dir / "prompt.txt"),
              "stable": sum(row["stable"] for row in rows),
              "buckets": dict(collections.Counter(row["error_bucket"] for row in rows)),
              "sample_warning": "样本量<30，候选极易过拟合，不能据此发布" if result["n"] < 30 else None}
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"评测 {args.name}：{result['correct']}/{result['n']} 正确（{result['accuracy']:.1%}）")
    for split, value in result["by_split"].items():
        print(f"  [{split}] {value['correct']}/{value['n']}（{value['accuracy']:.1%}）")
    for pair, count in report["confusion"].items():
        print(f"  {pair}: {count}")
    for row in result["rows"]:
        if not row["ok"]:
            print(f"  [错] {Path(row['image']).name} 人工={row['human_label']} 模型={row['predicted'] or '未检出'} IoU={row['iou']}")
    print(f"报告：{run_dir / 'report.json'}")
    if report["sample_warning"]:
        print(f"[警告] {report['sample_warning']}")
    return 0 if result["accuracy"] >= (args.min_accuracy or 0) else 3


def class_labels():
    from ls_common import load_config
    return [c["zh"] for c in load_config()["yolo"]["classes"]]


def missing_classes(candidate, labels=None):
    """候选必须保留全部类别词条；少一个都可能让该类在下游永久检不出。"""
    labels = labels or class_labels()
    lines = candidate.splitlines()
    return [label for label in labels if not any(line.startswith(f"- {label}") for line in lines)]


def _baseline_prompt(path=""):
    if path:
        return Path(path).read_text(encoding="utf-8").strip()
    from prompts import PROMPT
    return PROMPT


def _crop_for_prompt(image, box):
    """裁出案例框（无框取整图），返回 BGR 数组；失败返回 None。"""
    import cv2
    from ls_common import imread_cn
    img = imread_cn(str(image))
    if img is None:
        return None
    if not box:
        return img
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    h, w = img.shape[:2]
    x1, y1 = max(0, min(x1, w - 1)), max(0, min(y1, h - 1))
    x2, y2 = max(x1 + 2, min(x2, w)), max(y1 + 2, min(y2, h))
    return img[y1:y2, x1:x2]


def build_proposal_prompt(current_prompt, failing, protected=None):
    lines = []
    for i, case in enumerate(failing[:8], 1):
        predicted = case.get("predicted") or case.get("model_label") or "未记录"
        lines.append(f"{i}. 人工判定：{case['human_label']}；模型判定：{predicted}；"
                     f"备注：{case.get('note') or '无'}")
    directions = collections.Counter(
        f"{case['human_label']}→{case.get('predicted') or case.get('model_label') or '未检出'}"
        for case in failing)
    direction_text = "；".join(f"{name} {n} 条" for name, n in directions.most_common())
    keep = ""
    if protected:
        rows = [f"{i}. 人工判定：{case['human_label']}；模型判定：{case.get('predicted') or case.get('model_label') or '未记录'}"
                for i, case in enumerate(protected[:8], 1)]
        keep = ("\n【这些案例当前判对，修订后必须仍然判对（尤其不得把导管改判成其他类）】\n"
                + "\n".join(rows) + "\n")
    return f"""你在改进一段中药显微图像的目标检测提示词。下面是【当前提示词全文】【人工复核后发现的错判案例】和【必须保持正确的案例】。

【错判方向统计】{direction_text}

请只输出**修订后的完整提示词文本**：不要解释、不要 Markdown 代码围栏、不要增减类别、不要改输出 JSON 格式。
修订纪律：
- 保留全部类别词条与“只输出 JSON”的格式要求；
- 不得针对具体文件名、药材名或单张图硬编码；
- 只允许调整判据表述（正例/反例/判别顺序），最多改 3 处；
- 新增判据必须能靠肉眼在图上判断，不能用“如果模型不确定”这类空话；
- **每条修订必须直接对应上面列出的某个错判方向，并在改写里针对它**；禁止用泛化的“提高召回/都必须框出/不得漏框”条款代替纠错；
- 错判方向中占多数的那个必须先修；只有 1 条案例的方向不要为它新增规则；
- **修正错判的同时不得削弱其他类别**：上面“必须保持正确”的案例若因本次修改而判错，候选作废；
- 尤其注意：把导管误判成别的类（或反之）与把纤维误判成导管同样严重，不要用“更爱说纤维”的方式修错。

【当前提示词全文】
{current_prompt}

【错判案例】
{chr(10).join(lines)}
{keep}"""


def command_propose(args):
    if getattr(args, "split", "dev") != "dev":
        print("[FAIL] 不允许在 test 集上生成候选：测试集只能用于最终复核，不能参与调参")
        return 2
    cases = load_cases(split=getattr(args, "split", "dev") or "dev")
    protected = []
    if args.from_report:
        report_path = Path(args.from_report)
        if not report_path.is_file():
            print(f"[FAIL] 评测报告不存在：{report_path}")
            return 2
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("split") != "dev" or report.get("repeat", 0) < 3:
            print("[FAIL] 提案只接受至少三轮的 dev 评测")
            return 2
        rows = enrich_rows(report.get("rows", []))
        failed_ids = {row["case_id"] for row in rows if row.get("label_bucket") == "classification_error"}
        predicted = {row["case_id"]: row.get("predicted") for row in rows}
        failing = [{**c, "predicted": predicted.get(c["case_id"])}
                   for c in cases if c["case_id"] in failed_ids]
        # 只取 dev 的判对案例做“不得改坏”的护栏；test 标签绝不进入候选生成。
        protected = [{"human_label": row["human_label"], "predicted": row.get("predicted")}
                     for row in rows
                     if row.get("ok") and row.get("split") == "dev"][:8]
    else:
        failing = [c for c in cases if c.get("model_label") and c["model_label"] != c["human_label"]]
    if not failing:
        print("[FAIL] 没有可用失败案例（需要基线评测的错例，或记录 model_label 与 human_label 不一致的案例）")
        return 2
    current = _baseline_prompt(args.base_prompt)
    meta = build_proposal_prompt(current, failing, protected)
    from ls_common import load_config, resolve_key
    from vlm_api import call_vlm, encode_jpeg_b64
    cfg = load_config()
    key = resolve_key(cfg)
    sample = failing[0]
    crop = _crop_for_prompt(sample["image"], sample.get("box"))
    if crop is None:
        print("[FAIL] 无法读取失败案例图片")
        return 1
    b64, _ = encode_jpeg_b64(crop, max_side=1024, quality=88)
    text, err = call_vlm(cfg["vlm"], key, b64, meta)
    if not text:
        print(f"[FAIL] 候选生成失败：{err}")
        return 1
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = candidate.strip("`")
        candidate = candidate.split("\n", 1)[1] if "\n" in candidate else candidate
    if "只输出 JSON" not in candidate or "导管" not in candidate:
        print("[FAIL] 候选缺少必要的类别/输出约定，拒绝落盘（避免把提示词改坏）")
        return 1
    missing = missing_classes(candidate)
    if missing:
        print(f"[FAIL] 候选丢失类别 {missing}；拒绝落盘（少一个类会让该类永久检不出）")
        return 1
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(candidate, encoding="utf-8")
    print(f"[OK] 候选提示词 → {out}")
    print(f"     失败案例 {len(failing)} 条；当前提示词 {len(current)} 字符 → 候选 {len(candidate)} 字符")
    print("     下一步：--evaluate --name <名字> --prompt-file " + str(out))
    if len(cases) < 30:
        print(f"[警告] 案例仅 {len(cases)} 条：该候选必然过拟合，只用于验证机制，不能发布")
    return 0


def command_accept(args):
    source = Path(args.candidate)
    if not source.is_file():
        print(f"[FAIL] 候选不存在：{source}")
        return 2
    target = root_dir() / "accepted" / source.name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"[OK] 已留档 → {target}")
    print("     上线仍需：① 人工确认评测报告；② 合并进 目标检测/lib/prompts.py 并 bump PROMPT_VERSION；"
          "③ 跑 crop_gold/det_eval 全量回归。本命令不改生产提示词。")
    return 0


def selftest():
    """离线：案例校验/去重/匹配/汇总，不发网络请求、不写用户目录。"""
    box = [10, 10, 50, 50]
    case = validate_case({"image": "a.jpg", "image_sha256": "x", "box": box,
                          "human_label": "纤维", "model_label": "导管"})
    assert case["box"] == [10.0, 10.0, 50.0, 50.0]
    assert case_key(case) == case_key({**case, "box": "10,10,50,50"})
    for bad in ({"image": "a.jpg"}, {"human_label": "纤维"},
                {"image": "a.jpg", "human_label": "纤维", "split": "prod"}):
        try:
            validate_case(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"应拒绝非法案例 {bad}")
    assert validate_case({"image": "a.jpg", "human_label": "纤维", "split": "test"})["split"] == "test"
    dets = [{"object_label": "导管", "tight_box_source": [9, 9, 51, 51], "score": 0.92},
            {"object_label": "纤维", "tight_box_source": [200, 200, 300, 300], "score": 0.65}]
    best, iou = match_detection(dets, case)
    assert best["object_label"] == "导管" and iou > 0.9, (best, iou)
    result = score_cases([{**case, "case_id": "k1"}], {"a.jpg": dets})
    assert result["n"] == 1 and result["correct"] == 0
    assert result["confusion"] == {"纤维→导管": 1}
    assert result["by_split"]["dev"]["n"] == 1
    # 无框案例取最高分框；未检出必须计错而不是跳过
    no_box = {**case, "box": None}
    assert match_detection(dets, no_box)[0]["object_label"] == "导管"
    empty = score_cases([{**case, "case_id": "k2"}], {"a.jpg": []})
    assert empty["rows"][0]["predicted"] == "" and empty["rows"][0]["ok"] is False
    assert empty["confusion"] == {"纤维→未检出": 1}
    shifted = score_cases([case], {"a.jpg": dets[1:]})
    assert shifted["correct"] == 0 and shifted["rows"][0]["error_bucket"] == "localization_error"
    assert summarize_repeats([result] * 3)[0]["error_bucket"] == "classification_error"
    missed_same_case = score_cases([{**case, "case_id": "k1"}], {})
    assert summarize_repeats([result, result, missed_same_case])[0]["error_bucket"] == "unstable"
    prompt = build_proposal_prompt("【当前】", [case])
    assert "纤维" in prompt and "导管" in prompt and "只输出" in prompt
    guarded = build_proposal_prompt("【当前】", [case], [{"human_label": "导管", "predicted": "导管"}])
    assert "必须保持正确" in guarded and "不得把导管改判" in guarded
    assert missing_classes("- 纤维：x\n- 导管：y", ["纤维", "导管"]) == []
    assert missing_classes("- 纤维：x", ["纤维", "导管", "表皮细胞"]) == ["导管", "表皮细胞"]
    counts = _bucket_counts([{"ok": True, "error_bucket": "correct", "stable": True},
                             {"ok": False, "error_bucket": "unstable", "stable": False}])
    assert counts[("correct", True)] == 1 and counts[("unstable", False)] == 1
    assert _exact_mcnemar(0, 0) == 1.0
    assert _exact_mcnemar(6, 0) < 0.05 and _exact_mcnemar(3, 3) > 0.5
    print("REFLECT-DETECT SELFTEST: ALL ASSERTIONS PASS")
    return 0


def build_parser():
    ap = argparse.ArgumentParser(description="检测提示词的案例→候选→盲评闭环（离线）")
    ap.add_argument("--record", action="store_true", help="记录一条人工判定案例")
    ap.add_argument("--import-dataset", default="", help="从 YOLO 特征目录批量导入案例（classes.txt 为真值）")
    ap.add_argument("--list", action="store_true", help="查看案例库")
    ap.add_argument("--evaluate", action="store_true", help="用给定提示词重跑案例并比对")
    ap.add_argument("--propose", action="store_true", help="让模型生成候选提示词")
    ap.add_argument("--triage", action="store_true", help="分诊重复评测报告：稳定错/不稳定/能否进反射")
    ap.add_argument("--compare", action="store_true", help="同一批案例上配对比较基线与候选")
    ap.add_argument("--accept", action="store_true", help="留档已通过评测的候选")
    ap.add_argument("--selftest", action="store_true", help="离线自检")
    ap.add_argument("--image", default="", help="--record：图片路径")
    ap.add_argument("--label", default="", help="--record：人工判定（如 纤维）")
    ap.add_argument("--model-label", default="", help="--record：当时的模型判定")
    ap.add_argument("--box", default="", help="--record：x1,y1,x2,y2（原图坐标，可空）")
    ap.add_argument("--note", default="", help="--record：形态备注")
    ap.add_argument("--split", default="dev", choices=["dev", "test", "all"], help="案例切分/评测范围（默认 dev）")
    ap.add_argument("--cases", default="", help="案例库文件名/绝对路径（默认 cases.jsonl；legacy 与正式集分开）")
    ap.add_argument("--import-manifest", default="", help="按已审计的分组草案导入案例（保留其 split/box/真值）")
    ap.add_argument("--seed", type=int, default=0, help="--import-dataset：抽样随机种子（可复现）")
    ap.add_argument("--limit", type=int, default=0, help="--import-dataset：最多导入 N 条（0=全部）")
    ap.add_argument("--name", default="baseline", help="--evaluate：本次评测名")
    ap.add_argument("--report", default="", help="--triage：重复评测报告路径")
    ap.add_argument("--baseline-report", default="", help="--compare：基线报告")
    ap.add_argument("--candidate-report", default="", help="--compare：候选报告")
    ap.add_argument("--min-stable-errors", type=int, default=5, help="--triage：进入反射所需稳定分类错条数")
    ap.add_argument("--prompt-file", default="", help="--evaluate：候选提示词文件")
    ap.add_argument("--from-report", default="", help="--propose：从该基线评测报告里取错例")
    ap.add_argument("--base-prompt", default="", help="--propose：以该提示词为基线（默认内置 PROMPT）")
    ap.add_argument("--out", default="", help="--propose：候选输出路径")
    ap.add_argument("--candidate", default="", help="--accept：候选文件")
    ap.add_argument("--python", default="", help="--evaluate：检测解释器（默认 .venv/vision）")
    ap.add_argument("--timeout", type=int, default=1800, help="--evaluate：整批检测超时秒数")
    ap.add_argument("--repeat", type=int, default=1, choices=range(1, 11), help="独立重复轮次，最多10")
    ap.add_argument("--temperature", type=float, default=None,
                    help="--evaluate：覆盖检测温度（实验用；默认沿用 vlm_config）")
    ap.add_argument("--min-accuracy", type=float, default=0.0, help="低于该准确率返回非零")
    return ap


def main(argv=None):
    global CASE_FILE
    args = build_parser().parse_args(argv)
    if getattr(args, "cases", ""):
        CASE_FILE = args.cases
    actions = [args.record, bool(args.import_dataset), bool(args.import_manifest), args.list,
               args.evaluate, args.propose, args.triage, args.compare, args.accept, args.selftest]
    if sum(bool(a) for a in actions) != 1:
        build_parser().print_help()
        return 2
    if args.selftest:
        return selftest()
    if args.triage:
        if not args.report:
            print("[FAIL] --triage 需要 --report")
            return 2
        return command_triage(args)
    if args.compare:
        if not args.baseline_report or not args.candidate_report:
            print("[FAIL] --compare 需要 --baseline-report 与 --candidate-report")
            return 2
        return command_compare(args)
    if args.import_manifest:
        return command_import_manifest(args)
    if args.import_dataset:
        return command_import(args)
    if args.record:
        if not args.image or not args.label:
            print("[FAIL] --record 需要 --image 与 --label")
            return 2
        return command_record(args)
    if args.list:
        return command_list(args)
    if args.evaluate:
        return command_evaluate(args)
    if args.propose:
        if not args.out:
            print("[FAIL] --propose 需要 --out")
            return 2
        return command_propose(args)
    return command_accept(args)


if __name__ == "__main__":
    raise SystemExit(main())
