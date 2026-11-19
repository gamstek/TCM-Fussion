# -*- coding: utf-8 -*-
"""train_det.py — YOLO 检测基线训练入口（P1，2026-11-03）

**为什么存在**：P0（`lib/det_dataset.py` + `cli/build_det_dataset.py`）只负责把 14 个
单类导出集合并成可信的多类训练集；本入口按 `docs/目标检测_文献调研与实现方案_20261030.md`
§6.3 训练 **YOLOv8s 基线**（imgsz=1024、100 epochs、early stopping），产物落
`results/detection/yolo_runs/<name>/`（gitignored）。

**两条与本机相关的解析规则**（P0 产物按设计不改写）：
1. `data.yaml` 的 `path` 记录的是原 Windows 根（`d:/dataset/...`）。本入口在运行前生成一份
   本机解析后的 `data_local.yaml`（仅覆盖 `path`），P0 的 `data.yaml` 保持原样；
2. `train` 仍指向 P0 的 `train_list.txt`（稀有类 train-only 重采样已经写在该清单的重复行里），
   不退回 `images/train`，否则会静默丢掉重采样。

**分层**：本文件是入口（不进 import 注入点、不被任何库 import）；训练/评测逻辑全部转调
Ultralytics。配置在 `目标检测/config/detector.json`，数据合并仍是 P0 的 `build_det_dataset.py`。

用法（项目根执行；本机 venv 为 Linux 布局 `.venv/bin/python`）:
  python 目标检测/cli/train_det.py --dry-run                  # 只校验数据与参数，不训练
  python 目标检测/cli/train_det.py                            # 按 detector.json 训练
  python 目标检测/cli/train_det.py --epochs 120 --batch 16
  python 目标检测/cli/train_det.py --eval-only --split test    # 训练后在同一把尺子上评测
"""
import argparse
import json
import os
import sys

import paths

DEFAULT_CONFIG = os.path.join(paths.DETECTION_CONFIG_DIR, "detector.json")
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


def _load_cfg(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _deep_merge(base, over):
    """浅层递归合并：`over` 只写差异，未写的键保持 `base` 的值（避免整份配置复制后漂移）。"""
    out = dict(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _project_path(p):
    """配置里的相对路径一律相对项目根（与 AGENTS.md 的路径规则一致）。"""
    return p if os.path.isabs(p) else os.path.join(paths.PROJECT_ROOT, p)


def _resolve_data(data_yaml, out_dir):
    """读 P0 `data.yaml` → 生成 `data_local.yaml`（覆盖 path，不改 P0 产物）。

    返回 (本地 yaml 路径, 解析后的 dict, 数据集根)。`train` 若指向不存在的清单则报错，
    **不静默退回 images/train** —— 退回会丢掉 P0 的稀有类重采样。
    """
    import yaml

    with open(data_yaml, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    root = os.path.dirname(os.path.abspath(data_yaml))
    data["path"] = root
    data["yaml_file"] = os.path.abspath(data_yaml)   # 缓存新鲜度比对用
    # 类别名含中文时 Ultralytics 会去下载 Arial.Unicode.ttf（22MB，本机实测 ~10KB/s，
    # 会把训练启动卡住数十分钟；容器里也没有可用 CJK 字体）。训练/评估只依赖**类别 id**，
    # 故本机解析副本改用 ASCII 占位名，并把原中文名保留在 `names_zh` 供评测报告回映射；
    # P0 的 data.yaml / classes.txt 不改。
    names = data.get("names") or {}
    if isinstance(names, dict) and any(not str(v).isascii() for v in names.values()):
        data["names_zh"] = dict(names)
        data["names"] = {k: "class_{}".format(k) for k in names}
    for key in ("train", "val", "test"):
        val = data.get(key)
        if not isinstance(val, str) or not val:
            continue
        if os.path.isabs(val):
            data[key] = val
            continue
        target = os.path.join(root, val)
        if not os.path.exists(target):
            raise FileNotFoundError(
                "data.yaml 的 {} 指向不存在的路径：{}（data.yaml={}）".format(key, target, data_yaml))
        data[key] = val
    # 落盘副本不写绝对路径：去掉 path / yaml_file，train/val/test 改成本地副本所在目录
    # 的相对路径（Ultralytics 在 path 缺失时以 yaml 自身目录为根，见 check_det_dataset）。
    # 内存里的 data 仍保留绝对 root，供 _validate / _build_cache 解析。
    out_data = dict(data)
    out_data.pop("path", None)
    out_data.pop("yaml_file", None)
    for key in ("train", "val", "test"):
        val = out_data.get(key)
        if isinstance(val, str) and val:
            abs_val = val if os.path.isabs(val) else os.path.join(root, val)
            out_data[key] = os.path.relpath(abs_val, out_dir)
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "data_local.yaml")
    with open(out, "w", encoding="utf-8") as f:
        yaml.safe_dump(out_data, f, allow_unicode=True, sort_keys=False)
    return out, data, root


def _resolve_list(path):
    """按 Ultralytics 的 txt 清单规则解析图像路径：`./` 开头的相对**清单文件所在目录**。"""
    parent = os.path.dirname(os.path.abspath(path))
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            x = line.strip()
            if not x:
                continue
            if x.startswith("./"):
                x = os.path.join(parent, x[2:])
            out.append(os.path.abspath(x))
    return out


def _images_of(entry, root):
    """train/val/test 条目 → 图像绝对路径列表（目录或 txt）。"""
    path = entry if os.path.isabs(entry) else os.path.join(root, entry)
    if os.path.isdir(path):
        return sorted(os.path.join(path, f) for f in os.listdir(path)
                      if f.lower().endswith(IMG_EXTS))
    if path.lower().endswith(".txt"):
        return _resolve_list(path)
    return [path]


def _label_of(img_path):
    """YOLO 标签路径：`images/<split>/x.jpg` → `labels/<split>/x.txt`。"""
    parts = os.path.normpath(img_path).split(os.sep)
    for i, p in enumerate(parts):
        if p == "images":
            parts[i] = "labels"
            break
    else:
        return ""
    return os.path.splitext(os.sep.join(parts))[0] + ".txt"


def _cache_one(job):
    """子进程：读一张原图 → 等比缩到 max_side → JPEG q90 写入缓存。

    大图（2,048×1,536 / 3,840×2,160 TIFF）解码是 DataLoader 的瓶颈；预缩到 1,280
    （仍大于训练 imgsz=1,024）可显著减少每轮解码成本，且不改变训练分辨率的口径。
    """
    src, dst, max_side = job
    import cv2
    import numpy as np
    try:
        img = cv2.imdecode(np.fromfile(src, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return src, False, "imdecode 返回 None"
        h, w = img.shape[:2]
        s = min(1.0, float(max_side) / max(h, w))
        if s < 1.0:
            img = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))),
                             interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        if not ok:
            return src, False, "imencode 失败"
        with open(dst, "wb") as f:
            f.write(buf.tobytes())
        return src, True, ""
    except Exception as e:                       # noqa: BLE001
        return src, False, str(e)


def _cache_ready(cache_dir, max_side, source):
    meta = os.path.join(cache_dir, "cache_meta.json")
    yaml_path = os.path.join(cache_dir, "data_cache.yaml")
    if not (os.path.exists(meta) and os.path.exists(yaml_path)):
        return False
    try:
        with open(meta, encoding="utf-8") as f:
            m = json.load(f)
    except (OSError, ValueError):
        return False
    return int(m.get("max_side", 0)) == int(max_side) and m.get("source") == source


def _build_cache(data, root, cache_dir, max_side, workers):
    """把三分集图像预缩成 JPEG 缓存（`cache_dir/images/<split>/`），标签复制到
    `cache_dir/labels/<split>/`，并生成 `data_cache.yaml` 与 `train_list.txt`。

    只做**尺寸压缩**，不改图像内容、不改标签、不改划分；P0 数据集保持原样。
    `train_list.txt` 保留原清单里的重复行（稀有类 train-only 重采样）。
    """
    import shutil
    from concurrent.futures import ProcessPoolExecutor

    cache_dir = os.path.abspath(cache_dir)
    for sub in ("images", "labels"):
        for split in ("train", "val", "test"):
            os.makedirs(os.path.join(cache_dir, sub, split), exist_ok=True)

    entries, jobs, mapping = {}, [], {}
    for split in ("train", "val", "test"):
        entry = data.get(split)
        if not entry:
            continue
        imgs = _images_of(entry, root)
        unique = list(dict.fromkeys(imgs))
        entries[split] = (entry, imgs, unique)
        for src in unique:
            stem = os.path.splitext(os.path.basename(src))[0]
            dst = os.path.join(cache_dir, "images", split, stem + ".jpg")
            mapping[src] = dst
            jobs.append((src, dst, max_side))
            lab = _label_of(src)
            if lab and os.path.exists(lab):
                shutil.copyfile(lab, os.path.join(cache_dir, "labels", split, stem + ".txt"))

    print("[info] 构建图像缓存：{} 张 → {}（max_side={}）".format(
        len(jobs), cache_dir, max_side))
    failed = []
    with ProcessPoolExecutor(max_workers=max(1, int(workers))) as ex:
        for i, (src, ok, err) in enumerate(ex.map(_cache_one, jobs, chunksize=8), 1):
            if not ok:
                failed.append((src, err))
            if i % 1000 == 0:
                print("  {}/{}".format(i, len(jobs)))
    if failed:
        raise RuntimeError("缓存构建失败 {} 张，例如：{} -> {}".format(
            len(failed), failed[0][0], failed[0][1]))

    # 训练清单（保留重复行 = 保留 P0 的稀有类重采样）
    train_entry = entries.get("train")
    if train_entry:
        entry = train_entry[0]
        orig = entry if os.path.isabs(entry) else os.path.join(root, entry)
        srcs = _resolve_list(orig) if orig.lower().endswith(".txt") else train_entry[1]
        with open(os.path.join(cache_dir, "train_list.txt"), "w", encoding="utf-8") as f:
            for src in srcs:
                dst = mapping.get(os.path.abspath(src))
                if not dst:
                    raise RuntimeError("训练清单里的图像未进缓存：{}".format(src))
                f.write("./images/train/{}\n".format(os.path.basename(dst)))

    names = data.get("names") or {}
    names_zh = data.get("names_zh") or names
    names_ascii = ({k: "class_{}".format(k) for k in names}
                   if any(not str(v).isascii() for v in names.values()) else names)
    # 不写 path：Ultralytics 以 data_cache.yaml 自身目录（即 cache_dir）为根，
    # 下列条目与 train_list.txt 里的 "./images/train/..." 都是相对路径。
    cache_data = {
        "train": "train_list.txt" if train_entry else "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": names_ascii,
    }
    if any(not str(v).isascii() for v in names_zh.values()):
        cache_data["names_zh"] = dict(names_zh)
    import yaml
    yaml_path = os.path.join(cache_dir, "data_cache.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cache_data, f, allow_unicode=True, sort_keys=False)
    # 缓存新鲜度只记项目相对路径，避免把机器路径写进产物。
    src = data.get("yaml_file") or data.get("path") or ""
    src = os.path.relpath(os.path.abspath(src), paths.PROJECT_ROOT) if src else ""
    with open(os.path.join(cache_dir, "cache_meta.json"), "w", encoding="utf-8") as f:
        json.dump({"source": src,
                   "max_side": int(max_side),
                   "counts": {k: len(v[2]) for k, v in entries.items()}},
                  f, ensure_ascii=False, indent=2)
    return yaml_path


def _build_repeat_data(data_yaml, repeat_map, out_dir, names_src=None):
    """按类提高 train 清单的重复次数（稀有类增强臂），不改 P0 数据集、不改缓存原文件。

    `repeat_map` 是 {中文类名: 倍数}。只重写 train 清单（绝对路径指向缓存图）；
    val/test 原样保留。写出的 `data_repeat.yaml` 落在本次产物目录，避免影响基线复用同一份缓存。
    `names_src` 用于取中文类名（缓存 yaml 可能只有 ASCII 占位名；优先 P0 data.yaml）。
    """
    import yaml

    with open(data_yaml, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    root = data.get("path") or os.path.dirname(os.path.abspath(data_yaml))
    names_zh = {}
    if names_src and os.path.exists(names_src):
        with open(names_src, encoding="utf-8") as f:
            src = yaml.safe_load(f) or {}
        names_zh = src.get("names_zh") or src.get("names") or {}
    if not names_zh:
        names_zh = data.get("names_zh") or data.get("names") or {}
    zh_to_id = {str(v): int(k) for k, v in names_zh.items()}
    unknown = sorted(set(repeat_map) - set(zh_to_id))
    if unknown:
        raise ValueError("repeat_rare 的类别不在数据集里：{}（可选：{}）".format(
            unknown, sorted(zh_to_id)))
    reps = {zh_to_id[zh]: max(1, int(m)) for zh, m in repeat_map.items()}

    entry = data.get("train")
    if not isinstance(entry, str) or not entry:
        raise ValueError("data.yaml 没有 train 条目，无法重采样")
    target = entry if os.path.isabs(entry) else os.path.join(root, entry)
    imgs = _resolve_list(target) if target.lower().endswith(".txt") else _images_of(entry, root)

    from collections import Counter

    # 源清单已带 P0 的 train-only 重采样（气孔×3/花粉粒×2/菌丝×2/导管×3/淀粉粒×3）。
    # 这里先还原成“唯一图 + 原有倍数”，再只把 repeat_rare 指定的类**绝对倍数**覆盖，
    # 其余类保留 P0 原值；否则重复行会叠乘（如 3×6 = 18×）。
    base = Counter(imgs)
    lines, changed = [], 0
    for img in dict.fromkeys(imgs):
        mult = base[img]
        lab = _label_of(img)
        if lab and os.path.exists(lab):
            with open(lab, encoding="utf-8") as f:
                cls = {int(line.split()[0]) for line in f if line.strip()}
            mult = max([mult] + [reps[c] for c in cls if c in reps])
        if mult > base[img]:
            changed += 1
        lines += [img] * mult

    os.makedirs(out_dir, exist_ok=True)
    list_path = os.path.join(out_dir, "train_repeat.txt")
    # 清单与 yaml 都写相对路径：清单行以 "./" 开头（Ultralytics 相对清单所在目录解析），
    # yaml 不写 path（以 yaml 自身目录为根）。
    rel_lines = ["./" + os.path.relpath(x, out_dir).replace(os.sep, "/") for x in lines]
    with open(list_path, "w", encoding="utf-8") as f:
        f.write("\n".join(rel_lines) + "\n")

    out = dict(data)
    out.pop("path", None)
    out.pop("yaml_file", None)
    for key in ("train", "val", "test"):
        val = out.get(key)
        if isinstance(val, str) and val:
            abs_val = val if os.path.isabs(val) else os.path.join(root, val)
            out[key] = os.path.relpath(abs_val, out_dir).replace(os.sep, "/")
    out["train"] = "train_repeat.txt"
    yaml_path = os.path.join(out_dir, "data_repeat.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(out, f, allow_unicode=True, sort_keys=False)
    print("[info] 稀有类重采样：{} → 清单 {} 行（涉及 {} 张图；原 {} 行）".format(
        repeat_map, len(lines), changed, len(imgs)))
    return yaml_path


def _validate(data, root):
    """离线校验：三分集图像数、标签配对、类别 id 上界、框数；异常即抛。"""
    names = data.get("names") or {}
    n_cls = len(names)
    stats = {}
    for split in ("train", "val", "test"):
        entry = data.get(split)
        if not entry:
            continue
        imgs = _images_of(entry, root)
        missing, boxes = [], 0
        for img in imgs:
            lab = _label_of(img)
            if not lab or not os.path.exists(lab):
                missing.append(img)
                continue
            with open(lab, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    cid = int(line.split()[0])
                    if cid < 0 or cid >= n_cls:
                        raise ValueError("{} 的类别 id {} 超出 [0,{})".format(lab, cid, n_cls))
                    boxes += 1
        if missing:
            raise FileNotFoundError(
                "{} 有 {} 张图缺标签，例如：{}".format(split, len(missing), missing[0]))
        if not imgs:
            raise FileNotFoundError("{} 未解析到任何图像（entry={}）".format(split, entry))
        stats[split] = {"images": len(imgs), "boxes": boxes}
    return stats, n_cls


def _to_jsonable(obj):
    """Ultralytics 指标里含 numpy/Path/枚举，统一转 JSON 可写类型。"""
    import numpy as np
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def _train(cfg, data_yaml, args):
    # 容器 /dev/shm 过小（实测 64MB）时，DataLoader 多进程按默认 'file_descriptor'
    # 共享 batch 会报 `unable to allocate shared memory(shm) ... No space left on device`，
    # 且 PyTorch 只是丢掉该 worker 的 batch（不终止训练）→ 会**静默漏样本**。
    # 这里只在 shm < 1GB 时退回 file_system 策略；≥1GB 时保持默认（吞吐更好）。
    import torch
    import torch.multiprocessing as mp
    try:
        st = os.statvfs("/dev/shm")
        if st.f_blocks * st.f_frsize < (1 << 30):
            mp.set_sharing_strategy("file_system")
    except OSError:
        pass
    torch.backends.cudnn.benchmark = True   # 固定输入尺寸 1024，启用 cudnn 自动选优

    from ultralytics import YOLO

    model_path = _project_path(cfg["model"])
    project = _project_path(cfg["project"])
    if args.resume:
        # 续训必须从 last.pt 载入（含 optimizer/epoch）；从预训练权重 resume 无效。
        # 其余参数（data/imgsz/epochs/...) 由 checkpoint 自带的 args 恢复，不在此覆盖。
        model_path = os.path.join(project, args.name, "weights", "last.pt")
        if not os.path.exists(model_path):
            raise FileNotFoundError("续训权重不存在：{}".format(model_path))
        print("[info] 从断点续训：{}".format(model_path))
        model = YOLO(model_path)
        # 续训参数由 checkpoint 恢复；workers / save_dir 允许覆盖（Ultralytics check_resume 白名单）：
        # workers 供 /dev/shm 过小的容器用 0 规避共享内存不足；save_dir 用项目根拼出的绝对路径，
        # 避免 Ultralytics 把 checkpoint 里的相对 project 前缀成 runs/detect/...。
        model.train(resume=True, workers=args.workers,
                    save_dir=os.path.join(project, args.name))
        save_dir = str(getattr(model.trainer, "save_dir", ""))
        print("[done] 训练产物目录：{}".format(save_dir))
        return 0

    model = YOLO(model_path)
    kwargs = {
        "data": data_yaml,
        "imgsz": args.imgsz,
        "epochs": args.epochs,
        "batch": args.batch,
        "patience": args.patience,
        "workers": args.workers,
        "device": args.device,
        "seed": args.seed,
        "pretrained": cfg.get("pretrained", True),
        "single_cls": bool(args.single_cls or cfg.get("single_cls", False)),
        "close_mosaic": cfg.get("close_mosaic", 10),
        "cache": cfg.get("cache", False),
        "plots": cfg.get("plots", True),
        "project": project,
        "name": args.name,
        "exist_ok": args.exist_ok,
        "resume": args.resume,
        "amp": True,
        "val": True,
        "save": True,
    }
    kwargs.update(cfg.get("augment") or {})   # 增强臂：只覆盖显式给出的增强键
    print("[info] Ultralytics train 参数：")
    for k, v in kwargs.items():
        print("  {:<10} {}".format(k, v))
    model.train(**kwargs)
    save_dir = str(getattr(model.trainer, "save_dir", ""))
    print("[done] 训练产物目录：{}".format(save_dir))
    print("       最优权重：{}".format(os.path.join(save_dir, "weights", "best.pt")))
    return 0


def _eval(cfg, data_yaml, args):
    from ultralytics import YOLO

    weights = args.weights or os.path.join(
        _project_path(cfg["project"]), args.name, "weights", "best.pt")
    if not os.path.exists(weights):
        print("[FAIL] 权重不存在：{}（先训练或 --weights 指定）".format(weights), file=sys.stderr)
        return 2
    model = YOLO(weights)
    metrics = model.val(data=data_yaml, split=args.split, imgsz=args.imgsz,
                        batch=args.batch, workers=args.workers, device=args.device,
                        single_cls=bool(args.single_cls or cfg.get("single_cls", False)),
                        project=_project_path(cfg["project"]), name="{}_eval_{}".format(args.name, args.split),
                        exist_ok=True, plots=cfg.get("plots", True))
    # 评测报告只记项目相对路径，避免把机器路径写进可复现产物。
    weights_rel = os.path.relpath(os.path.abspath(weights), paths.PROJECT_ROOT).replace(os.sep, "/")
    data_rel = os.path.relpath(os.path.abspath(data_yaml), paths.PROJECT_ROOT).replace(os.sep, "/")
    out = {"weights": weights_rel, "split": args.split, "data": data_rel,
           "results_dict": _to_jsonable(metrics.results_dict),
           "per_class": _to_jsonable(getattr(metrics.box, "maps", None))}
    # 逐类指标（P1 验收需要逐类 AP50/AP50-95；Ultralytics 的字段名随版本略有差异，
    # 故按"存在才写"收集，不硬编码单一版本）。
    for key in ("p", "r", "ap50", "ap", "maps", "ap_class_index"):
        val = getattr(metrics.box, key, None)
        if val is not None:
            out.setdefault("box_metrics", {})[key] = _to_jsonable(val)
    cm = getattr(metrics, "confusion_matrix", None)
    if cm is not None and getattr(cm, "matrix", None) is not None:
        out["confusion_matrix"] = _to_jsonable(cm.matrix)
    import yaml
    # 中文类别名优先从**原始 P0 data.yaml**读取（缓存 yaml 的 names 可能是 ASCII 占位名）
    names_src = cfg["data"] if os.path.exists(cfg["data"]) else data_yaml
    with open(names_src, encoding="utf-8") as f:
        local = yaml.safe_load(f) or {}
    out["class_names"] = local.get("names") or {}
    out["class_names_zh"] = local.get("names_zh") or local.get("names") or {}
    report = os.path.join(_project_path(cfg["project"]),
                          "{}_eval_{}".format(args.name, args.split), "eval_report.json")
    os.makedirs(os.path.dirname(report), exist_ok=True)
    with open(report, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("[done] {} 评测：{}".format(args.split, report))
    print("       mAP50-95={:.4f} mAP50={:.4f}".format(
        out["results_dict"].get("metrics/mAP50-95(B)", float("nan")),
        out["results_dict"].get("metrics/mAP50(B)", float("nan"))))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description="YOLO 检测基线训练（P1，项目根执行）")
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--overlay", default=None,
                    help="叠加配置（只写差异键；相对项目根），例如 config/detector_aug.json")
    ap.add_argument("--data", default=None, help="覆盖 detector.json 的 data（P0 data.yaml）")
    ap.add_argument("--model", default=None, help="覆盖预训练权重（默认 detector.json 的 model）")
    ap.add_argument("--name", default=None, help="训练产物名（默认 detector.json 的 name）")
    ap.add_argument("--imgsz", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--patience", type=int, default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--single-cls", action="store_true",
                    help="把 14 类标签在训练内部折叠为 1 类（stage 1 无类别提框实验；数据/划分不变）")
    ap.add_argument("--resume", action="store_true", help="从上次中断处续训")
    ap.add_argument("--exist-ok", action="store_true", help="允许覆盖同名产物目录")
    ap.add_argument("--dry-run", action="store_true", help="只校验数据与参数，不训练")
    ap.add_argument("--eval-only", action="store_true", help="不训练，直接评测已有权重")
    ap.add_argument("--weights", default=None, help="评测权重（默认 <name>/weights/best.pt）")
    ap.add_argument("--split", choices=["train", "val", "test"], default="val", help="评测切分")
    ap.add_argument("--rebuild-cache", action="store_true",
                    help="强制重建预缩图像缓存（默认仅在缓存缺失/参数变化时重建）")
    ap.add_argument("--no-cache", action="store_true", help="不使用预缩缓存，直接读原图")
    args = ap.parse_args(argv)

    cfg = _load_cfg(args.config)
    if args.overlay:
        cfg = _deep_merge(cfg, _load_cfg(_project_path(args.overlay)))
    # 只有 data/model/project 是路径；name 是产物子目录名，**不能**转成绝对路径，
    # 否则 Ultralytics 会把它当作 save_dir 本身、绕过 project（实测：产物落到项目根）。
    for key in ("data", "model", "project"):
        if key in cfg:
            cfg[key] = _project_path(cfg[key])
    if args.data:
        cfg["data"] = _project_path(args.data)
    if args.model:
        cfg["model"] = _project_path(args.model)
    if args.name:
        cfg["name"] = args.name
    # 命令行优先；未给则把配置值回填到 args（Ultralytics 不接受 None，
    # 且下游 _train/_eval 统一从 args 取值，避免"配置有值但传进去是 None"）。
    for key in ("imgsz", "epochs", "batch", "patience", "workers", "device", "seed", "name"):
        if getattr(args, key) is not None:
            cfg[key] = getattr(args, key)
        else:
            setattr(args, key, cfg.get(key))
    if args.resume:
        # 续训从 last.pt 载入；这里先改 cfg["model"]，_train 里还会再确认一次。
        cfg["model"] = os.path.join(_project_path(cfg["project"]), args.name,
                                    "weights", "last.pt")
    if not os.path.exists(cfg["model"]) and not args.eval_only:
        print("[FAIL] 预训练/续训权重不存在：{}（可从 archive/models_yolo/ 取，或 --model 指定）".format(
            cfg["model"]), file=sys.stderr)
        return 2

    data_local, data, root = _resolve_data(cfg["data"], cfg["project"])
    stats, n_cls = _validate(data, root)
    print("[info] 数据：{}（本机解析副本 {}）".format(cfg["data"], data_local))
    print("[info] 类别数：{}".format(n_cls))
    for split, s in stats.items():
        print("  {:<5} 图 {:<6} 框 {}".format(split, s["images"], s["boxes"]))

    if args.dry_run:
        print("[dry-run] 校验通过，未训练")
        return 0

    # 预缩图像缓存：大图解码是 DataLoader 的瓶颈；缓存只改尺寸、不改标签/划分。
    if cfg.get("cache_images", True) and not args.no_cache:
        cache_dir = _project_path(cfg.get("cache_dir", "results/detection/yolo_runs/cache_dataset"))
        max_side = int(cfg.get("cache_max_side", 1280))
        # 缓存新鲜度比对用项目相对路径，和 cache_meta.json 的口径一致。
        source = os.path.relpath(os.path.abspath(cfg["data"]), paths.PROJECT_ROOT)
        if args.rebuild_cache or not _cache_ready(cache_dir, max_side, source):
            data_local = _build_cache(data, root, cache_dir, max_side, cfg.get("workers", 8))
        else:
            data_local = os.path.join(cache_dir, "data_cache.yaml")
            print("[info] 复用图像缓存：{}".format(data_local))

    # 稀有类增强臂：重采样只改本次 train 清单（绝对路径指向缓存图），不动 P0 数据集与缓存原文件。
    if cfg.get("repeat_rare") and not args.eval_only:
        # 清单放独立子目录：与 Ultralytics 的 save_dir（<project>/<name>）同名会触发自动加 -2 后缀。
        data_local = _build_repeat_data(
            data_local, cfg["repeat_rare"],
            os.path.join(_project_path(cfg["project"]), "rare_data", args.name),
            names_src=cfg["data"])

    if args.eval_only:
        return _eval(cfg, data_local, args)
    return _train(cfg, data_local, args)


if __name__ == "__main__":
    sys.exit(main())
