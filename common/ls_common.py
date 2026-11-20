# -*- coding: utf-8 -*-
"""检测和标定共用的路径、图像读取与配置适配。

从 paths.py 取得路径，在有历史侧车时读取双读一致的比例尺旁证。图像读取使用字节缓冲区，兼容 Windows 中文路径。

主要定义：load_config、imread_cn、resolve_key、zh_to_feature_id、is_trusted_status、family_bar_px 和 confirmed_bar_px。
关联：检测与分类 worker、IGM 和 scalebar_duo 消费这些接口。
"""
import json
import os
import sys

from contracts import ScaleReading

try:
    import paths as _paths
except ImportError as _e:  # 唯一注入点缺失 → 明确报错，不做静默降级
    raise ImportError(
        "无法 import paths：项目 import 注入点（site-packages/tcm_project.pth）缺失。\n"
        "请用项目 venv 的解释器执行：\n"
        r"    <项目>\.venv\Scripts\python.exe paths.py --install" + "\n"
        "（该 .pth 把 paths.py 所在的项目根挂上 sys.path；详见 paths.py 顶部说明）") from _e

PROJECT_ROOT = _paths.PROJECT_ROOT
DET_CONFIG = _paths.DETECTION_CONFIG_DIR
DATASET = _paths.DATASET
IMAGE_SUBDIRS = _paths.IMAGE_SUBDIRS
DET_RESULTS = _paths.DET_RESULTS

# 读取历史标定侧车时仍需解析其记录的原始路径。
_LEGACY_DATASET = os.path.dirname(_paths.PROJECT_ROOT)
CONFIG_PATH = os.path.join(DET_CONFIG, "vlm_config.json")
PRELABEL_DIR = os.path.join(DET_RESULTS, "prelabels")

STATUS_OK = "OK"
STATUS_OK_CONSENSUS = "OK-consensus"
TRUSTED_STATUSES = (STATUS_OK, STATUS_OK_CONSENSUS)

_FAM_INDEX = None
_OKBAR_CACHE = None


def canon_path(path):
    """迁移前旧绝对路径 → 当前路径；已是当前路径（或无法改写）则原样返回。

    兼容三类历史路径（prelabels/*.json、pilot_manifest.csv 等产物里仍是旧值；scale_table.csv 已于
    2026-09-20 停用，此条仅对更早的历史产物生效）：
      ① D:\\dataset\\显微镜像\\...            → <项目>\\dataset\\显微镜像\\...
      ② D:\\dataset\\数据量太少合集\\...       → <项目>\\dataset\\数据量太少合集\\...
      ③ D:\\dataset\\<项目名>\\标注体系\\...   → <项目>\\标注体系\\...
    """
    if not path:
        return path
    norm = os.path.normpath(str(path))
    if os.path.exists(norm):
        return norm
    legacy = os.path.normpath(_LEGACY_DATASET)
    if not norm.lower().startswith(legacy.lower() + os.sep):
        return norm
    tail = norm[len(legacy) + 1:]
    head = tail.split(os.sep)[0]
    if head in ("显微镜像", "数据量太少合集"):
        cand = os.path.join(PROJECT_ROOT, "dataset", tail)
    elif head.lower() == os.path.basename(PROJECT_ROOT).lower():
        cand = os.path.join(PROJECT_ROOT, tail[len(head) + 1:])
    else:
        cand = os.path.join(PROJECT_ROOT, tail)
    return cand if os.path.exists(cand) else norm


def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def imread_cn(path):
    """中文路径安全读取。返回 BGR ndarray 或 None。"""
    import cv2
    import numpy as np
    try:
        return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"[imread_cn] 读取失败 {path}: {e}", file=sys.stderr)
        return None


def resolve_key(cfg):
    """api_key：配置文件优先级低于环境变量。"""
    vlm = cfg["vlm"]
    key = os.environ.get(vlm.get("api_key_env", "VLM_API_KEY"), "").strip()
    if key:
        return key
    return vlm.get("api_key", "").strip()


def zh_to_feature_id(cfg):
    """中文标签 -> feature_id 映射（以 yolo.classes 为唯一事实源）。"""
    return {c["zh"]: c["feature_id"] for c in cfg["yolo"]["classes"]}


def is_trusted_status(status):
    """status → 是否可信（大小写/空白容错）。下游一律用本函数，勿再手拼字符串比较。"""
    return (status or "").strip().upper() in tuple(s.upper() for s in TRUSTED_STATUSES)


def family_bar_px(image_path, min_n=3):
    """同目录族先验：同文件夹 prelabel 中 status 精确 OK 的 scale.bar_px 众数。

    用途：A 引擎错块/未量到但 μm 已读到时，`scalebar_duo.cross_validate` 以
    px_fam/um 出 OK-low 兜底。族内实测 bar_px 恒定（同目录只有 233/585 一两种），
    而 um 标签可不同 → "族 px 众数 / 本图 um" 成立。
    门槛：族内精确 OK 样本 ≥ min_n 张才返回众数，否则 None（判定退回 manual，
    不出值）；索引首次调用时构建并缓存，其后 O(1)。
    """
    global _FAM_INDEX
    if _FAM_INDEX is None:
        import glob
        from collections import defaultdict
        idx = defaultdict(list)
        for fp in glob.glob(os.path.join(PRELABEL_DIR, "*.json")):
            try:
                with open(fp, encoding="utf-8") as f:
                    pre = json.load(f)
            except Exception:
                continue
            sc = ScaleReading.from_dict(pre.get("scale"))
            # 只收精确 OK（不含 OK-consensus）：族先验与旁证集都必须由双读一致的样本构成
            if sc.status != STATUS_OK or not sc.bar_px:
                continue
            folder = os.path.dirname(canon_path(pre.get("image_path", "")))
            idx[folder].append(int(sc.bar_px))
        _FAM_INDEX = idx
    vals = _FAM_INDEX.get(os.path.dirname(canon_path(str(image_path))), [])
    if len(vals) < min_n:
        return None
    from collections import Counter
    return Counter(vals).most_common(1)[0][0]


def confirmed_bar_px(min_n=3, prelabel_dir=None, use_cache=True):
    """已被旁证的 bar_px 集合 = 在 prelabel 语料里出现 ≥min_n 次的精确 OK 读数。

    用途：B 引擎未证实时（OK-medium 档）的第二独立证据。依据（2026-09-28 实测 422 份）：
    A 引擎的 bar_px 是**离散可重复**读数（精确 OK 样本 97% 只取 233/311 两档，对应两个
    镜头档），且 px_per_um = bar_px / bar_um 严格成立。实测 119 份 OK-medium 中 118 份的
    bar_px 落在本集合内 → "B 没证实"只是 B 的噪声，A 的读数与同批 OK 样本同值。

    **防自我放大**：只收 `status == "OK"`（精确双读一致），OK-consensus 自身不回灌——
    否则一次放宽会被下一次放宽当作证据，判据会逐轮松掉。
    门槛 min_n 默认 3：单次读到的新值可能是偶然（如实测 1 例 bar_px=331），≥3 张独立
    OK 才算可重复。索引首次构建后缓存，其后 O(1)（prelabel_dir/use_cache 供测试注入）。
    """
    global _OKBAR_CACHE
    if use_cache and prelabel_dir is None and _OKBAR_CACHE is not None:
        return _OKBAR_CACHE
    import glob
    from collections import Counter
    root = prelabel_dir or PRELABEL_DIR
    cnt = Counter()
    for fp in glob.glob(os.path.join(root, "*.json")):
        try:
            with open(fp, encoding="utf-8") as f:
                sc = ScaleReading.from_dict(json.load(f).get("scale"))
        except Exception:
            continue
        if sc.status == STATUS_OK and sc.bar_px:
            cnt[int(sc.bar_px)] += 1
    out = {bpx for bpx, n in cnt.items() if n >= min_n}
    if use_cache and prelabel_dir is None:
        _OKBAR_CACHE = out
    return out
