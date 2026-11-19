# -*- coding: utf-8 -*-
"""
ls_common.py — 两阶段标注管线公共工具
所有图像读取统一走 imread_cn（np.fromfile + cv2.imdecode），规避 cv2 中文路径静默失败坑。

**路径常量 2026-09-29 起不再在此定义**：唯一权威是项目根的 `paths.py`，本文件只做 re-export
（`PROJECT_ROOT` / `DATASET` / `BASE` / … 名字与取值保持原样）。`ROOT`（旧义"标注体系"）已随
2026-10-09 目录重构删除 —— 容器目录 标注体系/ 已整体退场。
"""
import json
import os
import sys

from contracts import PrelabelRecord, ScaleReading  # 第 ⑤ 段契约（prelabel）——2026-09-29 C1a/C2a
# 该模块只依赖标准库，是四/五段契约的唯一事实源；本文件取它只为「scale 字段名单点」，
# 不引入任何回向依赖（contracts 不 import 本项目任何模块）。

try:
    import paths as _paths
except ImportError as _e:  # 唯一注入点缺失 → 明确报错，不做静默降级
    raise ImportError(
        "无法 import paths：项目 import 注入点（site-packages/tcm_project.pth）缺失。\n"
        "请用项目 venv 的解释器执行：\n"
        r"    <项目>\.venv\Scripts\python.exe paths.py --install" + "\n"
        "（该 .pth 把 paths.py 所在的项目根挂上 sys.path；详见 paths.py 顶部说明）") from _e

HERE = _paths.COMMON_DIR         # <项目>/common（本文件所在）
PROJECT_ROOT = _paths.PROJECT_ROOT
BASE = _paths.DETECTION_DIR      # <项目>/目标检测（阶段一目录；配置在其 config/）
DET_CONFIG = _paths.DETECTION_CONFIG_DIR        # 阶段一配置：vlm_config / criteria_table / folder_feature / labeling_config
SEG_CONFIG = _paths.SEGMENTATION_CONFIG_DIR     # 分割层配置：seg_classes / labeling_config_mask

# 图像库根 = LS 的 LOCAL_FILES_DOCUMENT_ROOT。语义见 paths.py（2026-09-20 起取项目根）。
DATASET = _paths.DATASET
IMAGE_SUBDIRS = _paths.IMAGE_SUBDIRS

# 迁移前旧根（D:\dataset），仅供 canon_path 兼容历史绝对路径
_LEGACY_DATASET = os.path.dirname(_paths.PROJECT_ROOT)


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


def norm_path(p):
    """路径归一化，用于**跨来源比较**同一张图（LS 任务 / gold 清单 / prelabel 三处各自的写法）。

    统一为 normpath → 正斜杠 → 小写。Windows 盘符与分隔符大小写不敏感，且 canon_path 的
    产物在不同调用点可能带 `\\` 或 `/`，不归一化就会把"同一张图"判成两张。
    与 canon_path 的分工：canon_path 负责**改写**旧根路径，本函数只负责**比较键**。
    """
    return os.path.normpath(str(p)).replace("\\", "/").lower()

# 运行产物根（2026-09-20 目录整理：脚本/配置与产物分离，产物一律落 results/；权威在 paths.py）
RESULTS = _paths.RESULTS
DET_RESULTS = _paths.DET_RESULTS

CONFIG_PATH = os.path.join(DET_CONFIG, "vlm_config.json")
MANIFEST_PATH = os.path.join(PROJECT_ROOT, "manifest.csv")  # 2026-10-09 随知识资产上提到项目根
PRELABEL_DIR = os.path.join(DET_RESULTS, "prelabels")   # VLM 预标注原始输出（每图一 json）
LS_TASKS_DIR = os.path.join(DET_RESULTS, "ls_tasks")    # Label Studio 导入用 json
YOLO_OUT_DIR = os.path.join(DET_RESULTS, "yolo_export")  # 导出 YOLO txt


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


# ---------- Label Studio API（JWT refresh token 自动换取 access token） ----------

def ls_token(cfg):
    tok = os.environ.get("LS_TOKEN", "").strip() or cfg["labelstudio"].get("token", "")
    if not tok:
        print("错误: 未设置 LS_TOKEN 环境变量，且 vlm_config.json 中 token 为空。", file=sys.stderr)
        sys.exit(2)
    return tok


def ls_auth_header(cfg):
    tok = ls_token(cfg)
    if tok.startswith("eyJ"):  # JWT（refresh token）→ 换取短期 access token
        try:
            import requests
            r = requests.post(cfg["labelstudio"]["url"].rstrip("/") + "/api/token/refresh",
                              json={"refresh": tok}, timeout=30)
            if r.ok and r.json().get("access"):
                tok = r.json()["access"]
        except Exception as e:
            print(f"[警告] access token 换取失败，直接用原 JWT 尝试: {e}", file=sys.stderr)
        return {"Authorization": f"Bearer {tok}"}
    return {"Authorization": f"Token {tok}"}  # 旧版 legacy token


def ls_api(cfg, method, path, **kw):
    import requests
    # ok_statuses: 显式允许的非 2xx 状态码（默认 None = 一律视为致命错误并退出）。
    # 用途：LS 对越界分页返回 404 {"detail":"Invalid page"}，分页循环需把它当正常收尾，
    # 而不是让整个进程 sys.exit（见 目标检测/lib/ls_tasks.fetch_tasks）。
    ok = kw.pop("ok_statuses", None)
    url = cfg["labelstudio"]["url"].rstrip("/") + path
    headers = {**ls_auth_header(cfg), **kw.pop("headers", {})}
    resp = requests.request(method, url, headers=headers, timeout=300, **kw)
    if resp.status_code >= 400 and (not ok or resp.status_code not in ok):
        print(f"LS API {method} {path} → {resp.status_code}: {resp.text[:500]}", file=sys.stderr)
        sys.exit(1)
    return resp


def zh_to_feature_id(cfg):
    """中文标签 -> feature_id 映射（以 yolo.classes 为唯一事实源）。"""
    return {c["zh"]: c["feature_id"] for c in cfg["yolo"]["classes"]}


def feature_id_to_zh(cfg):
    return {c["feature_id"]: c["zh"] for c in cfg["yolo"]["classes"]}


# ---------- LS 本地文件显示 URL（tif 浏览器代理映射） ----------

LS_VIEW_SUBDIR = "ls_view"  # results/detection/ 下：tif→jpg 浏览器代理镜像（需注册对应 LS 本地存储）
VIEW_ROOT = os.path.join(DET_RESULTS, LS_VIEW_SUBDIR)


def local_file_url(image_path):
    """绝对路径 → LS 本地文件服务 URL（相对 DATASET 根，中文自动 URL 编码）。"""
    import urllib.parse
    rel = os.path.relpath(image_path, DATASET).replace("\\", "/")
    return "/data/local-files/?d=" + urllib.parse.quote(rel)


def proxy_abs_path(image_path):
    """tif 源图 → ls_view 下 jpg 代理的绝对路径（镜像原目录结构）。"""
    rel = os.path.relpath(image_path, DATASET).replace("\\", "/")
    parts = os.path.splitext(rel)[0].split("/")
    return os.path.join(VIEW_ROOT, *parts) + ".jpg"


def ensure_tif_proxy(image_path):
    """为 tif/tiff 生成浏览器可渲染的 jpg 代理；已存在则跳过；非 tif 返回 None。

    浏览器原生不支持 TIFF → LS 服务端即便 200 image/tiff，标注页仍报
    "There was an issue loading URL from $image value"。故 tif 入 LS 前必须先落代理，
    且在【任务导入前】调用（view_file_url 依代理是否存在决定用哪个 URL）。
    返回代理绝对路径（已存在或新建成功），读取/编码失败返回 None。
    """
    if not str(image_path).lower().endswith((".tif", ".tiff")):
        return None
    out = proxy_abs_path(image_path)
    if os.path.exists(out):
        return out
    img = imread_cn(image_path)
    if img is None:
        return None
    import cv2
    os.makedirs(os.path.dirname(out), exist_ok=True)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        return None
    buf.tofile(out)
    return out


def view_file_url(image_path):
    """标注界面显示用 URL：tif/tiff 浏览器原生不渲染，若已生成 ls_view 下同名 .jpg 代理则返回代理 URL，否则退回原始 URL。"""
    if str(image_path).lower().endswith((".tif", ".tiff")):
        proxy_abs = proxy_abs_path(image_path)
        if os.path.exists(proxy_abs):
            import urllib.parse
            # 相对 DATASET 的规范路径（模块搬迁后自动带上 TCM-Microscopic-Image-Recognition/ 前缀）
            url_rel = os.path.relpath(proxy_abs, DATASET).replace("\\", "/")
            return "/data/local-files/?d=" + urllib.parse.quote(url_rel)
    return local_file_url(image_path)


def check_local_file_url(cfg, url, timeout=30):
    """自检：LS 本地文件服务能否真正取到该 URL → (ok, status)。

    图片在标注页打不开时，最常见的两种错配都能在这里立刻暴露：
      * 404 —— LS 的 LOCAL_FILES_DOCUMENT_ROOT 与生成 URL 时用的根不一致
        （LS 默认值是盘根 os.path.abspath(os.sep)，未设置环境变量时就是它）；
      * 403 —— LOCAL_FILES_SERVING_ENABLED 未开。
    用 stream 只取响应头，不下载整图。
    """
    import requests
    try:
        r = requests.get(cfg["labelstudio"]["url"].rstrip("/") + url,
                         headers=ls_auth_header(cfg), timeout=timeout, stream=True)
        r.close()
        return 200 <= r.status_code < 300, r.status_code
    except Exception:
        return False, -1


def legacy_file_url(image_path):
    """按**迁移前旧根**（D:\\dataset）计算 LS 本地文件 URL。

    仅供【匹配】数据库中已存在的历史 data.image —— 那些 URL 是当时按旧根算出来的
    （如 `显微镜像/大黄导管/1.jpg`、`TCM-Microscopic-Image-Recognition/标注体系/...`）。
    新写入一律用 local_file_url/view_file_url（按当前 DATASET），不要用本函数生成新数据。
    """
    import urllib.parse
    rel = os.path.relpath(image_path, _LEGACY_DATASET).replace("\\", "/")
    return "/data/local-files/?d=" + urllib.parse.quote(rel)


def ensure_dirs():
    for d in (PRELABEL_DIR, LS_TASKS_DIR, YOLO_OUT_DIR):
        os.makedirs(d, exist_ok=True)


# ---------- 比例尺 status 取值（单一事实源，scalebar_duo 与各下游统一引用） ----------
# grade 编码进 status，便于下游零改动穿透判断：
#   OK            双读一致（A 本地形态学 与 B VLM 的 bar_px 互证）
#   OK-consensus  A 可信、B 未证实，但 A 的 bar_px 已被同批精确 OK 样本旁证过
#                 （2026-09-28 补：bar_px 是离散可重复读数，97% 只取 233/311 两档）
#   OK-medium     A 可信、B 未证实，且 bar_px 尚无旁证 → μm 读数仅供参照
#   OK-low        A 未量到尺，值来自族先验或 B（grade=low）→ 仅供参照
#   manual / n/a  须人工 / 无效
STATUS_OK = "OK"
STATUS_OK_CONSENSUS = "OK-consensus"
# 下游"可信"的唯一判定入口：凡 status ∈ 本元组即视为可用（不再追加 caveat、可进测量链）
TRUSTED_STATUSES = (STATUS_OK, STATUS_OK_CONSENSUS)


def is_trusted_status(status):
    """status → 是否可信（大小写/空白容错）。下游一律用本函数，勿再手拼字符串比较。"""
    return (status or "").strip().upper() in tuple(s.upper() for s in TRUSTED_STATUSES)


# ---------- 判据速查（criteria_table.json 单一事实源） ----------

CRITERIA_PATH = os.path.join(DET_CONFIG, "criteria_table.json")


def load_criteria():
    with open(CRITERIA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def criteria_hint_for(herb, feature_label, scale_status=""):
    """按 药材|特征 查判据文本；比例尺不可信时附加"μm 读数仅供参考"提示。

    scale_status 取自 prelabel JSON 的 `scale.status`（检测阶段双引擎标定）。
    可信档 = TRUSTED_STATUSES（OK 双读一致 / OK-consensus 有跨图旁证）；
    OK-medium / OK-low / manual / n/a 视为不可信、附加 caveat。
    走 is_trusted_status() 单一判定入口，不再各处手拼 `!= "OK"`。
    """
    crit = load_criteria()
    hint = crit["table"].get(f"{herb}|{feature_label}", crit["table"]["default"])
    # 空 status（早期 JSON 无 scale 字段）保持旧行为：不追加 caveat——避免既有 LS 任务
    # 重现时提示文案突变（刻意的最小改动；要改这个语义请另开一轮决策）。
    if scale_status and not is_trusted_status(scale_status):
        hint += " " + crit["scale_caveat"]
    return hint


# ---------- prelabel JSON 里的比例尺字段（检测阶段双引擎标定的产物） ----------

def prelabel_scale(pre):
    """从 prelabel JSON 读 检测阶段双引擎标定结果 → (px_per_um, status)。

    2026-09-20 起 px_per_um 的唯一生产来源（离线 scale_table.csv 已停用）。
    早期 JSON 无 `scale` 字段 → 返回 ("", "")，由调用方按"未标定"处理。
    status ∈ STATUS_OK | STATUS_OK_CONSENSUS | OK-medium | OK-low | manual | n/a
    （grade 编码进 status）；可信与否请用 is_trusted_status()，勿手拼字符串比较。

    字段名 2026-09-29 C1a 起单点：由 `contracts.ScaleReading` 承载（此前本函数直接
    `sc.get("px_per_um")`，与另外 9 个 reader 各按记忆取值）。
    **C2a 起同时接受** `contracts.PrelabelRecord`（调用方多已持有记录）与**裸 dict**
    （仍在 `json.load` 后直接取值的老代码/测试），以免调用点各自绕开本入口去摸 `scale`。
    """
    if isinstance(pre, PrelabelRecord):
        sc = pre.scale
    else:
        sc = ScaleReading.from_dict(pre.get("scale") if isinstance(pre, dict) else None)
    return (sc.px_per_um if sc.px_per_um else ""), sc.status


# ---------- 同目录族先验（cross_validate 的 OK-low 兜底数据源） ----------

_FAM_INDEX = None


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


# ---------- 已旁证的 bar_px 离散集（cross_validate 的 OK-consensus 第二证据源） ----------

_OKBAR_CACHE = None


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


# ---------- prelabel 语料索引（检测评测与离线实验共用） ----------

def load_prelabel_index():
    """prelabels/*.json → {norm_path: `contracts.PrelabelRecord`}（image_path 先过 canon_path）。

    2026-09-29 P4-3 从 `detection/cli/det_eval.py` 上收：它此前被"评测"与
    `segmentation/experiments/` 的离线实验共用，而两边都只能靠 import 编号脚本取到它。
    归位依据：本文件已是 prelabel 域的家（`PRELABEL_DIR` / `prelabel_scale` /
    `family_bar_px` / `confirmed_bar_px` 都在此）。

    **2026-09-29 C2a：返回值由裸 dict 改为契约记录**。此前它返回的 dict 被就地改写
    `image_path` 后**整份透传**给下游（评测 / 离线实验 / LS 注释），下游再按记忆取字段 ——
    正是"被读得最广却无契约"的那一环。需要原始结构时用记录的 `raw`
    （注意 `raw["image_path"]` 是**未规范化**的原值）。
    """
    idx = {}
    for fn in sorted(os.listdir(PRELABEL_DIR)):
        if not fn.endswith(".json"):
            continue
        with open(os.path.join(PRELABEL_DIR, fn), encoding="utf-8") as f:
            rec = PrelabelRecord.from_dict(json.load(f))
        rec.image_path = canon_path(rec.image_path)      # 原为就地改写 dict，现改记录字段
        idx[norm_path(rec.image_path)] = rec
    return idx
