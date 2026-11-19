# -*- coding: utf-8 -*-
"""
detect_core.py — 检测原语（阶段一共用）：整图/分块 VLM 检测 + 跨块 NMS + 逐图 px_per_um。

2026-09-29 解耦：这些原语原本混在 segmentation/fullimage_detect_seg.py 里，导致检测侧
（detection/cli/detect_sample.py）不得不反向 import segmentation/。现抽出到 detection/lib/，
依赖方向回归 common ← detection ← segmentation。

依赖白名单：`ls_common`、`vlm_api`（传输层）、`prompts`（提示词与类别白名单）、`geom`（框 IoU 原语）、`scalebar_duo`。
**不得 import 编号脚本**（2026-09-29 P1-3 起）——编号脚本按语言约束不能作为包成员被 import，
原先的 `import_module("02_vlm_prelabel")` 已由 `prompts` 取代。
**不得 import segmentation/ 下任何模块**——JPEG 编码走共享传输层 `vlm_api.encode_jpeg_b64`
（2026-09-29 P4-1b：本地副本已删。它当初是为"借 segmentation 的编码器只为传 quality"而自带的，
P1-3 拆掉那处循环依赖后已无存在理由）。

  tiles_of(w, h, ov)                   2×2 重叠块 + 全图（**生产链不再调用**，见 detect_adaptive）
  detect_once(...)                     单次 VLM 检测（唯一的"编码→调用→解析→映射"实现）
  detect_tiles(...)                    按区域列表逐块调用（含 upscale）；**返回已 NMS 合并**
  detect_adaptive(...)                 整图 1 次检测（**分块兜底已于 2026-09-30 关闭**）
  detect_consensus(...)                多轮自洽采样 + 按复现程度聚簇（V1：只加票不过滤）
  iou_box(a, b) / cluster_dets(dets, thr)  框 IoU 与"按类聚簇"（**唯一的贪心 IoU 分组实现**）
  nms_merge(dets, thr)                 框去重（= 取每个簇的代表；与 cluster_dets 同一实现）
  lookup_px_per_um(image_path, img)    逐图双引擎 px_per_um 查询（阶段一产物）
"""
import os
from concurrent.futures import ThreadPoolExecutor

import cv2

from geom import clamp_box, iou_box
from ls_common import imread_cn, load_config, zh_to_feature_id
from prompts import build_prompt
from vlm_api import call_vlm, encode_jpeg_b64, parse_vlm_json


# 中文标签 → feature_id 映射：**复用 ls_common.zh_to_feature_id**（与 prompts.VALID_LABELS 同源，
# 都以 vlm_config → yolo.classes 为唯一事实源）。2026-09-29 A1：此前本文件又自行推导了一遍，
# 是同一事实源的第二条路径（P4-1a/P4-1b 治过的"同算法多副本"在这条链上的残留）。
# 下面那份 7 项兜底只在 vlm_config 读不出来时生效（与 prompts 的 12 项兜底内容不同，属已知分叉，
# 仅在配置不可读的降级路径可见）。
def _load_feature_id():
    try:
        return zh_to_feature_id(load_config())
    except Exception:
        return {"草酸钙簇晶": "cluster_crystal", "草酸钙方晶": "single_crystal",
                "草酸钙针晶": "needle_crystal", "导管": "vessel",
                "石细胞": "stone_cell", "淀粉粒": "starch_grain", "疑似待定": "pending"}


FEATURE_ID = _load_feature_id()

# 检测 crop 的松框外扩量（短边比例）：检测链（`cli/detect_sample.py`）与 stage 2 分类器训练集
# （`lib/cls_dataset.py`，同一 crop 口径）共用这一份。**不能收小**：2026-09-29 实测，收到 geom
# 默认的 5%/24px 时，2/3 的 crop 被 IGM 极性探针判成 invert（1 个 mask 真把背景当前景）——
# 探针靠"四角必须是背景"，余量太小四角就落到实体上；另有 1 个结晶顶到画面边缘的框兜不回来。
# 取短边 30%：对近方框与旧的"每边 ×30%"等量，对长条目标（针晶/导管）也不会沿长轴过度外扩
# （这正是 `geom.loose_box` 用**短边**定 margin 的原因）。
DET_LOOSE_MARGIN_RATIO = 0.30


def lookup_px_per_um(image_path, img=None, full=False):
    """逐图 px_per_um：检测阶段双引擎标定（`scalebar_duo.measure_img_px_per_um`）。

    2026-09-20 起离线 scale_table.csv 已停用，比例尺一律现测；status 不在可信档时返回空串
    （下游按"未标定"处理，不会拿到不可信的 μm）。
    可信档 = ls_common.TRUSTED_STATUSES：OK（双读一致）与 **OK-consensus**（B 未证实但 A 的
    bar_px 已被同批精确 OK 样本旁证，2026-09-28 补）——后者使实测 118 份原本被卡的图
    得以进入测量链，而 OK-medium/OK-low 仍视为不可信。

    `full=True` 时返回引擎的完整 info dict（供需要写进产物 JSON 的调用方，如 vlm_prelabel 的
    `scale` 字段）；默认只返回可信时的 px_per_um 字符串。
    **这段编排是唯一一份**（2026-09-29 A2）：此前 vlm_prelabel 里另有一份逐字相同的调用。
    """
    import scalebar_duo                      # 延迟导入：scalebar_duo 顶层 import ls_common/vlm_api
    from ls_common import TRUSTED_STATUSES, confirmed_bar_px, family_bar_px
    if img is None:
        img = imread_cn(image_path)
    info = scalebar_duo.measure_img_px_per_um(img, os.path.basename(str(image_path)),
                                              load_config(), px_fam=family_bar_px(image_path),
                                              px_ok_set=confirmed_bar_px())
    if full:
        return info
    return info["px_per_um"] if info["status"] in TRUSTED_STATUSES else ""


# ---------------------------------------------------------------- 阶段一：检测原语
def tiles_of(w, h, ov=0.1):
    """2x2 重叠分块（ov 为相对重叠率），返回 (x1,y1,x2,y2) 列表，末尾附全图。

    **生产链已不再调用本函数**（2026-09-30 关闭空检分块兜底，理由见 detect_adaptive）；
    保留供离线实验/回归显式使用。detect_tiles 的 regions 也改为默认整图，避免"不传 regions
    就悄悄分块"这种可达性陷阱。
    """
    tw, th = int(w * 0.55), int(h * 0.55)
    xs = [0, w - tw]
    ys = [0, h - th]
    ts = [(x, y, min(w, x + tw), min(h, y + th)) for y in ys for x in xs]
    ts.append((0, 0, w, h))
    return ts


def detect_once(img, cfg_vlm, api_key, prompt, whitelist, *, max_side=None, quality=90,
                region=None, upscale=False, clamp="full", default_label=None):
    """一次 VLM 检测：编码 → 调用 → 解析 → 过滤 → 映射回**原图坐标**。

    这是"单次检测"的**唯一实现**（2026-09-29 B1）：`detect_tiles` 与 `segmentation` 侧
    `vlm_prelabel.process_one` 此前各自抄了一份（编码/调用/解析/白名单/退化框/坐标映射/
    score 公式共约 30 行，`score = 0.92 if high else 0.65` 这类魔数两处各写一遍）。

    **两处口径差异一律参数化，不"顺手统一"**（与 P4-1b 的 quality 同一原则）：
      - `quality`：分块侧 92 / 全图侧 90 —— 两边上线口径不同，改默认会静默改载荷字节；
      - `max_side`：只缩不放的上限（分块侧固定 1536；全图侧取 `cfg_vlm.max_side`）；
      - `upscale=True`：允许放大（插值转 INTER_CUBIC）。分块兜底已于 2026-09-30 关闭，
        故生产链恒为 False；仅显式调 `detect_tiles(..., upscale=True)` 的离线实验会用到；
      - `region`：待检窗口在原图中的位置（默认全图）；
      - `clamp`：坐标夹紧口径 —— `"blocks"`（分块侧原写法：夹到 W-1/H-1 且保证宽高 ≥1px）
        与 `"full"`（全图侧原写法：min 角只保下界、max 角只保上界，不保宽高）。
        两者都照抄原实现（"贴边坐标"下结果确实不同，不合并）；但**两种口径算完后统一再走一道
        归正 + 双向夹取**（2026-10-12，方案 2）——因为 VLM 会返回超出 0~1000 的坐标，只靠
        `full` 的单向夹紧会让 min 角越过画幅上限、与 max 角反向（详见该处注释）。

    whitelist：允许的类别名集合（分块侧传 `FEATURE_ID`、全图侧传 `VALID_LABELS`；
    两者成员相同，见 `_load_feature_id` 的说明）。

    default_label：**无类别提框（两阶段 stage 1）专用**。S1 的输出是
    `{"bbox_2d":…, "appearance":…}`——**根本没有 `label` 字段**，正常路径会被白名单整批丢掉。
    传了 `default_label`（配合 `whitelist={default_label}`）后，缺 label 的对象按该标签收下，
    使"编码→调用→解析→映射"仍只有这一份实现（不为两阶段另写一份检测原语）。
    `None`（默认）时行为与以前逐字段一致。

    返回 `(result, err)`；`err` 非空即失败。result 含：
      `dets`     —— 框列表（label / `tight_bbox`(原图坐标) / confidence / score / morphology）
      `accepted` —— 与 dets **逐位对应**的原始 object（调用方要额外字段时用，如 prelabel 的 note）
      `dropped`  —— 被白名单/退化框判据丢弃的原始 object
      `scale` / `sent_hw` / `region_hw` / `text`
    """
    H, W = img.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in (region if region is not None else (0, 0, W, H))]
    tile = img[y1:y2, x1:x2]
    th, tw = tile.shape[:2]
    s = 1.0
    if max_side:
        s = (max_side / max(th, tw)) if upscale else min(1.0, max_side / max(th, tw))
    if s != 1.0:
        interp = cv2.INTER_CUBIC if s > 1.0 else cv2.INTER_AREA
        sent = cv2.resize(tile, (max(1, int(tw * s)), max(1, int(th * s))), interpolation=interp)
    else:
        sent = tile
    b64, _ = encode_jpeg_b64(sent, quality=quality)
    text, err = call_vlm(cfg_vlm, api_key, b64, prompt)
    if text is None:
        return None, err
    try:
        parsed = parse_vlm_json(text)
    except Exception as e:
        return None, f"解析: {e}"

    dets, accepted, dropped = [], [], []
    for o in parsed.get("objects", []):
        lab = o.get("label") or default_label     # 无类别提框：S1 不输出 label，按兜底标签收下
        bb = o.get("bbox_2d")
        if lab not in whitelist or not bb or len(bb) != 4:
            dropped.append(o)
            continue
        bx1, by1, bx2, by2 = [float(v) for v in bb]
        if bx2 <= bx1 or by2 <= by1:        # 归一化空间里就已退化 → 丢
            dropped.append(o)
            continue
        if clamp == "blocks":
            # 原 detect_tiles 写法：0~1000 → 区域像素 → 原图，夹到 W-1/H-1 且保宽高 ≥1
            gx1 = min(W - 1.0, max(0.0, x1 + bx1 / 1000.0 * tw))
            gy1 = min(H - 1.0, max(0.0, y1 + by1 / 1000.0 * th))
            gx2 = min(float(W), max(gx1 + 1, x1 + bx2 / 1000.0 * tw))
            gy2 = min(float(H), max(gy1 + 1, y1 + by2 / 1000.0 * th))
        else:
            # 原 process_one 写法：min 角只保下界、max 角只保上界
            gx1 = max(0.0, x1 + bx1 / 1000.0 * tw)
            gy1 = max(0.0, y1 + by1 / 1000.0 * th)
            gx2 = min(float(W), x1 + bx2 / 1000.0 * tw)
            gy2 = min(float(H), y1 + by2 / 1000.0 * th)
        # 2026-10-12 补（方案 2，用户裁定）：上面两种 clamp 都只对「各自那一角」单向夹紧，而 VLM
        # 偶尔返回**超出 0~1000 归一化系**的坐标（实测 方晶 B20180079 的 x1≈1075、B20180413、
        # B20190072(3)）。在 `full` 口径下 min 角只保下界 → 映射后 gx1 会越过画幅右缘并与 gx2
        # 反向（gx1 > gx2），产物里留成退化框（`check_prelabel` 会报，导出侧只能整框丢弃）。
        # 故在两种口径**都算完之后**统一加一道归正 + 双向夹取：先保证 x1<x2 / y1<y2，再把两角都
        # 夹进 [0,W]×[0,H]；夹后不足 1px 的判为噪声丢弃（与导出侧 export_det_dataset 同口径）。
        # 注：`blocks` 口径本就双向夹紧，此处对其是恒等变换（不改它的既定结果）。
        cb = clamp_box((gx1, gy1, gx2, gy2), W, H)
        if cb is None:
            dropped.append(o)
            continue
        gx1, gy1, gx2, gy2 = cb
        dets.append({"label": lab,
                     "tight_bbox": [round(gx1, 1), round(gy1, 1), round(gx2, 1), round(gy2, 1)],
                     "confidence": o.get("confidence", "medium"),
                     "score": 0.92 if o.get("confidence") == "high" else 0.65,
                     "morphology": o.get("morphology", "")})
        accepted.append(o)
    return {"dets": dets, "accepted": accepted, "dropped": dropped, "scale": s,
            "sent_hw": [sent.shape[0], sent.shape[1]], "region_hw": [th, tw], "text": text}, None


def detect_tiles(img, cfg_vlm, api_key, dir_prior=None, priors=None,
                 regions=None, upscale=False, prompt=None, whitelist=None, default_label=None):
    """分块检测：逐区域调 `detect_once`，**返回已 NMS 合并**的框（2026-09-29 B2）。

    dir_prior = (dir_name, herb, feature)，**默认 None = 不注入目录先验**（2026-09-30）。检测器
    的输入只该是「图 + 检测任务定义」：目录名在标注语料里就是标签（注入有用），在送检管线里
    却是**待验证的样品身份**（注入=把答案喂给取证环节）。故本参数改为显式 opt-in，由调用方
    按用途给值，而不是从目录名自动推导后无条件塞进去。
    priors = [(特征名, 数量或None)] 为人工配置的特征先验与数量，经 build_prompt 注入 prompt。
    regions: 待检区域（原图坐标）；None = **整图一张**（2026-09-30 起不再是 2×2 分块：
        分块兜底已关闭，不传 regions 不该悄悄变成多区域检测）。要多区域请显式传列表。
    upscale: True 时把每个区域**强制放大**到 max_side（旧行为 s≤1 从不放大，小目标拿不到增益）。
    每框额外带 `tile` = 区域序号（1-based，与打印行一致），便于回查是哪个块出的。
    """
    H, W = img.shape[:2]
    ts = regions if regions is not None else [(0, 0, W, H)]
    # prompt 覆盖（两阶段 stage 1 传 `PROMPT_S1`）；None 时与以前**逐字一致**
    pr = prompt if prompt is not None else build_prompt(dir_prior, priors=priors)
    wl = whitelist if whitelist is not None else FEATURE_ID
    dets = []
    for i, (x1, y1, x2, y2) in enumerate(ts, 1):
        res, err = detect_once(img, cfg_vlm, api_key, pr, wl,
                               max_side=1536, quality=92, region=(x1, y1, x2, y2),
                               upscale=upscale, clamp="blocks", default_label=default_label)
        if res is None:
            print(f"  [tile {i}/{len(ts)}] FAIL: {err}")
            continue
        for d in res["dets"]:
            d["tile"] = i
        dets.extend(res["dets"])
        th, tw = res["region_hw"]
        print(f"  [tile {i}/{len(ts)}] {tw}x{th} @({x1},{y1}) → {len(res['dets'])} 框")
    return nms_merge(dets)


def detect_adaptive(img, cfg_vlm, api_key, dir_prior=None, priors=None,
                    prompt=None, whitelist=None, default_label=None):
    """整图 1 次检测（**分块兜底已于 2026-09-30 关闭**）。

    历史与关闭理由（2026-09-30，用户裁决）：
      · 2026-09-29「甲方案」曾保留"整图检出为空 → 2×2 重叠块放大重检（+4 次）"的兜底，
        因为当时 16 张对比里 12 张整图非空（都不走兜底），看起来代价很低。
      · 但对纤维类做的四轮实测表明这条兜底是**净负**：分块会切过目标（4 个块的边界就是
        切割线），且会往"模型本来认为没东西"的图上硬塞框——淀粉粒 14→36、菌丝 1→6，
        而它想救的小目标并没有被救回（7 个纤维框里仅 1 个达细胞级）。
      · 与 v7.7 那条"没有合法类别时如实留空、宁可少框也不要制造无依据的证据"直接冲突，
        故整条删除，而不是加开关：**整图空就是空**。
    要用多区域检测请显式调 `detect_tiles(regions=[...])`（离线实验/回归用）。

    dir_prior: (dir_name, herb, feature)；**默认 None = 不注入目录先验**（理由见 detect_tiles）。

    后三个参数**只给两阶段 stage 1 用**（`prompt=PROMPT_S1` + `whitelist={ENTITY_LABEL}` +
    `default_label=ENTITY_LABEL`），全部为 None 时与以前逐字一致。见 `detect_once` 的 default_label。

    返回**已 NMS 合并**的框（去重内联在 detect_tiles，2026-09-29 B2）。
    """
    H, W = img.shape[:2]
    return detect_tiles(img, cfg_vlm, api_key, dir_prior, priors, regions=[(0, 0, W, H)],
                        prompt=prompt, whitelist=whitelist, default_label=default_label)


def cluster_dets(dets, thr=0.5):
    """按类别 + 贪心 IoU 聚簇 —— **框去重与多轮投票共用的唯一实现**。

    2026-10-08：`nms_merge` 与 `detect_consensus` 都需要"同类别内按 IoU 分组"，若各写一份
    必然漂移（AGENTS.md 禁止重复实现算法原语）。故把分组抽到这里，`nms_merge` 退化为
    "取每个簇的代表"——语义与原贪心完全一致（成员判据都是"与所有已保留代表 IoU < thr"，
    迭代次序都是按 score 降序的稳定排序），由 `test_detect_adaptive` 的对拍断言钉住。

    返回 `[{label, rep, members}]`；`rep` 取该簇内首个元素（即 score 最高者）。
    """
    by_label = {}
    for d in dets:
        by_label.setdefault(d["label"], []).append(d)
    clusters = []
    for lab, ds in by_label.items():
        ds.sort(key=lambda d: -d["score"])          # 稳定排序：同分保持原相对次序
        reps = []
        for d in ds:
            for c in reps:
                if iou_box(d["tight_bbox"], c["rep"]["tight_bbox"]) < thr:
                    continue
                c["members"].append(d)
                break
            else:
                reps.append({"label": lab, "rep": d, "members": [d]})
        clusters.extend(reps)
    return clusters


def nms_merge(dets, thr=0.5):
    """框去重：取每个簇的代表，再按（类别, y, x）排序便于稳定编号。"""
    out = [c["rep"] for c in cluster_dets(dets, thr)]
    out.sort(key=lambda d: (d["label"], d["tight_bbox"][1], d["tight_bbox"][0]))
    return out


def detect_consensus(img, cfg_vlm, api_key, n=5, thr=0.5, temp=0.7,
                     dir_prior=None, priors=None, workers=None):
    """多轮自洽采样 + 按复现程度聚簇（**V1：只加票，不过滤**）。

    为什么存在（2026-09-30 论文分析结论）：本链的逐框置信度是**假的**——`detect_once` 里
    `score = 0.92 if confidence == "high" else 0.65`，即"模型自称 high/medium"直接当分、
    未经任何标定。多轮采样给一个**不依赖模型内部状态、也不依赖 GT** 的替代信号：
    同一实体被反复提出 → 更可能是真的。

    读法上的关键限制（写在这里防误读）：投票只能滤**随机幻觉**，滤不掉**系统性偏置**——
    若模型每一轮都在同一位置把针晶叫成纤维，N 轮全票通过、越投越"确信"。故本函数的首要
    产物是**票数分布**（把误出拆成随机/系统两类），而不是"精度一定会涨"。
    方案与判定线见 `docs/检测多轮采样投票_方案与实验设计.md`。

    实现要点：
      · 各轮只有采样噪声不同（prompt 完全相同、`temperature=temp`）→ 彼此独立、可并发；
        `call_vlm` 每次自建 urllib 请求、不持跨调用状态，线程安全。
      · **并发是单进程内的**：跨进程各自计数会叠加突破账户并发上限（glm-5.3-flash = 50），
        故这里用本进程的线程池，且 `workers` 不得超过 n。
      · 调用参数与 `detect_adaptive` 的整图路径**逐项一致**（同一 `detect_once`、同一
        prompt、`max_side=1536`/`quality=92`/`clamp="blocks"`），保证票数差异只来自采样噪声。
      · 单轮失败只跳过该轮（`continue` 语义），并在 `rollout_n` 里如实记**成功轮数**——
        否则"N=5"的产物会在缺轮时冒充满轮。

    返回 `(clusters, raw)`：
      · `clusters` —— 已聚簇的框（形状同 `detect_adaptive`，每框另带 `votes` / `rollout_n`；
        **不过滤**，1 票的框也保留，这正是 V1 的诊断价值）。
      · `raw` —— **逐轮原始框**（`[{rollout, objects:[{label,tight_bbox,confidence,score,
        morphology}]}]`，只含成功轮）。**必须与聚簇结果一并落盘**：这是事后做"忽略标签"
        再聚簇（V3）或换 `thr` 复盘的唯一数据源。2026-10-08 的 V1 只留了聚簇结果，
        导致 V3 想验证时**只能重跑**——这是设计疏漏，勿再犯。
    """
    H, W = img.shape[:2]
    prompt = build_prompt(dir_prior, priors=priors)   # 各轮同 prompt，循环外构造一次
    n = max(1, int(n))
    cfg_r = dict(cfg_vlm)
    cfg_r["temperature"] = temp                       # 不改调用方持有的 cfg（只改副本）
    w = max(1, min(int(workers or n), n))

    def one(i):
        res, err = detect_once(img, cfg_r, api_key, prompt, FEATURE_ID,
                               max_side=1536, quality=92, region=(0, 0, W, H),
                               clamp="blocks")
        return i, res, err

    all_dets, raw, n_ok, fails = [], [], 0, []
    with ThreadPoolExecutor(max_workers=w) as ex:
        for i, res, err in ex.map(one, range(n)):
            if res is None:
                fails.append((i + 1, err))
                print(f"  [rollout {i + 1}/{n}] FAIL: {err}", flush=True)
                continue
            n_ok += 1
            for d in res["dets"]:
                d["rollout"] = i + 1                  # 溯源：该框由第几轮提出
            all_dets.extend(res["dets"])
            raw.append({"rollout": i + 1,
                        "objects": [{k: d.get(k) for k in ("label", "tight_bbox", "confidence",
                                                           "score", "morphology")}
                                    for d in res["dets"]]})
            print(f"  [rollout {i + 1}/{n}] {len(res['dets'])} 框", flush=True)
    if n_ok == 0:
        print(f"  [consensus] {n} 轮全部失败，返回空：{fails}", flush=True)
        return [], raw
    out = []
    for c in cluster_dets(all_dets, thr):
        rep = dict(c["rep"])                          # 浅拷贝：改动不回写原始 dets
        rep["votes"] = len(c["members"])
        rep["rollout_n"] = n_ok
        out.append(rep)
    out.sort(key=lambda d: (d["label"], d["tight_bbox"][1], d["tight_bbox"][0]))
    hist = [sum(1 for d in out if d["votes"] == v) for v in range(1, n_ok + 1)]
    print(f"  [consensus] {n_ok}/{n} 轮成功 → 原始 {len(all_dets)} 框 → {len(out)} 簇"
          f"（V1 不过滤）；票数 1..{n_ok} 的簇数 {hist}", flush=True)
    return out, raw