# -*- coding: utf-8 -*-
"""igm_segment.py — IGM（图像生成大模型）分割：原图 + 分割指令描述 → 与原图同尺寸的二值 mask

与既有两条通路（VLM 输出多边形栅格化 / GrabCut·Otsu 经典双基线）并列的第三条分割通路：
不要求目标边界能被"点序列"表达，直接让图像生成模型按 prompt 生成前景掩码图，
再经「缩放回原尺寸 → 极性（提示词契约优先）→ 阈值二值化 → 形态学清理 → 目标框先验」
还原为严格同尺寸 mask。

接口：**通用 OpenAI 兼容** `POST {base_url}/images/generations`（JSON，不用 multipart），
base_url / model 全部来自 vlm_config.json 的 igm 段，不绑定任何厂商。

定位：**本模块就是现役的图像分割通路**——图像分割已不走"人工标注造真值 + 训练分割网络"那条路，
故 IGM 产物是**生产交付件**，不是喂标注/训练的预标注件。
可靠性不靠"不许当真值"这句话兜，而靠两道闸门：
  ① `info["status"]`（ok/empty/full/review）随交付清单流转，`igm_status != ok` 的实例在判药侧
     **被整条丢弃**（`rule_engine.pool_evidence` 丢弃带 `igm_review` 旗标的记录，不计入证据）；
  ② gen_raw.png / input_sent.jpg / overlay.jpg 落盘，供人眼核对。
⚠ 已知边界（2026-10-09 实测，务必知悉）：文生图模型是"画一个合理的目标"而非描摹输入，且生成是
随机的——同一张 crop 两次独立生成，mask 面积占比可差 3.6×（0.086→0.312）；16 张 raw 的离线 replay
（同源对照：后处理在同一份 raw 上**确定性**、16/16 一致）显示尺寸偏差方向不固定——同一味菊花的两个
实例量得 41.3 与 17.6 μm（药典 32~37）。故**尺寸量值仅在 `status == ok` 时可用**；review 实例只能
回答"检到 / 没检到"，不能回答"有多大"。

输入:
  --image    原图路径（中文路径安全）。**可省略**：省略时弹出文件资源管理器选择图片；
             保留参数是为了批处理/脚本调用时免弹窗
  --feature  特征类型（feature_taxonomy.json 的 feature_id），据此取该特征的形态描述注入模板；
             默认 cluster_crystal。可用值见项目根 feature_taxonomy.json 的 features[]
  --desc     前景特征描述自由文本，覆盖 --feature 的查表结果（用于本体表里没有/措辞不合用的情况）
  --prompt   完整提示词，给了就直接用（绕过模板与描述），用于完全自定义
输出（results/segmentation/igm_seg/{stem}/）:
  input_sent.jpg  发给 IGM 的**实际输入画布**（原图等比放大 + letterbox 补边，JPEG q95，
                  与 API 载荷逐字节同源），供目检放大倍率与补边效果
  mask.png        主产物：与输入**严格同尺寸**的 0/255 uint8 单通道 mask
  gen_raw.png     IGM 原始生成图（无损重编码），供核对模型是否在描摹输入
  overlay.jpg     原图 + 红色前景轮廓，目检用
  meta.json       prompt/特征类型/描述/模型/size/阈值/极性/面积占比/连通域数/耗时等
用法:
  python igm_segment.py                     # 弹出文件资源管理器选择原图
  python igm_segment.py --image "D:\\dataset\\显微镜像\\大黄草酸钙簇晶\\20230311.tif"
  python igm_segment.py --image X.tif --feature vessel        # 换特征即换描述
  python igm_segment.py --image X.tif --desc "石细胞：类圆形，壁厚，胞腔明显"   # 自由文本
  python igm_segment.py --image X.tif --prompt "完全自定义的整句提示词"        # 绕过模板
  python igm_segment.py --image X.tif --box "24,24,145,121"  # 目标框先验：丢框外邻粒并裁到框内
  python igm_segment.py --image X.tif --dry-run        # 只打印解析出的请求，不发 API
  python igm_segment.py --image X.tif --quiet          # 批处理：只打印每图结果与错误
  python igm_segment.py --selftest                     # 纯离线后处理自检，不耗 API
运行时按 `[1/7 输入]→[2/7 提示词]→[3/7 画布]→[4/7 调用]→[5/7 解码]→[6/7 后处理]→[7/7 落盘]`
打印阶段进度：含完整提示词、画布/载荷规模、API 每次 attempt 的往返耗时、后处理关键决策
（极性/阈值/填洞/清理）与耗时分解。`--quiet` 把这些详细日志静默，只保留
`[结果]`/`[输出]`/`[FAIL]` 三类行（批处理用）。日志均带 flush，重定向到文件也能实时看到。
"""
import argparse
import base64
import json
import math
import os
import re
import sys
import time
import urllib.error
import urllib.request

import cv2
import numpy as np

# 2026-09-29：本模块自 segmentation/ 移入 lib/。原先靠 ".." 相对自身推导的常量会整体错一层
# （本体表找不到 → IGM 阶段直接崩；结果目录落到 segmentation/results/），故改由 paths 统一派生。
import paths  # noqa: E402
from geom import to_polygon  # noqa: E402 掩码→多边形原语（2026-09-29 三引擎管线归档时上收）
from ls_common import load_config  # noqa: E402
import association_policy as assoc  # noqa: E402 连通域关联策略（规则表，非 if/else）

RESULTS_SEG = paths.SEG_RESULTS          # <项目根>/results/segmentation
IGM_DIR = os.path.join(RESULTS_SEG, "igm_seg")

# 提示词 = 固定模板（PROMPT_TEMPLATE：任务 + 形状保真 + 输出格式）× 前景特征描述。
# 描述**不另建词表**，按特征类型查 feature_taxonomy.json（特征本体的单一事实源），
# 优先取其中 igm_desc（专为 IGM 提示词清洗过的措辞），缺失回退 description。
# 优先级：--prompt（整句覆盖）> --desc（自由文本）> --feature（查本体表）；默认 feature=簇晶。
TAXONOMY_PATH = os.path.join(paths.PROJECT_ROOT, "feature_taxonomy.json")  # 2026-10-09 上提到项目根
DEFAULT_FEATURE = "cluster_crystal"
# "只分割中央那一个目标"这句是 2026-09-29 加的：crop 由检测阶段按 loose_box 裁出、目标
# 恒在画面中央，但 IGM 会把边缘露出的同类别实体一并"描摹+补全"，既污染 mask（邻粒直接
# 并入实例）又搞坏极性判定（见 detect_invert 的说明）。点名禁止后由目标框先验兜底（见
# apply_box_prior）。
PROMPT_TEMPLATE = ("请完成图像的二值分割任务（只有一个前景和一个背景的图像分割），图中的前景是{desc}。\n"
                   "只分割位于画面中央的那一个目标：画面边缘、四角出现的其他同类实体属于"
                   "背景，不要分割、不要补全、不要描出轮廓。\n"
                   "请严格按图中所见该实体的实际轮廓，把它从背景中完整分割出来，"
                   "保留原本形状，不要简化和复杂化，不要凭空生成图中不存在的实体。\n"
                   "输出二值掩码图：前景为纯白色，背景为纯黑色，不做灰度渐变、不画轮廓线。\n"
                   "背景必须整幅均匀纯黑：不允许出现半幅灰/白背景，也不允许出现对角或渐变的明暗分界。\n"
                   "不要添加任何文字、数字、水印、标识、logo、边框或装饰元素。")
# 末尾两句是 2026-10-09 用户逐图目检 IGM 原始产物后补的（原本只写"前景纯白/背景纯黑"）：
#   · 红花 10_01：模型交回**灰度图**且背景沿对角线半黑半白 → Otsu 把白的那半当成前景；
#   · 各行产物右下角恒带「AI生成」水印 → 既可能被阈值吃进 mask，又会**污染四角极性探针**
#     （角块边长=2% 短边，恰好落在水印上）。
# 单靠措辞不能保证模型遵守，故后处理侧另有护栏（见 CENTER_MAX_RATIO 与状态守卫）。
EMPTY_AREA_RATIO = 1e-4        # 前景占比低于此值视为"空 mask"
FULL_AREA_RATIO = 0.95         # 前景占比高于此值视为"铺满"（背景必然触边，故触发极性兜底）
MIN_AREA_DEFAULT = 50          # 连通域最小面积（px），与 to_polygon 的 min_area 口径一致
# ---- 目标框先验（--box，见 apply_box_prior）----
BOX_MARGIN_RATIO = 0.10        # 判定/裁剪用外扩比例（占框宽高），吸收检测框与生成轮廓的偏差
BOX_KEEP_OVERLAP = 0.5         # 连通域落在框内（含外扩）的面积占比低于此值 → 判为框外邻粒
CENTER_MIN_RATIO = 0.3         # legacy：不再作为 review 触发条件；保留给审计/旧报告
CENTER_MAX_RATIO = 1.0         # 中心块面积 / 框面积 **高于**此值 → 中心块比检测框还大（IGM 画大了，或
                               # 极性选反把背景当成了前景），交人工。圆/凸目标内接于其紧框时该比值
                               # ≤π/4≈0.79，铺满整框才到 1.0；超过 1.0 说明 mask 已溢出目标框，
                               # 尺寸无从谈起。2026-10-09 实测：1.121 / 1.793 / 2.177 三例都属此类，
                               # 而原先只有下限判据，全靠 0.15 那条框外占比兜——只比框大 5%、不外溢的
                               # blob 会被判 ok 直接进几何。
BOX_OUTSIDE_MAX = 0.15         # 紧框外的前景占比上限；超了说明目标与框外物体粘连 → 交人工
MIN_TARGET_AREA_RATIO = 0.005  # 目标面积/框面积低于此值视为过小斑点（替代“必须居中”的旧下限）
BOX_CENTER_REGION_RATIO = 0.50  # 新关联策略：紧框中央50%宽高；单点命中仅作诊断
BOX_ASSOC_EXTENT_MIN = 0.30     # 连通域包围范围/紧框面积，不再要求细长目标填满30%框面积
# ---- 极性探针（见 detect_invert）----
CORNER_PROBE_RATIO = 0.02      # 四角背景探针的角块边长（占短边），至少 CORNER_PROBE_MIN_PX
CORNER_PROBE_MIN_PX = 3        # 角块最小边长；过大（如 5%）会探进贴边实体里，反而探错
CORNER_FLAT_STD_RATIO = 0.10   # 角块 std / 255 超过此值 → 该角**不平坦**（文字/水印/纹理/实体边界）
                               # → 弃权，不作为背景样本。2026-10-09 实测：各行 IGM 原始产物右下角
                               # 恒带「AI生成」水印，而角块边长 = 2% 短边恰好落在水印上 —— 探针采到的
                               # "背景"其实是水印像素。只信任平坦角块即可自动规避，无需硬编码位置。
RING_MARGIN_RATIO = 0.20       # 四角不齐时，环带判据要求的相对差距门槛；不到则回退契约取亮侧

# ---- 生成画布尺寸约束（size_mode=auto 用；依据 vlm_config.json.example 实测记录）----
# doubao-seedream-5-0-lite 要求画布 ≥3686400 像素（=1920x1920），低于下限报 400
# InvalidParameter；pro/flash 可 1024x1024(=1048576)。故 min_pixels 是**已确认的 API
# 硬约束**，max_side 是"模型单边上限"（仓库内无实测记录，默认 4096，按实际调整）。
MIN_PIXELS_DEFAULT = 3686400   # 画布像素数下限（lite 口径）
MAX_SIDE_DEFAULT = 4096        # 单边上限（未实测，保守上限；超限时在 note 中标注）
SIZE_QUANTUM_DEFAULT = 64      # 边长对齐步长（现网可用的 1920 = 64×30）
RESIZE_FG_LEVEL = 128          # 回缩到原图尺寸时的前景判定阈值（面积平均后 ≥ 该值算前景）


# ---------------------------------------------------------------- 控制台进度日志
# 7 阶段进度：[1/7 输入] [2/7 提示词] [3/7 画布] [4/7 调用] [5/7 解码] [6/7 后处理] [7/7 落盘]。
# 全部 flush=True：本脚本常被重定向到文件/管道，不 flush 会整段缓冲、看不到实时进度
# （尤其 [4/7 调用] 的"发起…"必须在阻塞等待前落盘，否则等 15~30s 期间一片静默）。
# --quiet 只保留"每图一行结果 + 错误"，供批处理使用。
QUIET = False


def set_quiet(flag):
    """由 main 按 --quiet 设置（stage/box_print 静默，result/fail 仍打印）。"""
    global QUIET
    QUIET = bool(flag)


def stage(tag, msg=""):
    """阶段进度行：`[n/7 阶段] 内容`。--quiet 下静默。"""
    if not QUIET:
        print(f"[{tag}] {msg}".rstrip(), flush=True)


def result(msg):
    """每图一行的结果摘要（--quiet 下仍打印）。"""
    print(msg, flush=True)


def box_print(title, text, indent="  "):
    """把多行内容（如提示词）用框线包起来打印，避免与普通日志行混淆。--quiet 下静默。"""
    if QUIET:
        return
    lines = (text or "").splitlines() or [""]
    print(f"{indent}┌ {title}（{len(text or '')} 字符）", flush=True)
    for ln in lines:
        print(f"{indent}│ {ln}", flush=True)
    print(f"{indent}└" + "─" * 44, flush=True)


def fail(tag, msg):
    """带阶段号的失败输出（--quiet 下仍打印），便于一眼定位停在哪一步。"""
    sys.exit(f"[FAIL][{tag}] {msg}")


def build_prompt(desc):
    """前景特征描述注入固定模板，得到最终提示词。"""
    return PROMPT_TEMPLATE.format(desc=(desc or "").strip())


def load_feature_desc(feature_id):
    """按特征类型从 feature_taxonomy.json 取形态描述。

    优先取 igm_desc：该字段**自带中文名前缀**（如"导管：长管状结构…"），直接使用；
    缺失时回退 description 并补上"中文名："前缀。
    未知 feature_id 即报错并列出可用键（不静默回退到默认值，避免用错特征却看不出来）。
    """
    with open(TAXONOMY_PATH, "r", encoding="utf-8") as f:
        feats = json.load(f).get("features", [])
    idx = {it.get("feature_id"): it for it in feats}
    if feature_id not in idx:
        sys.exit(f"[FAIL] feature_taxonomy.json 中无特征 {feature_id!r}；可用：\n  "
                 + "、".join(sorted(k for k in idx if k)))
    it = idx[feature_id]
    zh = (it.get("zh_name") or feature_id).strip()
    igm_desc = (it.get("igm_desc") or "").strip()
    if igm_desc:
        return igm_desc                      # 已含"名称：描述"，勿重复加前缀
    desc = (it.get("description") or "").strip()
    if not desc:
        sys.exit(f"[FAIL] 特征 {feature_id} 在 feature_taxonomy.json 中无 igm_desc/description，"
                 f"请改用 --desc 直接给描述")
    return f"{zh}：{desc}"


# ---------------------------------------------------------------- 配置
def load_igm_cfg():
    """读取 vlm_config.json 的 igm 段。缺段或关键项缺失即报错退出（不做静默降级）。"""
    cfg = load_config()
    cfg_igm = cfg.get("igm")
    if not cfg_igm:
        sys.exit("[FAIL] vlm_config.json 缺少 igm 段。请照 vlm_config.json.example 的 igm 模板补充"
                 "（base_url / model 必填，api_key 可留空回退 vlm.api_key，或用环境变量 IGM_API_KEY）")
    if not cfg_igm.get("base_url") or not cfg_igm.get("model"):
        sys.exit("[FAIL] vlm_config.json 的 igm 段缺少 base_url 或 model")
    return cfg_igm, cfg.get("vlm", {})


def resolve_igm_key(cfg_igm, cfg_vlm):
    """密钥优先级：环境变量 IGM_API_KEY > igm.api_key > vlm.api_key（同厂商同账号可复用）。"""
    env_name = cfg_igm.get("api_key_env", "IGM_API_KEY")
    key = os.environ.get(env_name, "").strip()
    if key:
        return key, f"env:{env_name}"
    key = (cfg_igm.get("api_key") or "").strip()
    if key:
        return key, "igm.api_key"
    key = (cfg_vlm.get("api_key") or "").strip()
    if key:
        return key, "vlm.api_key(回退)"
    return "", ""


# ---------------------------------------------------------------- 图像编解码
def encode_image_jpeg(img, quality=95):
    """BGR ndarray → JPEG 字节串（=发给 IGM 的实际载荷）。带 alpha 的输入先丢弃 alpha。"""
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("jpeg encode 失败")
    return buf.tobytes()


def encode_image_b64(img, quality=95):
    """BGR ndarray → base64 JPEG 串（转调 encode_image_jpeg，保证与落盘载荷同源）。"""
    return base64.b64encode(encode_image_jpeg(img, quality)).decode()


def imread_any(path):
    """中文路径安全读取，保留通道（可能带 alpha）。"""
    try:
        return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    except Exception as e:
        print(f"[imread_any] 读取失败 {path}: {e}", file=sys.stderr)
        return None


def write_img(path, img, ext=None):
    """中文路径安全落盘：cv2.imencode → 经已打开的文件对象写入。

    注意不直接 `buf.tofile(path)`：ndarray.tofile 对 str 路径走 C 层 fopen，
    Windows 上遇中文路径/文件被占用（预览、看图软件锁定）会抛
    `OSError: [Errno 22] Invalid argument`（2026-09-23 实测）。经 Python
    open() 的文件对象写入则由 WinAPI 处理路径，规避该类错误；占用类失败
    翻译成人话提示。
    """
    ext = ext or os.path.splitext(path)[1].lower() or ".png"
    params = [cv2.IMWRITE_JPEG_QUALITY, 90] if ext in (".jpg", ".jpeg") else []
    ok, buf = cv2.imencode(ext, img, params)
    if not ok:
        raise RuntimeError(f"编码失败: {path}")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        with open(path, "wb") as f:
            buf.tofile(f)
    except OSError as e:
        raise RuntimeError(
            f"写盘失败: {path} —— 文件可能正被看图软件/预览窗格占用，请关闭后重试: {e}") from e


# ---------------------------------------------------------------- letterbox
def parse_size(size_str):
    """'1024x1024' → (1024, 1024)。"""
    m = re.match(r"^\s*(\d+)\s*[xX*×]\s*(\d+)\s*$", size_str or "")
    if not m:
        sys.exit(f"[FAIL] --size / igm.size 格式应为 WxH（如 1024x1024），收到: {size_str!r}")
    return int(m.group(1)), int(m.group(2))


def parse_box(text):
    """'x1,y1,x2,y2' → [int×4]；空串 → None（表示不做目标框先验）。"""
    if not (text or "").strip():
        return None
    parts = [p for p in re.split(r"[,;\s]+", text.strip()) if p]
    if len(parts) != 4:
        sys.exit(f"[FAIL] --box 格式应为 x1,y1,x2,y2（crop 局部紧框），收到: {text!r}")
    return [int(round(float(p))) for p in parts]


def pick_size(src_w, src_h, min_pixels=MIN_PIXELS_DEFAULT, max_side=MAX_SIDE_DEFAULT,
              quantum=SIZE_QUANTUM_DEFAULT):
    """按原图长宽比选生成画布 (out_w, out_h)，返回 (w, h, note)。

    取法：**满足像素下限所需的最小倍率，且不无谓放大/不越过单边上限**——
        scale = max(scale_floor, min(1.0, scale_cap))
    其中 scale_floor = sqrt(min_pixels / (src_w*src_h)) 是达标下限所需倍率，
    scale_cap = max_side / max(src_w, src_h) 是单边上限允许的倍率。效果：
      * 小图（如 154x152）：放大到刚好满足下限（≈1920x1920），不再无谓放大；
      * 中等图（如 2048x1536）：按需小幅放大到恰好达标（约 2240x1664），画布面积
        全部用于内容，而非被 letterbox 补边吃掉（原固定 1920x1920 时 4:3 图补边占 25%）；
      * 大图（如 8000x6000）：不缩到下限，而是用满单边上限（4096），保住回缩后的细节。
    像素下限与单边上限冲突（极端长宽比）时**以下限优先**（下限是已确认的 API 硬约束），
    并在 note 中标注。两边对齐到 quantum 的整数倍；取整后跌破下限则抬一侧（取长宽比
    失真较小的一侧）。
    """
    src_w, src_h = max(1, int(src_w)), max(1, int(src_h))
    aspect = src_w / float(src_h)
    scale_floor = math.sqrt(min_pixels / float(src_w * src_h))
    scale_cap = max_side / float(max(src_w, src_h))
    note = ""
    if scale_floor > scale_cap:
        scale = scale_floor
        note = (f"像素下限需倍率{scale_floor:.2f} 超过单边上限允许的{scale_cap:.2f}，"
                f"以下限优先（长边可能超 max_side={max_side}，若被拒请调大 max_side 或用固定 size）")
    else:
        scale = max(scale_floor, min(1.0, scale_cap))
    out_w = max(quantum, int(round(src_w * scale / quantum)) * quantum)
    out_h = max(quantum, int(round(src_h * scale / quantum)) * quantum)
    if out_w * out_h < min_pixels:          # 取整跌破下限 → 抬一侧，选长宽比失真小者
        cands = [(out_w + quantum, out_h), (out_w, out_h + quantum)]
        out_w, out_h = min(cands, key=lambda s: abs(math.log((s[0] / s[1]) / aspect)))
    return out_w, out_h, note


def letterbox_to(img, out_w, out_h):
    """等比缩放后在四周居中补边，组成 out_w×out_h 画布（避免直接把原图拉伸变形）。

    返回 (画布图, meta)。meta 存**比例**而非像素：服务端返回的尺寸未必等于请求尺寸，
    按比例回裁/回缩才能对上。用 BORDER_REPLICATE 而非纯黑：边缘保持背景延续更自然。
    """
    h, w = img.shape[:2]
    scale = min(out_w / w, out_h / h)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    resized = cv2.resize(img, (nw, nh), interpolation=interp)
    pad_y, pad_x = out_h - nh, out_w - nw
    top, left = pad_y // 2, pad_x // 2
    canvas = cv2.copyMakeBorder(resized, top, pad_y - top, left, pad_x - left,
                                cv2.BORDER_REPLICATE)
    meta = {
        "out_w": out_w, "out_h": out_h, "content_w": nw, "content_h": nh,
        "top_ratio": top / out_h, "left_ratio": left / out_w,
        "cw_ratio": nw / out_w, "ch_ratio": nh / out_h, "scale": round(scale, 6),
    }
    return canvas, meta


def _crop_content(arr, meta):
    """按 meta 的比例裁出内容区（丢掉 letterbox 补边），兼容服务端返回尺寸与请求不一致。"""
    gh, gw = arr.shape[:2]
    y0 = int(round(meta["top_ratio"] * gh))
    x0 = int(round(meta["left_ratio"] * gw))
    ch = int(round(meta["ch_ratio"] * gh))
    cw = int(round(meta["cw_ratio"] * gw))
    return arr[y0:y0 + ch, x0:x0 + cw]


# ---------------------------------------------------------------- API
def call_igm(cfg_igm, api_key, b64, prompt, size, dry_run=False):
    """POST {base_url}/images/generations。重试骨架与 common/vlm_api.call_vlm 一致。

    兼容点（各家实现不统一）：
      - 图生图入参字段名由 cfg_igm.image_field 指定（默认 image），值为 data URI；
      - response_format 仅在配置非空时下发（新版 OpenAI 图像模型已移除该字段，硬传会 400）；
      - n 显式置 1。
    返回 (payload_dict, err)；dry_run=True 时只打印请求摘要并返回 (None, None)。
    """
    body = {"model": cfg_igm["model"], "prompt": prompt, "n": 1, "size": size}
    if b64:
        body[cfg_igm.get("image_field", "image")] = f"data:image/jpeg;base64,{b64}"
    if cfg_igm.get("response_format"):
        body["response_format"] = cfg_igm["response_format"]

    url = cfg_igm["base_url"].rstrip("/") + "/images/generations"
    if dry_run:
        brief = {k: (v[:64] + f"...({len(v)} chars)" if isinstance(v, str) and len(v) > 64 else v)
                 for k, v in body.items()}
        print(f"[dry-run] {url}")
        print(f"[dry-run] body: {json.dumps(brief, ensure_ascii=False)}")
        return None, None

    req = urllib.request.Request(
        url, data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"})
    n_retry = cfg_igm.get("max_retries", 3)
    timeout_s = cfg_igm.get("timeout_s", 300)
    stage("4/7 调用", f"POST {url} | model={cfg_igm['model']} | size={size} | "
                      f"timeout={timeout_s}s | max_retries={n_retry}")
    last_err = None
    for attempt in range(1, n_retry + 1):
        stage("4/7 调用", f"发起 attempt {attempt}/{n_retry} …（等待期通常 15~30s）")
        t_attempt = time.time()
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                raw = resp.read()
                data = json.loads(raw.decode("utf-8"))
            if not (data.get("data") or []):
                last_err = f"返回无 data 项: {json.dumps(data, ensure_ascii=False)[:500]}"   # 瞬时异常，重试
                stage("4/7 调用", f"← attempt {attempt} 返回无 data（{time.time() - t_attempt:.1f}s），"
                                  f"{3 * attempt}s 后重试")
            else:
                stage("4/7 调用", f"← attempt {attempt} 成功（{time.time() - t_attempt:.1f}s）| "
                                  f"响应 {len(raw) / 1024:.0f}KB")
                return data, None
        except urllib.error.HTTPError as e:
            body_txt = e.read().decode("utf-8", "replace")[:500]
            last_err = f"HTTP {e.code}: {body_txt}"
            stage("4/7 调用", f"← attempt {attempt} HTTP {e.code}"
                              f"（{time.time() - t_attempt:.1f}s）：{body_txt[:160]}")
            if e.code in (401, 403):                   # 鉴权问题重试无意义
                return None, last_err
        except Exception as e:
            last_err = str(e)
            stage("4/7 调用", f"← attempt {attempt} 异常（{time.time() - t_attempt:.1f}s）："
                              f"{last_err[:160]}")
        time.sleep(3 * attempt)
    return None, last_err


def payload_to_image_bytes(payload):
    """从 images/generations 响应取原始图片字节。

    优先 b64_json；否则取 url（可能是 http(s) 也可能是 data: URI）。url 一律**立即下载**，
    因为该类链接通常短时过期。返回 (bytes, err)。
    """
    item = (payload.get("data") or [{}])[0]
    b64 = item.get("b64_json")
    if b64:
        try:
            b64 = b64.strip()
            return base64.b64decode(b64 + "=" * (-len(b64) % 4)), None   # 容忍缺 padding
        except Exception as e:
            return None, f"b64_json 解码失败: {e}"
    url = item.get("url") or item.get("image_url")
    if not url:
        return None, f"响应既无 b64_json 也无 url: {json.dumps(item, ensure_ascii=False)[:300]}"
    if url.startswith("data:"):
        try:
            return base64.b64decode(url.split(",", 1)[1]), None
        except Exception as e:
            return None, f"data URI 解码失败: {e}"
    try:
        with urllib.request.urlopen(url, timeout=120) as resp:
            return resp.read(), None
    except Exception as e:
        return None, f"下载生成图失败: {e}"


def decode_generated(raw):
    """图片字节 → (BGR, alpha, err)。alpha 无则 None；解不出（如 HTML 错误页）给明确报错。"""
    arr = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if arr is None:
        head = raw[:120].decode("utf-8", "replace")
        return None, None, f"生成图解码失败（可能返回的是错误页）: {head!r}"
    alpha = None
    if arr.ndim == 3 and arr.shape[2] == 4:
        alpha = arr[:, :, 3]
        bgr = arr[:, :, :3]
    elif arr.ndim == 2:
        bgr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    else:
        bgr = arr[:, :, :3]
    return bgr, alpha, None


# ---------------------------------------------------------------- 后处理
def detect_invert(gray_roi):
    """极性探针：判断前景落在暗侧（True）还是亮侧（False）。返回 (invert, note)。

    探针 1（首选）**四角小块**：crop 按 loose_box 裁出、目标恒在画面中央，故四角是背景
    （或其邻粒）最可靠的落脚点。四角同侧才下结论：均偏暗 ⇒ 背景暗 ⇒ 前景亮（False）；
    均偏亮 ⇒ 背景亮 ⇒ 前景暗（True）。
    探针 2（四角不齐时）环带均值 vs 全图均值，且要求**决定性差距**（相对 ≥ RING_MARGIN_RATIO），
    否则退回契约（提示词要求前景纯白 → 取亮侧）。
    兜底：按该极性算前景占比，>FULL_AREA_RATIO 则翻转（前景不该铺满整幅）。

    为什么不能只用环带判据（2026-09-29 实测）：IGM 会画出**多个实体**，实体自身也触边，
    此时"环带更亮"只说明画面边缘是实体、而非背景偏亮 → 判据会反。淀粉粒_img3_b1 即如此
    （环带 171.6 vs 全图 155.0 → 判暗侧；而该图实为"黑底白颗粒"，采用后 mask 盖到背景上：
    前景像素亮度均值 13.2、全图亮侧占比 0.61，且占比 0.398 够不到 0.95 兜底）。
    改看四角后，该图四角均暗（0/4 亮）→ 正确取亮侧；而 草酸钙簇晶_img2_b0 四角均亮
    （4/4）→ 正确取暗侧（该图确是白底黑对象）。全库 24 例逐例复核：除 b1 修正外全部不变。
    """
    h, w = gray_roi.shape[:2]
    t_otsu, _ = cv2.threshold(gray_roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    s = int(max(CORNER_PROBE_MIN_PX, round(CORNER_PROBE_RATIO * min(h, w))))   # 角块边长
    corners = [gray_roi[:s, :s], gray_roi[:s, -s:], gray_roi[-s:, :s], gray_roi[-s:, -s:]]
    # 只信**平坦**角块：角块本应是平整背景，带文字/水印/纹理/实体边界的角块一律弃权，否则探针会把
    # 水印像素当成背景样本（见 CORNER_FLAT_STD_RATIO 注释）。弃权后剩余角块仍须**全同侧**才下结论。
    trusted = [c for c in corners if float(c.std()) <= CORNER_FLAT_STD_RATIO * 255.0]
    n_bright = sum(1 for c in trusted if float(c.mean()) > t_otsu)
    if trusted and n_bright in (0, len(trusted)):
        invert = (n_bright == len(trusted))
        note = (f"四角平坦块 {n_bright}/{len(trusted)} 亮（角块 {s}px，相对 Otsu {t_otsu:.0f}；"
                f"{len(trusted)}/4 块可信）→ invert={invert}")
    else:
        t = int(max(2, round(0.02 * min(h, w))))          # 环带厚度：2% 边长，至少 2px
        ring = np.ones((h, w), bool)
        ring[t:h - t, t:w - t] = False
        ring_mean = float(gray_roi[ring].mean()) if ring.any() else float(gray_roi.mean())
        all_mean = float(gray_roi.mean())
        rel = (ring_mean - all_mean) / max(1.0, (ring_mean + all_mean) / 2.0)
        invert = rel >= RING_MARGIN_RATIO
        note = (f"四角不齐（{n_bright}/4 亮）→ 环带判据：环带均值{ring_mean:.1f} vs "
                f"全图{all_mean:.1f}（相对 {rel:+.2f}，门槛 {RING_MARGIN_RATIO}）→ invert={invert}")
    flag = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    ratio = float((cv2.threshold(gray_roi, t_otsu, 255, flag)[1] > 0).mean())
    if ratio > FULL_AREA_RATIO:
        invert = not invert
        note += f"；前景占比{ratio:.3f}> {FULL_AREA_RATIO}，翻转兜底 → invert={invert}"
    return invert, note


def fill_holes_limited(mask, max_hole_area=None):
    """填洞：填**不贴边界**（=被前景完全包围）的封闭背景区，max_hole_area 可选设上限。

    语义依据（2026-09-23 实测更正）：簇晶在明场下常呈"暗环带 + 较亮内核"——环带与
    内核**同属目标**，内核是被环带包围的封闭区，必须填掉，否则 mask 会退化成环带、
    目标内部出现"背景色"空洞（即前景/背景反转）。故 **max_hole_area=None/≤0 表示全填**
    （默认语义）；给定上限时只填不超过该面积的封闭区（用于"环形目标、内部确属背景"的
    特殊情形，由 --max-hole-ratio 控制）。

    注意调用时机：必须在**生成图分辨率**上调用、降采样之前（见 generated_to_mask）。
    贴边界 = 与画布外连通的外部背景，永不填。
    """
    out = mask.copy()
    if not (mask > 0).any():
        return out
    h, w = mask.shape
    bg = (mask == 0).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(bg, 8)
    for i in range(1, n):
        x, y, bw, bh, a = stats[i]
        touches_border = (x == 0 or y == 0 or x + bw >= w or y + bh >= h)
        if touches_border:
            continue
        if max_hole_area is None or max_hole_area <= 0 or a <= max_hole_area:
            out[lab == i] = 255
    return out


def clean_mask(mask, min_area, keep_largest):
    """形态学清理：闭运算补断口 → 开运算去毛刺 → 连通域按面积过滤 →（可选）只留最大块。

    不含填洞——填洞由 generated_to_mask 在**生成图分辨率**上完成（降采样会打断细环带，
    使内核不再闭合、填洞失效，见该函数说明）。
    也**不做**目标框先验（那一步在回缩到原图尺寸后由 apply_box_prior 做）。
    """
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    m = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k5)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k3)
    n, lab, stats, _ = cv2.connectedComponentsWithStats((m > 0).astype(np.uint8), 8)
    if n > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        keep = np.where(areas >= min_area)[0] + 1
        if keep.size and keep_largest:
            keep = np.array([keep[int(np.argmax(areas[keep - 1]))]])
        out = np.zeros_like(m)
        for lbl in keep:
            out[lab == lbl] = 255
        m = out
    return m


def apply_box_prior(mask, box, margin=BOX_MARGIN_RATIO, keep_overlap=BOX_KEEP_OVERLAP,
                    association="center_point", *, feature_id=None,
                    granularity="instance", association_policy=None):
    """目标框先验：按 crop 局部检测紧框做"目标感知"的连通域筛选与裁剪。返回 (mask, info)。

    依据：crop 是检测阶段按 loose_box 裁的，tight_box 即"本 crop 唯一目标"的范围，目标恒在
    框中心；框外（尤其贴着 crop 边界）的同类别实体是邻粒 —— 它们是**另一处**检测框的目标。
    若并进同一实例，几何会把全部连通域的点合起来求凸包（见 geomlib/descriptors.py 的
    _hull_points），Feret/面积被直接放大。实测 淀粉粒_img3_b2：4 个连通域里 3 个是贴边邻粒
    （3676px，占 mask 39%）→ 69.3 μm²/16.42 μm，只留目标块则是 42.3 μm²/7.61 μm。

    步骤：① 丢掉"落在框（外扩 margin）内面积占比 < keep_overlap"的连通域（整块邻粒）；
         ② 与框（外扩 margin）求交，裁掉粘连到框外的部分。
    判据用"是否落在框内"而非"只留最大块"：针晶束/复粒这种合法多部件目标不会被误删。

    info 另记框相关诊断（供审计与人工复核）：fg_outside_box_ratio / box_iou（**都按"丢掉框外
    邻粒后"的前景算**——否则会被即将丢掉的邻粒污染，把"框先验本来能修好的多连通域"误判成
    "目标与框外粘连"）/ center_comp_ratio（框中心连通域面积 ÷ 框面积，目标没被框住时明显偏小）
    / 丢弃域数与像素 / 裁剪像素。

    association=region_component 为融合链路的试验策略：中心区域+唯一框内连通域关联，
    用关联域包围范围代替单点/前景填充率下限；歧义、外溢及疑似截断仍review。
    center_point保持原B数值与像素行为，两种策略的mask像素不变。
    """
    if association not in {"center_point", "region_component"}:
        raise ValueError("未知目标关联策略")
    h, w = mask.shape
    x1, y1, x2, y2 = [int(round(float(v))) for v in box[:4]]
    x1, y1 = max(0, min(x1, w - 1)), max(0, min(y1, h - 1))
    x2, y2 = max(x1 + 1, min(x2, w)), max(y1 + 1, min(y2, h))
    bw, bh = x2 - x1, y2 - y1
    mx, my = int(round(bw * margin)), int(round(bh * margin))
    rx1, ry1 = max(0, x1 - mx), max(0, y1 - my)
    rx2, ry2 = min(w, x2 + mx), min(h, y2 + my)
    roi = np.zeros((h, w), bool)
    roi[ry1:ry2, rx1:rx2] = True

    fg = (mask > 0)
    tight = np.zeros((h, w), bool)
    tight[y1:y2, x1:x2] = True
    n, lab, stats, _ = cv2.connectedComponentsWithStats(fg.astype(np.uint8), 8)
    if n > 1:
        areas = stats[:, cv2.CC_STAT_AREA].astype(np.float64)
        inside = np.bincount(lab[roi].ravel(), minlength=n).astype(np.float64)
        ok = (inside / np.maximum(1.0, areas)) >= keep_overlap
        ok[0] = False
        keep = ok[lab]
        dropped_n = int(np.count_nonzero(~ok[1:]))
        dropped_px = int(areas[1:][~ok[1:]].sum())
    else:
        areas = np.zeros(2, np.float64)
        keep, dropped_n, dropped_px = fg.copy(), 0, 0
    kept_px, kept_in = int(keep.sum()), int((keep & tight).sum())
    out = np.where(keep & roi, 255, 0).astype(np.uint8)      # ② 裁掉粘连到框外的部分
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    hit = int(lab[min(h - 1, cy), min(w - 1, cx)]) if n > 1 else (1 if fg.any() else 0)
    info = {}
    info.update(
        box=[x1, y1, x2, y2], box_margin_ratio=margin, box_keep_overlap=keep_overlap,
        fg_outside_box_ratio=round((kept_px - kept_in) / max(1, kept_px), 4),
        box_iou=round(kept_in / max(1, kept_px + int(tight.sum()) - kept_in), 4),
        dropped_components=dropped_n, dropped_px=dropped_px,
        clipped_px=int((keep & ~roi).sum()),
        center_comp_ratio=(round(float(areas[hit]) / float(bw * bh), 3) if hit > 0 else 0.0),
    )
    touches_border = bool(out[0].any() or out[-1].any() or out[:, 0].any() or out[:, -1].any())
    fg_total = int((out > 0).sum())
    border_px = int((out[0] > 0).sum() + (out[-1] > 0).sum()
                    + (out[:, 0] > 0).sum() + (out[:, -1] > 0).sum())
    border_contact_ratio = round(border_px / max(1, fg_total), 6)
    info["touches_border"] = touches_border
    info["border_contact_ratio"] = border_contact_ratio
    if association == "region_component":
        dx, dy = bw * (1 - BOX_CENTER_REGION_RATIO) / 2, bh * (1 - BOX_CENTER_REGION_RATIO) / 2
        sx1, sy1 = int(x1 + dx), int(y1 + dy)
        sx2, sy2 = max(sx1 + 1, int(x2 - dx)), max(sy1 + 1, int(y2 - dy))
        candidates = [int(i) for i in np.unique(lab[tight & keep]) if i > 0]
        regional = [int(i) for i in np.unique(lab[sy1:sy2, sx1:sx2]) if i in candidates]
        components = assoc.components_from_cv(
            lab, stats, tight_box=(x1, y1, x2, y2), roi=roi, keep_overlap=keep_overlap)
        ctx = assoc.AssociationContext(
            components=components, feature_id=feature_id,
            shape=assoc.shape_of(feature_id), granularity=granularity or "instance",
            tight_box=(x1, y1, x2, y2), center_region=(sx1, sy1, sx2, sy2))
        decision = assoc.evaluate(ctx, association_policy or assoc.load_policy())
        selected_ids = tuple(decision.selected)
        dropped_ids = tuple(decision.dropped)
        if decision.method == "dominant_component":
            # 只有 dominant 会重画 mask；unique/multipart/ambiguous 保持原有 keep&roi 像素。
            out = np.where(np.isin(lab, selected_ids) & roi, 255, 0).astype(np.uint8)
        selected = selected_ids[0] if len(selected_ids) == 1 else None
        extent, component_ratio, touches = 0.0, 0.0, False
        if selected is not None:
            x, y, cw, ch, count = stats[selected]
            iw, ih = max(0, min(x + cw, x2) - max(x, x1)), max(0, min(y + ch, y2) - max(y, y1))
            extent = float(iw * ih) / (bw * bh)
            component_ratio = float(count) / (bw * bh)
            selected_mask = (lab == selected) & keep
            touches = bool(selected_mask[0].any() or selected_mask[-1].any()
                           or selected_mask[:, 0].any() or selected_mask[:, -1].any())
        info.update(
            box_association=association, association_version=decision.policy_version,
            center_region_box=[sx1, sy1, sx2, sy2], center_region_hit=bool(regional),
            association_candidates=len(candidates), association_method=decision.method,
            association_status=decision.status, association_reasons=list(decision.reasons),
            association_trace=list(decision.trace),
            association_dropped_components=len(dropped_ids),
            association_dropped_px=int(sum(areas[i] for i in dropped_ids)) if dropped_ids else 0,
            dominance=dict(decision.metrics), width_only=bool(decision.metrics.get("width_only", False)),
            associated_bbox_ratio=round(extent, 6), associated_comp_ratio=round(component_ratio, 6),
            associated_touches_border=touches,
            association_valid=bool(decision.status == "ok" and selected is not None
                                   and component_ratio >= MIN_TARGET_AREA_RATIO
                                   and not touches_border))
    return out, info


def generated_to_mask(gen, alpha, target_hw, letter_meta, threshold=None, invert=None,
                      keep_largest=False, min_area=MIN_AREA_DEFAULT, clean=True,
                      max_hole_ratio=0.0, box=None, box_margin=BOX_MARGIN_RATIO, audit_stages=None,
                      box_association="center_point", feature_id=None,
                      granularity="instance", association_policy=None):
    """IGM 生成图 → 与 target_hw 严格同尺寸的 0/255 uint8 mask，返回 (mask, info)。

    步骤：alpha 直取 → 裁 letterbox 补边 → ROI 灰度 → 阈值（Otsu/固定）→ 极性探针 → 二值化
    → **在生成分辨率上填封闭洞** → INTER_AREA 缩回原图尺寸 → 形态学清理 → 目标框先验
    （box 非空时）→ 状态守卫。

    invert：None（默认）= 由极性探针判定（四角背景探针优先，见 detect_invert）；探针判"暗侧"
    时说明模型把前景画成了暗色、与提示词契约（前景纯白）相悖，置 polarity_conflict 并把
    status 降为 review 交人工。显式 True/False 表示人工指定，直接采用、不报冲突。

    box：crop 局部的检测紧框 [x1,y1,x2,y2]。给了就启用目标框先验（见 apply_box_prior）：
    丢弃框外邻粒并与框求交，避免邻粒并入实例放大几何值。缺省 None = 不做（保持旧行为）。

    max_hole_ratio：封闭洞面积上限（占生成图比例）。**默认 0 = 全填**——簇晶的"暗环带
    + 亮内核"同属目标，内核必须被填成前景；>0 时只填不超过该比例的封闭区（保留大片被
    包围背景，适用于"环形目标"等特殊情形）。

    为什么填洞必须在**降采样之前**（2026-09-23 实测）：马3_b3 生成图 1920px 下环带闭合
    （最大封闭洞 715056px = 目标内核），填洞后占比 0.169→0.366、与检测框 IoU 0.76；
    若先缩到 154px，2~3px 细环带被 INTER_NEAREST 采样打断、内核不再闭合，填洞失效 →
    输出环带（占比 0.174、IoU 0.35），目标内部出现背景空洞（=前景/背景反转）。

    注意：补边在**裁剪**阶段就被丢掉（不参与后续任何映射），故 BORDER_REPLICATE 的边缘
    复制像素不可能渗入最终 mask；最终 mask 尺寸恒等于原图尺寸。
    """
    th, tw = target_hw
    info = {"threshold": None, "invert": None, "polarity_note": "", "source": "gray",
            "area_ratio": 0.0, "n_components": 0, "status": "ok",
            "polarity_hint": None, "polarity_conflict": False}

    use_alpha = alpha is not None and int(alpha.min()) < 255 and int(alpha.max()) > 0
    if use_alpha:
        # provider 返回透明底 PNG：alpha 即天然掩码，无需阈值与极性判断
        fg = (_crop_content(alpha, letter_meta) >= 128).astype(np.uint8) * 255
        info.update(source="alpha", polarity_note="alpha 通道直取")
    else:
        gray = cv2.cvtColor(gen, cv2.COLOR_BGR2GRAY) if gen.ndim == 3 else gen
        roi = _crop_content(gray, letter_meta)
        if threshold is not None:
            t = int(threshold)
            flag = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
            _, fg = cv2.threshold(roi, t, 255, flag)
            info.update(threshold=t, invert=bool(invert),
                        polarity_note="显式 --threshold" + (" + --invert" if invert else ""))
        else:
            t, _ = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            if invert is None:
                inv, hint_note = detect_invert(roi)
                info["polarity_hint"] = bool(inv)
                # 探针判"暗侧"= 模型把前景画成了暗色，与"前景纯白"的提示词契约相悖：
                # 掩码可能是对的（模型确实反着画），但值得人眼扫一下 overlay → 标 review
                info["polarity_conflict"] = bool(inv)
                note = (f"极性探针取{'暗侧' if inv else '亮侧'}"
                        + ("（与提示词契约'前景白'相悖 → review 交人工）" if inv else "")
                        + f"；{hint_note}")
            else:
                inv, note = bool(invert), "显式 --invert/--no-invert（人工指定）"
            flag = cv2.THRESH_BINARY_INV if inv else cv2.THRESH_BINARY
            _, fg = cv2.threshold(roi, t, 255, flag)
            info.update(threshold=float(t), invert=inv, polarity_note=note)

    # 关键：在生成分辨率上填封闭洞（降采样前）——细环带此处仍闭合，内核能被正确填成前景
    if audit_stages is not None:audit_stages['binary']=fg.copy()
    cap = None if max_hole_ratio <= 0 else round(max_hole_ratio * fg.size)
    fg = fill_holes_limited(fg, cap)
    if audit_stages is not None:audit_stages['filled']=fg.copy()
    info["hole_fill"] = "all" if cap is None else f"<={cap}px"

    # 回到原图尺寸：内容区 → (tw, th)。降采样用 INTER_AREA（面积平均）而非点采样：
    # 面积平均对采样网格不敏感（同一 mask 平移一个像素结果几乎不变），面积估计更稳，
    # 适合下游几何测量。实测对照（马3_b3/796_b0 当前 gen_raw）：AREA+128 与 NEAREST+128
    # 的占比差 <0.5%、IoU 差 <0.01，故这是"稳定性选择"而非纠错——细结构（细环带）真正
    # 的修复是"生成分辨率先填洞"（见上）。注意 AREA+128 对**亚像素级**细结构（每块覆盖率
    # <50%）会整体丢弃（点采样会留虚线）；若遇到这类结构，把下面这个 128 调低即可。
    small = cv2.resize(fg, (tw, th), interpolation=cv2.INTER_AREA)
    mask = np.where(small >= RESIZE_FG_LEVEL, 255, 0).astype(np.uint8)
    if audit_stages is not None:audit_stages['resized']=mask.copy()
    if clean:
        mask = clean_mask(mask, min_area, keep_largest)
    if box is not None:
        if audit_stages is not None:audit_stages['pre_box']=mask.copy()
        mask, binfo = apply_box_prior(mask, box, margin=box_margin, association=box_association,
                                      feature_id=feature_id, granularity=granularity,
                                      association_policy=association_policy)
        info.update(binfo)

    area = float((mask > 0).sum()) / float(mask.size)
    info["area_ratio"] = round(area, 6)
    info["n_components"] = max(0, int(cv2.connectedComponents((mask > 0).astype(np.uint8), 8)[0]) - 1)
    box_reasons = []
    if box is not None and box_association == "region_component":
        if not info.get("association_valid"):
            box_reasons.append("TARGET_ASSOCIATION_AMBIGUOUS_OR_TOO_SMALL")
        if area < MIN_TARGET_AREA_RATIO:
            box_reasons.append("TARGET_TOO_SMALL")
        if info.get("associated_comp_ratio", 0) > CENTER_MAX_RATIO:
            box_reasons.append("ASSOCIATED_COMPONENT_TOO_LARGE")
        if info.get("touches_border"):
            box_reasons.append("POSSIBLE_TRUNCATION")
        if info.get("fg_outside_box_ratio", 0) > BOX_OUTSIDE_MAX:
            box_reasons.append("FOREGROUND_OUTSIDE_BOX")
        if info.get("polarity_conflict"):
            box_reasons.append("POLARITY_CONFLICT")
        if info.get("association_status") == "review":
            for reason in info.get("association_reasons", []):
                if reason not in box_reasons:
                    box_reasons.append(reason)
        info["review_reason_codes"] = box_reasons
    if area <= EMPTY_AREA_RATIO:
        info["status"] = "empty"          # 生成图里没找到前景，交人工看 gen_raw
    elif area >= FULL_AREA_RATIO:
        info["status"] = "full"           # 铺满全幅，多半是极性判断失误，交人工
    elif info.get("polarity_conflict"):
        info["status"] = "review"
    elif info.get("association_method") == "dominant_component" and set(box_reasons) <= {
            "POSSIBLE_TRUNCATION"}:
        # 主体已与碎屑分开；仅因目标触边而不能保证完整长度，但宽度仍可用。
        # 这不是“无条件放行”：policy 已要求主体占绝对主导、与框重叠、碎屑不触边。
        info["status"] = "ok"
        info["width_only"] = True
    elif (box_reasons
          or (box is not None and box_association == "center_point" and (
              info.get("touches_border", False)
              or area < MIN_TARGET_AREA_RATIO
              or info.get("center_comp_ratio", 0.0) > CENTER_MAX_RATIO
              or info.get("fg_outside_box_ratio", 0.0) > BOX_OUTSIDE_MAX))):
        # 目标触边（可能截断）/ 前景过小 / **中心块比框还大** / 大片前景落在紧框外
        # （目标与框外物体粘连，框先验只能裁掉框外部分、框内的粘连段会留下 —— 实测 淀粉粒_img3_b5）
        # → 交人工复核。居中性不再作为 review 条件：细长目标可以偏在框一侧，只要不贴边。
        info["status"] = "review"
    if audit_stages is not None:
        audit_stages.setdefault('pre_box',mask.copy());audit_stages['legacy_final']=mask.copy()
    return mask, info


# ---------------------------------------------------------------- 落盘
def write_outputs(out_dir, img, gen_bgr, mask, meta, sent_jpeg=None):
    """写 input_sent.jpg / mask.png / gen_raw.png / overlay.jpg / meta.json。

    input_sent.jpg：**逐字节原样落盘**发给 IGM 的载荷图像（原图等比放大+letterbox 补边后的
    画布，JPEG q95），供目检"放大与补边效果"以及核对模型到底看到了什么。
    """
    os.makedirs(out_dir, exist_ok=True)
    sent_path = os.path.join(out_dir, "input_sent.jpg")
    if sent_jpeg:
        with open(sent_path, "wb") as f:        # 原样写字节，不经再编码
            f.write(sent_jpeg)
    mask_path = os.path.join(out_dir, "mask.png")
    write_img(mask_path, mask, ".png")
    if gen_bgr is not None:
        write_img(os.path.join(out_dir, "gen_raw.png"), gen_bgr, ".png")
    ov = img[:, :, :3].copy() if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if mask.any():
        polys = to_polygon(mask)
        cv2.drawContours(ov, [np.array(p, np.int32) for p in polys if len(p) >= 3],
                         -1, (0, 0, 255), 3)
    write_img(os.path.join(out_dir, "overlay.jpg"), ov, ".jpg")
    meta_path = os.path.join(out_dir, "meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return mask_path, meta_path


# ---------------------------------------------------------------- 自检（纯离线，不耗 API）
def selftest():
    """覆盖后处理链的离线用例：尺寸/值域、极性、多目标、letterbox、alpha、响应解析。"""
    passed, failed = 0, []

    def check(name, cond, extra=""):
        nonlocal passed
        if cond:
            passed += 1
            print(f"  [PASS] {name}")
        else:
            failed.append(f"{name} {extra}")
            print(f"  [FAIL] {name} {extra}")

    def meta_full(gh, gw):
        """生成图即内容区（无 letterbox 补边）。"""
        return {"out_w": gw, "out_h": gh, "content_w": gw, "content_h": gh,
                "top_ratio": 0.0, "left_ratio": 0.0, "cw_ratio": 1.0, "ch_ratio": 1.0,
                "scale": 1.0}

    # ① 生成图尺寸 ≠ 原图尺寸：输出必须严格等于原图尺寸，值域 ⊆ {0,255}
    # 夹具一律用"黑底白对象"：这是提示词契约要求的产物形态（前景纯白、背景纯黑）
    W, H = 300, 200
    GW, GH = 600, 400                                  # 2:1 等比，几何比例不失真
    gen = np.zeros((GH, GW, 3), np.uint8)
    r_gen = 60
    cv2.circle(gen, (GW // 2, GH // 2), r_gen, (255, 255, 255), -1)
    m, info = generated_to_mask(gen, None, (H, W), meta_full(GH, GW))
    check("尺寸: 输出恰为原图尺寸", m.shape == (H, W), f"got {m.shape}")
    check("尺寸: 值域 ⊆ {0,255}", set(np.unique(m)) <= {0, 255}, f"{np.unique(m)}")
    r_mask = r_gen * W / GW                            # 等比缩放后的半径
    exp = math.pi * r_mask ** 2 / (W * H)
    check("尺寸: 圆盘面积占比在容差内", abs(info["area_ratio"] - exp) < 0.02,
          f"got {info['area_ratio']:.3f} exp {exp:.3f}")

    # ② 极性：探针 1 看四角（背景必占四角）；探针 2 在四角不齐时退回环带判据且要求决定性
    #    差距。判"暗侧"意味着模型把前景画成了暗色、与"前景纯白"的契约相悖 → 标 review。
    #    2026-09-29 疤痕：淀粉粒_img3_b1 上环带判据因"实体自身触边"反判成暗侧，mask 盖到背景。
    inv1, _ = detect_invert(cv2.cvtColor(gen, cv2.COLOR_BGR2GRAY))
    check("极性: 黑底白盘（合规产物）判不反转", not inv1, f"got {inv1}")
    check("极性: 合规产物不报极性冲突", info["invert"] is False and info["status"] == "ok"
          and info["polarity_conflict"] is False, f"{info['invert']} {info['status']}")
    inv_bad, note_bad = detect_invert(cv2.cvtColor(255 - gen, cv2.COLOR_BGR2GRAY))
    check("极性: 白底黑盘（四角全亮）判反转", inv_bad and "四角平坦块 4/4 亮" in note_bad,
          f"got {inv_bad} {note_bad}")
    m_bad, i_bad = generated_to_mask(255 - gen, None, (H, W), meta_full(GH, GW))
    check("极性: 白底黑盘 → 取暗侧得到同一 mask，但标 review 交人工",
          i_bad["invert"] is True and i_bad["polarity_conflict"] is True
          and i_bad["status"] == "review" and int((m_bad != m).sum()) == 0,
          f"invert={i_bad['invert']} status={i_bad['status']} diff={int((m_bad != m).sum())}")
    gen5 = np.zeros((GH, GW, 3), np.uint8)
    cv2.circle(gen5, (300, 200), 90, (255, 255, 255), -1)      # 中央目标（亮）
    cv2.rectangle(gen5, (0, 0), (20, 20), (255, 255, 255), -1)  # 占住左上角 → 四角不齐
    inv5, note5 = detect_invert(cv2.cvtColor(gen5, cv2.COLOR_BGR2GRAY))
    check("极性: 四角不齐时退回环带判据（差距不足则取亮侧）",
          inv5 is False and "四角不齐" in note5, f"{inv5} {note5}")
    m_force, i_force = generated_to_mask(255 - gen, None, (H, W), meta_full(GH, GW), invert=True)
    check("极性: --invert 人工指定后与合规产物结果一致且不报冲突",
          int((m_force != m).sum()) == 0 and i_force["status"] == "ok"
          and i_force["polarity_conflict"] is False,
          f"diff={int((m_force != m).sum())} status={i_force['status']}")

    # ③ 多目标 + 噪点：默认保留 2 块，--keep-largest 只留 1 块，5px 噪点被面积过滤
    gen2 = np.zeros((GH, GW, 3), np.uint8)
    cv2.circle(gen2, (180, 200), 80, (255, 255, 255), -1)
    cv2.circle(gen2, (440, 200), 50, (255, 255, 255), -1)
    cv2.circle(gen2, (570, 30), 5, (255, 255, 255), -1)   # 噪点：缩放后面积 < min_area
    _, i3 = generated_to_mask(gen2, None, (H, W), meta_full(GH, GW))
    check("多目标: 默认保留 2 个连通域（小噪点被过滤）", i3["n_components"] == 2,
          f"got {i3['n_components']}")
    _, i4 = generated_to_mask(gen2, None, (H, W), meta_full(GH, GW), keep_largest=True)
    check("多目标: --keep-largest 只留 1 个", i4["n_components"] == 1, f"got {i4['n_components']}")

    # ③b 目标框先验（--box）：场景 = 框内目标 + 框外邻粒（不连通）+ 跨框右缘的条带
    #     （2026-09-29 实测疤痕：淀粉粒_img3_b2 的 4 连通域里 3 个是邻粒，占 mask 39%，
    #      几何把全部连通域点合起来求凸包 → 16.42 μm vs 只留目标 7.61 μm）
    box_out = (70, 70, 230, 130)                       # 输出坐标下紧框，目标恒在框中心
    gb = [v * GW // W for v in box_out]                # 映射到生成图坐标（本例 2:1）
    gen3 = np.zeros((GH, GW, 3), np.uint8)
    cv2.rectangle(gen3, (gb[0] + 10, gb[1] + 2), (gb[2] - 10, gb[3] - 2),
                  (255, 255, 255), -1)                                     # 目标（完全在框内）
    cv2.rectangle(gen3, (0, 20), (100, 116), (255, 255, 255), -1)          # 贴边邻粒（框外）
    cv2.rectangle(gen3, (gb[2] - 25, 190), (gb[2] + 40, 210),
                  (255, 255, 255), -1)                                     # 跨框右缘的条带（测裁剪）
    m_box, i_box = generated_to_mask(gen3, None, (H, W), meta_full(GH, GW), box=box_out)
    _, i_nobox = generated_to_mask(gen3, None, (H, W), meta_full(GH, GW))
    check("框先验: 不给 --box 时保持旧行为（邻粒仍在、无框诊断）",
          i_nobox["n_components"] == 2 and i_nobox.get("box") is None,
          f"n={i_nobox['n_components']} box={i_nobox.get('box')}")
    check("框先验: 框外邻粒被整块丢弃", i_box["dropped_components"] == 1
          and i_box["dropped_px"] > 0, f"got {i_box['dropped_components']} {i_box['dropped_px']}")
    check("框先验: 粘连到框外的部分被裁掉", i_box["clipped_px"] > 0, f"got {i_box['clipped_px']}")
    xs, ys = np.nonzero(m_box)[1], np.nonzero(m_box)[0]
    roi_out = (54, 64, 246, 136)                       # box 外扩 10% 后的范围（输出坐标）
    check("框先验: 输出不含任何框外（含外扩）前景",
          xs.size and xs.min() >= roi_out[0] and ys.min() >= roi_out[1]
          and xs.max() < roi_out[2] and ys.max() < roi_out[3],
          f"bbox=({xs.min()},{ys.min()},{xs.max()},{ys.max()})")
    check("框先验: 目标保住且状态仍为 ok（框外前景占比小）",
          i_box["status"] == "ok" and not i_box.get("touches_border")
          and i_box["fg_outside_box_ratio"] < BOX_OUTSIDE_MAX,
          f"{i_box['status']} border={i_box.get('touches_border')} "
          f"框外={i_box['fg_outside_box_ratio']}")
    # 目标与框外一大块白区粘连（实测 淀粉粒_img3_b5：模型把目标与右上角白区连成一体，
    # 紧框外前景占 35%、白区直抵画幅上/右边界）→ 框先验裁不干净，必须交人工
    gcx, gcy = (gb[0] + gb[2]) // 2, (gb[1] + gb[3]) // 2
    gen6 = np.zeros((GH, GW, 3), np.uint8)
    cv2.circle(gen6, (gcx, gcy), 90, (255, 255, 255), -1)                  # 中央目标
    cv2.rectangle(gen6, (0, gcy - 60), (gcx, gcy + 60), (255, 255, 255), -1)  # 黏到画幅左界的白块
    _, i_stick = generated_to_mask(gen6, None, (H, W), meta_full(GH, GW), box=box_out)
    check("框先验: 大片前景落在紧框外（目标与框外粘连）→ status=review",
          i_stick["status"] == "review" and i_stick["fg_outside_box_ratio"] > BOX_OUTSIDE_MAX,
          f"{i_stick['status']} 框外={i_stick['fg_outside_box_ratio']}")
    # 目标没有铺满框、但偏在一侧且不贴边：旧“必须居中/占框≥30%”会拒绝，新策略应放行
    gen4 = np.zeros((GH, GW, 3), np.uint8)
    cv2.rectangle(gen4, (gcx - 20, gcy - 20), (gcx + 20, gcy + 20), (255, 255, 255), -1)
    _, i_off = generated_to_mask(gen4, None, (H, W), meta_full(GH, GW), box=box_out)
    check("框先验: 目标未铺满框但不贴边 → 不再仅因居中失败拒绝",
          i_off["status"] == "ok" and not i_off.get("touches_border")
          and i_off["center_comp_ratio"] < CENTER_MIN_RATIO,
          f"{i_off['status']} center={i_off['center_comp_ratio']} "
          f"border={i_off.get('touches_border')}")
    # 真正过小的斑点：面积/框低于 MIN_TARGET_AREA_RATIO → review
    gen_tiny = np.zeros((GH, GW, 3), np.uint8)
    cv2.circle(gen_tiny, (gcx, gcy), 10, (255, 255, 255), -1)
    _, i_tiny = generated_to_mask(gen_tiny, None, (H, W), meta_full(GH, GW), box=box_out)
    check("框先验: 真正过小的斑点仍 review",
          i_tiny["status"] == "review" and i_tiny["area_ratio"] < MIN_TARGET_AREA_RATIO,
          f"{i_tiny['status']} area={i_tiny['area_ratio']}")
    # 中心块**比框还大**（旧判据只看下限，这条会漏）：前景略大于紧框、外溢不足 15%（框外占比
    # 那条兜不住）→ 必须由 CENTER_MAX_RATIO 拦下，否则一个"比目标还大的 blob"会被判 ok 进几何。
    gen7 = np.zeros((GH, GW, 3), np.uint8)
    d = 3                                              # 每边只外扩 3px（输出坐标）→ 外溢 <15%
    cv2.rectangle(gen7, ((box_out[0] - d) * GW // W, (box_out[1] - d) * GH // H),
                  ((box_out[2] + d) * GW // W, (box_out[3] + d) * GH // H), (255, 255, 255), -1)
    _, i_over = generated_to_mask(gen7, None, (H, W), meta_full(GH, GW), box=box_out)
    check("框先验: 中心块比紧框还大（外溢<15%）→ status=review",
          i_over["status"] == "review"
          and i_over["center_comp_ratio"] > CENTER_MAX_RATIO
          and i_over["fg_outside_box_ratio"] <= BOX_OUTSIDE_MAX,
          f"{i_over['status']} center={i_over['center_comp_ratio']} "
          f"框外={i_over['fg_outside_box_ratio']}")

    # ④ letterbox：方图尺寸正确、圆心回映误差 ≤2px、补边区内容被彻底丢弃
    side = 512
    src = np.zeros((H, W, 3), np.uint8)                # 黑底白盘（合规产物形态）
    cv2.circle(src, (150, 100), 40, (255, 255, 255), -1)
    canvas, pmeta = letterbox_to(src, side, side)
    check("letterbox: 画布尺寸正确", canvas.shape[:2] == (side, side), f"got {canvas.shape[:2]}")
    check("letterbox: 原图等比缩小后居中", abs(pmeta["cw_ratio"] - 1.0) < 1e-9
          and 0 < pmeta["ch_ratio"] < 1.0, f"meta={pmeta}")
    m5, _ = generated_to_mask(canvas, None, (H, W), pmeta)
    check("letterbox: 输出回到原图尺寸", m5.shape == (H, W), f"got {m5.shape}")
    ys, xs = np.nonzero(m5)
    cx, cy = (float(xs.mean()), float(ys.mean())) if xs.size else (-1.0, -1.0)
    check("letterbox: 圆心回映误差 ≤2px", abs(cx - 150) <= 2 and abs(cy - 100) <= 2,
          f"got ({cx:.1f},{cy:.1f})")
    # 只在补边区画前景（内容区保持纯黑）→ 最终 mask 必须为空：补边被丢弃而非映射到正文
    pad_only = np.zeros((side, side, 3), np.uint8)
    y = int(round((pmeta["top_ratio"] + pmeta["ch_ratio"]) * side)) + 2
    cv2.rectangle(pad_only, (0, y), (side - 1, side - 1), (255, 255, 255), -1)
    m5b, i5b = generated_to_mask(pad_only, None, (H, W), pmeta)
    check("letterbox: 补边区的生成内容被丢弃", i5b["area_ratio"] < 0.02,
          f"got {i5b['area_ratio']:.4f}")

    # ⑤ alpha 通路：透明底 PNG 直接取 alpha，不做阈值/极性判断
    alpha = np.zeros((GH, GW), np.uint8)
    cv2.circle(alpha, (GW // 2, GH // 2), 60, 255, -1)
    gen_a = np.full((GH, GW, 3), 200, np.uint8)
    m6, i6 = generated_to_mask(gen_a, alpha, (H, W), meta_full(GH, GW))
    check("alpha: 走 alpha 直取通路", i6["source"] == "alpha", f"got {i6['source']}")
    check("alpha: 尺寸与面积正确", m6.shape == (H, W)
          and abs(i6["area_ratio"] - exp) < 0.02, f"{m6.shape} {i6['area_ratio']:.3f}")

    # ⑥ 响应解析：缺 padding 的 b64 / data URI / http url / 空 data
    png = cv2.imencode(".png", gen)[1].tobytes()
    b64 = base64.b64encode(png).decode()
    r1, e1 = payload_to_image_bytes({"data": [{"b64_json": b64.rstrip("=")}]})   # 故意去 padding
    check("解析: 缺 padding 的 b64_json 可解", e1 is None and r1 == png, f"err={e1}")
    r2, e2 = payload_to_image_bytes({"data": [{"url": f"data:image/png;base64,{b64}"}]})
    check("解析: data URI 可解", e2 is None and r2 == png, f"err={e2}")
    r3, e3 = payload_to_image_bytes({"data": []})
    check("解析: 空 data 报错而非静默", r3 is None and bool(e3), f"err={e3}")
    r4, e4 = payload_to_image_bytes({"data": [{"url": "http://127.0.0.1:1/x.png"}]})
    check("解析: url 下载失败给明确报错", r4 is None and e4 is not None, f"err={e4}")
    bgr, a, err = decode_generated(png)
    check("解析: 生成图可解码且无 alpha", bgr is not None and a is None and err is None, f"err={err}")

    # ⑦ 参数解析与状态守卫
    check("参数: size 解析 1024x1024", parse_size("1024x1024") == (1024, 1024))
    m_empty, i_empty = generated_to_mask(np.zeros((GH, GW, 3), np.uint8), None, (H, W),
                                         meta_full(GH, GW))
    check("守卫: 全黑生成图 → status=empty 且 mask 全 0",
          i_empty["status"] == "empty" and int(m_empty.sum()) == 0,
          f"{i_empty['status']} sum={int(m_empty.sum())}")
    # 显式 --no-invert 时亮侧即前景：近全白图会被判为铺满 → status=full（交人工）
    gen_full = np.full((GH, GW, 3), 255, np.uint8)
    cv2.circle(gen_full, (GW // 2, GH // 2), 10, (0, 0, 0), -1)
    _, i_full = generated_to_mask(gen_full, None, (H, W), meta_full(GH, GW), invert=False)
    check("守卫: 强制不反转时近全白图 → status=full（交人工）", i_full["status"] == "full",
          f"got {i_full['status']} area={i_full['area_ratio']}")

    # 填洞语义 + 时机（2026-09-23 修正）：环带包住的内核同属目标 → 填成前景；
    # 且必须在生成分辨率上填（降采样会打断细环带使填洞失效）
    ring = np.zeros((64, 64), np.uint8)
    cv2.circle(ring, (32, 32), 24, 255, 2)           # 细环带（2px）
    ring[31:34, 54:58] = 0                           # 在环带上开缺口 → 内核与外部连通
    m_gap = fill_holes_limited(ring, None)
    check("填洞: 环带有缺口(内核不闭合) → 不填（诚实保留，不臆造）", m_gap[32, 32] == 0)
    closed = np.zeros((64, 64), np.uint8)
    cv2.circle(closed, (32, 32), 24, 255, 2)         # 闭合细环带
    m_cl = fill_holes_limited(closed, None)
    check("填洞: 闭合细环带 → 内核填成前景（簇晶暗环带+亮内核）", m_cl[32, 32] == 255)
    blk = np.zeros((64, 64), np.uint8)
    blk[16:48, 16:48] = 255
    blk[30:33, 30:33] = 0                            # 3x3 小洞（9px）
    check("填洞: 小洞照填", fill_holes_limited(blk, None)[31, 31] == 255)
    check("填洞: 上限 8px 时 9px 小洞不填（--max-hole-ratio 守门生效）",
          fill_holes_limited(blk, 8)[31, 31] == 0)
    edge = np.zeros((64, 64), np.uint8)
    edge[16:48, 16:48] = 255
    edge[16:48, 40] = 0                              # 开一条通到画面外的通道
    m_edge = fill_holes_limited(edge, None)
    check("填洞: 通到边界的背景（通道）永不填", m_edge[31, 40] == 0 and m_edge[5, 5] == 0)
    # 端到端：细环带在生成分辨率闭合、缩到 1/16 后必然被打断 → 只有"先填后缩"才能填成实心
    GH = GW = 400
    gen_ring = np.full((GH, GW, 3), 255, np.uint8)
    cv2.circle(gen_ring, (200, 200), 150, (0, 0, 0), 2)
    m_e2e, i_e2e = generated_to_mask(gen_ring, None, (25, 25), meta_full(GH, GW),
                                     invert=True, min_area=5)
    check("端到端: 生成分辨率先填洞 → 缩回后内核仍为前景", m_e2e[12, 12] == 255,
          f"center={m_e2e[12, 12]} area={i_e2e['area_ratio']}")
    check("端到端: 输出尺寸严格等于目标尺寸", m_e2e.shape == (25, 25))

    # 画布尺寸自适应（2026-09-23）：满足像素下限所需的最小倍率，且保持原图长宽比
    W1, H1, _ = pick_size(154, 152)
    check("pick_size: 小方图 154x152 → 刚好达标 1920x1920",
          W1 == 1920 and H1 == 1920, f"got {W1}x{H1}")
    W2, H2, _ = pick_size(2048, 1536)
    check("pick_size: 4:3 图长宽比保持（等比选尺寸）",
          abs(math.log((W2 / H2) / (2048 / 1536))) < 0.05, f"got {W2}x{H2}")
    _, lb2 = letterbox_to(np.zeros((1152, 1536, 3), np.uint8), W2, H2)
    pad2 = 1.0 - lb2["cw_ratio"] * lb2["ch_ratio"]
    check("pick_size: 4:3 图补边 <5%（原固定 1920x1920 时约 25%）",
          pad2 < 0.05 and max(W2, H2) <= MAX_SIDE_DEFAULT, f"got {W2}x{H2} pad={pad2:.3f}")
    W3, H3, _ = pick_size(8000, 6000)
    check("pick_size: 超大图用满单边上限而非缩到下限（保细节）",
          max(W3, H3) <= MAX_SIDE_DEFAULT and max(W3, H3) >= 4000, f"got {W3}x{H3}")
    check("pick_size: 边长均对齐到 quantum",
          all(v % SIZE_QUANTUM_DEFAULT == 0 for v in (W1, H1, W2, H2, W3, H3)))
    for sw, sh in ((154, 152), (2048, 1536), (8000, 6000), (669, 744), (326, 344)):
        pw, ph, _ = pick_size(sw, sh)
        check(f"pick_size: {sw}x{sh} 面积达下限且未超单边上限",
              pw * ph >= MIN_PIXELS_DEFAULT and max(pw, ph) <= MAX_SIDE_DEFAULT,
              f"got {pw}x{ph} area={pw * ph}")
    # 极端长宽比：像素下限（已确认的 API 硬约束）优先于单边上限（未实测），须给出警示而非静默
    pw, ph, note = pick_size(100, 900)
    check("pick_size: 极端长宽比以下限优先且给出 note 警示",
          pw * ph >= MIN_PIXELS_DEFAULT and max(pw, ph) > MAX_SIDE_DEFAULT and bool(note),
          f"got {pw}x{ph} note={note!r}")
    # 回缩保真：细结构靠"生成分辨率先填洞"保证闭合；本处锁定回缩的面积保真性
    disc = np.zeros((200, 200), np.uint8)
    cv2.circle(disc, (100, 100), 60, 255, -1)
    canvas_fill = float((disc > 0).mean())
    m_disc, i_disc = generated_to_mask(cv2.cvtColor(disc, cv2.COLOR_GRAY2BGR), None, (50, 50),
                                       meta_full(200, 200), invert=False, min_area=5)
    check("回缩: 面积占比与生成分辨率一致（面积平均保真，±20%）",
          abs(i_disc["area_ratio"] - canvas_fill) < 0.2 * canvas_fill,
          f"{i_disc['area_ratio']} vs {canvas_fill:.3f}")

    # 载荷落盘：发给模型的那张画布要原样存下来（同源字节，便于目检放大/补边）
    import tempfile
    _td = tempfile.mkdtemp(prefix="igm_selftest_")
    _img = np.full((60, 80, 3), 200, np.uint8)
    _jpeg = encode_image_jpeg(_img)
    check("载荷: encode_image_jpeg 产出可解码 JPEG",
          cv2.imdecode(np.frombuffer(_jpeg, np.uint8), cv2.IMREAD_COLOR) is not None)
    check("载荷: base64 串与字节同源",
          base64.b64decode(encode_image_b64(_img)) == _jpeg)
    write_outputs(_td, _img, None, np.zeros((60, 80), np.uint8), {}, sent_jpeg=_jpeg)
    _p = os.path.join(_td, "input_sent.jpg")
    check("载荷: write_outputs 落盘 input_sent.jpg 且字节与载荷一致",
          os.path.exists(_p) and open(_p, "rb").read() == _jpeg)
    import shutil as _sh
    _sh.rmtree(_td, ignore_errors=True)

    total = passed + len(failed)
    if failed:
        print(f"\nSELFTEST FAIL {passed}/{total}")
        for f in failed:
            print(f"  - {f}")
        return 1
    print(f"\nSELFTEST PASS {passed}/{total}")
    return 0


# ---------------------------------------------------------------- 主流程
def pick_image_dialog():
    """弹出文件资源管理器选择原图（tkinter；取消选择返回空串）。

    置顶 + 焦点还原：Windows 下对话框偶发被埋在后台，attributes("-topmost") 保证可见。
    """
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError:
        sys.exit("[FAIL] 缺少 tkinter，无法打开文件选择对话框；请改用 --image 指定路径")
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    path = filedialog.askopenfilename(
        title="选择待分割原图",
        filetypes=[("图像文件", "*.tif *.tiff *.jpg *.jpeg *.png *.bmp"),
                   ("所有文件", "*.*")])
    root.destroy()
    return path or ""


def main():
    ap = argparse.ArgumentParser(
        description="IGM（图像生成大模型）分割：原图 + 分割指令 → 与原图同尺寸的二值 mask")
    ap.add_argument("--image", nargs="?", default=None,
                    help="原图路径（可省略：省略时弹出文件资源管理器选择图片）")
    ap.add_argument("--feature", default=DEFAULT_FEATURE,
                    help="特征类型（feature_taxonomy.json 的 feature_id，如 cluster_crystal/vessel/"
                         "stone_cell/starch_grain/needle_crystal…），据此选对应形态描述注入模板")
    ap.add_argument("--desc", default=None,
                    help="前景特征描述自由文本；给了就覆盖 --feature 的查表结果")
    ap.add_argument("--prompt", default=None,
                    help="完整提示词；给了就直接用（绕过模板与描述），用于完全自定义")
    ap.add_argument("--out-dir", default="", help="输出目录（默认 results/segmentation/igm_seg/<stem>）")
    ap.add_argument("--size", default="",
                    help="发给 IGM 的生成尺寸 WxH（默认：按配置 size_mode=auto 自适应长宽比，"
                         "否则取配置 igm.size）")
    ap.add_argument("--threshold", type=int, default=None, help="固定二值化阈值（默认 Otsu）")
    ap.add_argument("--invert", dest="invert", action="store_true", default=None,
                    help="强制前景取暗侧（默认：按提示词契约取亮侧；环带判据只作提示并触发 review）")
    ap.add_argument("--no-invert", dest="invert", action="store_false",
                    help="强制前景取亮侧（默认行为，仅在需要显式声明时使用）")
    ap.add_argument("--box", default="",
                    help="crop 局部检测紧框 x1,y1,x2,y2（来自 crop 随行 json 的 tight_box_local）。"
                         "给了就启用目标框先验：丢弃框外邻粒并与框求交，只留本 crop 的目标")
    ap.add_argument("--box-association", choices=["center_point", "region_component"], default="center_point",
                    help="目标关联质量策略；原B默认单点，新融合显式采用区域/连通域")
    ap.add_argument("--granularity", default="instance",
                    help="crop 框粒度（instance/bundle/sheet/clump）；非 instance 不启用 dominant 连通域策略")
    ap.add_argument("--association-policy", default="",
                    help="关联策略 JSON 路径；缺省读 图像分割/config/association_policy.json")
    ap.add_argument("--box-margin", type=float, default=BOX_MARGIN_RATIO,
                    help=f"目标框先验的外扩比例（占框宽高，默认 {BOX_MARGIN_RATIO}）")
    ap.add_argument("--keep-largest", action="store_true",
                    help="只保留最大连通域（默认保留全部，仅按 --min-area 去噪）")
    ap.add_argument("--min-area", type=int, default=MIN_AREA_DEFAULT, help="连通域最小面积 px")
    ap.add_argument("--max-hole-ratio", type=float, default=0.0,
                    help="封闭洞填洞上限（占生成图比例）：默认 0 = 全填（簇晶'暗环带+亮内核'"
                         "同属目标，内核须填成前景）；>0 时只填不超过该比例的封闭区")
    ap.add_argument("--no-clean", action="store_true", help="跳过形态学清理与连通域过滤")
    ap.add_argument("--force", action="store_true", help="覆盖已有 mask")
    ap.add_argument("--dry-run", action="store_true", help="只打印解析出的请求，不发 API")
    ap.add_argument("--quiet", action="store_true",
                    help="只打印每图一行结果与错误（批处理用）；提示词、阶段进度、耗时等详细"
                         "日志静默")
    ap.add_argument("--selftest", action="store_true", help="跑离线自检（不耗 API）")
    args = ap.parse_args()
    set_quiet(args.quiet)

    if args.selftest:
        return selftest()
    if not args.image:
        print("[info] 未指定 --image，请通过文件资源管理器选择原图…")
        args.image = pick_image_dialog()
    if not args.image:
        fail("1/7 输入", "未选择图片（或使用 --image 指定路径 / --selftest）")

    # 输出目录与幂等判定：先判 skip，避免为已有产物白读大图（TIF 可能很大）
    stem = os.path.splitext(os.path.basename(args.image))[0]
    out_dir = args.out_dir or os.path.join(IGM_DIR, stem)
    mask_path = os.path.join(out_dir, "mask.png")
    if os.path.exists(mask_path) and not args.force and not args.dry_run:
        result(f"[跳过] {mask_path} 已存在（--force 覆盖）")
        return 0

    # [1/7 输入]
    img = imread_any(args.image)
    if img is None:
        fail("1/7 输入", f"读不到原图: {args.image}")
    h, w = img.shape[:2]
    n_ch = img.shape[2] if img.ndim == 3 else 1
    stage("1/7 输入", f"{args.image} | {w}x{h} | {n_ch}通道"
                      + ("（含 alpha）" if n_ch == 4 else ""))

    # [2/7 提示词] 优先级：--prompt 整句覆盖 > --desc 自由文本 > --feature 查特征本体表
    if args.prompt:
        prompt, desc, prompt_source = args.prompt, None, "cli:--prompt(绕过模板)"
    else:
        desc = args.desc if args.desc else load_feature_desc(args.feature)
        prompt = build_prompt(desc)
        prompt_source = "模板+--desc" if args.desc else f"模板+本体表({args.feature})"
    stage("2/7 提示词", f"来源={prompt_source}"
                        + (f" | 特征={args.feature}" if not args.prompt else ""))
    box_print("发给 IGM 的提示词（完整）", prompt)

    cfg_igm, cfg_vlm = load_igm_cfg()
    api_key, key_src = resolve_igm_key(cfg_igm, cfg_vlm)
    # 画布尺寸：--size 显式优先；否则 size_mode=auto 按原图长宽比自适应（见 pick_size），
    # 不然沿用配置里的固定 size。自适应会把画布面积真正用于内容，避免固定方形画布的补边浪费。
    size_note = ""
    if args.size:
        size_mode = "cli:--size"
        size = args.size
    elif (cfg_igm.get("size_mode") or "fixed").lower() == "auto":
        size_mode = "auto"
        side_w, side_h, size_note = pick_size(
            w, h, cfg_igm.get("min_pixels", MIN_PIXELS_DEFAULT),
            cfg_igm.get("max_side", MAX_SIDE_DEFAULT),
            cfg_igm.get("size_quantum", SIZE_QUANTUM_DEFAULT))
        size = f"{side_w}x{side_h}"
    else:
        size_mode = "fixed"
        size = cfg_igm.get("size") or "1024x1024"
    side_w, side_h = parse_size(size)

    canvas, letter_meta = letterbox_to(img, side_w, side_h)
    # 编码一次：既作 API 载荷（base64），又原样落盘成 input_sent.jpg 供目检（同源，不重复编码）
    sent_jpeg = encode_image_jpeg(canvas)
    b64 = base64.b64encode(sent_jpeg).decode()
    pad_ratio = round(1.0 - letter_meta["cw_ratio"] * letter_meta["ch_ratio"], 4)
    stage("3/7 画布", f"{size}（{size_mode}）| 缩放 {letter_meta['scale']:.2f}× | "
                      f"补边 {pad_ratio * 100:.1f}% | 载荷 JPEG q95 {len(sent_jpeg) / 1024:.0f}KB"
                      f" → base64 {len(b64) / 1024:.0f}KB | 模型={cfg_igm['model']}")
    stage("3/7 画布", f"密钥来源={key_src or '（空，将失败）'}")
    if size_note:
        stage("3/7 画布", f"[warn] {size_note}")

    if args.dry_run:
        stage("4/7 调用", "dry-run：只打印请求，不发 API")
        call_igm(cfg_igm, api_key, b64, prompt, size, dry_run=True)
        return 0
    if not api_key:
        fail("4/7 调用", "未配置 IGM API key（环境变量 IGM_API_KEY / igm.api_key / vlm.api_key 皆为空）")

    t0 = time.time()
    payload, err = call_igm(cfg_igm, api_key, b64, prompt, size)
    t_call = time.time()
    if payload is None:
        fail("4/7 调用", f"IGM 调用失败 | url={cfg_igm['base_url']} | model={cfg_igm['model']}"
                         f" | size={size} | {err}")
    raw, err = payload_to_image_bytes(payload)
    if raw is None:
        fail("4/7 调用", f"取生成图失败: {err}")
    gen_bgr, alpha, err = decode_generated(raw)
    if gen_bgr is None:
        fail("5/7 解码", f"{err}")
    t_dec = time.time()
    stage("5/7 解码", f"生成图 {gen_bgr.shape[1]}x{gen_bgr.shape[0]} | "
                      f"内容区 {letter_meta['content_w']}x{letter_meta['content_h']} | "
                      f"alpha={'有' if alpha is not None else '无'}")

    mask, info = generated_to_mask(
        gen_bgr, alpha, (h, w), letter_meta, threshold=args.threshold, invert=args.invert,
        keep_largest=args.keep_largest, min_area=args.min_area, clean=not args.no_clean,
        max_hole_ratio=args.max_hole_ratio, box=parse_box(args.box), box_margin=args.box_margin,
        box_association=args.box_association, feature_id=args.feature,
        granularity=args.granularity,
        association_policy=(assoc.load_policy(args.association_policy)
                            if args.association_policy else None))
    t_post = time.time()
    clean_txt = "跳过(--no-clean)" if args.no_clean else f"close5/open3/面积≥{args.min_area}"
    stage("6/7 后处理", f"通路={info['source']} | 极性 invert={info['invert']}"
                        f"（{info['polarity_note']}）| 阈值={info['threshold']} | "
                        f"填洞={info.get('hole_fill')} | 回缩 INTER_AREA≥{RESIZE_FG_LEVEL}")
    stage("6/7 后处理", f"清理={clean_txt} | 保留最大域={args.keep_largest}")
    if info.get("box") is not None:
        stage("6/7 后处理", f"目标框先验 box={info['box']}（外扩 {info['box_margin_ratio']}）| "
                            f"丢弃框外邻粒 {info['dropped_components']} 块/{info['dropped_px']}px | "
                            f"裁剪框外粘连 {info['clipped_px']}px | "
                            f"框内占比 {info.get('fg_outside_box_ratio')} 为框外 | "
                            f"中心块/框 {info.get('center_comp_ratio')}")
    if info.get("polarity_conflict"):
        stage("6/7 后处理", "[warn] 极性探针与提示词契约相悖 → status=review，请人工核对该 mask")
    if info.get("status") == "review" and info.get("box") is not None:
        stage("6/7 后处理", f"[warn] 框先验异常（中心块/框 {info.get('center_comp_ratio')}、"
                            f"框外占比 {info.get('fg_outside_box_ratio')}）→ status=review，"
                            f"多为目标与框外物体粘连或框偏大，请人工核对")

    meta = {
        "image": os.path.abspath(args.image), "image_hw": [h, w], "stem": stem,
        "prompt": prompt, "prompt_source": prompt_source,
        "feature": args.feature, "feature_desc": desc,
        "model": cfg_igm["model"], "base_url": cfg_igm["base_url"],
        "size_requested": size, "size_mode": size_mode, "size_note": size_note or None,
        "pad_ratio": pad_ratio,
        "input_sent": "input_sent.jpg", "input_sent_hw": [side_h, side_w],
        "gen_hw": list(gen_bgr.shape[:2]), "letterbox": letter_meta,
        "key_source": key_src, "threshold": info["threshold"], "invert": info["invert"],
        "polarity_note": info["polarity_note"], "mask_source": info["source"],
        "polarity_hint": info.get("polarity_hint"),
        "polarity_conflict": bool(info.get("polarity_conflict")),
        "box": info.get("box"), "box_margin_ratio": info.get("box_margin_ratio"),
        "box_keep_overlap": info.get("box_keep_overlap"),
        "box_iou": info.get("box_iou"),
        "fg_outside_box_ratio": info.get("fg_outside_box_ratio"),
        "center_comp_ratio": info.get("center_comp_ratio"),
        **{k: info[k] for k in ("box_association", "association_version", "center_region_box",
             "center_region_hit", "association_candidates", "association_method", "associated_bbox_ratio",
             "associated_comp_ratio", "associated_touches_border", "association_valid",
             "association_status", "association_reasons", "association_trace", "dominance",
             "width_only", "association_dropped_components", "association_dropped_px",
             "review_reason_codes") if k in info},
        "dropped_components": info.get("dropped_components"),
        "dropped_px": info.get("dropped_px"), "clipped_px": info.get("clipped_px"),
        "max_hole_ratio": args.max_hole_ratio,
        "hole_fill": info.get("hole_fill"),
        "area_ratio": info["area_ratio"], "n_components": info["n_components"],
        "status": info["status"], "mask_hw": list(mask.shape),
        "size_match": list(mask.shape) == [h, w],
        "revised_prompt": (payload.get("data") or [{}])[0].get("revised_prompt"),
        "elapsed_s": round(time.time() - t0, 1),
        "note": "IGM 是现役图像分割通路的产物（非预标注件）；status!=ok 的实例在判药侧被整条丢弃，尺寸量值仅在 status==ok 时可用",
    }
    write_outputs(out_dir, img, gen_bgr, mask, meta, sent_jpeg=sent_jpeg)
    t_end = time.time()
    outputs = ("input_sent.jpg", "mask.png", "gen_raw.png", "overlay.jpg", "meta.json")
    stage("7/7 落盘", " | ".join(
        f"{n} {os.path.getsize(os.path.join(out_dir, n)) / 1024:.1f}KB"
        for n in outputs if os.path.exists(os.path.join(out_dir, n))))
    stage("7/7 落盘", f"总耗时 {t_end - t0:.1f}s（API {t_call - t0:.1f}s"
                      f" + 解码 {t_dec - t_call:.1f}s + 后处理 {t_post - t_dec:.1f}s"
                      f" + 落盘 {t_end - t_post:.1f}s）")
    if not meta["size_match"]:
        stage("7/7 落盘", f"[warn] mask 尺寸 {mask.shape} 与原图 {(h, w)} 不一致（不应发生，请检查）")
    # 每图一行结果：--quiet 下仍打印；状态非 ok 时带显式警示，避免坏产物被当合格品
    result(f"[结果] {stem}: mask {mask.shape[1]}x{mask.shape[0]} | 占比 {info['area_ratio']:.3f} | "
           f"连通域 {info['n_components']} | 阈值 {info['threshold']} | invert={info['invert']} | "
           f"状态 {info['status']}" + ("" if info["status"] == "ok" else "  ⚠需人工确认")
           + (f" | 框先验: 丢邻粒 {info['dropped_components']} 块, 裁 {info['clipped_px']}px"
              if info.get("box") is not None else ""))
    result(f"[输出] {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
