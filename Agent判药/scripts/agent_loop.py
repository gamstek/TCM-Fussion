# -*- coding: utf-8 -*-
"""agent_loop — Agent（VLM）证据补全环（设计方案 §5.3，神经层）。

设计铁律：
  - 神经层只"补证据"，不改裁决：VLM 判读结果以"存在性证据"注入证据卡池
    （conf 由 VLM 给出，标记 conf_assumed），随后**整体重跑规则引擎**得出结论；
  - 有界循环：默认每次判药 ≤ --max-calls 次 VLM 调用；
  - 介入时机（§5.3）：结论为 probable / insufficient_evidence / needs_review，
    且 top 候选存在"检测器未见"的 required 判据（v1.0 仅对 P1/C 级特征发起，
    P0 级缺失信任检测链灵敏度）；
  - 场景判读（2026-09-22 用户决议：强制触发）：样本检出结晶类特征
    （scene_attributes.CRYSTAL_FEATURES）即逐视野发起 scene_check，不限裁决
    结果——宿主语境乘数 s(scene) 参与判据贡献评分；
  - 场景判读与候选身份解耦（2026-09-23 决议）：scene_check **不接收**候选 morph
    提示；结晶位置二值判读（位于/不位于细胞内，不设"无法判断"），并以 visual 前提
    cell_wall_visible 守卫胞内判读（详见 VLMClient.scene_check）；
  - VLM 不可用/调用失败 → 静默降级为 rule_only，报告标记 degraded。

VLM 配置复用上游 标注体系/detection/vlm_config.json（只读），OpenAI 兼容接口。
"""
import base64
import json
import re
import sys
import urllib.request
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rule_engine import EvidenceCard, decide  # noqa: E402
import scene_attributes  # noqa: E402

# 证据图索引与下发格式。源库原图多为 tif（白芍/马齿苋的 images/ 全是 .tif），
# 故 tif/tiff 必须进索引；但**只进索引不够**——TIFF 字节不能原样下发（mime 也没得填），
# 必须解码后转 JPEG。jpg/png 在尺寸允许时保持原样下发，不重编码、零画质损失。
_RAW_MIME = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}
_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")

# 下发前长边上限（px）。原图 4K/2K 原样下发会撞 API 读超时——实测 base64 载荷：
# 白芍 tif(3840x2160) 2.0MB、虎杖/马齿苋 jpg(2048x1536) 2.4~2.7MB，2026-09-23 白芍
# 报告的 "vlm_error: The read operation timed out" 即此（首个视野超时后 break，
# 导致整个样本一条场景判读都没有 → 宿主语境乘数静默失效 → 结论随网络时快时慢）。
# 1536 与检测阶段（detect_sample 的 min(1.0, 1536/max)）同一口径，
# 场景判读只需判断"能否看见细胞结构"，该分辨率足够。
IMAGE_MAX_SIDE = 1536


class VLMClient:
    """glm 系列 OpenAI 兼容客户端（最小实现：图像 + 文本 → JSON）。"""

    def __init__(self, config_path, timeout_s=60):
        cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
        vlm = cfg.get("vlm", cfg)
        self.base_url = vlm["base_url"].rstrip("/")
        self.model = vlm.get("model", "glm-4.6v")
        self.api_key = vlm.get("api_key") or ""
        if not self.api_key and vlm.get("api_key_env"):
            import os
            self.api_key = os.environ.get(vlm["api_key_env"], "")
        self.temperature = float(vlm.get("temperature", 0.1))
        self.max_tokens = int(vlm.get("max_tokens", 512))
        # 超时以配置为准（vlm_config.json 的 vlm.timeout_s，本仓库为 180s）；缺项才用入参默认。
        # 原实现只认入参、从不读配置，而 judge.py 按 VLMClient(path) 构造 → 实跑一直是 60s，
        # 与配置写的 180s 不符（2026-09-23 白芍场景判读超时即撞在这 60s 上）。
        self.timeout_s = float(vlm.get("timeout_s", timeout_s))

    def _post(self, payload):
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _encode_image(self, image_path):
        return _image_data_url(image_path)

    def presence_check(self, feature_zh, morph_hint, image_path):
        """图像级有无判定 → {present, conf, reason}（结构化输出，设计方案 §4.2）。"""
        prompt = (
            "你是中药粉末显微鉴别专家。请判断这张明视野显微照片中是否出现以下特征：\n"
            f"特征：{feature_zh}\n形态提示：{morph_hint or '（无）'}\n"
            '只输出 JSON：{"present": true/false, "conf": 0~1, "reason": "一句话依据"}'
        )
        try:
            # 编码放进 try：图像读不出/格式不收时必须降级为 ok=False，而不是抛出去中断整轮判药
            payload = {
                "model": self.model, "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "messages": [{"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": self._encode_image(image_path)}},
                    {"type": "text", "text": prompt},
                ]}],
            }
            data = self._post(payload)
            text = data["choices"][0]["message"]["content"]
            m = re.search(r"\{.*\}", text, re.S)
            obj = json.loads(m.group(0)) if m else {}
            return {"present": bool(obj.get("present")), "conf": float(obj.get("conf", 0.5)),
                    "reason": str(obj.get("reason", ""))[:200], "ok": True}
        except Exception as e:  # noqa: BLE001 — VLM 失败必须降级而不是中断判药
            return {"present": False, "conf": 0.0, "reason": f"vlm_error: {e}", "ok": False}

    def scene_check(self, image_path):
        """场景判读（三层模型第 2 层）：单视野图 → 视觉前提 + 结晶位置/排列/数量级。

        结构化输出，仿 presence_check；conf 为判读置信度（宿主语境乘数的评分门限
        SCENE_LOC_CONF_GATE 用）；失败返回 ok=False 由调用方静默降级。

        **与候选身份解耦**（2026-09-23 决议）：不再接收候选 morph 提示。候选 morph 原文
        含"簇晶存在于薄壁细胞中""常排列成行"，与被问维度同源，等于把答案写进问句——
        实测中 VLM 复述过候选名（"符合马齿苋或赤芍牡丹皮的簇晶特征"）并据此误判
        intracellular，把真值从 top1 压到 top3。

        **二值语义 + 视觉前提守卫**（cell_wall_visible，2026-09-23 用户决议）：
        crystal_location 只判"位于细胞内/不位于细胞内"两值，**不设"无法判断"**——
        画面中看不见细胞结构，本身就意味着结晶不位于细胞内（游离散在）。要求 VLM 先答
        "画面中能否看见细胞壁轮廓/细胞腔"，再答结晶位置；"位于细胞内"是正向主张，须以
        画面可见细胞结构举证：cell_wall_visible=false（含字段缺失/非 true）或模型答出
        二值之外的词（unknown/mixed/未识别值）一律判 free。原答留在 crystal_location_raw，
        guarded 标记改判；改判结果照常经 rule_engine._scene_location_factor 参与乘数
        （free 对 morph 含"细胞内含"的判据取相悖 ×0.60，对无相关表述者恒 1.0）。
        明场未染色时细胞壁常不可见，模型易以"簇晶位于薄壁细胞内"的药材学常识补足视觉
        缺失，该守卫即为此设。
        """
        # 措辞为 A/B 实测选定（v7，2026-09-24）：9 张真值图 × 3 轮，细胞内 15/18、
        # 游离 9/9、合计 26/33；对照的"收紧型"改法全部更差（v4 2/6、v5 0/6、v8 6/18、
        # v9 1/18，Fisher p=0.0059 / <0.0001）——该模型对"判定规则/举证责任/排除清单"
        # 一类约束过度敏感，会整体倒向 false。故本文案只做两件事：
        #   ① 删掉单向否证暗示（"看不见就答 false，不要因为粉末里通常有薄壁细胞就答 true"）；
        #   ② 加**一条**与正向描述并列的点名区分线索（簇晶自身不是细胞）。
        # 再加细节即回到过度约束，勿在无 A/B 复核的情况下扩写。
        prompt = (
            "你是中药粉末显微鉴别专家。请只依据画面中实际可见的结构作答，不要用药材学常识替代观察。\n"
            "第 0 步 cell_wall_visible：画面中是否存在细胞结构？出现下列任一形态即答 true——"
            "成片或成排排列的圆形/多角形细胞腔、断续但仍可辨认的细胞壁轮廓、"
            "细胞壁围成的多边形网格。以下**不构成**细胞结构、不得据此答 true："
            "孤立的圆形颗粒、气泡、油滴、离散的散点或杂质，"
            "以及**草酸钙簇晶自身**——簇晶呈多角星状或花瓣状团块、有折光边缘，"
            "很容易被误认成细胞，请勿把单个结晶颗粒的轮廓当作细胞结构；"
            "细胞结构应是多个相邻腔室成片排列、彼此能看到边界。"
            "若整幅画面只有孤立颗粒落在均匀空白的背景上、看不到任何上述排列，才答 false。\n"
            "第 1 步 crystal_location（二值，不答'无法判断'）："
            "intracellular＝结晶处在细胞结构所在的区域内；"
            "free＝结晶落在完全没有细胞结构的空白背景上、四周无可辨认的细胞。\n"
            "第 2 步 arrangement：多个结晶排成一行(rows)、散乱分布(scattered)、无法判断(unknown)；\n"
            "第 3 步 abundance：画面中结晶数量级，很少(few)、中等(many)、很多(very_many)、"
            "无法判断(unknown)。\n"
            "第 2/3 步只看结晶自身的排布与多寡，不以是否看见细胞结构为前提。\n"
            '只输出 JSON：{"cell_wall_visible": true/false, "crystal_location": "intracellular/free", '
            '"arrangement": "...", "abundance": "...", "conf": 0~1, "reason": "一句话依据"}'
        )
        arr_allowed = {"rows", "scattered", "unknown"}
        abd_allowed = {"few", "many", "very_many", "unknown"}
        try:
            # 编码放进 try：图像读不出/格式不收时必须降级为 ok=False，而不是抛出去中断整轮判药
            payload = {
                "model": self.model, "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "messages": [{"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": self._encode_image(image_path)}},
                    {"type": "text", "text": prompt},
                ]}],
            }
            data = self._post(payload)
            text = data["choices"][0]["message"]["content"]
            m = re.search(r"\{.*\}", text, re.S)
            obj = json.loads(m.group(0)) if m else {}
            raw_vis = obj.get("cell_wall_visible")           # 缺失/字符串"true" 一并规整
            vis = raw_vis is True or str(raw_vis).strip().lower() == "true"
            raw_loc = str(obj.get("crystal_location", "unknown")).strip().lower()
            # 二值语义（2026-09-23 用户决议）：只判"位于细胞内/不位于细胞内"。
            # "位于细胞内"是正向主张、须以画面可见细胞结构举证，故 cell_wall_visible=false
            # 或答出二值之外的词（unknown/mixed/未识别值）一律判 free；原答留痕供审计。
            loc = "intracellular" if (raw_loc == "intracellular" and vis) else "free"
            arr = str(obj.get("arrangement", "unknown"))
            abd = str(obj.get("abundance", "unknown"))
            try:
                conf = float(obj.get("conf", 0.5))
            except (TypeError, ValueError):
                conf = 0.5
            return {"crystal_location": loc, "crystal_location_raw": raw_loc,
                    "cell_wall_visible": vis, "guarded": loc != raw_loc,
                    "arrangement": arr if arr in arr_allowed else "unknown",
                    "abundance": abd if abd in abd_allowed else "unknown",
                    "conf": min(max(conf, 0.0), 1.0),
                    "reason": str(obj.get("reason", ""))[:200], "ok": True}
        except Exception as e:  # noqa: BLE001 — VLM 失败必须降级而不是中断判药
            return {"crystal_location": "unknown", "crystal_location_raw": "unknown",
                    "cell_wall_visible": False, "guarded": False,
                    "arrangement": "unknown", "abundance": "unknown", "conf": 0.0,
                    "reason": f"vlm_error: {e}", "ok": False}


def _image_data_url(path, quality=92, max_side=IMAGE_MAX_SIDE):
    """图像文件 → data URL（VLM 输入）。

    三条分支：
      * 长边 > max_side（原图 4K/2K 常见）→ cv2 解码、INTER_AREA 等比缩到 max_side、转 JPEG。
        这是**默认会走到的分支**，见 IMAGE_MAX_SIDE 的实测载荷说明；
      * 尺寸已达标且为 jpg/png → 原样下发（不重编码，零画质损失）；
      * 其余格式（tif/tiff/bmp/webp）→ 解码转 JPEG，尺寸不变。
    读不出即抛 ValueError：调用方在 try 内调用，故降级为 ok=False/degraded，不中断判药。
    """
    p = Path(path)
    ext = p.suffix.lower()
    import cv2                                   # 仅此处需要，保持模块导入轻量
    import numpy as np
    img = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"读不出图像（cv2 解码失败）: {p.name}")
    h, w = img.shape[:2]
    if ext in _RAW_MIME and max(h, w) <= max_side:
        return (f"data:{_RAW_MIME[ext]};base64,"       # 尺寸已达标：原样下发，零重编码
                + base64.b64encode(p.read_bytes()).decode("ascii"))
    if max(h, w) > max_side:                     # 等比缩小：长边对齐 max_side
        s = max_side / float(max(h, w))
        img = cv2.resize(img, (max(1, int(round(w * s))), max(1, int(round(h * s)))),
                         interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError(f"转 JPEG 失败: {p.name}")
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def build_image_index(image_roots):
    """stem → 图像路径 索引（含子目录，一次构建全程复用）。"""
    idx = {}
    for root in image_roots or []:
        root = Path(root)
        if not root.exists():
            continue
        for f in root.rglob("*"):
            if f.suffix.lower() in _IMAGE_EXTS and f.is_file():
                idx.setdefault(f.stem, f)
    return idx


def augment_and_rescore(kb, cards, decision, vlm, image_index, sample_records,
                        max_calls=6, max_images_per_check=2, scene=None):
    """按介入时机补证据并重评分。返回 (新decision, notes, agent_meta, scene)。

    神经层不直接产生结论——注入证据后整体重跑 decide()。

    场景判读触发（三层模型第 2 层，2026-09-22 用户决议：**强制触发**）：只要样本
    检出结晶类特征（scene_attributes.CRYSTAL_FEATURES）即对每视野发起 scene_check，
    **不限裁决结果**（confirmed/probable/insufficient 一律判读）——宿主语境乘数
    s(scene) 需参与判据贡献评分，若仅在平分时触发则评分通道在多数案例中无输入。
    结果经 merge_vlm_scene 并入 scene（VLM 与几何并存、VLM 优先），随后重跑
    decide(scene=...)。VLM 未开启/无结晶检出/预算耗尽/调用失败 → 静默降级。

    存在性补证据环（§5.3 原有）：verdict ∈ probable/insufficient 或 needs_review 时，
    对 top-3 候选"required 且未检出"的 P1/C 级判据发起 presence_check。
    两环共享 --vlm-max-calls 预算，场景判读先行。
    """
    meta = {"mode": "rule_only", "vlm_calls": 0, "degraded": False, "injected": [],
            "scene_vlm_calls": 0}
    notes = []
    if vlm is None:
        return decision, notes, meta, scene
    meta["mode"] = "rule+vlm"

    # ---- 场景判读（强制触发）：检出结晶类特征即判读，不限裁决结果 ----
    scene_updated = False
    crystal_present = any(fid in scene_attributes.CRYSTAL_FEATURES and card.present
                          for fid, card in cards.items())
    if crystal_present and decision.get("ranking"):
        # 不传候选 morph 提示：位置判断必须只来自画面观察（2026-09-23 决议）
        stems = []
        for r in sample_records or []:
            s = r.get("image_stem")
            if s and s not in stems:
                stems.append(s)
        answers = []
        for stem in stems:
            if meta["vlm_calls"] >= max_calls:
                notes.append(f"[agent] VLM 调用预算耗尽（{max_calls}），剩余视野未发起场景判读")
                break
            ipath = image_index.get(stem)
            if ipath is None:
                continue
            meta["vlm_calls"] += 1
            meta["scene_vlm_calls"] += 1
            res = vlm.scene_check(ipath)
            if not res["ok"]:
                # 单视野失败不能放弃其余视野：改 continue 继续判读（原为 break，一次超时
                # 就让整个样本一条场景判读都没有 → 宿主语境乘数静默失效，结论随网络时快时慢）
                meta["degraded"] = True
                notes.append(f"[agent] VLM 场景判读失败（{ipath.name}）：{res['reason']}"
                             f"（该视野跳过，继续其余视野）")
                continue
            answers.append({"fov": stem, "crystal_location": res["crystal_location"],
                            "crystal_location_raw": res.get("crystal_location_raw"),
                            "cell_wall_visible": res.get("cell_wall_visible"),
                            "arrangement": res["arrangement"], "abundance": res["abundance"],
                            "conf": res["conf"], "reason": res["reason"]})
            notes.append(f"[agent] VLM 场景判读 @ {ipath.name} → "
                         f"location={res['crystal_location']} arrangement={res['arrangement']} "
                         f"abundance={res['abundance']}（{res['reason']}）")
            if res.get("guarded"):
                raw = res.get("crystal_location_raw")
                why = ("原判 intracellular 但未见细胞结构（cell_wall_visible=false）"
                       if raw == "intracellular" else f"原答非二值（{raw}）")
                notes.append(f"[agent] 场景判读改判 @ {ipath.name}：{why} → "
                             f"判为不位于细胞内（游离散在），照常参与宿主语境乘数")
        if answers:
            scene = scene_attributes.merge_vlm_scene(scene or scene_attributes.empty_scene(),
                                                     answers)
            scene_updated = True
    if scene_updated:
        decision = decide(kb, cards, scene=scene)
        notes.append("[agent] 并入 VLM 场景证据后已重跑规则引擎"
                     "（宿主语境乘数参与判据贡献；丰度/排列仅平分排序）")

    # ---- 存在性补证据环（§5.3）：verdict ∈ probable/insufficient 或 needs_review ----
    needs = decision["verdict"] in ("probable", "insufficient_evidence") or decision.get("needs_review")
    if needs and decision.get("ranking"):
        # 取 top-3 候选中"required 且未检出"的 P1/C 级判据
        asks = []
        for cand_res in decision["ranking"][:3]:
            cand = _candidate_of(kb, cand_res["id"])
            if cand is None:
                continue
            for crit in cand["criteria"]:
                if crit["role"] != "required":
                    continue
                card = cards.get(crit["feature"])
                if card and card.present:
                    continue
                if kb.tier(crit["feature"]) == "P0":
                    continue  # P0 缺失信任检测链灵敏度（§5.3）
                asks.append((cand_res["display"], crit))
            if len(asks) >= max_calls:
                break

        injected_any = False
        for display, crit in asks:
            if meta["vlm_calls"] >= max_calls:
                notes.append(f"[agent] VLM 调用预算耗尽（{max_calls}），剩余判据未发起判读")
                break
            fid = crit["feature"]
            zh = kb.raw["feature_vocab"].get(kb.tier(fid), {}).get(fid, fid)
            stems = [r.get("image_stem") for r in sample_records if r.get("image_stem")]
            used = set()
            got = 0
            for stem in stems:
                if got >= max_images_per_check or meta["vlm_calls"] >= max_calls:
                    break
                ipath = image_index.get(stem)
                if ipath is None or stem in used:
                    continue
                used.add(stem)
                meta["vlm_calls"] += 1
                res = vlm.presence_check(zh, crit.get("morph"), ipath)
                if not res["ok"]:
                    # 同上：单图失败跳过该图、换下一视野，不放弃该判据的其余证据
                    meta["degraded"] = True
                    notes.append(f"[agent] VLM 调用失败（{display}/{fid}，{ipath.name}）："
                                 f"{res['reason']}（换下一视野）")
                    continue
                notes.append(f"[agent] VLM 判读 {display} / {zh} @ {ipath.name} → "
                             f"present={res['present']} conf={res['conf']:.2f}（{res['reason']}）")
                if res["present"] and res["conf"] >= 0.5:
                    card = cards.setdefault(fid, EvidenceCard(fid))
                    card.uncal_refs.append((f"vlm:{stem}", 0))     # 存在性证据（不可量化）
                    card.confs.append(res["conf"])
                    card.conf_assumed = True
                    meta["injected"].append({"candidate": display, "feature": fid,
                                             "conf": res["conf"], "image": str(ipath)})
                    injected_any = True
                    got += 1
        if injected_any:
            decision = decide(kb, cards, scene=scene)
            notes.append("[agent] 注入 VLM 证据后已重跑规则引擎（证据级注入，不改裁决逻辑）")
    return decision, notes, meta, scene


def _candidate_of(kb, cid):
    for cand in kb.candidates():
        if cand["id"] == cid:
            return cand
    return None
