# -*- coding: utf-8 -*-
"""
vlm_api.py — VLM API 传输层（两阶段共享）

检测阶段（detection/cli/vlm_prelabel）与比例尺双引擎（common/scalebar_duo）共用的
OpenAI 兼容 /chat/completions 传输与解析原语：图像编码、请求发送、返回 JSON 容错解析。
本模块只依赖标准库与 cv2，**不得 import 任何阶段模块**（避免共享层反向依赖阶段目录）。

  encode_jpeg_b64(img, max_side=None, quality=90)  等比缩放（最长边 max_side，只缩不放）+ JPEG + base64
  call_vlm(cfg_vlm, api_key, b64, prompt)   发请求（含重试）；prompt 必填
  parse_vlm_json(text)                剥 ```json 围栏 / 截取首个大括号块 / 截断修复
  _salvage_truncated(text)            parse_vlm_json 内部使用（截断修复）
"""
import base64
import json
import re
import time
import urllib.error
import urllib.request

import cv2


def encode_jpeg_b64(img, max_side=None, quality=90):
    """BGR 图 → base64 JPEG。返回 `(b64, scale)`。

    `max_side` 给定时按最长边等比**只缩不放**（scale ≤ 1）并返回实际缩放比；None 则原尺寸发送。
    `quality` 默认 90（本函数历史口径）；调用方若有自己的口径须**显式传入**——2026-09-29 P4-1b
    把检测侧的 q92 与分割侧的 q95 两份局部实现并入此处时，刻意不给它们改默认值，
    以免静默改变已上线载荷的字节。
    """
    scale = 1.0
    if max_side:
        h, w = img.shape[:2]
        scale = min(1.0, max_side / max(h, w))
        if scale < 1.0:
            img = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))),
                             interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("jpeg encode 失败")
    return base64.b64encode(buf.tobytes()).decode(), scale


def call_vlm(cfg_vlm, api_key, b64, prompt):
    payload = {
        "model": cfg_vlm["model"],
        "temperature": cfg_vlm.get("temperature", 0.1),
        "max_tokens": cfg_vlm.get("max_tokens", 4096),
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                {"type": "text", "text": prompt},
            ],
        }],
    }
    # 思考强度：**可选透传**——配置里没这个键就一个字都不发（保持各模型自己的默认）。
    # 背景（2026-09-30 换 glm-5.3-flash 时实测）：GLM-5.3/5.3-FLASH 默认 `thinking.type=enabled`
    # 且**不再支持关闭思考**（传 disabled 直接报错），`reasoning_effort` 默认 `max`（深度推理）。
    # max 强度下单图耗时数分钟，且 reasoning token 与答案**共享 max_tokens 预算**，会把 JSON 挤断。
    # 本链是"看图 → 输出固定 JSON"的感知+格式任务，深推理边际收益低，故配置里设 low。
    if cfg_vlm.get("reasoning_effort"):
        payload["reasoning_effort"] = cfg_vlm["reasoning_effort"]
    url = cfg_vlm["base_url"].rstrip("/") + "/chat/completions"
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {api_key}"})
    last_err = None
    for attempt in range(1, cfg_vlm.get("max_retries", 3) + 1):
        try:
            with urllib.request.urlopen(req, timeout=cfg_vlm.get("timeout_s", 180)) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            text = data["choices"][0]["message"]["content"]
            return text, None
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", "replace")[:300]
            last_err = f"HTTP {e.code}: {body}"
            if e.code in (401, 403):
                return None, last_err  # key 错误，重试无意义
        except Exception as e:
            last_err = str(e)
        time.sleep(3 * attempt)
    return None, last_err


def _salvage_truncated(text):
    """修复因 max_tokens 截断的 JSON：截到最后一个完整 object 并补齐收尾。"""
    m = re.search(r'"objects"\s*:\s*\[', text)
    if not m:
        return None
    body = text[m.end():]
    depth, end, in_str, esc = 0, -1, False, False
    for i, ch in enumerate(body):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i
    if end < 0:
        return None
    try:
        return {"objects": json.loads("[" + body[:end + 1] + "]")}
    except Exception:
        return None


def parse_vlm_json(text):
    """容错解析 VLM 返回的 JSON（剥 ```json 围栏 / 截取首个大括号块 / 截断修复）。"""
    text = text.strip()
    m = re.search(r"```(?:json)?(.*?)```", text, re.S)
    if m:
        text = m.group(1).strip()
    start = text.find("{")
    if start < 0:
        raise ValueError(f"返回中无 JSON: {text[:120]}")
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start:i + 1])
    raw = text[start:]
    try:
        return json.loads(raw)
    except Exception:
        salv = _salvage_truncated(raw)
        if salv is not None:
            return salv
        raise