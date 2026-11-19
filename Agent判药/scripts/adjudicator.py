# -*- coding: utf-8 -*-
"""adjudicator — LLM 有界仲裁者（方案 D，影子模式，v1.0）。

设计铁律（ADR-001 方案 D）：
  - 引擎（rule_engine.decide）照常出 top-k + 评分分解，本模块是引擎的**影子**辅助层；
  - 只在"带内模糊"案例介入：top1/top2 分差 < MARGIN_BAND，且 verdict ∈ probable/insufficient；
  - 输入限为判据卡（feature_id/state/贡献分），LLM 只能就这份证据表态；
  - 输出限为结构化 JSON，**必须引用候选判据且程序端校验引用一致性**——引用不存在/不匹配 → 建议作废；
  - 本模块**不修改 verdict**，只产出 shadow 建议落盘（verdict.json.shadow_bounded_adjudicator）。

纯标准库，失败静默降级（不影响 rule_only 生产路径）。
"""
import base64
import json
import re
import sys
import urllib.request
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# 带内模糊带宽：top1 与 top2 置信度分差小于该值才值得仲裁（对应引擎 MARGIN_DELTA=0.10 的模糊带）
MARGIN_BAND = 0.10
BAND_REVIEW = ("probable", "insufficient_evidence")   # 触发影子仲裁的裁决带
MAX_CANDIDATES = 3                                    # 最多交给 LLM 判据卡的候选数
MAX_REASON_LEN = 400


class AdjudicatorClient:
    """LLM 仲裁客户端（OpenAI 兼容 /chat/completions，模型名参数化）。

    复用上游 vlm_config 的连接方式；模型可经 init 另行指定（影子实验默认 glm-4.6v）。
    """

    def __init__(self, config_path, model=None, timeout_s=60):
        cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
        vlm = cfg.get("vlm", cfg)
        self.base_url = vlm["base_url"].rstrip("/")
        self.model = model or vlm.get("model", "glm-4.6v")
        self.api_key = vlm.get("api_key") or ""
        if not self.api_key and vlm.get("api_key_env"):
            import os
            self.api_key = os.environ.get(vlm["api_key_env"], "")
        self.temperature = float(vlm.get("temperature", 0.1))
        self.max_tokens = int(vlm.get("max_tokens", 512))
        # 默认采用配置里的 timeout_s（若 time 参数未显式指定）
        self.timeout_s = timeout_s if timeout_s != 60 else float(vlm.get("timeout_s", 60))

    def _post(self, payload):
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.api_key}"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
            self._http_headers_received(resp.status)
            return json.loads(resp.read().decode("utf-8"))

    def _http_headers_received(self, status):
        self.last_http_status = status

    def arbitrate(self, prompt):
        payload = {
            "model": self.model, "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        }
        try:
            data = self._post(payload)
            text = data["choices"][0]["message"]["content"]
            m = re.search(r"\{.*\}", text, re.S)
            obj = json.loads(m.group(0)) if m else {}
            return {"ok": True, "raw": obj}
        except Exception as e:  # noqa: BLE001 — 仲裁失败必须降级而不是中断判药
            return {"ok": False, "error": str(e), "raw": {}}


_SYSTEM_PROMPT = (
    "你是中药粉末显微鉴别领域的精确仲裁助手。你会收到两个候选药材的判据核验卡"
    "（feature_id / role / state / 贡献分 / 形态提示）。"
    "你的任务：仅依据这两份判据卡，在两者之间给出倾向，不得编造判据。\n"
    "输出严格 JSON："
    '{"choice": "top1"|"top2"|"tie", "cited": [{"candidate": "<candidate_id>", '
    '"feature": "<必须来自判据卡的feature_id>", "state": "<该feature的state>"}], '
    '"reason": "一句话依据（中文，≤120字）"}'
)


def build_prompt(cand_a, cand_b):
    """把两个候选的判据卡序列化成给 LLM 的案情摘要。"""
    def _block(cand):
        lines = [f"候选：{cand.get('id')}（{cand.get('display')}） 置信度 {cand.get('confidence'):.3f}"]
        for c in cand["checks"]:
            lines.append(
                f"  - {c['feature_id']} | role={c['role']} | state={c['state']} | "
                f"贡献分={c['score_contribution']} | 罚分={c['penalty']} | "
                f"形态={c.get('morph_text') or ''}")
        return "\n".join(lines)
    return f"请仲裁以下两个候选在证据上的相对倾向：\n\n{_block(cand_a)}\n\n{_block(cand_b)}"


def _resolve_candidate(sel, cand_a, cand_b):
    """把 LLM 填的候选选择符解析到 (top1/top2, cand)。兼容 top1/top2 及候选 id/display。"""
    if sel == "top1":
        return "top1", cand_a
    if sel == "top2":
        return "top2", cand_b
    for key, cand in (("top1", cand_a), ("top2", cand_b)):
        if sel in (cand.get("id"), cand.get("display")):
            return key, cand
    return None, None


def _cited_valid(choice, cited, cand_a, cand_b):
    """程序端校验引用一致性：choice 必须合法；cited 里的 (candidate, feature, state)
    必须存在于对应候选判据卡且 state 匹配。返回 (ok, invalid_reason)。"""
    if choice not in ("top1", "top2", "tie"):
        return False, f"choice 非法: {choice!r}"
    if not isinstance(cited, list):
        return False, "cited 非数组"
    for item in cited:
        if not isinstance(item, dict):
            return False, "cited 项非对象"
        sel = item.get("candidate")
        fid = item.get("feature")
        st = item.get("state")
        key, cand = _resolve_candidate(sel, cand_a, cand_b)
        if cand is None:
            return False, f"引用了不存在的候选: {sel!r}"
        hit = [c for c in cand["checks"]
               if c["feature_id"] == fid and c["state"] == st]
        if not hit:
            # 允许 feature 存在但 state 空（LLM 可不填 state）
            if any(c["feature_id"] == fid for c in cand["checks"]):
                if st is None:
                    continue
            return False, (f"候选 {key} 中判据 {fid!r} 状态应为 "
                           f"{[c['state'] for c in cand['checks'] if c['feature_id']==fid]} 而非 {st!r}")
    return True, ""

def _normalize_cand(cand):
    """把引擎候选 dict 规整为仲裁所需的判据卡（checks 去重字段、补 confidence/id）。"""
    return {
        "id": cand.get("id"),
        "display": cand.get("display"),
        "confidence": cand.get("confidence"),
        "checks": cand.get("checks") or [],
    }


def bounded_adjudicate(top_cands, client, max_calls=1):
    """带界仲裁入口（影子）：对 top2 产出仲裁建议，不经任何校验不修改 verdict。

    参数
      top_cands : decide() 输出的 ranking（按 confidence 降序）前 N 名
      client    : AdjudicatorClient
    Yields
      每次有效仲裁 yield 一个建议 dict；LLM 失败/引用不一致 → yield 失败说明。v1.0 单次仲裁。
    """
    if max_calls <= 0:
        return
    if len(top_cands) < 2:
        return
    a, b = _normalize_cand(top_cands[0]), _normalize_cand(top_cands[1])
    resp = client.arbitrate(build_prompt(a, b))
    if not resp["ok"]:
        yield {"status": "error", "detail": f"vlm_error: {resp.get('error')}",
               "choice": None, "cited": [], "reason": ""}
        return
    raw = resp["raw"]
    choice = raw.get("choice")
    cited = raw.get("cited") or []
    if not isinstance(cited, list):
        cited = []
    ok, why = _cited_valid(choice, cited, a, b)
    if not ok:
        yield {"status": "invalid_reference", "detail": why,
               "choice": choice, "cited": cited, "reason": raw.get("reason", "")}
        return
    yield {
        "status": "ok",
        "choice": choice,
        "choice_zh": {"top1": "倾向 top1", "top2": "倾向 top2",
                      "tie": "持平/无法区分"}.get(choice),
        "cited": cited,
        "reason": (raw.get("reason") or "")[:MAX_REASON_LEN],
        "note": "影子仲裁建议，仅记录参考，不改变引擎 verdict",
    }


def should_arbitrate(decision):
    """判定是否落在带内模糊区：verdict 属介入带、有 top1/top2、且分差 < MARGIN_BAND。"""
    v = decision.get("verdict")
    if v not in BAND_REVIEW:
        return False
    ranking = decision.get("ranking") or []
    if len(ranking) < 2:
        return False
    d = ranking[0]["confidence"] - ranking[1]["confidence"]
    if d is None:
        return False
    # 严格小于 MARGIN_BAND；用容差吸收浮点误差（0.60-0.50 不应误触发）
    return d < MARGIN_BAND - 1e-9


def run_shadow_adjudication(decision, client, max_calls=1):
    """对单个样本的 decide() 输出跑影子仲裁，返回要写入 verdict.json 的 shadow 结构。

    不改 decision；产出 shadow_bounded_adjudicator 字段内容。
    """
    if client is None or not should_arbitrate(decision):
        return None
    cands = decision.get("ranking") or []
    cands = cands[:MAX_CANDIDATES]
    results = list(bounded_adjudicate(cands, client, max_calls=max_calls))
    shadow = {
        "applied": True,
        "trigger": {
            "verdict": decision["verdict"],
            "top1": cands[0]["id"] if cands else None,
            "top2": cands[1]["id"] if len(cands) > 1 else None,
            "margin": round(cands[0]["confidence"] - cands[1]["confidence"], 4)
                      if len(cands) > 1 else None,
            "band_threshold": MARGIN_BAND,
        },
        "adjudications": results,
        "disclaimer": "影子仲裁（shadow）建议仅供参考，不改变引擎最终 verdict。",
    }
    return shadow
