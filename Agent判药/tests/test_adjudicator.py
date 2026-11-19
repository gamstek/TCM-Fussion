# -*- coding: utf-8 -*-
"""test_adjudicator — 影子仲裁（方案D）纯逻辑单测（数据驱动）。

用例数据存放于 tests/cases/adjudicator.json；本程序读取该文件，
用 fake client（不发起网络调用）按 checks[].kind 执行对应断言。
直接运行： python tests/test_adjudicator.py
"""
import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "scripts"))

from adjudicator import (MARGIN_BAND, _cited_valid, build_prompt,  # noqa: E402
                         run_shadow_adjudication, should_arbitrate)

CASES = HERE / "cases" / "adjudicator.json"
PASS = 0


def construct_decision(dec):
    """由紧凑参数展开为 canonical decision；已有完整 ranking 则直接采用。"""
    if "ranking" in dec:
        return {"verdict": dec["verdict"], "ranking": dec["ranking"]}
    c1, c2 = dec["c1"], dec["c2"]
    return {
        "verdict": dec["verdict"],
        "ranking": [
            {"id": "cand1", "display": "候选一", "confidence": c1,
             "checks": [{"feature_id": "cluster_crystal", "role": "required",
                         "state": "IN_RANGE", "score_contribution": 1.0,
                         "penalty": 0.0, "morph_text": "簇晶"}]},
            {"id": "cand2", "display": "候选二", "confidence": c2,
             "checks": [{"feature_id": "cluster_crystal", "role": "required",
                         "state": "IN_RANGE", "score_contribution": 0.8,
                         "penalty": 0.0, "morph_text": "簇晶"}]},
        ],
    }


class FakeClient:
    def __init__(self, result):
        self.result = result
        self.calls = 0

    def arbitrate(self, prompt):
        self.calls += 1
        self.last_prompt = prompt
        return self.result


def kind_check(ch):
    k = ch["kind"]
    if k == "should_arbitrate":
        return should_arbitrate(construct_decision(ch["decision"])) == ch["expect"]
    if k == "cited_valid":
        dec = construct_decision({"verdict": "probable", "c1": 0.60, "c2": 0.55})
        a, b = dec["ranking"][0], dec["ranking"][1]
        valid, _ = _cited_valid(ch["choice"], ch["cited"], a, b)
        return bool(valid) == ch["expect_valid"]
    if k == "prompt_build":
        dec = construct_decision(ch["decision"])
        a, b = dec["ranking"][0], dec["ranking"][1]
        p = build_prompt(a, b)
        return all(s in p for s in ch["expect_contains"])
    if k == "run_shadow":
        dec = construct_decision(ch["decision"])
        client = None if ch["client_result"] is None else FakeClient(ch["client_result"])
        shadow = run_shadow_adjudication(dec, client)
        exp = ch["expect"]
        if exp.get("is_none"):
            return shadow is None and exp.get("calls", 0) == (client.calls if client else 0)
        if not exp.get("applied"):
            raise ValueError(f"run_shadow 断言缺少 applied 字段: {ch['name']}")
        return (shadow is not None and shadow.get("applied") is True
                and shadow["adjudications"][0]["status"] == exp["status"]
                and (exp.get("choice") is None
                     or shadow["adjudications"][0]["choice"] == exp["choice"]))
    if k == "run_shadow_margin":
        dec = construct_decision(ch["decision"])
        client = FakeClient(ch["client_result"])
        shadow = run_shadow_adjudication(dec, client)
        return shadow is not None and abs(shadow["trigger"]["margin"] - ch["expect_margin"]) < 1e-6
    raise ValueError(f"未知断言语义 kind: {k}")


def check(name, cond):
    global PASS
    if not cond:
        raise AssertionError(f"用例失败: {name}")
    PASS += 1


def main():
    data = json.loads(CASES.read_text(encoding="utf-8"))
    for ch in data["checks"]:
        check(ch["name"], kind_check(ch))
    print(f"[ok] test_adjudicator 断言全部通过（{PASS} 项，数据源 {CASES.name}）")


if __name__ == "__main__":
    main()