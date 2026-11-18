"""构造候选投影测试所需的规则裁决、Agent 结果和 A 预测。调用方可覆写个别字段模拟交付边界，并用真实投影校验输出。

主要定义：
- _a_prediction、_verdict、_result 分别构建三种最小输入。

关联：
- 调用方 「tests/decision/test_output_projection.py」、「tests/integration/test_fair_investigation.py」。
- 统一执行见 tests/README.md。
"""

def _a_prediction():
    views = []
    for i in range(1, 6):
        folds = [[{"label": "知母", "probability": 0.998}] for _ in range(5)]
        views.append({"image": f"D:/case/view_0{i}.jpg", "input_sha256": f"sha{i}",
                      "fold_rankings": folds})
    return {"status": "ok", "ranking": [], "view_predictions": views}


def _verdict():
    def check(feature, role="required", present=True, state="IN_RANGE", contribution=0.4):
        return {"feature_id": feature, "role": role, "present": present, "state": state,
                "score_contribution": contribution, "tier": "P0"}

    return {
        "verdict": "insufficient_evidence", "needs_review": True, "conclusion": None,
        "ranking": [
            {"id": "射干", "display": "射干", "kind": "herb", "score": 0.4, "confidence": 0.4,
             "support_count": 1, "checks": [check("needle_crystal")]},
            {"id": "G2", "display": "半夏 / 法半夏 / 姜半夏", "kind": "group", "score": 0.4,
             "confidence": 0.4, "support_count": 1, "checks": [check("needle_crystal")]},
            {"id": "知母", "display": "知母", "kind": "herb", "score": 0.4, "confidence": 0.4,
             "support_count": 1, "checks": [check("needle_crystal")]},
            {"id": "菟丝子", "display": "菟丝子", "kind": "herb", "score": 0.3, "confidence": 0.6,
             "support_count": 1, "checks": [check("palisade_cell", "required", True, "NO_REF", 0.255)]},
        ],
    }


def _result():
    return {
        "observations": [
            {"accepted": True,
             "action": {"tool": "presence_check",
                        "args": {"image_id": "img_sha1", "feature_id": "palisade_cell"}},
             "result": {"present": True, "conf": 0.9}},
            {"accepted": True,
             "action": {"tool": "presence_check",
                        "args": {"image_id": "img_sha1", "feature_id": "fiber"}},
             "result": {"present": False, "conf": 0.88}},
        ],
        "operations": {},
    }
