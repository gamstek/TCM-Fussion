"""同一起点对照原B固定补查策略；仅用于显式模拟环境，不把回放当在线收益。"""
from __future__ import annotations

import copy

import scene_attributes
from agent_loop import augment_and_rescore
from rule_engine import decide, pool_evidence

from TCM_Fusion.lib.storage import IntegrationError


def compare_fixed_fixture(runtime, backend, agent_result, *, max_calls=6):
    if not runtime.simulation:
        raise IntegrationError("EVAL_REPLAY_ONLY", "该对照只允许模拟；真实对照须分别执行并记录真实费用")
    calls = []
    index = {stem: image["path"] for image in runtime.images.values() for stem in image["stems"]}
    identifiers = {str(image["path"]): key for key,image in runtime.images.items()}

    class FixedClient:
        def scene_check(self, path):
            action = runtime.action("scene_check", {"image_id": identifiers[str(path)]}, "scene_context")
            calls.append(action)
            return backend.call(runtime, action, path)[0]

        def presence_check(self, feature_zh, morph_hint, path):
            reverse = {zh: fid for group in runtime.kb.raw["feature_vocab"].values() for fid,zh in group.items()}
            action = runtime.action("presence_check", {"image_id": identifiers[str(path)],
                "feature_id": reverse[feature_zh], "morph_hint": morph_hint or ""}, "differential_presence")
            calls.append(action)
            return backend.call(runtime, action, path)[0]

    cards = pool_evidence(copy.deepcopy(runtime.records))
    scene = scene_attributes.build_scene(runtime.records)
    before = decide(runtime.kb, cards, scene=scene)
    decision, notes, meta, scene = augment_and_rescore(runtime.kb, cards, before, FixedClient(),
        index, runtime.records, max_calls=max_calls, scene=scene)
    old_keys = [a["action_id"] for a in calls]
    new_keys = [o["action"]["action_id"] for o in agent_result["observations"]]
    return {"simulation": True, "same_measurement_start": True,
            "fixed": {"decision": decision, "observation_requests": len(calls),
                      "planning_requests": 0, "duplicate_requests": len(old_keys)-len(set(old_keys))},
            "react": {"decision": agent_result["decision"],
                      "observation_requests": agent_result["observation_requests"],
                      "planning_requests": agent_result["planning_requests"],
                      "duplicate_requests": len(new_keys)-len(set(new_keys))},
            "accuracy_gain": None, "real_cost_gain": None,
            "note": "规划也计成本；模拟同响应对照不等于真实模型准确率/时延/成本评测"}
