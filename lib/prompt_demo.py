"""真实LangMem + 真实LangGraph的离线演示；聊天模型与观察均为显式替身。"""
import uuid

from . import prompt_optimization as P
from .evidence_runtime import EvidenceRuntime
from .storage import output_root, digest, write_json
from .react_graph import run_react


async def demo():
    from TCM_Fusion.tests.prompt_support import OfflineOptimizerModel, OfflinePlanner
    directory = output_root()/"prompt_optimization"/"demos"/uuid.uuid4().hex
    directory.mkdir(parents=True)
    images = []
    for i in range(2):
        p = directory/f"fov{i}.png"
        p.write_bytes(f"explicit-simulation-{i}".encode())
        images.append(str(p))
    a = {"status": "ok", "ranking": [{"label": "猪苓", "probability": 1}]}
    runtime = EvidenceRuntime([], images, sample_id="development_example", a=a, simulation=True)
    observations = [{"action": action, "accepted": True,
        "result": {"ok": True, "present": False, "conf": .9}} for action in runtime.legal_actions({})]
    trace = {"events": [{"phase": "action_observation", "accepted": True, "evidence_changed": False}]*2}
    training = {"split": "training", "examples": [{"group_id": "training_example", "trace": trace,
        "trace_sha256": digest(trace), "feedback": "连续观察没有新增证据；优先复核，减少重复调用。"}]}
    dataset = {"split": "development", "simulation": True, "cases": [{
        "group_id": "development_example", "records": [], "images": images, "a": a,
        "config_identity": "default", "observations": observations,
        "expected": {"verdict": "insufficient_evidence", "conclusion_id": None}}]}
    write_json(directory/"training.json", training)
    write_json(directory/"dataset.json", dataset)
    candidate = await P.optimize(training, model=OfflineOptimizerModel())
    write_json(directory/"candidate.json", candidate)
    report = await P.validate(candidate, dataset, planner_factory=OfflinePlanner)
    write_json(directory/"validation.json", report)
    release = await P.publish(candidate, report, planner_factory=OfflinePlanner)
    result = await run_react(runtime, P.RecordedTools(observations), planner_mode="llm",
        planner=OfflinePlanner(candidate["instruction"]), prompt_release=release)
    write_json(directory/"loaded_trace.json", result)
    write_json(directory/"summary.json", {"simulation": True, "generator": candidate["generator"],
        "baseline_requests": report["rows"][0]["baseline"]["requests"],
        "candidate_requests": report["rows"][0]["candidate"]["requests"],
        "release": str(release), "loaded_version": result["prompt_version"], "passed": report["passed"],
        "notice": "实际运行LangMem和LangGraph；模型为替身，不证明真实质量改善"})
    return directory
