"""离线Reflection全生命周期演示；图像是占位，模型与观察不联网。"""
import uuid
from .storage import output_root, write_json, digest
from .reflection import propose, validate, publish
from .agent_tools import LocalToolbox, FixtureBackend
from .evidence_runtime import EvidenceRuntime
from .react_graph import run_react


async def demo():
    directory = output_root()/"reflection"/"demos"/uuid.uuid4().hex
    directory.mkdir(parents=True)
    images = []
    for i in range(2):
        p = directory/f"view{i}.png"
        p.write_bytes(f"simulation-only-view-{i}".encode())
        images.append(str(p))
    case = {"records": [], "images": images,
            "a": {"status": "ok", "ranking": [{"label": "猪苓", "probability": 1}]},
            "replies": {"hypha": {"ok": True, "present": False, "conf": .9}},
            "expected": {"verdict": "insufficient_evidence", "conclusion_id": None}}
    runtime = EvidenceRuntime([], images, sample_id="reflection_demo", a=case["a"], simulation=True)
    tools = LocalToolbox(runtime, ledger_path=directory/"baseline_ledger.json", backend=FixtureBackend(case["replies"]))
    trace = await run_react(runtime, tools, planner_mode="heuristic")
    write_json(directory/"source_trace.json", trace)
    feedback = {"trace_sha256": digest(trace), "source": "program_check", "issue": "repeated_no_gain"}
    write_json(directory/"feedback.json", feedback)
    candidate = propose(trace, feedback)
    write_json(directory/"candidate.json", candidate)
    cases = {"split": "development", "cases": [case]}
    write_json(directory/"cases.json", cases)
    report = await validate(candidate, cases)
    write_json(directory/"validation.json", report)
    version = await publish(candidate, report)
    reused = LocalToolbox(runtime, ledger_path=directory/"after_ledger.json", backend=FixtureBackend(case["replies"]))
    after = await run_react(runtime, reused, planner_mode="heuristic", reflection_memory=version)
    write_json(directory/"after_trace.json", after)
    write_json(directory/"summary.json", {"simulation": True, "version": str(version),
        "before_requests": trace["requests"], "after_requests": after["requests"],
        "validation_passed": report["passed"], "real_accuracy_validated": False})
    return directory
