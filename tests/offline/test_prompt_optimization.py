"""真实 LangMem 优化、提示消费、封闭回放/发布隔离与共享账本回归。
该组用例由项目根 selftest 发现。涉及模型回复的用例使用可控输入，涉及进程的用例以各断言规定的执行边界为准。

主要定义：
- PromptTests 优化与真值校验: `P.optimize` 生成新 instruction；truth 模式接受已验证纠正而非要求答案一致。
- 发布与消费: `P.validate`+`P.publish` 后 `run_react` 实际使用发布提示，候选请求数低于基线。
- 数据集隔离: train/development 重叠、test 集参与优化或校验、未知观察被编造都报错或被判失败。
- 防伪造: guard 摘要变更、篡改报告、模拟 release 用于真实运行均被拒。
- 规划客户端: `LLMPlanner` 载荷含发布 instruction 与 GUARD，短编号 A1/A2 映射回注册 action_id。
- 共享账本与跨进程锁: 同账本两实例去重计费，`ledger_lock` 在另一进程持锁时返回 `LEDGER_BUSY`。

关联：
- 直接依赖 「lib/agent/react_graph.py」、「lib/offline/prompt_optimization.py」、「lib/runtime/ledger_lock.py」、「lib/runtime/storage.py」、「lib/tools/agent_tools.py」、「lib/workflow/evidence_runtime.py」、「tests/support/prompt_support.py」。
- 统一执行见 tests/README.md。
"""
import asyncio
import copy
import json
import tempfile
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from TCM_Fusion.lib.offline import prompt_optimization as P
from TCM_Fusion.lib.workflow.evidence_runtime import EvidenceRuntime
from TCM_Fusion.lib.tools.agent_tools import LocalToolbox, FixtureBackend
from TCM_Fusion.lib.agent.react_graph import run_react, LLMPlanner
from TCM_Fusion.lib.runtime.storage import IntegrationError, digest, read_json, write_json
from TCM_Fusion.tests.support.prompt_support import OfflineOptimizerModel, OfflinePlanner, IMPROVED


class PromptTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        images = []
        for i in range(2):
            p = self.root/f"fov{i}.png"; p.write_bytes(f"simulation{i}".encode()); images.append(str(p))
        self.a = {"status": "ok", "ranking": [{"label": "猪苓", "probability": 1}]}
        self.runtime = EvidenceRuntime([], images, sample_id="dev", a=self.a, simulation=True)
        self.observations = [{"action": a, "accepted": True, "result": {"ok": True, "present": False, "conf": .9}}
                             for a in self.runtime.legal_actions({})]
        trace = {"events": [{"phase": "action_observation", "accepted": True, "evidence_changed": False}]*2}
        self.training = {"split": "training", "examples": [{"group_id": "train", "trace": trace,
            "trace_sha256": digest(trace), "feedback": "减少连续无增益观察"}]}
        self.dataset = {"split": "development", "simulation": True, "cases": [{"group_id": "dev",
            "records": [], "images": images, "a": self.a, "config_identity": "default",
            "observations": self.observations, "expected": {"verdict": "insufficient_evidence", "conclusion_id": None}}]}
        self.candidate = await P.optimize(self.training, model=OfflineOptimizerModel())

    async def test_actual_langmem_optimizer_returns_new_instruction(self):
        self.assertEqual(self.candidate["generator"], "langmem.metaprompt")
        self.assertEqual(self.candidate["instruction"], IMPROVED)
        self.assertNotEqual(self.candidate["baseline"], self.candidate["instruction"])

    async def test_truth_validation_allows_verified_correction_instead_of_equal_answers(self):
        from unittest.mock import AsyncMock
        data=copy.deepcopy(self.dataset);data['mode']='truth'
        data['cases'][0]['reference']={'label':'target','label_source':'expert','label_scope':'sample_identity'}
        traces=[{'decision':{'verdict':'confirmed','conclusion':{'id':label}},'needs_review':False,
                 'requests':2,'events':[],'planner_degraded':False,'mandatory_complete':True} for label in ('wrong','target')]
        with patch('TCM_Fusion.lib.agent.react_graph.run_react',new=AsyncMock(side_effect=traces)):
            report=await P.validate(self.candidate,data,planner_factory=OfflinePlanner)
        self.assertTrue(report['passed']);self.assertFalse(report['real_accuracy_validated'])
        self.assertEqual(report['candidate_summary']['correct_determinate'],1)
        data['split']='test'
        with self.assertRaises(IntegrationError):await P.validate(self.candidate,data,planner_factory=OfflinePlanner)

    async def test_publish_and_graph_actually_consumes_prompt(self):
        with patch("urllib.request.urlopen") as network:
            report = await P.validate(self.candidate, self.dataset, planner_factory=OfflinePlanner)
            self.assertTrue(report["passed"])
            with patch.object(P, "output_root", return_value=self.root):
                release = await P.publish(self.candidate, report, planner_factory=OfflinePlanner)
            planner = OfflinePlanner(IMPROVED)
            result = await run_react(self.runtime, P.RecordedTools(self.observations),
                planner_mode="llm", planner=planner, prompt_release=release)
        network.assert_not_called()
        self.assertTrue(result["prompt_version"])
        self.assertEqual(planner.contexts[0]["planning_instruction"], IMPROVED)
        self.assertLess(report["rows"][0]["candidate"]["requests"], report["rows"][0]["baseline"]["requests"])

    async def test_unknown_observation_cannot_be_invented(self):
        data = copy.deepcopy(self.dataset); data["cases"][0]["observations"] = []
        report = await P.validate(self.candidate, data, planner_factory=OfflinePlanner)
        self.assertFalse(report["passed"])
        self.assertTrue(report["rows"][0]["baseline"]["unknown_actions"])

    async def test_train_development_groups_cannot_overlap(self):
        data = copy.deepcopy(self.dataset); data["cases"][0]["group_id"] = "train"
        with self.assertRaises(IntegrationError):
            await P.validate(self.candidate, data, planner_factory=OfflinePlanner)

    async def test_test_set_cannot_optimize_or_validate(self):
        with self.assertRaises(IntegrationError):
            await P.optimize({**self.training, "split": "test"}, model=OfflineOptimizerModel())
        with self.assertRaises(IntegrationError):
            await P.validate(self.candidate, {**self.dataset, "split": "test"}, planner_factory=OfflinePlanner)

    async def test_wrong_reference_blocks_publish(self):
        data = copy.deepcopy(self.dataset); data["cases"][0]["expected"]["verdict"] = "confirmed"
        report = await P.validate(self.candidate, data, planner_factory=OfflinePlanner)
        with self.assertRaises(IntegrationError):
            await P.publish(self.candidate, report, planner_factory=OfflinePlanner)

    async def test_fake_pass_revalidated(self):
        report = await P.validate(self.candidate, self.dataset, planner_factory=OfflinePlanner)
        report["dataset"]["cases"][0]["observations"] = []
        with self.assertRaises(IntegrationError):
            await P.publish(self.candidate, report, planner_factory=OfflinePlanner)

    async def test_no_network_without_explicit_opt_in(self):
        with self.assertRaises(IntegrationError):
            await P.optimize(self.training)
        with self.assertRaises(IntegrationError):
            await P.validate(self.candidate, self.dataset)

    async def test_changed_guard_rejected(self):
        with self.assertRaises(IntegrationError):
            await P.validate({**self.candidate, "guard_sha256": "bad"}, self.dataset, planner_factory=OfflinePlanner)

    async def test_simulated_version_not_used_live_and_tamper_rejected(self):
        report = await P.validate(self.candidate, self.dataset, planner_factory=OfflinePlanner)
        with patch.object(P, "output_root", return_value=self.root):
            release = await P.publish(self.candidate, report, planner_factory=OfflinePlanner)
        self.runtime.simulation = False
        with self.assertRaises(IntegrationError): P.load_release(release, self.runtime)
        self.runtime.simulation = True
        value = read_json(release); value["candidate"]["instruction"] = "tampered"; write_json(release, value)
        with self.assertRaises(IntegrationError): P.load_release(release, self.runtime)

    async def test_same_ledger_instances_deduplicate_and_account(self):
        backend = FixtureBackend({"hypha": {"ok": True, "present": False, "conf": .9}})
        a = LocalToolbox(self.runtime, ledger_path=self.root/"shared.json", backend=backend, max_requests=1)
        b = LocalToolbox(self.runtime, ledger_path=self.root/"shared.json", backend=backend, max_requests=1)
        action = self.runtime.legal_actions({})[0]
        results = await asyncio.gather(a.observe(action), b.observe(action))
        self.assertEqual(len(backend.calls), 1)
        self.assertEqual(sum(r["new_requests"] for r in results), 1)
        next_action = self.runtime.legal_actions({})[1]
        with self.assertRaises(IntegrationError): await b.observe(next_action)

    async def test_published_instruction_enters_real_client_payload(self):
        from adjudicator import AdjudicatorClient
        client = object.__new__(LLMPlanner)
        client.client = AdjudicatorClient.__new__(AdjudicatorClient)
        client.client.model = "test";client.client.timeout_s=1; client.config = {}
        with patch('TCM_Fusion.lib.runtime.hard_calls.post', return_value={"choices": [{"message": {"content": '{"action_id":"review"}'}}]}) as post:
            client.choose({"planning_instruction": IMPROVED})
        system = post.call_args.args[0]["messages"][0]["content"]
        self.assertIn(IMPROVED, system); self.assertIn(P.GUARD, system)

    async def test_adaptive_client_maps_short_option_back_to_registered_id(self):
        from adjudicator import AdjudicatorClient
        client = object.__new__(LLMPlanner)
        client.client = AdjudicatorClient.__new__(AdjudicatorClient)
        client.client.model = "test"
        client.client.timeout_s = 1
        client.config = {}
        context = {"stop_advisory": {"recommend": "continue"},
                   "actions": [{"action_id": "registered-hash", "tool": "presence_check"},
                               {"action_id": "review", "tool": "request_review"}],
                   "action_objectives": {"registered-hash": "verify"}}
        with patch("TCM_Fusion.lib.runtime.hard_calls.post",
                   return_value={"choices": [{"message": {"content": '{"action_id":"A2"}'}}]}) as post:
            reply, _ = client.choose(context)
        self.assertEqual(reply["action_id"], "review")
        payload = post.call_args.args[0]
        self.assertEqual(payload["response_format"], {"type": "json_object"})
        sent = json.loads(payload["messages"][1]["content"])
        self.assertEqual([action["action_id"] for action in sent["actions"]], ["A1", "A2"])
        self.assertEqual(sent["action_objectives"], {"A1": "verify"})
        self.assertEqual(context["actions"][0]["action_id"], "registered-hash")

    async def test_os_lock_excludes_another_process(self):
        from TCM_Fusion.lib.runtime.ledger_lock import acquire, release
        ledger = self.root/"cross_process.json"
        handle = acquire(ledger)
        code = ("import sys\nfrom TCM_Fusion.lib.runtime.ledger_lock import acquire,release\n"
                "from TCM_Fusion.lib.runtime.storage import IntegrationError\n"
                "try:\n h=acquire(sys.argv[1],timeout=.1)\n release(h)\n print('UNEXPECTED')\n"
                "except IntegrationError as e:\n print(e.code)\n")
        try:
            result = await asyncio.to_thread(subprocess.run, [sys.executable, "-X", "utf8", "-c", code, str(ledger)],
                                             capture_output=True, text=True, encoding="utf-8", timeout=15)
        finally:
            release(handle)
        self.assertEqual(result.returncode, 0)
        self.assertIn("LEDGER_BUSY", result.stdout)
