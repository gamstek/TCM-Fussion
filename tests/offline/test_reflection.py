"""反思发布的有效性、数据隔离与回放反例回归；不调用真实模型。

主要定义：
- ReflectionTests.test_publish_load_and_no_network: `propose`→`validate`→`publish`→`load_memory` 全链路且不联网。
- test_second_view_evidence_rejects_early_stop: 后续视图出现阳性时否定提前停止策略。
- test_feedback_cannot_be_unbound_or_unsupported、test_no_gain_claim_checked_against_trace: feedback 摘要必须绑定 trace、来源受支持，且无增益声明须与 trace 一致。
- test_test_set_and_missing_truth_rejected、test_arbitrary_policy_rejected: test 集、缺真值、任意 policy 均被拒。
- test_forged_pass_replayed、test_simulation_memory_not_used_in_production: 伪造报告重放失败，模拟记忆不能在真实运行使用或被篡改。
- test_llm_reflection_requires_opt_in、test_llm_path_uses_text_feedback_only: LLM 反思需显式 opt-in，且载荷不含图像。

关联：
- TCM_Fusion.lib.reflection: `propose`/`validate`/`publish`/`load_memory`。
- TCM_Fusion.lib.storage: `digest`/`read_json`/`write_json`/`IntegrationError`。
- TCM_Fusion.lib.agent_tools: `FixtureBackend` 的回放对照。
- adjudicator.AdjudicatorClient: LLM 反思客户端替身。
"""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from TCM_Fusion.lib.reflection import propose, validate, publish, load_memory
from TCM_Fusion.lib.storage import digest, IntegrationError, read_json, write_json


class ReflectionTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.images = []
        for i in range(2):
            p = self.root/f"v{i}.png"
            p.write_bytes(f"fixture{i}".encode())
            self.images.append(str(p))
        self.trace = {"events": [{"phase": "action_observation", "evidence_changed": False}]*2}
        self.feedback = {"trace_sha256": digest(self.trace), "source": "program_check", "issue": "repeated_no_gain"}
        self.candidate = propose(self.trace, self.feedback)
        self.cases = {"split": "development", "cases": [{"records": [], "images": self.images,
            "a": {"status": "ok", "ranking": [{"label": "猪苓", "probability": 1}]},
            "replies": {"hypha": {"ok": True, "present": False, "conf": .9}},
            "expected": {"verdict": "insufficient_evidence", "conclusion_id": None}}]}

    async def test_publish_load_and_no_network(self):
        with patch("urllib.request.urlopen") as network:
            report = await validate(self.candidate, self.cases)
            self.assertTrue(report["passed"])
            self.assertEqual(report["rows"][0]["baseline"]["requests"], 2)
            self.assertEqual(report["rows"][0]["candidate"]["requests"], 1)
            with patch("TCM_Fusion.lib.reflection.output_root", return_value=self.root):
                path = await publish(self.candidate, report)
            self.assertEqual(load_memory(path, simulation=True)["policy"], {"no_gain_limit": 1})
        network.assert_not_called()

    async def test_second_view_evidence_rejects_early_stop(self):
        from TCM_Fusion.lib.agent_tools import FixtureBackend
        class LaterPositive(FixtureBackend):
            def call(self, runtime, action, path):
                self.calls.append(1)
                return {"ok": True, "present": len(self.calls) == 2, "conf": .9}, None
        with patch("TCM_Fusion.lib.agent_tools.FixtureBackend", LaterPositive):
            report = await validate(self.candidate, self.cases)
        self.assertFalse(report["passed"])

    async def test_feedback_cannot_be_unbound_or_unsupported(self):
        for update in ({"trace_sha256": "bad"}, {"source": "self_confidence"}):
            with self.assertRaises(IntegrationError):
                propose(self.trace, {**self.feedback, **update})

    async def test_no_gain_claim_checked_against_trace(self):
        trace = {"events": [{"phase": "action_observation", "evidence_changed": True}]*2}
        with self.assertRaises(IntegrationError):
            propose(trace, {**self.feedback, "trace_sha256": digest(trace)})

    async def test_test_set_and_missing_truth_rejected(self):
        with self.assertRaises(IntegrationError):
            await validate(self.candidate, {**self.cases, "split": "test"})
        cases = copy.deepcopy(self.cases)
        del cases["cases"][0]["expected"]
        with self.assertRaises(IntegrationError):
            await validate(self.candidate, cases)

    async def test_arbitrary_policy_rejected(self):
        with self.assertRaises(IntegrationError):
            await validate({**self.candidate, "policy": {"disable_p0_guard": True}}, self.cases)

    async def test_forged_pass_replayed(self):
        report = await validate(self.candidate, self.cases)
        report["replay_cases"]["cases"][0]["expected"]["verdict"] = "confirmed"
        with self.assertRaises(IntegrationError):
            await publish(self.candidate, report)

    async def test_simulation_memory_not_used_in_production(self):
        report = await validate(self.candidate, self.cases)
        with patch("TCM_Fusion.lib.reflection.output_root", return_value=self.root):
            path = await publish(self.candidate, report)
        with self.assertRaises(IntegrationError):
            load_memory(path, simulation=False)
        value = read_json(path)
        value["policy"] = {"no_gain_limit": 2}
        write_json(path, value)
        with self.assertRaises(IntegrationError):
            load_memory(path, simulation=True)

    async def test_llm_reflection_requires_opt_in(self):
        with self.assertRaises(IntegrationError):
            propose(self.trace, self.feedback, llm=True)

    async def test_llm_path_uses_text_feedback_only(self):
        with patch("adjudicator.AdjudicatorClient") as factory:
            client = factory.return_value
            client.api_key = "fixture"
            client.model = "fixture"
            client._post.return_value = {"choices": [{"message": {"content": "候选经验：更早停止无增益循环"}}]}
            result = propose(self.trace, self.feedback, llm=True, allow_network=True)
        client._post.assert_called_once()
        payload = client._post.call_args.args[0]
        self.assertNotIn("image_url", str(payload))
        self.assertEqual(result["mode"], "llm_reflection")
        self.assertEqual(result["policy"], {"no_gain_limit": 1})
