"""真实LangGraph与stdio MCP边界回归；模型响应使用可控替身，绝不联网推理。"""
from __future__ import annotations

import copy
import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import paths

from TCM_Fusion.lib.agent_tools import FixtureBackend, LocalToolbox
from TCM_Fusion.lib.evidence_runtime import EvidenceRuntime
from TCM_Fusion.lib.mcp_bridge import connect_stdio
from TCM_Fusion.lib.react_graph import ScriptedPlanner, run_react
from TCM_Fusion.lib.storage import IntegrationError, digest, read_json, write_json


class ReactTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="microscope-react-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.image = self.root/"fov.png"
        self.image.write_bytes(b"simulation-only-not-decoded")
        self.fixture = read_json(Path(paths.PROJECT_ROOT)/"TCM_Fusion/tests/fixtures/react_case.json")
        self.runtime = EvidenceRuntime(self.fixture["records"], [self.image], sample_id="react_demo",
                                       a=self.fixture["a"], simulation=True)
        self.backend = FixtureBackend(self.fixture["replies"])
        self.tools = LocalToolbox(self.runtime, ledger_path=self.root/"ledger.json", backend=self.backend)

    async def run_graph(self, **kwargs):
        return await run_react(self.runtime, self.tools, planner_mode="scripted",
                               planner=kwargs.pop("planner", ScriptedPlanner()), **kwargs)

    async def scene_first(self):
        action = self.runtime.legal_actions({})[0]
        result = await self.tools.observe(action)
        observation = {"action": action, "accepted": result["accepted"], "result": result["result"]}
        return {"observations": [observation], "attempted": [action["action_id"]]}

    async def test_graph_actually_changes_evidence_and_rule_verdict(self):
        before = copy.deepcopy(self.runtime.original)
        result = await self.run_graph()
        self.assertEqual(result["framework"], "langgraph")
        self.assertFalse(result["shadow_only"])
        self.assertTrue(result["simulation"])
        self.assertEqual(result["decision"]["ranking"][0]["id"], "猪苓")
        self.assertEqual(before["verdict"], "insufficient_evidence")
        self.assertEqual(result["decision"]["verdict"], "probable")
        self.assertEqual(result["decision"]["support_feature_count"], 2)
        self.assertEqual(self.runtime.original, before)
        self.assertTrue(any(o["accepted"] and o["action"]["tool"] == "presence_check" for o in result["observations"]))
        cards = self.runtime.score(result["observations"])["cards"]
        self.assertEqual(cards["hypha"].n_instances, 0)
        self.assertEqual(cards["hypha"].dims, {})

    async def test_mandatory_precedes_planner_and_budget_is_shared(self):
        planner = ScriptedPlanner()
        result = await self.run_graph(max_requests=1, planner=planner)
        self.assertEqual(result["requests"], 1)
        self.assertEqual(result["planning_requests"], 0)
        self.assertEqual(result["observations"][0]["action"]["tool"], "scene_check")
        self.assertEqual(planner.contexts, [])
        self.assertTrue(result["needs_review"])

    async def test_zero_budget_no_backend(self):
        result = await self.run_graph(max_requests=0)
        self.assertEqual(result["observations_executed"], 0)
        self.assertFalse(result["mandatory_complete"])
        self.assertEqual(self.backend.calls, [])

    async def test_max_steps_stops(self):
        result = await self.run_graph(max_steps=1)
        self.assertEqual(result["steps"], 1)
        self.assertEqual(result["stop_reason"], "max_steps")

    async def test_no_future_observations_in_thought(self):
        planner = ScriptedPlanner()
        await self.run_graph(planner=planner)
        first = planner.contexts[0]
        self.assertEqual([o["action"]["tool"] for o in first["previous_observations"]], ["scene_check"])
        self.assertLessEqual(len(first["actions"]), 5)

    async def test_no_gain_twice_stops(self):
        another = self.root/"fov2.png"
        another.write_bytes(b"another-simulation-view")
        self.runtime = EvidenceRuntime([], [self.image, another], sample_id="empty", a=self.fixture["a"], simulation=True)
        class NegativeBackend:
            def call(self, runtime, action, path):
                return {"ok": True, "present": False, "conf": 0.9}, None
        self.tools = LocalToolbox(self.runtime, ledger_path=self.root/"negative.json", backend=NegativeBackend())
        result = await self.run_graph()
        self.assertEqual(result["stop_reason"], "no_gain_twice")
        self.assertEqual(result["observation_requests"], 2)
        self.assertEqual(result["requests"], 4)

    async def test_illegal_planner_falls_back_without_arbitrary_tool(self):
        result = await self.run_graph(planner=ScriptedPlanner(["shell_exec"]))
        self.assertTrue(result["planner_degraded"])
        self.assertTrue(all(o["action"]["tool"] in {"scene_check", "presence_check"} for o in result["observations"]))

    async def test_planner_finish_does_not_clear_review(self):
        result = await self.run_graph(planner=ScriptedPlanner(["finish"]))
        self.assertEqual(result["stop_reason"], "planner_finish")
        self.assertTrue(result["needs_review"])

    async def test_scene_failure_requires_review(self):
        self.backend.replies["scene_check"] = TimeoutError("credential-do-not-log")
        result = await self.run_graph()
        self.assertFalse(result["mandatory_complete"])
        self.assertTrue(result["needs_review"])
        self.assertEqual(result["stop_reason"], "mandatory_context_incomplete")
        self.assertEqual(result["requests"], 1)
        self.assertNotIn("credential-do-not-log", json.dumps(result))

    async def test_sizes_cannot_be_injected(self):
        self.backend.replies["hypha"] = {"ok": True, "present": True, "conf": 0.95, "size_um": 10}
        state = await self.scene_first()
        action = next(a for a in self.runtime.legal_actions(state) if a["args"].get("feature_id") == "hypha")
        result = await self.tools.observe(action)
        self.assertFalse(result["accepted"])
        self.assertNotIn("hypha", self.runtime.score(state["observations"])["cards"])

    async def test_tool_cache_persists_across_toolbox_restart(self):
        action = self.runtime.legal_actions({})[0]
        first = await self.tools.observe(action)
        restarted = LocalToolbox(self.runtime, ledger_path=self.root/"ledger.json", backend=self.backend)
        second = await restarted.observe(action)
        self.assertTrue(first["accepted"])
        self.assertTrue(second["cache_hit"])
        self.assertEqual(second["new_requests"], 0)
        self.assertEqual(len(self.backend.calls), 1)

    async def test_pending_entry_never_automatically_retried(self):
        action = self.runtime.legal_actions({})[0]
        await self.tools.observe(action)
        value = read_json(self.root/"ledger.json")
        value["entries"][action["action_id"]].update(status="pending", accepted=False)
        write_json(self.root/"ledger.json", value)
        restarted = LocalToolbox(self.runtime, ledger_path=self.root/"ledger.json", backend=self.backend)
        result = await restarted.observe(action)
        self.assertFalse(result["accepted"])
        self.assertEqual(len(self.backend.calls), 1)

    async def test_image_mutation_rejected_before_request(self):
        action = self.runtime.legal_actions({})[0]
        self.image.write_bytes(b"changed")
        with self.assertRaises(IntegrationError):
            await self.tools.observe(action)
        self.assertEqual(self.backend.calls, [])

    async def test_skip_mandatory_denied(self):
        criterion = next(c for c in self.runtime.kb.herbs["猪苓"]["criteria"] if c["feature"] == "hypha")
        action = self.runtime.action("presence_check", {"image_id": next(iter(self.runtime.images)),
            "feature_id": "hypha", "morph_hint": criterion["morph"]}, "differential_presence")
        with self.assertRaises(IntegrationError):
            await self.tools.observe(action)
        self.assertEqual(self.backend.calls, [])

    async def test_p0_tool_and_skill_bypass_denied(self):
        state = await self.scene_first()
        action = self.runtime.legal_actions(state)[0]
        p0 = copy.deepcopy(action)
        p0["args"]["feature_id"] = "single_crystal"
        with self.assertRaises(IntegrationError):
            await self.tools.observe(p0)
        altered = self.runtime.action(action["tool"], action["args"], "../../secret")
        with self.assertRaises(IntegrationError):
            await self.tools.observe(altered)

    async def test_strict_observation_types(self):
        state = await self.scene_first()
        action = self.runtime.legal_actions(state)[0]
        for value in ({"ok": True, "present": "false", "conf": .9},
                      {"ok": True, "present": True, "conf": float('nan')},
                      {"ok": True, "present": True, "conf": True}):
            self.assertFalse(self.runtime.validate_observation(action, value))

    async def test_duplicate_stem_different_contents_rejected(self):
        other = self.root/"other/fov.png"
        other.parent.mkdir()
        other.write_bytes(b"another-image")
        with self.assertRaises(IntegrationError) as ctx:
            EvidenceRuntime(self.fixture["records"], [self.image, other], sample_id="demo")
        self.assertEqual(ctx.exception.code, "IMAGE_STEM_AMBIGUOUS")

    async def test_duplicate_image_content_deduplicated(self):
        other = self.root/"other.png"
        other.write_bytes(self.image.read_bytes())
        runtime = EvidenceRuntime(self.fixture["records"], [self.image, other], sample_id="demo")
        self.assertEqual(len(runtime.images), 1)
        self.assertEqual(len(runtime.legal_actions({})), 1)

    async def test_skill_read_cannot_escape_registry(self):
        self.assertEqual(len(self.runtime.skills.list()), 5)
        with self.assertRaises(IntegrationError):
            self.runtime.skills.get("../README")

    async def test_source_truth_not_in_planner_evidence(self):
        records = copy.deepcopy(self.fixture["records"])
        records[0]["herb"] = "TRUE_LABEL_MUST_NOT_LEAK"
        runtime = EvidenceRuntime(records, [self.image], sample_id="demo")
        self.assertNotIn("TRUE_LABEL", json.dumps(runtime.summary({}), ensure_ascii=False))
        self.assertNotIn("herb", runtime.records[0])

    async def test_real_network_disabled_in_toolbox(self):
        runtime = EvidenceRuntime(self.fixture["records"], [self.image], sample_id="real")
        tools = LocalToolbox(runtime, ledger_path=self.root/"real.json", backend=self.backend)
        with self.assertRaises(IntegrationError) as ctx:
            await tools.observe(runtime.legal_actions({})[0])
        self.assertEqual(ctx.exception.code, "NETWORK_DISABLED")

    async def test_mcp_stdio_handshake_tools_and_active_loop(self):
        context = self.root/"context.json"
        write_json(context, {"sample_id": "react_demo", "simulation": True,
                             "config_identity": self.runtime.config_identity,
                             "fixture": {**self.fixture, "images": [str(self.image)]}, "allow_network": False})
        async with connect_stdio(context) as tools:
            listed = await tools.session.list_tools()
            self.assertEqual({t.name for t in listed.tools}, {"list_skills", "get_skill", "read_evidence", "observe", "list_tools_registry", "read_tool", "recompute_tool"})
            self.assertEqual(len((await tools.call("list_skills"))["skills"]), 5)
            result = await run_react(self.runtime, tools, planner_mode="scripted", planner=ScriptedPlanner())
            self.assertEqual(result["decision"]["ranking"][0]["id"], "猪苓")
            self.assertEqual(result["decision"]["verdict"], "probable")
            # 同一个action再次请求，由服务器持久去重，不重发模型请求。
            cached = await tools.observe(result["observations"][0]["action"])
            self.assertEqual(cached["new_requests"], 0)
            with self.assertRaises(IntegrationError):
                await tools.call("get_skill", {"skill_id": "../README"})

    async def test_cold_workflow_mcp_returns_denial_without_native_import_stall(self):
        import sys
        from TCM_Fusion.lib.workflow_runtime import from_task
        weights=Path(paths.PROJECT_ROOT)/'TCM_Fusion/yolo-results/detection/yolo_runs/v1_yolov10s_1024/weights/best.pt'
        if not weights.exists():
            self.skipTest('real workflow asset not installed')
        config={'sample_id':'cold_native_probe','allow_network':False,'b':{'input_mode':'images'},
                'agent':{'image_roots':[str(self.image)],'workflow':{
                    'python':sys.executable,'execution':'active','agent_policy':'adaptive_v2',
                    'targeted_recovery':True,'max_seconds':30}}}
        directory=self.root/'cold';directory.mkdir()
        a={'status':'disabled','ranking':[]};write_json(directory/'a.json',a)
        runtime=from_task(config,a,directory)
        basis=runtime.score([])['evidence_digest']
        runtime.investigate(['G1'],'protocol lifecycle probe',basis)
        runtime.update_hypotheses([{'candidate_id':'G1','status':'paused',
                                    'reason':'parent paused branch'}],basis)
        action=runtime.action('detect_fov_initial',{'target_id':next(iter(runtime.images)),
            'evidence_version':runtime.workflow.store.current()['version']},'workflow_repair')
        context=directory/'context.json';write_json(context,{'workflow_config':config,
            'a_prediction':str(directory/'a.json'),'directory':str(directory),'allow_network':False})
        async def check():
            async with connect_stdio(context) as tools:
                snapshot=await tools.call('read_evidence')
                self.assertIn('G1',snapshot['candidate_interest']['inactive_ids'])
                self.assertNotIn('G1',snapshot['candidate_interest']['ids'])
                return await tools.recompute(action)
        result=await asyncio.wait_for(check(),20)
        self.assertFalse(result['accepted'])
        self.assertIn(result['error_code'],{'ACTION_DENIED','NETWORK_DISABLED'})
        self.assertEqual(read_json(runtime.workflow.operations)['entries'],{})

    async def test_from_b_binding_and_live_backend_paths(self):
        """覆盖真实run manifest读取分支；只替换模型返回，不替换规则/契约。"""
        from TCM_Fusion.lib.evidence_runtime import from_b_artifacts
        root = self.root/"b"
        write_json(root/"run_manifest.json", {"config": {"sample": "react_demo"}})
        (root/"measurements").mkdir()
        (root/"measurements/test_measurements.jsonl").write_text(
            json.dumps(self.fixture["records"][0])+'\n', encoding="utf-8")
        b = {"run_dir": str(root), "verdict": {**self.runtime.original, "source_images": [str(self.image)]}}
        with patch("TCM_Fusion.lib.evidence_runtime.file_hash", return_value="controlled-config-hash"):
            bound = from_b_artifacts(b, {"image_roots": [str(self.root)]}, sample_id="react_demo")
        self.assertEqual(len(bound.records), 1)
        self.assertEqual(bound.records[0]["feature"], "single_crystal")
        b["verdict"]["agent"] = {"vlm_calls": 1}
        with self.assertRaises(IntegrationError) as ctx:
            from_b_artifacts(b, {"image_roots": [str(self.root)]}, sample_id="react_demo")
        self.assertEqual(ctx.exception.code, "AGENT_HISTORY_MISSING")

    async def test_live_backend_reuses_original_visual_client_and_strict_parse(self):
        from TCM_Fusion.lib.agent_tools import LiveBackend
        from agent_loop import VLMClient
        config = self.root/"vlm.json"
        write_json(config, {"vlm": {"base_url": "https://invalid.example", "model": "fixture",
                                   "api_key": "fake", "reasoning_effort": "low"}})
        with patch.object(paths, "DETECTION_CONFIG_DIR", str(self.root)):
            config.rename(self.root/"vlm_config.json")
            backend = LiveBackend()
        self.assertIsInstance(backend.client, VLMClient)
        state = await self.scene_first()
        action = self.runtime.legal_actions(state)[0]
        body = {"choices": [{"message": {"content": '{"present":"false","conf":0.9}'}}]}
        with patch("TCM_Fusion.lib.hard_calls.post", return_value=body) as post, \
                patch("TCM_Fusion.lib.hard_calls.encode_image", return_value="data:image/png;base64,fixture"):
            with self.assertRaises(IntegrationError) as failure:
                backend.call(self.runtime, action, self.image)
        post.assert_called_once()
        self.assertEqual(failure.exception.code, "MODEL_FIELD_INVALID")
        ledger = self.root/"encode_timeout.json"
        toolbox = LocalToolbox(self.runtime, ledger_path=ledger, backend=backend)
        action = self.runtime.legal_actions({"observations": [], "attempted": []})[0]
        with patch("TCM_Fusion.lib.hard_calls.encode_image",
                   side_effect=IntegrationError("IMAGE_ENCODE_TIMEOUT", "test deadline")), \
                patch("TCM_Fusion.lib.hard_calls.post") as post:
            result = await toolbox.observe(action)
        post.assert_not_called()
        self.assertFalse(result["accepted"])
        self.assertEqual(result["error_code"], "IMAGE_ENCODE_TIMEOUT")
        self.assertEqual(result["result"]["error_code"], "IMAGE_ENCODE_TIMEOUT")
        self.assertEqual(read_json(ledger)["entries"][action["action_id"]]["status"], "failed")

        def truncated(payload, seconds, *, diagnostics):
            diagnostics["http_status"] = 200
            return {"choices": [{"finish_reason": "length", "message": {"content": '{"conf":'}}],
                    "usage": {"completion_tokens": 512}}

        toolbox = LocalToolbox(self.runtime, ledger_path=self.root/"scene_truncated.json", backend=backend)
        with patch("TCM_Fusion.lib.hard_calls.encode_image", return_value="data:image/png;base64,fixture"), \
                patch("TCM_Fusion.lib.hard_calls.post", side_effect=truncated):
            result = await toolbox.observe(action)
        self.assertFalse(result["accepted"])
        self.assertEqual(result["error_code"], "MODEL_OUTPUT_TRUNCATED")
        self.assertEqual(result["diagnostics"]["http_status"], 200)
        self.assertEqual(result["diagnostics"]["finish_reason"], "length")
        self.assertEqual(result["usage"], {"completion_tokens": 512})

        from types import SimpleNamespace
        from unittest.mock import Mock
        charge = Mock()
        valid = {"choices": [{"finish_reason": "stop", "message": {"content":
                 '{"conf":0.8,"cell_wall_visible":false,"crystal_location":"free",'
                 '"arrangement":"scattered","abundance":"many"}'}}]}
        with patch.object(self.runtime, "workflow", SimpleNamespace(charge=charge), create=True), \
                patch("TCM_Fusion.lib.hard_calls.encode_image", return_value="data:image/png;base64,fixture"), \
                patch("TCM_Fusion.lib.hard_calls.post", side_effect=[
                    IntegrationError("UPSTREAM_REMOTE_DISCONNECTED", "test disconnect"), valid
                ]) as post:
            value, _ = backend.call(self.runtime, action, self.image)
        self.assertTrue(value["ok"])
        self.assertEqual(post.call_count, 2)
        self.assertEqual(post.call_args.args[0]["reasoning_effort"], "low")
        charge.assert_called_once_with("network", "scene-retry:" + action["action_id"])
        self.assertEqual(backend.client.diagnostics["previous_error_code"], "UPSTREAM_REMOTE_DISCONNECTED")

    async def test_new_evidence_does_not_force_wrong_ranking_change(self):
        records = copy.deepcopy(self.fixture["records"])
        records[0]["primary_value_um"] = 30
        self.runtime = EvidenceRuntime(records, [self.image], sample_id="conflict", a=self.fixture["a"], simulation=True)
        self.tools = LocalToolbox(self.runtime, ledger_path=self.root/"conflict.json", backend=self.backend)
        result = await self.run_graph()
        self.assertEqual(result["decision"]["ranking"][0]["id"], "G3")
        self.assertTrue(result["needs_review"])

    async def test_original_fixed_policy_comparison_counts_planning(self):
        from TCM_Fusion.tests.fixed_baseline import compare_fixed_fixture
        result = await self.run_graph()
        comparison = compare_fixed_fixture(self.runtime, FixtureBackend(self.fixture["replies"]), result)
        self.assertTrue(comparison["same_measurement_start"])
        self.assertGreater(comparison["react"]["planning_requests"], 0)
        self.assertEqual(comparison["react"]["duplicate_requests"], 0)
        self.assertIsNone(comparison["accuracy_gain"])
        self.assertGreater(comparison["fixed"]["observation_requests"], 0)

    async def test_zero_support_candidates_do_not_trigger_noise_checks(self):
        state = await self.scene_first()
        actions = self.runtime.legal_actions(state)
        self.assertTrue(actions)
        self.assertEqual({a["args"]["feature_id"] for a in actions}, {"hypha"})

    async def test_direct_library_llm_does_not_bypass_network_opt_in(self):
        with self.assertRaises(IntegrationError) as ctx:
            await run_react(self.runtime, self.tools, planner_mode="llm")
        self.assertEqual(ctx.exception.code, "NETWORK_DISABLED")

    async def test_retired_active_path_rejected_preserves_baseline(self):
        from TCM_Fusion.lib.pipeline import run_pipeline
        root = self.root/"complete_b"
        write_json(root/"run_manifest.json", {"run_id": root.name, "config": {"sample": "react_demo"}})
        (root/"measurements").mkdir()
        (root/"measurements/test_measurements.jsonl").write_text(
            json.dumps(self.fixture["records"][0])+'\n', encoding="utf-8")
        vpath = root/"verdicts/reports/react_demo/verdict.json"
        write_json(vpath, {"sample_id": "react_demo", **self.runtime.original,
                           "source_images": [str(self.image)], "agent": {"vlm_calls": 0}})
        before = vpath.read_bytes()
        write_json(self.root/"vlm_config.json", {"vlm": {"model": "fake", "base_url": "https://invalid.example"}})
        config = {"sample_id": "react_demo", "allow_network": True,
                  "a": {"mode": "unavailable"}, "b": {"mode": "existing", "run_dir": str(root)},
                  "agent": {"mode": "active", "transport": "local", "planner": "llm",
                            "image_roots": [str(self.image)], "judgement_request_budget": 6}}
        # The retired active path must fail before inference or modifying B artifacts.
        with patch("urllib.request.urlopen") as network, patch("subprocess.run") as processes:
            with self.assertRaises(IntegrationError) as error:
                await asyncio.to_thread(run_pipeline, config, base=self.root/"integration")
        self.assertEqual(error.exception.code, "CONFIG_INVALID")
        network.assert_not_called()
        processes.assert_not_called()
        self.assertEqual(vpath.read_bytes(), before)
