"""可控输入与假子进程下的适配边界回归，绝不把模拟结果当真实模型效果。
该组用例由项目根 selftest 发现。涉及模型回复的用例使用可控输入，涉及进程的用例以各断言规定的执行边界为准。

主要定义：
- IntegrationTests A 输出解析: `parse_a_stdout` 的精度与顺序，Top-K 不完整、类别重复、概率和异常均被拒。
- A 预测缓存: `predict_a` 缓存命中、权重变更失效、损坏缓存 `CACHE_INVALID`、输入变更 `INPUT_CHANGED`、缺图与缺折检查点的错误码。
- 原子写入: `write_json` 在 Windows 共享冲突下重试、永久拒绝次数有界且保留旧内容、非 Windows 拒绝不重试。
- 融合与影子选择: `fuse` 冲突不改写 B、`select_shadow` 仅影子不注入卡片、非法 LLM 动作降级、调用失败不重试。
- 配置与流水线: `validate_config` 拒绝跨样本、未知字段与未声明的 recorded 数据；`run_pipeline` 失败任务落盘。
- B 适配: `read_b_verdict` 样本契约、`execute_b` 只读回放不改动工件、网络未 opt-in 先失败、子进程退出码映射。

关联：
- 直接依赖 「lib/application/pipeline.py」、「lib/decision/candidate_policy.py」、「lib/decision/collaboration.py」、「lib/runtime/storage.py」、「lib/workflow/adapters.py」。
- 统一执行见 tests/README.md。
"""
from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import paths
from rule_engine import KnowledgeBase
from TCM_Fusion.lib.workflow.adapters import (execute_b, parse_a_stdout, predict_a, read_b_verdict,
                                      run_process)
from TCM_Fusion.lib.decision.collaboration import fuse, knowledge_base, propose_actions, select_shadow
from TCM_Fusion.lib.decision.candidate_policy import candidates
from TCM_Fusion.lib.application.pipeline import run_pipeline, validate_config
from TCM_Fusion.lib.runtime.storage import IntegrationError, read_json, write_json


TEXT = """模型: ResNet50 Full Fine-tuning, 5-fold ensemble
类别数: 2
Top-2 预测结果:
  1. 甘草                0.8000  (80.00%)
  2. 白芍                0.2000  (20.00%)
"""
FOLD_TEXT = TEXT + "FOLD_RANKINGS_JSON=" + json.dumps([
    [{"rank": 1, "label": "甘草", "probability": .9},
     {"rank": 2, "label": "白芍", "probability": .1}]
] * 4 + [[{"rank": 1, "label": "白芍", "probability": .6},
           {"rank": 2, "label": "甘草", "probability": .4}]])
BASE = Path(paths.PROJECT_ROOT)


class IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="fusion-test-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.a = read_json(BASE/"tests/fixtures/a_prediction.json")
        self.a["status"] = "ok"
        self.b = read_json(BASE/"tests/fixtures/b_verdict.json")
        self.kb = knowledge_base()

    def test_original_cli_precision_and_order(self):
        value = parse_a_stdout(TEXT)
        self.assertEqual(value["ranking"][0]["label"], "甘草")
        self.assertEqual(value["probability_precision"], 4)
        self.assertFalse(value["resident_models"])

    def test_atomic_write_retries_windows_sharing_and_preserves_bytes(self):
        obj={'evidence':'不改内容','count':3};baseline=self.root/'baseline.json';target=self.root/'target.json'
        write_json(baseline,obj);write_json(target,{'previous':True})
        original=os.replace;calls=[]
        def replace(source,destination):
            calls.append(1)
            if len(calls)<3:
                exc=PermissionError('simulated sharing');exc.winerror=32;raise exc
            return original(source,destination)
        with patch('TCM_Fusion.lib.runtime.storage.os.replace',side_effect=replace),patch('TCM_Fusion.lib.runtime.storage.time.sleep'):
            write_json(target,obj)
        self.assertEqual(target.read_bytes(),baseline.read_bytes());self.assertEqual(len(calls),3)

    def test_atomic_write_permanent_denial_is_bounded_and_keeps_previous(self):
        target=self.root/'target.json';write_json(target,{'previous':True});before=target.read_bytes()
        exc=PermissionError('simulated permanent denial');exc.winerror=5
        with patch('TCM_Fusion.lib.runtime.storage.os.replace',side_effect=exc) as replace,patch('TCM_Fusion.lib.runtime.storage.time.sleep'):
            with self.assertRaises(PermissionError):write_json(target,{'new':True})
        self.assertEqual(replace.call_count,8);self.assertEqual(target.read_bytes(),before)
        self.assertEqual(list(self.root.glob('.write-*.json')),[])

    def test_atomic_write_non_windows_denial_is_not_retried(self):
        with patch('TCM_Fusion.lib.runtime.storage.os.replace',side_effect=PermissionError('denied')) as replace:
            with self.assertRaises(PermissionError):write_json(self.root/'target.json',{})
        self.assertEqual(replace.call_count,1)

    def test_incomplete_top_k_rejected(self):
        with self.assertRaises(IntegrationError):
            parse_a_stdout(TEXT.replace("  2. 白芍                0.2000  (20.00%)", ""))

    def test_duplicate_class_rejected(self):
        with self.assertRaises(IntegrationError):
            parse_a_stdout(TEXT.replace("白芍", "甘草"))

    def test_probability_sum_rejected(self):
        with self.assertRaises(IntegrationError):
            parse_a_stdout(TEXT.replace("0.2000", "0.9000"))

    def _a_config(self):
        image = self.root/"图像.jpg"
        image.write_bytes(b"controlled-image-for-cli-adapter")
        weights = self.root/"weights"
        weights.mkdir()
        for i in range(1, 6):
            (weights/f"fold{i}_best.pt").write_bytes(f"not-real-weight-{i}".encode())
        return {"mode": "cli", "image": str(image), "checkpoint_dir": str(weights), "python": sys.executable}

    def test_cache_hit_no_repeated_process(self):
        cfg = self._a_config()
        with patch("TCM_Fusion.lib.workflow.adapters.run_process", return_value=FOLD_TEXT) as process:
            first = predict_a(cfg, self.root/"cache")
            second = predict_a(cfg, self.root/"cache")
        self.assertFalse(first["cache_hit"])
        self.assertTrue(second["cache_hit"])
        self.assertEqual(process.call_count, 1)
        self.assertIn("1000000", process.call_args.args[0])
        self.assertIn("--emit-fold-rankings", process.call_args.args[0])
        kb = KnowledgeBase(Path(paths.JUDGE_DIR)/"data"/"herb_criteria.json")
        interest = candidates(second, kb, config={"top_k": 1})
        self.assertEqual(interest["ids"], ["甘草", "白芍"])
        self.assertEqual([r["ranking"][0]["label"] for r in interest["fold_top_k"]], ["甘草"] * 4 + ["白芍"])
        self.assertNotIn("selected_mass", interest)

    def test_cache_weight_change_invalidates(self):
        cfg = self._a_config()
        with patch("TCM_Fusion.lib.workflow.adapters.run_process", return_value=FOLD_TEXT) as process:
            one = predict_a(cfg, self.root/"cache")
            (Path(cfg["checkpoint_dir"])/"fold3_best.pt").write_bytes(b"changed")
            two = predict_a(cfg, self.root/"cache")
        self.assertNotEqual(one["cache_key"], two["cache_key"])
        self.assertEqual(process.call_count, 2)

    def test_corrupt_cache_rejected_without_inference(self):
        cfg = self._a_config()
        with patch("TCM_Fusion.lib.workflow.adapters.run_process", return_value=FOLD_TEXT) as process:
            predict_a(cfg, self.root/"cache")
            path = next((self.root/"cache").glob("*.json"))
            value = read_json(path)
            value["ranking"] = ["invalid"]
            write_json(path, value)
            with self.assertRaises(IntegrationError) as ctx:
                predict_a(cfg, self.root/"cache")
        self.assertEqual(ctx.exception.code, "CACHE_INVALID")
        self.assertEqual(process.call_count, 1)

    def test_meta_fold_count_cannot_be_overridden(self):
        cfg = self._a_config()
        with patch("TCM_Fusion.lib.workflow.adapters.run_process", return_value=TEXT.replace("5-fold", "4-fold")) as process:
            with self.assertRaises(IntegrationError) as ctx:
                predict_a(cfg, self.root/"cache")
        self.assertEqual(ctx.exception.code, "A_CHECKPOINT_MISMATCH")
        self.assertNotIn("--k-folds", process.call_args.args[0])

    def test_input_mutation_not_cached(self):
        cfg = self._a_config()
        def changed(argv, **kwargs):
            Path(cfg["image"]).write_bytes(b"changed-during-inference")
            return FOLD_TEXT
        with patch("TCM_Fusion.lib.workflow.adapters.run_process", side_effect=changed):
            with self.assertRaises(IntegrationError) as ctx:
                predict_a(cfg, self.root/"cache")
        self.assertEqual(ctx.exception.code, "INPUT_CHANGED")
        self.assertFalse((self.root/"cache").exists())

    def test_missing_a_input_explicit(self):
        with self.assertRaises(IntegrationError) as ctx:
            predict_a({"image": str(self.root/"missing")}, self.root/"cache")
        self.assertEqual(ctx.exception.code, "A_IMAGE_MISSING")

    def test_fold_gap_rejected(self):
        cfg = self._a_config()
        (Path(cfg["checkpoint_dir"])/"fold2_best.pt").unlink()
        with self.assertRaises(IntegrationError) as ctx:
            predict_a(cfg, self.root/"cache")
        self.assertEqual(ctx.exception.code, "A_CHECKPOINT_MISSING")

    def test_conflict_never_mutates_b(self):
        original = copy.deepcopy(self.b)
        result = fuse(self.a, self.b, self.kb)
        self.assertEqual(result["agreement"], "conflict")
        self.assertTrue(result["needs_review"])
        self.assertEqual(self.b, original)
        self.assertEqual(result["b_verdict"], original)

    def test_agreement_does_not_add_support(self):
        self.a["ranking"][0]["label"] = "白芍"
        result = fuse(self.a, self.b, self.kb)
        self.assertEqual(result["agreement"], "agree")
        self.assertEqual(result["b_verdict"]["support_feature_count"], 2)

    def test_group_compatible_not_upgraded(self):
        kb = {"herbs": [{"herb": "甲"}], "groups": {"G": {"members": ["甲", "乙"]}}}
        a = {"status": "ok", "ranking": [{"label": "甲", "probability": 1.0}]}
        self.b.update(verdict="group_level", conclusion={"id": "G", "display": "甲/乙"})
        result = fuse(a, self.b, kb)
        self.assertEqual(result["agreement"], "group_compatible")
        self.assertEqual(result["b_verdict"]["verdict"], "group_level")

    def test_unknown_a_and_b_abstention(self):
        self.a["ranking"][0]["label"] = "未支持类别"
        self.assertEqual(fuse(self.a, self.b, self.kb)["agreement"], "a_unsupported")
        self.a["ranking"][0]["label"] = "甘草"
        self.b.update(verdict="insufficient_evidence", conclusion=None)
        self.assertEqual(fuse(self.a, self.b, self.kb)["agreement"], "b_insufficient")

    def test_only_missing_p1_c_actions_deduplicate(self):
        image = self.root/"原图.jpg"
        image.write_bytes(b"fixture-not-decoded-by-planner")
        self.b["source_images"] = [str(image), str(image)]
        self.b["score_breakdown"] = {"terms": [
            {"feature_id": "f", "role": "required", "tier": "P1", "state": "NOT_DETECTED"},
            {"feature_id": "p0", "role": "required", "tier": "P0", "state": "NOT_DETECTED"},
            {"feature_id": "present", "role": "required", "tier": "C", "state": "IN_RANGE"}]}
        actions = propose_actions(self.b)
        checks = [a for a in actions if a["kind"] == "presence_check"]
        self.assertEqual(len(checks), 1)
        self.assertEqual(checks[0]["feature_id"], "f")
        self.assertTrue(checks[0]["requires_history_verification"])

    def test_shadow_only_no_card_injection(self):
        original = copy.deepcopy(self.b)
        fusion = fuse(self.a, self.b, self.kb)
        value = select_shadow(fusion, self.b, {})
        self.assertTrue(value["shadow_only"])
        self.assertEqual(value["observations_executed"], 0)
        self.assertEqual(self.b, original)

    def test_llm_invalid_action_falls_back(self):
        fusion = fuse(self.a, self.b, self.kb)
        value = select_shadow(fusion, self.b, {"mode": "llm_shadow"},
                              client=lambda actions: {"action_id": "execute_arbitrary_code"})
        self.assertTrue(value["degraded"])
        self.assertEqual(value["choice"], "review")
        self.assertEqual(value["planning_requests"], 1)

    def test_llm_failure_counted_without_retry(self):
        calls = []
        def failed(actions):
            calls.append(1)
            raise TimeoutError("secret-must-not-be-logged")
        value = select_shadow(fuse(self.a, self.b, self.kb), self.b,
                              {"mode": "llm_shadow"}, client=failed)
        self.assertEqual(calls, [1])
        self.assertNotIn("secret-must", json.dumps(value))

    def test_llm_reuses_original_transport_once(self):
        with patch("TCM_Fusion.lib.decision.collaboration.read_json", return_value={"vlm": {"max_tokens": 32768}}), \
                patch("adjudicator.AdjudicatorClient") as transport:
            transport.return_value.api_key = "fixture-not-real"
            transport.return_value.model = "test"
            transport.return_value._post.return_value = {
                "choices": [{"message": {"content": '{"action_id":"review"}'}}]}
            value = select_shadow(fuse(self.a, self.b, self.kb), self.b,
                                  {"mode": "llm_shadow"}, allow_network=True)
        self.assertFalse(value["degraded"])
        transport.return_value._post.assert_called_once()
        transport.return_value.arbitrate.assert_not_called()

    def test_invalid_agent_config_fails_before_b(self):
        with self.assertRaises(IntegrationError):
            validate_config({"agent": {"mode": "active"}})

    def test_budget_exhaustion_no_planning(self):
        self.b["agent"]["vlm_calls"] = 6
        with patch("urllib.request.urlopen") as network:
            value = select_shadow(fuse(self.a, self.b, self.kb), self.b,
                                  {"mode": "llm_shadow"}, allow_network=True)
        self.assertEqual(value["planning_requests"], 0)
        network.assert_not_called()

    def test_network_and_active_modes_rejected(self):
        with self.assertRaises(IntegrationError):
            select_shadow(fuse(self.a, self.b, self.kb), self.b, {"mode": "llm_shadow"})
        with self.assertRaises(IntegrationError):
            select_shadow(fuse(self.a, self.b, self.kb), self.b, {"mode": "active"})

    def test_sample_traversal_and_unknown_fields_rejected(self):
        for config in ({"sample_id": "../outside"}, {"a": {"shell": "cmd"}}, {"allow_network": "false"}):
            with self.subTest(config=config), self.assertRaises(IntegrationError):
                validate_config(config)

    def test_cross_sample_images_rejected(self):
        with self.assertRaises(IntegrationError) as ctx:
            validate_config({"a": {"image": str(self.root/"outside.jpg")},
                             "b": {"input": str(self.root/"sample")}})
        self.assertEqual(ctx.exception.code, "SAMPLE_MISMATCH")

    def test_recorded_data_require_opt_in(self):
        cfg = read_json(BASE/"tests/fixtures/recorded_pipeline.json")
        cfg["allow_recorded"] = False
        with self.assertRaises(IntegrationError):
            validate_config(cfg)

    def test_demo_exercises_real_orchestration_without_network(self):
        with patch("urllib.request.urlopen") as network, patch("subprocess.run") as processes:
            directory = run_pipeline(read_json(BASE/"tests/fixtures/recorded_pipeline.json"), base=self.root)
        value = read_json(directory/"fusion.json")
        self.assertTrue(value["simulation"])
        self.assertEqual(value["fusion"]["agreement"], "conflict")
        self.assertEqual(read_json(directory/"job.json")["status"], "completed")
        network.assert_not_called()
        processes.assert_not_called()

    def test_failed_job_persisted(self):
        cfg = {"sample_id": "test", "a": {"image": str(self.root/"missing")}}
        with self.assertRaises(IntegrationError):
            run_pipeline(cfg, base=self.root)
        statuses = list((self.root/"runs").glob("*/job.json"))
        self.assertEqual(len(statuses), 1)
        self.assertEqual(read_json(statuses[0])["status"], "failed")

    def test_b_record_sample_contract(self):
        with self.assertRaises(IntegrationError):
            read_b_verdict(BASE/"tests/fixtures/b_verdict.json", "different_sample")

    def test_existing_b_directory_not_proof_of_completion(self):
        incomplete = self.root/"b"
        (incomplete/"verdicts").mkdir(parents=True)
        with self.assertRaises(IntegrationError) as ctx:
            execute_b({"mode": "existing", "run_dir": str(incomplete)}, "demo", "unused")
        self.assertEqual(ctx.exception.code, "B_INCOMPLETE")

    def _complete_b(self, directory):
        write_json(directory/"run_manifest.json", {"run_id": directory.name})
        measurements = directory/"measurements"
        measurements.mkdir(exist_ok=True)
        (measurements/"demo_measurements.jsonl").write_text(
            json.dumps({"image_stem": "s", "instance_id": 1, "feature": "cluster_crystal",
                        "calibration_px_per_um": 2, "calibration_status": "ok"})+'\n', encoding="utf-8")
        write_json(directory/"verdicts/reports/demo/verdict.json", self.b)

    def test_complete_b_read_only_roundtrip(self):
        directory = self.root/"existing"
        self._complete_b(directory)
        before = {p.relative_to(directory): p.read_bytes() for p in directory.rglob("*") if p.is_file()}
        result = execute_b({"mode": "existing", "run_dir": str(directory)}, "demo", "unused")
        self.assertEqual(result["verdict"], self.b)
        self.assertEqual(before, {p.relative_to(directory): p.read_bytes()
                                  for p in directory.rglob("*") if p.is_file()})

    def test_b_run_invokes_original_entry_then_reads_contract(self):
        source = self.root/"masks"
        source.mkdir()
        (source/"placeholder.png").write_bytes(b"fake-child-does-not-decode")
        def child(argv):
            self.assertEqual(Path(argv[3]), Path(paths.PROJECT_ROOT)/"run_pipeline.py")
            self.assertIn("--masks", argv)
            self.assertNotIn("--force", argv)
            self._complete_b(self.root/"b_runs"/"integration_test")
            return ""
        with patch.object(paths, "RUNS_ROOT", str(self.root/"b_runs")), \
                patch("TCM_Fusion.lib.workflow.adapters.run_process", side_effect=child):
            result = execute_b({"input_mode": "masks", "input": str(source)},
                               "demo", "integration_test")
        self.assertEqual(result["verdict"], self.b)

    def test_optional_a_unavailable_preserves_b(self):
        cfg = read_json(BASE/"tests/fixtures/recorded_pipeline.json")
        cfg["a"] = {"mode": "cli", "required": False, "image": str(self.root/"missing")}
        directory = run_pipeline(cfg, base=self.root)
        value = read_json(directory/"fusion.json")
        self.assertEqual(value["fusion"]["agreement"], "a_unavailable")
        self.assertEqual(value["b"]["verdict"], self.b)

    def test_b_network_opt_in_before_process(self):
        with patch("subprocess.run") as processes:
            with self.assertRaises(IntegrationError) as ctx:
                execute_b({"input_mode": "images", "input": str(self.root)}, "demo", "unused")
        self.assertEqual(ctx.exception.code, "NETWORK_DISABLED")
        processes.assert_not_called()

    def test_real_subprocess_exit_code(self):
        with self.assertRaises(IntegrationError) as ctx:
            run_process([sys.executable, "-c", "raise SystemExit(7)"])
        self.assertEqual(ctx.exception.code, "PROCESS_FAILED")

    def test_namespace_paths_use_project_authority(self):
        from TCM_Fusion.lib.runtime.storage import output_root, project_path
        self.assertEqual(project_path("MICROSCOPE-main"), Path(paths.PROJECT_ROOT)/"MICROSCOPE-main")
        self.assertTrue(output_root().is_relative_to(Path(paths.UI_OUT_ROOT)))
