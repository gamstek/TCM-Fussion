"""覆盖本地 LangSmith 投影、观察中的 D4 分支与上传门控。

主要定义：TraceTests 使用四个视野的纯 JSON 工件模拟运行中与结束两种状态。
关联：cli.langsmith_trace、lib.offline.langsmith_trace。SDK 发送端使用记录替身。
"""

from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timedelta, timezone
from io import StringIO
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from uuid import NAMESPACE_URL, UUID, uuid5

import paths
import requests
from TCM_Fusion.cli import langsmith_trace as cli
from TCM_Fusion.lib.offline.langsmith_trace import LangSmithPublisher, Span, find_run, project_trace
from TCM_Fusion.lib.runtime.storage import write_json


class CaptureClient:
    def __init__(self):
        self.created = {}
        self.patched = []
        self.updated = set()

    def create_run(self, **run):
        from langsmith.utils import LangSmithConflictError
        if run["id"] in self.created:
            raise LangSmithConflictError("payload already received")
        self.created[run["id"]] = run

    def update_run(self, **run):
        from langsmith.utils import LangSmithConflictError
        if run["run_id"] in self.updated:
            raise LangSmithConflictError("payload already received")
        self.updated.add(run["run_id"])
        self.patched.append(run)


class TraceTests(unittest.TestCase):
    def test_observation_waves_keep_one_authorizing_round(self):
        from TCM_Fusion.lib.offline.langsmith_trace import _events
        thought = {'phase':'thought', 'action_id':'a', 'execution_choice':{'kind':'observe_questions'}}
        events = [{'phase':'planning_started','request_number':2}, thought,
                  {'phase':'observation_group_wave','action_ids':['a','b']},
                  {'phase':'observation_group_wave','action_ids':['c']},
                  {'phase':'observation_group_return','unissued':['d']}]
        _, choices, _, _, _ = _events(events)
        self.assertEqual(choices, dict.fromkeys(['a','b','c'], (2,thought)))

    def test_sequential_identity_items_keep_authorizing_planning_round(self):
        from TCM_Fusion.lib.offline.langsmith_trace import _events, _plan_choice
        thought = {'phase':'thought', 'action_id':'first', 'tool':'classify_target',
                   'execution_choice':{'kind':'identity_group','requested':{'action_ids':['A3','A5','A6']}}}
        events = [{'phase':'planning_started','request_number':3}, thought,
                  {'phase':'identity_group_item','action_id':'first'},
                  {'phase':'identity_group_item','action_id':'late-bound-second'},
                  {'phase':'identity_group_return','unissued':['third']}]
        plans, choices, _, _, _ = _events(events)
        self.assertEqual(set(choices), {'first','late-bound-second'})
        self.assertEqual(choices['late-bound-second'], (3,thought))
        self.assertEqual(_plan_choice(plans[3], {}, include_decision_summary=False)['decision_kind'], 'identity_group')

    def setUp(self):
        directory = Path(paths.RESULTS)
        directory.mkdir(exist_ok=True)
        temporary = TemporaryDirectory(dir=directory)
        self.addCleanup(temporary.cleanup)
        self.case = Path(temporary.name)
        self.job_id = "fusion_" + "b" * 32
        self.run = self.case / "runs" / self.job_id
        self.start = 1_700_000_000
        self.secret = "RAW_SECRET_PAYLOAD"
        images = [str(i) * 64 for i in range(1, 5)]
        self.write(self.case / "input_proof.json", {"views": [
            {"source": f"D:/private/{self.secret}.jpg", "sha256": sha} for sha in images]})
        self.write(self.run / "code_signature_start.json", {"at": "2023-11-14T22:13:20+00:00"})
        self.write(self.run / "code_signature_before_agent.json", {"at": "2023-11-14T22:13:22+00:00"})
        self.write(self.run / "job.json", {"job_id": self.job_id, "status": "running", "stage": "workflow_agent"})
        self.rows = {}
        for i, sha in enumerate(images, 1):
            self.rows[str(i)] = {
                "tool": "detect_fov_initial", "epoch": "initial", "image_id": f"img_{sha}",
                "action_id": f"action-{i}", "selected_at": self.start + 8,
                "started_at": self.start + 9 + i / 10 if i < 4 else None,
                "ended_at": self.start + 10 + i / 10 if i < 3 else None,
                "status": "done" if i < 3 else "pending",
            }
        self.events = [
            {"phase": "planning_started", "request_number": 1, "started_at": self.start + 3},
            {"phase": "planning_timing", "request_number": 1, "ended_at": self.start + 7},
            {"phase": "thought", "tool": "detect_fov_initial", "action_id": "action-1",
             "physical_partner_ids": ["action-2", "action-3", "action-4"], "decision_summary": self.secret,
             "execution_choice": {"requested": {"action_id": "A1"}},
             "usage": {"prompt_tokens": 31, "completion_tokens": 7, "total_tokens": 38}},
            {"phase": "recompute", "action_id": "action-1", "accepted": True,
             "feedback": {"candidate_support_changes": {"白芍": ["cluster_crystal"]},
                          "criterion_transitions": [{"candidate_id": "白芍", "feature_id": "cluster_crystal",
                                                     "role": "required", "before": "OUT_OF_RANGE", "after": "IN_RANGE"}]},
             "result": {"image_payload": self.secret}},
            {"phase": "planning_started", "request_number": 2, "started_at": self.start + 11},
        ]
        self.save_snapshots()

    @staticmethod
    def write(path, data):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    def save_snapshots(self):
        self.write(self.run / "agent_trace.json", {"events": self.events, "simulation": True})
        self.write(self.run / "workflow" / "operations.json", {"entries": self.rows})

    def snapshot(self, **kwargs):
        return project_trace(self.run, ({"白芍"}, {"cluster_crystal"}), **kwargs)[1]

    def planner_offer(self):
        image = "img_" + "1" * 64
        return {"phase": "planner_offer", "request_number": 1, "actual_llm_short_ids": True, "menu": {
            "actions": [
                {"action_id": "A1", "tool": "detect_fov_initial", "args": {"target_id": f"fov:{image}"}},
                {"action_id": "A2", "tool": "presence_check", "args": {
                    "image_id": image, "feature_id": "cluster_crystal", "morph_hint": self.secret}},
                {"action_id": "A3", "tool": "segment_instance"},
                {"action_id": "A4", "tool": "list_features"},
            ],
            "action_state": {"A3": {"image_id": image, "feature_id": "cluster_crystal"}},
            "remaining_budget": {"planning": 7, "observation": 4, "unused": self.secret},
            "parallel_observations": [{"action_id": "O1", "tool": "presence_check", "args_from_action": "A2"}],
            "execution_options": {
                "questions": [{"question_id": "Q1", "action_id": "A2", "purpose": "identity",
                               "shared_consumers": ["白芍", self.secret]}],
                "goals": [{"goal_id": "G1", "entry_action_id": "A3", "goal": {
                    "semantic_task": "measure_length", "instance_ref": self.secret}}],
                "problem_goals": [{"choice_id": "P1", "entry": "reuse", "image_id": image,
                                   "feature_id": "cluster_crystal", "dimension": "feret_max"}],
            },
            "investigation": {
                "investigation_focus": {"a_leads": ["白芍"]},
                "candidate_audit": {"candidates": [
                    {"candidate_id": "白芍", "required": {"cluster_crystal": "unchecked"}},
                ]},
                "comparison_problems": {"active_candidates": ["白芍"], "questions": [
                    {"feature_id": "cluster_crystal", "purpose": "measure", "dimension": "feret_max",
                     "consumers": [{"candidate_id": "白芍", "role": "required"}],
                     "entry_paths": [{"action_id": "A1", "kind": "discovery_prerequisite"},
                                     {"action_id": self.secret, "kind": "feature_step"}]},
                ]},
            },
        }}

    def test_visual_question_tool_receipt_and_next_menu_have_opt_in_text(self):
        brief = '请沿图中清晰的边缘跟踪纹理。' * 20 + '末尾请指出最清楚的一段。'
        self.assertGreater(len(brief), 240)
        receipt = {'status':'reported','brief':brief,'tiles':[
            {'tile':0,'status':'reported','brief':brief,'finding':'边缘有横纹','location':'右侧','readability':'partial'},
            {'tile':1,'status':'unreported','brief':brief+' D:/private/sample.png','finding':'D:/private/secret'}]}
        for row in self.events:
            if row['phase'] == 'thought':
                row['visual_briefs'] = {'action-1':brief}
            if row['phase'] == 'recompute':
                row['result'] = {'accepted':True,'status':'done','visual_observation':receipt}
        self.rows['1']['tool'] = 'feature_scan'
        offer = self.planner_offer()
        offer['request_number'] = 2
        offer['menu']['previous_visual_observations'] = [receipt]
        self.events.append(offer)
        self.save_snapshots()
        compact = {s.key:s for s in self.snapshot()}
        expanded = {s.key:s for s in self.snapshot(include_visual_observations=True)}
        self.assertEqual(compact['plan-1'].outputs['requested_visual_briefs'], 1)
        self.assertNotIn('planned_visual_questions', compact['plan-1'].outputs)
        self.assertEqual(expanded['plan-1'].outputs['planned_visual_questions'], [brief])
        next_input = expanded['plan-2'].inputs['previous_visual_observations'][0]
        actual = next(s for s in expanded.values() if s.inputs.get('tool') == 'feature_scan')
        self.assertEqual(actual.outputs['visual_observation'], next_input)
        self.assertEqual(next_input['brief'], brief)
        self.assertEqual(next_input['tiles'][0]['brief'], brief)
        self.assertNotIn('brief', next_input['tiles'][1])
        self.assertEqual(next_input['tiles'][0]['finding'], '边缘有横纹')
        self.assertNotIn('finding', next_input['tiles'][1])
        self.assertNotIn('brief', compact['plan-2'].inputs['previous_visual_observations'][0])

    def test_live_group_carries_views_and_qualified_feedback(self):
        self.assertEqual(find_run(self.case), self.run)
        projected = {span.key: span for span in self.snapshot()}
        self.assertEqual(projected["case"].inputs["view_count"], 4)
        self.assertIsNone(projected["plan-2"].end)
        self.assertIsNone(projected["d4-initial"].end)
        self.assertEqual(projected["d4-initial"].inputs["branch_count"], 4)
        self.assertEqual(projected["d4-initial-3"].inputs["view"], "view_03")
        self.assertNotIn("d4-initial-4", projected)
        completed = projected["d4-initial-1"]
        self.assertEqual(completed.parent, "d4-initial")
        self.assertEqual(completed.inputs["planning_request"], 1)
        self.assertEqual(completed.outputs["criterion_transitions"][0]["after"], "IN_RANGE")
        self.assertEqual(projected["plan-1"].outputs["usage_metadata"]["total_tokens"], 38)
        self.assertNotIn(self.secret, repr(projected))

    def test_shortened_view_is_bound_only_by_this_menu(self):
        from TCM_Fusion.lib.offline.langsmith_trace import _menu_options
        view = 'img_' + 'f'*64
        menu = {'view_id_map':{'@V1@':view}, 'actions':[
            {'action_id':'A1', 'tool':'detect_fov_initial', 'args':{'target_id':'crop:@V1@:c'}}]}
        self.assertEqual(_menu_options(menu, {view:'view_01'}, ({},{}))['A1']['view'], 'view_01')
        self.assertNotIn('view', _menu_options(menu, {}, ({},{}))['A1'])

    def test_planning_context_describes_real_menu_choices(self):
        self.events.insert(1, self.planner_offer())
        thought = next(row for row in self.events if row["phase"] == "thought")
        for kind, field, choice, expected in (
            ("action", "action_id", "A1", {"tool": "detect_fov_initial"}),
            ("instance_goal", "goal_id", "G1", {"entry": "A3", "goal": "measure_length"}),
            ("observe_questions", "question_ids", ["Q1"], {"entry": "A2", "candidates": ["白芍"]}),
            ("problem_goal", "goal_id", "P1", {"entry": "reuse", "dimension": "feret_max"}),
        ):
            with self.subTest(kind=kind):
                thought["execution_choice"] = {"kind": kind, "requested": {field: choice}}
                self.save_snapshots()
                plan = next(span for span in self.snapshot() if span.key == "plan-1")
                self.assertEqual(plan.inputs["required_criteria_at_request"], {"白芍": {"cluster_crystal": "unchecked"}})
                self.assertEqual(plan.inputs["remaining_budget"], {"planning": 7, "observation": 4})
                self.assertEqual(plan.inputs["comparison_questions"], [{
                    "feature": "cluster_crystal", "purpose": "measure", "dimension": "feret_max",
                    "criteria": [{"candidate": "白芍", "role": "required"}], "discovery_actions": ["A1"],
                }])
                options = {row["id"]: row for row in plan.inputs["available_choices"]}
                self.assertEqual(options["O1"]["view"], "view_01")
                self.assertEqual(options["O1"]["feature"], "cluster_crystal")
                self.assertIn("A4", options)
                selected = plan.outputs["requested_choices"][0]
                self.assertEqual(selected["view"], "view_01")
                self.assertEqual({key: selected[key] for key in expected}, expected)
                self.assertEqual(plan.outputs["decision_kind"], kind)
                self.assertNotIn(self.secret, repr(plan))
        self.events[1]["actual_llm_short_ids"] = False
        self.save_snapshots()
        plan = next(span for span in self.snapshot() if span.key == "plan-1")
        self.assertEqual(plan.inputs, {"planning_request": 1})
        self.assertNotIn("requested_choices", plan.outputs)

    def test_parallel_observations_show_accepted_and_unknown_receipts(self):
        self.events.extend([
            {"phase": "thought", "action_id": "observe-1", "tool": "presence_check"},
            {"phase": "parallel_join", "actions": ["observe-1", "observe-2"]},
            {"phase": "execution_timing", "action_id": "observe-1", "parallel": True,
             "started_at": self.start + 14, "ended_at": self.start + 16},
        ])
        rows = []
        for index, (accepted, present) in enumerate(((True, False), (False, False)), 1):
            rows.append({"action": {"action_id": f"observe-{index}", "tool": "presence_check",
                "args": {"image_id": "img_" + str(index) * 64, "feature_id": "cluster_crystal"}},
                "result": {"ok": True, "present": present, "conf": .95, "reason": self.secret},
                "accepted": accepted})
        self.write(self.run / "agent_trace.json", {"events": self.events, "observations": rows})
        group = next(span for span in self.snapshot() if span.name == "并行观察")
        self.assertEqual(group.inputs["planning_request"], 2)
        observations = group.outputs["view_observations"]
        self.assertEqual([row["view"] for row in observations], ["view_01", "view_02"])
        self.assertEqual([row["state"] for row in observations], ["not_seen", "unknown"])
        self.assertIn("本视野未见", group.outputs["summary"])
        self.assertNotIn(self.secret, repr(group.outputs))

    def test_terminal_details_show_scores_qualified_measurement_and_scoped_usage(self):
        measurement = {"candidate_id": "白芍", "feature_id": "cluster_crystal", "role": "required",
                       "dimension_eligible": True, "state": "IN_RANGE", "quant_note": self.secret,
                       "stats": {"median": 20.35, "n_measured": 1},
                       "quant_bounds": {"dim": "feret_max", "min": 11, "max": 35}}
        self.write(self.run / "final_assessment.json", {
            "formal_verdict": "confirmed", "review_reasons": ["CANDIDATE_AUDIT_INCOMPLETE"],
            "candidate_comparison": [{"candidate_id": "白芍", "rule_score": .8, "fused_score": .79}],
            "candidate_audit": {"candidates": [{"candidate_id": "白芍", "review_complete": False}],
                                "unmapped": [{"label": self.secret}]},
            "supporting_evidence": [{"kind": "criterion", "data": measurement}, {"kind": "criterion",
                "data": {**measurement, "dimension_eligible": False, "stats": {"median": 99}}}],
            "usage": {"prompt_tokens": 48, "completion_tokens": 5, "total_tokens": 53},
        })
        self.write(self.run / "output.json", {"top_candidate": {"id": "白芍"}, "review_required": True,
            "status": "tentative", "rule_score_version": "b-rule/4", "fusion_ranking": [{"id": "白芍", "score": .79}]})
        self.write(self.run / "job.json", {"job_id": self.job_id, "status": "completed", "elapsed_seconds": 25})
        a_rows = [{"label": "白芍", "probability": .91}, {"label": self.secret, "probability": .05}]
        self.write(self.run / "a_prediction.json", {"view_predictions": [{"ranking": a_rows, "fold_rankings": [a_rows]}]})
        root, plan_a, *_ = self.snapshot()
        self.assertIn("暂定 白芍", root.outputs["summary"])
        self.assertEqual(root.outputs["formal_b_verdict"], "confirmed")
        self.assertEqual(root.outputs["candidate_review"], {"mapped": 1, "complete": 0, "unmapped": 1})
        report, = root.outputs["supporting_measurements"]
        self.assertEqual((report["median"], report["n_measured"], report["reference_max"]), (20.35, 1, 35))
        self.assertEqual(root.outputs["run_metrics"]["planning_returned_tokens"]["total_tokens"], 38)
        self.assertEqual(root.outputs["run_metrics"]["final_assessment_returned_tokens"]["total_tokens"], 53)
        self.assertEqual(plan_a.outputs["views"][0]["fold_top1"][0]["label"], "白芍")
        self.assertNotIn(self.secret, repr((root.outputs, plan_a.outputs)))

    def test_model_decision_summary_is_opt_in(self):
        thought = next(row for row in self.events if row["phase"] == "thought")
        reason = "白芍/G1 的簇晶尚未核验，先做视野初检。"
        thought.update(decision_source="model", decision_summary=reason)
        self.save_snapshots()
        for flags, included in (([], False), (["--include-decision-summary"], True)):
            with (patch.object(cli, "_publisher", side_effect=AssertionError("unexpected network")),
                  redirect_stdout(StringIO()) as stream):
                self.assertEqual(cli.main([str(self.case), *flags]), 0)
            self.assertEqual(reason in stream.getvalue(), included)
        for value in ("D:/private/sample.png", "Authorization: credential", "img_" + "1" * 64):
            thought["decision_summary"] = value
            self.save_snapshots()
            plan = next(span for span in self.snapshot(include_decision_summary=True) if span.key == "plan-1")
            self.assertNotIn("model_decision_summary", plan.outputs)

    def test_whole_public_summary_is_opt_in_and_sensitive_tail_is_checked(self):
        thought = next(row for row in self.events if row['phase'] == 'thought')
        reason = '现有视野的簇晶身份已确认。' * 24 + '下一轮可检查可用的尺寸。'
        self.assertGreater(len(reason), 240)
        thought.update(decision_source='model', decision_summary=reason, decision_summary_capture='complete')
        thought['usage']['completion_tokens_details'] = {'reasoning_tokens': 3}
        self.save_snapshots()

        def output(enabled):
            return next(s.outputs for s in self.snapshot(include_decision_summary=enabled) if s.key == 'plan-1')

        self.assertNotIn('model_decision_summary', output(False))
        expanded = output(True)
        self.assertEqual((expanded['model_decision_summary'], expanded['model_summary_capture']), (reason, 'complete'))
        self.assertEqual(expanded['usage_metadata']['output_token_details'], {'reasoning': 3})
        self.assertEqual(expanded['usage_metadata']['total_tokens'], 38)
        thought['decision_summary'] = reason + ' D:/private/sensitive.jpg'
        self.save_snapshots()
        filtered = output(True)
        self.assertNotIn('model_decision_summary', filtered)
        self.assertEqual(filtered['model_summary_visibility'], 'withheld_by_text_filter')
        thought['decision_summary'] = reason[:240]
        self.save_snapshots()
        self.assertEqual(output(True)['model_summary_capture'], 'complete')
        del thought['decision_summary_capture']
        self.save_snapshots()
        self.assertEqual(output(True)['model_summary_capture'], 'legacy_240_char_boundary')

    def test_feature_scan_reports_each_dispatched_brief_and_checks_corresponding_receipt(self):
        key = 'c' * 64
        brief = '沿清晰的部分观察颗粒轮廓。' * 20 + '右侧边缘与周边纹理如何相接？'
        self.assertGreater(len(brief), 240)
        preset = '观察外缘和几何轮廓。'
        self.rows[key] = self.rows.pop('1')
        self.rows[key].update(tool='feature_scan', feature_id='cluster_crystal', visual_brief=brief)
        thought = next(r for r in self.events if r['phase'] == 'thought')
        thought.update(tool='feature_scan', visual_briefs={'action-1':brief},
                       visual_guidance_choices={'A1':{'mode':'agent_brief','action_id':'action-1'}})
        offer = self.planner_offer()
        offer['menu']['actions'][0]['tool'] = 'feature_scan'
        offer['menu']['action_state']['A1'] = {'feature_id':'cluster_crystal','candidate_regions_px':[[0,0,100,100]]}
        offer['menu']['scan_feature_hints'] = {'cluster_crystal':preset}
        self.events.insert(1, offer)
        self.save_snapshots()
        folder = self.run/'workflow'/'ops'/key[:16]/'scan'
        for index, extra in enumerate(({'visual_brief':brief}, {}, {'visual_brief':'来源未确认的文字'})):
            tile = folder / f'tile_{index:02d}'
            roi = [index*100, 0, (index+1)*100, 100]
            self.write(tile/'locate.json', {'tool':'locate_feature', 'feature_id':'cluster_crystal',
                                            'roi':roi, 'feature_hint':preset, **extra})
            if index < 2:
                self.write(tile/'detection'/'region_result.json', {'feature_id':'cluster_crystal', 'count':0,
                           'roi':roi if index == 0 else [0,0,1,1]})
        (folder/'tile_02'/'detection').mkdir()
        (folder/'tile_02'/'detection'/'region_result.json').write_text('{', encoding='utf-8')
        for expanded in (False, True):
            spans = self.snapshot(include_visual_observations=expanded)
            plan = next(s for s in spans if s.key == 'plan-1')
            first = next(a for a in plan.inputs['available_choices'] if a['id'] == 'A1')
            self.assertEqual(first['candidate_regions_px'], [[0,0,100,100]])
            self.assertEqual(plan.inputs.get('scan_feature_hints'), {'cluster_crystal':preset} if expanded else None)
            choice = plan.outputs['visual_guidance_choices']['A1']
            self.assertEqual(choice['mode'], 'agent_brief')
            self.assertEqual(choice.get('question'), brief if expanded else None)
            scan = next(s.outputs['scan_requests'] for s in spans if 'scan_requests' in s.outputs)
            tiles = scan['tiles']
            self.assertEqual([t['guidance_source'] for t in tiles],
                             ['project_preset_and_agent','project_preset','additional_brief_unattributed'])
            self.assertEqual([t['status'] for t in tiles], ['worker_returned','prepared','prepared'])
            self.assertEqual(tiles[0]['crop_count'], 0)
            self.assertNotIn('crop_count', tiles[1])
            self.assertEqual(tiles[0].get('visual_brief'), brief if expanded else None)
            self.assertEqual(tiles[1].get('feature_hint'), preset if expanded else None)

    def test_trace_distinguishes_model_proposal_from_receipt_based_stop(self):
        from TCM_Fusion.lib.offline.langsmith_trace import _plan_choice
        thought = {'decision_source': 'model', 'decision_summary_source': 'candidate_audit',
                   'proposed_decision_summary': '所有候选均已核验', 'decision_summary': '未查（香附/secretory_tissue）'}
        self.assertEqual(_plan_choice({'thought': thought}, {}, include_decision_summary=True),
                         {'model_decision_summary': '所有候选均已核验',
                          'required_investigation_summary': '未查（香附/secretory_tissue）'})
        self.assertEqual(_plan_choice({'thought': thought}, {}, include_decision_summary=False), {})

    def test_live_plan_waits_for_outcome_then_uploads_rich_input_once(self):
        thought = self.events.pop(2)
        self.save_snapshots()
        client = CaptureClient()
        publisher = LangSmithPublisher(client, "trace-test", project_name="TCM-Fusion")
        trace_id = publisher.sync(self.job_id, self.snapshot())
        first_count = len(client.created)
        self.assertFalse(any(run["name"] == "规划请求 1" for run in client.patched))
        self.events.insert(1, self.planner_offer())
        self.events.insert(3, thought)
        self.save_snapshots()
        self.assertEqual(publisher.sync(self.job_id, self.snapshot()), trace_id)
        self.assertEqual(publisher.sync(self.job_id, self.snapshot()), trace_id)
        self.assertEqual(len(client.created), first_count)
        plan, = [run for run in client.patched if run["name"] == "规划请求 1"]
        self.assertIn("available_choices", plan["inputs"])
        self.assertEqual(plan["outputs"]["requested_choices"][0]["id"], "A1")

    def test_publish_pending_then_terminal_in_same_trace(self):
        client = CaptureClient()
        publisher = LangSmithPublisher(client, "trace-test", project_name="TCM-Fusion")
        trace_id = publisher.sync(self.job_id, self.snapshot())
        first = len(client.created), len(client.patched)
        self.assertEqual(publisher.sync(self.job_id, self.snapshot()), trace_id)
        self.assertEqual((len(client.created), len(client.patched)), first)
        plan = next(run for run in client.patched if run["name"] == "规划请求 1")
        self.assertEqual(plan["extra"]["metadata"]["usage_metadata"]["total_tokens"], 38)
        sdk_root = client.created[trace_id]
        self.assertIsNone(sdk_root.get("end_time"))
        self.assertEqual(sdk_root["inputs"]["view_count"], 4)

        self.rows["3"].update(status="done", ended_at=self.start + 13)
        self.rows["4"].update(status="failed", started_at=self.start + 12, ended_at=self.start + 14,
                              error_code="WORKFLOW_TIMEOUT")
        self.events.append({"phase": "planning_timing", "request_number": 2,
                            "ended_at": self.start + 15})
        self.save_snapshots()
        self.write(self.run / "job.json", {"job_id": self.job_id, "status": "completed",
                                          "stage": "done", "elapsed_seconds": 17})
        self.write(self.run / "output.json", {"top_candidate": {"id": "白芍"}, "status": "tentative",
                                              "review_required": True})
        terminal = self.snapshot()
        self.assertEqual(publisher.sync(self.job_id, terminal), trace_id)
        self.assertEqual(client.patched[-1]["run_id"], trace_id)
        self.assertEqual(client.patched[-1]["outputs"]["decision"], "tentative")
        group = next(run for run in client.patched if run["name"] == "D4 · 跨视野并发")
        failure = next(run for run in client.patched if run["error"] == "WORKFLOW_TIMEOUT")
        self.assertEqual(failure["parent_run_id"], group["run_id"])
        self.assertEqual(group["outputs"]["failed_branches"], 1)
        self.assertNotIn(self.secret, repr((client.created, client.patched)))

    def test_feedback_can_arrive_after_the_tool_ends(self):
        feedback = self.events.pop(3)
        tool = self.rows.pop("1")
        self.write(self.run / "workflow" / "action_timeline.json", {"actions": {"one": tool}})
        self.save_snapshots()
        client = CaptureClient()
        publisher = LangSmithPublisher(client, "trace-test", project_name="TCM-Fusion")
        publisher.sync(self.job_id, self.snapshot())
        self.assertTrue(any(run["name"] == "view_01 · detect_fov_initial" and not run.get("end_time")
                            for run in client.created.values()))
        self.events.append(feedback)
        self.save_snapshots()
        publisher.sync(self.job_id, self.snapshot())
        self.assertFalse(any(run["name"] == "view_01 · detect_fov_initial" for run in client.patched))
        self.events.extend([
            {"phase": "automatic_measurement_feedback", "feedback": {"accepted": True, "latency_ms": 42}},
            {"phase": "execution_timing", "action_id": "action-1",
             "started_at": self.start + 9, "ended_at": self.start + 12},
        ])
        self.save_snapshots()
        publisher.sync(self.job_id, self.snapshot())
        result, = [run for run in client.patched if run["name"] == "view_01 · detect_fov_initial"]
        self.assertIn("criterion_transitions", result["outputs"])
        self.assertEqual(result["outputs"]["automatic_measurement"]["latency_ms"], 42)

    def test_resumed_follow_replays_if_previous_observer_closed_calls(self):
        client = CaptureClient()
        publisher = LangSmithPublisher(client, "trace-test", project_name="TCM-Fusion")
        trace_id = publisher.sync(self.job_id, self.snapshot())
        resumed = LangSmithPublisher(client, "trace-test", project_name="TCM-Fusion")
        self.assertEqual(resumed.sync(self.job_id, self.snapshot()), trace_id)
        self.write(self.run / "job.json", {"job_id": self.job_id, "status": "completed", "elapsed_seconds": 17})
        terminal = self.snapshot()
        replay_id = resumed.sync(self.job_id, terminal)
        self.assertNotEqual(replay_id, trace_id)
        self.assertEqual(client.created[replay_id]["name"], "TCM_Fusion · 样品鉴别回放")
        self.assertEqual(client.created[replay_id]["outputs"]["status"], "completed")
        self.assertEqual(client.patched[-1]["run_id"], trace_id)

    def test_preview_does_not_create_client(self):
        with (patch.object(cli, "_publisher", side_effect=AssertionError("unexpected network")),
              redirect_stdout(StringIO()) as stream):
            status = cli.main([str(self.case)])
        self.assertEqual(status, 0)
        self.assertIn("仅本地预览", stream.getvalue())
        self.assertNotIn(self.secret, stream.getvalue())

    def test_follow_updates_one_trace_and_stops_at_job_completion(self):
        client = CaptureClient()
        publisher = LangSmithPublisher(client, "trace-test", project_name="TCM-Fusion")

        def finish_after_first_snapshot(_seconds):
            self.assertTrue(client.created)
            self.write(self.run / "job.json", {"job_id": self.job_id, "status": "failed",
                                              "error_code": "WORKFLOW_DEADLINE"})

        with (patch.object(cli.time, "sleep", side_effect=finish_after_first_snapshot) as wait,
              redirect_stdout(StringIO()) as stream):
            cli.observe(self.case, follow=True, publisher=publisher)
        wait.assert_called_once()
        first, last = stream.getvalue().splitlines()
        self.assertIn("进行中", first)
        self.assertIn("已结束", last)
        self.assertEqual(first.split()[0], last.split()[0])
        self.assertEqual(client.patched[-1]["error"], "WORKFLOW_DEADLINE")

    def test_follow_waits_for_directory_then_tracks_replaced_checkpoints(self):
        future = self.case / "not created yet"
        run = future / "runs" / self.job_id
        client = CaptureClient()
        publisher = LangSmithPublisher(client, "trace-test", project_name="TCM-Fusion")
        running = {"job_id": self.job_id, "status": "running", "stage": "workflow_agent"}
        started = {"phase": "planning_started", "request_number": 1, "started_at": self.start + 1}

        def create_partial_job():
            run.mkdir(parents=True)
            (run / "job.json").write_text("{", encoding="utf-8")

        def publish_running():
            write_json(run / "code_signature_start.json", {"at": "2023-11-14T22:13:20+00:00"})
            write_json(run / "job.json", running)

        def publish_first_plan():
            self.assertEqual(len(client.created), 1)
            write_json(run / "agent_trace.json", {"events": [started]})

        def publish_completion():
            self.assertEqual(len(client.created), 2)
            write_json(run / "agent_trace.json", {"events": [started, self.planner_offer(),
                {"phase": "planning_timing", "request_number": 1, "ended_at": self.start + 3},
                {"phase": "thought", "tool": "detect_fov_initial", "decision_source": "model",
                 "decision_summary": "查看第一视野。", "execution_choice": {"requested": {"action_id": "A1"}}}]})
            write_json(run / "job.json", {**running, "status": "completed", "stage": "done", "elapsed_seconds": 4})

        ticks = iter((create_partial_job, publish_running, publish_first_plan, publish_completion))
        with (patch.object(cli, "_publisher", return_value=publisher), patch.object(client, "close", create=True),
              patch.object(cli.time, "sleep", side_effect=lambda _seconds: next(ticks)()),
              redirect_stdout(StringIO()) as stream):
            self.assertEqual(cli.main([str(future), "--follow", "--upload", "--include-decision-summary"]), 0)
        lines = stream.getvalue().splitlines()
        self.assertIn("等待样品目录", lines[0])
        self.assertIn("暂时不可读", lines[1])
        self.assertEqual(lines[2].split()[0], lines[-1].split()[0])
        self.assertIn("已结束", lines[-1])
        plan, = [entry for entry in client.patched if entry["name"] == "规划请求 1"]
        self.assertEqual(plan["outputs"]["model_decision_summary"], "查看第一视野。")

    def test_follow_stops_when_preparation_finishes_without_a_job(self):
        prepared = self.case / "prepared"
        self.write(prepared / "interface_status.json", {"status": "prepared"})
        with (patch.object(cli.time, "sleep", side_effect=AssertionError("unexpected wait")),
              redirect_stderr(StringIO()) as error):
            self.assertEqual(cli.main([str(prepared), "--follow"]), 2)
        self.assertIn("样品准备已结束", error.getvalue())

    def test_upload_destination_requires_key(self):
        with (patch.dict(os.environ, {"LANGSMITH_API_KEY": ""}, clear=False),
              patch.object(cli, "optional_json", return_value={}),
              patch("langsmith.Client", side_effect=AssertionError("unexpected client"))):
            with self.assertRaisesRegex(RuntimeError, "config/langsmith.local.json"):
                cli._publisher()

    def test_upload_uses_local_api_key_with_explicit_environment_override(self):
        self.write(self.case / "config" / "langsmith.local.json", {"api_key": "LOCAL_FILE_TEST_ONLY"})
        with (patch.object(cli.paths, "PROJECT_ROOT", str(self.case)),
              patch.dict(os.environ, {}, clear=True), patch("langsmith.Client") as client):
            cli._publisher()
            self.assertEqual(client.call_args.kwargs["api_key"], "LOCAL_FILE_TEST_ONLY")
            os.environ["LANGSMITH_API_KEY"] = "LOCAL_ENV_TEST_ONLY"
            cli._publisher()
            self.assertEqual(client.call_args.kwargs["api_key"], "LOCAL_ENV_TEST_ONLY")

    def test_local_agent_defaults_and_explicit_project_override(self):
        local = {"endpoint": "https://api.smith.langchain.com", "agent_id": "tcm-fussion", "environment": "local"}
        with (patch.dict(os.environ, {"LANGSMITH_API_KEY": "LOCAL_TEST_ONLY"}, clear=True),
              patch.object(cli, "optional_json", return_value=local), patch("langsmith.Client") as client):
            publisher = cli._publisher()
            self.assertEqual(publisher.address["id"], "tcm-fussion")
            self.assertEqual(publisher.address["environment"].lower(), "local")
            self.assertEqual(client.call_args.kwargs["api_url"], local["endpoint"])
            self.assertIsNone(publisher.project_name)
            os.environ["LANGSMITH_PROJECT"] = "selected-project"
            publisher = cli._publisher()
            self.assertEqual(publisher.project_name, "selected-project")
            self.assertIsNone(publisher.address)

    def test_locked_sdk_serializes_both_web_destinations(self):
        from langsmith import Client, agent_address

        self.events.insert(1, self.planner_offer())
        self.save_snapshots()
        calls = []

        def capture_http(method, url, *args, **kwargs):
            self.assertTrue(url.startswith("https://trace-test.invalid/runs"))
            body = kwargs.get("json") or json.loads(kwargs.get("data") or "{}")
            calls.append((method, body))
            response = requests.Response()
            response.status_code = 200
            response._content = b"{}"
            response.url = url
            return response

        with patch.object(requests.Session, "request", side_effect=capture_http):
            for project, address in (("TCM-Fusion", None), (None, agent_address("tcm-fusion", "local"))):
                calls.clear()
                client = Client(api_url="https://trace-test.invalid", api_key="LOCAL_TEST_ONLY",
                                info={}, auto_batch_tracing=False, omit_traced_runtime_info=True,
                                tracing_sampling_rate=1)
                try:
                    publisher = LangSmithPublisher(client, "wire-test", project_name=project, address=address)
                    publisher.sync(self.job_id, self.snapshot())
                finally:
                    client.close()
                start = calls[0][1]
                self.assertEqual(calls[0][0], "POST")
                if address:
                    self.assertEqual(start["address"], "lrn:agents/tcm-fusion/environments/local")
                else:
                    self.assertEqual(start["session_name"], "TCM-Fusion")
                finished_call = next(body for method, body in calls if method == "PATCH" and body["run_type"] == "llm")
                self.assertEqual(finished_call["extra"]["metadata"]["usage_metadata"]["total_tokens"], 38)
                self.assertEqual(finished_call["inputs"]["available_choices"][0]["id"], "A1")
                self.assertNotIn(self.secret, repr(calls))

    def test_archived_replay_reuses_its_id_without_patching_completed_runs(self):
        from langsmith import Client, agent_address

        self.events.insert(1, self.planner_offer())
        thought = next(row for row in self.events if row["phase"] == "thought")
        thought.update(decision_source="model", decision_summary="先定位簇晶实例。")
        self.save_snapshots()
        self.write(self.run / "job.json", {"job_id": self.job_id, "status": "completed", "elapsed_seconds": 17})
        spans = self.snapshot(include_decision_summary=True)

        destination = "wire-test"
        original_id = str(uuid5(NAMESPACE_URL, f"TCM_Fusion:{destination}:{self.job_id}"))
        received = {original_id: {"outputs": {"status": "completed"}}}
        calls = []
        now = datetime.now(timezone.utc)

        def capture_http(method, url, *args, **kwargs):
            self.assertTrue(url.startswith("https://trace-test.invalid/runs"))
            body = kwargs.get("json") or json.loads(kwargs.get("data") or "{}")
            response = requests.Response()
            response.url = url
            self.assertLess(abs(datetime.fromisoformat(body["start_time"]) - now), timedelta(hours=24))
            if method == "POST" and body["id"] not in received:
                received[body["id"]] = body
                response.status_code, response._content = 202, b"{}"
            else:
                response.status_code = 409
                response._content = b'{"error":"Conflict: payload already received"}'
            calls.append((method, response.status_code))
            return response

        address = agent_address("tcm-fusion", "local")
        with patch.object(requests.Session, "request", side_effect=capture_http):
            client = Client(api_url="https://trace-test.invalid", api_key="LOCAL_TEST_ONLY", info={},
                            auto_batch_tracing=False, omit_traced_runtime_info=True, tracing_sampling_rate=1)
            try:
                publisher = LangSmithPublisher(client, destination, address=address)
                replay_id = str(publisher.sync(self.job_id, spans))
                self.assertNotEqual(replay_id, original_id)
                self.assertEqual(calls, [("POST", 202)] * len(spans))
                repeated = LangSmithPublisher(client, destination, address=address)
                self.assertEqual(str(repeated.sync(self.job_id, spans)), replay_id)
                self.assertEqual(calls[len(spans):], [("POST", 409)] * len(spans))
            finally:
                client.close()
        self.assertEqual(len(received), len(spans) + 1)
        root = received[replay_id]
        self.assertEqual(root["name"], "TCM_Fusion · 样品鉴别回放")
        self.assertEqual(root["address"], "lrn:agents/tcm-fusion/environments/local")
        self.assertEqual(root["inputs"]["fusion_run_id"], self.job_id)
        plan = next(run for run in received.values() if run.get("name") == "规划请求 1")
        self.assertEqual(plan["parent_run_id"], replay_id)
        self.assertEqual(plan["inputs"]["available_choices"][0]["id"], "A1")
        self.assertEqual(plan["outputs"]["model_decision_summary"], "先定位簇晶实例。")
        self.assertEqual(plan["extra"]["metadata"]["usage_metadata"]["total_tokens"], 38)
        self.assertIn("end_time", plan)
        offset = datetime.fromisoformat(root["start_time"]) - datetime.fromtimestamp(spans[0].start, timezone.utc)
        self.assertGreater(offset, timedelta(hours=24))
        for span in spans:
            row = root if span.parent is None else received[str(uuid5(UUID(replay_id), span.key))]
            meta = row["extra"]["metadata"]
            original_start = datetime.fromtimestamp(span.start, timezone.utc)
            self.assertTrue(meta["replay_time_shifted"])
            self.assertEqual(datetime.fromisoformat(meta["original_start_time"]), original_start)
            self.assertEqual(datetime.fromisoformat(row["start_time"]), original_start + offset)
            if span.end is not None:
                original_end = datetime.fromtimestamp(span.end, timezone.utc)
                self.assertEqual(datetime.fromisoformat(meta["original_end_time"]), original_end)
                self.assertEqual(datetime.fromisoformat(row["end_time"]), original_end + offset)
        self.assertNotIn(self.secret, repr(received))

    def test_recent_replay_keeps_original_times(self):
        client = CaptureClient()
        start = datetime.now(timezone.utc).replace(microsecond=0) - timedelta(minutes=4)
        end = start + timedelta(seconds=30)
        span = Span("case", "样品", start.timestamp(), end.timestamp(), parent=None)
        publisher = LangSmithPublisher(client, "trace-test", project_name="TCM-Fusion")
        trace_id = publisher.sync(self.job_id, [span])
        row = client.created[trace_id]
        self.assertEqual(row["start_time"], start)
        self.assertEqual(row["end_time"], end)
        self.assertNotIn("replay_time_shifted", row.get("extra", {}).get("metadata", {}))
