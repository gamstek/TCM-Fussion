"""构造 Agent 领域运行时：从当前测量和合格观察建立规则卡池，生成融合专用 B 裁决，只读原图与输入工件。
加载 skills/registry.json 与 SKILL.md 的领域策略卡；工具权限来自代码白名单。由「lib/mcp/mcp_bridge.py」、「lib/offline/prompt_optimization.py」、「lib/offline/reflection.py」调用，结果按调用方的任务阶段消费。

主要定义：
- Skills: 加载 skills/registry.json 与 SKILL.md 的领域策略卡；工具权限来自代码白名单。
- EvidenceRuntime: 持有 records/images/知识库/工具表，提供 score、action、legal_actions、candidate_interest、read_revision、validate_action、validate_observation、summary。
- from_b_artifacts: 从原 B 的 run_dir/measurements 还原运行时，拒绝缺完整轨迹的旧 VLM 补查与样品不绑定的图像。

关联：
- 调用方 「lib/mcp/mcp_bridge.py」、「lib/offline/prompt_optimization.py」、「lib/offline/reflection.py」、「lib/offline/truth_prompt_validation.py」、「lib/workflow/workflow_runtime.py」、「tests/application/test_lifecycle.py」。
- 直接依赖 「lib/decision/candidate_policy.py」、「lib/decision/comparison_problems.py」、「lib/runtime/probe_budget.py」、「lib/runtime/storage.py」、「lib/tools/tool_registry.py」。
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import paths
import scene_attributes
from rule_engine import EvidenceCard, KnowledgeBase, pool_evidence
from TCM_Fusion.lib.decision.rule_scoring import decide

from TCM_Fusion.lib.tools.tool_registry import ToolRegistry
from TCM_Fusion.lib.decision.candidate_policy import candidates as a_candidates, criteria_diff, validate_candidates
from TCM_Fusion.lib.runtime.storage import (
    IntegrationError, digest, file_hash, finite_number, project_path, read_json, reuse_read,
)


class Skills:
    """技能是可加载的领域策略卡；工具权限来自代码白名单。"""
    def __init__(self):
        self.root = Path(paths.PROJECT_ROOT)/"skills"
        self.registry = read_json(self.root/"registry.json")

    def list(self):
        return [{**row, "sha256": file_hash(self.root/row["id"]/"SKILL.md")}
                for row in self.registry]

    def get(self, identifier):
        if identifier not in {s["id"] for s in self.registry}:
            raise IntegrationError("SKILL_UNKNOWN", "未注册技能")
        return (self.root/identifier/"SKILL.md").read_text(encoding="utf-8")


class EvidenceRuntime:
    def __init__(self, records, images, *, sample_id, baseline=None, a=None, aliases=None,
                 kb_path=None, config_identity="default", simulation=False, candidate_policy=None):
        self.candidate_policy = validate_candidates(candidate_policy)
        self.artifact_roots = []
        self.sample_id = sample_id
        self.records = copy.deepcopy(records)
        for rec in self.records:
            rec.pop("herb", None)  # 真值绝不进入推理或planner
        self.kb_path = Path(kb_path or Path(paths.JUDGE_DIR)/"data/herb_criteria.json")
        self.kb = KnowledgeBase(self.kb_path)
        self.kb_sha256 = file_hash(self.kb_path)
        self.skills = Skills()
        self.tools = ToolRegistry()
        self.a = copy.deepcopy(a or {})
        self.aliases = aliases or {}
        self.config_identity = config_identity
        self.simulation = simulation
        self.images = {}
        stems = {}
        for value in images:
            p = project_path(value)
            sha = file_hash(p)
            if p.stem in stems and stems[p.stem] != sha:
                raise IntegrationError("IMAGE_STEM_AMBIGUOUS", "同stem不同图，拒绝静默绑定")
            stems[p.stem] = sha
            identifier = "img_"+sha
            item = self.images.setdefault(identifier, {"path": p, "sha256": sha, "stems": []})
            if p.stem not in item["stems"]:
                item["stems"].append(p.stem)
        self.original = copy.deepcopy(baseline or self.score([])["decision"])
        self.original_sha256 = digest(self.original)

    def score(self, observations):
        if file_hash(self.kb_path) != self.kb_sha256:
            raise IntegrationError("KB_CHANGED", "运行期间知识库改变")
        workflow = getattr(self, 'workflow', None)
        threshold = workflow.config.get('min_support_features') if workflow is not None else None
        return reuse_read(
            (self, 'rules'),
            lambda: (self.records, observations, threshold),
            lambda: self._score_rules(observations, threshold),
            mutable=True,
        )

    def _score_rules(self, observations, threshold):
        from TCM_Fusion.lib.evidence.evidence_projection import presence_state

        cards = pool_evidence(self.records)
        instance_features = {fid for fid, card in cards.items() if card.present}
        scene = scene_attributes.build_scene(self.records)
        answers = []
        injected = set()
        presence_questions = {}
        observation_states = []
        for observation in observations:
            action = observation["action"]
            if action["tool"] == "scene_check" and observation.get("accepted"):
                for stem in self.images[action["args"]["image_id"]]["stems"]:
                    answers.append({"fov": stem, **observation["result"]})
            if action["tool"] == "presence_check":
                fid = action["args"]["feature_id"]
                image = action["args"]["image_id"]
                state = presence_state(observation, self.kb, self.images)
                observation_states.append({'image_id': image, 'feature_id': fid,
                    'state': state, 'scope': 'observed_fov_only'})
                if state != 'present':
                    continue
                if fid not in instance_features:
                    hint = str(action['args'].get('morph_hint') or '').strip()
                    presence_questions.setdefault(fid, set()).add(hint)
                key = (fid, image)
                if key in injected:
                    continue
                injected.add(key)
                card = cards.setdefault(fid, EvidenceCard(fid))
                card.uncal_refs.append(("agent:"+key[1], 0))
                card.confs.append(observation["result"]["conf"])
                card.conf_assumed = True  # 不创建dims/px_per_um/尺寸
        if answers:
            scene = scene_attributes.merge_vlm_scene(scene, answers)
        decision = decide(self.kb, cards, scene=scene, min_support_features=threshold,
                          presence_questions=presence_questions)
        fingerprint = {"scene": scene, "cards": {k: vars(v) for k, v in cards.items()}}
        if presence_questions:
            fingerprint['presence_questions'] = {fid: sorted(hints) for fid, hints in presence_questions.items()}
        return {"cards": cards, "scene": scene, "decision": decision,
                'observation_states':observation_states,
                "evidence_digest": digest(fingerprint)}

    def action(self, tool, args, skill, *, mandatory=False, priority=()):
        body = {"tool": tool, "args": args, "skill": skill,
                "config_identity": self.config_identity, "kb_sha256": self.kb_sha256}
        return {**body, "action_id": digest(body), "mandatory": mandatory, "priority": list(priority)}

    def legal_actions(self, state):
        return self.tools.propose(self,state)

    def candidate_interest(self):
        """Investigation scope only; never rule evidence or confirmation."""
        return a_candidates(self.a, self.kb, self.aliases, self.candidate_policy)

    def read_revision(self, state):
        """Legacy evidence-only reads; workflow adds physical and control revisions."""
        return state['evidence_digest']

    def _legacy_proposals(self, state):
        scored = self.score(state.get("observations", []))
        cards, decision = scored["cards"], scored["decision"]
        attempted = set(state.get("attempted", []))
        actions = []
        crystal = any(fid in scene_attributes.CRYSTAL_FEATURES and c.present for fid,c in cards.items())
        if crystal:
            for identifier in sorted(self.images):
                action = self.action("scene_check", {"image_id": identifier}, "scene_context", mandatory=True)
                if action["action_id"] not in attempted:
                    actions.append(action)
            workflow = getattr(self, "workflow", None)
            if actions and not (workflow and workflow.config.get("agent_policy") == "adaptive_v2"):
                return actions  # 场景必查先行，不让LLM绕开
        candidates = {c["id"]: c for c in self.kb.candidates()}
        # 原排名含大量零支持候选；它们不能仅凭并列占位触发无关补查。
        selected_ids = [r["id"] for r in decision.get("ranking", [])
                        if r.get("support_count", 0) > 0 and r.get("confidence", 0) > 0][:3]
        # A仅扩大补查兴趣集合；不筛掉B候选、不把类别名传给观察工具。
        interest = self.candidate_interest()
        selected_ids = [cid for cid in selected_ids if cid not in interest.get('inactive_ids', [])]
        for aid in interest["ids"]:
            if aid in candidates and aid not in selected_ids:
                selected_ids.append(aid)
        selected = [candidates[x] for x in selected_ids if x in candidates]
        required_sets = [{c["feature"] for c in row["criteria"] if c["role"] == "required"} for row in selected]
        common = set.intersection(*required_sets) if required_sets else set()
        workflow = getattr(self, 'workflow', None)
        from TCM_Fusion.lib.runtime.probe_budget import enabled as plans_enabled, available as probe_available
        unified_probes = plans_enabled(self)
        wf_config = getattr(workflow, 'config', {}) or {}
        support_total_cap = int(wf_config.get('supporting_probe_total', 0) or 0)
        support_per_candidate = max(1, int(wf_config.get('supporting_probe_per_candidate', 2) or 2))
        support_total = 0
        support_used = {}
        support_seen = set()
        if support_total_cap and not unified_probes:
            for obs in state.get('observations', []):
                action = obs.get('action') or {}
                if action.get('tool') != 'presence_check':
                    continue
                fid = (action.get('args') or {}).get('feature_id')
                cids = {row['id'] for row in selected
                        for c in row['criteria'] if c.get('feature') == fid and c.get('role') == 'supporting'}
                if not cids:
                    continue
                support_total += 1
                support_seen.add(fid)
                for cid in cids:
                    support_used[cid] = support_used.get(cid, 0) + 1
            if workflow is not None:
                for row in read_json(workflow.operations).get('entries', {}).values():
                    if row.get('tool') != 'feature_scan':
                        continue
                    cids = {c.get('candidate_id') for c in (row.get('criteria') or [])
                            if c.get('role') == 'supporting'}
                    if not cids:
                        continue
                    support_total += 1
                    support_seen.add(row.get('feature_id'))
                    for cid in cids:
                        support_used[cid] = support_used.get(cid, 0) + 1
        for candidate_rank, candidate in enumerate(selected):
            for criterion in candidate["criteria"]:
                fid = criterion["feature"]
                role = criterion.get("role")
                if role not in {"required", "supporting"}:
                    continue
                if (self.kb.tier(fid) not in {"P1", "C"}
                        or (fid in cards and cards[fid].present)):
                    continue
                if role == 'supporting' and support_total_cap and not unified_probes:
                    if fid in support_seen or support_total >= support_total_cap:
                        continue
                    if support_used.get(candidate["id"], 0) >= support_per_candidate:
                        continue
                for identifier in sorted(self.images):
                    if role == 'supporting' and fid in support_seen and not unified_probes:
                        break
                    args = {"image_id": identifier, "feature_id": fid,
                            "morph_hint": self._presence_hint(criterion)}
                    action = self.action("presence_check", args, "differential_presence",
                                         priority=(role == "required", fid in common, candidate_rank,
                                                   fid, identifier, args["morph_hint"]))
                    if action["action_id"] not in attempted:
                        if unified_probes and not probe_available(self, action):
                            continue
                        if role == 'supporting' and support_total_cap and not unified_probes:
                            if support_total >= support_total_cap or support_used.get(candidate["id"], 0) >= support_per_candidate:
                                continue
                            support_total += 1
                            support_seen.add(fid)
                            support_used[candidate["id"]] = support_used.get(candidate["id"], 0) + 1
                        actions.append(action)
        unique = {a["action_id"]: a for a in actions}
        ranked = sorted(unique.values(), key=lambda a: a["priority"])
        return ranked

    def validate_action(self, action):
        """服务器也独立验权限：只能 registered image、合法required P1/C与本体形态提示。"""
        tool, args = action.get("tool"), action.get("args", {})
        spec = self.tools.validate(action, self)
        if spec.layer == "read":
            raise IntegrationError("ACTION_DENIED", "观察入口禁止只读动作")
        return self._validate_action_semantics(action)

    def _presence_hint(self, criterion):
        workflow = getattr(self, 'workflow', None)
        if workflow and workflow.config.get('agent_policy') == 'adaptive_v2':
            from TCM_Fusion.lib.decision.comparison_problems import canonical_presence_hint
            return canonical_presence_hint(self, criterion['feature'])
        return criterion.get('morph') or ''

    def _validate_action_semantics(self, action):
        tool, args = action["tool"], action["args"]
        if args.get("image_id") not in self.images:
            raise IntegrationError("ACTION_DENIED", "观察入口禁止只读动作或未知图像")
        image = self.images[args["image_id"]]
        if file_hash(image["path"]) != image["sha256"]:
            raise IntegrationError("INPUT_CHANGED", "观察前原图内容已变更")
        if tool == "presence_check":
            fid = args.get("feature_id")
            allowed = any(c["feature"] == fid and c["role"] in {"required", "supporting"}
                          and self._presence_hint(c) == args.get("morph_hint", "")
                          for row in self.kb.candidates() for c in row["criteria"])
            if self.kb.tier(fid) not in {"P1", "C"} or not allowed:
                raise IntegrationError("ACTION_DENIED", "禁止P0或非知识库判据检查")
        return image["path"]

    def validate_observation(self, action, result):
        spec = self.tools.get(action.get("tool"))
        return self.tools.validate_result(self, action, result)

    def _validate_observation_v1(self, action, result):
        if not isinstance(result, dict) or result.get("ok") is not True:
            return False
        if not finite_number(result.get("conf"), maximum=1):
            return False
        if action["tool"] == "presence_check":
            return isinstance(result.get("present"), bool) and not any(
                k in result for k in ("size_um", "primary_value_um", "px_per_um", "conclusion"))
        return (isinstance(result.get("cell_wall_visible"), bool)
                and result.get("crystal_location") in {"free", "intracellular"}
                and (result["crystal_location"] != "intracellular" or result["cell_wall_visible"])
                and result.get("arrangement") in {"rows", "scattered", "unknown"}
                and result.get("abundance") in {"few", "many", "very_many", "unknown"})

    def summary(self, state, *, scored=None):
        if scored is None:
            scored = self.score(state.get("observations", []))
        interest = self.candidate_interest()
        d = scored["decision"]
        compact = {"verdict": d["verdict"], "conclusion": d.get("conclusion"),
                   "needs_review": d.get("needs_review", False),
                   "ranking": [{"id": r["id"], "confidence": r["confidence"],
                                "support_count": r["support_count"],
                                "checks": [{k: c.get(k) for k in ("feature_id", "state", "role")}
                                           for c in r["checks"]]} for r in d.get("ranking", [])[:3]]}
        return {"sample_id": self.sample_id, "decision": compact,
                'presence_localizations': [o['presence_localization'] for o in state.get('observations', [])
                                           if 'presence_localization' in o][-8:],
                'observation_states':scored.get('observation_states',[]),'unobserved_state':'unknown',
                "evidence_digest": scored["evidence_digest"],
                "cards": {fid: {"present": c.present, "n_measured": c.n_instances,
                                 "n_seen": c.n_seen} for fid,c in scored["cards"].items()},
                "skills": self.skills.list(),
                "candidate_interest": interest,
                "criteria_diff": criteria_diff(self.kb, interest["ids"], scored["cards"])}


def from_b_artifacts(b, agent_config, *, sample_id, a=None, aliases=None):
    """只接受可还原的测量起点；已做旧VLM补查但缺完整历史的run不静默重放。"""
    from evidence_pool import load_measurements
    root = project_path(b["run_dir"])
    manifest = read_json(root/"run_manifest.json")
    if (manifest.get("config") or {}).get("sample") != sample_id:
        raise IntegrationError("AGENT_SAMPLE_UNBOUND", "主动模式需要manifest.config.sample精确绑定样品")
    if (b["verdict"].get("agent") or {}).get("vlm_calls", 0):
        raise IntegrationError("AGENT_HISTORY_MISSING", "旧VLM补查无完整轨迹；主动模式请以enable_vlm=false生成独立B基线")
    records = load_measurements([root/"measurements"])
    images = []
    for value in agent_config.get("image_roots", []):
        p = project_path(value)
        images.extend([p] if p.is_file() else sorted(
            x for x in p.rglob("*") if x.is_file() and x.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}))
    if not records or not images:
        raise IntegrationError("AGENT_INPUT_MISSING", "主动Agent需要测量记录与同样品原图")
    stems = {p.stem for p in images}
    if any(r.get("image_stem") not in stems for r in records):
        raise IntegrationError("AGENT_FOV_MISSING", "测量视野无法绑定原图，拒绝猜测配对")
    referenced = b["verdict"].get("source_images") or []
    if referenced:
        bound = {file_hash(project_path(p)) for p in referenced}
        if any(file_hash(p) not in bound for p in images):
            raise IntegrationError("AGENT_SAMPLE_UNBOUND", "新增图像不属于原B来源集合")
    runtime = EvidenceRuntime(records, images, sample_id=sample_id, baseline=b["verdict"], a=a, candidate_policy=agent_config.get("candidate_policy"),
                           aliases=aliases, config_identity=digest({"vlm": file_hash(
                               Path(paths.DETECTION_CONFIG_DIR)/"vlm_config.json")}))
    runtime.artifact_roots = [root]
    return runtime
