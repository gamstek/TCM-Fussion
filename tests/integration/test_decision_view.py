"""D2 离线回归：共享查询、区分问题、资格引用及歧义关联，随既有 selftest 发现。

真实 WorkflowRuntime/LocalToolbox 与明确 fixture 结合；不调用视觉 API、不评价药材准确率。
该组用例由项目根 selftest 发现。涉及模型回复的用例使用可控输入，涉及进程的用例以各断言规定的执行边界为准。

主要定义：
- DecisionViewTests。

关联：
- 直接依赖 「lib/agent/react_graph.py」、「lib/decision/comparison_problems.py」、「lib/decision/final_assessment.py」、「lib/evidence/evidence_projection.py」、「lib/evidence/instance_ledger.py」、「lib/runtime/probe_budget.py」、「lib/runtime/storage.py」、「lib/tools/agent_tools.py」。
- 统一执行见 tests/README.md。
"""
import asyncio
import copy
import unittest
from unittest.mock import patch

from TCM_Fusion.lib.tools.agent_tools import FixtureBackend, LocalToolbox
from TCM_Fusion.lib.decision.comparison_problems import build as comparison, presence_key
from TCM_Fusion.lib.evidence.evidence_projection import build, reference_allowed
from TCM_Fusion.lib.decision.final_assessment import build_packet, validate_reply
from TCM_Fusion.lib.evidence.instance_ledger import project
from TCM_Fusion.lib.runtime.probe_budget import describe as describe_probe
from TCM_Fusion.lib.agent.react_graph import run_react, compact_action_menu
from TCM_Fusion.lib.runtime.storage import IntegrationError, read_json, write_json
from TCM_Fusion.tests.support import workflow_fixtures as fixtures


class DecisionViewTests(unittest.TestCase):
    setUp = fixtures.InstanceFixture.setUp
    runtime = fixtures.InstanceFixture.runtime
    detected = fixtures.InstanceFixture.detected
    alias = fixtures.InstanceFixture.alias

    def with_candidates(self, **config):
        r = self.runtime(agent_policy='adaptive_v2', early_verification=True, **config)
        r.a = {'status':'ok','fold_rankings':[[{'label':'白芍','probability':.6},
                                             {'label':'G1','probability':.4}]]*5}
        return r

    def test_shared_feature_query_is_one_action_and_one_backend_call(self):
        r = self.with_candidates(); state={'observations':[],'attempted':[]}
        actions=r.legal_actions(state)
        fiber=[a for a in actions if a['tool']=='presence_check' and a['args']['feature_id']=='fiber']
        self.assertEqual(len(fiber),1)
        backend=FixtureBackend(); tools=LocalToolbox(r,ledger_path=self.root/'obs.json',backend=backend)
        first=asyncio.run(tools.observe(fiber[0]))
        again=asyncio.run(tools.observe(fiber[0]))
        self.assertTrue(again['cache_hit']);self.assertEqual(again['new_requests'],0)
        self.assertEqual(len(backend.calls),1)
        view=r.decision_view({**state,'observations':[first],'actions':actions})
        query=next(q for q in view['comparison']['shared_queries'] if q['feature_id']=='fiber')
        self.assertGreaterEqual(len(query['candidate_ids']),2)
        self.assertIn('candidate_morphology',query['does_not_verify'])
        self.assertFalse(any(g.get('is_exclusion') for q in view['comparison']['questions'] for g in q['consumers']))

    def test_different_morphology_questions_keep_distinct_query_keys(self):
        r=self.with_candidates()
        first=next(a for a in r.legal_actions({'observations':[]}) if a['tool']=='presence_check')
        second={**first,'args':{**first['args'],'morph_hint':'another actual morphology question'}}
        self.assertNotEqual(presence_key(r,first),presence_key(r,second))
        # 工具实体键保留实际问题。当前预算策略按同图同特征合并支持性额度。
        p1,p2=describe_probe(r,first),describe_probe(r,second)
        if p1 is not None:
            self.assertEqual(p1['key'],p2['key'])
        with self.assertRaises(IntegrationError):
            r.validate_action(second)

    def test_shared_dimension_different_intervals_are_not_exclusion(self):
        r=self.with_candidates()
        rows=[{'id':cid,'kind':'herb','criteria':[{'feature':'fiber','role':'required',
            'morph':'same structural question','quant':{'dim':'mean_width','min':lo,'max':hi}}]}
            for cid,lo,hi in [('A',8,14),('B',11,27)]]
        scored={'decision':{'ranking':[]}}
        with patch.object(r,'candidate_interest',return_value={'ids':['A','B'],'a_topk_ids':['A','B']}),\
             patch.object(r.kb,'candidates',return_value=rows):
            view=comparison(r,scored,{'projections':[],'evidence_version':'v'},[])
        self.assertEqual(len(view['questions']),1)
        q=view['questions'][0]
        self.assertTrue(q['overlapping_ranges']);self.assertEqual(len(q['consumers']),2)
        self.assertTrue(all(not c['is_exclusion'] for c in q['consumers']))

    def test_different_morphology_or_dimension_not_merged(self):
        r=self.with_candidates()
        rows=[{'id':cid,'kind':'herb','criteria':[{'feature':'fiber','role':'required',
            'morph':morph,'quant':{'dim':dim,'min':1,'max':5}}]}
            for cid,morph,dim in [('A','thick','width'),('B','thin','width'),('C','thin','length')]]
        with patch.object(r,'candidate_interest',return_value={'ids':['A','B','C']}),\
             patch.object(r.kb,'candidates',return_value=rows):
            view=comparison(r,{'decision':{'ranking':[]}}, {'projections':[],'evidence_version':'v'},[])
        self.assertEqual(len(view['questions']),3)

    def test_kb_indistinguishable_has_explicit_terminal_boundary(self):
        r=self.with_candidates();r.a={'status':'ok','fold_rankings':[[{'label':'陈皮','probability':1.}]]*5}
        view=r.decision_view({'observations':[]})['comparison']
        boundary=next(x for x in view['boundaries'] if x['candidate_id']=='G3')
        self.assertEqual(boundary['capability'],'kb_indistinguishable')
        self.assertFalse(boundary['repeat_will_help']);self.assertEqual(boundary['legal_action_ids'],[])
        self.assertIn('枳壳',boundary['members'])
        self.assertEqual(view['active_candidates'],['G3'])

    def test_five_fold_union_is_not_replaced_by_top_two(self):
        r=self.with_candidates();names=['秦皮','桔梗','龙胆','甘草','苦参']
        r.a={'status':'ok','fold_rankings':[[{'label':name,'probability':1.}] for name in names]}
        view=r.decision_view({'observations':[]})['comparison']
        self.assertEqual(set(view['a_topk_union']),set(names))
        self.assertEqual(set(view['active_candidates']),set(names))

    def observation(self,r,accepted=True):
        image=next(iter(r.images))
        return {'accepted':accepted,'action':{'tool':'presence_check','action_id':'fixture',
            'args':{'image_id':image,'feature_id':'sand_crystal','morph_hint':''}},
            'result':{'ok':accepted,'present':True,'conf':.9}}

    def test_early_and_final_criterion_refs_and_policy_are_identical(self):
        r=self.with_candidates();r.a={'status':'ok','fold_rankings':[[{'label':'秦皮','probability':1.}]]*5}
        observations=[self.observation(r)]; scored=r.score(observations)
        early=build(r,scored,observations)
        packet=build_packet(r,scored['decision'],{'observations':observations})
        # 收尾可能额外列出正式榜单的其他候选；共同事实的身份和资格必须逐字段一致。
        final_facts={f['id']:f for f in packet['facts']}
        for f in early['facts']:
            self.assertEqual(f,final_facts[f['id']])
        support=next(f for f in early['facts'] if reference_allowed(f,'秦皮'))
        reply={'candidate_id':'秦皮','support_refs':[support['id']],
               'counter_refs':[],'unresolved_refs':[],'explanation':'fixture'}
        self.assertEqual(validate_reply(packet,reply),reply)

    def test_unrelated_commit_does_not_renumber_refs_but_view_is_fresh(self):
        r=self.with_candidates();observations=[self.observation(r)]
        before=r.decision_view({'observations':observations})
        marker=self.root/'marker.json';write_json(marker,{'diagnostic':'unrelated'})
        snap=r.workflow.store.current();image=next(iter(r.images))
        r.workflow.store.commit(snap['version'],[r.workflow._art('marker','diagnostic',marker,[image])])
        after=r.decision_view({'observations':observations})
        self.assertNotEqual(before['eligibility']['evidence_version'],after['eligibility']['evidence_version'])
        self.assertEqual([f['id'] for f in before['eligibility']['facts']],
                         [f['id'] for f in after['eligibility']['facts']])

    def test_failed_observation_is_unknown_not_absence_or_counter(self):
        r=self.with_candidates();obs=self.observation(r,False)
        view=build(r,r.score([obs]),[obs])
        f=next(f for f in view['facts'] if f['kind']=='observation')
        self.assertEqual(f['data']['observation_state'],'unknown')
        self.assertFalse(f['supports_candidate_ids']);self.assertFalse(f['counters_candidate_ids'])

    def test_uncalibrated_size_and_a_prior_cannot_support_final_candidate(self):
        r=self.with_candidates();r.a={'status':'ok','fold_rankings':[[{'label':'陈皮','probability':1.}]]*5}
        scored=r.score([]);row=next(c for c in scored['decision']['ranking'] if c['id']=='G3')
        check=row['checks'][0];check.update(present=True,state='UNCAL',score_contribution=.5)
        with patch.object(r,'score',return_value=scored):packet=build_packet(r,scored['decision'],{'observations':[]})
        bad=next(f for f in packet['facts'] if f['kind']=='criterion' and
                 f['data']['candidate_id']=='G3' and f['data']['state']=='UNCAL')
        self.assertFalse(reference_allowed(bad,'G3'))
        prior=next(f for f in packet['facts'] if f['kind']=='a_prior')
        for f in [bad,prior]:
            with self.assertRaises(IntegrationError):
                validate_reply(packet,{'candidate_id':'G3','support_refs':[f['id']],
                    'counter_refs':[],'unresolved_refs':[],'explanation':'invalid support'})
        projection=next(p for p in packet['eligibility_projection'] if p['candidate_id']=='G3')
        self.assertTrue(projection['invalid_dimensions'])

    def test_high_overlap_is_ambiguous_without_merging_or_authorization(self):
        r,crop=self.detected();changed=self.alias(r,crop,box=[0,0,8.5,8.5])
        index=project(r.workflow.store.current(),read_json(r.workflow.operations))
        self.assertEqual(len(index['instances']),2)
        association=index['associations'][0]
        self.assertEqual(association['state'],'ambiguous');self.assertFalse(association['auto_merged'])
        self.assertGreaterEqual(association['iou'],.85)
        self.assertEqual(association['qualification_effect'],'none')

    def test_changed_roi_after_failed_segment_is_not_blocked_by_high_iou(self):
        r,crop=self.detected(fixtures.StageFixture(fail='segment_instance'))
        r.workflow.execute(next(a for a in r.workflow.propose() if a['tool']=='segment_instance'))
        changed=self.alias(r,crop,box=[0,0,8.5,8.5])
        self.assertFalse(r.workflow._segment_reviewed(changed))
        self.assertTrue(any(a['tool']=='segment_instance' and a['args']['target_id']==changed['id']
                            for a in r.workflow.propose()))

    def test_read_tools_are_paginated_and_do_not_change_evidence_or_budget(self):
        r=self.with_candidates();snap=r.workflow.store.current();budget=read_json(r.workflow.budget_path)
        for kind in ['eligibility','eligibility_facts','comparison','instances']:
            value=r.tools.read(r,'read_evidence_detail',{'kind':kind,'offset':0,'limit':1},{'observations':[]})
            self.assertLessEqual(len(value['rows']),1)
        self.assertEqual(snap,r.workflow.store.current());self.assertEqual(budget,read_json(r.workflow.budget_path))

    def test_actual_react_context_contains_fresh_projection(self):
        r=self.with_candidates();seen=[]
        class Planner:
            def choose(self,context):
                seen.append(copy.deepcopy(context))
                detect=next((a for a in context['actions'] if a['tool']=='detect_fov_initial'),None)
                if detect:return {'action_id':detect['action_id']},None
                return {'action_id':'review','override_reason':'offline inspection complete'},None
        tools=LocalToolbox(r,ledger_path=self.root/'react_obs.json',backend=FixtureBackend())
        trace=asyncio.run(run_react(r,tools,planner=Planner(),planner_mode='scripted',max_steps=4))
        self.assertTrue(seen)
        view=seen[-1]['evidence']['workflow']
        self.assertIn('eligibility_projection',view);self.assertIn('comparison_problems',view)
        self.assertEqual(view['eligibility_projection']['evidence_version'],r.workflow.store.current()['version'])
        self.assertEqual(trace['stop_reason'],'planner_request_review')

    def test_related_assessment_version_changes_reference_binding(self):
        r,crop=self.detected();r.a={'status':'ok','fold_rankings':[[{'label':'陈皮','probability':1.}]]*5}
        before=build(r,r.score([]),[])
        assessment=self.root/'assessment_marker.json';write_json(assessment,{'policy_version':'fixture_v2'})
        snap=r.workflow.store.current()
        r.workflow.store.commit(snap['version'],[r.workflow._art('qa_marker','quality_assessment',
                                   assessment,[crop['id']],image_id=crop['image_id'])])
        after=build(r,r.score([]),[])
        def ref(view):return next(f['id'] for f in view['facts'] if f['kind']=='criterion' and
                                 f['data']['candidate_id']=='G3' and f['data']['feature_id']=='single_crystal')
        self.assertNotEqual(ref(before),ref(after))

    def test_shared_query_unknown_failure_is_not_retried_under_other_candidate(self):
        r=self.with_candidates();state={'observations':[],'attempted':[]}
        action=next(a for a in r.legal_actions(state) if a['tool']=='presence_check' and a['args']['feature_id']=='fiber')
        backend=FixtureBackend({'fiber':IntegrationError('FIXTURE_SERVICE_FAILURE','fixture')})
        toolbox=LocalToolbox(r,ledger_path=self.root/'failure_observations.json',backend=backend)
        first=asyncio.run(toolbox.observe(action));self.assertFalse(first['accepted'])
        before=read_json(r.workflow.budget_path)
        second=asyncio.run(toolbox.observe({**action,'consumer_candidate':'another'}))
        self.assertTrue(second['cache_hit']);self.assertEqual(len(backend.calls),1)
        self.assertEqual(before,read_json(r.workflow.budget_path))
        self.assertEqual(first['status'],'failed')

    def test_agent_can_select_detail_read_from_actual_menu(self):
        r=self.with_candidates();reads=[]
        class Planner:
            def choose(self,c):
                detect=next((a for a in c['actions'] if a['tool']=='detect_fov_initial'),None)
                if detect:return {'action_id':detect['action_id']},None
                if not reads:
                    action=next(a for a in c['actions'] if a['tool']=='read_evidence_detail' and
                                a['args']['kind']=='eligibility')
                    reads.append(action['action_id']);return {'action_id':action['action_id']},None
                return {'action_id':'review','override_reason':'read complete'},None
        toolbox=LocalToolbox(r,ledger_path=self.root/'detail_obs.json',backend=FixtureBackend())
        trace=asyncio.run(run_react(r,toolbox,planner=Planner(),planner_mode='scripted',max_steps=5))
        self.assertEqual(len(reads),1)
        self.assertTrue(any(row['tool']=='read_evidence_detail' and row['result']['kind']=='eligibility'
                            for row in trace['read_results']))

    def test_compact_menu_preserves_optional_detail_slot(self):
        actions=[{'tool':'segment_instance','action_id':str(i)} for i in range(30)]
        detail={'tool':'read_evidence_detail','action_id':'detail','args':{'kind':'comparison'}}
        visible,metadata=compact_action_menu(actions+[detail,{'tool':'finish','action_id':'finish'}],20)
        self.assertIn(detail,visible);self.assertEqual(len(visible),21)
        self.assertGreater(metadata['omitted'],0)

    def test_exhausted_observation_budget_is_visible_not_a_new_scan(self):
        r=self.with_candidates(max_observations=0)
        action=next(a for a in r.legal_actions({'observations':[]}) if a['tool']=='presence_check')
        # 纯元数据用例：让判据问题恰好对应当前合法观察，不调用工具。
        rows=[{'id':'fixture','kind':'herb','criteria':[{'feature':action['args']['feature_id'],
               'role':'required','quant':None,'morph':action['args']['morph_hint']}]}]
        with patch.object(r,'candidate_interest',return_value={'ids':['fixture']}),\
             patch.object(r.kb,'candidates',return_value=rows):
            view=comparison(r,{'decision':{'ranking':[]}}, {'projections':[],'evidence_version':'v'},[action])
        self.assertEqual(view['questions'][0]['capability'],'budget_blocked')

    def test_mcp_style_read_without_client_menu_computes_current_actions(self):
        r=self.with_candidates();view=r.decision_view({'observations':[]})
        self.assertTrue(view['comparison']['shared_queries'])
        self.assertEqual(read_json(r.workflow.budget_path)['used'],{})
