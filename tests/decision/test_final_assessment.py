"""收尾研判回归：只读判定、来源区分、共享预算与非法引用回退，不发起真实模型请求。
该组用例由项目根 selftest 发现。涉及模型回复的用例使用可控输入，涉及进程的用例以各断言规定的执行边界为准。

主要定义：
- FinalAssessmentTests（停止码收尾）: 任一 stop_reason 都仍产出只读 judgement，且不改证据版本与判决对象。
- FinalAssessmentTests（上报包）: compact_packet 限制载荷体积同时保留事实 ID、候选支持关系与融合分。
- FinalAssessmentTests（来源区分）: 注入 finalizer 标记为 tentative/simulation，不冒充规则确认。
- FinalAssessmentTests（终止与复核）: termination.reason 与 review_reasons 分离，D8b 单类复核不被服务中断掩盖。
- FinalAssessmentTests（数值防护）: 模型自由文本中的错误融合分被替换为确定性数值摘要并留警告。
- FinalAssessmentTests（引用校验）: 未知候选、伪造引用和失败观察都不能作为正向支持。
- FinalAssessmentTests（预算与时限）: 收尾保留单独请求池；超时或关闭后不再请求模型。
- FinalAssessmentTests（真实范围）: 非 simulation 下拒绝注入 finalizer。

关联：
- 直接依赖 「lib/agent/react_graph.py」、「lib/decision/candidate_audit.py」、「lib/decision/final_assessment.py」、「lib/runtime/storage.py」、「lib/tools/agent_tools.py」、「lib/workflow/workflow_runtime.py」、「tests/support/workflow_fixtures.py」。
- 统一执行见 tests/README.md。
"""
import asyncio
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from TCM_Fusion.lib.runtime.storage import IntegrationError,read_json,write_json
from TCM_Fusion.lib.workflow.workflow_runtime import WorkflowRuntime
from TCM_Fusion.lib.decision.final_assessment import build_packet,compact_packet,finalize,validate_reply,request_model
from TCM_Fusion.lib.tools.agent_tools import LocalToolbox,FixtureBackend
from TCM_Fusion.lib.agent.react_graph import run_react,ScriptedPlanner
from TCM_Fusion.tests.support.workflow_fixtures import StageFixture


class FinalAssessmentTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.image=self.root/'input.png';self.image.write_bytes(b'fixture')
        a={'status':'ok','fold_rankings':[[{'label':'秦皮','probability':.7},
                                          {'label':'桔梗','probability':.3}]]*5}
        self.runtime=WorkflowRuntime([self.image],root=self.root/'workflow',sample_id='s',a=a,
            simulation=True,backend=StageFixture(),workflow_config={'agent_policy':'adaptive_v2'})
        image=next(iter(self.runtime.images))
        # 冻结的旧定向观察已按秦皮判据提问。通用阳性待核验由决策视图用例覆盖。
        candidate = next(c for c in self.runtime.kb.candidates() if c['id'] == '秦皮')
        criterion = next(c for c in candidate['criteria'] if c['feature'] == 'sand_crystal')
        self.observation={'accepted':True,'action':{'tool':'presence_check','action_id':'fixture',
            'args':{'image_id':image,'feature_id':'sand_crystal','morph_hint':criterion['morph']}},
            'result':{'ok':True,'present':True,'conf':.75}}
        self.result={'observations':[self.observation],'stop_reason':'override_exhausted',
                     'steps':2,'events':[],'needs_review':True}
        decision=self.runtime.score(self.result['observations'])['decision']
        self.verdict=copy.deepcopy(decision)
        self.verdict['ranking']=[r for r in decision['ranking'] if r['id'] in {'秦皮','桔梗'}]
        self.verdict['needs_review']=True

    def reply(self,packet):
        ref=next(f['id'] for f in packet['facts'] if f['kind']=='criterion'
                 and '秦皮' in f['supports_candidate_ids'])
        return {'candidate_id':'秦皮','support_refs':[ref],'counter_refs':[],
                'unresolved_refs':[],'explanation':'砂晶存在性支持秦皮，但未形成充分鉴别证据。'},None

    def test_every_stop_reason_still_produces_read_only_judgement(self):
        before=self.runtime.workflow.store.current()['version'];original=copy.deepcopy(self.verdict)
        for reason in ['override_exhausted','max_steps','workflow_deadline','request_budget',
                       'no_legal_action','planner_request_review']:
            result={**self.result,'stop_reason':reason}
            value=finalize(self.runtime,self.verdict,result,self.root/reason)
            self.assertTrue(value['judgement']);self.assertEqual(value['candidate_id'],'秦皮')
            self.assertTrue(value['supporting_evidence']);self.assertTrue(value['candidate_comparison'])
            self.assertEqual(value['new_image_calls'],0);self.assertTrue(value['review_required'])
        self.assertEqual(self.runtime.workflow.store.current()['version'],before)
        self.assertEqual(self.verdict,original)

    def test_compact_packet_bounds_model_payload_and_keeps_refs(self):
        facts=[{'id':f'observation:{i}','kind':'observation',
                'data':{'accepted':True,'action':{'tool':'presence_check',
                         'args':{'image_id':'img','feature_id':'f'}},
                        'result':{'ok':True,'present':True,'reason':'x'*2000}},
                'supports_candidate_ids':['c'],'counters_candidate_ids':[]}
               for i in range(200)]
        packet={'schema':'assessment-evidence/v1','sample_id':'s','evidence_version':'v',
                'formal_verdict':{'ranking':[{'id':'c','checks':[{'feature_id':'f'}]*200}]},
                'candidates':[{'id':'c'}],'facts':facts,'artifact_index':{'x':'y'*10000}}
        model=compact_packet(packet)
        self.assertLess(len(json.dumps(model,ensure_ascii=False)),120000)
        self.assertEqual([f['id'] for f in model['facts']], [f['id'] for f in facts])
        self.assertEqual(model['facts'][0]['supports_candidate_ids'],['c'])
        self.assertNotIn('artifact_index',model)

    def test_injected_model_assessment_is_distinct_from_rule_confirmation(self):
        value=finalize(self.runtime,self.verdict,self.result,self.root/'model',responder=self.reply)
        self.assertEqual(value['method'],'injected_finalizer')
        self.assertEqual(value['status'],'tentative');self.assertTrue(value['simulation'])
        self.assertIsNone(value['confidence_basis']['calibrated_probability'])

    def single_feature_verdict(self):
        verdict=copy.deepcopy(self.verdict)
        verdict.update(verdict='confirmed',conclusion={'id':'秦皮'},needs_review=True,
                       notes=['D8b 单类证据条件放行：独立证据仅 1 类，仍需复核'])
        for row in verdict['ranking']:
            row['fused_score']=.872168 if row['id']=='秦皮' else .240057
        scored=self.runtime.score(self.result['observations'])
        top=next(r for r in scored['decision']['ranking'] if r['id']=='秦皮')
        top.update(rule_score=.8,confidence=.8,support_count=1)
        next(r for r in verdict['ranking'] if r['id']=='秦皮').update(rule_score=.8,confidence=.8)
        return verdict,scored

    def test_service_termination_and_d8b_review_have_separate_causes(self):
        verdict,scored=self.single_feature_verdict();original=copy.deepcopy(verdict)
        result={**self.result,'stop_reason':'runtime_request_review','planner_degraded':True,
                'events':[{'phase':'stop','decision_source':'runtime_fallback',
                           'reason':'UPSTREAM_REMOTE_DISCONNECTED'}]}
        with patch.object(self.runtime,'score',return_value=scored):
            value=finalize(self.runtime,verdict,result,self.root/'d8b',responder=self.reply)
        self.assertEqual(value['termination']['reason'],'UPSTREAM_REMOTE_DISCONNECTED')
        self.assertEqual(value['termination']['category'],'service')
        self.assertEqual(value['formal_verdict'],'confirmed')
        self.assertEqual(value['status'],'tentative');self.assertTrue(value['review_required'])
        self.assertIn('INSUFFICIENT_FEATURE_TYPES',value['review_reasons'])
        self.assertNotIn('UPSTREAM_REMOTE_DISCONNECTED',value['review_reasons'])
        self.assertEqual(value['confidence_basis']['rule_score'],.8)
        self.assertEqual(value['confidence_basis']['rule_confidence'],.8)
        self.assertEqual(value['confidence_basis']['support_count'],1)
        self.assertEqual(value['confidence_basis']['fused_score'],.872168)
        self.assertEqual(verdict,original)

    def test_swapped_candidate_score_cannot_enter_published_explanation(self):
        verdict,scored=self.single_feature_verdict()
        def swapped(packet):
            reply,usage=self.reply(packet)
            reply['explanation']='倾向秦皮但融合分 0.240057，仍需复核。'
            return reply,usage
        with patch.object(self.runtime,'score',return_value=scored):
            value=finalize(self.runtime,verdict,self.result,self.root/'swapped',responder=swapped)
        self.assertNotIn('0.240057',value['explanation'])
        self.assertIn('0.240057',value['model_proposal']['explanation'])
        self.assertEqual(value['explanation_source'],'deterministic_numeric_guard')
        self.assertIn('0.872168',value['score_summary'])
        self.assertIn('MODEL_NUMERIC_CLAIM_UNVERIFIED',value['narrative_warnings'])
        self.assertEqual(value['status'],'tentative')

    def test_service_interruption_alone_does_not_revoke_formal_authorization(self):
        verdict,scored=self.single_feature_verdict()
        verdict.update(needs_review=False,notes=[])
        next(r for r in scored['decision']['ranking'] if r['id']=='秦皮')['support_count']=2
        result={**self.result,'stop_reason':'runtime_request_review',
                'termination_reason':'UPSTREAM_REMOTE_DISCONNECTED','planner_degraded':True}
        # 此例仅隔离“服务中断本身”：明确提供已完成的审查，覆盖缺口另有独立回归。
        from TCM_Fusion.lib.decision.candidate_audit import build as audit_candidates
        audit=audit_candidates(self.runtime,scored,result['observations'])
        audit.update(complete=True,incomplete_candidate_ids=[],unmapped=[])
        with patch.object(self.runtime,'score',return_value=scored), \
             patch('TCM_Fusion.lib.decision.candidate_audit.build',return_value=audit):
            value=finalize(self.runtime,verdict,result,self.root/'authorized',responder=self.reply)
        self.assertEqual(value['termination']['category'],'service')
        self.assertEqual(value['status'],'confirmed');self.assertFalse(value['review_required'])
        self.assertEqual(value['review_reasons'],[])

    def test_fact_reference_numbers_do_not_discard_valid_model_analysis(self):
        def referenced(packet):
            reply,usage=self.reply(packet)
            reply['explanation']='见 '+reply['support_refs'][0]+'，砂晶支持秦皮，但证据不足。'
            return reply,usage
        value=finalize(self.runtime,self.verdict,self.result,self.root/'refs',responder=referenced)
        self.assertEqual(value['explanation_source'],'injected_finalizer')
        self.assertEqual(value['narrative_warnings'],[])

    def test_compact_packet_preserves_fused_scores_by_candidate_id(self):
        verdict,_=self.single_feature_verdict()
        packet=build_packet(self.runtime,verdict,self.result)
        compact=compact_packet(packet)
        indexed={r['id']:r for r in compact['formal_verdict']['ranking']}
        self.assertEqual(indexed['秦皮']['fused_score'],.872168)
        self.assertEqual(indexed['桔梗']['fused_score'],.240057)

    def test_snapshot_does_not_trigger_legacy_planning_or_review_calls(self):
        with patch.object(self.runtime,'summary',side_effect=AssertionError('no planning callbacks')):
            value=finalize(self.runtime,self.verdict,self.result,self.root/'readonly')
        self.assertTrue(value['judgement'])

    def test_text_request_uses_reserved_pool_and_records_actual_usage(self):
        from types import SimpleNamespace
        packet=build_packet(self.runtime,self.verdict,self.result)
        self.runtime.workflow.limits['finalization']=1
        reply,_=self.reply(packet)
        import json
        body={'choices':[{'message':{'content':json.dumps(reply,ensure_ascii=False)}}],
              'usage':{'total_tokens':123}}
        client=SimpleNamespace(model='fixture',api_key='fixture')
        with patch('adjudicator.AdjudicatorClient',return_value=client), \
             patch('TCM_Fusion.lib.runtime.hard_calls.post',return_value=body) as post:
            result,usage=request_model(self.runtime,packet)
        self.assertEqual(result,reply);self.assertEqual(usage,body['usage'])
        self.assertEqual(post.call_count,1)
        self.assertLessEqual(post.call_args.args[1],30)
        self.assertEqual(read_json(self.runtime.workflow.budget_path)['used']['finalization'],1)

    def test_unknown_candidate_or_fabricated_reference_is_rejected(self):
        packet=build_packet(self.runtime,self.verdict,self.result)
        good,_=self.reply(packet)
        for invalid in [{**good,'candidate_id':'imaginary'},
                        {**good,'support_refs':['nonexistent:1']}]:
            with self.assertRaises(IntegrationError):validate_reply(packet,invalid)

    def test_one_eligible_reference_survives_mixed_citations_in_delivery(self):
        def mixed(packet):
            good, _ = self.reply(packet)
            prior = next(x['id'] for x in packet['facts'] if x['kind'] == 'a_prior')
            rival = next(x['id'] for x in packet['facts']
                         if x['kind'] == 'criterion' and x['data']['candidate_id'] == '桔梗')
            return {**good, 'support_refs':[prior, good['support_refs'][0], rival, 'missing-id'],
                    'unresolved_refs':['missing-gap'],
                    'explanation':'其他药材的判据和模型先验也确认了本药材。'}, None
        with patch.object(self, 'reply', wraps=self.reply) as calls:
            value = finalize(self.runtime, self.verdict, self.result, self.root/'mixed', responder=mixed)
            calls.assert_called_once()
        raw = value['model_proposal']
        audit = value['model_citation_audit']
        self.assertEqual(value['candidate_id'], '秦皮')
        self.assertIsNone(value['model_failure'])
        self.assertEqual(value['method'], 'injected_finalizer')
        self.assertEqual(value['explanation_source'], 'validated_references')
        self.assertNotEqual(raw['explanation'], value['explanation'])
        self.assertEqual([f['id'] for f in value['supporting_evidence']], [raw['support_refs'][1]])
        self.assertEqual(audit['accepted']['support_refs'], [raw['support_refs'][1]])
        self.assertEqual([x['reason'] for x in audit['rejected']],
                         ['prior_only','different_candidate','unknown_reference','unknown_reference'])
        self.assertEqual(value['unresolved'], [])
        self.assertTrue(value['review_required'])
        self.assertEqual(value['new_tool_calls'], 0)

    def test_local_positive_alone_survives_other_unusable_observations(self):
        packet = build_packet(self.runtime, self.verdict, self.result)
        good, _ = self.reply(packet)
        positive = next(x['id'] for x in packet['facts']
                        if x['kind']=='observation' and '秦皮' in x['supports_candidate_ids'])
        prior = next(x['id'] for x in packet['facts'] if x['kind']=='a_prior')
        checked = validate_reply(packet, {**good, 'support_refs':[prior,positive]})
        self.assertEqual(checked['support_refs'], [positive])
        audit = {}
        with self.assertRaisesRegex(IntegrationError, '没有一条合格支持'):
            validate_reply(packet, {**good, 'support_refs':[prior]}, citation_audit=audit)
        self.assertEqual(audit['accepted']['support_refs'], [])
        self.assertEqual(audit['rejected'][0]['reason'], 'prior_only')

    def test_failed_observation_cannot_support_positive_judgement(self):
        result={**self.result,'observations':[{'accepted':False,'action':self.observation['action'],
                   'result':{'ok':False,'error_code':'TIMEOUT'}}]}
        packet=build_packet(self.runtime,self.verdict,result)
        failed=next(f for f in packet['facts'] if f['kind']=='observation')
        self.assertEqual(failed['supports_candidate_ids'],[])
        self.assertEqual(failed['counters_candidate_ids'],[])
        good,_=self.reply(build_packet(self.runtime,self.verdict,self.result))
        with self.assertRaises(IntegrationError):
            validate_reply(packet,{**good,'support_refs':[failed['id']]})

    def test_local_absence_in_another_view_is_not_candidate_counterevidence(self):
        second=self.root/'second.png'
        second.write_bytes(b'second-fixture')
        runtime=WorkflowRuntime([self.image,second],root=self.root/'two-views',sample_id='s',
            a=self.runtime.a,simulation=True,backend=StageFixture(),
            workflow_config={'agent_policy':'adaptive_v2'})
        negative=copy.deepcopy(self.observation)
        image=next(img for img in runtime.images if img!=negative['action']['args']['image_id'])
        negative['action']['args']['image_id']=image
        negative['result']['present']=False
        result={**self.result,'observations':[self.observation,negative]}
        packet=build_packet(runtime,self.verdict,result)
        obs=next(f for f in packet['facts'] if f['kind']=='observation'
                 and f['data']['observation_state']=='absent')
        self.assertIn('秦皮',obs['data']['related_candidate_ids'])
        self.assertEqual(obs['counters_candidate_ids'],[])
        self.assertIn('bound field of view',obs['data']['scope'])
        good,_=self.reply(packet)
        audit = {}
        checked = validate_reply(packet, {**good, 'counter_refs':[obs['id']]}, citation_audit=audit)
        self.assertEqual(checked['support_refs'], good['support_refs'])
        self.assertEqual(checked['counter_refs'], [])
        self.assertEqual(audit['rejected'][0]['reason'], 'ineligible_counter')
        value=finalize(runtime,self.verdict,result,self.root/'same-feature')
        self.assertEqual(value['local_absence_observations'],[obs])
        self.assertTrue(value['presence_observations'])
        self.assertEqual(value['counterevidence'],[])
        self.assertTrue(all(f['kind']=='criterion' for f in value['supporting_evidence']))

    def test_uncalibrated_score_contribution_does_not_count_as_a_verified_feature(self):
        verdict,scored=self.single_feature_verdict()
        top=next(r for r in scored['decision']['ranking'] if r['id']=='秦皮')
        top['support_count']=2
        top['checks'].append({'feature_id':'fiber','present':True,'role':'supporting',
                              'state':'UNCAL','quant_bounds':{'dim':'width','min':1,'max':2},
                              'score_contribution':.5})
        with patch.object(self.runtime,'score',return_value=scored):
            value=finalize(self.runtime,verdict,self.result,self.root/'two-counts',responder=self.reply)
        self.assertEqual(value['confidence_basis']['support_count'],2)
        self.assertEqual(value['confidence_basis']['verified_feature_count'],1)
        self.assertIn('已验证独立特征类数=1',value['score_summary'])
        self.assertIn('B规则计分特征类数=2',value['score_summary'])

    def test_only_quantitative_criterion_can_be_referenced_as_counterevidence(self):
        scored=self.runtime.score(self.result['observations'])
        top=next(r for r in scored['decision']['ranking'] if r['id']=='秦皮')
        top['checks'].append({'feature_id':'fiber','present':True,'role':'supporting',
                              'state':'OUT_OF_RANGE','quant_bounds':{'dim':'width','min':1,'max':2},
                              'score_contribution':0.0})
        with patch.object(self.runtime,'score',return_value=scored):
            packet=build_packet(self.runtime,self.verdict,self.result)
        counter=next(f for f in packet['facts'] if f['kind']=='criterion'
                     and '秦皮' in f['counters_candidate_ids'])
        good,_=self.reply(packet)
        reply={**good,'counter_refs':[counter['id']]}
        self.assertEqual(validate_reply(packet,reply),reply)

    def test_invalid_model_answer_falls_back_without_new_tools(self):
        def bad(packet):
            good,usage=self.reply(packet)
            return {**good,'support_refs':['made_up']},usage
        value=finalize(self.runtime,self.verdict,self.result,self.root/'invalid',responder=bad)
        self.assertEqual(value['method'],'deterministic')
        self.assertEqual(value['model_failure'],'ASSESSMENT_INVALID')
        self.assertEqual(value['candidate_id'],'秦皮')

    def test_no_model_request_after_deadline_or_close(self):
        budget=read_json(self.runtime.workflow.budget_path);budget['deadline']=0
        write_json(self.runtime.workflow.budget_path,budget)
        responder=unittest.mock.Mock(side_effect=AssertionError('must not call'))
        value=finalize(self.runtime,self.verdict,self.result,self.root/'expired',responder=responder)
        self.assertEqual(value['model_failure'],'WORKFLOW_TIMEOUT');responder.assert_not_called()
        self.assertTrue(value['judgement'])

    def test_budget_keeps_one_closing_request_after_planning_is_exhausted(self):
        w=self.runtime.workflow;w.limits.update(planning=10,network=2,finalization=1,total=20)
        w.charge('planning','p1')
        with self.assertRaises(IntegrationError):w.charge('planning','p2')
        w.charge('finalization','last')
        used=read_json(w.budget_path)['used']
        self.assertEqual(used['network'],2);self.assertEqual(used['finalization'],1)

    def test_injected_finalizer_is_denied_in_real_scope(self):
        self.runtime.simulation=False
        with self.assertRaises(IntegrationError):
            finalize(self.runtime,self.verdict,self.result,self.root/'denied',responder=self.reply)

    def test_reserves_time_then_exits_investigation_without_calling_tool(self):
        w=self.runtime.workflow;w.limits['finalization']=1
        budget=read_json(w.budget_path);import time
        budget['deadline']=time.time()+5;write_json(w.budget_path,budget)
        result=asyncio.run(run_react(self.runtime,LocalToolbox(self.runtime,
            ledger_path=self.root/'ledger.json',backend=FixtureBackend()),
            planner=ScriptedPlanner(),planner_mode='scripted'))
        self.assertEqual(result['stop_reason'],'finalization_time_reserve')
        self.assertEqual(result['steps'],0)


    def test_provisional_selection_rebinds_explanation_and_support_to_same_candidate(self):
        provisional={'ranking':[{'candidate_id':'桔梗',
            'criterion_states':[{'state':'DETECTED_PENDING'}],
            'pending_verification':[{'role':'required','feature_id':'inulin'}]}]}
        before=self.runtime.workflow.store.current()['version']
        with patch('TCM_Fusion.lib.evidence.provisional_evidence.build',return_value=provisional):
            value=finalize(self.runtime,self.verdict,self.result,self.root/'rebound',responder=self.reply)
        self.assertEqual(value['candidate_id'],'桔梗')
        self.assertEqual(value['model_proposal']['candidate_id'],'秦皮')
        self.assertEqual(value['explanation_source'],'deterministic_selection_guard')
        self.assertIn('ASSESSMENT_SELECTION_REBOUND',value['narrative_warnings'])
        self.assertFalse(value['supporting_evidence']);self.assertFalse(value['confidence_basis']['has_admitted_b_support'])
        self.assertNotIn('砂晶存在性支持秦皮',value['explanation'])
        self.assertIn('待核验',value['judgement']);self.assertNotIn('仅有A先验',value['judgement'])
        self.assertEqual(self.runtime.workflow.store.current()['version'],before)
        self.assertTrue(value['review_required'])
