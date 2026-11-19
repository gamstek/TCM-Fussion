"""用真实候选审查投影核对逐折候选与已取得的证据。断言未映射、暂停、无合格尺寸和局部阴性状态，以及交付复核理由。视觉输入来自替身。

主要定义：
- CandidateAuditTests。

关联：
- 直接依赖 「lib/agent/react_graph.py」、「lib/decision/candidate_audit.py」、「lib/decision/final_assessment.py」、「lib/decision/output_projection.py」、「lib/tools/agent_tools.py」、「lib/workflow/workflow_runtime.py」、「tests/support/workflow_fixtures.py」。
- 统一执行见 tests/README.md。
"""
import asyncio
import copy
import unittest
from pathlib import Path

from TCM_Fusion.tests.support import workflow_fixtures as fixtures
from TCM_Fusion.lib.decision.candidate_audit import build,compact
from TCM_Fusion.lib.tools.agent_tools import LocalToolbox,FixtureBackend
from TCM_Fusion.lib.agent.react_graph import run_react
from TCM_Fusion.lib.decision.final_assessment import finalize
from TCM_Fusion.lib.decision.output_projection import project_output


class CandidateAuditTests(unittest.TestCase):
    setUp=fixtures.WorkflowFixture.setUp
    runtime=fixtures.WorkflowFixture.runtime

    def prepare(self,**kw):
        r=self.runtime(agent_policy='adaptive_v2',fast_recovery_selection='per_feature',
                       targeted_recovery=True,**kw)
        r.a={'status':'ok','fold_rankings':[[{'label':'白芍','probability':.5},
            {'label':'陈皮','probability':.3},{'label':'郁金','probability':.2}]]*5}
        return r

    def audit(self,r,obs=None):
        return build(r,r.score(obs or []),obs or [])

    def observation(self,r,feature,present,accepted=True):
        return {'accepted':accepted,'action':{'tool':'presence_check','action_id':'fixture:'+feature,
            'args':{'image_id':next(iter(r.images)),'feature_id':feature,'morph_hint':''}},
            'result':{'ok':accepted,'present':present,'conf':.9}}

    def test_five_fold_cohort_and_paused_candidates_are_not_removed(self):
        r=self.prepare();original=copy.deepcopy(r.a)
        r.update_hypotheses([{'candidate_id':'G3','status':'paused','reason':'explicit fixture defer'}],r.score([])['evidence_digest'])
        a=self.audit(r);row=next(x for x in a['candidates']if x['candidate_id']=='G3')
        self.assertFalse(row['review_complete']);self.assertFalse(row['paused_is_excluded'])
        self.assertEqual(row['hypothesis_status'],'paused');self.assertIn('G3',a['candidate_ids'])
        self.assertEqual(r.a,original)

    def test_initial_detection_does_not_establish_candidate_specific_absence(self):
        r=self.prepare();r.workflow.execute(next(a for a in r.workflow.propose()if a['tool']=='detect_fov_initial'))
        audit=self.audit(r);g=next(c for c in audit['candidates']if c['candidate_id']=='G3')
        self.assertTrue(all(x['state']=='unchecked'for x in g['required']))
        self.assertFalse(g['review_complete'])

    def test_supporting_gaps_are_preserved_alongside_required(self):
        r=self.prepare();row=next(c for c in self.audit(r)['candidates']if c['candidate_id']=='白芍')
        self.assertIn('cluster_crystal',[x['feature_id']for x in row['required']])
        self.assertIn('vessel',[x['feature_id']for x in row['supporting']])
        self.assertIn('vessel',row['unassessed_features'])

    def test_unmapped_fifth_fold_is_explicit_and_cannot_complete_audit(self):
        r=self.prepare();r.a['fold_rankings'][4]=[{'label':'玄明粉','probability':.9833}]
        a=self.audit(r);self.assertIn('玄明粉',[x['label']for x in a['unmapped']])
        self.assertFalse(a['complete']);self.assertTrue(a['comparison_required'])

    def test_positive_presence_does_not_satisfy_quantitative_supporting(self):
        r=self.prepare();a=self.audit(r,[self.observation(r,'fiber',True)])
        row=next(c for c in a['candidates']if c['candidate_id']=='白芍')
        fiber=next(x for x in row['supporting']if x['feature_id']=='fiber')
        self.assertEqual(fiber['state'],'present_pending');self.assertFalse(fiber['assessed'])

    def test_local_negative_is_observed_but_not_a_sample_exclusion(self):
        r=self.prepare();a=self.audit(r,[self.observation(r,'stoma',False)])
        row=next(c for c in a['candidates']if c['candidate_id']=='G3')
        self.assertEqual(row['supporting'][0]['state'],'local_absent')
        self.assertFalse(row['paused_is_excluded']);self.assertFalse(row['review_complete'])

    def test_failure_with_false_payload_remains_unknown(self):
        r=self.prepare();a=self.audit(r,[self.observation(r,'stoma',False,accepted=False)])
        row=next(c for c in a['candidates']if c['candidate_id']=='G3')
        self.assertEqual(row['supporting'][0]['state'],'failed_or_unknown')
        self.assertFalse(row['supporting'][0]['assessed'])

    def test_unclassified_crop_does_not_block_other_feature_in_v2(self):
        r=self.prepare();r.workflow.execute(next(a for a in r.workflow.propose()if a['tool']=='detect_fov_initial'))
        snap=r.workflow.store.current()
        crop=next(v for v in snap['artifacts'].values()if v.get('kind')=='crop')
        item={k:v for k,v in crop.items()if k!='valid'};item['identity_required']=True
        r.workflow.store.commit(snap['version'],[item])
        actions=r.workflow.propose();targets=r.workflow.targets()
        self.assertTrue(any(a['tool']=='feature_scan'and targets[a['args']['target_id']]['feature_id']=='cluster_crystal'for a in actions))
        self.assertFalse(any(a['tool']=='segment_instance'for a in actions))

    def test_legacy_classification_gate_stays_unchanged(self):
        r=self.runtime(fast_recovery_selection='per_feature',targeted_recovery=True)
        r.a={'status':'ok','fold_rankings':[[{'label':'白芍','probability':1}]]*5}
        r.workflow.execute(next(a for a in r.workflow.propose()if a['tool']=='detect_fov_initial'))
        self.assertFalse(any(a['tool']=='feature_scan'for a in r.workflow.propose()))

    def run_agent(self,r,p):
        toolbox=LocalToolbox(r,ledger_path=self.root/'obs.json',backend=FixtureBackend())
        return asyncio.run(run_react(r,toolbox,planner=p,planner_mode='scripted',max_steps=3))

    def test_stop_suggestions_are_consistent_and_review_does_not_require_extra_plan(self):
        r=self.prepare();seen=[]
        class Planner:
            def choose(inner,c):
                seen.append(c);self.assertEqual(c['stop_advisory']['recommend'],c['stop_decision']['recommended_exit'])
                audit=c['evidence']['workflow']['candidate_audit'];self.assertFalse(audit['complete'])
                return {'action_id':'review'},None
        t=self.run_agent(r,Planner());self.assertEqual(t['planning_requests'],1)
        self.assertEqual(t['stop_reason'],'planner_request_review')

    def test_finish_without_comparison_is_projected_to_review_in_same_call(self):
        r=self.prepare()
        class Planner:
            def choose(inner,c):return {'action_id':'finish'},None
        t=self.run_agent(r,Planner());self.assertEqual(t['planning_requests'],1)
        self.assertEqual(t['stop_reason'],'planner_request_review')
        self.assertIn('candidate_audit_incomplete',t['finish_veto'])

    def test_final_report_cannot_hide_cohort_gaps_behind_empty_selected_unresolved(self):
        r=self.prepare();obs=[self.observation(r,'gelatinized_starch',True)]
        scored=r.score(obs);verdict=copy.deepcopy(scored['decision'])
        verdict.update(verdict='confirmed',needs_review=False,conclusion={'id':'郁金'})
        result={'observations':obs,'events':[],'stop_reason':'planner_request_review','steps':0}
        value=finalize(r,verdict,result,self.root/'final')
        self.assertFalse(value['candidate_review_complete']);self.assertTrue(value['review_required'])
        self.assertEqual(value['status'],'tentative');self.assertIn('CANDIDATE_AUDIT_INCOMPLETE',value['review_reasons'])
        self.assertIn('selected candidate',value['unresolved_scope'])
        out=project_output('fixture',verdict,{'final_assessment':value},r.a)
        self.assertEqual(out['status'],'tentative');self.assertIsNotNone(out['candidate_audit'])

    def test_registered_detail_reader_returns_complete_candidate_audit(self):
        r=self.prepare();state={'observations':[],'attempted':[]}
        value=r.tools.read(r,'read_evidence_detail',{'kind':'eligibility','offset':0,'limit':20},state)
        self.assertIn('candidate_audit',value);self.assertIn('G3',value['candidate_audit']['candidate_ids'])

    def test_compact_audit_does_not_truncate_paused_or_fold_union(self):
        r=self.prepare();a=self.audit(r);short=compact(a)
        self.assertEqual(short['candidate_ids'],a['candidate_ids'])
        self.assertEqual(len(short['candidates']),len(a['candidates']))

    def test_pausing_candidate_does_not_erase_admitted_positive_support(self):
        runtime = self.prepare()
        runtime.a['fold_rankings'][0] = [{'label': '猪苓', 'probability': 1}]
        for tool in ('detect_fov_initial', 'segment_instance', 'measure_instances'):
            action = next(a for a in runtime.workflow.propose() if a['tool'] == tool)
            self.assertEqual(runtime.workflow.execute(action)['status'], 'done')
        runtime.update_hypotheses([{'candidate_id': '猪苓', 'status': 'paused', 'reason': 'explicit defer'}],
                                  runtime.score([])['evidence_digest'])
        row = next(c for c in self.audit(runtime)['candidates'] if c['candidate_id'] == '猪苓')
        self.assertEqual(row['hypothesis_status'], 'paused')
        self.assertTrue(row['required'][0]['satisfies_criterion'])
        self.assertFalse(row['paused_is_excluded'])

    def test_top_required_complete_does_not_hide_remaining_cohort_gaps(self):
        r=self.prepare();seen=[]
        class Planner:
            def choose(inner,c):
                seen.append(c)
                if len(seen)==1:
                    a=next(a for a in c['actions']if a['tool']=='presence_check'and a['args']['feature_id']=='gelatinized_starch')
                    return {'action_id':a['action_id']},None
                self.assertEqual(c['stop_advisory']['why'],'lead_evidence_gap')
                self.assertEqual(c['stop_advisory']['recommend'],c['stop_decision']['recommended_exit'])
                return {'action_id':'review'},None
        toolbox=LocalToolbox(r,ledger_path=self.root/'obs.json',backend=FixtureBackend(
            {'gelatinized_starch':{'ok':True,'present':True,'conf':.9}}))
        t=asyncio.run(run_react(r,toolbox,planner=Planner(),planner_mode='scripted',max_steps=3))
        self.assertEqual(t['planning_requests'],2)

    def multi(self):
        from TCM_Fusion.lib.workflow.workflow_runtime import WorkflowRuntime
        other=self.root/'second.png';other.write_bytes(b'explicit second fixture')
        return WorkflowRuntime([self.image,other],root=self.root/'multi',sample_id='multi',
            a={'status':'ok','fold_rankings':[[{'label':'陈皮','probability':1}]]*5},simulation=True,
            backend=fixtures.StageFixture(),workflow_config={'agent_policy':'adaptive_v2'})

    def test_one_negative_field_does_not_complete_multiview_audit(self):
        r=self.multi();obs=[self.observation(r,'stoma',False)]
        row=next(c for c in self.audit(r,obs)['candidates']if c['candidate_id']=='G3')['supporting'][0]
        self.assertEqual(row['state'],'local_absent_partial');self.assertFalse(row['assessed'])
        self.assertEqual((row['observed_fov_count'],row['provided_fov_count']),(1,2))

    def test_all_provided_fields_negative_still_does_not_exclude_specimen(self):
        r=self.multi();obs=[]
        for image in r.images:
            o=self.observation(r,'stoma',False);o['action']['args']['image_id']=image;obs.append(o)
        row=next(c for c in self.audit(r,obs)['candidates']if c['candidate_id']=='G3')
        self.assertEqual(row['supporting'][0]['state'],'local_absent')
        self.assertFalse(row['paused_is_excluded']);self.assertFalse(row['review_complete'])
