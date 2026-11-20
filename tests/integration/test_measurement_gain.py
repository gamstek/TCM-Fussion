"""量纲收益的菜单与交付回归。可选冻结案例直接读原签名版本，全部计算在本机完成。

FrozenMeasurementTests 重建七轮选择前的物理证据、消费者及 A/G 菜单，另核对三路交付。
MeasurementPolicyTests 独立验证来源变化时重新开放，及上下文摘要保持当轮可独立理解。
依赖原B卡池、planning_choices、measurement_gain、planning_context 与 final_assessment。
"""
import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import paths
from rule_engine import KnowledgeBase, pool_evidence
from TCM_Fusion.lib.agent.action_feedback import evaluate, snapshot
from TCM_Fusion.lib.agent.measurement_gain import dimension_gain, measurement_menu
from TCM_Fusion.lib.agent.planning_choices import attach, model_view
from TCM_Fusion.lib.agent.planning_context import compact, size
from TCM_Fusion.lib.decision.final_assessment import decision_layers, finalize, layer_report, validate_reply
from TCM_Fusion.lib.decision.rule_scoring import decide
from TCM_Fusion.lib.evidence.provisional_evidence import build
from TCM_Fusion.lib.runtime.storage import digest, read_json, write_json


class MeasurementPolicyTests(unittest.TestCase):
    def test_another_view_can_resolve_an_observed_disagreement(self):
        checks = [('X', {'quant_bounds': {'dim':'mean_width','min':8,'max':25},
                         'state':'OUT_OF_RANGE','role':'required'})]
        values = [7.0, 7.8, 9.1]
        def reasons(view):
            return dimension_gain(values, checks, {'old'}, {view}, {'X'})['consumers'][0]['reasons']
        self.assertEqual(reasons('old'), [])
        self.assertEqual(reasons('new'), ['test_observed_disagreement_in_new_view'])

    def test_opposing_outliers_and_automatic_samples_are_not_double_counted(self):
        bounds = {'dim':'mean_width','min':8,'max':25}
        check = {'role':'required','quant_bounds':bounds}
        group = dimension_gain([5,30],[('candidate',check)],{'one'},{'one'},{'candidate'})
        self.assertTrue(group['worth_measuring'])
        self.assertEqual(group['consumers'][0]['sides'],{'below_minimum':1,'above_maximum':1})
        from TCM_Fusion.lib.agent.action_feedback import planning_changes
        fb = {'artifacts_added':{'measurement':1},'measurement_sample_change':{'f':{'before':1,'after':2}},
              'criterion_transitions':[],'candidate_order_changed':False}
        changes = planning_changes([{'phase':'recompute','tool':'segment_instance','feedback':fb},
                                    {'phase':'automatic_measurement_feedback','feedback':fb}])
        self.assertEqual(len(changes),1)
        self.assertEqual(changes[0]['criterion_changes'],[])
        self.assertFalse(changes[0]['ranking_changed'])

    def test_repeated_question_reuses_text_without_mutating_shared_live_args(self):
        shared={'image_id':'img','feature_id':'fiber','morph_hint':'同一个形态问题'*20}
        different={**shared,'morph_hint':shared['morph_hint']+'但限于该区域'}
        live={'action_id':'registered','tool':'presence_check','args':shared}
        ctx={'actions':[live], 'previous_observations':[{'action':a,'result':{}} for a in
            (live,live,{**live,'args':different})]}
        view,_,_=compact(ctx)
        self.assertEqual(view['actions'][0]['args'],shared)
        rows=view['previous_observations']
        self.assertEqual(rows[0]['action']['args']['morph_hint'],shared['morph_hint'])
        self.assertEqual(rows[1]['action']['args']['morph_hint_from_observation'],0)
        self.assertEqual(rows[2]['action']['args'],different)

    def test_localization_only_links_exact_source_even_across_window_boundaries(self):
        from TCM_Fusion.lib.agent.react_graph import compact_round_context
        from TCM_Fusion.lib.evidence.evidence_projection import recent_localizations
        def obs(i, other):
            return {'action':{'action_id':str(i),'tool':'presence_check',
                    'args':{'feature_id':'fiber','image_id':'img_s'}},'presence_localization':{
                    'source_image_sha256':'s','feature_id':'fiber','identity':'pending',
                    'usage_eligibility':{'length':False},'state':'present_pending',
                    'spatial_agreement':[{'artifact_id':other,'feature_id':other,'iou':.82,
                                          'possible_identity_conflict':True}]}}
        history = [obs(i, 'vessel' if i%2 else 'hypha') for i in range(35)]
        for width in (6,9,32):
            with self.subTest(window=width):
                context = {'actions':[],'previous_observations':history,
                           'evidence':{'presence_localizations':recent_localizations(history)}}
                wf = SimpleNamespace(config={'context_compact':True,'context_recent_observations':width})
                view = compact(compact_round_context(context,wf))[0]
                shown = view['previous_observations']
                expected = history[-width:]
                self.assertEqual(len(shown),width)
                self.assertEqual(view['context_meta']['previous_observations_total'],35)
                for index, row in enumerate(shown):
                    raw = expected[index]
                    self.assertEqual(raw['action']['action_id'],row['action']['action_id'])
                    loc = row['presence_localization']
                    ref = loc.get('index_in_presence_localizations')
                    if index < width-8:
                        self.assertIsNone(ref)
                    else:
                        self.assertIsNotNone(ref)
                        loc = view['evidence']['presence_localizations'][ref]
                    self.assertEqual(loc['source_action_id'],raw['action']['action_id'])
                    original = raw['presence_localization']['spatial_agreement'][0]
                    self.assertEqual(loc['spatial_agreement']['identity_conflicts'],
                                     [{k:original[k] for k in ('artifact_id','feature_id','iou')}])

        original = obs(1,'vessel')
        revised = copy.deepcopy(original)
        revised['presence_localization']['transform'] = {'revision':2}
        missing = obs(2,'hypha')
        del missing['action']['action_id']
        wrong = obs(3,'fiber')
        wrong['action']['args']['image_id'] = 'img_other'
        for seen, catalog in [(original,[original,original]),(original,[revised]),
                              (missing,[missing]),(wrong,[wrong]),(original,[])]:
            context={'actions':[],'previous_observations':[seen],
                     'evidence':{'presence_localizations':recent_localizations(catalog)}}
            result=compact(compact_round_context(context,wf))[0]
            self.assertNotIn('index_in_presence_localizations',result['previous_observations'][0]['presence_localization'])

    def test_unchanged_history_compacts_with_all_active_roles_and_index(self):
        audit = {'candidates':[{'candidate_id':'X','required':{'f':'supported'},
                                'supporting':{'g':'unchecked'}}], 'incomplete_candidate_ids':['X']}
        ctx = {'evidence':{'candidate_interest':{'ids':['X']},
            'criteria_diff':[{'candidate_id':'X','criteria':[{'role':'required','feature':'f',
                                                               'morph':'visible detail ' * 90}]}],
            'workflow':{'candidate_audit':audit, 'evidence_index':{'detail_access':['read_evidence_detail']}}},
            'previous_observations':[{'action':{}, 'accepted':True, 'result':{'present':True},
                'diagnostics':{'raw':'a' * 2000}}], 'actions':[]}
        first, cursor, _ = compact(ctx)
        second, _, _ = compact(ctx, cursor)
        self.assertLess(size(second),size(first))
        self.assertTrue(second['round_delta']['baseline_available'])
        self.assertEqual(second['round_delta']['criterion_changes'], [])
        self.assertEqual(second['evidence']['workflow']['candidate_audit'],audit)
        self.assertIn('evidence_index',second['evidence']['workflow'])
        new = copy.deepcopy(ctx)
        new['evidence']['workflow']['candidate_audit']['candidates'][0]['supporting']['g']='supported'
        view, _, _ = compact(new,cursor)
        self.assertEqual(view['round_delta']['criterion_changes'][0]['feature_id'],'g')


class FrozenMeasurementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        run_id = 'fusion_3f83139fa29e4be5b7b28be887699664'
        matches = list(Path(paths.RUN_RESULTS).glob('*/runs/' + run_id + '/final_evidence.json'))
        if not matches:
            raise unittest.SkipTest('本机没有登记的连翘四图冻结案例')
        cls.directory, = [p.parent for p in matches]
        cls.trace = read_json(cls.directory / 'agent_trace.json')
        cls.packet = read_json(matches[0])
        cls.kb = KnowledgeBase(Path(paths.JUDGE_DIR) / 'data/herb_criteria.json')

    def version(self, version):
        snap = read_json(self.directory / 'workflow/evidence/versions' / (version+'.json'))
        self.assertEqual(digest({k:v for k,v in snap.items() if k!='version'}), version)
        return snap

    def test_seven_frozen_pre_action_menus(self):
        events = self.trace['events']
        offers = [e['menu'] for e in events if e['phase']=='planner_offer']
        round_number = 0
        tested = []
        for event in events:
            if event['phase']=='thought':
                round_number += 1
            if event['phase']!='automatic_measurement_feedback':
                continue
            op = self.trace['operations'][event['operation_key']]
            target = op['target_id'].removeprefix('seg:')
            if 'secretory_tissue_view_03' not in target:
                continue
            old = offers[round_number-1]
            version = next(a['args']['evidence_version'] for a in old['actions']
                           if a.get('args', {}).get('target_id') == target)
            snap = self.version(version)
            arts = snap['artifacts']
            crops = {key:v for key,v in arts.items() if v.get('valid')}
            active = old['investigation']['comparison_problems']['active_candidates']
            bound_image = crops[target]['image_id']
            images = {key:{'stems':[Path(a['path']).stem]} for key,a in arts.items() if a['kind']=='image'}
            runtime = SimpleNamespace(kb=self.kb, images=images, candidate_interest=lambda:{'ids':active},
                tools=SimpleNamespace(get=lambda name:SimpleNamespace(layer='recompute')),
                workflow=SimpleNamespace(config={'semantic_execution':True},targets=lambda:crops,
                                         store=SimpleNamespace(current=lambda:snap)))
            def scoring(version_data):
                cards = pool_evidence([r['data'] for r in version_data['records']])
                return {'cards':cards,'decision':decide(self.kb,cards)}
            scored = scoring(snap)
            # 复用原菜单中已合法的实例动作。另向同一策略输入可直接分割的对照入口。
            old_actions = [a for a in old['actions'] if a['tool'] in
                {'classify_target','segment_instance','measure_instances'}]
            actions, groups, keys = measurement_menu(runtime, old_actions, scored, crops)
            direct = {'action_id':'segment-control','tool':'segment_instance','args':{'target_id':target}}
            direct_choices, _, _ = measurement_menu(runtime,[direct],scored,crops)
            rendered, _ = model_view(attach(runtime, {'actions':actions,'measurement_groups':groups},
                                              targets=crops, can_pair=False, measurement_actions=keys))
            goals = [g for g in rendered['execution_options']['goals'] if g['goal']['instance_ref']==op['instance_id']]
            width_goal = next((g for g in goals if g['goal']['semantic_task']=='measure_width'),None)
            self.assertTrue(any(g['goal']['semantic_task']=='verify_identity' for g in goals))
            group = groups['secretory_tissue@'+bound_image]['dimensions']['mean_width']
            n = len(tested)
            self.assertEqual(group['n_measured'],n)
            self.assertEqual(bool(direct_choices),bool(width_goal))
            feedback = evaluate(snapshot(scored),snapshot(scoring(self.version(op['evidence_version']))),
                                direct,accepted=True)
            self.assertEqual(feedback['measurement_sample_change']['secretory_tissue']['after'],n+1)
            if n in {0,1}:
                self.assertIn('厚朴',width_goal['criterion_consumers'])
            elif n == 2:
                self.assertEqual(width_goal['criterion_consumers'], ['柴胡'])
            else:
                self.assertIsNone(width_goal)
            transitions = {(c['candidate_id'],c['after']) for c in feedback['criterion_transitions']
                           if c['feature_id']=='secretory_tissue'}
            if n == 0:
                self.assertIn(('厚朴','OUT_OF_RANGE'),transitions)
            if n == 2:
                self.assertIn(('柴胡','OUT_OF_RANGE'),transitions)
            if n >= 3:
                self.assertFalse(transitions)
                archived = next(e['feedback'] for e in events if e.get('phase')=='recompute'
                    and e.get('tool')=='segment_instance' and e.get('evidence_version')==op['evidence_version'])
                self.assertFalse(archived['candidate_order_changed'])
            tested.append(round_number)
        self.assertEqual(tested, [17,22,28,29,30,31,32])

    def test_all_frozen_rounds_bind_local_results_to_the_same_source_and_purpose(self):
        from TCM_Fusion.lib.agent.react_graph import compact_round_context
        from TCM_Fusion.lib.evidence.evidence_projection import recent_localizations
        ledger = read_json(self.directory/'agent_tool_ledger.json')['entries']
        for width in (6,9,32):
            with self.subTest(window=width):
                observed = []
                rounds = 0
                cursor = None
                saw_conflict = False
                wf = SimpleNamespace(config={'context_compact':True,'context_recent_observations':width})
                for event in self.trace['events']:
                    if event['phase']=='action_observation':
                        action_id = event['action_id']
                        record = ledger[action_id]
                        self.assertEqual((record['action_id'],record['action']['action_id'],record['status']),
                                         (action_id,action_id,'done'))
                        observed.append(record)
                    if event['phase']!='context_profile':
                        continue
                    rounds+=1
                    payload={'actions':[],'previous_observations':observed,
                             'evidence':{'presence_localizations':recent_localizations(observed)}}
                    view,cursor,_=compact(compact_round_context(payload,wf),cursor)
                    wire,_=model_view(view)
                    evidence=wire['evidence']
                    shown=wire['previous_observations']
                    expected=observed[-width:]
                    source_rows=[r for r in observed if 'presence_localization' in r][-8:]
                    self.assertEqual([r['source_action_id'] for r in evidence['presence_localizations']],
                                     [r['action_id'] for r in source_rows])
                    self.assertEqual(wire['context_meta']['previous_observations_total'],len(observed))
                    self.assertEqual(len(shown),len(expected))
                    for index,record in enumerate(expected):
                        item=shown[index]
                        args=dict(item['action']['args'])
                        if 'morph_hint_from_observation' in args:
                            prior=shown[args.pop('morph_hint_from_observation')]
                            args['morph_hint']=prior['action']['args']['morph_hint']
                        self.assertEqual((item['action']['action_id'],args,item['result'],item['accepted']),
                                         (record['action_id'],record['action']['args'],record['result'],record['accepted']))
                        row=evidence['presence_localizations'][item['presence_localization']['index_in_presence_localizations']]
                        self.assertEqual(row['source_action_id'],record['action_id'])
                        state={**evidence.get('localization_states',{}).get(row.get('state_ref'),{}),**row}
                        raw=record['presence_localization']
                        for k in ('source_image_sha256','feature_id','quality','identity','state','observation_scope',
                                  'findings','usage_eligibility','source_tight_bbox','crop_bbox'):
                            self.assertEqual(state.get(k),raw.get(k))
                        conflicts=[{k:h[k] for k in ('artifact_id','feature_id','iou')}
                                   for h in raw.get('spatial_agreement',[]) if h.get('possible_identity_conflict')]
                        self.assertEqual(state.get('spatial_agreement',{}).get('identity_conflicts',[]),conflicts)
                        saw_conflict |= bool(conflicts)
                self.assertTrue(saw_conflict)
                self.assertEqual((rounds,len(observed),len(ledger)),(41,8,8))
                self.assertEqual({r['action_id'] for r in observed},set(ledger))

    def test_frozen_decision_views_and_qualified_citations(self):
        p = self.packet
        self.assertEqual(p['formal_verdict']['fusion_weights'],{'a':0.5,'b':0.5,'a_folds':20})
        layers = decision_layers(p,False)
        self.assertFalse(layers['formal']['authorized'])
        fusion, b = layers['fusion_leader'],layers['b_leader']
        self.assertEqual((fusion['id'],fusion['score']),('连翘',0.531722))
        self.assertEqual((b['id'],b['score']),('厚朴',0.728571))
        self.assertIn(('fiber','NO_REF'),[(f['feature_id'],f['state']) for f in fusion['support']])
        self.assertEqual({x['feature_id'] for x in b['support']},{'stone_cell','fiber'})
        self.assertEqual({x['feature_id'] for x in b['counter']},{'secretory_tissue'})
        self.assertIn('定性判据已采信', layer_report(layers))
        verdict = p['formal_verdict']
        provisional = build(verdict, {'valid_classifications':self.trace['valid_classifications']})
        state = next(c for r in provisional['ranking'] if r['candidate_id']=='连翘'
                     for c in r['criterion_states'] if c['feature_id']=='fiber')
        self.assertEqual(state['state'],'QUALITATIVE_ADMITTED_NO_RANGE')
        self.assertFalse(state['pending'])
        self.assertEqual(validate_reply(p,self.trace['final_assessment']['model_proposal'])['candidate_id'],'厚朴')
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            budget = out / 'budget.json'
            write_json(budget, {'closed': False})
            wf = SimpleNamespace(limits={}, config={'agent_policy':'adaptive_v2'}, budget_path=budget,
                remaining_seconds=lambda:30, store=SimpleNamespace(current=lambda:{'version':p['evidence_version']}))
            runtime = SimpleNamespace(simulation=True, workflow=wf,
                valid_classifications=lambda:self.trace['valid_classifications'])
            with patch('TCM_Fusion.lib.decision.final_assessment.build_packet',return_value=p):
                result = finalize(runtime,verdict,self.trace,out,
                    responder=lambda _: (self.trace['final_assessment']['model_proposal'],None))
            self.assertEqual(result['decision_layers'],layers)
            self.assertEqual((result['candidate_id'], result['status']),('连翘','tentative'))
            self.assertTrue(result['review_required'])
            self.assertNotIn('fiber 待核验',result['judgement'])
            self.assertEqual(result['new_tool_calls'],0)

