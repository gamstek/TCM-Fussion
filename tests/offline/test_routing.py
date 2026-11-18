"""路由真实编排分支、候选权限与零 API 只读验证；不把替身当识别效果。

主要定义：
- RoutingTests shadow/trial 路由: shadow 只建议、trial 真正跳过 B；边界分数、非法概率分布与不支持的结构标签都不释放。
- 候选与多视图: `candidates` 的 top_k、未知 mass 不重归一化、跨视图折序与视图顺序保持。
- 只读工具: `LocalToolbox.read` 不产生模型请求、不改变证据摘要，越权动作与超限参数被拒。
- 动作注册与权限: `action_id` 摘要稳定，额外参数（如 `size_um`）在执行前被拒。
- stdio MCP: 工具注册表条目、只读读取不含真实标签、预算耗尽返回 `RESOURCE_BUDGET` 且不落账。

关联：
- TCM_Fusion.lib.routing_policy: `route` 与 `validate_policy`。
- TCM_Fusion.lib.pipeline: `run_pipeline` 与 `validate_config`。
- TCM_Fusion.lib.candidate_policy: `candidates` 候选重组。
- TCM_Fusion.lib.evidence_runtime/agent_tools/react_graph/mcp_bridge: 运行时、工具、ReAct 与 MCP 边界。
- tests/fixtures/recorded_pipeline.json、react_case.json: 配置与记录夹具。
"""
import asyncio
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import paths
from TCM_Fusion.lib.storage import IntegrationError, read_json, digest
from TCM_Fusion.lib.collaboration import knowledge_base
from TCM_Fusion.lib.pipeline import run_pipeline, validate_config
from TCM_Fusion.lib.routing_policy import route, validate_policy
from TCM_Fusion.lib.candidate_policy import candidates
from TCM_Fusion.lib.evidence_runtime import EvidenceRuntime
from TCM_Fusion.lib.agent_tools import LocalToolbox, FixtureBackend
from TCM_Fusion.lib.react_graph import run_react, ScriptedPlanner
from TCM_Fusion.lib.mcp_bridge import connect_stdio

BASE = Path(paths.PROJECT_ROOT)/'TCM_Fusion'


class RoutingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.a = {'status':'ok', 'ranking':[{'label':'甘草','probability':0.999},
                                         {'label':'白芍','probability':0.001}]}
        self.policy = {'mode':'shadow','min_score':0.99,'min_margin':0.9,'buffer':0.0001}
        self.config = read_json(BASE/'tests/fixtures/recorded_pipeline.json')
        self.config['routing'] = self.policy

    def runtime(self, policy=None):
        image = self.root/'fov.png'; image.write_bytes(b'simulation-only')
        f = read_json(BASE/'tests/fixtures/react_case.json')
        return EvidenceRuntime(f['records'], [image], sample_id='test', a=self.a,
                               simulation=True, candidate_policy=policy)

    def test_shadow_and_trial(self):
        r = route(self.a,self.policy,knowledge_base()); self.assertTrue(r['recommend_skip']); self.assertFalse(r['skip'])
        self.assertTrue(route(self.a,{**self.policy,'mode':'trial'},knowledge_base())['skip'])

    def test_boundary_goes_to_agent(self):
        self.a['ranking'][0]['probability']=0.99;self.a['ranking'][1]['probability']=0.01
        self.assertFalse(route(self.a,self.policy,knowledge_base())['skip'])
        self.assertFalse(route(self.a,self.policy,knowledge_base())['recommend_skip'])

    def test_invalid_distribution(self):
        for probability in (float('nan'), True, -1, 2):
            self.a['ranking'][0]['probability']=probability
            self.assertFalse(route(self.a,self.policy,knowledge_base())['recommend_skip'])
        self.a['ranking'][0]['probability']=0.9
        self.assertEqual(route(self.a,self.policy,knowledge_base())['reason'],'incomplete_distribution')

    def test_unsupported_structural_label_not_released(self):
        self.a['ranking'][0]['label']='大黄导管'
        self.assertFalse(route(self.a,self.policy,knowledge_base(),{'大黄导管':'大黄'})['recommend_skip'])

    def test_no_silent_validated_mode_or_bad_threshold(self):
        for change in ({'mode':'validated'}, {'buffer':0}, {'min_score':True}, {'top_k':True}):
            with self.assertRaises(IntegrationError):validate_policy({**self.policy,**change})

    def test_trial_really_skips_b(self):
        self.config['routing']={**self.policy,'mode':'trial'}
        with patch('TCM_Fusion.lib.pipeline.predict_a',return_value=self.a), patch('TCM_Fusion.lib.pipeline.execute_b') as b:
            directory=run_pipeline(self.config,base=self.root)
        b.assert_not_called();value=read_json(directory/'fusion.json')
        self.assertIsNone(value['b']);self.assertFalse(value['agent']['executed']);self.assertTrue(value['needs_review'])
        self.assertNotIn('confirmed',(directory/'report.md').read_text('utf-8').split('不代表')[0])

    def test_shadow_does_not_skip_b(self):
        from TCM_Fusion.lib import pipeline
        original=pipeline.execute_b
        with patch('TCM_Fusion.lib.pipeline.predict_a',return_value=self.a),patch('TCM_Fusion.lib.pipeline.execute_b',wraps=original) as b:
            directory=run_pipeline(self.config,base=self.root)
        b.assert_called_once();self.assertTrue(read_json(directory/'fusion.json')['b_executed'])

    def test_top_k_interest_does_not_modify_rule_result(self):
        r=self.runtime({'top_k':2});before=r.score([])
        c=candidates(self.a,r.kb,{}, {'top_k':2})
        self.assertEqual(len(c['ids']),2)
        r.legal_actions({'observations':[]});after=r.score([])
        self.assertEqual(before['decision'],after['decision']);self.assertEqual(before['evidence_digest'],after['evidence_digest'])

    def test_unknown_mass_and_no_renormalization(self):
        r=self.runtime();a=copy.deepcopy(self.a);a['ranking'][0]['label']='not-in-kb'
        c=candidates(a,r.kb,{}, {'top_k':1})
        self.assertTrue(c['candidate_incomplete']);self.assertEqual(c['selected_mass'],0.999);self.assertEqual(c['ids'],[])

    def test_multiview_candidates_preserve_view_and_fold_order(self):
        views = [{'status': 'ok', 'image': f'view_{i}.jpg',
                  'fold_rankings': [[{'label': label, 'probability': (j+1)/10}]
                                    for j in range(5)]}
                 for i, label in enumerate(('甘草', '白芍', '甘草', '白芍'), 1)]
        result = candidates({'view_predictions': views}, self.runtime().kb, config={'top_k': 1})
        self.assertEqual(result['source'], 'multiview_fold_top_k')
        self.assertEqual([v['image_stem'] for v in result['views']], [f'view_{i}' for i in range(1, 5)])
        for original, actual in zip(views, result['views']):
            self.assertEqual([f['fold'] for f in actual['fold_top_k']], list(range(1, 6)))
            self.assertEqual([(r['label'], r['probability']) for f in actual['fold_top_k'] for r in f['ranking']],
                             [(r['label'], r['probability']) for f in original['fold_rankings'] for r in f])
        self.assertEqual(len(result['ids']), 2)

    def test_read_tools_no_network_or_mutation(self):
        r=self.runtime();before=digest(r.records);backend=FixtureBackend()
        tools=LocalToolbox(r,ledger_path=self.root/'ledger.json',backend=backend)
        for name,args in [('read_evidence',{'offset':0,'limit':20}),('list_features',{}),('list_fovs',{}),('read_scale',{}),('criteria_diff',{})]:
            value=asyncio.run(tools.read(name,args,{}));self.assertIn('evidence_digest',value)
        self.assertEqual(backend.calls,[]);self.assertEqual(tools.ledger['requests'],0);self.assertEqual(digest(r.records),before)
        with self.assertRaises(IntegrationError):r.tools.read(r,'observe',{}, {})
        with self.assertRaises(IntegrationError):r.tools.read(r,'read_evidence',{'offset':0,'limit':500}, {})

    def test_registry_preserves_old_action_identity_and_denies_extra_args(self):
        r=self.runtime();actions=r.legal_actions({});a=actions[0]
        body={k:a[k] for k in ('tool','args','skill','config_identity','kb_sha256')}
        self.assertEqual(a['action_id'],digest(body));r.validate_action(a)
        bad=copy.deepcopy(a);bad['args']['size_um']=5
        with self.assertRaises(IntegrationError):r.validate_action(bad)

    def test_planner_can_read_without_observation_charge(self):
        r=self.runtime({'top_k':2});tools=LocalToolbox(r,ledger_path=self.root/'ledger.json',backend=FixtureBackend())
        class Reader(ScriptedPlanner):
            def choose(self, context):
                action=next((a for a in context['actions'] if a['tool']=='read_scale'),None)
                return {'action_id':action['action_id'] if action else 'review'},None
        result=asyncio.run(run_react(r,tools,planner=Reader(),planner_mode='scripted',max_steps=5,max_requests=6))
        self.assertTrue(any(e['phase']=='read' for e in result['events']))
        self.assertEqual(result['observation_requests'],tools.ledger['requests'])

    def test_mcp_registry_and_read_transport(self):
        image=self.root/'fov.png';image.write_bytes(b'simulated')
        from TCM_Fusion.lib.storage import write_json
        f=read_json(BASE/'tests/fixtures/react_case.json');f['images']=[str(image)]
        path=self.root/'context.json';write_json(path,{'simulation':True,'sample_id':'test','fixture':f})
        async def check():
            async with connect_stdio(path) as tools:
                specs=await tools.call('list_tools_registry')
                self.assertEqual(len(specs['tools']),13)
                self.assertIn('read_quality',{s['name'] for s in specs['tools']})
                self.assertIn('read_action_catalog',{s['name'] for s in specs['tools']})
                self.assertIn('read_evidence_detail',{s['name'] for s in specs['tools']})
                result=await tools.read('read_evidence',{'offset':0,'limit':5},{})
                self.assertIn('records',result)
                self.assertTrue(all('herb' not in row for row in result['records']))
        asyncio.run(check())

    def test_mcp_recompute_returns_structured_budget_error_without_api(self):
        from TCM_Fusion.lib.storage import write_json
        from TCM_Fusion.lib.workflow_runtime import from_task
        image=self.root/'fov.png';image.write_bytes(b'explicit-test-no-api')
        directory=self.root/'task';directory.mkdir()
        cfg={'sample_id':'budget_test','allow_network':True,'a':{'mode':'unavailable'},
             'b':{'mode':'run','input_mode':'images','input':str(image),'enable_vlm':False},
             'agent':{'mode':'active','planner':'llm','image_roots':[str(image)],
                      'workflow':{'max_initializations':0,'max_recompute':0,'max_api_calls':0}}}
        a={'status':'unavailable','ranking':[]};write_json(directory/'a.json',a)
        runtime=from_task(cfg,a,directory);action=runtime.workflow.propose()[0]
        context=self.root/'workflow_context.json'
        write_json(context,{'workflow_config':cfg,'a_prediction':str(directory/'a.json'),
                            'directory':str(directory),'allow_network':True})
        async def check():
            async with connect_stdio(context) as tools:
                result=await tools.recompute(action)
                self.assertFalse(result['accepted']);self.assertEqual(result['error_code'],'RESOURCE_BUDGET')
        asyncio.run(check())
        budget=read_json(directory/'workflow/resources.json')
        self.assertEqual(budget['used'],{});self.assertEqual(budget['entries'],{})
