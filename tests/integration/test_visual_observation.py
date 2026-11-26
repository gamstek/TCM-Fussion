"""核对实际规划短编号、MCP 动作、视觉请求以及下一轮可见的观察。

接口：拦截规划及视觉 HTTP，使用同一裁图对拍开放观察和自由追问。
范围：验证投递、回执、去重与证据边界，测试图像和回复均为模拟。
"""
import asyncio
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PIL import Image
from mcp.shared.memory import create_connected_server_and_client_session

from TCM_Fusion.cli.classify_worker import main as classify_worker
from TCM_Fusion.cli.workflow_worker import main as locate_worker
from TCM_Fusion.lib.agent.react_graph import LLMPlanner, run_react
from TCM_Fusion.lib.mcp.mcp_bridge import make_server, MCPToolbox
from TCM_Fusion.lib.offline.workflow_replay import RecordedObservations
from TCM_Fusion.lib.runtime.storage import IntegrationError, read_json, write_json
from TCM_Fusion.lib.tools.agent_tools import LocalToolbox
from TCM_Fusion.lib.vision.visual_observation import (
    prompt_suffix, receipt, resolve_guidance, selected_briefs, validate_brief,
)
from TCM_Fusion.lib.workflow.workflow_backend import WorkflowBackend
from TCM_Fusion.tests.support.workflow_fixtures import StageFixture, WorkflowFixture


FIRST = {'finding': '右侧可见一段管壁', 'location': '右侧', 'readability': 'partial'}
QUESTION = '右侧管壁的增厚纹理怎样排列？'
ANSWER = {'finding': '右上可辨斜向平行纹', 'location': '右上', 'readability': 'clear'}


def envelope(data):
    return {'choices': [{'message': {'content': json.dumps(data, ensure_ascii=False)}}]}


class InitialCrop(StageFixture):
    def execute(self, session, tool, target, work, context):
        result = super().execute(session, tool, target, work, context)
        directory = Path(result['crops_dir'])
        sidecar = directory / 'crop.json'
        write_json(sidecar, {**read_json(sidecar), 'feature_id': 'vessel', 'object_label': '导管', 'box_source':'yolo'})
        with Image.open(session.runtime.images[target['image_id']]['path']) as source:
            source.crop((0, 0, 8, 8)).save(directory / 'crop.png')
            write_json(result['completion'], {**read_json(result['completion']), 'image_size':list(source.size)})
        return result


class VisualObservationTests(WorkflowFixture, unittest.TestCase):
    def setUp(self):
        super().setUp()
        Image.new('RGB', (320, 320), 'white').save(self.image)
        self.enterContext(patch('ls_common.load_config', return_value={
            'vlm': {'model': 'fixture', 'base_url': 'https://offline.invalid', 'max_retries': 1}}))
        self.enterContext(patch('ls_common.resolve_key', return_value='fixture'))

    def detected(self, **settings):
        runtime = self.runtime(InitialCrop(), agent_policy='adaptive_v2', context_compact=True,
                               enabled_tools=['detect_fov_initial','classify_target','inspect_crop','feature_scan'],
                               **settings)
        session = runtime.workflow
        session.execute(next(a for a in session.propose() if a['tool'] == 'detect_fov_initial'))
        session.backend = WorkflowBackend()
        return runtime

    async def run_through_mcp(self, runtime, max_steps):
        local = LocalToolbox(runtime, ledger_path=self.root/'obs.json')
        with patch('TCM_Fusion.lib.mcp.mcp_bridge.load_bound_context', return_value=(runtime, local)):
            server = make_server(self.root/'context.json')
        async with create_connected_server_and_client_session(server) as client:
            planner = LLMPlanner.__new__(LLMPlanner)
            planner.client = SimpleNamespace(model='fixture', timeout_s=2)
            planner.config = {}
            return await run_react(runtime, MCPToolbox(client), planner=planner, planner_mode='scripted',
                                   max_steps=max_steps)

    def test_model_authors_followup_on_same_pixels_through_mcp(self):
        runtime = self.detected()
        workflow = runtime.workflow
        version = workflow.store.current()['version']
        previous_score = runtime.score([])['evidence_digest']
        payloads, menus = [], []
        rationale = '右侧管壁已有可读回执，需要区分邻近纹理。' * 16 + '末尾保留本轮的选择依据。'

        def plan(payload, timeout):
            menu = json.loads(payload['messages'][1]['content'])
            menus.append(menu)
            tool = 'inspect_crop' if len(menus) <= 2 else 'request_review'
            aid = next(a['action_id'] for a in menu['actions'] if a['tool'] == tool)
            reply = {'decision': {'kind':'action', 'action_id':aid}, 'decision_summary':'这句理由仅供规划'}
            if len(menus) == 1:
                reply['decision']['visual_briefs'] = {aid: {'use': 'preset'}}
            elif len(menus) == 2:
                reply['decision']['visual_briefs'] = {aid: QUESTION, 'A999': '未选动作上的文字'}
                del reply['decision_summary']
                reply['decision']['rationale'] = rationale
            result = envelope(reply)
            result['choices'][0]['message']['reasoning_content'] = 'PRIVATE_FIXTURE_FIELD'
            return result

        def vision(request, timeout):
            payloads.append(json.loads(request.data))
            data = {'visual_observation': FIRST if len(payloads) == 1 else ANSWER,
                    'identity_verified': True, 'primary_value_um': 37}
            return io.BytesIO(json.dumps(envelope(data)).encode())

        with patch('TCM_Fusion.lib.runtime.hard_calls.post', side_effect=plan), \
             patch('TCM_Fusion.lib.workflow.workflow_backend.run',
                   side_effect=lambda argv, **opts: classify_worker(argv[-2:])), \
             patch('urllib.request.urlopen', side_effect=vision):
            trace = asyncio.run(self.run_through_mcp(runtime, max_steps=5))

        self.assertEqual(len(payloads), 2, [(e.get('phase'), e.get('reason') or e.get('error_code'),
                                           e.get('last_rejection'), e.get('result', {}).get('error_code'))
                                          for e in trace['events']])
        first, second = (p['messages'][0]['content'] for p in payloads)
        self.assertEqual(first[0], second[0])
        self.assertNotIn(QUESTION, first[1]['text'])
        self.assertIn(QUESTION, second[1]['text'])
        self.assertNotIn(rationale, second[1]['text'])
        self.assertNotIn('这句理由', second[1]['text'])
        self.assertNotIn('未选动作', second[1]['text'])
        self.assertIn(FIRST['finding'], json.dumps(menus[1]['recent_action_feedback'], ensure_ascii=False))
        self.assertIn(ANSWER['finding'], json.dumps(menus[2]['recent_action_feedback'], ensure_ascii=False))
        self.assertEqual(trace['planning_requests'], 3)
        thoughts = [row for row in trace['events'] if row['phase'] == 'thought']
        choices = [next(iter(row['visual_guidance_choices'].values())) for row in thoughts[:2]]
        self.assertEqual([choice['mode'] for choice in choices], ['explicit_preset','agent_brief'])
        self.assertEqual(thoughts[1]['decision_summary'], rationale)
        self.assertEqual(thoughts[1]['decision_summary_capture'], 'complete')
        self.assertNotIn('PRIVATE_FIXTURE_FIELD', json.dumps(trace, ensure_ascii=False))
        rows = [r for r in read_json(workflow.operations)['entries'].values() if r['tool'] == 'inspect_crop']
        self.assertEqual(len(rows), 2)
        follow = next(r for r in rows if r.get('visual_brief'))
        self.assertEqual(follow['visual_observation'], {**ANSWER,'brief':QUESTION,'status':'reported'})
        self.assertEqual(workflow.store.current()['version'], version)
        self.assertEqual(runtime.score([])['evidence_digest'], previous_score)
        self.assertFalse(any(a['tool'] == 'inspect_crop' for a in workflow.propose()))

    def test_unreadable_and_reworded_briefs_do_not_reissue_or_verify(self):
        runtime = self.detected()
        workflow = runtime.workflow
        action = next(a for a in workflow.propose() if a['tool'] == 'inspect_crop')
        response = {'visual_observation': {'finding':'关键区域失焦', 'location':'', 'readability':'unreadable'}}
        with patch('TCM_Fusion.lib.workflow.workflow_backend.run',
                   side_effect=lambda argv, **opts: classify_worker(argv[-2:])), \
             patch('vlm_api.call_vlm', return_value=(json.dumps(response), None)) as call:
            first = workflow.execute(action)
            cached = workflow.execute({**action, 'visual_brief': QUESTION})
        self.assertTrue(first['accepted'])
        self.assertTrue(cached['cache_hit'])
        self.assertNotIn('brief', cached['visual_observation'])
        self.assertEqual(call.call_count, 1)
        self.assertFalse(any(a['tool'] == 'inspect_crop' for a in workflow.propose()))
        self.assertEqual(runtime.valid_classifications(), [])

    def test_identity_and_locator_receive_brief_without_changing_base_result(self):
        runtime = self.detected()
        workflow = runtime.workflow
        action = next(a for a in workflow.propose() if a['tool'] == 'classify_target')
        with patch('TCM_Fusion.lib.workflow.workflow_backend.run',
                   side_effect=lambda argv, **opts: classify_worker(argv[-2:])), \
             patch('classify_crop.call_vlm', return_value=(json.dumps({
                 'label':'图像不可判', 'reason':'目标失焦', 'confidence':'low',
                 'visual_observation':ANSWER}), None)) as call:
            row = workflow.execute({**action,'visual_brief':QUESTION})
        self.assertTrue(row['accepted'])
        self.assertIn(QUESTION, call.call_args.args[-1])
        self.assertEqual(row['visual_observation']['finding'], ANSWER['finding'])
        artifact = workflow.store.current()['artifacts']['classification:' + action['args']['target_id']]
        self.assertNotIn('visual_observation', read_json(artifact['path']))
        self.assertFalse(any(c['identity_verified'] for c in runtime.valid_classifications()))
        ctx = self.root/'tile.json'
        output = self.root/'locate'
        write_json(ctx, {'tool':'locate_feature','roi':[0,0,160,160],'source':str(self.image),
                         'feature_id':'vessel','output':str(output),'visual_brief':QUESTION})
        with patch('detect_core.call_vlm', return_value=(json.dumps({'objects':[], 'visual_observation':FIRST}), None)) as call:
            locate_worker(['--context',str(ctx)])
        self.assertIn(QUESTION, call.call_args.args[-1])
        self.assertEqual(read_json(output/'region_result.json')['count'], 0)
        self.assertEqual(read_json(output/'visual_observation.json')['finding'], FIRST['finding'])

    def test_cnn_only_keeps_separate_visual_action_available(self):
        workflow = self.detected().workflow
        classify = next(a for a in workflow.propose() if a['tool'] == 'classify_target')
        crop_id = classify['args']['target_id']
        pred = {'rows':[{'id':crop_id, 'top1':'vessel', 'top1_prob':.99, 'entropy':.01}]}
        workflow.config['cnn_weights'] = 'fixture'
        workflow.yolo_python = sys.executable
        with patch('TCM_Fusion.lib.vision.stage2_classifier.model_profile', return_value=workflow.cnn_signature), \
             patch('TCM_Fusion.lib.vision.stage2_classifier.request_signature', return_value='fixture'), \
             patch('TCM_Fusion.lib.workflow.workflow_backend._cnn_predictions', return_value=pred), \
             patch('TCM_Fusion.lib.workflow.workflow_backend.run') as worker:
            row = workflow.execute({**classify,'visual_brief':QUESTION})
        self.assertTrue(row['accepted'])
        self.assertEqual(row['visual_observation'], {'brief':QUESTION, 'status':'cnn_only'})
        worker.assert_not_called()
        inspect = next(a for a in workflow.propose() if a['tool'] == 'inspect_crop')
        self.assertEqual(inspect['args']['target_id'], crop_id)
        self.assertTrue(workflow.store.current()['artifacts']['classification:' + crop_id]['identity_verified'])

    def test_per_tile_readings_keep_their_scope_with_empty_boxes(self):
        workflow = self.detected().workflow
        workflow.config.update(feature_scan_tile_policy='full', feature_scan_concurrency=1)
        work = self.root / 'scan'
        work.mkdir()
        target = {'id':'scan', 'source':self.image, 'feature_id':'vessel', 'max_tiles':2, 'visual_brief':QUESTION}
        replies = [(json.dumps({'objects':[], 'visual_observation':FIRST}),None),
                   (json.dumps({'objects':[]}),None)]
        with patch('TCM_Fusion.lib.workflow.workflow_backend.run',
                   side_effect=lambda argv, **opts: locate_worker(argv[-2:])), \
             patch('detect_core.call_vlm', side_effect=replies) as call:
            result = workflow.backend.execute(workflow, 'feature_scan', target, work, work/'context.json')
        self.assertEqual(call.call_count, 2)
        self.assertTrue(all(QUESTION in entry.args[-1] for entry in call.call_args_list))
        self.assertEqual(read_json(result['completion'])['count'], 0)
        detail = read_json(result['visual_observation_file'])
        self.assertEqual([r['status'] for r in detail['tiles']], ['reported','unreported'])
        self.assertEqual([r['roi'] for r in detail['tiles']], result['tiles'])
        self.assertNotEqual(*result['tiles'])

    def test_selected_scan_passes_menu_hint_and_agent_question_to_each_actual_tile(self):
        preset = '检查边缘与背景的交界。'
        question = (
            '请沿每个图块内能看清的颗粒边缘观察，从亮度变化较缓的一侧追到边线交会处，说明在什么位置能连续看清外缘、哪里被相邻碎片遮挡。'
            '如果似乎有两层轮廓，请比较两条线的走向、间距以及它们在转角处是否一起改变方向，并说明判断依据。'
            '遇到深色条纹经过颗粒时，请区分条纹是止于边缘、穿过边缘还是因失焦无法追踪，只描述该块实际可见的部分。'
            '图块边缘如有被截断的结构，也请指出它在图内的位置，以及截断会影响哪一项观察。遇到反光，请说明仍可辨认的边线和不确定区域。'
            '最后描述最清楚的一个转角两侧分别连接到什么轮廓。如果没有可判读的转角，请说明限制来自背景重叠、失焦还是图块范围。')
        self.assertGreater(len(question), 240)
        runtime = self.detected(fast_recovery_selection='per_feature', fast_fallback_max_features=4,
                                feature_scan_max_tiles=2, feature_scan_tile_policy='full', feature_scan_concurrency=1,
                                feature_profiles={'single_crystal':{'positive':preset}})
        runtime.investigate(['G3'], '模拟候选调查', runtime.score([])['evidence_digest'])
        menus = []
        prompts = []

        def plan(payload, timeout):
            menu = json.loads(payload['messages'][1]['content'])
            menus.append(menu)
            if len(menus) == 1:
                aid = next(a['action_id'] for a in menu['actions'] if a['tool'] == 'feature_scan'
                           and menu['action_state'][a['action_id']]['feature_id'] == 'single_crystal')
                self.assertIn(preset, menu['scan_feature_hints']['single_crystal'])
                self.assertEqual(len(menu['action_state'][aid]['candidate_regions_px']), 2)
                return envelope({'decision':{'kind':'action','action_id':aid,'visual_briefs':{aid:question}},
                                 'decision_summary':'先调查同一视野的竞争判据。'})
            aid = next(a['action_id'] for a in menu['actions'] if a['tool'] == 'request_review')
            return envelope({'decision':{'kind':'action','action_id':aid}})

        def vision(request, timeout):
            content = json.loads(request.data)['messages'][0]['content']
            prompts.append(next(part['text'] for part in content if part['type'] == 'text'))
            return io.BytesIO(json.dumps(envelope({'objects':[], 'visual_observation':ANSWER})).encode())

        with patch('TCM_Fusion.lib.runtime.hard_calls.post', side_effect=plan), \
             patch('TCM_Fusion.lib.workflow.workflow_backend.run', side_effect=lambda argv, **opts: locate_worker(argv[-2:])), \
             patch('urllib.request.urlopen', side_effect=vision):
            trace = asyncio.run(self.run_through_mcp(runtime, max_steps=3))
        self.assertEqual(trace['planning_requests'], 2)
        self.assertEqual(len(prompts), 2)
        thought = next(r for r in trace['events'] if r['phase'] == 'thought')
        self.assertEqual(list(thought['visual_briefs'].values()), [question])
        for prompt in prompts:
            self.assertIn(question, prompt)
            self.assertIn(preset, prompt)
            self.assertNotIn('先调查同一视野的竞争判据', prompt)
        feedback = next(r['visual_observation'] for r in menus[1]['recent_action_feedback'] if r.get('visual_observation'))
        self.assertEqual(feedback['brief'], question)
        self.assertEqual([tile['finding'] for tile in feedback['tiles']], [ANSWER['finding']]*2)

    def test_raw_observation_replay_requires_the_recorded_brief(self):
        action = {'action_id':'test', 'visual_brief':QUESTION}
        result = {'ok':True,'present':False,'conf':.9}
        record = {'action':action,'accepted':True,'result':result,
                  'visual_observation':receipt(QUESTION, {'visual_observation':ANSWER})}
        recorded = RecordedObservations([record])
        response, _ = recorded.call(None, action, self.image)
        self.assertEqual(response['visual_observation'], record['visual_observation'])
        self.assertNotIn('visual_observation', result)
        with self.assertRaises(IntegrationError) as exc:
            recorded.call(None, {**action, 'visual_brief':'另一个问题'}, self.image)
        self.assertEqual(exc.exception.code, 'REPLAY_MISS')

    def test_invalid_metadata_preserves_eligible_action_and_report_fields(self):
        actions = [{'action_id':'a','tool':'classify_target'}, {'action_id':'b','tool':'measure_instances'}]
        for requested, mode, rejected in ((None, 'unspecified', 0), ({'a':{'use':'preset'}}, 'explicit_preset', 0),
                                          ({'a':{'use':'something else'}}, 'invalid', 1)):
            self.assertEqual(resolve_guidance(requested, actions), ({}, {'a':mode}, rejected))
        self.assertEqual(selected_briefs({'a':QUESTION, 'b':QUESTION, 'none':QUESTION}, actions), ({'a':QUESTION},2))
        for invalid in ('', '图内\n细节', {'path':'anything'}, 'API_KEY=value'):
            self.assertEqual(selected_briefs({'a':invalid}, actions), ({},1))
        kb = SimpleNamespace(herbs={'白芍':{}}, groups={'G2':{}})
        self.assertEqual(selected_briefs({'a':'按白芍的结论检查'}, actions, kb), ({},1))
        self.assertEqual(receipt(QUESTION, {'visual_observation':{'finding': True, 'readability':'clear'}}),
                         {'status':'unreported','brief':QUESTION})

    def test_visual_brief_crosses_former_boundary_and_validates_the_whole_question(self):
        action = {'action_id':'a', 'tool':'classify_target'}
        kb = SimpleNamespace(herbs={'白芍':{}}, groups={'G2':{}})
        for length in (239, 240, 241, 1024):
            with self.subTest(length=length):
                question = '纹' * (length-1) + '？'
                self.assertEqual(resolve_guidance({'a':question}, [action], kb), ({'a':question}, {'a':'agent_brief'}, 0))
                validate_brief({**action, 'visual_brief':question}, kb)
                self.assertIn(question, prompt_suffix(question))
        for tail in ('白芍', ' API_KEY=fixture', ' D:/private/sample.png'):
            question = '纹' * 241 + tail
            self.assertEqual(resolve_guidance({'a':question}, [action], kb), ({}, {'a':'invalid'}, 1))
            with self.assertRaises(IntegrationError):
                validate_brief({**action, 'visual_brief':question}, kb)
