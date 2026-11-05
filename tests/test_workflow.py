"""受信插件、证据CAS、资源幂等与原图启动闭环；视觉响应为明确替身。"""
import asyncio
import copy
import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import contracts
import paths
from TCM_Fusion.lib.storage import IntegrationError, digest, file_hash, read_json, write_json
from TCM_Fusion.lib.tool_registry import ToolSpec, validate_arguments
from TCM_Fusion.lib.evidence_store import EvidenceStore
from TCM_Fusion.lib.workflow_runtime import WorkflowRuntime
from TCM_Fusion.lib.agent_tools import LocalToolbox, FixtureBackend
from TCM_Fusion.lib.react_graph import run_react
from TCM_Fusion.lib.artifact_index import ArtifactIndex
from TCM_Fusion.lib.resource_budget import install_http_guard


class StageFixture:
    def __init__(self, scale=2, fail=None):self.scale=scale;self.fail=fail;self.calls=[]
    def execute(self,s,tool,target,work,context):
        self.calls.append(tool)
        if tool==self.fail:raise IntegrationError('SIMULATED_FAILURE','故障注入')
        image=next(iter(s.runtime.images.values()))['path']
        if tool in {'detect_fov_initial','detect_region'}:
            folder=work/'detection/crops';folder.mkdir(parents=True)
            (folder/'crop.png').write_bytes(b'simulation-crop')
            side=contracts.crop_sidecar('crop','crop.png',str(image),image.stem,'single_crystal','方晶','',
                self.scale,[0,0,8,8],[0,0,8,8],[0,0,8,8],'8x8')
            write_json(folder/'crop.json',side);write_json(work/'done.json',{'count':1})
            return {'crops_dir':str(folder),'completion':str(work/'done.json')}
        if tool=='calibrate_fov':
            write_json(work/'scale.json',{'status':'OK','px_per_um':2})
            return {'scale':str(work/'scale.json')}
        if tool=='segment_instance':
            folder=work/'segmentation/masks';folder.mkdir(parents=True)
            mask=folder/(image.stem+'__single_crystal__inst01.png');mask.write_bytes(b'simulation-mask')
            write_json(folder.parent/'instances.json',{'instances':[{'igm_status':'ok'}]})
            return {'segmentation':str(folder.parent),'quality':'valid'}
        if tool=='classify_target':
            write_json(work/'classification/classification.json',
                       {'label':'纤维','feature_id':'fiber','confidence':'high','reason':'模拟复核',
                        'granularity':'instance','source':'vlm','crop_id':target.get('crop_id')})
            return {'classification':str(work/'classification')}
        if tool=='measure_instances':
            row={'image_stem':image.stem,'instance_id':1,'feature':'single_crystal','calibration_status':'ok',
                 'calibration_px_per_um':2,'primary_dimension':'feret_max_um','primary_value_um':55,'conf':.9,'flags':[]}
            f=work/'result_measurements.jsonl';f.write_text(json.dumps(row)+'\n','utf-8')
            return {'measurements':str(f)}
        raise AssertionError(tool)


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        self.image=self.root/'fov.png';self.image.write_bytes(b'simulation-fov')

    def runtime(self,backend=None,**config):
        return WorkflowRuntime([self.image],root=self.root/'task',sample_id='sample',a={},
            workflow_config={'max_recompute':8,**config},simulation=True,backend=backend or StageFixture())

    def test_nested_jsonschema_and_external_ref(self):
        schema={'type':'object','required':['x'],'additionalProperties':False,'properties':{
            'x':{'type':'array','items':{'type':'integer','minimum':1},'minItems':2}}}
        validate_arguments(schema,{'x':[1,2]})
        for obj in ({'x':[0,2]},{'x':[True,2]},{'x':[1]}, {'x':[1,2],'shell':'bad'}):
            with self.assertRaises(IntegrationError):validate_arguments(schema,obj)
        with self.assertRaises(IntegrationError):validate_arguments({'$ref':'https://example.com/schema'}, {})

    def test_independent_handler_precondition_and_output(self):
        r=self.runtime();reg=r.tools
        reg.register_handler('trusted_test',lambda r,a,s:{'value':a['n']*2})
        reg.register_precondition('positive_test',lambda r,a,s:a['n']>0)
        reg.register(ToolSpec('trusted_test','read','1',{'type':'object','required':['n'],'properties':{'n':{'type':'integer'}}},
            {'type':'object','required':['value'],'properties':{'value':{'type':'integer'}}},
            ('evidence_inspection',),'read',False,'trusted_test','positive_test'))
        for row in r.skills.registry:
            if row['id']=='evidence_inspection':row['tools'].append('trusted_test')
        self.assertEqual(reg.read(r,'trusted_test',{'n':2},{}),{'value':4})
        with self.assertRaises(IntegrationError):reg.read(r,'trusted_test',{'n':-1},{})
        with self.assertRaises(IntegrationError):reg.register(ToolSpec('unbound','read','1',{}, {},(), 'read',False,'arbitrary.import'))

    def test_registry_proposer_is_not_hardcoded(self):
        r=self.runtime();r.tools.register_proposer('test_proposals',lambda r,s:[r.action('trusted_query',{},'evidence_inspection')])
        r.tools.register_handler('trusted_query',lambda r,a,s:{})
        r.tools.register(ToolSpec('trusted_query','read','1',{'type':'object'}, {'type':'object'},
            ('evidence_inspection',),'read',False,'trusted_query',proposal_id='test_proposals'))
        for row in r.skills.registry:
            if row['id']=='evidence_inspection':row['tools'].append('trusted_query')
        self.assertTrue(any(a['tool']=='trusted_query' for a in r.tools.propose(r,{})))

    def test_raw_images_bootstrap_actual_langgraph(self):
        backend=StageFixture();r=self.runtime(backend);tools=LocalToolbox(r,ledger_path=self.root/'observations.json',backend=FixtureBackend())
        result=asyncio.run(run_react(r,tools,planner_mode='heuristic',max_requests=12,max_steps=12))
        self.assertEqual(backend.calls[:3],['detect_fov_initial','segment_instance','measure_instances'])
        self.assertTrue(r.workflow.store.current()['records']);self.assertTrue(any(e['phase']=='recompute' for e in result['events']))
        self.assertEqual(self.image.read_bytes(),b'simulation-fov')

    def test_label_conflict_offers_classification_and_relabels_crop(self):
        """观察到fiber与crop类别single_crystal相左：应提供复核，并把结论落成可重分割的新crop。"""
        backend=StageFixture();r=self.runtime(backend)
        detect=next(a for a in r.workflow.propose() if a['tool']=='detect_fov_initial')
        r.workflow.execute(detect)
        image_id=next(iter(r.images))
        r.score([{'accepted':True,'action':{'tool':'presence_check',
                                            'args':{'image_id':image_id,'feature_id':'fiber'}},
                  'result':{'present':True,'conf':0.9,'reason':'模拟观察'}}])
        offered=[a for a in r.workflow.propose() if a['tool']=='classify_target']
        self.assertEqual(len(offered),1,'类别冲突应恰好提供一次复核动作')
        record=r.workflow.execute(offered[0])
        self.assertEqual(record['status'],'done')
        arts=r.workflow.store.current()['artifacts']
        relabeled=[k for k,v in arts.items() if v['valid'] and v['kind']=='crop' and '__as_fiber' in k]
        self.assertEqual(len(relabeled),1,'复核出不同类别应产出可重分割的新crop')
        meta=contracts.CropMeta.from_json(arts[relabeled[0]]['path'])
        original=contracts.CropMeta.from_json(arts['crop:'+image_id+':crop']['path'])
        self.assertEqual((meta.feature_id,meta.object_label),('fiber','纤维'))
        self.assertEqual(Path(meta.png).resolve(),Path(original.png).resolve(),'只改类别，不复制或改写像素')
        self.assertTrue(any(a['tool']=='segment_instance' and a['args']['target_id']==relabeled[0]
                            for a in r.workflow.propose()),'新crop应被提议重新分割')
        self.assertEqual(backend.calls.count('classify_target'),1)

    def test_missing_scale_is_repaired_before_segmentation(self):
        backend=StageFixture(scale=None);r=self.runtime(backend)
        tools=LocalToolbox(r,ledger_path=self.root/'observations.json',backend=FixtureBackend())
        asyncio.run(run_react(r,tools,planner_mode='heuristic',max_requests=12,max_steps=12))
        self.assertEqual(backend.calls[:4],['detect_fov_initial','calibrate_fov','segment_instance','measure_instances'])

    def test_segment_failure_is_visible_and_not_retried(self):
        backend=StageFixture(fail='segment_instance');r=self.runtime(backend)
        tools=LocalToolbox(r,ledger_path=self.root/'observations.json',backend=FixtureBackend())
        result=asyncio.run(run_react(r,tools,planner_mode='heuristic',max_steps=10))
        self.assertTrue(result['needs_review']);self.assertEqual(backend.calls.count('segment_instance'),1)
        self.assertEqual(r.workflow.store.current()['records'],[])
        self.assertTrue(any(v.get('error_code')=='SIMULATED_FAILURE' for v in read_json(r.workflow.operations)['entries'].values()))

    def test_service_404_blocks_segmentation_without_spatial_repair(self):
        class ServiceFailure(StageFixture):
            def execute(self,s,tool,target,work,context):
                if tool=='segment_instance':
                    self.calls.append(tool);raise IntegrationError('UPSTREAM_HTTP_404','simulated service failure')
                return super().execute(s,tool,target,work,context)
        backend=ServiceFailure();r=self.runtime(backend)
        tools=LocalToolbox(r,ledger_path=self.root/'service_obs.json',backend=FixtureBackend())
        trace=asyncio.run(run_react(r,tools,planner_mode='heuristic',max_steps=10))
        self.assertEqual(backend.calls,['detect_fov_initial','segment_instance'])
        self.assertEqual(trace['stop_reason'],'workflow_service_unavailable');self.assertTrue(trace['needs_review'])
        self.assertEqual(r.workflow.blocked_tools(),{'segment_instance':'UPSTREAM_HTTP_404'})
        self.assertFalse(any(k.startswith('roi:auto:') for k in r.workflow.targets()))
        self.assertEqual(r.summary({'observations':[]})['workflow']['blocked_tools'],r.workflow.blocked_tools())
        # 新目标、新证据版本也不能越过同一任务内的服务熔断。
        crop=next(v for v in r.workflow.targets().values() if v.get('kind')=='crop')
        raw=read_json(crop['path']);raw.update(crop_id='second',png=str(Path(crop['path']).parent/'crop.png'))
        file=self.root/'second.json';write_json(file,raw)
        cid='crop:'+crop['image_id']+':second'
        art=r.workflow._art(cid,'crop',file,crop['parents'],image_id=crop['image_id'])
        snap=r.workflow.store.commit(r.workflow.store.current()['version'],[art])
        action=r.action('segment_instance',{'target_id':cid,'evidence_version':snap['version']},'workflow_repair')
        with self.assertRaises(IntegrationError) as caught:r.tools.invoke(r,action)
        self.assertEqual(caught.exception.code,'TOOL_SERVICE_BLOCKED')
        self.assertEqual(backend.calls.count('segment_instance'),1)
        self.assertFalse(any(a['tool']=='segment_instance' for a in r.workflow.propose()))
        fresh=WorkflowRuntime([self.image],root=self.root/'fresh',workflow_config={},sample_id='fresh',
                              simulation=True,backend=StageFixture())
        self.assertEqual(fresh.workflow.blocked_tools(),{})

    def test_empty_mask_still_proposes_local_image_repair(self):
        class EmptyMask(StageFixture):
            def execute(self,s,tool,target,work,context):
                if tool=='segment_instance':raise IntegrationError('EMPTY_MASK','simulated empty mask')
                return super().execute(s,tool,target,work,context)
        r=self.runtime(EmptyMask())
        r.tools.invoke(r,r.workflow.propose()[0]);r.tools.invoke(r,r.workflow.propose()[0])
        self.assertEqual(r.workflow.blocked_tools(),{})
        self.assertTrue(any(a['tool']=='detect_region' for a in r.workflow.propose()))

    def test_unknown_worker_failure_is_not_guessed_as_a_bad_region(self):
        class UnknownFailure(StageFixture):
            def execute(self,s,tool,target,work,context):
                if tool=='segment_instance':raise IntegrationError('WORKER_FAILED','unknown failure')
                return super().execute(s,tool,target,work,context)
        r=self.runtime(UnknownFailure())
        r.tools.invoke(r,r.workflow.propose()[0]);r.tools.invoke(r,r.workflow.propose()[0])
        self.assertEqual(r.workflow.blocked_tools(),{})
        self.assertFalse(any(k.startswith('roi:auto:') for k in r.workflow.targets()))
        self.assertTrue(r.workflow.unresolved())

    def test_review_mask_keeps_image_repair_available(self):
        class ReviewMask(StageFixture):
            def execute(self,s,tool,target,work,context):
                value=super().execute(s,tool,target,work,context)
                if tool=='segment_instance':value['quality']='review'
                return value
        r=self.runtime(ReviewMask())
        r.tools.invoke(r,r.workflow.propose()[0]);result=r.tools.invoke(r,r.workflow.propose()[0])
        self.assertTrue(result['accepted']);self.assertFalse(result['repair_completed'])
        self.assertTrue(any(a['tool']=='detect_region' for a in r.workflow.propose()))

    def test_recompute_budget_zero_never_calls_backend(self):
        backend=StageFixture();r=self.runtime(backend,max_recompute=0,max_initializations=0)
        tools=LocalToolbox(r,ledger_path=self.root/'observations.json',backend=FixtureBackend())
        result=asyncio.run(run_react(r,tools,planner_mode='heuristic',max_steps=3))
        self.assertEqual(backend.calls,[]);self.assertTrue(result['needs_review'])

    def test_operation_cache_does_not_execute_twice(self):
        backend=StageFixture();r=self.runtime(backend);action=r.workflow.propose()[0]
        first=r.tools.invoke(r,action);second=r.tools.invoke(r,action)
        self.assertTrue(first['accepted']);self.assertTrue(second['cache_hit']);self.assertEqual(len(backend.calls),1)

    def test_recorded_workflow_replay_has_no_new_image_calls(self):
        from TCM_Fusion.lib.workflow_replay import replay_report,RecordedStages
        r=self.runtime();tools=LocalToolbox(r,ledger_path=self.root/'obs.json',backend=FixtureBackend(),max_requests=100)
        trace=asyncio.run(run_react(r,tools,planner_mode='heuristic',max_requests=100,max_steps=100))
        operations=read_json(r.workflow.operations)['entries']
        report={'route':'workflow_agent','a':{},'sample_id':'sample','workflow_options':r.workflow.config,
            'agent':{'operations':operations,'observations':trace['observations']},
            'b':{'verdict':{**trace['decision'],'source_images':[str(self.image)]}}}
        value=asyncio.run(replay_report(report,self.root/'replay'))
        self.assertTrue(value['covered']);self.assertTrue(value['matches_source_verdict'])
        self.assertEqual(value['new_image_api_calls'],0)
        missing=RecordedStages([])
        with self.assertRaises(IntegrationError):missing.execute(r.workflow,'detect_fov_initial',
            next(iter(r.workflow.targets().values())),self.root/'unused',None)
        self.assertEqual(missing.misses,1)

    def test_store_rejects_cycles_and_tampering(self):
        store=EvidenceStore(self.root/'evidence','sample');p=self.root/'p';p.write_bytes(b'data')
        with self.assertRaises(IntegrationError):store.commit(store.current()['version'],[
            {'id':'a','path':str(p),'sha256':file_hash(p),'parents':['a']}])
        store.commit(store.current()['version'],[{'id':'a','path':str(p),'sha256':file_hash(p),'parents':[]}])
        p.write_bytes(b'changed')
        with self.assertRaises(IntegrationError):store.verified_records()

    def test_failed_and_crop_index_has_no_truth(self):
        root=self.root/'b';(root/'detection/crops').mkdir(parents=True)
        write_json(root/'job.json',{'status':'failed','error_code':'EMPTY_MASK','secret':'not-for-agent'})
        write_json(root/'detection/crops/crop.json',{'source_image_path':str(self.image),'herb':'truth'})
        result=ArtifactIndex([root]).scan();text=json.dumps(result,ensure_ascii=False)
        self.assertIn('EMPTY_MASK',text);self.assertNotIn('not-for-agent',text);self.assertNotIn('truth',text)
        self.assertTrue(any(a['status']=='missing_pair' for a in result['artifacts']))

    def test_store_cas_invalidation_and_immutable_history(self):
        store=EvidenceStore(self.root/'evidence','sample');p=self.root/'p';p.write_text('first');m=self.root/'mask';m.write_text('mask')
        initial=store.current()['version'];one=store.commit(initial,[{'id':'crop','path':str(p),'sha256':file_hash(p),'parents':[]}])
        two=store.commit(one['version'],[{'id':'mask','path':str(m),'sha256':file_hash(m),'parents':['crop']}])
        new=self.root/'new';new.write_text('second')
        three=store.commit(two['version'],[{'id':'crop','path':str(new),'sha256':file_hash(new),'parents':[]}])
        self.assertFalse(three['artifacts']['mask']['valid']);self.assertTrue((store.root/'versions'/(one['version']+'.json')).is_file())
        with self.assertRaises(IntegrationError):store.commit(one['version'],[])

    def test_invalidation_traverses_resubmitted_unchanged_sidecar(self):
        store=EvidenceStore(self.root/'store','sample')
        pixel=self.root/'pixel';pixel.write_bytes(b'old')
        side=self.root/'side';side.write_bytes(b'same-sidecar')
        mask=self.root/'mask';mask.write_bytes(b'mask')
        artifacts=[{'id':'pixels','path':str(pixel),'sha256':file_hash(pixel),'parents':[]},
                   {'id':'crop','path':str(side),'sha256':file_hash(side),'parents':['pixels']},
                   {'id':'mask','path':str(mask),'sha256':file_hash(mask),'parents':['crop']}]
        first=store.commit(store.current()['version'],artifacts)
        new=self.root/'newpixel';new.write_bytes(b'changed')
        second=store.commit(first['version'],[
            {**artifacts[0],'path':str(new),'sha256':file_hash(new)},artifacts[1]])
        self.assertTrue(second['artifacts']['crop']['valid'])
        self.assertFalse(second['artifacts']['mask']['valid'])

    def test_network_guard_counts_attempt_and_blocks_retry(self):
        import urllib.request
        budget=self.root/'budget.json';ctx=self.root/'ctx.json'
        write_json(budget,{'identity':'test','deadline':10**12,'used':{},'entries':{}})
        write_json(ctx,{'identity':'test','budget_path':str(budget),'operation_key':'op','limits':{'network':2,'total':2}})
        original=urllib.request.urlopen;redirect=urllib.request.HTTPRedirectHandler.redirect_request
        try:
            with patch('urllib.request.urlopen',side_effect=OSError('network failure')) as remote:
                install_http_guard(ctx)
                with self.assertRaises(OSError):urllib.request.urlopen('https://example.com')
                with self.assertRaises(IntegrationError):urllib.request.urlopen('https://example.com')
                self.assertEqual(remote.call_count,1)
            self.assertEqual(read_json(budget)['used']['network'],1)
        finally:urllib.request.urlopen=original;urllib.request.HTTPRedirectHandler.redirect_request=redirect

    def test_real_worker_process_budget_and_timeout(self):
        from TCM_Fusion.lib.controlled_process import run
        budget=self.root/'budget.json';ctx=self.root/'ctx.json'
        write_json(budget,{'identity':'test','deadline':10**12,'used':{},'entries':{}})
        write_json(ctx,{'identity':'test','budget_path':str(budget),'operation_key':'op','limits':{'network':0,'total':0}})
        code="import urllib.request; urllib.request.urlopen('https://example.com')"
        with self.assertRaises(IntegrationError):run([sys.executable,'-c',code],context_path=ctx,timeout=5,log_path=self.root/'worker.log')
        with self.assertRaises(IntegrationError):run([sys.executable,'-c','import time; time.sleep(10)'],context_path=ctx,timeout=.15,log_path=self.root/'timeout.log')

    def test_original_geometry_worker_with_partial_mask_and_real_mcp(self):
        self._geometry_reuse_case(False)

    def test_diagnostic_review_reuses_mask_and_preserves_rule_exclusion(self):
        self._geometry_reuse_case(True)

    def _geometry_reuse_case(self, diagnostic):
        """真正执行原measure.py与stdio；合成图仅用于接口/量纲验证，不联网。"""
        import subprocess
        from TCM_Fusion.lib.pipeline import run_pipeline
        py=Path(sys.executable)
        if not py.is_file():self.skipTest('没有可用的CV解释器')
        probe=subprocess.run([str(py),'-c','import cv2,numpy'],capture_output=True)
        if probe.returncode:self.skipTest('CV解释器缺numpy/opencv；其它回归仍运行')
        subprocess.run([str(py),'-c',
            "import cv2,numpy as np,sys; a=np.full((64,64,3),220,np.uint8); m=np.zeros((64,64),np.uint8); cv2.circle(m,(32,32),10,255,-1); cv2.imwrite(sys.argv[1],a); cv2.imwrite(sys.argv[2],m)",
            str(self.image),str(self.root/'mask.png')],check=True,capture_output=True)
        original=self.root/'original_b';crop=original/'detection/crops';crop.mkdir(parents=True)
        import shutil
        shutil.copy2(self.image,crop/'crop.png')
        side=contracts.crop_sidecar('crop','crop.png',str(self.image),'fov','single_crystal','方晶','',2,
                                   [20,20,44,44],[20,20,44,44],[0,0,64,64],'64x64')
        write_json(crop/'crop.json',side)
        masks=original/'segmentation/masks';masks.mkdir(parents=True)
        name='fov__single_crystal__inst01.png';shutil.copy2(self.root/'mask.png',masks/name)
        raw={k:'' for k in contracts.MANIFEST_COLUMNS};raw.update(instance_id=1,inst_mask=name,crop_id='crop',
             source_image_path=str(self.image),feature_id='single_crystal',object_label='方晶',
             px_per_um='2',igm_status='review' if diagnostic else 'ok',granularity='instance',tight_bbox='[20,20,44,44]')
        with (masks/('fov'+contracts.MANIFEST_SUFFIX)).open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=contracts.MANIFEST_COLUMNS);w.writeheader();w.writerow(raw)
        shutil.copytree(crop,masks/'crops')  # 原B交付的同源侧车副本不应造成工件ID冲突。
        before={str(p):file_hash(p) for p in original.rglob('*') if p.is_file()}
        config={'sample_id':'cpu_test','allow_network':False,'a':{'mode':'unavailable','required':False},
                'b':{'mode':'existing','run_dir':str(original),'enable_vlm':False},
                'agent':{'mode':'active','planner':'heuristic','transport':'mcp','image_roots':[str(self.image)],
                         'max_steps':6,'workflow':{'python':str(py),'max_api_calls':0,'max_seconds':15}}}
        if diagnostic:
            config['agent']['workflow'].update(quality_policy='diagnostic_review',enabled_tools=['measure_instances'])
        directory=run_pipeline(config,base=self.root/'runs')
        value=read_json(directory/'fusion.json');self.assertEqual(value['route'],'workflow_agent')
        self.assertTrue(any(o['tool']=='measure_instances' and o['accepted'] for o in value['agent']['operations'].values()))
        snap=read_json(directory/'workflow/evidence/head.json')
        content=read_json(directory/'workflow/evidence/versions'/(snap['version']+'.json'))
        self.assertTrue(content['records']);self.assertEqual(content['records'][0]['data']['calibration_px_per_um'],2)
        self.assertEqual(value['agent']['resources']['used'].get('network',0),0)
        self.assertEqual(before,{str(p):file_hash(p) for p in original.rglob('*') if p.is_file()})
        if diagnostic:
            self.assertTrue(value['diagnostic_only']);self.assertTrue(value['needs_review'])
            self.assertIn('igm_review',content['records'][0]['data']['flags'])
            from rule_engine import pool_evidence
            cards=pool_evidence([r['data'] for r in content['records']])
            self.assertTrue(all(not c.present and c.n_instances==0 for c in cards.values()))
            self.assertEqual(value['agent']['resources']['used'].get('observation',0),0)

    def test_workflow_prompt_dual_replay_and_release(self):
        from TCM_Fusion.lib import prompt_optimization as P
        r=self.runtime(max_planning=30,max_resource_units=200)
        tools=LocalToolbox(r,ledger_path=self.root/'recorded_obs.json',backend=FixtureBackend(),max_requests=100)
        trace=asyncio.run(run_react(r,tools,planner_mode='heuristic',max_requests=100,max_steps=100))
        report={'route':'workflow_agent','simulation':True,'a':{},'sample_id':'sample',
                'workflow_options':r.workflow.config,'agent':{
                    'operations':read_json(r.workflow.operations)['entries'],'observations':trace['observations']},
                'b':{'verdict':{**trace['decision'],'source_images':[str(self.image)]}}}
        candidate={'schema':'prompt-candidate/v1','generator':'unit-test-only','instruction':'choose without extra read',
                   'baseline':P.DEFAULT_INSTRUCTION,'guard_sha256':digest(P.GUARD),
                   'training_groups':['train'],'training_sha256':'fixture','profile':P.profile(),'simulation':True}
        expected={'verdict':trace['decision']['verdict'],'conclusion_id':(trace['decision'].get('conclusion') or {}).get('id')}
        dataset={'split':'development','mode':'workflow','simulation':True,'cases':[{
            'group_id':'development','report':report,'report_sha256':digest(report),'expected':expected}]}
        class Planner:
            def __init__(self,instruction):self.extra=instruction==P.DEFAULT_INSTRUCTION
            def choose(self,context):
                selected=context['actions'][0]
                if self.extra:
                    selected=next(a for a in context['actions'] if a['tool']=='read_evidence');self.extra=False
                return {'action_id':selected['action_id']},None
        with patch('TCM_Fusion.lib.workflow_prompt_validation.output_root',return_value=self.root), \
             patch('TCM_Fusion.lib.prompt_optimization.output_root',return_value=self.root):
            validation=asyncio.run(P.validate(candidate,dataset,planner_factory=Planner))
            self.assertTrue(validation['passed']);self.assertEqual(validation['new_image_api_calls'],0)
            version=asyncio.run(P.publish(candidate,validation,planner_factory=Planner))
            self.assertTrue(P.load_release(version,r)['simulation'])
            truth_data=copy.deepcopy(dataset);truth_data['mode']='workflow_truth'
            truth_data['cases'][0]['reference']={'label':'independent_fixture_truth','label_source':'test_fixture','label_scope':'sample_identity'}
            truth_validation=asyncio.run(P.validate(candidate,truth_data,planner_factory=Planner))
            self.assertEqual(truth_validation['metric'],'paired_truth_with_bounded_cost')
            self.assertIn('outcome',truth_validation['rows'][0]['candidate'])
            bad=copy.deepcopy(dataset)
            ops=bad['cases'][0]['report']['agent']['operations']
            for key in list(ops):
                if ops[key]['tool']=='segment_instance':del ops[key]
            bad['cases'][0]['report_sha256']=digest(bad['cases'][0]['report'])
            self.assertFalse(asyncio.run(P.validate(candidate,bad,planner_factory=Planner))['passed'])

    def test_offline_missing_view_still_requires_review(self):
        r=self.runtime();self.assertTrue(r.workflow.unresolved())
        r.simulation=False;r.workflow.allow_network=False
        self.assertEqual(r.workflow.propose(),[])
        self.assertTrue(r.workflow.unresolved())

    def test_read_scale_exposes_contract_and_version(self):
        r=self.runtime();r.tools.invoke(r,r.workflow.propose()[0])
        result=r.tools.read(r,'read_scale',{}, {})
        self.assertEqual(result['scales'][0]['px_per_um'],2)
        self.assertIn('evidence_version',result['scales'][0])

    def test_initialization_and_api_pools_are_separate_but_globally_bounded(self):
        r=self.runtime(max_recompute=0,max_api_calls=1)
        self.assertTrue(r.tools.invoke(r,r.workflow.propose()[0])['accepted'])
        used=read_json(r.workflow.budget_path)['used']
        self.assertEqual(used.get('initialization'),1);self.assertNotIn('recompute',used)
        r.workflow.charge('planning','plan')
        with self.assertRaises(IntegrationError):r.workflow.charge('observation','observe')
        self.assertEqual(read_json(r.workflow.budget_path)['used']['network'],1)

    def test_shadow_generates_plan_but_refuses_execution(self):
        backend=StageFixture();r=self.runtime(backend,execution='shadow')
        action=r.workflow.propose()[0]
        with self.assertRaises(IntegrationError):r.tools.invoke(r,action)
        self.assertEqual(backend.calls,[])
        from TCM_Fusion.lib.pipeline import run_pipeline,validate_config
        config={'sample_id':'shadow','allow_network':False,
                'a':{'mode':'unavailable','required':False},
                'b':{'mode':'run','input_mode':'images','input':str(self.root),'enable_vlm':False},
                'agent':{'mode':'active','planner':'llm','image_roots':[str(self.image)],
                         'workflow':{'execution':'shadow'}}}
        directory=run_pipeline(config,base=self.root/'runs')
        value=read_json(directory/'fusion.json')
        self.assertEqual(value['route'],'workflow_shadow');self.assertFalse(value['agent']['executed'])
        self.assertEqual(value['agent']['actions'][0]['tool'],'detect_fov_initial')
        config['allow_recorded']=True;config['a']={'mode':'recorded','record':'unused'}
        with self.assertRaises(IntegrationError):validate_config(config)

    def test_transport_failure_retains_valid_state_and_requires_review(self):
        class BrokenTransport:
            async def recompute(self,action):raise RuntimeError('simulated transport loss')
        r=self.runtime()
        result=asyncio.run(run_react(r,BrokenTransport(),planner_mode='heuristic',max_steps=3))
        self.assertTrue(result['needs_review']);self.assertEqual(r.records,[])
        self.assertEqual(result['events'][-1]['error_code'],'TOOL_TRANSPORT_FAILED')

    def test_measure_cache_and_replay_bind_transitive_mask_content(self):
        r=self.runtime()
        for _ in range(2):r.tools.invoke(r,r.workflow.propose()[0])
        target=next(v for v in r.workflow.targets().values() if v.get('kind')=='segmentation')
        before=r.workflow.operation_key('measure_instances',target)
        original=r.tools.invoke(r,r.workflow.propose()[0])
        from TCM_Fusion.lib.workflow_replay import RecordedStages
        recorded=RecordedStages([original])
        snap=r.workflow.store.current();mask=next(v for v in snap['artifacts'].values()
                                                if v['kind']=='segmentation_file' and v['path'].endswith('.png'))
        new=self.root/'changed-mask.png';new.write_bytes(b'new-mask-content')
        replacement={k:v for k,v in mask.items() if k!='valid'}
        replacement.update(path=str(new),sha256=file_hash(new))
        marker={k:v for k,v in snap['artifacts'][target['id']].items() if k!='valid'}
        r.workflow.store.commit(snap['version'],[replacement,marker])
        current=r.workflow.targets()[target['id']]
        self.assertNotEqual(before,r.workflow.operation_key('measure_instances',current))
        self.assertEqual(r.workflow.store.verified_records(),[])
        with self.assertRaises(IntegrationError):recorded.execute(r.workflow,'measure_instances',current,self.root/'unused',None)
        self.assertEqual(recorded.misses,1)
