"""组合临时测试目录、阶段替身和可控候选场景，供工作流及跨组件用例复用。替身写出最小 crop、分割与测量工件，使调用方能够验证真实契约和提交。

主要定义：
- StageFixture / D3StageFixture 提供检测、分割、复核和测量工件。
- WorkflowFixture 及其子类组装各类调查场景。

关联：
- 调用方 「tests/agent/test_investigation_control.py」、「tests/agent/test_planning_choices.py」、「tests/agent/test_reflection_strategy.py」、「tests/application/test_lifecycle.py」、「tests/decision/test_candidate_audit.py」、「tests/decision/test_final_assessment.py」。
- 直接依赖 「lib/agent/react_graph.py」、「lib/agent/reflection_controller.py」、「lib/agent/semantic_execution.py」、「lib/runtime/storage.py」、「lib/tools/agent_tools.py」、「lib/workflow/workflow_runtime.py」。
- 统一执行见 tests/README.md。
"""
import asyncio
import json
import tempfile
from pathlib import Path

import contracts

from TCM_Fusion.lib.runtime.storage import IntegrationError, read_json, write_json
from TCM_Fusion.lib.workflow.workflow_runtime import WorkflowRuntime
from TCM_Fusion.lib.tools.agent_tools import LocalToolbox, FixtureBackend
from TCM_Fusion.lib.agent.react_graph import run_react
from TCM_Fusion.lib.agent.reflection_controller import ReflectionController
from TCM_Fusion.lib.agent.semantic_execution import options


class StageFixture:
    def __init__(self, scale=2, fail=None, classification_feature=None):
        self.scale=scale;self.fail=fail;self.calls=[];self.classification_feature=classification_feature
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
            meta=contracts.CropMeta.from_json(target['path'])
            feature=self.classification_feature or meta.feature_id
            write_json(work/'classification/classification.json',
                       {'label':'纤维' if feature=='fiber' else meta.object_label,
                        'feature_id':feature,'confidence':'high','reason':'模拟复核',
                        'granularity':'instance','source':'vlm','crop_id':target.get('crop_id')})
            return {'classification':str(work/'classification')}
        if tool=='measure_instances':
            row={'image_stem':image.stem,'instance_id':1,'feature':'single_crystal','calibration_status':'ok',
                 'calibration_px_per_um':2,'primary_dimension':'feret_max_um','primary_value_um':55,'conf':.9,'flags':[]}
            f=work/'result_measurements.jsonl';f.write_text(json.dumps(row)+'\n','utf-8')
            return {'measurements':str(f)}
        raise AssertionError(tool)


class D3StageFixture(StageFixture):
    """显式模拟几何返回宽度；原StageFixture仅有长度，用于缺量纲反例。"""
    def execute(self,s,tool,target,work,context):
        result=super().execute(s,tool,target,work,context)
        if tool=='measure_instances':
            p=Path(result['measurements']);row=json.loads(p.read_text(encoding='utf-8'))
            row.update(mean_width_um=5.5,area_um2=30)
            p.write_text(json.dumps(row)+'\n',encoding='utf-8')
        return result


class WorkflowFixture:
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        self.image=self.root/'fov.png';self.image.write_bytes(b'simulation-fov')

    def runtime(self,backend=None,**config):
        return WorkflowRuntime([self.image],root=self.root/'task',sample_id='sample',a={},
            workflow_config={'max_recompute':8,**config},simulation=True,backend=backend or StageFixture())


class InstanceFixture(WorkflowFixture):
    def detected(self, backend=None):
        r = self.runtime(backend)
        action = next(a for a in r.workflow.propose() if a['tool'] == 'detect_fov_initial')
        self.assertEqual(r.workflow.execute(action)['status'], 'done')
        crop = next(t for t in r.workflow.targets().values() if t['kind'] == 'crop')
        return r, crop

    def alias(self, r, crop, *, name='renamed', box=None, pixels=None):
        doc = read_json(crop['path']); doc['crop_id'] = name
        doc['png'] = str(Path(crop['path']).parent/'crop.png')
        if box is not None:
            doc['tight_box'] = box; doc['loose_box'] = box
        if pixels is not None:
            path = self.root/(name+'.png'); path.write_bytes(pixels); doc['png'] = str(path)
        side = self.root/(name+'.json'); write_json(side, doc)
        cid = 'crop:'+crop['image_id']+':'+name
        art = r.workflow._art(cid, 'crop', side, crop['parents'], image_id=crop['image_id'])
        r.workflow.store.commit(r.workflow.store.current()['version'], [art])
        return r.workflow.targets()[cid]


class D3Fixture(WorkflowFixture):
    def detected(self, backend=None, **config):
        r = self.runtime(backend=backend or D3StageFixture(), agent_policy='adaptive_v2', max_planning=20,
                         max_resource_units=200, **config)
        r.workflow.execute(next(a for a in r.workflow.propose() if a['tool']=='detect_fov_initial'))
        return r

    def state(self):
        return {'observations': [], 'attempted': [], 'events': []}

    def action(self, r, tool):
        return next(a for a in r.workflow.propose() if a['tool']==tool)

    def goal(self, r):
        option = options(r, r.workflow.propose())[0]
        return {'semantic_task':'measure_width','instance_ref':option['instance_ref'],'problem_id':'width_question'}

    def run_agent(self, r, planner, **opts):
        tools = LocalToolbox(r, ledger_path=self.root/'observations.json', backend=FixtureBackend())
        return asyncio.run(run_react(r, tools, planner=planner, planner_mode='scripted',
                                     max_steps=opts.pop('max_steps',12), **opts))

    def controller(self):
        r=self.detected();c=ReflectionController(r)
        action=r.action('read_scale',{},'evidence_inspection')
        for _ in range(2): c.record(action,progress=False,feedback={'problem_id':'same_question','accepted':True},
                                  evidence_version=r.workflow.store.current()['version'])
        card=c.propose({'no_gain':2})
        return r,c,action,card


class FairInvestigationFixture(WorkflowFixture):
    def prepare(self):
        r = self.runtime(agent_policy='adaptive_v2', semantic_execution=True,
                         fast_recovery_selection='per_feature', targeted_recovery=True)
        r.a = {'status':'ok', 'fold_rankings':[[{'label':'白芍','probability':.8},
            {'label':'郁金','probability':.12},{'label':'陈皮','probability':.08}]]*4 +
            [[{'label':'玄明粉','probability':.9833},{'label':'郁金','probability':.0167}]]}
        r.a['fold_rankings'][2]=[{'label':'白芍','probability':.8},
            {'label':'郁金','probability':.12},{'label':'牡丹皮','probability':.08}]
        return r
