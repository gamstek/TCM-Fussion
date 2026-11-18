"""验证显微特征分类模型档案、按 crop 内容组装的缓存键及阈值入账。阶段测试使用临时权重元信息和可控推理回复，训练侧变换接口执行真实绑定检查。

主要定义：
- Stage2Tests（模板）: 校验药材 A 保留五折，stage2 权重来自「分类器设计权重与CNN权重」。
- Stage2Tests（配置与资产）: 缺失权重或未配对 cnn_weights/cnn_meta 明确失败；药材标签、重复类名、未知预处理被拒绝。
- Stage2Tests（指纹）: request_signature 随权重、meta 和 crop 内容变化而失效；重复 crop id 被拒绝。
- Stage2Tests（缓存契约）: cache_matches 要求签名、概率与 topk 契约一致，零熵仍可接受。
- Stage2Tests（后端）: 复用有效缓存不发起远端调用；低置信保留候选与 pending_review。
- Stage2Tests（错误隔离）: 模型加载失败报 CNN_ASSET_MISSING，而不是当作低置信。
- Stage2Tests（worker）: 复用 cls_net.build_transforms 与类别顺序，类别反转时报 CNN_META_MISMATCH。

关联：
- 直接依赖 「lib/runtime/storage.py」、「lib/vision/stage2_classifier.py」、「lib/workflow/workflow_backend.py」。
- 统一执行见 tests/README.md。
"""
import copy
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import paths
from TCM_Fusion.lib.vision.stage2_classifier import cache_matches, infer, model_profile, request_signature
from TCM_Fusion.lib.runtime.storage import IntegrationError, read_json, write_json
from TCM_Fusion.lib.workflow.workflow_backend import WorkflowBackend


class Stage2Tests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.weights=self.root/'best.pt';self.weights.write_bytes(b'test-only-not-a-model')
        self.meta=self.root/'model_meta.json'
        self.metadata={'arch':'resnet18','imgsz':32,'input_mode':'letterbox',
                       'classes':['00_fiber','01_other']}
        write_json(self.meta,self.metadata)
        self.cfg={'cnn_weights':str(self.weights),'cnn_meta':str(self.meta)}
        self.image=self.root/'crop.png';self.image.write_bytes(b'fixture-image')
        self.crops=[{'id':'crop1','png':str(self.image)}]

    def signature(self):
        return request_signature(model_profile(self.cfg),self.crops,sys.executable)

    def cached(self):
        return {'signature':self.signature(),'rows':[{'id':'crop1','top1':'fiber',
            'top1_class':'00_fiber','top1_prob':.99,'entropy':0.,'topk':[
                {'class':'00_fiber','feature_id':'fiber','prob':.99},
                {'class':'01_other','feature_id':'other','prob':.01}]}]}

    def test_template_preserves_herb_folds_and_uses_requested_stage2_root(self):
        cfg=read_json(Path(paths.PROJECT_ROOT)/'TCM_Fusion/config/multi_view.json')
        self.assertEqual(cfg['a']['checkpoint_dir'],'MICROSCOPE-main/v3_resnet50_full')
        self.assertEqual(cfg['agent']['candidate_policy']['top_k'],3)
        self.assertTrue(cfg['agent']['workflow']['cnn_weights'].startswith('分类器设计权重与CNN权重/'))

    def test_missing_assets_and_unpaired_config_fail_explicitly(self):
        with self.assertRaises(IntegrationError) as result:model_profile({'cnn_weights':str(self.weights)})
        self.assertEqual(result.exception.code,'CNN_CONFIG_INVALID')
        self.weights.unlink()
        with self.assertRaises(IntegrationError) as result:model_profile(self.cfg)
        self.assertEqual(result.exception.code,'CNN_ASSET_MISSING')

    def test_herb_labels_duplicates_and_unknown_preprocessing_rejected(self):
        for changes in ({'classes':['白芍','龙胆']},{'classes':['00_fiber','01_fiber']},
                        {'input_mode':'guess'},{'imgsz':False}):
            with self.subTest(changes=changes):
                write_json(self.meta,{**self.metadata,**changes})
                with self.assertRaises(IntegrationError):model_profile(self.cfg)

    def test_signature_invalidates_weights_meta_and_crop_content(self):
        before=self.signature();self.weights.write_bytes(b'new-weights')
        self.assertNotEqual(before,self.signature())
        before=self.signature();write_json(self.meta,{**self.metadata,'imgsz':64})
        self.assertNotEqual(before,self.signature())
        before=self.signature();self.image.write_bytes(b'replaced-image')
        self.assertNotEqual(before,self.signature())

    def test_cache_requires_matching_input_and_probability_contract(self):
        value=self.cached();self.assertTrue(cache_matches(value,self.signature()))
        self.assertFalse(cache_matches({'rows':value['rows']},self.signature()))
        for key,val in [('entropy',float('nan')),('top1','other'),('top1_prob',.5),('topk',[])]:
            changed=copy.deepcopy(value);changed['rows'][0][key]=val
            self.assertFalse(cache_matches(changed,self.signature()))
        self.image.write_bytes(b'new-pixels')
        self.assertFalse(cache_matches(value,self.signature()))

    def test_duplicate_crop_ids_rejected(self):
        with self.assertRaises(IntegrationError):
            request_signature(model_profile(self.cfg),self.crops*2,sys.executable)

    def backend_session(self):
        target={'id':'crop1','kind':'crop','path':'fixture-crop.json','batch_cnn_only':True}
        meta=SimpleNamespace(png=self.image,crop_id='crop1',granularity='instance')
        session=SimpleNamespace(config=self.cfg,root=self.root,yolo_python=Path(sys.executable),
            python=Path(sys.executable),cancel_event=None,targets=lambda:{'crop1':target},
            remaining_seconds=lambda:30)
        return session,target,meta

    def test_backend_reuses_valid_cache_and_accepts_zero_entropy(self):
        session,target,meta=self.backend_session()
        write_json(self.root/'cnn/predictions.json',self.cached())
        work=self.root/'op';work.mkdir()
        with patch('TCM_Fusion.lib.workflow.workflow_backend.contracts.CropMeta.from_json',return_value=meta),\
             patch('TCM_Fusion.lib.workflow.workflow_backend.run') as remote:
            out=WorkflowBackend().execute(session,'classify_target',target,work,None)
        remote.assert_not_called()
        doc=read_json(Path(out['classification'])/'classification.json')
        self.assertEqual(doc['feature_id'],'fiber');self.assertEqual(doc['cnn_entropy'],0.)

    def test_low_confidence_batch_preserves_candidate_details_without_glm(self):
        session,target,meta=self.backend_session()
        cached=self.cached();cached['rows'][0]['top1_prob']=.6;cached['rows'][0]['topk'][0]['prob']=.6
        write_json(self.root/'cnn/predictions.json',cached)
        work=self.root/'op';work.mkdir()
        with patch('TCM_Fusion.lib.workflow.workflow_backend.contracts.CropMeta.from_json',return_value=meta),\
             patch('TCM_Fusion.lib.workflow.workflow_backend.run') as remote:
            out=WorkflowBackend().execute(session,'classify_target',target,work,None)
        remote.assert_not_called()
        doc=read_json(Path(out['classification'])/'classification.json')
        self.assertEqual(doc['reject'],'pending_review');self.assertEqual(doc['cnn_score'],.6)
        self.assertEqual(len(doc['cnn_topk']),2)

    def test_load_failure_is_not_low_confidence(self):
        session,target,meta=self.backend_session();self.weights.unlink()
        with patch('TCM_Fusion.lib.workflow.workflow_backend.contracts.CropMeta.from_json',return_value=meta):
            with self.assertRaises(IntegrationError) as result:
                WorkflowBackend().execute(session,'classify_target',target,self.root,None)
        self.assertEqual(result.exception.code,'CNN_ASSET_MISSING')

    def test_worker_reuses_training_transform_and_class_order(self):
        if not importlib.util.find_spec('torch') or not importlib.util.find_spec('torchvision'):
            self.skipTest('此解释器没有可选 Torch；真实 worker 在 YOLO 环境运行')
        import torch
        from PIL import Image
        from 目标检测.lib.cls_net import build_transforms
        Image.new('RGB',(13,7),(80,120,180)).save(self.image)
        def model(batch):
            self.assertEqual(tuple(batch.shape),(1,3,32,32))
            with Image.open(self.image) as image:
                expected=build_transforms(32,'letterbox')[1](image.convert('RGB'))
            self.assertTrue(torch.equal(batch[0],expected))
            return torch.tensor([[5.,0.]])
        loaded={**self.metadata,'names':{}}
        with patch('目标检测.lib.cls_net.load_model',return_value=(model,loaded,torch.device('cpu'))):
            value=infer({'weights':str(self.weights),'meta':str(self.meta),'crops':self.crops})
        self.assertTrue(cache_matches(value,self.signature()))
        self.assertEqual(value['rows'][0]['top1'],'fiber')
        with patch('目标检测.lib.cls_net.load_model',return_value=(model,{**loaded,'classes':list(reversed(loaded['classes']))},'cpu')):
            with self.assertRaises(IntegrationError) as result:
                infer({'weights':str(self.weights),'meta':str(self.meta),'crops':self.crops})
        self.assertEqual(result.exception.code,'CNN_META_MISMATCH')
