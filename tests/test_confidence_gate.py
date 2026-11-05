"""统计门控离线单测；临时构造数据不证明真实分类效果或样品独立性。"""
import copy
import tempfile
import unittest
from pathlib import Path
from TCM_Fusion.lib import confidence_gate as G
from TCM_Fusion.lib.storage import IntegrationError, file_hash, write_json
from TCM_Fusion.lib.routing_policy import route,validate_policy
from TCM_Fusion.lib.collaboration import knowledge_base


class ConfidenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        self.routing={'min_score':.5,'min_margin':.1,'buffer':.0001,'top_k':2}
        self.policy={'frozen':True,'development_groups':['development'], 'development_images':['separate-development-hash'],
                     'routing':self.routing,'alpha':.05,'risk_limit':.3,'minimum_released':2}

    def dataset(self,split,n=4):
        cases=[]
        for i in range(n):
            group=split+str(i);p=self.root/(group+'.png');p.write_bytes(group.encode())
            a={'status':'ok','provenance':'recorded_simulation','signature':{'weights':['weight-hash'],
                'script':'script-hash','python_sha256':'interpreter-hash'},'ranking':[
                {'label':'甘草','probability':.99},{'label':'白芍','probability':.01}]}
            cases.append({'group_id':group,'image':str(p),'image_sha256':file_hash(p),'prediction':a,'truth':'甘草'})
        return {'split':split,'simulation':True,'independence_confirmed':True,'cases':cases}

    def test_temperature_fit_and_model_identity(self):
        fit=G.fit(self.dataset('calibration'))
        self.assertLessEqual(fit['nll_after'],fit['nll_before']);self.assertTrue(fit['simulation'])
        bad=self.dataset('calibration');bad['cases'][1]['prediction']['signature']['weights']=['different']
        with self.assertRaises(IntegrationError):G.fit(bad)

    def test_group_and_image_split_overlap_rejected(self):
        cal=self.dataset('calibration');fit=G.fit(cal);data=self.dataset('test')
        data['cases'][0]['group_id']=cal['cases'][0]['group_id']
        with self.assertRaises(IntegrationError):G.validate(fit,self.policy,data)

    def test_upper_bound_and_no_released_samples(self):
        self.assertIsNone(G.upper_error(0,0,.05))
        self.assertAlmostEqual(G.upper_error(0,100,.05),1-.05**.01)
        self.assertGreater(G.upper_error(3,100,.05),.03)
        self.assertEqual(G.upper_error(4,4,.05),1)

    def test_failed_risk_report_cannot_publish(self):
        fit=G.fit(self.dataset('calibration'));data=self.dataset('test')
        report=G.validate(fit,{**self.policy,'risk_limit':.001},data)
        self.assertFalse(report['passed'])
        with self.assertRaises(IntegrationError):G.publish(report)

    def test_simulation_release_cannot_enable_real_routing(self):
        fit=G.fit(self.dataset('calibration'));data=self.dataset('test',20)
        report=G.validate(fit,self.policy,data);self.assertTrue(report['passed'])
        release=G.publish(report);p=self.root/'release.json';write_json(p,release)
        result=route(data['cases'][0]['prediction'],{'mode':'validated','release':str(p)},knowledge_base())
        self.assertFalse(result['skip']);self.assertEqual(result['reason'],'GATE_SCOPE')

    def test_test_image_change_invalidates_publish_replay(self):
        fit=G.fit(self.dataset('calibration'));data=self.dataset('test',20)
        report=G.validate(fit,self.policy,data);Path(data['cases'][0]['image']).write_bytes(b'changed')
        with self.assertRaises(IntegrationError):G.publish(report)

    def test_missing_artifact_is_fail_closed(self):
        a=self.dataset('test')['cases'][0]['prediction']
        result=route(a,{'mode':'validated','release':str(self.root/'missing.json')},knowledge_base())
        self.assertFalse(result['skip']);self.assertEqual(result['mode'],'validated')
        with self.assertRaises(IntegrationError):validate_policy({'mode':'validated'})

    def test_real_dataset_rejects_fixture_predictions(self):
        data=self.dataset('calibration');data['simulation']=False
        with self.assertRaises(IntegrationError):G.fit(data)

    def real_data(self,split,n=4):
        data=self.dataset(split,n);data['simulation']=False
        for row in data['cases']:row['prediction']['provenance']='original_cli'
        identity=G.model_identity(data['cases'][0]['prediction'])
        data['training_audit']={'verified':True,'reference':'unit-test-only-not-real-data',
            'model_identity':identity,'group_ids':['training-sample'],
            'image_sha256s':['training-image-hash']}
        return data

    def test_training_overlap_audit_is_required_and_checked(self):
        data=self.real_data('calibration');bad=copy.deepcopy(data);bad.pop('training_audit')
        with self.assertRaises(IntegrationError):G.fit(bad)
        bad=copy.deepcopy(data);bad['training_audit']['group_ids'].append(data['cases'][0]['group_id'])
        with self.assertRaises(IntegrationError):G.fit(bad)
        fit=G.fit(data);test=self.real_data('test',20)
        test['training_audit']['reference']='changed-training-audit'
        with self.assertRaises(IntegrationError):G.validate(fit,self.policy,test)

    def test_validated_routing_positive_path_uses_frozen_rule(self):
        # 单元测试伪构造身份用于覆盖控制分支；不是一次真实风险验证。
        fit=G.fit(self.real_data('calibration'));data=self.real_data('test',20)
        report=G.validate(fit,self.policy,data);release=G.publish(report)
        p=self.root/'release.json';write_json(p,release)
        result=route(data['cases'][0]['prediction'],{'mode':'validated','release':str(p)},knowledge_base())
        self.assertTrue(result['skip']);self.assertTrue(result['calibrated'])
        a=copy.deepcopy(data['cases'][0]['prediction']);a['signature']['weights']=['changed']
        self.assertFalse(route(a,{'mode':'validated','release':str(p)},knowledge_base())['skip'])
