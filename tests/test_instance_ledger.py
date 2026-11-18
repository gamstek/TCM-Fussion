"""D1 回归：真实 WorkflowSession 路由/提交，视觉后端为显式离线替身。

消费 StageFixture、contracts 和实例索引；由既有 selftest 自动发现。
验证首次 GLM 可达、语义去重、输入变化可重开及失败/pending 不重发；不证明识别准确率。
"""
import copy
import unittest
from pathlib import Path

from TCM_Fusion.lib.instance_ledger import attempt, describe, project
from TCM_Fusion.lib.storage import read_json, write_json
from TCM_Fusion.tests import test_workflow as fixtures

StageFixture = fixtures.StageFixture


class InstanceLedgerTests(unittest.TestCase):
    setUp = fixtures.WorkflowTests.setUp
    runtime = fixtures.WorkflowTests.runtime

    def detected(self, backend=None):
        r = self.runtime(backend)
        action = next(a for a in r.workflow.propose() if a['tool'] == 'detect_fov_initial')
        self.assertEqual(r.workflow.execute(action)['status'], 'done')
        crop = next(t for t in r.workflow.targets().values() if t['kind'] == 'crop')
        return r, crop

    def pending_cnn(self, r, crop):
        root = self.root/'cnn_result'
        write_json(root/'classification.json', {'label': '待复核', 'reject': 'pending_review',
            'feature_id': '', 'source': 'cnn', 'confidence_source': 'cnn'})
        r.workflow.accept('classify_target', crop, {'classification': str(root)},
                          r.workflow.store.current()['version'])

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

    def test_cnn_pending_review_offers_and_executes_one_glm(self):
        r, crop = self.detected(); self.pending_cnn(r, crop)
        actions = [a for a in r.workflow.propose() if a['tool'] == 'classify_target']
        self.assertEqual(len(actions), 1)
        result = r.workflow.execute(actions[0])
        self.assertEqual(result['status'], 'done')
        self.assertEqual(r.workflow.backend.calls.count('classify_target'), 1)
        self.assertFalse(any(a['tool'] == 'classify_target' and
                             a['args']['target_id'] == crop['id'] for a in r.workflow.propose()))

    def test_unreadable_glm_is_terminal_and_indexed(self):
        class Unreadable(StageFixture):
            def execute(self, s, tool, target, work, context):
                if tool == 'classify_target':
                    self.calls.append(tool)
                    write_json(work/'classification/classification.json',
                               {'label':'不可辨认', 'feature_id':'', 'reject':'unreadable', 'source':'vlm'})
                    return {'classification': str(work/'classification')}
                return super().execute(s, tool, target, work, context)
        r, crop = self.detected(Unreadable()); self.pending_cnn(r, crop)
        a = next(a for a in r.workflow.propose() if a['tool'] == 'classify_target')
        self.assertEqual(r.workflow.execute(a)['status'], 'done')
        renamed = self.alias(r, crop)
        self.assertTrue(r.workflow._classify_attempted(renamed))
        index = read_json(r.workflow.root/'instance_ledger.json')
        rows = list(index['instances'].values())
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['attempts'][0]['status'], 'done')

    def test_renamed_crop_shares_instance_and_attempt_but_pixels_can_change(self):
        r, crop = self.detected(); renamed = self.alias(r, crop)
        snap = r.workflow.store.current()
        self.assertEqual(describe(crop, snap)['instance_id'], describe(renamed, snap)['instance_id'])
        for tool in ('classify_target', 'segment_instance'):
            self.assertEqual(attempt(tool, crop, snap, 'method'), attempt(tool, renamed, snap, 'method'))
        changed = self.alias(r, crop, name='changed', pixels=b'more-context')
        snap = r.workflow.store.current()
        self.assertNotEqual(attempt('classify_target',crop,snap,'method'),
                            attempt('classify_target',changed,snap,'method'))

    def test_expanded_roi_new_method_and_scale_are_distinct_inputs(self):
        r, crop = self.detected(); enlarged = self.alias(r, crop, box=[0,0,16,16])
        snap = r.workflow.store.current()
        self.assertNotEqual(describe(crop,snap)['instance_id'], describe(enlarged,snap)['instance_id'])
        self.assertNotEqual(attempt('segment_instance',crop,snap,'v1'),
                            attempt('segment_instance',crop,snap,'v2'))
        changed = copy.deepcopy(snap)
        changed['artifacts']['scale:'+crop['image_id']]['sha256'] = 'new-scale'
        self.assertNotEqual(attempt('segment_instance',crop,snap,'v1'),
                            attempt('segment_instance',crop,changed,'v1'))

    def test_different_source_images_are_not_shared(self):
        r, crop = self.detected(); snap = r.workflow.store.current()
        changed = copy.deepcopy(snap)
        changed['artifacts'][crop['image_id']]['sha256'] = 'another-source'
        self.assertNotEqual(describe(crop,snap)['instance_id'], describe(crop,changed)['instance_id'])

    def test_failed_segmentation_renamed_crop_is_pruned_without_new_cost(self):
        r, crop = self.detected(StageFixture(fail='segment_instance'))
        a = next(a for a in r.workflow.propose() if a['tool'] == 'segment_instance')
        failed = r.workflow.execute(a); self.assertEqual(failed['status'], 'failed')
        renamed = self.alias(r, crop)
        self.assertFalse(any(a['tool']=='segment_instance' and a['args']['target_id']==renamed['id']
                             for a in r.workflow.propose()))
        before = read_json(r.workflow.budget_path)
        requested = r.action('segment_instance', {'target_id':renamed['id'],
            'evidence_version':r.workflow.store.current()['version']}, 'workflow_repair')
        reused = r.workflow.execute(requested)
        self.assertTrue(reused['semantic_cache_hit']); self.assertEqual(reused['status'], 'failed')
        self.assertEqual(before, read_json(r.workflow.budget_path))
        self.assertEqual(r.workflow.backend.calls.count('segment_instance'), 1)

    def test_pending_semantic_attempt_cannot_be_reissued(self):
        r, crop = self.detected(); semantic = r.workflow.semantic_attempt('classify_target',crop)
        doc = read_json(r.workflow.operations)
        doc['entries']['in-flight'] = {**semantic,'tool':'classify_target','target_id':crop['id'],
                                       'status':'pending','accepted':False}
        write_json(r.workflow.operations,doc)
        renamed = self.alias(r,crop)
        a = r.action('classify_target',{'target_id':renamed['id'],
            'evidence_version':r.workflow.store.current()['version']},'workflow_repair')
        self.assertEqual(r.workflow.execute(a)['status'],'pending')
        self.assertEqual(r.workflow.backend.calls.count('classify_target'),0)

    def test_projection_links_classification_mask_measurement_without_new_support(self):
        r, crop = self.detected()
        seg = next(a for a in r.workflow.propose() if a['tool']=='segment_instance')
        self.assertEqual(r.workflow.execute(seg)['status'],'done')
        measure = next(a for a in r.workflow.propose() if a['tool']=='measure_instances')
        self.assertEqual(r.workflow.execute(measure)['status'],'done')
        before = r.workflow.store.current()
        view = project(before,read_json(r.workflow.operations))
        row = next(iter(view['instances'].values()))
        self.assertEqual(len(row['mask_versions']),1); self.assertEqual(len(row['measurements']),1)
        self.assertEqual({x['status'] for x in row['attempts']},{'done'})
        self.assertEqual(before,r.workflow.store.current())

    def test_measurement_key_changes_when_mask_content_changes(self):
        r, crop = self.detected()
        r.workflow.execute(next(a for a in r.workflow.propose() if a['tool']=='segment_instance'))
        target = next(t for t in r.workflow.targets().values() if t['kind']=='segmentation')
        snap=r.workflow.store.current(); before=attempt('measure_instances',target,snap,'method')
        mask=next(Path(target['masks']).glob('*.png')); mask.write_bytes(b'changed-mask')
        self.assertNotEqual(before,attempt('measure_instances',target,snap,'method'))

    def test_auto_measurement_is_indexed_and_not_offered_again(self):
        r = self.runtime(auto_measure_after_segment=True)
        r.workflow.execute(next(a for a in r.workflow.propose() if a['tool']=='detect_fov_initial'))
        r.workflow.execute(next(a for a in r.workflow.propose() if a['tool']=='segment_instance'))
        rows = list(read_json(r.workflow.operations)['entries'].values())
        measured = [x for x in rows if x['tool']=='measure_instances']
        self.assertEqual(len(measured),1); self.assertEqual(measured[0]['status'],'done')
        self.assertTrue(measured[0]['automatic'])
        self.assertTrue(r.workflow.store.current()['records'])
        self.assertFalse(any(a['tool']=='measure_instances' for a in r.workflow.propose()))

    def test_auto_measurement_failure_is_terminal_and_visible(self):
        r = self.runtime(StageFixture(fail='measure_instances'), auto_measure_after_segment=True)
        r.workflow.execute(next(a for a in r.workflow.propose() if a['tool']=='detect_fov_initial'))
        result = r.workflow.execute(next(a for a in r.workflow.propose() if a['tool']=='segment_instance'))
        self.assertEqual(result['status'],'done')  # 合法 mask 不因测量失败而回滚。
        measured = [x for x in read_json(r.workflow.operations)['entries'].values()
                    if x['tool']=='measure_instances']
        self.assertEqual(len(measured),1); self.assertEqual(measured[0]['status'],'failed')
        self.assertEqual(measured[0]['error_code'],'SIMULATED_FAILURE')
        self.assertFalse(any(a['tool']=='measure_instances' for a in r.workflow.propose()))
