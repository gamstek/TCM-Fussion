"""调查进展的本地指纹：约束无收益循环，不规定模型应调查哪个候选。

消费当前原规则检查和已采信工件；不把重命名、文件哈希变化、实例数量累加或假设切换
当成调查进展。react_graph 在执行前后对拍，不增加模型请求、不授予证据资格。

主要定义：
- progress_state。同一事实源上的分轴指纹：能力/证据/输入诊断分别评价，不新增证据授权。
- progress_signature。
- progress_axes。新能力不等于正式支持，已诊断不可用不等于阴性；重命名/无关文件不计。
- continuation_count。只有实际无进展的继续执行才扣宽限次数；换假设或改建议文案不能清零。

关联：
- 调用方 「lib/agent/react_graph.py」、「tests/agent/test_investigation_control.py」、「tests/integration/test_agent_efficiency.py」。
- 直接依赖 「lib/evidence/instance_ledger.py」、「lib/runtime/storage.py」。
"""
import contracts
from TCM_Fusion.lib.runtime.storage import digest, read_json


def progress_state(runtime, scored):
    """同一事实源上的分轴指纹：能力/证据/输入诊断分别评价，不新增证据授权。"""
    snapshot = runtime.workflow.store.current(); arts = snapshot['artifacts']
    stages = set(); diagnostics = set()
    for aid, artifact in arts.items():
        if not artifact.get('valid'):
            continue
        if artifact.get('kind') == 'detection' and aid.startswith('detected:'):
            stages.add(digest(['view_examined', artifact.get('image_id')]))
        elif artifact.get('kind') == 'scale':
            stages.add(digest(['scale', artifact.get('image_id'), artifact.get('px_per_um')]))
        elif artifact.get('kind') == 'crop':
            from TCM_Fusion.lib.evidence.instance_ledger import describe
            bound = describe(artifact, snapshot)
            stages.add(digest(['crop_available', bound['source_image_sha256'],
                              bound['feature_id'], bound['geometry'], bound['crop_sha256']]))
        elif artifact.get('kind') in {'classification','segmentation'}:
            uses = None
            if artifact['kind'] == 'classification' and not artifact.get('identity_verified'):
                crop = arts.get(artifact.get('crop_id'), {})
                if crop:
                    from TCM_Fusion.lib.evidence.instance_ledger import describe
                    bound = describe(crop, snapshot)
                    result = read_json(artifact['path'])
                    # 只记录受信分类结果已评估输入；不能将不可判当成目标不存在。
                    if result.get('reject'):
                        diagnostics.add(digest(['identity_input_assessed',
                            bound['source_image_sha256'], bound['geometry'], bound['crop_sha256'],
                            result.get('reject'), result.get('feature_id')]))
                continue
            if artifact['kind'] == 'segmentation':
                if artifact.get('quality') != 'valid':
                    continue
                if runtime.workflow.purpose_quality():
                    qa = runtime.workflow.assessment(artifact)
                    if not qa or not qa['integrity'] or not any(u['allowed'] for u in qa['uses'].values()):
                        continue
                    uses = sorted(k for k,u in qa['uses'].items() if u['allowed'])
            crop = arts.get(artifact.get('crop_id'))
            if not crop:
                crop = next((arts[p] for p in artifact.get('parents', [])
                             if arts.get(p,{}).get('kind') == 'crop'), None)
            if crop:
                meta = contracts.CropMeta.from_json(crop['path'])
                stages.add(digest([artifact['kind'], crop.get('image_id'),
                                   meta.tight_box or crop['id'], artifact.get('feature_id') or meta.feature_id, uses]))
    checks = sorted((r['id'], c.get('feature_id'), c.get('state'))
                    for r in scored['decision'].get('ranking', []) for c in r.get('checks', []))
    observed = sorted({(o['image_id'], o['feature_id'], o['state'])
                       for o in scored.get('observation_states', []) if o['state'] in {'present','absent'}})
    scene = scored.get('scene') or {}
    evidence = {digest(['check', *c]) for c in checks}
    evidence.update(digest(['observation', *o]) for o in observed)
    evidence.update(digest(['feature', fid]) for fid, card in scored['cards'].items() if card.present)
    scene_facts = {k:scene.get(k) for k in ('cell_wall_visible','crystal_location','arrangement','abundance_obs')}
    if any(v is not None for v in scene_facts.values()):
        evidence.add(digest(['scene', scene_facts]))
    return {'evidence': sorted(evidence), 'capability': sorted(stages), 'diagnostic': sorted(diagnostics)}


def progress_signature(runtime, scored):
    return digest(progress_state(runtime, scored))


def progress_axes(before, after):
    """新能力不等于正式支持，已诊断不可用不等于阴性；重命名/无关文件不计。"""
    return {axis: bool(set(after[axis]) - set(before[axis]))
            for axis in ('evidence', 'capability', 'diagnostic')}


def continuation_count(previous, *, progress, continued_after_advice):
    """只有实际无进展的继续执行才扣宽限次数；换假设或改建议文案不能清零。"""
    return 0 if progress else previous + int(continued_after_advice)
