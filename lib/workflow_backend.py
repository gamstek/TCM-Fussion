"""真实B阶段包装：固定入口子进程/具名库worker；单视野或单crop，不重跑整条B。"""
import shutil
from pathlib import Path
import contracts
import paths
from .controlled_process import run
from .storage import IntegrationError, read_json, write_json, file_hash, digest


class WorkflowBackend:
    @staticmethod
    def code_signature():
        root=Path(paths.PROJECT_ROOT)
        roots=[Path(paths.DETECTION_LIB_DIR),Path(paths.SEGMENTATION_LIB_DIR),Path(paths.COMMON_DIR),
               Path(paths.GEOM_DIR),Path(paths.JUDGE_SCRIPTS_DIR),root/'TCM_Fusion/lib',root/'TCM_Fusion/cli']
        files={p for folder in roots for p in folder.rglob('*.py')}
        files.update([root/'contracts.py',root/'feature_taxonomy.json',
                      Path(paths.SEGMENTATION_CONFIG_DIR)/'quality_policy.json',
                      Path(paths.SEGMENTATION_CONFIG_DIR)/'association_policy.json',
                      Path(paths.DETECTION_CLI_DIR)/'detect_sample.py',
                      Path(paths.SEGMENTATION_CLI_DIR)/'igm_segment_batch.py'])
        return digest([(p.relative_to(root).as_posix(),file_hash(p)) for p in sorted(files)])

    def execute(self, session, tool, target, work, context):
        py=session.python;common=[py,'-X','utf8']
        def command(args):
            run(common+list(map(str,args)),context_path=context,timeout=session.remaining_seconds(),log_path=work/'stage.log',cancel_event=session.cancel_event)
        if tool=='detect_fov_initial':
            out=work/'detection'
            command([Path(paths.DETECTION_CLI_DIR)/'detect_sample.py','--images',target['path'],
                     '--sample',session.sample_id,'--herb','unknown','--out-dir',out])
            if not (out/'crops.json').is_file():raise IntegrationError('DETECTION_INCOMPLETE','检测没有完成清单')
            return {'crops_dir':str(out/'crops'),'completion':str(out/'crops.json')}
        if tool in {'detect_region','calibrate_fov'}:
            out=work/'detection';ctx={'tool':tool,'source':target['source'],'output':str(out)}
            if tool=='detect_region':
                ctx.update(roi=target['roi'],px_per_um=target.get('px_per_um'),existing_boxes=session.existing_boxes(target['image_id']),replace_id=target.get('replace_id'))
            f=work/'worker.json';write_json(f,ctx)
            command(['-m','TCM_Fusion.cli.workflow_worker','--context',f])
            return {'scale':str(out/'scale.json')} if tool=='calibrate_fov' else {
                'crops_dir':str(out/'crops'),'completion':str(out/'region_result.json')}
        if tool=='classify_target':
            # 单crop类别复核：复用检测侧 classify_crop（提示词/解析/白名单只此一份），
            # 在子进程内跑，网络请求由 worker_bootstrap 的HTTP守卫按请求键记账。
            meta=contracts.CropMeta.from_json(target['path'])
            ctx={'tool':tool,'crop_id':meta.crop_id,'crop_json':str(target['path']),
                 'crop_png':str(meta.png),'source':meta.source_image_path,
                 'tight_box':meta.tight_box,'output':str(work/'classification')}
            f=work/'worker.json';write_json(f,ctx)
            out=work/'classification'
            command(['-m','TCM_Fusion.cli.classify_worker','--context',f])
            if not (out/'classification.json').is_file():
                raise IntegrationError('CLASSIFY_INCOMPLETE','分类阶段没有产出清单')
            return {'classification':str(out)}
        if tool=='segment_instance':
            src=Path(target['path']);meta=contracts.CropMeta.from_json(str(src))
            contracts.check_crops([meta]);inp=work/'crops';inp.mkdir()
            shutil.copy2(src,inp/src.name);shutil.copy2(meta.png,inp/Path(meta.png).name)
            out=work/'segmentation'
            command([Path(paths.SEGMENTATION_CLI_DIR)/'igm_segment_batch.py','--crops',inp,'--out-dir',out,
                     '--box-association', session.config.get('box_association','region_component')])
            doc=read_json(out/'instances.json')
            if not doc.get('instances'):raise IntegrationError('EMPTY_MASK','没有分割实例')
            return {'segmentation':str(out),'quality':'valid' if all(
                i.get('igm_status')=='ok' for i in doc['instances']) else 'review'}
        if tool=='measure_instances':
            # 局部阶段各自从inst01开始；先在新交付副本做确定性命名空间，
            # 使原measure输出的instance_id仍严格对应实际mask文件名。
            import csv
            source=Path(target['masks']);delivery=work/'delivery';delivery.mkdir()
            rows=[];offset=int(digest(target['id'])[:8],16)*10000
            for f in source.glob('*__measure_manifest.csv'):
                rows=[]
                with f.open(encoding='utf-8-sig',newline='') as stream:
                    for raw in csv.DictReader(stream):
                        row=contracts.MeasureManifestRow.from_row(raw)
                        if row.instance_id is None or not 0<=row.instance_id<10000:
                            raise IntegrationError('INSTANCE_ID','局部mask编号非法')
                        src=(source/row.inst_mask).resolve()
                        if not src.is_relative_to(source.resolve()) or not src.is_file():
                            raise IntegrationError('MASK_SCOPE','mask越出交付目录或不存在')
                        if session.purpose_quality():
                            qa=session.assessment(target)
                            if not qa or not qa['integrity']:raise IntegrationError('QUALITY_SCOPE','无受信用途资格')
                            aid=target.get('assessment_id');assessment=session.store.current()['artifacts'][aid]
                            src=Path(assessment['path']).parent/'associated_mask.png'
                            if file_hash(src)!=qa['bindings']['mask']:raise IntegrationError('QUALITY_SCOPE','关联mask变化')
                            if rows:raise IntegrationError('QUALITY_MULTIPLE_INSTANCES','单目标资格不能用于多实例重复测量')
                        iid=offset+row.instance_id;name=f"{Path(row.source_image_path).stem}__{row.feature_id}__inst{iid}.png"
                        shutil.copy2(src,delivery/name)
                        rows.append({**raw,'instance_id':iid,'inst_mask':name,'herb':''})
                with (delivery/f.name).open('w',encoding='utf-8-sig',newline='') as stream:
                    w=csv.DictWriter(stream,fieldnames=contracts.MANIFEST_COLUMNS);w.writeheader();w.writerows(rows)
            if (source/'crops').is_dir():shutil.copytree(source/'crops',delivery/'crops')
            out=work/'measurement';out.mkdir()
            command([Path(paths.GEOM_DIR)/'measure.py','--masks',delivery,
                     '--out-prefix',out/'result','--split-policy','all'])
            f=out/'result_measurements.jsonl'
            rows=list(contracts.iter_measurement_records(str(f)))
            if not rows:raise IntegrationError('EMPTY_MEASUREMENT','测量没有实例')
            return {'measurements':str(f),'delivery':str(delivery)}
        raise IntegrationError('TOOL_UNIMPLEMENTED','没有真实阶段实现')
