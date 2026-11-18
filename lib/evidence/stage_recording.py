"""把受控阶段的结果绑定到独立工作目录和文件摘要。

记录单阶段文件、批量分类的逐裁图结果，以及示例审计及其拼图。回放按相对路径重建输出，批量失败仅保留结构化状态。

主要定义：
- bundle 整理阶段产物相对路径、摘要及回放结果。

关联：
- workflow_session 在提交阶段结果前记录工件。
- workflow_replay 校验并复制录制文件。
"""
from pathlib import Path
from TCM_Fusion.lib.runtime.storage import IntegrationError, file_hash


def bundle(result, work):
    root = Path(work).resolve()
    files = {}

    def record_path(value, *, required_file=False):
        path = Path(value).resolve()
        if not path.is_relative_to(root):
            raise IntegrationError('STAGE_OUTPUT_SCOPE', '后端产物越出独立工作区')
        if required_file and not path.is_file():
            raise IntegrationError('STAGE_OUTPUT_MISSING', '缺少可回放的阶段文件')
        for file in ([path] if path.is_file() else sorted(path.rglob('*'))):
            if file.is_file():
                if not file.resolve().is_relative_to(root):
                    raise IntegrationError('STAGE_OUTPUT_SCOPE', '后端产物越出独立工作区')
                files[file.relative_to(root).as_posix()] = file_hash(file)
        return path.relative_to(root).as_posix()

    fields = ('crops_dir', 'completion', 'regions', 'scale', 'segmentation', 'classification',
              'measurements', 'delivery', 'audit', 'contact_sheet')
    paths = {
        key: record_path(result[key], required_file=key in {'audit', 'contact_sheet'})
        for key in fields if key in result
    }
    value = {'root': str(root), 'paths': paths, 'files': files, 'quality': result.get('quality')}
    if 'classifications' in result:
        rows = []
        for row in result['classifications']:
            classification = Path(row['classification']) / 'classification.json'
            recorded_file = record_path(classification, required_file=True)
            rows.append({'crop_id': row['crop_id'], 'status': row['status'],
                         'classification': Path(recorded_file).parent.as_posix()})
        value['classifications'] = rows
        value['failures'] = [
            {key: row[key] for key in ('crop_id', 'status', 'error_code')}
            for row in result.get('failures', [])
        ]
    return value
