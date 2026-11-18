"""绑定受控阶段输出与工作目录。接受后端产物路径，记录相对落点和 SHA256，供离线回放核对。
由「lib/workflow/workflow_session.py」调用，结果按调用方的任务阶段消费。

主要定义：
- bundle。

关联：
- 调用方 「lib/workflow/workflow_session.py」。
- 直接依赖 「lib/runtime/storage.py」。
"""
from pathlib import Path
from TCM_Fusion.lib.runtime.storage import IntegrationError, file_hash


def bundle(result,work):
    root=Path(work).resolve();paths={};files={}
    for key in ('crops_dir','completion','regions','scale','segmentation','classification','measurements','delivery'):
        if key not in result:continue
        p=Path(result[key]).resolve()
        if not p.is_relative_to(root):raise IntegrationError('STAGE_OUTPUT_SCOPE','后端产物越出独立工作区')
        paths[key]=p.relative_to(root).as_posix()
        for f in ([p] if p.is_file() else p.rglob('*')):
            if f.is_file():files[f.relative_to(root).as_posix()]=file_hash(f)
    return {'root':str(root),'paths':paths,'files':files,'quality':result.get('quality')}
