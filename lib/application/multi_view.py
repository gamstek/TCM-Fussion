"""通用多图入口：原图/manifest→新建 case→doctor→既有工作流；图片数量由 manifest 决定。

由 cli.main 的 multi-view 与 Python 调用方使用；只复制源像素，参考标签不进入规划
配置；dry-run 只做准备与本地检查，不运行模型。

主要定义：
- inspect_sources: 校验目录/manifest 中的图片路径、SHA256 与重复性。
- bind_config: 把源图与样品绑定进运行模板，生成新 case 配置。
- run_multi_view: 创建独立 case、执行 doctor 并启动 run_pipeline。
- `_validate_case_name` 等系列：Windows 合法/保留 case 名的内部校验。

关联：
- 调用方 「cli/main.py」、「tests/application/test_multi_view.py」。
- 直接依赖 「lib/application/pipeline.py」、「lib/runtime/storage.py」。
"""
from __future__ import annotations

import copy
import os
import re
import shutil
import uuid
from pathlib import Path

import paths
from TCM_Fusion.lib.application.pipeline import doctor, run_pipeline, validate_config
from TCM_Fusion.lib.runtime.storage import IntegrationError, file_hash, project_path, read_json, write_json

_INVALID_NAME_CHARS = frozenset('<>:"/\\|?*') | frozenset(chr(i) for i in range(32))
_RESERVED_NAMES = frozenset({'CON', 'PRN', 'AUX', 'NUL'}) | frozenset(
    f'{prefix}{i}' for prefix in ('COM', 'LPT') for i in range(1, 10))
_MAX_CASE_NAME = 100


def _validate_case_name(case_name):
    """Reject names that cannot be used as a portable directory component."""
    if not isinstance(case_name, str) or not case_name.strip():
        raise IntegrationError('MULTI_VIEW_NAME', 'case_name不能为空')
    if len(case_name) > _MAX_CASE_NAME:
        raise IntegrationError('MULTI_VIEW_NAME', f'case_name不能超过{_MAX_CASE_NAME}个字符')
    if case_name in {'.', '..'} or case_name != case_name.strip() or case_name[-1] in ' .':
        raise IntegrationError('MULTI_VIEW_NAME', 'case_name不能为"."/".."，且不能以空格或点结尾')
    invalid = sorted({ch for ch in case_name if ch in _INVALID_NAME_CHARS})
    if invalid:
        raise IntegrationError(
            'MULTI_VIEW_NAME', 'case_name包含非法字符: ' + ' '.join(repr(ch) for ch in invalid))
    if case_name.split('.', 1)[0].upper() in _RESERVED_NAMES:
        raise IntegrationError('MULTI_VIEW_NAME', f'case_name不能使用系统保留名: {case_name}')
    return case_name


def inspect_sources(source_dir, manifest=None):
    """Validate any positive number of distinct original files inside source_dir."""
    source = project_path(source_dir)
    proof = read_json(project_path(manifest) if manifest else source / 'input_proof.json')
    views = proof.get('views')
    if not isinstance(views, list) or not views:
        raise IntegrationError('MULTI_VIEW_INPUT', 'manifest必须包含至少1个views')
    bindings = []
    for row in views:
        if not isinstance(row, dict):
            raise IntegrationError('MULTI_VIEW_INPUT', 'view必须是对象')
        name = row.get('path') or row.get('view')
        if not isinstance(name, str) or not name:
            raise IntegrationError('MULTI_VIEW_INPUT', 'view缺少path或view文件名')
        candidate = Path(name)
        image = (candidate if candidate.is_absolute() else source / candidate).resolve()
        expected = row.get('sha256')
        if not image.is_relative_to(source) or not image.is_file():
            raise IntegrationError('MULTI_VIEW_SCOPE', '原图必须存在且位于source-dir内')
        if image.suffix.lower() not in {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}:
            raise IntegrationError('MULTI_VIEW_INPUT', '不是支持的图像文件')
        if not isinstance(expected, str) or not re.fullmatch('[a-fA-F0-9]{64}', expected):
            raise IntegrationError('MULTI_VIEW_HASH', '每张图必须提供SHA256')
        actual = file_hash(image)
        if actual != expected.lower():
            raise IntegrationError('MULTI_VIEW_HASH', '原图SHA256与manifest不一致')
        bindings.append({'source': str(image), 'sha256': actual})
    if len({x['source'] for x in bindings}) != len(bindings) or len({x['sha256'] for x in bindings}) != len(bindings):
        raise IntegrationError('MULTI_VIEW_DUPLICATE', '路径及内容必须各自唯一')
    return bindings


def bind_config(template, images, *, sample_id, a_python=None):
    """Use current policy unchanged; inject neutral paths and local interpreter."""
    cfg = copy.deepcopy(template)
    if 'routing' in cfg or cfg.get('allow_recorded', False):
        raise IntegrationError('MULTI_VIEW_CONFIG', '多图真实接口不接受routing/recorded')
    agent = cfg.get('agent', {})
    workflow = agent.get('workflow', {})
    if (cfg.get('a', {}).get('mode') != 'cli' or cfg.get('b', {}).get('mode') != 'run'
            or not cfg.get('allow_network') or agent.get('mode') != 'active'
            or agent.get('planner') != 'llm' or agent.get('transport') != 'mcp'
            or workflow.get('agent_policy') != 'adaptive_v2'
            or workflow.get('execution') != 'active' or workflow.get('detector') != 'yolo'
            or cfg['b'].get('enable_vlm', False)):
        raise IntegrationError('MULTI_VIEW_CONFIG', '必须使用真实active/llm/mcp/adaptive_v2/YOLO')
    cfg['sample_id'] = sample_id
    cfg['a'].pop('image', None)
    cfg['a']['images'] = list(images)
    a_python = a_python or os.environ.get('TCM_A_PYTHON')
    if not a_python:
        raise IntegrationError(
            'MULTI_VIEW_A_PYTHON_MISSING',
            '必须用 --a-python 或 TCM_A_PYTHON 指定 A 解释器；不默认使用 Agent venv')
    cfg['a']['python'] = str(project_path(a_python))
    cfg['b'].update(input_mode='images', input=images[0])
    cfg['agent']['image_roots'] = list(images)
    return validate_config(cfg)


def run_multi_view(source_dir, *, output_root=None, manifest=None,
                  config_template=None, a_python=None, dry_run=False, case_name=None):
    """Return fresh case on dry-run, completed run directory on live execution.

    case_name optionally names the case directory; the config sample_id is
    sanitized separately so Chinese/display names remain usable on disk.
    All failures after case creation write interface_status.json. Retrying means
    another case; no recorded observations, old A cache or hidden fallback.
    """
    bindings = inspect_sources(source_dir, manifest)
    a_python = a_python or os.environ.get('TCM_A_PYTHON')
    if not a_python:
        raise IntegrationError(
            'MULTI_VIEW_A_PYTHON_MISSING',
            '必须用 --a-python 或 TCM_A_PYTHON 指定 A 解释器；不默认使用 Agent venv')
    template_path = project_path(config_template) if config_template else (
        Path(paths.PROJECT_ROOT) / 'TCM_Fusion/config/multi_view.json')
    template = read_json(template_path)
    root = project_path(output_root) if output_root else (
        Path(paths.PROJECT_ROOT) / 'TCM_Fusion/运行结果')
    source = project_path(source_dir)
    if root == source or root.is_relative_to(source):
        raise IntegrationError('MULTI_VIEW_SCOPE', '输出根不能位于输入图像目录内')
    if case_name is not None:
        _validate_case_name(case_name)
        base = root / case_name
    else:
        base = root / ('case_' + uuid.uuid4().hex[:8])
    base.mkdir(parents=True, exist_ok=False)
    sample_id = re.sub(r'[^0-9A-Za-z_-]+','_', case_name)[:64].strip('_') if case_name else base.name
    sample_id = sample_id or base.name
    status = {'schema': 'multi-view-interface/v1', 'dry_run': bool(dry_run),
              'base': str(base), 'stage': 'copying', 'status': 'running'}
    write_json(base / 'interface_status.json', status)
    try:
        images = []
        for index, row in enumerate(bindings, 1):
            original = Path(row['source'])
            image = base / f'view_{index:02d}{original.suffix.lower()}'
            shutil.copyfile(original, image)
            if file_hash(image) != row['sha256']:
                raise IntegrationError('MULTI_VIEW_HASH', '复制期间原图内容发生变化')
            images.append(str(image))
        cfg = bind_config(template, images, sample_id=sample_id, a_python=a_python)
        write_json(base / 'config.json', cfg)
        write_json(base / 'input_proof.json', {'views': [
            {**row, 'view': Path(image).name} for row, image in zip(bindings, images)],
            'reference_excluded_from_agent': True,
            'previous_visual_artifacts_reused': False, 'a_cache_reused': False,
            'same_physical_specimen_verified': False})
        status['stage'] = 'doctor'
        write_json(base / 'interface_status.json', status)
        readiness = doctor(cfg)
        write_json(base / 'doctor.json', readiness)
        if not readiness['ready']:
            raise IntegrationError('MULTI_VIEW_NOT_READY', f'本地检查失败，见{base / "doctor.json"}')
        if dry_run:
            status.update(stage='prepared', status='prepared')
            write_json(base / 'interface_status.json', status)
            return base
        status['stage'] = 'pipeline'
        write_json(base / 'interface_status.json', status)
        assert not (base / 'a_cache').exists() and not (base / 'runs').exists()
        run = run_pipeline(cfg, base=base)
        write_json(base / 'run_location.json', {'directory': str(run)})
        status.update(stage='done', status='completed', directory=str(run))
        write_json(base / 'interface_status.json', status)
        return run
    except Exception as exc:
        status.update(status='failed', error_code=getattr(exc, 'code', type(exc).__name__))
        write_json(base / 'interface_status.json', status)
        raise
