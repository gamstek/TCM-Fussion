"""Five original images + manifest -> fresh case -> doctor -> existing workflow.

Consumed by cli.main five-view and Python callers. Only source pixels are copied;
reference labels remain outside planner configuration. Dry-run never runs models.
"""
from __future__ import annotations

import copy
import os
import re
import shutil
import uuid
from pathlib import Path

import paths
from .pipeline import doctor, run_pipeline, validate_config
from .storage import IntegrationError, file_hash, project_path, read_json, write_json


def inspect_sources(source_dir, manifest=None):
    """Validate five distinct original files inside source_dir, bound by SHA256."""
    source = project_path(source_dir)
    proof = read_json(project_path(manifest) if manifest else source / 'input_proof.json')
    views = proof.get('views')
    if not isinstance(views, list) or len(views) != 5:
        raise IntegrationError('FIVE_VIEW_INPUT', 'manifest必须包含恰好5个views')
    bindings = []
    for row in views:
        if not isinstance(row, dict):
            raise IntegrationError('FIVE_VIEW_INPUT', 'view必须是对象')
        name = row.get('path') or row.get('view')
        if not isinstance(name, str) or not name:
            raise IntegrationError('FIVE_VIEW_INPUT', 'view缺少path或view文件名')
        candidate = Path(name)
        image = (candidate if candidate.is_absolute() else source / candidate).resolve()
        expected = row.get('sha256')
        if not image.is_relative_to(source) or not image.is_file():
            raise IntegrationError('FIVE_VIEW_SCOPE', '原图必须存在且位于source-dir内')
        if image.suffix.lower() not in {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}:
            raise IntegrationError('FIVE_VIEW_INPUT', '不是支持的图像文件')
        if not isinstance(expected, str) or not re.fullmatch('[a-fA-F0-9]{64}', expected):
            raise IntegrationError('FIVE_VIEW_HASH', '每张图必须提供SHA256')
        actual = file_hash(image)
        if actual != expected.lower():
            raise IntegrationError('FIVE_VIEW_HASH', '原图SHA256与manifest不一致')
        bindings.append({'source': str(image), 'sha256': actual})
    if len({x['source'] for x in bindings}) != 5 or len({x['sha256'] for x in bindings}) != 5:
        raise IntegrationError('FIVE_VIEW_DUPLICATE', '必须是5张路径及内容不同的原图')
    return bindings


def bind_config(template, images, *, sample_id, a_python=None):
    """Use current policy unchanged; inject neutral paths and local interpreter."""
    cfg = copy.deepcopy(template)
    if 'routing' in cfg or cfg.get('allow_recorded', False):
        raise IntegrationError('FIVE_VIEW_CONFIG', '五图真实接口不接受routing/recorded')
    agent = cfg.get('agent', {})
    workflow = agent.get('workflow', {})
    if (cfg.get('a', {}).get('mode') != 'cli' or cfg.get('b', {}).get('mode') != 'run'
            or not cfg.get('allow_network') or agent.get('mode') != 'active'
            or agent.get('planner') != 'llm' or agent.get('transport') != 'mcp'
            or workflow.get('agent_policy') != 'adaptive_v2'
            or workflow.get('execution') != 'active' or workflow.get('detector') != 'yolo'
            or cfg['b'].get('enable_vlm', False)):
        raise IntegrationError('FIVE_VIEW_CONFIG', '必须使用真实active/llm/mcp/adaptive_v2/YOLO')
    cfg['sample_id'] = sample_id
    cfg['a'].pop('image', None)
    cfg['a']['images'] = list(images)
    a_python = a_python or os.environ.get('TCM_A_PYTHON')
    if not a_python:
        raise IntegrationError(
            'FIVE_VIEW_A_PYTHON_MISSING',
            '必须用 --a-python 或 TCM_A_PYTHON 指定 A 解释器；不默认使用 Agent venv')
    cfg['a']['python'] = str(project_path(a_python))
    cfg['b'].update(input_mode='images', input=images[0])
    cfg['agent']['image_roots'] = list(images)
    return validate_config(cfg)


def run_five_view(source_dir, *, output_root=None, manifest=None,
                  config_template=None, a_python=None, dry_run=False):
    """Return fresh case on dry-run, completed run directory on live execution.

    All failures after case creation write interface_status.json. Retrying means
    another case; no recorded observations, old A cache or hidden fallback.
    """
    bindings = inspect_sources(source_dir, manifest)
    a_python = a_python or os.environ.get('TCM_A_PYTHON')
    if not a_python:
        raise IntegrationError(
            'FIVE_VIEW_A_PYTHON_MISSING',
            '必须用 --a-python 或 TCM_A_PYTHON 指定 A 解释器；不默认使用 Agent venv')
    template_path = project_path(config_template) if config_template else (
        Path(paths.PROJECT_ROOT) / 'TCM_Fusion/config/five_view.json')
    template = read_json(template_path)
    root = project_path(output_root) if output_root else (
        Path(paths.PROJECT_ROOT) / 'TCM_Fusion/运行结果')
    source = project_path(source_dir)
    if root == source or root.is_relative_to(source):
        raise IntegrationError('FIVE_VIEW_SCOPE', '输出根不能位于输入图像目录内')
    base = root / ('case_' + uuid.uuid4().hex[:8])
    base.mkdir(parents=True, exist_ok=False)
    status = {'schema': 'five-view-interface/v1', 'dry_run': bool(dry_run),
              'base': str(base), 'stage': 'copying', 'status': 'running'}
    write_json(base / 'interface_status.json', status)
    try:
        images = []
        for index, row in enumerate(bindings, 1):
            original = Path(row['source'])
            image = base / f'view_{index:02d}{original.suffix.lower()}'
            shutil.copyfile(original, image)
            if file_hash(image) != row['sha256']:
                raise IntegrationError('FIVE_VIEW_HASH', '复制期间原图内容发生变化')
            images.append(str(image))
        cfg = bind_config(template, images, sample_id=base.name, a_python=a_python)
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
            raise IntegrationError('FIVE_VIEW_NOT_READY', f'本地检查失败，见{base / "doctor.json"}')
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
