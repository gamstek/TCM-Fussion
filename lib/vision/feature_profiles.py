"""从项目 taxonomy 提取逐特征默认说明，叠加配置中同特征的正例、易混负例和几何提示。生成供定位、身份分类及存在性观察使用的文本。特征资格与权限由运行时检查。

主要定义：
- load_taxonomy: 读取项目根 feature_taxonomy.json。
- default_profiles: 从 taxonomy 生成逐特征 profile（positive/negative/geometry/source）。
- build_profiles: 用配置覆盖默认 profile，仅接受已存在特征。
- profile_for: 返回单个特征的深拷贝 profile。
- hint_text: 拼出正例与易混负例提示文本。

关联：
- 调用方 「lib/decision/comparison_problems.py」、「lib/workflow/workflow_backend.py」、「tests/vision/test_feature_profiles.py」。
- 直接依赖 「lib/runtime/storage.py」。
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import paths
from TCM_Fusion.lib.runtime.storage import read_json


def load_taxonomy():
    path = Path(paths.PROJECT_ROOT) / 'feature_taxonomy.json'
    return read_json(path)


def default_profiles():
    taxonomy = load_taxonomy()
    profiles = {}
    for row in taxonomy.get('features') or []:
        fid = str(row.get('feature_id') or '').strip()
        if not fid:
            continue
        positive = str(row.get('igm_desc') or row.get('description') or '').strip()
        profiles[fid] = {
            'feature_id': fid,
            'name': row.get('zh_name') or fid,
            'positive': positive[:500],
            'negative': [],
            'geometry': row.get('category') or '',
            'source': 'taxonomy',
        }
    return profiles


def build_profiles(overrides=None):
    profiles = default_profiles()
    for fid, value in (overrides or {}).items():
        if fid not in profiles or not isinstance(value, dict):
            continue
        row = profiles[fid]
        if value.get('positive'):
            row['positive'] = str(value['positive'])[:500]
        if value.get('negative'):
            negatives = value['negative']
            if isinstance(negatives, str):
                negatives = [negatives]
            row['negative'] = [str(x)[:300] for x in negatives if str(x).strip()]
        if value.get('geometry'):
            row['geometry'] = str(value['geometry'])
        row['source'] = 'config'
    return profiles


def profile_for(feature_id, overrides=None):
    return deepcopy(build_profiles(overrides).get(str(feature_id), {}))


def hint_text(profile):
    if not profile:
        return ''
    parts = []
    if profile.get('positive'):
        parts.append('目标正例：' + str(profile['positive']))
    if profile.get('negative'):
        parts.append('易混负例（不要当目标）：' + '；'.join(profile['negative']))
    return ' '.join(parts).strip()
