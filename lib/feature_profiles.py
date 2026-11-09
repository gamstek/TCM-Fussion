"""特征 profile：以 taxonomy 为全集，避免把针晶提示写进所有特征。

profile 只提供提示/负例/几何标签，不决定工具是否可用；模型不支持的特征仍由
worker 的受信类别表拒绝，不能因为 profile 存在就假装可识别。
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import paths
from .storage import read_json


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
