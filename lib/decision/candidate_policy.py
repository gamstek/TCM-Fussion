"""把多视野逐模型榜、单图平均榜或历史逐折榜映射为 Agent 调查候选。

历史记录缺少逐折排名时保留旧的整榜口径（按 top_k 与累计质量截断）；
候选分数未经校准，各折训练数据重叠，因此候选不满足 required，也不能冒充证据。

主要定义：
- validate_candidates: 校验候选数及历史累计概率阈值。
- candidates: 提名候选，保留每个来源的排名和未映射标签。
- criteria_diff: 输出候选的鉴别判据清单，只读展示，unknown 不等同否定。

关联：
- 调用方 「lib/application/pipeline.py」、「lib/decision/candidate_scope.py」、「lib/workflow/evidence_runtime.py」、「lib/workflow/workflow_runtime.py」、「lib/workflow/workflow_session.py」、「tests/integration/test_integration.py」。
- 直接依赖 「lib/runtime/storage.py」。
"""
from TCM_Fusion.lib.runtime.storage import IntegrationError, digest, finite_number
from pathlib import Path


def validate_candidates(config):
    if config is None:
        return None
    if not isinstance(config, dict) or set(config)-{'top_k', 'cumulative_mass', 'multiview_per_model_top_k'}:
        raise IntegrationError('CONFIG_INVALID', 'candidate_policy字段非法')
    k = config.get('top_k', 8 if 'multiview_per_model_top_k' in config else 3)
    mass = config.get('cumulative_mass', 1.0)
    if type(k) is not int or not 1 <= k <= 68 or not finite_number(mass, minimum=0.0001, maximum=1):
        raise IntegrationError('CONFIG_INVALID', '候选数/累计质量非法')
    normalized = {'top_k': k, 'cumulative_mass': mass}
    if 'multiview_per_model_top_k' in config:
        per_model = config['multiview_per_model_top_k']
        if type(per_model) is not int or not 1 <= per_model <= 86:
            raise IntegrationError('CONFIG_INVALID', '多视野逐模型候选数须为 1 至 86 的整数')
        normalized['multiview_per_model_top_k'] = per_model
    return normalized


def _map_top_k(rows, k, mapping, kb, known, selected, unmapped):
    ranking = []
    for row in rows[:k]:
        label = row['label']
        herb = mapping.get(label, label)
        identifier = kb.group_of.get(herb, herb)
        if identifier in known:
            selected[identifier] = None
        else:
            unmapped[label] = None
        ranking.append({**row, 'candidate_id': identifier if identifier in known else None})
    return ranking


def _multiview_model_candidates(views, k, kb, mapping, known):
    selected, unmapped, per_view = {}, {}, []
    for view in views:
        models = view.get('model_predictions') or []
        if (view.get('status') != 'ok' or view.get('ensemble_method') != 'mean_softmax'
                or len(models) != 3 or len({model['model_id'] for model in models}) != 3
                or any(len(model['ranking']) < k for model in models)):
            raise IntegrationError('A_RECORD_INVALID', '多视野 A 须为每张图提供三个完整的模型榜')
        per_view.append({'image_stem': Path(view['image']).stem, 'model_top_k': [
            {'model_id': model['model_id'],
             'ranking': _map_top_k(model['ranking'], k, mapping, kb, known, selected, unmapped)}
            for model in models]})
    return {'source': 'multiview_model_top_k', 'views': per_view, 'ids': list(selected),
            'unmapped': list(unmapped), 'mapping_version': digest(mapping),
            'policy': {'multiview_per_model_top_k': k},
            'candidate_incomplete': bool(unmapped) or any(
                len(model['ranking']) > k for view in views for model in view['model_predictions']),
            'interpretation': '逐视野逐模型提名，再按候选 ID 取并集。未映射标签留待复核。'}


def candidates(a, kb, aliases=None, config=None):
    if a.get('ensemble_method') == 'mean_softmax':
        cfg = validate_candidates(config)
        k = cfg['top_k'] if config and 'top_k' in config else 8
        mapping = aliases or {}
        known = {row['id'] for row in kb.candidates()}
        views = a.get('view_predictions') or []
        per_model = (cfg or {}).get('multiview_per_model_top_k')
        if len(views) > 1 and per_model is not None:
            return _multiview_model_candidates(views, per_model, kb, mapping, known)
        ranking = a.get('ranking', []) if a.get('status') == 'ok' else []
        if len(ranking) < k:
            raise IntegrationError('A_RECORD_INVALID', 'A 集成平均榜不足指定候选数')
        selected, unmapped = {}, {}
        top_k = _map_top_k(ranking, k, mapping, kb, known, selected, unmapped)
        return {'source': 'ensemble_top_k', 'ids': list(selected), 'unmapped': list(unmapped),
                'ensemble_top_k': top_k, 'mapping_version': digest(mapping), 'policy': {'top_k': k},
                'candidate_incomplete': bool(unmapped) or len(ranking) > k,
                'interpretation': '按样品平均概率提名候选。映射后去重，未映射项留待复核。'}
    if a.get('view_predictions'):
        views = []
        ids, unmapped = {}, {}
        for view in a['view_predictions']:
            entry = candidates(view, kb, aliases, config)
            if entry.get('source') != 'fold_top_k':
                raise IntegrationError('A_RECORD_INVALID', '多视野输入缺少逐折结果')
            ids.update(dict.fromkeys(entry['ids']))
            unmapped.update(dict.fromkeys(entry['unmapped']))
            views.append({'image_stem': Path(view['image']).stem, 'fold_top_k': entry['fold_top_k']})
        return {'source': 'multiview_fold_top_k', 'views': views,
                'ids': list(ids), 'unmapped': list(unmapped), 'candidate_incomplete': True,
                'policy': entry['policy'], 'mapping_version': entry['mapping_version'],
                'interpretation': '每视野分别保留各折候选；不平均，不作为显微证据。'}
    cfg = validate_candidates(config)
    mapping = aliases or {}
    known = {row['id'] for row in kb.candidates()}
    folds = a.get('fold_rankings') if a.get('status') == 'ok' else None
    if folds:
        k = cfg['top_k'] if cfg else 3
        selected, unresolved, fold_top_k = {}, {}, []
        for fold, rows in enumerate(folds, 1):
            top_k = _map_top_k(rows, k, mapping, kb, known, selected, unresolved)
            fold_top_k.append({'fold': fold, 'ranking': top_k})
        return {'source': 'fold_top_k', 'ids': list(selected), 'unmapped': list(unresolved),
                'fold_top_k': fold_top_k, 'mapping_version': digest(mapping),
                'candidate_incomplete': bool(unresolved) or any(k < len(rows) for rows in folds),
                'policy': {'top_k': k},
                'interpretation': '逐折分数未经校准。各折训练数据重叠，候选仅供查证，不能作为显微证据。'}
    cfg = cfg or {'top_k': 1, 'cumulative_mass': 1.0}
    ranking = a.get('ranking', []) if a.get('status') == 'ok' else []
    selected, unresolved, mass = [], [], 0.0
    for row in ranking[:cfg['top_k']]:
        prob = row.get('probability', 0)
        if not finite_number(prob, maximum=1):
            raise IntegrationError('A_RECORD_INVALID', '候选概率非法')
        label = row['label']; herb = mapping.get(label, label)
        identifier = kb.group_of.get(herb, herb)
        mass += prob
        if identifier in known:
            if identifier not in selected:
                selected.append(identifier)
        else:
            unresolved.append({'label': label, 'probability': prob})
        if mass >= cfg['cumulative_mass']:
            break
    return {'ids': selected, 'unmapped': unresolved, 'selected_mass': mass,
            'remaining_mass': max(0.0, 1-mass), 'mapping_version': digest(mapping),
            'candidate_incomplete': bool(unresolved or mass+1e-4 < cfg['cumulative_mass']),
            'policy': cfg}


def criteria_diff(kb, ids, cards):
    """只读鉴别清单；不把unknown等同否定，不改变decide候选。"""
    rows = {r['id']: r for r in kb.candidates()}
    return [{'candidate_id': identifier,
             'criteria': [{**c, 'tier': kb.tier(c['feature']),
                           'observed_present': bool(cards.get(c['feature']) and cards[c['feature']].present)}
                          for c in rows[identifier]['criteria']]}
            for identifier in ids if identifier in rows]
