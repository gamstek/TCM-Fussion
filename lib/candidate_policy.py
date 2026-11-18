"""把 A 的逐折 Top-K 预测映射为 Agent 候选兴趣清单，仅作查证入口而非显微证据。

历史记录缺少逐折排名时保留旧的整榜口径（按 top_k 与累计质量截断）；
候选分数未经校准，各折训练数据重叠，因此候选不满足 required，也不能冒充证据。

主要定义：
- validate_candidates: 校验 top_k 与 cumulative_mass 的取值范围，返回规范化配置。
- candidates: 按多视野逐折 / 单视野逐折 / 旧整榜三种口径生成候选 ID 与未映射项。
- criteria_diff: 输出候选的鉴别判据清单，只读展示，unknown 不等同否定。

关联：
- .storage: 使用 IntegrationError、digest、finite_number 做配置与概率校验。
- .workflow_session / .workflow_runtime: 在会话初始化与候选扩展时调用 candidates。
- .evidence_runtime / .candidate_scope: 复用 candidates 与 criteria_diff 生成本轮候选。
- .pipeline: 调用 validate_candidates 在离线流程中校验候选配置。
"""
from .storage import IntegrationError, digest, finite_number
from pathlib import Path


def validate_candidates(config):
    if config is None:
        return None
    if not isinstance(config, dict) or set(config)-{'top_k', 'cumulative_mass'}:
        raise IntegrationError('CONFIG_INVALID', 'candidate_policy字段非法')
    k = config.get('top_k', 3)
    mass = config.get('cumulative_mass', 1.0)
    if type(k) is not int or not 1 <= k <= 68 or not finite_number(mass, minimum=0.0001, maximum=1):
        raise IntegrationError('CONFIG_INVALID', '候选数/累计质量非法')
    return {'top_k': k, 'cumulative_mass': mass}


def candidates(a, kb, aliases=None, config=None):
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
            top_k = []
            for row in rows[:k]:
                label = row['label']
                herb = mapping.get(label, label)
                identifier = kb.group_of.get(herb, herb)
                if identifier in known:
                    selected[identifier] = None
                else:
                    unresolved[label] = None
                top_k.append({**row, 'candidate_id': identifier if identifier in known else None})
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
