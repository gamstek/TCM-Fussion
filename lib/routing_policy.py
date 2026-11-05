"""A原始完整排序→启发式分流；仅消费推理结果，不校准、不推造B证据。"""
from .storage import IntegrationError, digest, finite_number


def validate_policy(config):
    if config is None:
        return None
    if not isinstance(config, dict) or set(config)-{'mode', 'min_score', 'min_margin', 'buffer', 'top_k', 'release'}:
        raise IntegrationError('CONFIG_INVALID', 'routing存在未知字段')
    if config.get('mode') == 'validated':
        if not isinstance(config.get('release'), str) or not config['release']:
            raise IntegrationError('CONFIG_INVALID', 'validated需要经过风险复验的发布文件')
        if set(config)-{'mode','release'}:
            raise IntegrationError('CONFIG_INVALID', 'validated阈值来自发布文件，禁止现场覆盖')
        return dict(config)
    if config.get('mode', 'shadow') not in {'shadow', 'trial'}:
        raise IntegrationError('CONFIG_INVALID', 'routing仅支持shadow/trial，不支持未验证的validated')
    for key in ('min_score', 'min_margin', 'buffer'):
        if not finite_number(config.get(key), maximum=1):
            raise IntegrationError('CONFIG_INVALID', 'routing须显式配置有效分数阈值、差距与buffer')
    if config['buffer'] < 0.0001:
        raise IntegrationError('CONFIG_INVALID', 'buffer至少0.0001以覆盖文本舍入误差')
    if type(config.get('top_k', 3)) is not int or not 1 <= config.get('top_k', 3) <= 68:
        raise IntegrationError('CONFIG_INVALID', 'routing.top_k须为1～68整数')
    return {**config, 'mode': config.get('mode', 'shadow'), 'top_k': config.get('top_k', 3)}


def route(a, config, kb, aliases=None):
    cfg = validate_policy(config)
    if cfg is None:
        return {'mode': 'off', 'recommend_skip': False, 'skip': False, 'reason': 'legacy_path'}
    if cfg['mode'] == 'validated':
        from .confidence_gate import decide
        try: return decide(a, cfg['release'], kb, aliases)
        except (IntegrationError, OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
            return {'mode':'validated','recommend_skip':False,'skip':False,'calibrated':False,
                    'reason':getattr(exc,'code','GATE_UNAVAILABLE')}
    result = {'mode': cfg['mode'], 'policy_id': digest(cfg), 'calibrated': False,
              'rounding_decimals': 4, 'recommend_skip': False, 'skip': False,
              'reason': 'invalid_or_unavailable_a', 'top_k': cfg['top_k']}
    ranking = a.get('ranking') or []
    if a.get('status') != 'ok' or len(ranking) < 2:
        return result
    names = [r.get('label') for r in ranking]
    probs = [r.get('probability') for r in ranking]
    if (any(not isinstance(n, str) for n in names) or len(set(names)) != len(names)
            or any(not finite_number(p, maximum=1) for p in probs)
            or any(x < y for x, y in zip(probs, probs[1:]))):
        return result
    # CLI请求全排序；四舍五入后的总量允许最多每项半个末位误差。
    if abs(sum(probs)-1) > len(probs)*0.00005 + 1e-8:
        return {**result, 'reason': 'incomplete_distribution'}
    herbs = {h['herb'] for h in kb['herbs'] if not h.get('no_criteria')}
    # 首版只放行直接、明确的药材标签，别名/结构类只供候选规划。
    if names[0] not in herbs or (aliases or {}).get(names[0], names[0]) != names[0]:
        return {**result, 'reason': 'non_direct_label'}
    gap = probs[0]-probs[1]
    passed = probs[0] >= cfg['min_score']+cfg['buffer'] and gap >= cfg['min_margin']+cfg['buffer']
    return {**result, 'raw_score': probs[0], 'raw_margin': gap,
            'recommend_skip': passed, 'skip': passed and cfg['mode'] == 'trial',
            'reason': 'heuristic_pass' if passed else 'below_threshold_or_boundary'}
