# -*- coding: utf-8 -*-
"""融合层的有界决策 LLM：只做 review 修复策略与候选准入，不产生测量或最终分数。

设计边界：
- 输入只给结构化证据（review 字段、候选 checks、A 候选 id），不传原图/真值；
- 输出必须是有限枚举 JSON；非法输出由调用方回退到确定性策略；
- 这里不写证据、不写 verdict、不改 rule_engine 的 score。
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import paths
from .storage import IntegrationError, digest, read_json


REVIEW_ACTIONS = {'relocate', 'split_component', 'diagnostic_measure', 'abandon'}
ADMISSION_ACTIONS = {'admit', 'verify', 'suppress'}


class DecisionLLM:
    """复用 adjudicator 的文本传输；只接受有限枚举结果。"""

    def __init__(self):
        from adjudicator import AdjudicatorClient
        config_path = Path(paths.DETECTION_CONFIG_DIR) / 'vlm_config.json'
        self.client = AdjudicatorClient(config_path)
        self.config = read_json(config_path).get('vlm', {})
        if not self.client.api_key:
            raise IntegrationError('CONFIG_MISSING', '缺决策 LLM 凭据')
        self.cache = {}

    def _ask(self, kind, payload):
        key = digest({'kind': kind, 'payload': payload})
        if key in self.cache:
            return self.cache[key]
        prompt = (
            '你是显微鉴定融合链路的决策审查器，只做两件事：'
            '① 根据分割 review 的结构化原因选择修复策略；'
            '② 根据候选证据判断非 A 候选是否应准入。'
            '不得编造测量、不得给药材结论、不得修改分数。只返回一个 JSON 对象。'
        )
        if kind == 'review_repair':
            prompt += (' review_repair 必须返回 '
                       '{"repair":"relocate|split_component|diagnostic_measure|abandon",'
                       '"component_ids":[整数,...],"component_policy":"largest|selected|all|merge",'
                       '"reason":"..."}。component_ids 只能取 input.review.components 中的 id；'
                       '只有 split_component/diagnostic_measure 需要 components。')
        user = json.dumps({'task': kind, 'input': payload}, ensure_ascii=False)
        body = None
        try:
            from .hard_calls import post
            self.client.timeout_s = min(float(self.client.timeout_s), 60.0)
            body = post({'model': self.client.model, 'temperature': 0,
                         'max_tokens': min(int(self.config.get('max_tokens', 1024)), 1024),
                         'response_format': {'type': 'json_object'},
                         'messages': [{'role': 'system', 'content': prompt},
                                      {'role': 'user', 'content': user}]},
                        self.client.timeout_s)
            text = body['choices'][0]['message']['content'].strip()
            if text.startswith('```'):
                text = '\n'.join(text.splitlines()[1:-1])
            match = re.search(r'\{.*\}', text, re.S)
            value = json.loads(match.group(0) if match else text)
        except Exception as exc:
            value = {'available': False, 'error': type(exc).__name__}
        if not isinstance(value, dict):
            value = {'available': False, 'error': 'invalid_json_object'}
        value['available'] = bool(value.get('available', True))
        value['cache_key'] = key
        self.cache[key] = value
        return value

    def review(self, review, candidate=None):
        payload = {'review': review, 'candidate': candidate or {},
                   'component_options': review.get('components') or []}
        value = self._ask('review_repair', payload)
        if value.get('available') and value.get('repair') in REVIEW_ACTIONS:
            valid = {int(c['id']) for c in (review.get('components') or []) if c.get('id') is not None}
            ids = []
            for raw in value.get('component_ids') or []:
                try:
                    cid = int(raw)
                except (TypeError, ValueError):
                    continue
                if cid in valid and cid not in ids:
                    ids.append(cid)
            policy = str(value.get('component_policy') or '')
            if policy not in {'largest', 'selected', 'all', 'merge'}:
                policy = 'selected' if ids else 'largest'
            if ids:
                policy = 'selected'
            return {**value, 'component_ids': ids, 'component_policy': policy}
        return {'available': False, 'repair': 'abandon',
                'component_ids': [], 'component_policy': 'largest',
                'reason': 'decision LLM unavailable or invalid; default abandon',
                'fallback': True}

    def admission(self, candidate, evidence, a_ids):
        payload = {'candidate': candidate, 'evidence': evidence, 'a_candidate_ids': a_ids}
        value = self._ask('candidate_admission', payload)
        if value.get('available') and value.get('decision') in ADMISSION_ACTIONS:
            return value
        # 确定性回退：非 A 候选、只有通用 required 证据 → verify
        checks = candidate.get('checks') or []
        specific = [c for c in checks if c.get('present') and c.get('role') == 'required'
                    and c.get('feature_id') not in {'epidermis_cell', 'secretory_tissue'}]
        decision = 'admit' if (candidate.get('id') in a_ids or specific) else 'verify'
        return {'available': False, 'decision': decision,
                'reason': 'decision LLM unavailable or invalid; deterministic fallback',
                'fallback': True}
