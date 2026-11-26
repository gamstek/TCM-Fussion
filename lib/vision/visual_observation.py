"""在已绑定的视觉动作上携带 Agent 的可选观察委托。

接口：selected_briefs 将本轮委托绑定到实际选择。prompt_suffix 和 receipt 对应视觉请求和回执。
调用方：ReAct、原图观察和受控视觉 worker。观察文字只供规划使用，物理操作键及正式判据由各自组件维护。
"""
import json
import re

from TCM_Fusion.lib.runtime.storage import IntegrationError, write_json


VISUAL_TOOLS = frozenset({'presence_check', 'scene_check', 'locate_feature', 'feature_scan',
                          'classify_target', 'inspect_crop'})
_PRIVATE = re.compile(r'(?:https?|file|data)://|data:image/|[A-Za-z]:[\\/]|\\\\|'
                      r'\b(?:bearer\s+|api[_ -]?key\b|authorization\b|sk-|lsv2_)', re.I)


def plain_text(value, limit=None):
    if not isinstance(value, str):
        return ''
    text = value.strip()
    if limit is not None and len(text) > limit:
        return ''
    return text if text and text.isprintable() and not _PRIVATE.search(text) else ''


def _brief_text(value, kb):
    text = plain_text(value)
    if text and kb and any(name in text for name in (*kb.herbs, *kb.groups)):
        return ''
    return text


def validate_brief(action, kb=None):
    if 'visual_brief' not in action:
        return
    if action.get('tool') not in VISUAL_TOOLS or not _brief_text(action['visual_brief'], kb):
        raise IntegrationError('VISUAL_BRIEF_INVALID', '观察委托须绑定视觉动作，使用不含候选名称的单行文字')


def resolve_guidance(requested, selected, kb=None):
    """记录本轮视觉动作的委托选择。只有模型明确写出的有效问题会进入视觉请求。"""
    modes = {a['action_id']: 'unspecified' for a in selected if a and a['tool'] in VISUAL_TOOLS}
    briefs = {}
    if not isinstance(requested, dict):
        return briefs, modes, int(requested is not None)
    rejected = 0
    for key, value in requested.items():
        if key in modes:
            if value == {'use': 'preset'}:
                modes[key] = 'explicit_preset'
                continue
            if text := _brief_text(value, kb):
                modes[key] = 'agent_brief'
                briefs[key] = text
                continue
            modes[key] = 'invalid'
        rejected += 1
    return briefs, modes, rejected


def selected_briefs(requested, selected, kb=None):
    """兼容只消费有效委托和拒绝数量的调用方。"""
    briefs, _, rejected = resolve_guidance(requested, selected, kb)
    return briefs, rejected


def binding_key(action):
    target = action.get('args', {}).get('target_id')
    return (action['tool'] + '/' + target) if target else action['action_id']


def for_dispatch(action, briefs):
    text = briefs.get(binding_key(action))
    return {**action, 'visual_brief': text} if text else action


def prompt_suffix(brief=None, *, open_observation=False):
    if brief is None and not open_observation:
        return ''
    question = plain_text(brief)
    if brief is not None and not question:
        raise IntegrationError('VISUAL_BRIEF_INVALID', '观察委托格式无效')
    focus = ('\n规划器对当前目标的待查问题（仅作观察方向）' + json.dumps(question, ensure_ascii=False)
             if question else '')
    return (focus + '\n仅依据目标图像回答，允许报告不同结构。原字段按原任务独立判读；即使细节未见，目标仍可能存在。'
            '在同一 JSON 增加 visual_observation 对象，finding 写实际可见细节或无法判读的原因，'
            'location 写目标区内位置，无法定位填空，readability 取 clear、partial 或 unreadable。'
            '推测、参照图和药材结论不能充当实见。')


def receipt(brief, response):
    """主机记录原委托，模型只能填写所见、部位和可读性。缺项不会变成阴性。"""
    row = {'brief': brief} if brief else {}
    report = response.get('visual_observation') if isinstance(response, dict) else None
    if isinstance(report, dict):
        finding = plain_text(report.get('finding'), 320)
        raw_location = report.get('location', '')
        location = plain_text(raw_location, 100)
        readability = report.get('readability')
        if finding and readability in ('clear', 'partial', 'unreadable') and (location or raw_location == ''):
            return {**row, 'status': 'reported', 'finding': finding, 'location': location,
                    'readability': readability}
    return {**row, 'status': 'unreported'}


def record(work, observation):
    path = work / 'visual_observation.json'
    write_json(path, observation)
    return {'visual_observation_file': str(path)}
