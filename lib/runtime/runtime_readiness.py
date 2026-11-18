"""读取模型配置与调用环境，检查模型名、服务根地址和凭据配置的一致性。返回不含凭据正文的结构化诊断，供 doctor 与真实文本规划启动前检查。

主要定义：
- profile_checks: 对指定配置段返回固定文字检查清单（配置存在性/编码/地址/模型名/凭据一致性），不返回密钥、原始 JSON 或服务响应。
- planner_preflight: 在 planner=llm 时汇总启动前检查；CPU/heuristic 路径不受模型凭据约束。

关联：
- 调用方 「lib/application/pipeline.py」、「lib/application/workflow_entry.py」、「tests/runtime/test_readiness.py」。
- 直接依赖 「lib/runtime/storage.py」。
"""
import os
from pathlib import Path
from urllib.parse import urlsplit

import paths
from TCM_Fusion.lib.runtime.storage import read_json


def profile_checks(sections=('vlm',)):
    """仅返回固定诊断文字，不返回密钥、原始JSON或服务响应。"""
    path=Path(paths.DETECTION_CONFIG_DIR)/'vlm_config.json'
    def row(name,ok,code,detail):
        return {'name':name,'ok':bool(ok),'code':'OK' if ok else code,'detail':detail}
    if not path.is_file():
        return [row('模型运行配置',False,'CONFIG_MISSING',
                    '缺目标检测/config/vlm_config.json；从同目录.example生成本地配置')]
    try:
        if path.read_bytes().startswith(b'\xef\xbb\xbf'):
            return [row('模型配置编码',False,'CONFIG_ENCODING_UNSUPPORTED',
                        '原B读取器要求UTF-8无BOM；不要用PowerShell5.1默认UTF8编码重写配置')]
        cfg=read_json(path)
        if not isinstance(cfg,dict):raise ValueError()
    except (OSError,ValueError):
        return [row('模型配置解析',False,'CONFIG_INVALID','配置必须是有效UTF-8 JSON对象')]
    checks=[]
    for section in dict.fromkeys(sections):
        values=cfg.get(section)
        if not isinstance(values,dict):
            checks.append(row(section+'配置段',False,'CONFIG_INVALID','缺少所需模型配置段'))
            continue
        endpoint=values.get('base_url');model=values.get('model')
        try:
            parsed=urlsplit(endpoint) if isinstance(endpoint,str) else None
            endpoint_ok=bool(parsed and parsed.scheme in {'http','https'} and parsed.hostname
                             and not parsed.username and not parsed.password and not parsed.query and not parsed.fragment)
        except ValueError:endpoint_ok=False
        checks.append(row(section+'服务地址',endpoint_ok,'CONFIG_ENDPOINT_INVALID',
                          'base_url须为兼容服务根地址，不在URL中携带凭据；此检查不证明服务可达'))
        checks.append(row(section+'模型名称',isinstance(model,str) and bool(model.strip()),
                          'CONFIG_MODEL_MISSING','model须由账号实际开通的模型名称填写'))
        literal=values.get('api_key','');env_name=values.get('api_key_env')
        if not isinstance(literal,str) or (env_name is not None and (not isinstance(env_name,str) or not env_name.strip())):
            checks.append(row(section+'凭据字段',False,'CONFIG_INVALID','api_key/api_key_env须为字符串'))
            continue
        env_value=os.environ.get(env_name,'').strip() if env_name else ''
        literal=literal.strip()
        conflict=bool(literal and env_value and literal!=env_value)
        checks.append(row(section+'凭据一致性',not conflict,'CREDENTIAL_AMBIGUOUS',
                          '原规划与检测客户端优先级不同；不要同时配置不同的文件密钥和环境密钥'))
        # 不跨厂商静默借用VLM密钥；同账号确需共用时显式指定同一环境变量即可。
        checks.append(row(section+'凭据已提供',bool(literal or env_value),'CREDENTIALS_MISSING',
                          '请在实际启动进程中设置api_key_env指定的变量；不要把密钥写进报告'))
    return checks


def planner_preflight(config):
    """新workflow真实LLM启动前检查；CPU/heuristic路径不受模型凭据约束。"""
    agent=config.get('agent',{});workflow=agent.get('workflow',{})
    if agent.get('planner','llm')!='llm':
        return {'ready':True,'required':False,'checks':[],'network_tested':False}
    checks=profile_checks(('vlm',))
    return {'ready':all(c['ok'] for c in checks),'required':True,'checks':checks,'network_tested':False}
