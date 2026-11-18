"""候选置信度工件的共同约束。提供版本摘要、模型身份检查和温度映射，供发布和在线读取复用。
由「lib/decision/routing_policy.py」、「lib/offline/confidence_gate.py」调用，结果按调用方的任务阶段消费。

主要定义：
- seal。
- check。
- model_identity。
- calibrated。

关联：
- 调用方 「lib/decision/routing_policy.py」、「lib/offline/confidence_gate.py」。
- 直接依赖 「lib/runtime/storage.py」。
"""
import copy
import math
from TCM_Fusion.lib.runtime.storage import IntegrationError, digest


def seal(value):return {**value,'sha256':digest(value)}


def check(value,kind):
    if value.get('schema')!=kind or value.get('sha256')!=digest({k:v for k,v in value.items() if k!='sha256'}):
        raise IntegrationError('GATE_ARTIFACT','门控工件格式或哈希错误')


def model_identity(a):
    s=a.get('signature') or {}
    if not s.get('weights') or not s.get('script') or not s.get('python_sha256'):
        raise IntegrationError('GATE_MODEL','预测缺模型身份')
    return digest({'weights':s['weights'],'script':s['script'],'python':s['python_sha256'],
                   'classes':sorted(r['label'] for r in a['ranking'])})


def calibrated(a,temperature):
    result=copy.deepcopy(a);rows=result['ranking']
    logs=[math.log(max(r['probability'],1e-12))/temperature for r in rows]
    m=max(logs);zs=[math.exp(v-m) for v in logs];z=sum(zs)
    for r,p in zip(rows,zs):r['probability']=p/z
    result['ranking']=sorted(rows,key=lambda r:r['probability'],reverse=True)
    return result
