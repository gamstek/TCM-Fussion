"""加载通过回放验证的经验工件。校验内容摘要和阈值，限制在显式模拟上下文中使用。
由「lib/agent/react_graph.py」、「lib/offline/reflection.py」调用，结果按调用方的任务阶段消费。

主要定义：
- check_candidate。
- load_memory。

关联：
- 调用方 「lib/agent/react_graph.py」、「lib/offline/reflection.py」。
- 直接依赖 「lib/runtime/storage.py」。
"""
from TCM_Fusion.lib.runtime.storage import IntegrationError, digest, read_json


def check_candidate(candidate):
    if (candidate.get("schema") != "reflection-candidate/v1" or candidate.get("status") != "candidate"
            or candidate.get("policy") != {"no_gain_limit": 1} or candidate.get("scope") != "simulation"):
        raise IntegrationError("REFLECTION_POLICY", "不支持的候选策略；禁止自由改写规则")


def load_memory(path, *, simulation):
    value = read_json(path)
    body = {k:v for k,v in value.items() if k != "sha256"}
    if value.get("schema") != "reflection-memory/v1" or digest(body) != value.get("sha256"):
        raise IntegrationError("REFLECTION_MEMORY", "经验版本校验失败")
    check_candidate(value["candidate"])
    if (not simulation or value.get("scope") != "simulation" or not value["validation"].get("passed")
            or value["policy"] != value["candidate"]["policy"]):
        raise IntegrationError("REFLECTION_SCOPE", "模拟回放经验不得进入真实识别")
    return value
