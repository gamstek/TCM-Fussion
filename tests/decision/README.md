# decision

候选范围、审查、融合、收尾与交付投影。

| 文件 | 阅读指引 |
| --- | --- |
| [test_ab_fusion.py](test_ab_fusion.py) | 核验 80∶20 B 分、量纲缺口、佐证去重、封顶、裁决分差及 A/B 融合。 |
| [test_candidate_audit.py](test_candidate_audit.py) | 用真实候选审查投影核对逐折候选与已取得的证据。 |
| [test_candidate_scope.py](test_candidate_scope.py) | 候选范围门回归：B 只能从 A Top-K 取候选，非 A 必须严格证明。 |
| [test_final_assessment.py](test_final_assessment.py) | 收尾研判回归：只读判定、来源区分、共享预算与非法引用回退，不发起真实模型请求。 |
| [test_output_projection.py](test_output_projection.py) | always-output 投影的离线回归；不联网、不执行阶段。 |

统一命令和测试输入边界见 [tests 导航](../README.md)。
