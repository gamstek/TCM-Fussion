# evidence

证据与工件，包括内容寻址版本、实例关联、用途投影和录制元信息。

| 文件 | 阅读指引 |
| --- | --- |
| [artifact_index.py](artifact_index.py) | 绑定样品下的只读 B 工件索引：契约检查与像素质量检查分开标记。 |
| [evidence_projection.py](evidence_projection.py) | D2：循环内与最终研判共用的只读证据资格投影。 |
| [evidence_store.py](evidence_store.py) | 独立不可变证据快照库：内容寻址提交、依赖失效传播，原 B 工件永不覆写。 |
| [instance_ledger.py](instance_ledger.py) | 任务内实例索引与语义尝试键（D1）。 |
| [provisional_evidence.py](provisional_evidence.py) | 把已产生的分类结果转成暂定候选排序，供超时场景保留中间证据。 |
| [purpose_evidence.py](purpose_evidence.py) | 受信用途适配：只消费当前证据库绑定评估，把测量记录投影成规则引擎可用的行。 |
| [search_ledger.py](search_ledger.py) | 单智能体共享检索账本视图：只描述和汇总已有预留与结果，不改变证据、不调用模型。 |
| [stage_recording.py](stage_recording.py) | 绑定受控阶段输出与工作目录。 |

直接依赖 runtime。

对应回归见 [tests/evidence](../../tests/evidence/README.md)。

[返回上级导航](../README.md)
