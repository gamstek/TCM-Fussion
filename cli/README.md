# cli

命令入口、MCP 进程入口和隔离 worker。命令从项目根执行。

| 文件 | 阅读指引 |
| --- | --- |
| [api_worker.py](api_worker.py) | 读取调用方写入的请求 JSON，运行图像编码或模型请求，并将结果写入指定路径。 |
| [classify_worker.py](classify_worker.py) | 单 crop 类别复核。在受控子进程中复用 `classify_crop`，按配置准备同图幅对照并记录目标区的可见结构。 |
| [cnn_classifier_worker.py](cnn_classifier_worker.py) | 受控 stage2 worker：context → 训练侧 cls_net 推理 → workflow 身份概率，从项目根以 `-m` 启动。 |
| [langsmith_trace.py](langsmith_trace.py) | 观察一个 case，将规划、逐视野回执、合格实测和 D4 分支投影成 LangSmith 调用。日常用 [PowerShell 入口](../Observe-LangSmith.ps1) 输入目录。 |
| [main.py](main.py) | 提供真实任务、就绪检查、多图准备和统一回归命令。 |
| [mcp_server.py](mcp_server.py) | 项目根启动绑定样品的 MCP stdio 服务 worker；协议 stdout 仅由官方 SDK 写入。 |
| [offline.py](offline.py) | 离线验证/发布入口：项目根 `python -m TCM_Fusion.cli.offline <command>`。 |
| [workflow_worker.py](workflow_worker.py) | 固定检测侧库包装 worker：全图/局部检测、重新标定与 GLM 定向定位，从项目根以 `-m` 启动，绝不 import CLI。 |
| [yolo_worker.py](yolo_worker.py) | 受控 YOLO 检测 worker：只输出原图坐标与模型溯源，从项目根以 `-m` 启动。 |

[返回上级导航](../README.md)
