# cli

命令入口、MCP 进程入口和隔离 worker。命令从项目根执行。

| 文件 | 阅读指引 |
| --- | --- |
| [api_worker.py](api_worker.py) | 读取调用方写入的请求 JSON，运行图像编码或模型请求，并将结果写入指定路径。 |
| [classify_worker.py](classify_worker.py) | 单 crop 类别复核 worker：在受控子进程内复用检测侧 `classify_crop`，从项目根以 `-m` 启动。 |
| [cnn_classifier_worker.py](cnn_classifier_worker.py) | 受控 stage2 worker：context → 训练侧 cls_net 推理 → workflow 身份概率，从项目根以 `-m` 启动。 |
| [image_audit_worker.py](image_audit_worker.py) | 图片示例审计 worker：拼 contact sheet 并发起一次 VLM 旁路判断，从项目根以 `-m` 启动。 |
| [main.py](main.py) | 提供真实任务、就绪检查、多图准备和统一回归命令。 |
| [mcp_server.py](mcp_server.py) | 项目根启动绑定样品的 MCP stdio 服务 worker；协议 stdout 仅由官方 SDK 写入。 |
| [offline.py](offline.py) | 离线验证/发布入口：项目根 `python -m TCM_Fusion.cli.offline <command>`。 |
| [workflow_worker.py](workflow_worker.py) | 固定检测侧库包装 worker：全图/局部检测、重新标定与 GLM 定向定位，从项目根以 `-m` 启动，绝不 import CLI。 |
| [yolo_worker.py](yolo_worker.py) | 受控 YOLO 检测 worker：只输出原图坐标与模型溯源，从项目根以 `-m` 启动。 |

[返回上级导航](../README.md)
