# 在 LangSmith 查看一次鉴别

LangSmith Tracing 接收调用摘要并提供网页调用树。观察命令读取本机工件，可跟随正在执行的鉴别，也可将已完成的样品上传为回放。单图与多图都按样品组织调用，D4 的分支显示在同一调用组内。

## PowerShell 日常用法

在项目根运行脚本，再粘贴样品目录或内部的运行目录。也可以直接把路径作为参数传入。

```powershell
.\Observe-LangSmith.ps1
.\Observe-LangSmith.ps1 'D:\TCM_Fusion\运行结果\你的样品目录'
```

已完成的目录直接生成回放并退出。进行中的目录每两秒核对工件变化，任务结束后退出。可以在指定目录或 `job.json` 尚未创建时开启，终端会先提示等待。输入包含多个运行的样品目录时，传入具体的 `runs/fusion_<id>`。按 **Ctrl+C** 退出观察。

脚本自动读取本机 `config/langsmith.local.json` 的密钥与目的地，附上原记录中的模型选择简述。加 `-Preview` 可做本地预览。更换密钥时更新该文件中的 `api_key`。

## 准备网页目的地

1. 登录 LangSmith，选定工作区。若左侧是 **Tracing Projects**，新建一个名为 `TCM-Fusion` 的项目或选择已有项目。
2. 若工作区按 Agent 管理，在 **+ Agent → Connect → Connect an existing agent** 中注册外部 Agent。当前工作区已选用标识 `tcm-fussion` 和 **Local** 环境。在其他工作区使用时填写对应 Agent 的标识。[注册外部 Agent](https://docs.langchain.com/langsmith/create-an-agent#name-an-agent-traced-from-outside-langsmith)
3. 首次配置或更换账号时，在 **Settings → API Keys** 创建追踪密钥，写入 `config/langsmith.local.json` 的 `api_key`。若密钥跨多个工作区，在 **Settings → General** 复制对应的 Workspace ID 并写入 `workspace_id`。[密钥、工作区和 API 区域](https://docs.langchain.com/langsmith/create-account-api-key)

## 先用已完成的 case 预览

在 `D:\TCM_Fusion` 打开 PowerShell，把 `$case` 改成已有工件的样品目录。也可直接传入含 `job.json` 的 `runs/fusion_<id>/`。

```powershell
$python = '.\.venv\Scripts\python.exe'
$case = 'D:\TCM_Fusion\运行结果\你的样品目录'
& $python -X utf8 -m TCM_Fusion.cli.langsmith_trace $case
```

预览会列出此次能投影的调用、时间和字段。它以现成任务记录为输入，新增模型调用和 LangSmith 请求均为 0。已有的 Python 锁文件包含本桥接使用的 LangSmith SDK 版本。

## 把选中的追踪发到网页

Python 观察入口读取同一份本机配置。临时切换目的地可用下列环境变量覆盖。

Project 工作区使用项目名。

```powershell
$env:LANGSMITH_PROJECT = 'TCM-Fusion'
```

Agent 工作区使用注册时填写的标识和环境。本机配置已填写 `tcm-fussion / local` 和美国区 API 地址。换到其他目的地时可以显式设置环境变量。

```powershell
$env:LANGSMITH_PROJECT = $null
$env:LANGCHAIN_PROJECT = $null
$env:LANGCHAIN_SESSION = $null
$env:LANGSMITH_ENDPOINT = 'https://api.smith.langchain.com'
$env:LANGSMITH_AGENT_ID = 'tcm-fussion'
$env:LANGSMITH_AGENT_ENVIRONMENT = 'local'
```

需要指定工作区或区域时，在观察终端设置 `LANGSMITH_WORKSPACE_ID` 或 `LANGSMITH_ENDPOINT`。桥接通过显式命令控制上传。

```powershell
& $python -X utf8 -m TCM_Fusion.cli.langsmith_trace $case --upload
```

在网页切到刚选的项目或 Agent 环境，进入 **Tracing → Traces**，按命令输出的 trace ID 找到 **TCM_Fusion · 样品鉴别回放**。用 **Details** 展开具体调用，看开始、结束、输出和失败码。[LangSmith 追踪界面](https://docs.langchain.com/langsmith/view-traces)

## 看懂一次鉴别

展开 **规划请求 N**。**Input** 显示该轮落盘的活动候选、必需判据状态、待鉴别问题和短编号菜单。问题的 `direct_actions` 是当前可执行的核验步骤，`discovery_actions` 是定位实例的前置动作。**Output** 显示模型选择的编号、对应的视野与目标、工具和实报 token。匹配 `planning_request` 相同的工具调用，可以看到动作是否接受、是否推进证据获取以及新增的合格支持或反证。早期工件缺少的字段以空缺显示。

| 在调用树选中 | Details 里的内容 |
| --- | --- |
| **Plan A** | 各视野的五折首选、平均榜前三名及原模型报分。规划实际纳入的候选与判据见当轮 Input。 |
| **规划请求 N** | 当轮预算余额、所选动作、选择简述和实报 token。 |
| **并行观察** | 逐项显示视野、特征、观察置信度及可见／本视野未见／未能判读。这里的结果范围是单个视野。 |
| **样品根节点** | 交付结论、正式 B 裁决、B 分和融合排名、审查覆盖、合格实测中位数及参考界限。规划用量与终审用量分别显示，按已有回执汇总。 |

如需阅读模型随选择返回的简述，先用下列第一条命令检查内容，再用第二条命令发送。LangSmith 对同一调用只接受一次结束更新。归档回放直接提交完整的 Input、Output 和原始时间，使用独立的 trace ID。在同一目的地重复上传同一份回放会复用该 ID。[LangSmith 维护者说明](https://github.com/langchain-ai/langsmith-sdk/issues/2111)

```powershell
& $python -X utf8 -m TCM_Fusion.cli.langsmith_trace $case --include-decision-summary
& $python -X utf8 -m TCM_Fusion.cli.langsmith_trace $case --include-decision-summary --upload
```

## 跟随你正在运行的任务

在你启动正常鉴别后，将 `$case` 改成该任务目录，在观察终端加 `--follow`。

```powershell
& $python -X utf8 -m TCM_Fusion.cli.langsmith_trace $case --upload --follow
```

观察端每两秒检查工件变化。调用开始落盘后登记为进行中，收到对应反馈后提交一次结束。主任务结束时，观察端提交已有的剩余回执并退出。重新连接时，若发现先前的观察端已经提交过结束，观察端会在任务完成后上传完整回放，以终端最后输出的 trace ID 为准。

网页调用树包含 Plan A 的候选提名阶段、实际规划请求、工具及 D4 分支。多图工具显示 `view_01`、`view_02` 等序号。具有原始记录的规划附回传 token，工具输出可包含新支持、定量反证和逐候选判据状态变化。最终结论与复核原因在样品根调用下。时间线使用工件时间，缺少独立起止的子步骤留在已有调用摘要中。

默认上传运行 ID、时间、视野序号、工具及状态、实际报量与结构化判据摘要。源码、模型权重、图像、完整 Prompt、凭据和本机路径保存在本地。`--include-decision-summary` 单独开启最多 240 字符的模型选择简述，跳过带有常见路径或凭据特征的简述。

## 源码入口

从 [PowerShell 入口](../Observe-LangSmith.ps1) 进入 [观察命令](../cli/langsmith_trace.py) 的 `observe`，再到 [追踪投影](../lib/offline/langsmith_trace.py) 的 `project_trace`。`LangSmithPublisher.sync` 提交实时调用与归档回放。定向测试覆盖逐视野回执、目录后建、运行工件原子替换和重复上传。[定向测试](../tests/offline/test_langsmith_trace.py)
