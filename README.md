# TCM_Fusion｜真实五折CNN＋VLM Agent融合链路

现役入口是 `cli.main → pipeline → workflow_entry → workflow_runtime/session → LangGraph ReAct → MCP → B工具/worker → 原规则 → 融合及最终研判`。A每张图五折各自Top-3，不用平均榜替代五折输入。YOLO默认先出框；GLM原图观察可并行，局部定位是有条件补救，不是必经关卡。分割、标定和测量仍须真实工件及用途资格。

## 一条命令跑龙胆五图

从项目根运行（不是TCM_Fusion子目录）。先做本地预检：

```powershell
.venv\Scripts\python.exe -X utf8 -m TCM_Fusion.cli.main five-view --source-dir TCM_Fusion/运行结果/needle_5view_final_1f76fd81 --a-python D:/Anaconda/python.exe --dry-run --json
```

启动一次全新真实测试：

```powershell
.venv\Scripts\python.exe -X utf8 -m TCM_Fusion.cli.main five-view --source-dir TCM_Fusion/运行结果/needle_5view_final_1f76fd81 --a-python D:/Anaconda/python.exe --json
```

接口自行校验manifest与五张原图哈希、复制中性文件名、建立空A缓存任务、生成配置、doctor并调用既有链路。每条命令均创建新case；dry-run目录不直接续跑。输出JSON的directory在dry-run指准备目录，真实运行指runs下的运行目录。也可设TCM_A_PYTHON或传入本机其他A解释器；不自动切换环境。doctor不联网，不证明远端API可用。真实运行会产生费用。

[五图接口说明](龙胆五图测试运行方案.md)不是Agent注入材料；[单图手册](单图测试操作手册.md)用于理解单图配置。程序只消费JSON与图像，不执行文档里的脚本。

## 保留的组件

| 位置 | 职责 |
|---|---|
| cli/main.py | 只提供run、doctor、selftest、five-view |
| lib/five_view.py、config/five_view.json | 可调用五图接口及规范模板，无旧case图像路径和缓存 |
| lib/workflow_*、react_graph、tool_*、agent_tools、mcp_bridge | 最新主动流程、真实MCP、前置条件、模型规划及反馈 |
| cli/*_worker、lib/worker_bootstrap | 隔离解释器、API、YOLO、分类和B阶段执行；禁止删成普通演示代码 |
| lib/final_assessment、output_projection | 调查停止后仍综合判断；证据不足不强制确认 |
| skills、requirements-agent*.txt | 注册技能与锁定依赖；实际环境还依赖原A/B、vision/yolo解释器 |
| yolo-results | 当前best.pt、class_map及当前训练/评测来源指标；不包含旧权重/数据缓存 |
| tests、tests/offline | 主链契约与可选离线功能测试，统一selftest递归发现；模拟不是线上识别效果 |

`case_9cc59640`完整保留为历史参考，不能宣称验证了其结束之后的代码。五张源图位于needle_5view_final_1f76fd81，只保留图像与input_proof。它们跨两个group_key，同一物理样品关系未确认。

## 在线与离线分开

```powershell
.venv\Scripts\python.exe -X utf8 -m TCM_Fusion.cli.main selftest
.venv\Scripts\python.exe -X utf8 -m TCM_Fusion.cli.offline --help
```

离线入口保留LangMem提示优化/验证/发布、任务经验发布、按来源划分数据、真值隔离评估、质量回放、校准门及工作流回放。不能因不参与当前五图推理就删除这些回归和发布依赖。在线仍可显式加载通过验证的prompt_release；默认模板不加载未知提示。离线命令仅显式allow-network的优化操作可请求文本模型；并不等于自动训练。

旧demo与不带workflow的active运行器已移出活动包，不再提供命令。主动模式缺agent.workflow明确报错。历史recorded只作为tests/fixtures/recorded_pipeline.json的契约样本，正式CLI run不接受旧影子/录制入口。

## 调度与交付边界

模型在权限、预算和真实前置条件允许的动作中选择；现行初检优先约束仍在，未在清理中更改策略。P0在adaptive_v2仅建议；模板删除误导性p0_priority并显式冻结GLM阈值0.8/0.8、扩框1.5和工具列表。绑定五图不等于五图均执行所有B步骤，需查agent_trace和operations。override_exhausted是防循环停止码，不是诊断；正常停止后生成final_assessment，文本模型失败/硬截止时程序化回退。

主要产物：job.json、a_prediction.json、agent_trace.json、workflow/ops、agent_verdict.json、fusion.json、output.json、final_evidence.json、final_assessment.json、report.md。无目标/QA不通过时可能无mask或尺寸，不能为凑结果绕过真实证据。

审计与归档：项目根docs/方国昊/TCM_Fusion清理审计_CODE-066；历史文件移至项目根archive/TCM_Fusion_CODE-066_20261106，包内不保留可被误扫的旧Python。详见[修改记录](修改记录.md)、[统一调试经验](../融合链路/测试与调试经验.md)、[架构索引](融合链路文档/00_INDEX.md)。
