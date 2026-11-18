# 融合回归导航

从项目根运行统一离线回归。真实 LangGraph、MCP 和计算阶段可与明确的替身组合，各组 README 说明覆盖范围。

```powershell
.venv\Scripts\python.exe -X utf8 -m TCM_Fusion.cli.main selftest
```

新测试按主要验证对象归属。共享构造逻辑进入 support，静态输入进入 [fixtures](fixtures/README.md)。

| 组件 | 职责 |
| --- | --- |
| [application](application/README.md) | 任务入口、多图输入和准备流程。 |
| [agent](agent/README.md) | 动作选择、调查控制、优先级和任务内反思。 |
| [workflow](workflow/README.md) | 工作流权限、阶段执行、几何工件、重试与证据提交。 |
| [vision](vision/README.md) | 检测分类、图像分块、定位、视觉质量与旁路审计。 |
| [evidence](evidence/README.md) | 实例索引、语义尝试及暂定证据排序。 |
| [decision](decision/README.md) | 候选范围、审查、融合、收尾与交付投影。 |
| [runtime](runtime/README.md) | 进程、资源预算、环境就绪、代码签名及异常行为。 |
| [integration](integration/README.md) | 跨组件闭环，包含真实 LangGraph、MCP stdio 或原几何 worker 配合可控输入。 |
| [offline](offline/README.md) | 离线划分、路由测试、置信门、提示优化、反思和冻结工件的回归。 |
| [support](support/README.md) | 可复用的测试替身和工件构造器。测试包引用这些辅助函数。 |

[返回融合入口](../README.md)
