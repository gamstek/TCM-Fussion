# 主链与离线回归测试

所有测试用临时目录、fixture、mock 或假模型，不证明识别准确率；验证契约、权限、预算、错误处理、身份门和输出不被改坏。

## 主链测试

`tests/` 当前包括：

```text
multi_view
feature_profiles
negative_feature_bank
image_exemplar_audit
needle_observation
workflow
adaptive_quality
react
p0_priority
retry_policy
service_failures
time_budget
code_guard
integration
readiness
final_assessment
ab_fusion
candidate_scope
output_projection
association_policy
```

重点覆盖：

- `multi_view` 任意 N 张、SHA256/路径/重复校验、A 解释器缺失、dry-run/live；
- `WorkflowSession` 动态初始化/locate 预算、locate 去重、身份门、批量 screen 目标；
- `adaptive_v2` P0 建议、停止、复核和 final_assessment；
- 输出层不把 A 先验、locate 候选或假设写成确认证据。

## 离线测试

`tests/offline/` 包括：

```text
reflection
prompt_optimization
workflow_prompt
evaluation
dataset_split
confidence_gate
routing
```

迁移不是停用。`fixed_baseline.py` 是固定 B 模拟对照辅助；`prompt_support.py` 是离线提示测试替身；`fixtures/recorded_pipeline.json` 只用于契约回归，不是线上配置。

## 运行

项目根：

```powershell
.venv\Scripts\python.exe -X utf8 -m TCM_Fusion.cli.main selftest
```

`selftest` 递归发现两组，并包含外部原 B 规则/场景回归。禁止通过删离线断言让清理看似通过。

## 边界

- 模拟/固定 fixture 不等于真实 A/B/GLM/IGM 可用；
- 测试通过不等于当前工具能对任意图片产出 mask 或尺寸；
- `classify_batch`、YOLO batch/GPU、locate 批量并发尚未实现，测试中不应伪造其通过；
- 新改动如涉及真实流程分支，应在现有测试文件中补断言，而不是只测本地合成路径。

[返回 README](../README.md) · [多图接口](../多图测试运行方案.md) · [单图手册](../单图测试操作手册.md)
