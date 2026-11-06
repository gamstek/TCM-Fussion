# 离线回归测试

所有测试用临时目录、fixture、mock或假模型，不证明识别准确率；验证契约、权限、预算、错误处理与输出不被改坏。

主目录：five_view、workflow、adaptive_quality、react、p0_priority、retry_policy、service_failures、time_budget、code_guard、integration、readiness、final_assessment、ab_fusion、candidate_scope、output_projection、association_policy。

`offline/`：reflection、prompt_optimization、workflow_prompt、evaluation、dataset_split、confidence_gate、routing。迁移不是停用；顶层统一selftest递归发现两组。`fixed_baseline.py`为固定B模拟对照辅助，`prompt_support.py`为离线提示测试替身；fixtures/recorded_pipeline.json只用于契约回归，不是线上配置。

项目根运行：`.venv\Scripts\python.exe -X utf8 -m TCM_Fusion.cli.main selftest`。外部原B59/21/125组也在同一命令内。禁止通过删离线断言让清理看似通过。
