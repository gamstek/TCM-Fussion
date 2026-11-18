# vision

检测分类、图像分块、定位、视觉质量与旁路审计。

| 文件 | 阅读指引 |
| --- | --- |
| [test_box_quality.py](test_box_quality.py) | 框内容质量评估回归：只阻断空白内容，不因成簇/相邻而拒绝分割。 |
| [test_feature_profiles.py](test_feature_profiles.py) | 特征 profile 回归：taxonomy 提供默认全集，覆盖项只作用于目标特征，不串扰其他特征。 |
| [test_image_exemplar_audit.py](test_image_exemplar_audit.py) | 图片示例审计回归：contact sheet 拼图与分数规范化，不调用真实 VLM。 |
| [test_image_tiling.py](test_image_tiling.py) | 在固定图幅上验证分块网格覆盖边缘、上限和小图行为，记录旧参数的兼容语义。 |
| [test_needle_observation.py](test_needle_observation.py) | 旁路针晶审计与 mask 拆分的离线回归，只验证非阻断输出，不证明识别准确率。 |
| [test_negative_feature_bank.py](test_negative_feature_bank.py) | 负例特征库结构回归：按 P0 测试错误归因构建，只作提示与审计，不替代召回。 |
| [test_presence_localization.py](test_presence_localization.py) | presence 可选定位与影子线索的边界回归：确定性与兼容性验证，不宣称真实 API 或真值。 |
| [test_stage2.py](test_stage2.py) | 验证显微特征分类模型档案、按 crop 内容组装的缓存键及阈值入账。 |

统一命令和测试输入边界见 [tests 导航](../README.md)。
