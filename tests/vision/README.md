# vision

检测分类、图像分块、定位、视觉质量与旁路审计。

| 文件 | 阅读指引 |
| --- | --- |
| [test_box_quality.py](test_box_quality.py) | 框内容质量评估回归：只阻断空白内容，不因成簇/相邻而拒绝分割。 |
| [test_feature_audit.py](test_feature_audit.py) | 用真实快照提交验证初检、通用补扫、批量分类、工件失效和 mask 落点，视觉输入为本地替身。 |
| [test_feature_profiles.py](test_feature_profiles.py) | 特征 profile 回归：taxonomy 提供默认全集，覆盖项只作用于目标特征，不串扰其他特征。 |
| [test_identity_review.py](test_identity_review.py) | 七组已确认示例的双向配对及单请求、图幅、目标原生分辨率、可见部位与身份待定回归。 |
| [test_image_tiling.py](test_image_tiling.py) | 在固定图幅上验证分块网格覆盖边缘、上限和小图行为，记录旧参数的兼容语义。 |
| [test_negative_feature_bank.py](test_negative_feature_bank.py) | 负例特征库结构回归：按 P0 测试错误归因构建，只作提示与审计，不替代召回。 |
| [test_presence_localization.py](test_presence_localization.py) | presence 可选定位与影子线索的边界回归：确定性与兼容性验证，不宣称真实 API 或真值。 |
| [test_stage2.py](test_stage2.py) | 验证显微特征分类模型档案、按 crop 内容组装的缓存键及阈值入账。 |

统一命令和测试输入边界见 [tests 导航](../README.md)。
