# vision

图像与模型适配，处理出框、分块、特征身份、裁图质量和诊断审计。

| 文件 | 阅读指引 |
| --- | --- |
| [box_quality.py](box_quality.py) | 读取裁图灰度像素和可选框尺寸。 |
| [detector_yolo.py](detector_yolo.py) | 读取模型和配置，结合权威类别表将 YOLO 结果转换为原图坐标检测框。 |
| [feature_audit.py](feature_audit.py) | 从调查结束时的证据快照汇总有效裁图、分类和 mask，写旁路报告及连通域目检图。 |
| [feature_profiles.py](feature_profiles.py) | 从项目 taxonomy 提取逐特征默认说明，叠加配置中同特征的正例、易混负例和几何提示。 |
| [image_exemplar_audit.py](image_exemplar_audit.py) | 将目标裁图和指定特征的正负示例拼成带标签的图像，并将返回分数转为审计 JSON。 |
| [image_tiling.py](image_tiling.py) | 将原图像素宽高划分为覆盖全幅的网格，输出数量受限的原图坐标框。 |
| [presence_localization.py](presence_localization.py) | 把 presence 观察的可选 box 投影为影子定位线索，供账本与规划器参考，不产生正式工件。 |
| [stage2_classifier.py](stage2_classifier.py) | stage2 适配层：训练侧 cls_net → crop 显微特征身份概率 → workflow_backend。 |

直接依赖 runtime。

对应回归见 [tests/vision](../../tests/vision/README.md)。

[返回上级导航](../README.md)
