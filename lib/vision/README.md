# vision

图像与模型适配，处理出框、分块、特征身份、裁图质量和诊断审计。

交付后审计汇总输入视野数、有候选的视野数、单裁图和带示例复核次数。每张候选保留实际输入模式、回退原因和拒绝状态，供独立标注集复核。

| 文件 | 阅读指引 |
| --- | --- |
| [box_quality.py](box_quality.py) | 读取裁图灰度像素和可选框尺寸。 |
| [detector_yolo.py](detector_yolo.py) | 读取模型和配置，结合权威类别表将 YOLO 结果转换为原图坐标检测框。 |
| [feature_audit.py](feature_audit.py) | 从调查结束时的证据快照汇总有效裁图、分类和 mask，写旁路报告及连通域目检图。 |
| [feature_window.py](feature_window.py) | 按[形态取景规则](#feature-window-method)给定位框留原像素上下文。 |
| [feature_profiles.py](feature_profiles.py) | 从项目 taxonomy 提取逐特征默认说明，叠加配置中同特征的正例、易混负例和几何提示。 |
| [identity_review.py](identity_review.py) | 用已有 CNN 排名和出框类别取同一易混组的两张已确认示例，组装本来就要发出的单 crop 复核。检查人工示例、原图幅及可见部位。 |
| [image_tiling.py](image_tiling.py) | 生成均匀网格。定向定位只取合法局部 ROI，单块预算按部分覆盖返回。 |
| [presence_localization.py](presence_localization.py) | 把 presence 观察的可选 box 投影为影子定位线索，供账本与规划器参考，不产生正式工件。 |
| [stage2_classifier.py](stage2_classifier.py) | stage2 适配层：训练侧 cls_net → crop 显微特征身份概率 → workflow_backend。 |
| [visual_observation.py](visual_observation.py) | 按当轮已选动作绑定规划器自拟的观察委托，组装实际视觉请求，校验实见、部位及可读性。自由观察回执供下一轮调查；正式身份与尺寸仍按各自的工具核验。 |

直接依赖 runtime。

对应回归见 [tests/vision](../../tests/vision/README.md)。

<a id="feature-window-method"></a>

## 按鉴别结构选原像素视窗

同为定位紧框，颗粒类需要保住外缘和内部纹理，细长组织需要沿轴看到一段连续的壁。视窗据此保留上下文。检测给出的 `tight_box` 仍用于表示同一实例，`loose_box` 记录从原图切下的像素范围。[检测裁图入口](../../cli/workflow_worker.py)

| 特征 | 取景依据 |
| --- | --- |
| 簇晶、方晶、石细胞、淀粉粒 | 保留整个紧框并对称留边，给外缘、晶棱和颗粒内部的判别结构留出观察空间。 |
| 导管、纤维、分泌组织 | 框长边达到短边两倍时用框轴。其余情况下用紧框内的 Sobel 灰度梯度估计纹理方向，方向性足够稳定才沿该轴多留一段。弱纹理或方向不明确时对称留边。 |

设框短边为 `s`，长边为 `l`，均按原图像素计。每边基础余量是 `max(24, 0.05 × s, min(72, 0.18 × s))`。有可靠轴向时，把 `min(96, 0.20 × l)` 超过基础余量的部分投影到横竖方向，两端各留相应余量。最终窗口包含原分类范围，边界取整并夹在原图内。裁图直接截取原像素。纹理方向只决定留边，类别和可见结构由相应的分类与复核环节判断。[取景实现](feature_window.py)

定位紧框、原分类范围、视窗原点和原图校验值保存在裁图侧车中。Stage2 按 `classification_box_local` 取原分类子窗并把坐标纳入缓存签名。已有单实例视觉复核沿用对应的分类或身份子窗。分割读取完整视窗并按 `tight_box_local` 关联目标。几何由 mask 和标定计算，用视窗的左上角将外接框与端点平移回原图。[分类缓存](stage2_classifier.py) · [几何坐标与截断记录](../../几何测量/README.md#原像素视窗与量测坐标)

[返回上级导航](../README.md)
