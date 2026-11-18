# YOLO 14 类标注集审计与图片数据集

生成时间：2026-11-09  
源目录：`E:\目标检测标注`  
模型：`TCM_Fusion/yolo-results/detection/yolo_runs/v1_yolov10s_1024/weights/best.pt`  
运行环境：`.venv/yolo`，CPU  
推理参数：`imgsz=1024, conf=0.25, iou=0.45, max_det=300`  
匹配规则：类感知贪心匹配，`IoU>=0.5`

## 重要限制

这是从 `E:\目标检测标注` 中每类随机抽取 **30 张**，共 **420 张 / 1307 个 GT 框** 的审计样本：

```text
不是完整 4,961 张
不是你的 P0 test 993 张
E 集很可能与 v10s 训练数据重叠，属于近似 in-sample
```

因此以下数字只能用于**观察错误结构**，不能直接与 P0 test 的 52.3% / 43.4% / 4.3% 作严格对比。

## 样本结果

```text
GT boxes      1307
预测框        1037
TP            818
漏检 FN       489
类别错误      29
背景 FP       190
```

折算：

```text
正确率      818 / 1307 = 62.6%
漏检率      489 / 1307 = 37.4%
类别错误    29 / 1307 = 2.2%
背景 FP     190 / 1037 = 18.3%
```

与 P0 test 的方向一致：

```text
漏检是主要问题
类别混淆很少
组织类簇仍是主要混淆源
淀粉粒/针晶等小目标或成团目标召回低
```

但不能声称数值一致；样本和训练重叠不同。

## 分类型结果

| GT 类 | TP | 类别错误 | 背景 FP |
|---|---:|---:|---:|
| 其他 | 49 | 3 | 22 |
| 分泌组织 | 37 | 2 | 14 |
| 导管 | 14 | 10 | 6 |
| 气孔 | 31 | 1 | 3 |
| 淀粉粒 | 165 | 0 | 42 |
| 石细胞 | 33 | 2 | 14 |
| 纤维 | 31 | 2 | 7 |
| 花粉粒 | 63 | 0 | 2 |
| 草酸钙方晶 | 119 | 2 | 21 |
| 草酸钙簇晶 | 83 | 0 | 10 |
| 草酸钙针晶 | 96 | 0 | 33 |
| 菌丝 | 38 | 5 | 12 |
| 表皮细胞 | 30 | 2 | 4 |
| 非腺毛 | 29 | 0 | 0 |

主要混淆方向仍集中在：

```text
导管 → 纤维 / 表皮细胞 / 其他
菌丝 → 其他 / 表皮细胞
石细胞 → 其他 / 分泌组织
纤维 → 石细胞 / 表皮细胞
表皮细胞 → 其他
```

## 目录结构

```text
crops/
  <predicted_class>/
    <gt_class>__<source_stem>__pXX.jpg

positives/
  needle_crystal/
    ...jpg            # 预测为针晶且 TP

needle_false_positives/
  <gt_class>/
    ...jpg            # 预测为针晶但非 TP；含漏定位的针晶和少量其他类，仅作审计，不是纯负例库

tissue_negatives/
  for_needle/
    <gt_class>/
      ...jpg          # 从 GT 框裁出的组织类图像，用于人工构建针晶负例库

mismatch/
  <gt_class>__to__<pred_class>/
    ...jpg            # 类别混淆 crop

missed_gt/
  <gt_class>/
    ...jpg            # GT 未被任何预测匹配的 GT 框裁剪

dataset_manifest.csv   # 逐预测框：source/gt/pred/conf/IoU/match/crop
dataset_summary.json   # 按 GT 类的 match 统计
dataset_dataset_summary.json # 生成后的图片数据集统计
summary.json           # 推理总账
per_image.jsonl        # 逐图计数
predictions.jsonl      # 全部预测框
```

当前文件数：

```text
crops 1037
positives 96
needle_false_positives 34
mismatch 29
tissue_negatives/for_needle:
  纤维 37
  表皮细胞 32
  导管 31
  石细胞 55
  分泌组织 52
missed_gt 323（其中淀粉粒 221、针晶 21）
```

`needle_false_positives` 中 34 个包含：

```text
32 个 GT=针晶但 IoU/定位不匹配
2 个 GT=分泌组织但预测为针晶
```

所以它不能直接当“针晶负例库”；真正可用于构建负例库的是 `tissue_negatives/for_needle`，仍需人工复核。

## 与用户 P0 test 的关系

用户 P0 test：

```text
993 图 / 1882 GT
985 正确 / 817 漏 / 80 类别错误 / 486 FP
```

本样本：

```text
420 图 / 1307 GT
818 TP / 489 FN / 29 类别错误 / 190 FP
```

结构一致，数值不同：

- 本样本正确率更高；
- 漏检率更低；
- 类别错误更低；
- 背景 FP 更低。

最可能原因：

```text
E 集与训练集重叠
样本量小
抽样不是 P0 的 993 张
```

因此**不能把本结果当作独立验证**；独立结论仍以 P0 test 为准。

## 用途

这个目录适合用于：

```text
1. 快速检查 14 类在 E 集上的输出结构；
2. 从 hard_negatives/needle_crystal 中挑真正困难负例；
3. 从 missed_gt 中挑漏检样本，修复标注单元/训练样本；
4. 从 mismatch 中挑组织类簇的混淆样本；
5. 不应直接把全部 hard_negatives 当成负例库，需人工复核。
```

特别是：

```text
tissue_negatives/for_needle 中约 207 个组织 crop
```

这些是从 GT 框裁出的组织图像，可作为人工构建负例库的候选；仍需人工看形态后再入负例库，且不能把整张组织图当负例。

`needle_false_positives` 不是纯负例库：其中 32 个其实是“GT 为针晶但定位不到 IoU 0.5”的漏定位样本，只有 2 个是其他类被预测为针晶。

## 复现命令

推理与构建是一次性脚本，未写入生产代码。关键参数：

```text
每类 30 张
imgsz=1024
conf=0.25
iou=0.45
class-aware matching IoU=0.5
random seed=20261109
```

[返回运行结果索引](../../00_INDEX.md) · [负例特征库](../../config/negative_feature_bank.json)
