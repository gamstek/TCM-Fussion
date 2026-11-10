# 针晶示例候选池（E 集人工框）

来源：

```text
E:\目标检测标注\<类别>\labels\*.txt
```

这些 label 文件是人工审核过的 YOLO 框，可以用于**第一步的数据集候选池**，但不能直接等同于最终的“图片示例库”。

## 已生成内容

```text
positive_gt/
  needle_crystal/        117 个 GT 针晶框 crop

negative_gt/
  fiber/                 37
  epidermis_cell/        32
  vessel/                34
  stone_cell/            55
  secretory_tissue/      51
  合计                   209

manifest.csv             326 条记录
summary.json             数量和限制
```

`manifest.csv` 字段：

```text
group
class
source_image
source_sha256
label_file
gt_index
box_px
width_px
height_px
aspect
area_ratio
touches_border
visibility
hard_negative
split
source
crop
```

## 可以直接用作什么

可以作为：

```text
1. 正例候选：单根针晶、成簇中可辨单根
2. 负例候选：纤维/表皮/导管/石细胞/分泌组织
3. 输入给人工审核的 contact sheet
```

因为框来自人工 label，不是模型预测，所以比直接用 `positives/` 的模型 TP 更干净。

## 还不能直接用作什么

不能直接当作：

```text
最终训练集
最终示例库
独立测试集
hard-negative 库
```

原因：

```text
visibility = unreviewed
hard_negative = false
split = unverified_e_training_overlap
```

### 1. 没有可见性标签

117 个针晶框里可能包含：

```text
单根完整可见
成簇中的可辨单根
整簇
部分截断
模糊
```

需要人工标：

```text
single_visible
cluster_visible_single
cluster_merged
partial
unknown
```

最终正例建议只收：

```text
single_visible
cluster_visible_single
```

### 2. 负例不是自动 hard negative

`negative_gt` 只是**GT 组织框**，不等于模型会误判为针晶的困难负例。

要变成 hard negative，还需要：

```text
模型在这些 crop 上预测为针晶
或
候选审计给出较高 needle_score
或
人工认定它与针晶形态最接近
```

本次 YOLO 审计中，真正“其他类被预测为针晶”的只有 2 个，所以不要把所有 GT 组织框都叫 hard negative。

### 3. 不能当独立测试

`E:\目标检测标注` 很可能与 v10s 训练数据重叠。

本池统一标记：

```text
split = unverified_e_training_overlap
```

在完成独立分组前，不能用于声称泛化效果，也不能作为 T2/T3 的独立评测。

## 建议人工审核流程

### 正例

从 `positive_gt/needle_crystal` 117 个中选 30–50 个：

```text
必须单根可见，或成簇中可辨单根
不要收整簇
不要收明显模糊/截断
记录 source_sha256
```

### 负例

从 `negative_gt` 209 个中选 80–150 个：

```text
fiber 30
stone_cell 20
secretory_tissue 15
epidermis 15
vessel 10
```

优先选：

```text
与针晶形态最接近
框内结构清晰
不是纯背景
不是整张组织图
```

并补：

```text
hard_negative=true/false
negative_reason
```

## 与当前负例特征库的关系

当前：

```text
TCM_Fusion/config/negative_feature_bank.json
```

是**文本形态特征库**。

本目录是**GT 框图片候选池**。

两者合并后才是：

```text
文本负例特征
+ 真实负例图片
+ 人工审核状态
```

## 结论

**可以用于第一步数据集。**

但正确说法是：

> 这些 label 是人工审核框，可以直接生成正负例 crop 候选池；  
> 但还需要人工做可见性、困难负例和独立分组审核，才能进入正式示例库。

在此之前，它们只能作为：

```text
candidate pool
```

不能作为：

```text
reviewed exemplar bank
independent benchmark
```

[返回数据集索引](../00_INDEX.md)
