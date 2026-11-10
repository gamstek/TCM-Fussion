# 自动整理的针晶示例候选池

来源：

```text
E:\目标检测标注\<类别>\labels\*.txt
```

label 是人工审核过的 GT 框。本文档中的“自动”指：**用代码从人工框和 feature_taxonomy 自动整理、筛选、裁剪**，不是重新人工审核。

## 产物

```text
all_candidates.csv
  全部候选：7,482
  positive_gt 647
  negative_gt 2,849
  other_gt 3,986

selected_manifest.csv
  自动筛选：170
  positive 50
  negative 120

exemplar_bank.json
  可直接供图片示例审计读取的 50 正例 + 120 负例

positive_gt/needle_crystal/
negative_gt/<feature_id>/
positive_contact.jpg
negative_contact.jpg
summary.json
```

负例分布：

```text
fiber            24
epidermis_cell   24
vessel           24
stone_cell       24
secretory_tissue 24
```

## 自动筛选规则

正例：

```text
GT 框
非触边
1.5 <= aspect <= 10
0.0005 <= area_ratio <= 0.03
按 area_ratio 升序、aspect 降序
取前 50
```

负例：

```text
组织类 GT 框
非触边
每类按 area_ratio 取前 24
共 120
```

这些规则只是**几何启发式**，不能真正判断：

```text
单根是否完整可见
是否成簇
是否模糊
是否只是组织局部
```

因此字段保持：

```text
review_status = auto_unreviewed
visibility = auto_single_heuristic / auto_tissue
hard_negative = unverified
split = unverified_e_training_overlap
```

## 与 JSON 描述信息的关系

`all_candidates.csv` 和 `exemplar_bank.json` 已把 `feature_taxonomy.json` 写入每条记录：

```text
taxonomy_priority
taxonomy_description
taxonomy_igm_desc
taxonomy_qa_shape
taxonomy_measurement
```

负例还对应：

```text
TCM_Fusion/config/negative_feature_bank.json
```

因此示例图片与形态描述已经能成对读取，但**图片示例审计 worker 尚未实现**。

## 能用于什么

可以用于：

```text
1. 接入图片示例审计的候选数据
2. 作为 50 正例 + 120 负例的初始 exemplar bank
3. 生成示例 contact sheet
4. 训练/审计的候选池
```

不能直接用于：

```text
1. 声称已人工审核
2. 独立测试集
3. 计算泛化准确率
4. 自动证明负例全部正确
```

## 关键限制

```text
E:\目标检测标注 很可能与训练集重叠
```

所以：

```text
split = unverified_e_training_overlap
```

它不能替代 P0 test 的 993 图独立评测。

## 结论

已经完成**不需要人工逐个裁剪**的数据整理：

```text
人工 GT 框
+ 代码自动裁剪
+ taxonomy JSON 描述
+ geometric heuristic
= 50 正例 + 120 负例的 exemplar bank 候选
```

但它是：

```text
auto_unreviewed candidate bank
```

不是：

```text
human-reviewed benchmark
```

[返回数据集索引](../00_INDEX.md)
