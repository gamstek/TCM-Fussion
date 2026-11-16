# 14 类 CNN 分类器探针

模型：`分类器设计与权重/cls_runs/abl_320_lb_e40_ls01_mix01_s20261031/best.pt`
（ResNet18, 320, letterbox, 14 类）

输入：`龙胆_hybrid_tile4` 的 21 个 GLM feature crop。

```text
worker_predictions.json  原始 top-3/entropy
predictions.csv          CSV
worker_context.json      推理上下文
```

CPU batch 推理约 0.67 秒；这是 batch，不是 21 次网络请求。

## 说明

- 分类器只做 14 类特征归类，不判药材；
- 没有 unreadable / not_feature 类；
- 本次 21 个 crop 无人工标签，只验证速度和输出形态；
- 测试集报告 accuracy 0.8629 / macro-F1 0.8349，但不等于本批准确率。
