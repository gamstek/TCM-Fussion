# experiments — IGM 分割实验代码

本目录是 **实验代码**，不属于生产管线；生产脚本仍是上一级的 `igm_segment.py`（未被本目录改动）。

实验目的：解决低对比小目标上 IGM 生成的 mask **明显偏小**的问题，验证两条改进思路
（① 提示词改为灰度概率输出；② 同图多次采样后聚合），并给出可复现的数据。

**完整结论、数据表与保留意见见**：`标注体系/docs/IGM分割_灰度提示与集成采样实验报告.md`

## 文件

| 文件 | 作用 |
|---|---|
| `igm_exp_common.py` | 公共工具：灰度提示词构造、检测框推导（兼容清洗前后三种 crop json schema）、软图读取/聚合度量、幂等调用 igm_segment |
| `exp_gray_prompt.py` | **实验A**：灰度概率提示词的机制验证（中间灰阶占比 / 与原图相关性 / 阈值扫描） |
| `exp_ensemble_vote.py` | **实验B**：多采样聚合对比 —— 方式1 先二值化再按 `255*x/n` 投票 vs 方式2 软图平均 + 操作点 |
| `exp_multisample_validate.py` | **实验C**：4 样本 × n=3 扩大验证（含较清晰的对照样本，检验低操作点是否过度扩张） |

## 运行

```powershell
$py = "d:\dataset\TCM-Microscopic-Image-Recognition\标注体系\labelstudio_env\Scripts\python.exe"
$ex = "d:\dataset\TCM-Microscopic-Image-Recognition\标注体系\segmentation\experiments"

# 默认行为：复用已有产物做离线分析，不消耗 API 额度
& $py -X utf8 "$ex\exp_gray_prompt.py"
& $py -X utf8 "$ex\exp_ensemble_vote.py"
& $py -X utf8 "$ex\exp_multisample_validate.py" --no-api   # 缺产物即报错，防误扣费

# 需要重新生成时才加 --force（消耗 IGM API 额度）
& $py -X utf8 "$ex\exp_ensemble_vote.py" --n 4 --force
```

**额度提示**：三个脚本默认**幂等**——产物目录已有 `meta.json` 时直接复用，不重复调用 API；
只有 `--force` 才会真正发起调用（实验 A 1 次 / B 4 次 / C 12 次）。

## 产物位置（`results/` 为 gitignore，不进版本库）

| 目录 | 内容 |
|---|---|
| `标注体系/results/segmentation/igm_seg_expA/<stem>/` | 实验A 的完整产物（含 `gen_raw.png`）+ 阈值扫描结论来自脚本输出 |
| `标注体系/results/segmentation/igm_seg_expB/run1..n/` | 实验B 各次采样产物；`soft_avg_prob.png`（软图平均=经验概率图）、`ensemble_x*.png`（各投票档） |
| `标注体系/results/segmentation/igm_seg_expC/<stem>/run1..3/` | 实验C 各样本各次产物；`<stem>/soft_avg_prob.png` |

对照基线为生产目录 `标注体系/results/segmentation/igm_seg/<stem>/`（**会随人工重跑而变化**，
报告中的"生产单次"均为报告时盘上读数）。