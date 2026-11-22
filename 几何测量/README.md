# 几何测量模块（第 4 步 · Step 4-geometry）

> 独立版的融合 worker 调用 [measure.py](measure.py)，项目根的 `selftest` 覆盖几何 worker 与交付契约。下文的区间核查、合成自测和伪掩码脚本在 `D:\TCM-Microscopic-Image-Recognition-main\几何测量` 使用。

流水线位置：**实例分割的 Mask 进 → 各实例几何特征（形状 + 真实尺寸 μm）出**，
供第 5 步 Agent 规则推理对照判据（形态 + 量化尺寸）裁决。

```
分割交付 mask + *_measure_manifest.csv ──► mask_io(统一读取) ──► descriptors(像素几何量) ──► ×px/μm 标定 ──► measurements.csv/jsonl
                        （px_per_um 取自清单）                                     ▲              │
                                                      validate_ranges.py ←─────────┘              │ → 判据核验报告
```

本目录**自包含**，不依赖、不修改 `标注体系/`（只读取其 `manifest.csv` 数据底表，
以及**分割交付的 `*_measure_manifest.csv`** 中人工实测的 `px_per_um`）。

---

## 一、输入契约（mask 是尺寸测量的唯一真值）

`measure.py --masks` 支持三种格式（`geomlib/mask_io.py` 自动识别）：

| 格式 | 说明 | 文件名约定 |
|---|---|---|
| 实例编号 PNG | 0=背景，1..255=实例编号，一张图一文件 | `<图像stem>__<feature_id>.png` |
| 每实例 PNG | 二值图（>0 为前景），一个实例一文件；**instance_id = 文件名 `<k>`**（稳定身份） | `<图像stem>__<feature_id>__inst<k>.png` |
| COCO JSON | `segmentation` 支持 polygon 与未压缩 RLE | categories 提供特征名 |

* `<feature_id>` 用特征本体键（`cluster_crystal`/`single_crystal`/`needle_crystal`/`vessel`/
  `stone_cell`/`starch_grain`/`fiber`/`crystal_fiber`/`secretory_tissue`/`pollen_grain`/`hypha`…），
  缺省回退 `generic` 通用剖面；
* 一个实例允许多连通域（针晶束 / 成束纤维 / 复粒），自动打 `multi_component` flag；
* feature 标签缺失时只做通用测量，范围核查返回 `no_reference`；
* **分割环节交付件**（《分割方案与设计》§1.3）：mask 为 **crop 局部尺寸**的二值图，按
  `<源图stem>__<feature_id>__inst<k>.png` 命名交付，**同目录带一份 `*_measure_manifest.csv`**
  （每实例的 `px_per_um`，分割/检测环节实测）。几何量在 crop 像素上计算后 ×px/μm，
  与 mask 是否为 crop 尺寸无关（同图多 crop 的 inst 序号连续不冲突）。
* 亦接受放回原图全 0 背景的**整图尺寸** mask：因 crop 未重采样、像素网格分辨率与原图一致，
  `px_per_um` 相同、μm 测量结果不变，仅 `bbox_x/bbox_y` 落入原图坐标系（与 crop 局部交付的差异仅在坐标框架）。

### 原像素视窗与量测坐标

同一张原图可能有两个大小相同、位置不同的裁图。几何入口先按实际 mask 文件名取清单行，再用 `crop_id` 找对应侧车，核对 mask 尺寸、紧框和视窗之间的坐标关系。[原像素取景规则](../lib/vision/README.md#feature-window-method)

`feature-window/v1` 的裁图直接取自原图。使用清单中的标定计算尺寸，原图坐标通过加上 `loose_box` 的左上角还原。JSONL 的 `crop_frame` 保存原图视窗、mask 外接框、Feret 端点，并记录是触及裁图边缘还是同时触及原图边缘。既有像素坐标字段仍以 mask 为坐标框架。

坐标绑定缺失或失配、前景触边、分割关联裁掉目标连通域时，保留诊断记录和 `igm_review`，主判尺寸留空。[坐标核验与回映](measure.py)

## 二、标定（px → μm）——只认分割交付清单，缺失即报错

全库倍率不统一（CV 31.9%），**绝不假设统一倍率**。比例尺测定是**检测阶段的职责**：
由 `标注体系/common/scalebar_duo.py` 双引擎（A 本地形态学+OCR / B VLM，交叉验证）逐图测出，
随 mask 一起交付。**`scale_table.csv` 已于 2026-09-20 全项目停用**，几何模块不读任何离线标定表、
也不做比例尺自动探测。逐图标定来源唯一：

1. **分割交付清单 `*_measure_manifest.csv` 的 `px_per_um`**（唯一来源，`calibration_source=delivery_manifest`、`calibration_status=ok`）；
2. 交付清单里没有该图的有效 `px_per_um` → **报错退出（exit code 2）**，一次列出全部缺项源图 stem、
   交付目录与期望文件名，**不产出任何无 μm 的测量行**。请先补测比例尺再跑测量。

> 变更说明：此前"未标定降级为 `uncalibrated`、只出像素量"的兜底已按 2026-09-20 决定取消——
> 静默降级会让下游 Agent 拿到一批只有像素值的记录，问题被掩盖而不是被暴露。

## 三、输出：每个实例的几何特征

`<prefix>_measurements.csv`（平面表）/ `.jsonl`(全量) 主要字段：

| 组 | 字段 | 对应药典措辞举例 |
|---|---|---|
| 主判尺寸 | `primary_dimension` / `primary_value_um` | 按特征自动选（见下） |
| 尺寸 | `feret_max/feret_min`（精确 Feret 长短径）、`eq_diameter`、`max_inscribed_dia`、`perimeter`、`area_um2`、最小外接矩形宽高与角度 | 簇晶"直径"、针晶"长"、方晶"长/径" |
| 细长结构 | `skeleton_len`（骨架长）、`mean/median_width`（沿骨架宽）、`tortuosity`（弯曲度）、`width_p25/p75` | 纤维/导管/菌丝"直径 8~25μm"、乳管管径 |
| 形状 | `aspect_ratio`、`eccentricity`、`solidity`（实心度）、`extent`、`convexity`（凹凸度）、`circularity`、`roundness`、`rectangularity` | 石细胞"分枝状"、方晶"菱形/长方形" |
| 结构 | `defect_count` + `mean_defect_depth_ratio`（凸包凹角计数/深度，**棱角钝锐**的明场代理）、`dt_peak_count`（复粒分粒数近似）、`n_components` | 簇晶"棱角多短钝"、淀粉粒"复粒由2~8分粒组成" |

**per-feature 主判尺寸**（`geomlib/profiles.py`）：簇晶/方晶/石细胞/淀粉粒/花粉粒 → `feret_max`（"直径"）；
针晶（含柱晶）→ `feret_max`（"长"）；纤维/晶纤维/导管/菌丝 → `mean_width`（"径/宽"）。

**字段载体约定（Agent `evidence_pool` 按此解析，改动需双侧同步）**：

| 字段 | CSV 载体 | JSONL 载体 | 备注 |
|---|---|---|---|
| `instance_id` | int | int | = 交付 mask 文件名的 `inst<k>`（实例编号图=像素编号；COCO=annotation id）。**稳定身份**：换目录/换批次重测不漂移，可回溯到 mask 文件与叠加图 `<stem>__inst<id>_测量.jpg` |
| `flags` | 分号连接串（`uncalibrated;multi_component`，空=无旗标） | 字符串数组 | 可能值：`empty` / `degenerate_tiny` / `multi_component` / `uncalibrated` |
| `calibration_status` | `ok` \| `failed` | 同左 | 仅 `ok`（源自交付清单 `px_per_um`）的 μm 值进量化统计；`failed`=未标定 → 仅存在性证据 |
| `bbox_x/y、bbox_w/h_px` | int | int | 与所测 mask 同一坐标框架：原图级 mask = 原图坐标；crop 级交付（分割 §1.3）= crop 局部坐标，不可与原图 bbox 直接混用 |
| 叠加图 | `<stem>__inst<instance_id>_测量.jpg` | — | crop 局部 mask 按交付 `crops/*.json` 的 `loose_box` 回贴源图对应区域（绿=轮廓、红=最大长径）；源图与 crop 小图都拿不到才退回灰底，文件名不变 |

**边界说明（诚实清单）**：脐点类型（星状/裂缝状）、导管纹理（网纹/螺纹）、壁厚/层纹、
木化反应、颜色、偏光双折射——这些需要灰度/纹理/多光谱信息，**纯 mask 几何测不到**，
留给 Agent 的形态描述与人工复核项（光学项本就因无配对偏振图像不进自动链路）。

## 四、判据范围核查（validate_ranges.py）

`geomlib/profiles.py` 内置 `REFERENCE_RANGES`：按《85味中药显微特征_修正版》逐条录入的
量化尺寸（大黄簇晶 20~160(190)μm、半夏针晶 20~144μm、板蓝根木纤维径 14~25μm、
山楂石细胞 19~125μm……共 40 余条；药典无数值者不设范围 → `no_reference` 而非误判）。

每条测量输出状态：`within` / `out_of_range`（附出界方向与边界值）/
`extreme_only`（超上限但落在"个别可达"极值内）/ `no_reference` / `uncalibrated`。
~~另执行大黄排他判据的几何可核查部分~~（2026-09-22 删除：原 `EXCLUSION_GEOMETRIC` 的"大黄簇晶
<45μm → 提示非正品"flag 出自示例文档、非药典依据；尺寸出界已由 `out_of_range` 状态覆盖）。

## 五、用法

```bat
:: 1) 批量测量（mask → 几何特征 μm；px_per_um 从分割交付清单自动读取）
python measure.py --masks <分割交付目录|PNG|COCO.json> ^
    --images ..\dataset\显微镜像 ^
    --manifest "..\标注体系\manifest.csv" ^
    --out-prefix out\batch01 --overlay-dir out\overlay

:: 2) 判据范围核查
python validate_ranges.py --measurements out\batch01_measurements.csv --out out\batch01_range_check.csv

:: 3) 自测（合成已知真值，33 项断言）
python selftest_synthetic.py

:: 4)（可选）分割模块上线前的冒烟：真实图 Otsu 伪 mask（非正式分割！）
python make_pseudo_mask.py --images ..\dataset\显微镜像\大黄草酸钙簇晶 --filter "^W2016" --out-dir pseudo_masks
```

`--split-policy test_full_train20`：按 manifest 的 `suggested_split` 落实
"量化属性只对 test 集全标、训练集抽 20%"（train 按 image+instance 的 MD5 确定性抽 20%；
`instance_id` = 交付文件名 `inst<k>` 后，抽样子集与 mask 目录怎么切分完全无关，跨批次严格可复现）。

## 六、与项目文档的衔接与已踩坑的对应

| 项目教训 | 本模块落实 |
|---|---|
| mask 是尺寸测量唯一真值，bbox 由 mask 派生 | 全部尺寸量从 mask 轮廓/距离变换/骨架计算，bbox 仅作派生输出 |
| 倍率 CV 31.9%，不可全库一率；比例尺测定归检测阶段 | 逐图标定只认分割交付清单 `px_per_um`；缺标定直接报错，不出 μm 假数、也不静默降级 |
| 标定随交付走，而非几何自测 | `calibration_source/status` 逐行落表；缺项在跑测量前就报错列出 |
| 中文路径 imread 静默失败 | `geomlib/imutil.py` 统一 fromfile+imdecode |
| Otsu 假溢出轮廓 | `make_pseudo_mask.py` 剔除贴四缘组件（正式分割上属分割环节） |
| 弱标签 stem 歧义 | manifest 完整路径优先 + stem 歧义告警（真实教训：`10.jpg` 多药材同名） |

## 七、自测与验收口径

* `selftest_synthetic.py`：33/33 PASS——圆/椭圆 Feret 精度 ±1.5px、星形簇晶凹角数、
  针晶长细比、弯纤维骨架宽 ±1.6px、导管管径 ±2px、复粒分粒数、μm 换算、标定缺失时的降级行为、
  交付清单 `load_delivery_manifest` 读取与 stem 匹配、**缺 px_per_um → measure.py 退出码 2 且报错点名**、
  **量化可用性闸门 L1**（表皮细胞/气孔恒不产 `primary_value_um` 且打 `coarse_quant`；细长/单粒特征不受影响）。
* 真实数据冒烟（分割交付 demo：manual_masks_measure 的 7 实例，2026-09-17 跑通）：
  标定全部 `delivery_manifest/ok`（px_per_um 12.3/11.7 取自交付清单），依次产出
  measurements → overlays → Agent 判决（单类证据 samples 均按 D8 门槛降级
  `insufficient_evidence`，符合设计）。
* 交接给 Agent 的验收线（引用全流程文档）：标定随分割交付；量化属性 test 全标、train 抽 20%。

## 八、目录结构

```
几何测量/
├── measure.py              批量测量入口（CLI）
├── validate_ranges.py      药典判据范围核查（CLI）
├── selftest_synthetic.py   合成真值自测（33 项断言）+ demo 物料生成
├── make_pseudo_mask.py     伪 mask 生成器（冒烟专用，非正式分割）
├── geomlib/
│   ├── imutil.py           中文路径安全读写
│   ├── calibration.py      读取分割交付清单 px_per_um + Calibration（scale_table 相关 API 已退役）
│   ├── mask_io.py          PNG/COCO mask 统一读取（输入契约）
│   ├── descriptors.py      几何描述子库（Feret/骨架/凹角/分粒…）
│   └── profiles.py         per-feature 测量剖面 + 药典参考范围
├── demo_masks/             合成 demo mask（自测产物，内含 demo__measure_manifest.csv，可复跑 measure.py）
├── demo_manifest.csv       demo 配套底表
├── pseudo_masks/           真实图伪 mask（冒烟产物）
└── out/                    demo / pseudo 的测量、核查结果与叠加图
    └── _stale_overlay_backup/  旧枚举命名叠加图备份（instance_id 契约重构前）
```
