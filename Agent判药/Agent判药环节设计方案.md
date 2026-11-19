# Agent 判药环节 · 设计方案

> 版本 v1.0（设计稿，暂不含代码实现）
> 上游衔接：目标检测 → 实例分割 → **几何测量**（`几何测量/measure.py`）→ **本环节**
> 配套阅读：`中药显微鉴别项目全流程说明.md` §4.6、`85味中药显微特征_修正版.md`、`_attach_大黄示例.md`、`标注体系/feature_taxonomy.json`

---

## 目录

1. [定位与设计原则](#一定位与设计原则)
2. [输入输出契约](#二输入输出契约)
3. [知识层：85味判据知识库](#三知识层85味判据知识库herb_criteriajson)
4. [证据层：从测量表到证据卡](#四证据层从测量表到证据卡)
5. [推理层：规则引擎 + Agent 编排](#五推理层规则引擎--agent-编排)
6. [输出：结构化鉴别报告](#六输出结构化鉴别报告)
7. [端到端走查：以大黄为例](#七端到端走查以大黄为例)
8. [评测与验收](#八评测与验收)
9. [工程结构与实施计划](#九工程结构与实施计划)
10. [风险与已知坑](#十风险与已知坑)
11. [开放问题（需确认后冻结）](#十一开放问题需确认后冻结)

---

## 一、定位与设计原则

### 1.1 在流水线中的位置

```
显微图像 ─► 目标检测 ─► 实例分割 ─► 几何测量 ─► 【Agent 判药】 ─► 鉴别报告
                                  measurements.jsonl ──┘（上游产物，只读；
                                  逐行内嵌 calibration_source/status 与 px_per_um）
```

> 标定来源（2026-09-20 契约）：几何只认**分割交付清单 `*_measure_manifest.csv` 的 `px_per_um`**，
> 拿不到即报错退出；标定来源/状态已逐行内嵌 measurements，判药层不读取任何标定表（README D6）。
> `scale_table.csv` 已全项目停用，比例尺由检测阶段 `scalebar_duo` 双引擎测定。

Agent 判药是流水线**最后一环**：把上游产出的结构化事实（检出了什么特征、每个实例多大、什么形状）对照 85 味药的**判据四元组**（形态 + 量化为自动裁决依据；光学/偏光项因数据集限制不参与自动裁决）做规则推理，输出结论与完整证据链。

> **2026-09-22 变更**：原判据四元组中的"排他性判据"及其一票否决逻辑**已整体拆除**（含 `EX-DH-1/2`、
> `suspect/suspect_adulterated` 裁决带、`suspect_subject`、报告"排他性检查"节）。理由：库内两条排他
> 规则的唯一出处是一份**示例文档**（《一中药显微鉴别标准示例大黄粉末显微特征.docx》），非药典依据；
> 以"与多味药材共有"的特征去否决真品，风险大于收益。裁决只由判据评分 + 裁决带 + D8 门槛做出。

### 1.2 技术路线：神经-符号系统（与 `_attach_大黄示例.md` 对齐）

- **符号层（规则引擎，纯 Python，确定性）**：候选生成、量化核验、评分、组级归并、裁决带。**最终裁决只能由这一层做出**——可解释、可复现、能执行硬约束。
- **神经层（Agent + VLM，GLM 系列）**：做规则做不了的三件事——
  ① P1/C 特征的图像级判读（纤维成束、分泌组织、非腺毛、矿物光泽等检测类别之外的证据）；
  ② 细粒度形态确认（导管网纹/螺纹、石细胞分枝状、脐点形态等"灰度/纹理类"属性）；
  ③ 报告叙述生成。
- **铁律：神经层只"补证据、写叙述"，不"改裁决"。** LLM 提出的一切定量说法必须引用上游测量记录（instance_id），VLM 的目视判断一律降级为弱证据并打标，裁决结果不可被 LLM 推翻。

### 1.3 设计原则

| 原则 | 落地方式 |
|---|---|
| 证据先行，药典为纲 | 判据知识库逐条溯源到《85味》原文；评分公式中每个加/减分都能指回某条判据和某个实例 |
| 面向全部 85 味，不特化单味药 | 知识库按统一 schema 覆盖 85 味（含 4 组"显微不可分组"、无判据的玄明粉、矿物/动物药特殊通道），大黄只是其中一条普通记录 |
| 无证据 ≠ 证据缺失 | 阴性证据（"该药应有的特征没检出"）必须乘以该特征的检出可靠性权重，防止 FOV 偏差导致误杀 |
| 防泄漏 | 判药输入中**不得出现 manifest 的 herb 字段**；文件夹名只允许在评测阶段当 ground truth 用 |
| 尺寸只信测量链 | μm 一律来自 measurements（mask 真值 + 分割交付清单 `px_per_um`），VLM 自估尺寸已证实不可靠，禁入 |
| 诚实的不确定性 | 检测类别覆盖不到、图像未标定、判据无量化范围、组内不可分——全部显式降级并在报告中声明，不硬凑结论 |

---

## 二、输入输出契约

### 2.1 输入（全部只读，不修改 `标注体系/` 与 `几何测量/`）

| 输入 | 来源 | 用途 |
|---|---|---|
| `<prefix>_measurements.jsonl/csv` | `几何测量/measure.py`（经 run_pipeline 落 `流水线数据/runs/<id>/measurements/`） | 核心证据：每实例 feature、主判尺寸 primary_value_um、形状组描述子（solidity/defect_count/roundness/dt_peak_count…）、calibration_status、flags。载体约定：`instance_id` = 交付 mask 文件名 `inst<k>`（稳定身份）；flags CSV 为分号串、JSONL 为数组；calibration_status ∈ ok\|failed（来源 delivery_manifest\|uncalibrated；px_per_um 由分割交付清单提供，几何精测量）。目录输入时同 prefix 的 csv/jsonl 只取 jsonl（防双份计数） |
| `*_measure_manifest.csv`（px_per_um） | `标注体系/results/segmentation/` 交付 | 几何标定的唯一来源（随 mask 交付的 px_per_um，由检测阶段双引擎实测）；标定来源/状态已逐行内嵌 measurements，判药不直接读此文件 |
| ~~`scale_table.csv`~~ | — | **已于 2026-09-20 全项目停用**（产出脚本 `06_scale_calibrate.py` 删除、文件移入 `archive/`）；判药本就不读取（见 README D6） |
| 检测/分割原始输出 | 第 2~3 步模块 | bbox、mask、检出置信度（measurements 未包含 conf 时需从上游补齐，见 §2.3） |
| 原图目录 `显微镜像/` | 数据集 | 证据叠加图：`judge.py --overlays <流水线数据/runs/<id>/measurements/overlay>` 按文件名 `<stem>__inst<id>_测量.jpg`（measure.py 产物契约）关联；报告生成时**复制进样本报告目录 `证据图/` 内嵌于判据核验表**，并从 `verdict.json.evidence_files` 记录原路径（零写入上游）。实例 crop 取图工具（`crop_instance`）待分割交付 crop 目录后补 |
| `manifest.csv` | `标注体系/` | 图像→样本（文件夹）归并、group_key/split；**herb 列在推理时屏蔽，仅评测用**。另支持 `judge.py --sample <名字>`：把同一样品（同一份送检粉末）的多张显微子图强制归并为 1 个样本裁决，缺省按 manifest 文件夹/图 stem 归并 |
| `feature_taxonomy.json` | `标注体系/` | feature_id 词表对账（只读引用，不回写） |
| `85味中药显微特征_修正版.md` | 项目文档 | 判据知识库的抽取语料（原文级溯源） |

> **重要约定**：`标注体系/` 与 `几何测量/` 正被其他对话编辑，本环节对它们**零写入**。`profiles.py` 的 `REFERENCE_RANGES`/`EXCLUSION_GEOMETRIC` 仅作为知识库构建脚本的**种子数据导入**，本环节维护自己的派生副本（见 §3.5）。

### 2.2 处理单元：样本（sample）

- 一个 **样本 = manifest 中同一药材文件夹下的同批图像组**（生产环境中即"一份送检粉末样品的一组视野图"）。
- 判决在**样本级**做出，图像级只产中间证据：单张视野只代表部分粉末，"这一张没看到簇晶"不能推出样品无簇晶。
- 多图证据汇聚规则见 §4.4。

### 2.3 需与上游确认补齐的字段

measurements 现有 CSV 列**不含检出置信度 conf 和 mask IoU 质量**，而证据加权需要它。两个方案（开放问题 Q1）：
- A（推荐）：分割模块输出时附带 per-instance conf，measure.py 透传——需上游一处小改；
- B：本环节另行读检测/分割原始输出，按 (image_stem, instance_id) join。

> **接口现状（对齐检查后修正）**：方案 B 的 join 键 `(image_stem, instance_id)` 只有在
> `geomlib/mask_io` 把 `instance_id` 定义为**交付文件名中的 `inst<k>`**（而非目录扫描序号）
> 之后才稳定——此前换目录/换批次重测编号会整体漂移，join 会悄悄错位。上游透传 conf 后，
> evidence_pool 现有代码路径自动生效（README D3 的默认 0.85 仅在 conf 缺失时使用）。
> 另："疑似待定（class 6）"实例上游已约定按 `feature_id=suspect` 裁剪交付（《分割方案》§1.2），
> 本环节 v1.0 的实际行为是 `pool_evidence` 过滤（不进评分，VLM 复判未实现，README D7），
> Q3 定案前不要按"已实现复判"设计下游流程。

### 2.4 输出

1. **`reports/<sample_id>/verdict.json`** —— 机器可读裁决（schema 见 §6.1）；
2. **`reports/<sample_id>/report.md`** —— 人读鉴别报告：结论、逐条判据核验表、证据图链、告警；
3. **证据图链**：报告内嵌（自包含）——叠加图复制进该样本 `报告目录/证据图/` 并内嵌于「判据核验表」的证据图列；源视野图复制进 `报告目录/源图/`。`verdict.json.evidence_files` 记录叠加图**原始路径**、`source_images` 记录源图路径（只引用不额外复制，零写入上游；实现口径见 README D5）；
4. **`run_summary.jsonl`** —— 批量运行台账（每样本一行：结论、用时、VLM 调用数、降级标记）。
   `verdict.json` 的 schema 标识为 `agent-judgement/v1`（见 §6.1）。

---

## 三、知识层：85味判据知识库（herb_criteria.json）

### 3.1 为什么需要独立知识库

- `feature_taxonomy.json` 是**特征本体**（30 类特征），不是**判据库**（85 味 × 各自判据）；
- `profiles.py::REFERENCE_RANGES` 已录入部分量化范围（约 30 余条），但只覆盖"有尺寸数字"的品种，且属于 `几何测量/`（不可写）；
- 85 味中大量判据是**纯定性**的（如"当归：韧皮薄壁细胞纺锤形，壁略厚，表面有极微细的斜向交错纹理"），需要结构化后才能参与推理。

因此本环节新建派生知识库 `Agent判药/data/herb_criteria.json`，**以《85味》原文为唯一语料**，`REFERENCE_RANGES` 作种子对账，构建后版本化冻结。

### 3.2 构建流程（LLM 辅助抽取 + 人工审核，两阶段，复用项目标注方法论）

```
85味原文逐条 ─► LLM 结构化抽取(判据四元组拆解) ─► 校验脚本(词表/尺寸/组归属对账) ─► 人工审核(对照药典) ─► v1.0 冻结
                     │                                                    ▲
                     └── 种子对账: profiles.REFERENCE_RANGES 差异清单 ──────┘
```

- 抽取时强制 LLM 输出原文来源（写入判据/记录的 `notes` 字段），人工审核逐条核对；
- 对账脚本输出三类差异报告：词表外特征名、量化范围与种子数据不符、组归属不一致——人工裁决后才入库。

> **溯源口径（v1.0 现状）**：实现后的 KB **不设** herb 级 `source_sentence` / `version` /
> `special_channels` 字段——原文溯源改由判据级 `notes` 与本文档 §3.2 的构建记录承担；
> "特殊制片项"（如大黄升华物）改为按普通 supporting 判据入库（见 §3.3 大黄示例）。
> 版本冻结由 `data/criteria_changelog.md` 记录，不写进每条 herb 记录。

### 3.3 记录 schema（每味药一条；v1.0 已冻结为下列字段名，设计稿早期的
feature_id/umin/umax/tier/morphology{zh,shape_focus,vlm_check}/字符串 condition 均已按实现回写）

```jsonc
{
  "herb": "大黄",
  "group_id": null,                      // 显微不可分组归属，见 §3.4
  "criteria": [                          // 本药"应出现"的特征判据
    {
      "feature": "cluster_crystal",      // 引用 feature_vocab 词表
      "role": "required",                // required(主判) | supporting(辅助)
      "quant": {                         // 无量化判据时为 null（如牡丹皮簇晶）
        "dim": "feret_max",              // 枚举（与 profiles 对账，audit A3 把关）：feret_max | mean_width
        "min": 20, "max": 160, "extreme": 190
      },
      "morph": "草酸钙簇晶，多角形星状集合体，棱角短钝",   // 形态描述原文（作 VLM 提问素材）
      "vlm": "簇晶棱角是否短钝（非尖锐）",               // 交给 VLM 的定性复核问题，可空
      "abundance": null                  // 甚多|极多|较多|众多=1.0；多见=0.95；可见|偶见|少见=0.9 → 软证据，见 §5.2
    },
    { "feature": "vessel", "role": "supporting", "quant": null,
      "morph": "具缘纹孔、网纹、螺纹及环纹导管，非木化",
      "vlm": "导管纹理类型（网纹/螺纹/环纹/具缘纹孔）", "abundance": null },
    { "feature": "starch_grain", "role": "supporting",
      "quant": {"dim": "feret_max", "min": 3, "max": 45, "extreme": null},
      "morph": "单粒类球形或多角形，脐点星状；复粒由2～8分粒组成", "vlm": "淀粉粒脐点是否星状",
      "abundance": "甚多" },
    { "feature": "sublimate_crystal", "role": "supporting", "quant": null,
      "morph": "升华物可见菱状针晶或羽状结晶（特殊制片项）", "abundance": null }
  ],
  "no_criteria": false,                  // true = 药典无显微判据（玄明粉），跳过自动判药并声明
  "notes": "偏光双折射判据保留为人工复核知识，不参与自动裁决"   // 可选；亦承载原文溯源
}
```

> **2026-09-22**：`exclusions` 字段（原 EX-DH-1 石细胞 / EX-DH-2 簇晶<45μm 两条大黄排他规则）
> 已从库中及引擎中整体删除，理由见 §1.1 变更说明。`audit_criteria.py` 的 A5 改为**残留检查**
> （库内再出现 `exclusions` 即报错），防止无源规则回流。

> 与初稿的实质差异：① tier 不再逐条入库，由知识库顶层 `feature_vocab`（P0/P1/C → 特征集）
> 派生，保证全库一致（划分依据与升级纪律见 **§3.6**）；② `shape_focus` 从 schema 移除——形态签名软核验改为 rule_engine 内置的
> `morphology_plausibility(feature_id)` 分特征规则集（阈值与评测联动调参，不占知识库字段）。

### 3.4 85 味全量覆盖的特殊处理（不能只照搬大黄模式）

通读《85味》后，判据分为六类情形，schema 全部兜住：

| 情形 | 味数/例子 | 处理方式 |
|---|---|---|
| ① P0 特征 + 量化判据 | 大黄、白芍、虎杖、半夏类、白术/苍术、山楂、肉桂、黄花类纤维药、柴胡/防风（油管）等 | 走主推理链（§5），量化核验完全可自动执行 |
| ② P0 特征、无量化 | 牡丹皮/赤芍簇晶、枳壳/陈皮方晶"成片"、诃子纤维 | 量化核验返回 `no_reference`，退化为"检出即支持"+ 形态签名软核验 |
| ③ 仅 P1/C 定性判据 | 当归（韧皮薄壁细胞纹理）、泽泻（纹孔群）、艾叶（T形非腺毛）、槟榔（内胚乳）、莱菔子（栅状细胞）等 | 检测链覆盖不到 → **VLM 图像级判读通道**（弱证据，按 §5.2 的 w(tier) 降权），结论上限封顶为"倾向"级，强制人工复核标记 |
| ④ 显微不可分组（4 组 9 味） | 赤芍≈牡丹皮；半夏≈法半夏≈姜半夏；枳壳≈陈皮；白术≈苍术 | 知识库建 `group` 记录，组内成员共享同一组判据；裁决输出**组级结论**（§5.4） |
| ⑤ 矿物/动物药 | 滑石、石膏、雄黄、牡蛎、水牛角、蝉蜕、青黛 | 无 P0 组织特征，走**整图 VLM 判读通道**（颜色/光泽/碎片形态），可靠性最低，报告显著声明"建议人工复核"；与植物类药判据天然不冲突（互斥候选池） |
| ⑥ 无判据 | 玄明粉（原文仅 "default"） | 显式标记 `no_criteria=true`：不参与自动裁决，报告输出"该品种暂无显微判据，跳过自动判药"——**不猜** |

另注意：⑤ 矿物药与 ③ 定性药的数量不少（合计约 20+ 味），意味着**本环节的准确率指标必须按判据类型分层统计**（§8.2），不能只报一个总数字。

### 3.5 与上游数据的关系（防写冲突）

```
标注体系/feature_taxonomy.json   （只读）──► 词表对账
几何测量/profiles.py             （只读）──► REFERENCE_RANGES 种子 + shape_focus 语义
85味中药显微特征_修正版.md        （只读）──► 唯一判据语料
          │
          ▼ 构建脚本 build_criteria.py（LLM抽取+对账+人工审核）
Agent判药/data/herb_criteria.json  （本环节自有、版本化、冻结后变更走 changelog）
```

每次上游 feature_taxonomy 或 85 味文档更新后重跑对账脚本，输出 diff 报告人工确认。

### 3.6 特征档位（P0/P1/C）的划分依据与升级纪律

本项目的档位有**两套并存的语义**，混用会直接改错权重，故在此显式区分。

**① 两套档位**

| 体系 | 位置 | 档位 | 语义 |
|---|---|---|---|
| 标注优先级 | `标注体系/feature_taxonomy.json` → `priority_tiers` | P0/P1/P2 | 数据与模型的**实现顺序**：P0=核心必做、P1=进阶、P2=可选扩展 |
| 证据通道 | `Agent判药/data/herb_criteria.json` → `feature_vocab` | **P0/P1/C** | **证据从哪来**：P0=通道A(实例级检测+测量 w=1.0)、P1=通道B(图像级/VLM 判读 w=0.5)、C=通道C(整图VLM 判读 w=0.3) |

> **运行时权威是 KB 的 `feature_vocab`**——`rule_engine.KnowledgeBase.tier_of` 由它派生，
> `w(tier)`/`sens(tier)` 与 `morphology_plausibility` 的分档全部取自这里。
> taxonomy 的 `priority_tiers` 只用于标注侧对账（`audit_criteria.py`），不参与推理。
> **本文档以下提到 P0/P1/C 一律指 KB 侧语义**（设计稿早期沿用的 "P2" 词已废弃，对应 KB 的 P1 或 C）。

**② 划分依据（KB 侧的三条判据）**

| 档位 | 通道 | 划分依据 | 典型特征 |
|---|---|---|---|
| **P0** | A 实例级检测+测量 | 检测类别已覆盖（YOLO 0~5）；且尺寸可由 **mask 几何量直接测出** → 可量化核验 | 簇晶、方晶、针晶、导管、石细胞、淀粉粒 |
| **P1** | B 图像级/VLM 判读 | 仍是**离散组织形态**，但需"成束/成片/跨实例"判读，或实例化成本高、量程不全 → 仅作存在性/弱量化 | 纤维、晶纤维、分泌组织、表皮细胞、木栓细胞、花粉粒、非腺毛、腺毛、硅质块、菊糖、菌丝、栅状细胞、气孔 |
| **C** | C 整图VLM 判读 | **非离散实体**，或需特殊制片/理化手段（几何与检测都拿不到） | 砂晶（过小）、钟乳体、升华物（特殊制片）、动物类、矿物类、靛蓝结晶、棕色物、薄壁细胞纹理、内胚乳细胞、真菌团块、糊化淀粉粒 |

**③ 两套档位的对照（词表收口已完成 2026-09-20）**

| 特征 | taxonomy（标注优先级） | KB（证据通道） | 说明 |
|---|---|---|---|
| 6 类 P0 特征 | P0 | P0 | 一致 |
| fiber / crystal_fiber / secretory_tissue / cork_cell / epidermis_cell / pollen_grain | P1 | P1 | 一致 |
| non_glandular_hair / glandular_hair / palisade_cell / stoma | P1 | P1 | **已对齐**（收口前 taxonomy 缺这 4 个词，已回补） |
| sand_crystal / sublimate_crystal | P2 | C | **已对齐**（收口前 taxonomy 未登记档位，已补） |
| 其余 C 级词：brown_content / endosperm_cell / fungal_mass / gelatinized_starch / indigo_crystal / parenchyma_cell | P2 | C | 一致（taxonomy 的 P2 对应 KB 的 C） |
| **inulin / silica_body / hypha** | P2 | **P1** | **语义差异（有意保留）**：taxonomy 视为"可选扩展"，KB 归入通道 B（形态可辨、有量程） |
| **animal_feature / mineral_feature / calcium_carbonate** | P2 | **C** | **语义差异（有意保留）**：KB 归入通道 C（整图判读） |

> **词表收口结果**：taxonomy 的 `features[]`（20→**30** 条）与 `priority_tiers`（19→**30** 条）
> 已与 KB `feature_vocab`（**30** 条）**完全对齐**——无缺项、无悬空、无重复；原悬空引用的
> `aleurone` 已移除（KB 未收录、85 味判据中亦无此特征）。
> 上表最后两行是**有意的语义差异而非缺口**：taxonomy 表达"标注优先级"、KB 表达"证据通道"，
> 映射声明已写入 taxonomy 的 `meta.tier_mapping_note`。
> **推理侧唯一权威仍是 KB 的 `feature_vocab`**；taxonomy 的 `priority_tiers` 只用于标注侧对账
> （`audit_criteria.py`），不参与推理。

**④ 升级纪律（三条，不得违背）**

1. **tier 表达"证据从哪来"，不表达"这味药多重要"**。把某特征升 P0 的真实含义是"必须改用通道 A 取证"，
   即**必须先有实例级检测器**；它不是一个"重要性"声明。
2. **升档必须与检测能力同步，否则净负收益**：
   - **权重通胀**：w 0.5→1.0，但证据仍来自 VLM/图像级 → 弱证据被当强证据计分（正是 D8 要防的假阳性）；
   - **灵敏度反噬**（更危险）：`sens(tier)` 0.4→0.8，使 `required` 缺失罚由 `0.3·0.5·0.4=0.06`
     升至 `0.3·1.0·0.8=0.24`（**4 倍**）。检测器不存在时，FOV 偏差造成的"未检出"会被系统性重罚，
     **弃权率反而升高**；
   - **形态把关缺失**：`morphology_plausibility` 对表皮细胞/气孔/腺毛/栅状细胞/菊糖/硅质块
     一律返回 0.85 中性，拿不到通道 A 的质检红利。
3. **升档对 D8 门槛零作用**。`_support_feature_ids` 只统计**不同 feature_id 的类数**，
   完全不看 tier（`score_contribution>0` 只要 w>0 即成立）。即使把全部 P1 升为 P0，
   "仅 1 类判据"的候选（当前 50/79）仍会弃权——**弃权的成因是判据类数不足，不是档位低**。

**⑤ 升档判据与流程**

升 P0 需同时满足：① 有（或可建）实例级检测器；② 量程齐、几何可量化；③ 共享度高（w 被低估的影响大，
如 fiber 用于 13 味、非腺毛 10 味、表皮细胞 9 味）。

```
① 建实例级检测器（detection 标注 + 训练 + label_config）
② 补 profiles.py 剖面（缺失的 primary_dims，如 non_glandular_hair）
③ 补 KB quant 量程
④ 最后 才把 id 在 feature_vocab 的 P1↔P0 之间移动（一行改动）
⑤ audit_criteria + 单测 + demo 回归，并登记 data/criteria_changelog.md
```

**⑥ 升档的真实收益**

不在"提高该药的分"（`C=S/S_max`，分子分母同步变大），而在**压低不该有该特征的药**——
`evaluate_candidate` 的"错误特征出现罚"有一道 P0 闸门（非 P0 特征不参与该罚分）。
升 P0 后，"某药判据不含纤维但检出 ≥3 个纤维实例"才会计罚，这才是 fiber / 非腺毛 / 分泌组织升级的价值所在。

> 备注（机制改进方向，未实施）：tier 目前在同时承担"取证通道"与"特征诊断力"两件事
> （纤维＝高频低特异；大黄升华物＝低频高特异）。若要根治，应**新增独立字段**（如 `specificity`），
> 并让 D8 从"数特征类数"改为"要求 ≥1 类高特异证据 + ≥2 类证据"。属机制变更，
> 需走 `ADR-001` 追记 + 全量回归。

---

## 四、证据层：从测量表到证据卡

### 4.1 证据卡（Evidence Card）

把 measurements 按样本聚合后，每个 (feature_id) 生成一张证据卡：

```jsonc
{
  "feature_id": "cluster_crystal",
  "present": true,
  "source": "detector",                // detector(实例级) | vlm_image(图像级) | vlm_whole(整图) | folder_prior(禁用)
  "n_instances": 12,
  "conf": {"max": 0.92, "median": 0.83},
  "quant_stats": {                     // 主判尺寸的稳健统计
    "dim": "feret_max",
    "n_measured": 10,                  // 剔除 uncalibrated 后
    "median_um": 86.4, "p25_um": 61.0, "p75_um": 118.2,
    "min_um": 48.1, "max_um": 171.3,
    "n_uncalibrated": 2
  },
  "shape_signature": {                 // 对齐 profiles.shape_focus 的实测摘要
    "defect_count_median": 6, "mean_defect_depth_ratio_median": 0.11, "solidity_median": 0.78
  },
  "instance_refs": ["B20190056#3", "B20190056#7", "..."],   // 证据可溯源
  "reliability": 0.86                  // 可靠性权重，见 §4.3
}
```

### 4.2 三条证据通道

> 通道 ↔ 特征档位（P0/P1/C）的**划分依据、对照关系与升级纪律见 §3.6**；本节只讲通道本身。

| 通道 | 证据来源 | 权重档 | 覆盖 |
|---|---|---|---|
| A 实例级（强） | P0 检测+分割+测量链 | w=1.0 × conf | 6 类 P0 特征，全部量化判据 |
| B 图像级（中） | P1 图像分类器（用 L0 文件夹弱标签训练）+ VLM 定向判读（"图中是否有纤维束？"） | w=0.5，VLM 判读再乘 0.8 | 13 类 P1 特征（纤维、分泌组织、木栓细胞、花粉粒、表皮/毛茸类、硅质块、菊糖、菌丝、栅状细胞、气孔） |
| C 整图判读（弱） | VLM 对样本全部图的整图描述与匹配 | w=0.3，且单项封顶 | 11 类 C 特征（矿物/动物类、砂晶、钟乳体、特殊制片如升华物、色素/薄壁纹理/内胚乳/真菌团块/糊化淀粉粒） |

VLM 判读的硬要求：结构化输出（JSON schema：present/conf/依据描述）、坐标遵循 0~1000 归一化约定（换新模型先抽 5 张目检——项目已踩坑条款）、同一样本的多次判读独立记录不互相污染。

### 4.3 可靠性权重（reliability，v1.0 已按实现回写——README D1）

```
reliability = conf_factor × quality_factor
  （通道权重 w(tier) 只在判据贡献与罚项里体现一次；rel 再含 w 会对弱通道双重降权，
    使 P1/C-only 药材永远进不了 probable 带，与"弱通道封顶为倾向"自相矛盾）

  conf_factor   = median_conf 映射（<0.5 的证据不参与强支持，只可触发存疑）；
                  上游未透传 conf 时按 0.85 默认并打 conf_assumed（README D3）
  量化有效性不入 rel：calibration_status ≠ ok（即 failed：该图未在分割交付清单列出
                  或无 px_per_um）的实例只作存在性证据（uncal_refs），量化核验走 UNCAL 档（q=0.5）
  quality_factor= 1 − 0.05 × min(multi_component 实例数, 4)，下限 0.8
                  （v1.0 仅罚 multi_component；degenerate_tiny 罚项留待过分割数据积累后增设）
```

### 4.4 样本级多图汇聚

- 尺寸统计用 **median + IQR**（不用 mean/max：max 对过分割碎块和个别巨晶都敏感；药典"个别可达 190μm"由 `extreme` 分支处理而非直接取 max）；
- 存在性判定：任一图可靠检出即 present；**"全图均未检出"才生成阴性证据**，且阴性强度随"该特征的通道灵敏度"打折（P0 检测器灵敏度高→阴性可用；P1/VLM 通道灵敏度未知→阴性只降 0.05 分，近乎忽略）；
- 同一样本多图间的批内一致性（同批应同判）作为 sanity check，冲突时整样本标记 `needs_review`。

---

## 五、推理层：规则引擎 + Agent 编排

### 5.1 总体流程

```
测量表+图目录 ─► 证据卡池 ─► ① 候选生成 ─► ② 逐候选评分
                                     │                       │
                                     ▼                       ▼
                        (Agent 可在此补充 B/C 通道证据)  ③ 组级归并 + 置信带 ─► ④ 报告生成
```

### 5.2 ① 候选生成与 ② 评分（符号层，确定性）

**候选生成**：取"任一 required/supporting 判据特征在证据卡池中 present"的药材（含组）。**不使用 manifest.herb / 文件夹名**。若候选池为空或全部弱证据 → 直接输出"证据不足"。

**量化核验五态**（对每条带 quant 的判据，用 primary 尺寸统计量核对）：

| 状态 | 判定 | 得分 q |
|---|---|---|
| IN_RANGE | median ∈ [umin, umax] | 1.0 |
| EXTREME | median 出界但 ≤ extreme（或少量实例落在 [umax, extreme]） | 0.7 |
| NO_REF | 判据无量化范围，检出即形态相符 | 0.6（形态签名明显不符时 0.2） |
| UNCAL | 检出但图像未标定，无法核尺寸 | 0.5（仅计数/形态证据） |
| OUT_OF_RANGE | median 显著出界且超 extreme | 0 → 转为负分 −0.5 |

**形态签名软核验**：由 `rule_engine.morphology_plausibility(feature_id, card)` 按特征分档打分（如簇晶：defect_count≥3→1.0、≥1→0.6、否则 0.4；淀粉粒：roundness≥0.65 或 dt_peak_count≥2→1.0；无形状信息→0.85 中性），规则集内置于引擎、与 §8 评测联动调参，**先宽后严**；知识库不再逐条携带 shape_focus 区间。

**丰度软证据**：药典"甚多/极多/较多/可见/偶见"映射到实例密度（每 mm² 检出数）的软区间，v1.0 权重设很低（0.1 档），待标注数据积累后校准。

**评分公式**：

```
S(h) = Σ_criteria  w(tier) · q(state) · morph · m(abundance) · rel_i   （支持分）
     − Σ_present_but_wrong  0.4 · rel_j                                （该药不该大量出现/尺寸不符的特征）
     − Σ_absent_required    0.3 · w(tier) · sens(tier)                 （主判特征缺失，按通道灵敏度打折）
置信度 C(h) = min(1, max(0, S_required(h)) / Σ_required w)   （**只由必备判据决定**：
                                              supporting 不进分子也不进分母，其作用是按 v1.2 意图
                                              "增强证据广度"——进 support_count / D8 门槛与并列排序；
                                              排序键 = (confidence, support_count, score)）
```

> 候选级 cap（2026-09-16 演进，见 ADR-001 §7）：若 `C(h)` 由单一判据支配（贡献占比 ≥0.55）且该判据量化跨度 ≥60μm，置信度额外设上限 0.60——抑制"宽区间独证"（如马齿苋 cluster 7~108）造成的武断 confirmed；多判据真药贡献分散不受影响。

**D8 最少独立证据门槛（2026-09-15 采纳，见 ADR-001 §3）**：阳性和解（`confirmed` / `probable` /
`group_level`）必须由 **≥2 类独立证据特征**支撑，否则一律降级。原因：单一类特征命中（如仅 1 类簇晶）
无法区分"是该药"与"该药恰有常见特征"，是 probable 假阳性的主要来源（实证：20250378 单类簇晶 →
`probable·虎杖 0.57`，而辣蓼/大黄/G1 仅一步之遥）。

- **独立证据特征**＝该候选 `checks` 中 `present`、状态非 `OUT_OF_RANGE`/`NOT_DETECTED`、
  且 `score_contribution > 0` 的**不同 feature_id**；**同一特征的多个量纲判据只计 1 类**
  （如射干针晶"长+宽"是两条判据、一类证据）；
- 常量 `MIN_SUPPORT_FEATURES = 2`；门槛代码位于裁决带赋值**之后**，对 `confirmed`/`probable`/
  `group_level` 一视同仁（组级也不豁免：单特征同样分不出组）；
- **D8b 单类证据条件放行（2026-09-22，见 ADR-001 §8）**：药典只给 1 个显微判据的品种
  （59/85 味；非组 47 味：红花/板蓝根/虎杖/荜茇…）在"必须 ≥2 类"下**永远**只能弃权，
  与该事实不符。故例外条件为**「≥2 类 或（唯一证据类 且 `decisive` 且 conf ≥ 0.75）」**：
  ① `decisive`＝该类证据已把 top1 与并列者分开（分差 ≥δ=0.10，或独立证据类数不同）；
  ② 置信度已达 confirmed 带。放行样本仍 `needs_review=True`。
  实证：白芍样本（单类簇晶 conf 1.00、分差 0.40）→ `confirmed·白芍`；
  虎杖样本（大黄/虎杖 conf 均 0.60、分差 0.00）与"方晶 G3/猪苓"、"极多簇晶"一律继续拦下。
  无量化 `required` 判据的品种（`NO_REF=0.6`）到不了 0.75，故本放行只对**有量化的单判据品种**生效；
- 不满足 → `verdict=insufficient_evidence`、`conclusion=None`，原 top1 插到 `alternatives` 首位
  （信息不丢），`needs_review=True`，`verdict.json` 记录 `support_feature_count`；
- 该门槛对**所有**阳性和解一视同仁（`confirmed`/`probable`/`group_level`），无豁免路径
  （原"排他升级豁免"随 2026-09-22 排他层拆除而废止）；
- 与 VLM 环的协同：`insufficient_evidence` 恰在 `augment_and_rescore()` 的介入时机内——被门槛
  拦下的样本若 VLM 补到第 2 类判据证据，重评分后可自然回到 `probable`。
  **门槛不降低召回，只拒绝"无佐证的独证"。**

**排他扫描（原 ②，2026-09-22 已删除）**：引擎不再扫描候选 `exclusions`、不再有 `vetoed` 名单与
`suspect`/`suspect_adulterated` 升级带。裁决 = 逐候选评分 → 排序 → 裁决带 + D8 门槛。

### 5.3 Agent 编排（神经层）

用带工具调用的 LLM 循环（ReAct 式，有界）在**证据补全**与**冲突澄清**两个时机介入：

**工具清单**：

| 工具 | 作用 | 备注 |
|---|---|---|
| `get_sample_evidence()` | 读证据卡池摘要 | 只读 |
| `get_criteria(herb)` | 读候选药判据 | 只读 |
| `crop_instance(instance_id)` | 取实例裁剪图+叠加图 | 引用 measure 产物 |
| `vlm_look(image, question, schema)` | 定向目视问答（纹理类型/分枝状/脐点形态/颜色光泽） | 结构化输出；≤2 问/图 |
| `presence_check(feature, image_set)` | B 通道图像级有无判定 | P1 特征 |
| `rescore(hypothesis)` | 触发规则引擎重算 | 神经层唯一能"写"的状态 |
| `submit_report(report)` | 生成报告并结束循环 | — |

**循环护栏**：

- 默认每样本 ≤ 8 次工具调用、≤ 12 张 VLM 图（成本上限，超限按置信度排序取前 N）；
- LLM 的任何数字论断必须带 instance 引用，报告生成器校验引用存在性，缺失即拒绝提交并要求重写；
- `rescore` 后仍由规则引擎出结论；LLM 若对结论有异议，只能在报告 `narrative` 与 `agent_dissent` 字段中陈述理由，供人工复核——**不改 verdict**；
- 循环超时/异常 → 降级为纯规则引擎结果 + `degraded=true`。

**介入时机**（避免无谓调用）：仅当 ① 候选间分差 < δ 且差异特征属于 B/C 通道；② 存在 `suspect` 待定实例（class 6）需 VLM 复判归类；③ 某候选的 required 特征"检测器未见但 VLM 可能看见"（P1 型判据）时才进入循环。典型大黄样本（P0 证据充分、无冲突）**零 VLM 调用**直出报告。

### 5.4 ④ 组级归并与裁决类型

| verdict | 条件 | 报告表现 |
|---|---|---|
| `confirmed` | C ≥ 0.75 且 top1−top2 ≥ δ(0.10) | "鉴别为 X" |
| `probable` | 0.50 ≤ C < 0.75 或分差不足 | "倾向为 X（备选 Y）"，标人工复核 |
| `group_level` | top1/top2 同属显微不可分组 | "可鉴别至组（赤芍/牡丹皮），组内显微不可分"——**按设计这是正确答案而非失败** |
| `insufficient_evidence` | 候选池空；或 top1 无有效支持分（C < 0.50）；**或 D8 门槛未过**（独立证据特征 <2 类且不满足 D8b 条件放行） | 列出已检出证据与缺失判据，转人工；D8 降级时原 top1 保留为首个备选并记录降级原因 |

> **2026-09-22**：原 `suspect` / `suspect_adulterated` 两带（一票否决升级）已随排他层拆除而删除，
> 裁决带只剩上表四档。

组级归并细节：半夏组内 半夏(20~144) 与 法半夏(32~144) 的下界差异小于测量误差，v1.0 明确按同组处理、在报告中提示"若为饮片可结合炮制工艺信息区分"；白术/苍术同。枳壳/陈皮、赤芍/牡丹皮完全同判据，永久组级。

---

## 六、输出：结构化鉴别报告

### 6.1 verdict.json schema（要点）

```jsonc
{
  "schema": "agent-judgement/v1",
  "sample_id": "送检样品A",                // 生产环境为送检批号；评测中为文件夹（仅作归并键）
  "generated_at": "2026-09-18T17:55:42",
  "source_images": ["<视野原图绝对路径>", "..."],
  "evidence_files": ["<叠加图绝对路径>", "..."],   // 只引用不额外复制；报告另复制进 证据图/
  "verdict": "confirmed",
  "verdict_zh": "鉴别成立",
  "conclusion": {"id": "大黄", "kind": "herb", "display": "大黄", "confidence": 0.7879},
  "support_feature_count": 3,             // D8：独立证据特征类数（<2 类且不满足 D8b 条件放行时为 insufficient_evidence）
  "margin": 0.1879,                       // top1 − top2
  "alternatives": [{"id": "G1", "display": "赤芍 / 牡丹皮", "confidence": 0.6}],
  "ranking_top5": [{"id": "...", "display": "...", "confidence": 0.0}],
  "criteria_check": [                     // 证据链：逐条判据核验
    {"feature_id": "cluster_crystal", "role": "required", "tier": "P0",
     "present": true, "state": "IN_RANGE", "quant_note": "median 57.6μm ∈ [20, 160]μm",
     "stats": {"median": 57.6, "p25": 57.6, "p75": 57.6, "min": 57.6, "max": 57.6,
               "n_measured": 1, "n_uncal": 0},
     "morph_text": "草酸钙簇晶，多角形星状集合体，棱角短钝", "morph_plausibility": 1.0,
     "reliability": 0.9, "n_instances": 1, "score_contribution": 0.9, "penalty": 0.0,
     "instance_refs": ["img1#1"]}
  ],
  "notes": ["D8 证据数量门槛：独立证据特征仅 1 类…"],
  "needs_review": false,
  "agent": {"mode": "rule_only", "vlm_calls": 0, "degraded": false},
  "shadow_bounded_adjudicator": null      // ADR-001 方案 D 影子仲裁建议，仅 --shadow-adjudicat 时非空
}
```

> 报告成品为两份自有格式：`report.md`（人读，见 §6.2）与上表 `verdict.json`；
> `kind` ∈ `herb | group`，`verdict` 枚举见 §5.4；`ranking_top5` 供错误分析与复核取证。

### 6.2 人读报告（report.md）固定骨架

1. 结论卡（verdict + 置信度 + 告警）
2. 判据核验表（判据原文 → 检出情况 → 尺寸统计 vs 药典范围 → ✓/△/✗）
3. 证据图链（缩略图 + instance 链接，复用 measure 叠加图）＋ 3·b 源视野图
4. 局限声明（未标定图、未覆盖特征、组级不可分说明）
5. （如启用 Agent）VLM 补充判读及 dissent
6. （如启用影子仲裁）仅参考、不改结论

> **2026-09-22**：原骨架第 3 节"排他性检查结果"已随排他层拆除而删除（其后各节序号前移）。

---

## 七、端到端走查：以大黄为例

> 大黄只是 85 条判据记录中普通的一条；此走查用于验证机制，路径对所有 ① 类药材同构。

1. **证据池**：measurements 中 `cluster_crystal` ×12（median feret_max 86.4μm，IQR 61~118）、`starch_grain` ×27（median 18μm，dt_peak_count 多为 3~6）、`vessel` ×4（mean_width 31μm）；`stone_cell` ×0；2 个实例 uncalibrated。
2. **候选生成**：簇晶+淀粉粒+导管 present → 大黄、虎杖、白芍、牡丹皮/赤芍组、马齿苋、辣蓼入池（皆含簇晶判据）。
3. **量化核验**：大黄 = 簇晶 IN_RANGE(1.0) + 淀粉粒 IN_RANGE(1.0，复粒 dt_peak_count 2~8 相符) + 导管 NO_REF(0.6)；白芍簇晶判据 11~35μm vs 实测 median 86 → OUT_OF_RANGE（自然落榜，无需否决规则）；虎杖 30~100μm 基本相符但其判据仅此一条、缺淀粉粒支撑。
4. **评分**：大黄 C≈0.83（只计 required 簇晶，佐证进 support_count）；虎杖 ≈0.5 量级；白芍被 OUT_OF_RANGE 压到低位。top1−top2 = 0.42 > δ。
5. **D8 门槛**：大黄的独立证据特征 = {cluster_crystal, starch_grain, vessel} = **3 类 ≥ 2** → 门槛通过（`support_feature_count=3`）。对照：仅 1 类簇晶的样本（如虎杖）会被拦下 → `insufficient_evidence`。
6. **裁决**：`confirmed · 大黄`（分差 > δ）；零 VLM 调用。
7. **报告**：判据核验表逐条引用 instance；声明"脐点星状未自动核验（灰度类判据），建议人工复核项"。

> 实测对照（2026-09-17，分割交付 demo `manual_masks_measure` 三视野归并）：`--sample 送检样品A`
> → 3 类证据（簇晶 IN_RANGE 57.6μm、导管检出、淀粉粒 IN_RANGE 17.5μm）→ **`confirmed · 大黄` 0.79**；
> 若按图拆 3 样本，则各样本仅 1 类特征 → 均按 D8 降级 `insufficient_evidence`。
> 这条对照正是 D8 的设计动机：**样本聚合（多视野送检）是多类证据落到同一候选的前提**。

对照 `_attach_大黄示例.md` 的预期：簇晶尺寸核验、淀粉粒复粒、证据图链全部落地；偏光判据按约定留在人工复核知识中；**该示例文档另给出的"排他"一条已于 2026-09-22 判定为非药典依据并删除**。

---

## 八、评测与验收

### 8.1 评测集构建

- 单位：**样本（文件夹）**，从 manifest 组感知 test 集（及"数据量太少合集"中独立成组的文件夹）取样，同 group_key 不跨集；
- ground truth = 文件夹药材名（**仅评测阶段使用**，推理链路屏蔽）；
- 4 个不可分组按**组**计分（组内任意成员都算对），分层指标另列；
- 玄明粉单独列"不适用"桶。

### 8.2 指标与验收线（提案，待项目组确认）

| 指标 | 定义 | 提案验收线 |
|---|---|---|
| Top-1 准确率（①类量化判据药） | 分层：有 P0 量化判据的药材 | ≥ 0.70（对照弱基线 RF 0.589、随机 0.333） |
| Top-3 准确率 | 全体可判药 | ≥ 0.85 |
| 组级正确归组率 | 4 组 9 味 | ≥ 0.90（输出 group_level 即得分） |
| 诚实弃权率 | 证据不足时正确输出 insufficient（而非硬猜） | 弃权样本的人工复核命中率 ≥ 0.6 |
| 置信度校准 | 分桶 C vs 实测准确率 | ECE ≤ 0.15 |
| 弱通道可靠性 | ③⑤类药材的 confirmed 判定准确率 | 单列报告，v1.0 不设硬线（预期显著低于①类） |

### 8.3 掺伪负例构造（随排他层拆除而废止）

> **2026-09-22**：原"证据级仿真掺伪负例（A 药实例 + B 药排他特征实例 → 验证 veto 触发）"随排他层拆除
> 一并废止；相应的"排他召回/误杀率"验收线已从 §8.2 移除。若日后获得**药典依据**的伪品区分判据，
> 需先重新定义裁决语义（是否恢复否决层）再重建该负例集。

### 8.4 消融与基线

- 规则引擎 only vs +Agent(VLM) 补证据：量化 B/C 通道的实际增益（预期主要在③⑤类）；
- 端到端 CNN 对照：同评测集训练整图分类器，验证项目"工具化+规则推理"路线主张（可解释性、小样本）；
- 错误分析产出：混淆矩阵按"特征集相似度"聚类（簇晶类药互混、纤维径重叠药互混），回喂知识库/检测器迭代。

---

## 九、工程结构与实施计划

### 9.1 目录（本环节新建，自包含）

```
Agent判药/
├── README.md                      # 本目录导读与协作约定
├── Agent判药环节设计方案.md        # 本文档
├── data/
│   ├── herb_criteria.json         # 85味判据知识库（构建+人工审核后冻结 v1.0）
│   └── criteria_changelog.md      # 知识库变更记录
├── scripts/                       # （已实现，与 README 目录导读一一对应）
│   ├── build_criteria.py          # ③ 85味原文 → LLM 抽取 → 对账 → 草稿（一次性构建，未入库归档；重建按 §3.2 流程）
│   ├── audit_criteria.py          # 词表/范围/组归属对账报告
│   ├── evidence_pool.py           # measurements → 证据卡
│   ├── rule_engine.py             # 候选/否决/评分/组归并（纯函数，可单测）
│   ├── agent_loop.py              # 工具编排与护栏
│   ├── report.py                  # verdict.json + report.md
│   ├── judge.py                   # 批量入口（CLI，风格对齐 measure.py）
│   └── eval_agent.py              # 分层评测 + 混淆矩阵
├── tests/                         # 规则引擎单测（合成证据卡用例，含大黄走查回归）
└── reports/                       # 运行输出
```

依赖纪律：只 import `几何测量/geomlib` 的读取类工具（如需要）与标准库；对 `标注体系/`、`几何测量/` 无任何写操作。

### 9.2 阶段计划（嵌入项目第 6~7 周，知识库可提前并行）

| 阶段 | 内容 | 依赖 |
|---|---|---|
| P-a（可与第 2~5 周并行） | 判据知识库构建 + 人工审核（85 条） | 仅 85 味文档 |
| P-b（第 6 周末） | 证据卡池 + 规则引擎 + 大黄/白芍/虎杖等 5 味走查 | measurements 样例 |
| P-c（第 7 周） | Agent 编排（VLM 工具）+ 报告生成 | 上线 VLM 配置 |
| P-d（第 7 周末） | 评测 harness + 分层指标报告 + 消融 | 全流程联调 |

### 9.3 成本与运行

- VLM 调用仅发生在冲突/弱证据样本（预期 <30% 样本），单样本 ≤12 图上限；批处理夜间跑，断点续跑（沿用 vlm_prelabel.py 的模式）；
- 全链路沿用项目既有工程修正：中文路径 imread 封装、组感知划分、predictions/annotations 两层思想（本环节的 Agent 判读记录独立存放，不混入人工证据）。

---

## 十、风险与已知坑

| # | 风险 | 对策 |
|---|---|---|
| 1 | 检测类别（6 类 P0）远小于判据词表（30 类特征），③类药材证据天然残缺 | B/C 通道 + 分层验收 + 弃权机制；不硬凑 |
| 2 | 过分割（导管拆段）虚增实例数、扭曲密度统计 | 沿用"整段一框"上游硬条款；证据层对 multi_component/同图碎片做合并启发式；密度软证据低权重 |
| 3 | 尺寸重叠药对（黄芩/桑白皮/黄芪纤维径区间交叠）单特征不可分 | 评分设计为**判据组合**区分而非单条命中；分差 <δ 输出双候选 |
| 4 | VLM 幻觉与坐标漂移 | 结构化输出 + 引用校验 + 0~1000 坐标映射纪律 + 新模型先抽 5 张目检 |
| 5 | 未标定图像使量化核验失效 | 非 ok 标定一律只作存在性证据（量化走 UNCAL 档，§4.3）+ caveats 显式声明 |
| 6 | 文件夹名泄漏造成虚高 | 推理链路代码级禁读 manifest.herb；评测 harness 单独注入 |
| 7 | 判据知识库抽取错误（LLM 误读药典原文） | 原文溯源字段 + 人工审核 + 对账脚本三重把关；冻结版本制 |
| 8 | ~~一票否决被误触发~~（**2026-09-22 已消除风险源**） | 原否决层（EX-DH-1/2）连同其"达标即升级 suspect"路径一并删除：规则的唯一出处是示例文档、非药典依据，且用与多味药材共有的特征否决真品风险大于收益。等价风险改为由**判据量化核验**承担（如簇晶尺寸出界 → OUT_OF_RANGE 直接压低该候选分值，不需额外规则） |
| 9 | 与上游目录的并行编辑冲突 | 零写入 + 派生副本 + 对账脚本（§3.5） |

---

## 十一、开放问题（需确认后冻结）

| # | 问题 | 建议 |
|---|---|---|
| Q1 | 检出置信度 conf 如何进入本环节（上游透传 vs 本环节 join）？ | 建议 A：分割模块透传，measure 增列（§2.3） |
| Q2 | 验收线数值（§8.2 提案值）需项目组确认 | 按①类分层讨论 |
| Q3 | "疑似待定 class 6"实例是否进测量链、是否由 VLM 归类 | 建议进入并由 Agent 复判。**现状**：上游已约定按 `feature_id=suspect` 裁剪交付；本环节 v1.0 在 `pool_evidence` 过滤（不评分、不复判，README D7），复判待本项定案后开工 |
| Q4 | 玄明粉等无判据品种的产品策略（跳过 vs 整图 VLM 兜底） | v1.0 跳过并声明 |
| Q5 | 组级结论在业务上是否可接受（赤芍/牡丹皮无法显微区分是药典事实） | 建议接受并输出组级 + 提示理化/性状鉴别补充 |
| Q6 | VLM 供应商与预算（glm-4.5v 复用标注链路配置） | 复用 vlm_config.json |
