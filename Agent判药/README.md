# Agent判药 · 目录导读

> 独立版保留原 B 判据、报告、知识库对账和 `selftest` 所调用的三套测试。下文的 `eval_agent.py` 和单图实验在 `D:\TCM-Microscopic-Image-Recognition-main\Agent判药` 使用。融合项目的离线评估入口见 [cli.offline](../cli/offline.py)。

本目录是中药显微鉴别流水线**最后一环（Agent 判药）**的实现目录。

- **设计方案**：`Agent判药环节设计方案.md`（v1.0，本实现的依据）
- **当前状态**：✅ v1.0 已实现并通过验证（知识库对账 PASS、规则引擎单测 PASS、真实测量数据端到端 demo 与评测链路打通；2026-09-15 与几何测量做过一轮接口对齐修复，见下表 D5~D7 与协作约定 5；同日新增 **D8 证据数量门槛**，裁决带语义变更，演进决策见 `ADR-001_裁决层演进.md`）。2026-09-16 追加：用例数据驱动化（`tests/cases/*.json`），规则引擎 57 项 + 影子仲裁 21 项单测 PASS；评分层新增候选级"宽区间主导"置信度 cap（修马齿苋假阳性），详见 ADR-001 §7。

## 目录结构

```
Agent判药/
├── Agent判药环节设计方案.md     # 主设计文档
├── ADR-001_裁决层演进.md         # 决策记录：D8 证据门槛已采纳；LLM 裁决/学习式打分评估
├── README.md                    # 本文件
├── data/
│   ├── herb_criteria.json       # 85味判据知识库 v1.0（85味+4组；量化范围与上游 profiles 对账，见 changelog v1.5）
│   └── criteria_changelog.md    # 知识库变更记录
├── scripts/
│   ├── audit_criteria.py        # 知识库对账（A1~A7：含 profiles.REFERENCE_RANGES 量纲+范围对账、量纲可供给性）
│   ├── evidence_pool.py         # measurements → 样本归并 → 证据卡池
│   ├── rule_engine.py           # 符号规则引擎：候选/评分/组归并/裁决带（纯标准库）
│   ├── agent_loop.py            # Agent(VLM) 证据补全环（可选，失败自动降级 rule_only）
│   ├── report.py                # verdict.json + report.md 生成
│   ├── judge.py                 # ★ 批量判药 CLI 入口
│   └── eval_agent.py            # 评测 harness（top1/top3/组级/弃权率/混淆对）
├── tests/
│   ├── test_rule_engine.py      # 规则引擎单测（59 项，读取 cases/rule_engine.json，直接运行）
│   ├── test_adjudicator.py      # 影子仲裁单测（21 项，读取 cases/adjudicator.json，直接运行）
│   └── cases/                   # 用例数据文件（rule_engine.json / adjudicator.json）
└── reports/
    ├── demo/                    # 真实数据 demo 输出（当前契约下可复跑复现）
    ├── demo_vlm/                # VLM 模式 demo 输出（历史快照：旧枚举 instance_id 语义，勿作 join 示例）
    └── eval_summary.json        # 评测摘要
```

## 快速使用

```powershell
# 1. 知识库对账（改动 KB 后必跑）
python scripts/audit_criteria.py

# 2. 规则引擎单测
python tests/test_rule_engine.py

# 3. 批量判药（纯规则引擎；--overlays 复用几何测量叠加图 → verdict.evidence_files）
#    几何输出经 run_pipeline 落 流水线数据\runs\<run>\measurements 后判药：
python scripts/judge.py --measurements ..\流水线数据\runs\<run>\measurements\<run>_measurements.jsonl `
    --manifest ..\标注体系\manifest.csv --out reports\run1 `
    --overlays ..\流水线数据\runs\<run>\measurements\overlay

# 3b. 同一份送检粉末的多视野 crop 归并为 1 个样本再判决（--sample）
python scripts/judge.py --measurements ..\流水线数据\runs\<run>\measurements\<run>_measurements.jsonl `
    --sample 送检样品A --out reports\run1 `
    --overlays ..\流水线数据\runs\<run>\measurements\overlay

# 4. 启用 Agent（VLM）证据补全（自动读取上游 vlm_config，失败降级）
python scripts/judge.py ... --enable-vlm --images ..\显微镜像

# 5. 评测（manifest 的 herb 列仅在评测 harness 中作为 ground truth）
python scripts/eval_agent.py --measurements ... --manifest ... --out reports\eval_summary.json
```

> **样本聚合语义**：缺省 `group_samples` 按 manifest 的 `folder_name`（或回退图 `image_stem`）把
> 测量记录归并为样本；`--sample <名字>` 则把所有记录强制归为这一个样本，用于**同一份送检粉末的
> 多张显微子图**（分割交付的多个 crop；逐图 still 各自几何测量，聚合只发生在裁决层）。

## 协作约定（重要）

1. **零写入上游**：`标注体系/` 与 `几何测量/` 对本目录**只读**；`profiles.py::REFERENCE_RANGES` 仅作知识库种子，派生副本在 `data/herb_criteria.json`。两侧差异由 `audit_criteria` A6 逐条报出（当前 31 条完全一致、4 条组并集差异属预期、1 条待裁决：茯苓 hypha `3~8|16` vs `4~6`），**在本目录内不改上游**。
2. **防泄漏**：`evidence_pool.load_measurements` 强制丢弃测量记录中的 herb 字段；manifest 的 herb 列只在 `eval_agent.py` 中作 ground truth。
3. **裁决权**：最终裁决只由 `rule_engine.decide()` 做出；VLM 只注入存在性证据后触发整体重评分，不可改裁决逻辑。
4. **确定性**：`rule_engine.py` 纯标准库、无随机性、可单测回归。
5. **测量表装载约定**（对齐 `几何测量/measure.py` 输出契约）：`instance_id` = 交付 mask 文件名的 `inst<k>`（稳定身份，可回溯 mask/叠加图，也是 Q1 conf join 的键）；目录扫描时同一 prefix 的 csv/jsonl 只取其一（优先 jsonl，避免双份记录把计数翻倍）；`image_stem` 自动剥图像扩展名、`flags` 兼容数组/分号串/Python repr（历史一次性产物如 `out/measure_草酸钙簇晶_20250378.csv` 也走同一入口规整）。

## 实现说明与设计偏差记录

| # | 事项 | 说明 |
|---|---|---|
| D1 | **reliability 不再含通道权重 w** | 设计方案 §4.3 的 `rel=w·conf·quality` 与归一化 `C=S/Σw` 叠加，导致 P1/C 通道双重降权、弱通道-only 药材永远无法进入 probable 带（违背"弱通道上限封顶为倾向"）。实现改为 `rel=conf·quality`，w 仅在判据贡献与罚项中体现一次。已回写为设计修正。 |
| D2 | 未标定实例计入存在性证据 | `present` 包含"检出但未标定"的实例（状态 UNCAL，q=0.5），而非按未检出处理，避免误触发 required 缺失罚。 |
| D3 | 测量记录暂无 conf 字段 | 上游 measurements 未透传检出置信度（设计方案开放问题 Q1），证据卡默认 conf=0.85 并标记 `conf_assumed`；上游透传后自动生效（join 键 `(image_stem, instance_id)` 已随 mask_io 契约修正而稳定）。 |
| D4 | VLM v1.0 仅接受 jpg/png 类图像 | tif 原图不做转码（避免引入 PIL 依赖），索引与调用自动跳过并在报告注明。 |
| D5 | 证据图链：报告内嵌（自包含） | 叠加图经 `judge.py --overlays <流水线数据/runs/<id>/measurements/overlay>` 关联 `<stem>__inst<id>_测量.jpg`，报告生成时**复制进该样本 `报告目录/证据图/`，内嵌于判据核验表**并从 `verdict.json.evidence_files` 记录原路径；零写入上游（不写 `标注体系/`、`几何测量/`）。 |
| D6 | 判药不读标定表 / 标定内嵌 measurements | 标定来源/状态（calibration_source/status）逐行内嵌于 measurements；几何自 2026-09-20 起只认分割交付 `*_measure_manifest.csv`，**拿不到 px_per_um 直接报错退出**，故线上来源恒为 `delivery_manifest`、状态恒为 `ok`（`uncalibrated/failed` 仅存在于历史测量产物）。`scale_table.csv` 已全项目停用（产出脚本已删），判药层始终不读取任何标定表。 |
| D7 | suspect（class 6）实例 v1.0 只过滤不复判 | `pool_evidence` 丢弃 feature=`suspect` 记录：不进评分、也不做 Q3 设想的 VLM 复判归类（该功能仍待 Q3 确认后实现）。 |
| D8 | **最少独立证据门槛 `MIN_SUPPORT_FEATURES=2`（含 D8b 单类条件放行）** | 阳性和解（confirmed/probable/group_level）需 ≥2 类**独立证据特征**（不同 feature_id、present、非 OUT、贡献分>0；同一特征的多量纲判据如针晶长+宽只计 1 类）。单类特征命中无法区分"是该药"与"该药恰有常见特征"（实证：20250378 单类簇晶 → probable·虎杖 0.57，而辣蓼/大黄/G1 仅一步之遥）。**D8b（2026-09-22）**：药典只给 1 个显微判据的品种（59/85 味，非组 47 味）在 D8=2 下永远弃权，故例外为「≥2 类 **或**（唯一证据类 且 decisive 且 conf ≥0.75）」——白芍样本（单类簇晶 conf 1.00、分差 0.40）→ confirmed·白芍，虎杖样本（大黄/虎杖 均 0.60、分差 0.00）仍弃权；放行样本仍标 `needs_review`。不达门槛一律降级 `insufficient_evidence` + `needs_review`，原 top1 保留为首个备选，`verdict.json` 记录 `support_feature_count`。动机与演进见 `ADR-001_裁决层演进.md` §3/§8。**suspect/suspect_adulterated（排他升级）与整个排他层已于 2026-09-22 拆除**——库内两条否决规则（EX-DH-1 石细胞、EX-DH-2 簇晶<45μm）的唯一出处是**示例文档**《一中药显微鉴别标准示例大黄粉末显微特征.docx》，非药典依据；以与多味药材共有的特征否决真品，误杀风险大于收益（实证：白芍样本簇晶 20.9μm ∈ 白芍 11~35 却曾被升成 `suspect(大黄)`）。故 `exclusions`/`scope`/`suspect_subject`/`vetoed` 一并删除，`verdict` 只剩 confirmed/probable/group_level/insufficient_evidence 四档，audit A5 改为"`exclusions` 残留即报错"。 |

## 真实数据 demo 结果（D8 门槛后重跑，2026-09-15）

> ⚠ 本节与下一节记录的是**当时口径**下的历史实测：其中的 `EX-DH-2 一票否决`、置信度数值（旧分母
> 含 supporting）均已被后续变更取代（排他层 2026-09-22 拆除、置信度分母只计 required、A6 差异由
> 34 条收敛为 33 条）。保留原文供演进追溯，勿作为当前行为依据；当前行为以单测 59 项 + audit PASS 为准。

`pseudo_measurements.csv`（55 个簇晶实例，伪标签掩码；标定全命中，主判中位 11.1μm）→
大黄被 **EX-DH-2**（簇晶长径中位 <45μm 存疑）一票否决；独立证据仅簇晶 1 类 →
**D8 证据数量门槛**生效：`insufficient_evidence`（原 top1 白芍 0.50 保留为首个备选、
needs_review），eval 混淆对由 D8 前的「大黄 → 白芍」变为「大黄 → (弃权)」
（对照产物：`reports/eval_summary_preD8.json` 与 `reports/single_20250378/verdict_preD8.json`）。
**解读**：D8 前该样本判 `probable·白芍`（次选辣蓼 0.4615，分差 0.039）——单类证据本就无法
区分多候选，"虚假倾向"不如"诚实弃权"；这仍是上游伪标签碎片化的真实反馈，
真实分割 mask 上线、多特征测量到位后重跑复评。
（首跑记录：median 10.9μm → `insufficient_evidence`；量化结论随 scale_table 版本漂移，属预期。
`reports/demo_vlm/` 保留的是重构前旧 instance_id 枚举语义下的历史快照。）

## 真实数据 demo 结果（分割交付 3 视野归并，2026-09-17）

分割交付 demo `流水线数据/input/manual_masks_measure`（7 个每实例 mask：img1 簇晶、
img2 导管、img3 淀粉粒×5；`px_per_um` 全部取自交付清单 `delivery_manifest/ok`），经
`run_pipeline.py --masks manual_masks_measure --sample 送检样品A` 归并为 1 个样本判决：

- **单视图各自判**（缺省按图像拆 3 样本）：每样本只见 1 类特征 → **均 D8 证据不足**
  `insufficient_evidence`；
- **归并视图判**（`--sample 送检样品A`）：3 类独立证据特征（簇晶 IN_RANGE 57.6μm、
  导管检出、淀粉粒 IN_RANGE 17.5μm）→ **`confirmed · 大黄` conf 0.79**（备选赤芍/牡丹皮、
  虎杖、马齿苋各 0.60），符合大黄"簇晶+淀粉粒+导管"判据形态。

这一对照正好验证 D8 门槛的设计动机：**样本聚合（多视野送检）是多类特征证据落到一个候选的
前提**。产物：`流水线数据/runs/demo_manual_measure`（拆分）与
`流水线数据/runs/demo_manual_measure_merged`（归并）。

## 设计待确认项

见设计文档 §十一（Q1 置信度透传、Q2 验收线数值、Q3 疑似待定处理、Q4 玄明粉策略、Q5 组级结论业务接受度、Q6 VLM 预算）。当前实现按建议方案处理（conf 缺省 0.85、玄明粉跳过、组级输出、VLM 默认关闭按需启用）。
