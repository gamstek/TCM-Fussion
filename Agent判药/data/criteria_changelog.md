# herb_criteria.json 变更记录

> 说明（2026-09-28）：本文为知识库变更记录，条目日期标识**变更发生时间**；其中的测试计数
> 已统一为**当前回归基线**（几何 `selftest_synthetic` 33/33、规则引擎 59 项、仲裁器 21 项），
> 增量描述（如"55 → 42 项"）保留当时值并标注现值。

## v1.9（2026-10-09）——分泌组织量纲对齐：油管/乳管由 feret_max 改 mean_width

**起因**：药典对油管/乳管给的是**管径**（柴胡 8~25、防风 17~60、桔梗 14~25），而判据 `dim=feret_max`。
卡片对分泌组织的主判量纲原为 `feret_max_um` → 长管实例上 `feret_max`=管段长度（切片里常常几十上百 μm），
**恒判 `OUT_OF_RANGE` 倒扣真品**；短管段才偶然落进区间——与 v1.5「量纲可供给性」类的错误同源
（`profiles.py` 分泌组织条目注释早已指出「油管/乳管长管状可改看 mean_width_um」）。

**知识库（`herb_criteria.json` → v1.9）**
- `secretory_tissue.quant.dim`：`feret_max → mean_width`，共 **6 条**——
  柴胡(8~25)/防风(17~60)/桔梗(14~25) 三条药典原文即「管径」；
  厚朴(50~85)/香附(35~72)/鱼腥草(28~104) 三条为**类圆形**油细胞/分泌细胞，其「直径」用 mean_width 量取
  **实测仅差约 3.5%**（合成圆盘 feret_max 120.9 vs mean_width 116.7 px），区间原样保留、不改数值。
  **为什么必须一并改这 3 条**：`audit_criteria.py` A7 要求判据 dim 恒等于该特征卡片
  `primary_dims` 首位量纲；卡片只有一张，改首位即对全部 6 条同构生效，漏改任一条即 A7 报错（该条恒判 UNCAL）。
  数值全部并入 `morph`（标注「管径/直径」）便于人工核对。
- `meta.version` → `1.9`。

**几何卡片刻面（`几何测量/geomlib/profiles.py`，本轮同步上游）**
- `PROFILES["secretory_tissue"].primary_dims`：`["feret_max_um","eq_diameter_um"] →
  ["mean_width_um","feret_max_um","eq_diameter_um"]`（feret_max 退为后备，椭圆油细胞可回看长径）；
- `REFERENCE_RANGES` 对应 5 条（柴胡/防风/桔梗/鱼腥草/香附）`dim` 由 `feret_max → mean_width`，**区间不变**
  （厚朴无 profiles 条目，仅 KB 侧）。此为 A6「按 (herb,feature) 比对量纲+范围」所需；
  跨模块同步（`Agent判药/` 与 `几何测量/`）已在同一提交内完成，便于追溯。

**验证**：`audit_criteria` **PASS（0 错误 0 提示）**——A6 与 profiles 对账 **32 条完全一致**、
量纲不一致 **0**、范围不一致 **0**；A7 量纲可供给性不合规 **0**。
规则引擎 57 → **59 项**、仲裁器 21 项、场景 125 项；几何 `selftest_synthetic` 31 → **33/33**
（新增两条为同批「量化可用性闸门」护栏，非本量纲改动；无用例锁定分泌组织的 `primary_dimension`，
故本改动对既有断言语义无影响）。

## v1.8（2026-09-22）——依《合并版》重排 role：正文一律 required、额外补充一律 supporting

**规则变更**（用户决议：撤销 v1.3 的"弱判别力降 supporting"例外）。新增《85味中药显微特征_合并版.md》
（原始版逐字为基底，修正版独有内容收进"（额外补充：…）"括号），知识库 role 改按文档来源机械划分：
**原始正文特征 → required；合并版括号内补充 → supporting**。

**知识库（herb_criteria.json → v1.8）**
- 恢复 required（9 条，原 v1.3 herb 级弱判别力例外全部撤销）：大黄/vessel、大黄/starch_grain、
  大黄/sublimate_crystal、大腹皮/silica_body、淡竹叶/stoma、党参/secretory_tissue、
  麦冬/stone_cell、木香/fiber、猪苓/hypha；
- 拆出混入 required morph 的补充内容（引擎按 feature_id 取证据卡，同特征重复条目会双重计分，
  故**不另立同特征判据**，改记 notes）：艾叶（单列性非腺毛）、薄荷（小腺毛）、淡竹叶（副卫细胞
  近圆三角形）、牡蛎（棱柱层断面）、水牛角（纵/横断面观）、葶苈子（纤维素柱）、栀子（内果皮石细胞）；
- 异特征补充内容正常立 supporting 判据：大青叶/epidermis_cell（连珠状增厚）、
  苦杏仁/epidermis_cell（种皮外表皮细胞）；大青叶"橙皮苷样结晶"无词表对应，仅记 notes；
- `meta`：version 1.8；generator 改指《合并版》；role_rule_note 重写；role_exceptions 删除
  scope=herb 整块（金银花×3/栀子×2/厚朴×2/黄连×1 本就是补充来源，按来源规则维持 supporting，
  无需例外条目）；group/quant/non_official 三块保留不变。

**影响**（用户决议：接受严格语义，逐用例更新测试锁定新行为）：
- 大黄"簇晶+淀粉"双证据样本：conf 1.00→0.53（导管/升华物未检出计入 required 分母）→ 不再
  自动 confirmed，诚实弃权；导管/升华物证据补齐后仍可达 confirmed（大黄走查用例 0.7852）；
- 猪苓"菌丝+方晶"双证据样本：conf 1.00→0.87，G3（陈皮/枳壳组单判据满分 1.00）反超登顶，
  输出 group_level + 人工复核，猪苓降为首备选；
- 混淆×1：马齿苋（宽区间 cap 0.6）登顶为 probable 结论，cap 防伪仍有效（未到 confirmed）；
- `tests/cases/rule_engine.json` 同步更新 3 个用例（方晶单特征、双特征放行、混淆×4），
  复跑 47/47 PASS；audit PASS（0 错误 0 警告）。

**影响**：正文特征缺失罚变严（required 集扩大），G1 组"赤芍正文并含淀粉粒"仍按组交集设计维持
supporting（组 reason 已注明）。

## v1.7（2026-09-22）——非药典来源的特征存在性一律 supporting（青黛补正 + audit A8）

**规则澄清**（用户决议：非药典来源的特征都设为 supporting）。文档标记 `**非药典来源：…口径**`
按**标注覆盖的范围**分两类，判据库分别处理——这一区分是必须的，否则会归零药典药材的置信度：

| 类型 | 例 | 处理 |
|---|---|---|
| ① 覆盖**特征本身**（存在性来自非官方） | 青黛靛蓝（教材）、辣蓼簇晶/棕色块物（广东地方标准）、淫羊藿非腺毛（教材） | **一律 `supporting`**，不得 `required` |
| ② 只覆盖**若干 μm 区间**（存在性是药典正文） | 牡丹皮/赤芍簇晶区间（教材）、茯苓菌丝区间（成方）、连翘纤维（教材）、麦冬/白头翁直径 | 保留 `required`，**区间置 `null`** |

若把②也降为 supporting：G1（赤芍/牡丹皮）与连翘的唯一 `required` 会变空 → 分母为 0 → 置信度归零，
药典药材再也判不出来（这些特征的存在性本是药典正文，只是尺寸数字不成文）。

**知识库**
- `青黛/indigo_crystal`：`required → supporting`。v1.3 当时只加了候选级 `non_official_basis`
  （conf 封顶 0.49），**未按同一条规则的"存在性不得 required"改 role** —— 本轮补正，
  required 集随之为空 → 置信度 0，仅作备选出列；`notes` 同步说明；
- 辣蓼（cluster_crystal、brown_content）与淫羊藿（non_glandular_hair）本就 correct，无需改动；
- `meta`：`role_exceptions` 新增 `scope=non_official` 一类（4 条）逐条登记；新增 `non_official_note`
  说明 `non_official_basis` 是**候选级**旗标，若日后出现"同一候选内既有正文判据、又有非官方判据"
  的混合情形，须改为判据级 `basis` 字段。

**工具（`audit_criteria.py`）**
- 新增 **A8 非药典来源不得 required**（**错误级**）：对带 `non_official_basis` 的候选，任一 `required`
  判据即 FAIL。反例自测（临时把青黛改回 required）确认报 `[ERROR]`、退出码 1，不会静默放过；
- 复跑：audit PASS（A6 31 一致 + 已知分叉 2 + 组并集 3、A7 0、A8 ✓）、0 错误 0 警告。

**文档**
- `85味中药显微特征_修正版.md` 图例改写为上述①②两类的精确表述（原表述把两类混为一谈，
  按字面会把牡丹皮/赤芍簇晶也降级）；淫羊藿条目补注"非腺毛细胞数与长度均系教材口径，
  药典正文未载该特征"，避免与"正文有特征但无值"混淆。

**验证**：audit PASS（A8 反例 FAIL 已自测）；规则引擎 57 项、仲裁器 21 项全 PASS；
白芍/虎杖 run 复跑结果不变（青黛不参与这两个样本的候选池）。

## v1.6（2026-09-22）——D8b 单类证据条件放行；文档来源标记约定；A6 已知分叉登记

**起因**：v1.5 对账时发现的结构性冲突——**59/85 味药典只给 1 个显微判据**（非组 47 味），
在 D8「≥2 类独立证据」下这批药（红花/板蓝根/虎杖/荜茇/麻黄/桂枝…）**永远**只能输出
`insufficient_evidence`。第一个真实反例：白芍样本（簇晶 16.1~23.2μm ∈ 药典 11~35，conf 1.00）
被弃权。用户决议：条件放行（不做完整机制变更）、纯定性药材维持 `NO_REF=0.6`。

**A. 裁决层 D8b（`scripts/rule_engine.py`，见 ADR-001 §8）**
- D8 门槛由"必须 ≥2 类"改为「**≥2 类 或（唯一证据类 且 `decisive` 且 conf ≥ 0.75）**」；
- 两道前提缺一不可：① `decisive`＝该类证据已把 top1 与并列者分开（分差 ≥δ=0.10，或独立证据类数不同）；
  ② 达到 confirmed 置信带。放行样本仍 `needs_review=True` 并在 notes 记录 D8b 放行依据；
- **为什么不是直接把门槛降到 1**：实测 D8=1 时白芍→confirmed，但**虎杖样本→probable·大黄**
  （大黄/虎杖 conf 均 0.60、分差 0.00）——正是 D8 当初要拦的假阳性；加 `decisive` 前提后两者分开；
- 边界（回归锁定）：单类且分差 0 的"方晶 G3/猪苓"、"极多簇晶 大黄/虎杖/马齿苋"、"针晶宽范 半夏/射干"、
  "未标定 UNCAL" 全部继续弃权；宽区间判据被 DOM cap 压到 ≤0.60，天然到不了 0.75；
- 无量化 `required` 判据的品种（`NO_REF=0.6`）到不了 0.75 → 本放行只对**有量化的单判据品种**生效，
  纯定性药材 20 味（荜茇/槟榔/麻黄/连翘/滑石…）维持弃权（用户决议）。

**B. 文档来源标记（`85味中药显微特征_修正版.md`）**
- 新增统一标记 `**非药典来源：…口径**`，覆盖 6 处原先只写在自然语言里、机器不可读的非药典依据：
  牡丹皮簇晶 9~45（教材）、赤芍簇晶 7~40/12~43（教材）、茯苓 4~6（成方显微）、辣蓼 5~35（广东地方标准/
  中华本草）、青黛（教材）、淫羊藿非腺毛 200~1000（教材）；文首补"来源标记约定"图例；
- 补记缺失量纲词：白芍"具缘纹孔及网纹导管 **直径**20～65μm"、"纤维长梭形，**直径**15～40μm"
  （原文漏"直径"，与判据库 `mean_width` 口径对齐）；
- 数值零改动（profiles/KB 均不受影响），只让 v1.3 的 role 规则有机器可读锚点。

**C. 工具（`scripts/audit_criteria.py`）**
- 新增 `A6_KNOWN_DIVERGENCE` 豁免登记：`茯苓/hypha`（KB 单味正文 3~8|16 vs profiles 成方 4~6）、
  `陈皮/single_crystal`（KB 组并集 3~53 vs profiles 5~53）**已裁决"不同步上游"**，
  由 `[warn]` 降为 `[info]` 并在行内附裁决理由——不再每次 audit 重复告警；
- 结果：A6 31 条一致 + 已知分叉 2 + 组并集 3（预期），量纲不一致 0、范围不一致 0；A7 不合规 0；
  **audit 首次达到 0 错误 0 提示（0 warn）**。

**测试**：规则引擎 42 → **44 项**（现 57 项）——"小簇晶单特征 → D8 拦截"改判为"D8b 条件放行 → confirmed·白芍
（并断言 conf ≥0.75、`needs_review` 为真）"；仲裁器 21 项不变。

**验证**：`audit_criteria` PASS（0 warn）；规则引擎 57 项、仲裁器 21 项全 PASS；
几何 `selftest_synthetic` 31/31、`validate_ranges` 白芍 5/5 `within`；12 模块 import 链通过；
三个真实样本 run 复跑——**白芍 confirmed·白芍（原弃权）**、**虎杖 仍弃权（无假阳性）**、
大黄 confirmed·大黄（三类证据不受影响）。

**仍挂起**：① 按通道分级放行（P0 独证可 probable、P1/C 独证不行）未采纳；
② 新增 `criteria.specificity` 字段 + D8"≥1 类高特异 + ≥2 类证据"的完整机制变更（设计方案 §3 备注）
未启动；③ 纯定性药材的结论上限（`NO_REF=0.6`）维持，靠 VLM 通道补第二类证据。

## v1.5（2026-09-22）——文档↔判据库全量对账：量纲可供给性、role 组内交集、来源口径

**起因**：对 `85味中药显微特征_修正版.md` 与 `herb_criteria.json` 做逐条对账（85 味名称/组归属/
数值/role 四路核对）。名称、组归属、绝大多数数值一致；查出三类问题：① 教材/成方口径区间仍在做
硬判定；② 判据 `dim` 与检测链实际写出的量纲不符 → 药典区间形同摆设（恒判 `UNCAL`）；
③ 组判据把"仅个别成员有的补充特征"当 `required`，用缺失罚把该成员结构性降级。

**A. 区间退出硬判定（v1.3 同类残留）**
- `连翘/fiber`：`mean_width 24~32 → null`。药典连翘显微只述"内果皮纤维上下层纵横交错，纤维短梭形"，
  24~32μm 系教材口径（v1.1 已加注来源更正，但**未按 v1.3 处理**）→ 实测 20μm 会被判
  `OUT_OF_RANGE` 倒扣真品。数值移入 `morph`，判据级 `notes` 键删除（回归 `meta.criterion_keys` 白名单）。
- `麦冬/needle_crystal`：`mean_width ≤10 → null`。`needle_crystal` 的卡片量纲是 `feret_max`
  （profiles `primary_dims=["feret_max_um"]`），该条恒 `UNCAL` → 实测直径 8μm（药典范围内）时
  麦冬唯一 `required` 上限只有 conf 0.425。处理口径与 v1.3 删除射干 `mean_width ≤49` 完全一致，直径并入 `morph`。
- `白头翁/non_glandular_hair`：同上（非腺毛卡片量纲为 `feret_max`/长，13~33μm 是直径）→ `null`，直径入 `morph`。
- `G1/cluster_crystal`：维持 `null`，但把组理由写实——牡丹皮文档所载 9~45μm 系教材口径，按 v1.3 不作判据。

**B. 量纲对齐检测链（4 条导管判据）**
- `白芍/地黄/防风/G2 的 vessel`：`feret_max → mean_width`。profiles 对导管的剖面是
  "管径=沿骨架中位宽"（`primary_dims=["mean_width_um","max_inscribed_dia_um"]`），
  而卡片每条记录只写一个主判量纲 → 原 4 条管径判据（20~65 / ≤92 / 14~85 / 10~24μm）全部恒 `UNCAL`。
  改后实测 42.5μm 管径：白芍 `IN_RANGE`、G2 `OUT_OF_RANGE`（螺纹导管 10~24μm）均正常生效。

**C. 组判据 required 取"成员交集"（v1.2 提级的反噬）**
- `G2/starch_grain`、`G4/stone_cell`、`G4/inulin`：`required → supporting`。淀粉粒只出现在半夏条目、
  石细胞/菊糖只出现在白术条目（法半夏/姜半夏/苍术文档均无），设为 required 即对缺该特征的成员施加
  `0.3·w·sens` 缺失罚：实测"仅检出针晶"时 G2 conf 0.305（排名第 5，落后射干/知母 0.600）、
  G4 conf 0.220 → 双双被 D8/裁决带判 `insufficient_evidence`；而"针晶+淀粉粒"齐备时两种口径
  同为 `group_level` 0.850 —— **提级在证据齐时零增益、在证据缺时只有害**。
  与设计方案 §3 ④-3 一致：升档对 D8 门槛零作用（`_support_feature_ids` 只看 feature_id 类数）。
- `G3/single_crystal`：`5~53 → 3~53`。下限补入枳壳正文"直径3~30μm"（原只取陈皮"长5~53"）。

**D. 补录文档有而库内缺的特征（同时缓解"单类判据即弃权"）**
- 厚朴 `fiber 15~32`、`secretory_tissue 油细胞 50~85`；黄连 `stone_cell 25~64`（长至102 入 morph）；
  淡竹叶 `silica_body`（哑铃形硅质细胞/方形栓质细胞）；防风 `fiber`（叶基维管束伴纤维束）；
  乌梅 `non_glandular_hair`（毛茸脱落痕迹）；
  G1 组补 `vessel 25~78`、`cork_cell`、`fiber`，并把 `starch_grain` 量化为 `2~16`（赤芍2~13 ∪ 牡丹皮3~16）。
- **未补并登记理由**：甘草/黄柏/苦参的"方晶"与既有 `crystal_fiber` 系同一解剖证据（同源计两类会虚假
  抬高 D8 广度）；牡蛎棱柱层、葶苈子纤维素柱、栀子内果皮石细胞等仍按 v1.2"并入 morph、不新建判据"。

**E. 其他修正**
- `白芍`：删 `gelatinized_starch`（与 `starch_grain` 的 morph 是同一句"糊化淀粉粒团块甚多"，
  违反 v1.2"白芍糊化淀粉粒并入 morph、不新建判据"）。**保留 `starch_grain` 作唯一载体**：若两条都删，
  检测到的 P0 淀粉粒会触发 `wrong_present` 罚反伤真品（单测"中簇晶→白芍"即因此失败，实测确认）。
- `茯苓`：`hypha 4~6 → 3~8|16`（改按文档"单味正文"口径；4~6 系成方显微口径，且与 profiles 通用条目 3~8 冲突）；
  `fungal_mass supporting → required`（正文"不规则分枝状团块遇水合氯醛溶化"为主特征）。
- `淫羊藿`：`non_glandular_hair`（教材口径）`required → supporting`；正文"沿叶脉异细胞含草酸钙柱晶"
  升 `needle_crystal required` —— 原状是"教材特征当支柱、正文特征当佐证"。
- `大腹皮/silica_body`：`8~8 → 4~12`（"约8μm"给容差，与 profiles 对齐）。
- 补录 18 条药典丰度措辞（众多/甚多/较多/多见/可见/少见），`M_ABUNDANCE` 增 `众多=1.0`、`少见=0.9`，
  `meta.abundance_note` 与引擎口径对齐。
- `meta`：补 `criterion_keys`（含 herb 级键）、`abundance_note`、`role_rule_note`、
  `role_exceptions`（正文但置 supporting 的 17 条 + 组交集 5 条 + 区间置 null 5 条，逐条登记理由）。

**工具（`scripts/audit_criteria.py`）**
- A6 改为按 `(herb, feature)` 配对后再比 **量纲 + 范围**（旧实现把 `dim` 写错当成"没这条"而静默跳过，
  射干即此类）；组判据按成员展开，组并集差异计入 `[info]`；
- 新增 **A7 量纲可供给性**（错误级）：判据 `dim` 必须等于该特征卡片实际写出的主判量纲
  （`profiles.get_profile` 首位），否则恒判 `UNCAL` —— 这是一条可回归的错误，而非提示。

**上游未写入（遵守 README「零写入上游」）**：本轮只改 `Agent判药/`，`几何测量/` 未动，两处差异留待人工裁决：
- `陈皮/single_crystal`：KB G3 组并集 `3~53` vs profiles `5~53`（枳壳正文下限 3μm 未进 profiles）；
- `茯苓/hypha`：KB `3~8|16`（文档"单味正文"口径）vs profiles `4~6`（成方显微口径，与本文件外的通用条目 `hypha 3~8` 亦不自洽）。
- 若裁决为"同步上游"，这两行改动须单独提交（跨模块写入，与本轮"只改判据库"的范围分开追溯）。

**验证**：`audit_criteria` PASS（A6：31 条范围完全一致、4 条组并集差异（预期）、1 条范围差异（茯苓，待裁决）、
量纲不一致 0；A7 量纲可供给性不合规 0）；
规则引擎单测 57 项、仲裁器 21 项通过；几何 `selftest_synthetic` 31/31、`validate_ranges` 白芍 5/5 `within`（16.1~23.2μm ∈ 11~35）；
12 个模块 import 链通过；重跑 白芍/虎杖 run 与改前逐字一致（`insufficient_evidence` + 同一 D8 note，无回归）。

**待拍板（机制层，未改）**：D8 `≥2 类独立证据` 与"药典只给 1 个判据的药材"结构性冲突
（59/85 味仅 1 条判据，其中非组 50 味永远只能输出 `insufficient_evidence`，含红花/板蓝根/虎杖/荜茇等；
白芍样本簇晶 20.9μm 合规、conf 1.00 仍弃权）。设计方案 §3 ④-3 与备注（新增 `specificity` 字段、
D8 改为"≥1 类高特异 + ≥2 类证据"）已指出方向，但属机制变更，需 `ADR-001` 追记 + 全量回归后再动。

## v1.4（2026-09-22）——拆除排他（一票否决）层：两条大黄否决规则无药典依据

**判定依据**：库内唯二两条否决规则（`EX-DH-1` 成群石细胞 → 疑似掺伪；`EX-DH-2` 簇晶中位数
<45μm → 存疑）的唯一出处是《一中药显微鉴别标准示例大黄粉末显微特征.docx》——一份**示例文档**
（在 `标注体系/docs/标注方案与最小测试报告.md` §3.2 被引用为"黄金样例"，原文只有
"检出大量石细胞→提示伪品；簇晶<45μm→存疑"一句），**非《中国药典》依据**。以白芍（11~35μm）、
赤芍、牡丹皮等 10 味共有、并不指向大黄的特征去否决真品，误杀风险大于收益（实证：白芍样本簇晶
20.9μm 本合规，却曾被升成 `suspect(大黄)`）。故连同其逻辑整体删除，而非继续收紧前件。

**知识库**：
- 删除 85 味 + 4 组的 `exclusions` 字段与 `meta.exclusion_note`（净删 124 行，无其他字段改动）。

**引擎（`scripts/rule_engine.py`）**：
- 删除 `_check_when` / `check_exclusions` / `_has_sample_alarm`、vetoed/clean 分流、`escalate`
  升级分支、`scope` 透传；`KnowledgeBase.candidates()` 不再读取 exclusions；
- `decide()` 输出不再有 `exclusions_triggered` / `vetoed` / `suspect_subject`，`ranking` 即全量排序；
- 裁决带只剩 `confirmed` / `probable` / `group_level` / `insufficient_evidence` 四档；
- D8 门槛不再有"排他升级豁免"路径，对全部阳性和解一视同仁。

**消费者同步**：`report.py`（VERDICT_ZH 去 suspect 两档、verdict.json 去 3 个字段、删除
"排他性检查"节并把证据图链/源视野图/局限声明/Agent/影子仲裁前移为三~六节、`_full_checks`
还原为"结论候选"）、`audit_criteria.py`（A5 由"结构校验"改为**残留检查**：库内再出现
`exclusions` 即报错）、`eval_agent.py`（pred 口径回到 `conclusion.display`）、
`diagnose_single.py`（去掉 `check_exclusions` 导入与否决打印）、`run_single.py`（去掉被否决候选段与
排他节）、`adjudicator.py`（文档串）。

**测试**：删除"掺伪负例（一票否决）"与"EX-DH-2 scope=candidate"两个用例，及
`no_exclusions`/`vetoed_contains`/`vetoed_entry_exclusion`/`suspect_subject_and_review`/
`no_suspect_subject`/`conclusion_and_review` 六类断言与 `_any_vetoed` 辅助；
"混淆×2"改名并改判为"结论为白芍（大黄簇晶 OUT_OF_RANGE 自然落榜）"。
规则引擎单测 **55 → 42 项**（现 57 项）、仲裁器 21 项、audit PASS。

**上游同源规则一并清理**（同一示例文档来源，避免回流）：
- `标注体系/feature_taxonomy.json`：删除顶层 `exclusion_rules`；
- `标注体系/detection/criteria_table.json`：删除「大黄|石细胞」提示行（原"大黄中不得检出石细胞！
  检出即提示掺伪"），`comment` 同步；**注意：已推送到 LS 的任务需重跑 `detection/annot/sync_criteria.py` 才会刷新**；
- `几何测量/geomlib/profiles.py`：删除 `EXCLUSION_GEOMETRIC`；`几何测量/validate_ranges.py`
  去掉其导入与"大黄<45μm 打 flag"逻辑（尺寸出界仍由 `out_of_range` 状态覆盖）；
- 相关 README/docs（几何测量 README、分割方案与设计、标注方案与最小测试报告、detection 操作手册）同步标注。

**待办**：若日后取得**药典依据**的伪品/易混品显微区分判据，须作为新 ADR 重新论证"是否恢复否决层"
（语义、评测集、误杀率验收线都需重建），不得沿用本轮删除的规则文本。

## v1.3（2026-09-21）——role 归属规范化：非官方依据不得作 required；非官方区间退出硬判定

**规则**（据《85味中药显微特征_修正版.md》的来源分段）：药典正文明文的主特征才可 `required`；
「药典补充」段一律 `supporting`；**非官方来源**（地方标准/中华本草/教材/成方口径）分两种处理——
特征存在性来自非官方者**不得 required**，特征在正文但 μm 区间非官方者**区间置 null**（不得用于
`OUT_OF_RANGE` 判罚，否则会倒扣真品）。

- `辣蓼`：`cluster_crystal` **required → supporting**，并新增 `non_official_basis: true`
  （《药典》未收载本级，5~35μm 系广东地方标准/中华本草口径）。影响：辣蓼 required 集为空 →
  分母为 0 → 置信度 0，不再与药典药材争排名（此前在白芍样本上与白芍并列 1.0000 并列第 1）。
- `青黛`：新增 `non_official_basis: true`（《药典》鉴别以理化为主，蓝色颗粒/块片为教材口径）。
  引擎据此把该候选置信度封顶到 `NONOFF_MAX_CONF=0.49`（低于 probable 带）：可作备选列出，不得作为结论。
- `淫羊藿`：`non_glandular_hair` 区间 `200~1000 → null`（教材口径，药典正文未给非腺毛数值），
  数值移入 `morph` 作参考。与既有正确先例（赤芍/麦冬 已置 null）口径统一。
  影响：A6 与 profiles 对账由 34 条一致变为 33 条（该条本就不应参与量化判定）。
- `射干`：删除第二条 `needle_crystal`（`mean_width ≤49`）。针晶/柱晶主判量纲是 `feret_max`，
  卡片中不含 `mean_width` → 该条恒判 `UNCAL`；若卡片确有 mean_width 则 135μm 判 `OUT_OF_RANGE`
  ——两种情形都是对同一证据的二次计分与误罚。直径信息并入 `morph`。

**引擎同步**（`scripts/rule_engine.py`）：

- 置信度改为**只由必备判据决定**：`C(h) = min(1, max(0, S_required) / Σ_required w)`。
  supporting 的净贡献从分子剔除、也不进分母 —— 按 v1.2 立库意图，supporting 的作用是
  "增强证据广度、助力 D8≥2 类门槛"，不是抬置信度。一处同时消除三个方向的偏差：
  ① 未检出的 supporting 不再稀释（原"知识库记得越细、置信度越低"）；
  ② 已检出的 supporting 不再造成"多检出佐证反而降分"（原猪苓 1.0000 → 0.8667）；
  ③ required 未检出的候选无法再靠佐证顶上来（原川楝子：晶纤维未检出、簇晶作佐证仍得 0.6267 → 现归 0）。
- 排序键改为 `(confidence, support_count, score)`：置信度并列时以独立证据类数分高下。
- "窄分差"判定改为 `decisive = 分差≥0.10 或 独立证据类数不同`，避免把已明确判出的候选降级。
- 新增 `non_official_basis` 候选封顶 `NONOFF_MAX_CONF`。
- **排他升级前件收紧 + 嫌疑对象另立字段**：`escalate` 原只要求被否决候选 `conf ≥ 0.50`，
  于是「簇晶中位数<45μm」（EX-DH-2，与 10 味含簇晶药材共有）在白芍样本上把整样本升成
  `suspect(大黄)`——而排名首位是白芍 1.00、大黄 0.60，且大黄已被否决，报告自相矛盾。
  现要求被否决候选必须**本来就是最佳解释**（`conf ≥ BAND_PROBABLE` **且** `≥ 未否决候选最高分`）；
  判为升级时 `conclusion=null`（识别结论不成立），嫌疑对象写入新增字段 `suspect_subject`
  （报告 §一 显示"存疑对象"，§二 核验表回退到嫌疑对象、§三 仍记录否决明细）。掺伪用例（EX-DH-1：
  大黄 conf 1.00 高于任何未否决候选）不受影响，安全线保持。
- **否决条件新增 `scope`（样本级 vs 候选级）**：`EX-DH-2` 的谓词（簇晶中位数<45μm）与白芍
  （11~35μm）等小簇晶药材共有、**本身不指向大黄**，作为"整样本掺伪警报"语义不成立。
  按"是否指向本药"分级，令其 `scope: "candidate"`——命中只取消大黄的识别资格（退出排名、
  不可作结论），永不升级整样本；`EX-DH-1`（成群石细胞）显式 `scope: "sample"`，仍为样本级
  安全线。升级路径前件因此变为：`scope=sample` + `conf ≥ 0.50` + `conf ≥ 未否决候选最高分`。
  `meta.exclusion_note` 与 audit A5（scope 白名单）同步；报告 §三 仍照常列出候选级否决明细。

**评测口径同步**：`eval_agent.py` 的混淆对/`pred` 原用 `conclusion.display`，升级样本因
`conclusion=null` 一律记成"(弃权)"，与 top1/top3 走 `ranking` 的口径不一致；现回退到
`suspect_subject.display`（升级样本显示"被误判成谁"，`per_sample` 另记 `suspect_subject` 字段）。

**实测效果（真实 run 复跑）**：白芍样本候选排名由「辣蓼 0.769 / 白芍 0.263（第 9）」变为
「**白芍 1.0（唯一首位）** / 马齿苋 0.6 / 赤芍·牡丹皮 0.6 / 辣蓼 0 / 川楝子 0」；
白芍样本裁决由 `suspect(大黄)` 回归 `insufficient_evidence`（conclusion/suspect_subject 均 null，
提示"独立证据仅 1 类簇晶"）；虎杖样本维持 `insufficient_evidence`（大黄 0.60 与虎杖 0.60 并列，
双候选转人工）。

**未完成（待输入源）**：85 味目前仍只有大黄带排他规则。补齐"逐味排他规则"需要各药伪品/易混品的
显微区分依据，而当前知识库唯一种子《85味中药显微特征_修正版.md》未收载伪品章节（grep 无
伪品/易混/不应出现等表述），凭空补录会引入无源判据，故本轮只做机制（`scope` 分级 + 升级前件），
待取得伪品区分依据（药典「鉴别」项外的易混品对照文献）后按药逐条补录。

**验证**：audit PASS（A6 差异仍仅大腹皮 1 条）；规则引擎单测 57 项、仲裁器 21 项全部通过。

## v1.2（2026-09-16）——补录文档"药典补充"独立可测特征（supporting）

- 依据：《85味中药显微特征_修正版.md》中标注"药典补充"的内容，将**可独立检测的新特征**补录为 `supporting` 判据（增强证据广度、助力 D8≥2 类门槛）；纯形态细化并入已有判据 `morph`，不新建判据。
- 补录 supporting 新判据（非组药）：艾叶 cluster_crystal 3~7 / glandular_hair、白芍 vessel 20~65 / fiber 15~40 / gelatinized_starch、薄荷 non_glandular_hair / stoma、川楝子 stone_cell 14~54/ext150 / cluster_crystal 5~27、穿心莲 non_glandular_hair / glandular_hair~40 / stoma、大青叶 stoma、淡竹叶 non_glandular_hair、党参 inulin、地黄 secretory_tissue / vessel~92、防风 vessel 14~85 / stone_cell、诃子 cluster_crystal 5~40、栀子 fiber(内果皮~10) / cluster_crystal 19~34、使君子 fiber、金银花 glandular_hair~108 / non_glandular_hair / cluster_crystal 6~45、淫羊藿 needle_crystal(柱晶) / stoma。
- 组判据补录：G2(+starch_grain 2~20、vessel 螺纹10~24)、G3(+stoma 气孔环式16~34)、G4(+fiber~40、stone_cell 37~64、inulin)。
- 纯形态并入 morph（不新建判据）：牡蛎棱柱层、水牛角纵/横断面纹理、葶苈子种皮纤维素柱、黄芩纤维长60~250、苦杏仁种皮外表皮、白芍糊化淀粉粒、艾叶单列非腺毛、薄荷小腺毛、穿心莲钟乳体螺状、大青叶橙皮苷样结晶/连珠状壁、淡竹叶气孔副卫细胞、栀子内果皮石细胞、川楝子种皮含晶细胞。
- 判别性特征提级 required（区分易混淆近缘药）：G2 淀粉粒 2~20（半夏必含且"甚多"，区别于知母/射干仅针晶）、G4 石细胞 37~64 与 菊糖（白术/苍术菊科共性）；G4 纤维判别力弱，维持 supporting。
- 验证：audit PASS（A6 仅存此前"只锁文档值"遗留的大腹皮 silica 提示 1 条）；规则引擎单测 57 项、仲裁器单测 21 项全部通过。

## v1.1（2026-09-16）——按药典完善显微特征，量化区间对账

- 依据：《85味中药显微特征_修正版.md》逐味对照《中国药典》完善后，与herb_criteria.json量化区间对账。
- 新增 quant（14条）：厚朴 stone_cell 11~65、苦杏仁 stone_cell 25~150、栀子 stone_cell 60~112、黄芩 fiber 9~33、连翘 fiber 24~32、甘草 crystal_fiber 8~14、苦参 crystal_fiber 11~27、马齿苋 cluster_crystal 7~108、辣蓼 cluster_crystal 5~35、猪苓 hypha 2~10、麦冬 needle_crystal ~10、诃子 stone_cell 14~40、使君子 epidermis_cell 14~43、穿心莲 calcium_carbonate ~36。
- notes 更正：连翘 fiber 的24~32为教材口径（药典未给连翘纤维μm值），已加注来源更正。
- 明确不补（药典/文档未给μm值、纯定性判据）：党参/麦冬/黄连/荆芥/菟丝子/麻黄/麦芽等石细胞/纤维/表皮/气孔、金银花/菊花花粉粒直径、青黛/石膏/滑石/雄黄/水牛角等；玄明粉标 no_features。
- 组级处理：G1~G4 组判据维持，成员冗余 quant 不补（规则引擎候选只认组判据）。
- 与 profiles.REFERENCE_RANGES 对账：A6 保持 35 条一致、差异 0。

## v1.1.1（2026-09-16）——量化区间收敛为"只锁文档值"

- 依据：《85味中药显微特征_修正版.md》与知识库逐条对账后，将量化区间收敛为仅取文档明文数值（不再沿用来自几何测量 profiles 的容差扩展）。
- 修订 2 条：
  - 大腹皮 `silica_body`：4~12μm → **8~8μm**（文档"直径约 8μm"单点值）。
  - 猪苓 `hypha`：去除 quant 区间 2~10μm → **`quant: null`**（文档未给菌丝 μm 值，仅保留存在性 supporting 判据）。
- 对账结果：audit PASS；A6 与 `profiles.REFERENCE_RANGES` 变为 34 条一致、**差异 1 条（大腹皮 silica_body 8 vs profiles 4~12，仅供人工裁决）**。profiles 属`几何测量`只读目录，未改动。

## v1.0（2026-09-15）

- 初版发布：85 味全覆盖（79 味独立判据 + 4 组 9 味归组）+ 玄明粉显式 no_criteria。
- 判据来源：《85味中药显微特征_修正版.md》逐条结构化；`source` 溯源以 `meta.generator` 说明为准（人工审核版）。
- 对账结果（`scripts/audit_criteria.py`）：
  - A1~A5 结构校验 PASS（85 味齐全、词表内引用、范围合法、组归属互指、否决条件合法）；
  - A6 与 `几何测量/geomlib/profiles.py::REFERENCE_RANGES` 对账：35 条量化范围完全一致，差异 0 条。
- 特征词表：feature_taxonomy.json v0.1 超集——共用 id 语义一致；新增 P1 级 `palisade_cell / stoma / glandular_hair` 与 C 级 `indigo_crystal / brown_content / parenchyma_cell / endosperm_cell / fungal_mass / gelatinized_starch`（85 味判据所需，详见 meta.vocab_note）。
- 已知限制：判据中"灰度/纹理类"属性（脐点形态、壁木化、颜色光泽等）以 `vlm` 字段标注，待 VLM 通道启用后参与弱证据判读；v1.0 仅形态几何代理生效。
