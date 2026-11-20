# Label Studio 两阶段标注操作手册（P0 试点）

> 本页保留原工程的标注与数据整理流程。下文的 Label Studio、预标注、数据集构建和训练命令在 `D:\TCM-Microscopic-Image-Recognition-main\目标检测` 使用。独立版的检测、分类和标定由融合工作流调用，运行从 [独立部署](../独立部署.md) 开始。

> 流程：**阶段一 AI 预标注**（多模态大模型 API 批量出紧框）→ **阶段二 人工审核**（Label Studio 内采信/修正 AI 框）→ **自检**（10% 盲标算 Kappa）→ **导出**（YOLO 紧框 + μm 属性表）。
> 全部脚本在 `目标检测/`（**可执行入口在其 `cli/`——其中 `annot/` 专收 ↔Label Studio 回路（导入/同步/判据回填/盲标自检），具名库在其 `lib/`，配置在其 `config/`，回归测试在其 `tests/`**），运行统一用项目根 `.venv\Scripts\python.exe`。下面命令一律先 `cd` 到 `目标检测/`。
> **脚本已全改具名（2026-09-29）**：原先的 `02_`~`13_` 编号前缀一律去掉（下表已按新名列出）；`01/04/06/08/09` 更早已归档，故**本目录不再有编号脚本**。各入口定位与工序顺序见根 README 的「入口总表」。
> **产物位置（2026-09-20 起、2026-10-09 上提）**：本手册中小写的 `prelabels/`、`prelabel_review/`、`sample_detect/`、`yolo_export/`、`ls_tasks/`、`ls_view/`、`pilot_manifest.csv`、`selfcheck_state.json` 一律指 **项目根 `results/detection/`** 下的同名条目（路径常量见 `ls_common.DET_RESULTS`）；`目标检测/` 目录本身只放脚本、具名库（`lib/`）、配置（`config/`）与回归测试（`tests/`）。
> **比例尺标定（2026-09-20 迁移）**：离线 `scale_table.csv` 与产出脚本 `06_scale_calibrate.py` 已退役。
> px_per_um 改由**检测阶段双引擎现测**（`scalebar_duo.py`：A 本地形态学+OCR / B VLM，交叉验证），
> 结果写进 prelabel JSON 的 `scale` 字段，下游（LS 任务 data、`seg_manifest.csv`、几何测量交付清单）一律从该字段取。
> 检测层历史备份（`prelabels_backup_v6/`、`prelabels_miss_backup_v4/` 等）已移至项目根 `archive/detection_history/`。

## ⭐ 当前进度与续跑（2026-02 更新）

- ✅ 管线全链路就绪并对真实服务实测；试点项目 **id=1** 已建
- ✅ 比例尺标定（旧，已退役）：曾用 `scale_table.csv` 覆盖 216 张（157 实测 px/μm + 59 同药材中位数兜底）。**2026-09-20 起改为检测阶段双引擎现测，见下**
- ✅ **v7.1 已同步 LS（2026-09-15）**：`sync_prelabels_to_ls.py` 完成 216 张旧任务 predictions 替换（旧 339 框弃用，用户确认一并清除 36 条旧人工标注）+ 201 张新任务导入；项目 id=1 现为 **417 任务 / 417 条 v7.1 predictions / 803 框 / 0 人工标注**。manifest.csv 曾被外部进程重建（丢 数据量太少合集 745 行，11:32），已用 `build_manifest.py` 恢复为 5145 行全量（外部版本备份 `manifest_外部重建4400行_20260915.bak.csv`）
- ▶ **下一步**：人工审核 http://127.0.0.1:8080/projects/1/data（201 张新图曾无比例尺标定；2026-09-20 后跑 `vlm_prelabel.py --only-scale` 补测 prelabel JSON，再跑 `refresh_task_data.py` 即可同步进任务 data）
- ✅ 漏检补标（v5/v6）已被 v7.1 全量覆盖，`archive/detection_history/prelabels_miss_backup_v4/` 仅存档
- ✅ v7.1 同步已完成（见上 2026-09-15 条目：216 张替换 predictions + 201 张新建任务），本条旧计划作废；当前待办以上一条"下一步"为准
- 重跑/续跑命令（脚本均有断点跳过逻辑，重复运行安全）：

```powershell
cd D:\dataset\TCM-Microscopic-Image-Recognition\目标检测
..\.venv\Scripts\python.exe cli\vlm_prelabel.py --dirs "大黄导管,针晶" --limit 5   # 目录先验模式试点
..\.venv\Scripts\python.exe cli\vlm_prelabel.py --dirs "大黄草酸钙簇晶,大黄导管,大黄淀粉粒,党参石细胞,板蓝根石细胞,常山导管,常山淀粉粒,常山草酸钙针晶,黄柏石细胞,麦冬石细胞,麦冬针晶,龙胆针晶,猪苓方晶" --force
..\.venv\Scripts\python.exe cli\vlm_prelabel.py          # 旧 pilot 模式仍可用（默认 pilot_manifest.csv）
```

- `02` 目录先验模式参数：`--dirs`（目录名或关键词，命中根 manifest.csv，自动启用先验 prompt）、`--only`（按文件名 stem 精确重试）、`--force`（覆盖已有 JSON）、`--sample`（按归一化类名抽样）。标签自动归一化：簇晶→草酸钙簇晶、方晶→草酸钙方晶、针晶→草酸钙针晶；目录先验与 manifest 标签不一致时以目录为准并告警。
- 输出 JSON 新增 `prompt_version` 与 `dir_prior` 字段，便于按版本追溯。

- 修坑 2 则（全量跑时发现）：
  7. **不同目录同名图（如 石细胞_1 在两个目录）会碰撞**：02 输出键与 07 叠加图名已加路径短哈希消歧（`石细胞_1_b488d4`），跳过逻辑先核对 image_path 再跳。
  8. **`GET /api/tasks?project=N` 默认不返回 predictions**：必须带 `fields=all`（04 导出脚本已内置）；核验用 `total_predictions` 计数或 `GET /api/tasks/{id}`。

### 踩坑与修正记录（重要，换模型/换图前必读）

1. **glm-4.5v 返回 0~1000 归一化坐标**，不遵守"按像素坐标"的 prompt 要求——首版按像素解析导致框整体偏移、切断实体。已改为 prompt 明确要求 0~1000 坐标系 + `x/1000*W` 映射。**换其他厂商模型时必须先抽 5 张目检框位置**（`render_prelabels.py` 渲染后人工看）。
2. **疑似待定策略**：大面积组织块/纤维束不得框成"疑似待定"（首版曾把整片组织框进去）；离散小实体拿不准才归此类。
3. **导管严禁过分割**：密度提示曾导致一根导管的具缘纹孔段被拆成 8 个"方晶"框；已加"导管是连续单一结构、整段一框"硬条款 + 方晶排他描述（独立锐利单晶 vs 相连壁物质）。
4. **VLM 尺寸自估不可靠**（普遍偏小且粗糙），已从输出中弃用；μm 走几何换算——分割上线前用紧框径 × prelabel JSON 里的 px_per_um 作临时估算（该值由双引擎实测），正式测量以 mask 为准。
5. LS 1.23 API **不接受 zip 导入**；用本地文件直挂（`LOCAL_FILES_SERVING_ENABLED` + `/data/local-files/?d=<相对D:\dataset的路径>`，中文需 URL 编码）。
6. 检出的每图预标注框数量跨度大（1~12 框）：单框大目标正常，密集淀粉粒逐个框；审核时按需补漏。
7. **v7 先验 prompt 的两个失败模式（已修复，改 prompt 前先读）**：① 只强调"必框不能漏"会诱发**空白背景网格幻觉**（40 个等距虚拟框、真目标反而不框）——必须同时写明"找不到就如实输出空 objects、严禁凑数/网格布框"；② 针晶束被拆成几十个小框——已加"整束一框"硬条款。**每次改 prompt 先用 `--dirs … --only … --force` 抽 5~10 张渲染目检再全量**。
8. **max_tokens 不足导致 JSON 截断**：密集针晶图输出 30~46 框时 4096 token 不够，已调到 8192；`parse_vlm_json` 另有截断修复兜底（截到最后一个完整 object）。


## 0. 一次性准备

| 步骤 | 操作 |
|---|---|
| 启动服务 | 双击/运行 `start_labelstudio.ps1`，地址 http://127.0.0.1:8080 |
| 注册账号 | 首次打开网页 → Sign Up 注册（第一个账号即管理员；单人审核就用它） |
| 拿 LS Token | 网页右上角 Account & Settings → Access Token，复制（legacy Token 或 JWT refresh token 均可，JWT 会自动换取 access token） |
| 设置环境变量 | `setx LS_TOKEN "你的token"`（或直接填进 vlm_config.json 的 labelstudio.token）；`setx VLM_API_KEY "你的多模态大模型Key"`（新开终端生效） |
| 确认 VLM 配置 | `vlm_config.json` 里 `base_url`/`model` 当前默认智谱 `glm-5.3-flash` + `reasoning_effort: low`（OpenAI 兼容接口，换厂商只改这两行；**`reasoning_effort` 仅 GLM-5.2 及以上支持，换回 GLM-4.6V 时必须删掉该键**，否则请求被拒） |

## 1. 阶段一：AI 预标注

```powershell
cd D:\dataset\TCM-Microscopic-Image-Recognition\目标检测
# ⓪ 抽样：01_sample_pilot.py 已于 2026-09-29 归档到 ..\archive\detection_scripts\
#    （P0 试点项目在 LS 中已退役，现存项目全是 ui检测-*/ui分割-*/检测评测-gold）；
#    现有 results/detection/pilot_manifest.csv 可直接用，无需重建。
..\.venv\Scripts\python.exe cli\vlm_prelabel.py --limit 5   # ① 先跑5张看效果（检查框贴合度、类别对不对）
..\.venv\Scripts\python.exe cli\vlm_prelabel.py        # ② 满意后全量，4并发可断点续跑（已存在的json自动跳过）
```

- 输出：`prelabels/<类>_<图名>.json`，每图含 `tight_bbox`（像素坐标，已从推理分辨率映射回原图 2048×1536）、`confidence`（high=可采信 / medium=必须复核）、`morphology`、`size_um`（**紧框径 × px/μm 的临时估算**，仅作判据核验参考；正式测量以分割 mask 为准）。
- 预标注**只作参考不直接入库**，全部经阶段二人工确认（方案硬性要求）。

## 2. 阶段二：导入 Label Studio 人工审核

```powershell
..\.venv\Scripts\python.exe annot\import_to_labelstudio.py --project-id 1
```

- 图片**不复制不打包**：通过 LS 本地文件服务直挂原始路径（`start_labelstudio.ps1` 已开 `LOCAL_FILES_SERVING_ENABLED`，`LOCAL_FILES_DOCUMENT_ROOT` = **项目根**（`$PSScriptRoot\..\..`，2026-09-20 图像库迁入 `<项目>\dataset\` 后调整），脚本会自动注册存储）。注意：LS 1.23 的 API 不接受 zip 导入。
- 项目「中药显微P0试点-簇晶石细胞导管」已建好（**id=1**）；AI 预标注以 **predictions** 进来，人工提交的是权威 annotations，两不混淆。
- 标注操作：打开任务 → AI 框显示为虚线建议 → 逐框**采信/拖动修正/删除/补框** → 按 §4.5 框画法硬条款（紧贴 ±2%、禁止切断实体、重叠目标各自成框）→ Submit。
- 每个框可填「判据核验」（大黄簇晶 60~140μm 等），存入属性表供后续规则 Agent 使用。
- 人工审核完成后，在项目 Settings → Collaboration & Access 打开 **Review Stream**，走一遍审核队列把所有任务 Accept（单人即自查二审）。

## 3. 自检：10% 盲标 + Kappa

```powershell
..\.venv\Scripts\python.exe annot\selfcheck_kappa.py --source-project-id <项目id> --create
# → 自动建「自检(盲标)」项目，抽约22张（不看第一轮结果）
# 间隔至少一天后独立盲标完成，再跑：
..\.venv\Scripts\python.exe annot\selfcheck_kappa.py --source-project-id <项目id> --report
```

- 验收线（方案 §六）：**Kappa ≥ 0.7 且 IoU≥0.5 匹配对平均 IoU ≥ 0.7** → PASS 可量产；否则先修规范。

## 4. 导出

```powershell
# 04_export_yolo.py 已于 2026-09-29 归档到 ..\archive\detection_scripts\
#（YOLO 导出从未落地：results/detection/yolo_export/ 为空、models/ 无 YOLO 权重；
#  若后续要训本地检测器，从 archive 取回即可）
# ..\.venv\Scripts\python.exe 04_export_yolo.py --project-id <项目id>
```

- 产出 `yolo_export/`：`images/` + `labels/`（YOLO 归一化紧框）+ `classes.txt`（id 对应 feature_taxonomy 的 feature_id）+ `attributes.csv`（判据核验/备注/组号）+ `data.yaml`。
- 类别表（`vlm_config.json` yolo.classes）：0 簇晶 / 1 方晶 / 2 针晶 / 3 导管 / 4 石细胞 / 5 淀粉粒 / 6 疑似待定（默认不进 YOLO）。
- **防泄漏提醒**：试点合集导出仅用于检查；正式训练前按 `attributes.csv` 的 `group_key` 做组感知划分。

## 5. 脚本清单

| 文件 | 作用 |
|---|---|
| `vlm_config.json` | 全部配置：VLM 接口、试点抽样、LS 地址/token、YOLO 类别表 |
| `../common/ls_common.py` | 公共库（中文路径安全 imread 等，坑已封装；2026-09-29 上收至共享层，2026-10-09 起在项目根 `common/`） |
| ~~`01_sample_pilot.py`~~ | **已归档（2026-09-29 → `..\archive\detection_scripts\`）**：manifest → P0 试点抽样清单；试点项目已在 LS 中退役，现有 `pilot_manifest.csv` 直接可用 |
| `vlm_prelabel.py` | 多模态大模型 API 批量预标注（断点续跑、重试、坐标映射；**v7.1 目录先验模式**：`--dirs` 按目录名注入"该特征必框+其他类别也框+禁凑数"prompt，`--only/--force/--sample` 辅助） |
| `import_to_labelstudio.py` | 打包图+预标注 → 创建项目导入 |
| ~~`04_export_yolo.py`~~ | **已归档（2026-09-29 → `..\archive\detection_scripts\`）**：LS 人工标注 → YOLO + 属性表；从未落地（`yolo_export/` 空、无 YOLO 权重），要训本地检测器时再取回 |
| ~~`08_relabel_miss.py`~~ / ~~`09_update_miss_predictions.py`~~ | **已归档（2026-09-29 → `..\archive\detection_scripts\`）**：v5/v6 补漏战役（定向重跑漏检图 + 回写 LS 预测）；`prelabels/` 现存全部为 v7.1/v7.2，已无 v5/v6 可作用 |
| `selfcheck_kappa.py` | 盲标抽样 + IoU/Kappa 验收报告 |
| ~~`06_scale_calibrate.py`~~ | **已删除（2026-09-20）**：离线批量比例尺标定（→ `scale_table.csv`）退役，改由 `scalebar_duo.py` 双引擎在检测阶段现测；存量 prelabel 用 `vlm_prelabel.py --only-scale` 回填，回填后若有一批 `OK-medium` 可再跑 `vlm_prelabel.py --relax-consensus`（离线升为 `OK-consensus`，不耗 API） |
| `render_prelabels.py` | 渲染预标注叠加图 → `prelabel_review/`（绿=high 橙=复核 灰=疑似待定，标注框径μm），prompt 调优/换模型的目检验收工具 |
| `tests/` 下 `test_import_chain.py` / `test_scalebar_consensus.py` / `test_import_all.py` / `test_detect_adaptive.py` / `cleanup_e2e.py` / `smoke_cleanup.py` | 离线/在线回归测试与清理（2026-09-20 由 detection/ 根移入 tests/；`test_scalebar_consensus.py` 2026-09-28 新增，覆盖 OK-consensus 判据与防自我放大；`test_export_selfcheck_e2e.py` 因写死已退役的 `PID=1` 已于 2026-09-29 归档） |
| `config/vlm_config.json` 内 `token` | 已存 LS JWT refresh token（环境变量 LS_TOKEN 优先） |
| `config/labeling_config.xml` | 标注界面配置（7类矩形 + 逐框判据核验 + 备注）；判据页头为动态 `$criteria_hint`（按任务显示该药材×特征的判据） |
| `config/criteria_table.json` | 药材×特征 判据速查表（单一事实源；含比例尺不可信提示），`ls_common.criteria_hint_for()` 统一取用（status 不在可信档时追加 `scale_caveat`；可信档 = `ls_common.TRUSTED_STATUSES` = OK / **OK-consensus**）。2026-09-22 已删除「大黄\|石细胞」排他提示行（唯一出处为示例文档、非药典依据）；已推送到 LS 项目的任务需重跑 `annot/sync_criteria.py` 才会刷新 |
| `config/folder_feature.json` | 药材级目录 → 主特征映射（目录先验；`prompts.prior_from_dir()` 唯一取用处） |
| `scalebar_duo.py` | 比例尺双引擎标定（A 本地形态学+OCR / B VLM，交叉验证）→ px_per_um。公共 API `measure_px_per_um(path, cfg, px_fam=…, px_ok_set=…)` / `measure_img_px_per_um(...)`；02 检测阶段逐图调用并写入 prelabel JSON 的 `scale` 字段；独立运行只出对照报告 `results/detection/scalebar_duo/scalebar_duo_report.csv`。**status 五档**：OK（双读一致）/ **OK-consensus**（B 未证实但 A 的 bar_px 已被同批精确 OK 样本旁证，2026-09-28 补）/ OK-medium（无旁证）/ OK-low（族先验·B 兜底）/ manual·n/a；可信与否统一走 `ls_common.is_trusted_status()` |
| `sync_prelabels_to_ls.py` | 把 prelabels 中 v7.x 预标注同步到 LS：已有任务替换 predictions（`--purge-annotations` 连人工标注一起清）、缺失任务组装导入（含 criteria_hint），`--dry-run` 预览 |
| `refresh_task_data.py` | 从 prelabels **重建完整任务 data**（prelabel_summary/criteria_hint/px_per_um 等 9 字段）。⚠️ LS 任务级 PATCH 对 data 是**整体替换**——提交部分字段会抹掉 image/herb 等其余列（2026-09-15 曾发生并由本脚本修复）；改动任务 data 一律走本脚本或提交完整字典 |
| `build_tif_proxies.py` | 为任务引用的 tif 生成浏览器可渲染的 jpg 代理 → `results/detection/ls_view/`（镜像原目录结构）并注册 LS 本地存储；`ls_common.view_file_url()` 统一映射（tif 任务 image 指向代理，jpg 原样）。⚠️ 浏览器原生不渲染 TIFF，tif 任务必须走代理。⚠️ 代理目录里还需带 `dataset/` 一层（与 `DATASET` 的 relpath 一致），现有代理仍是旧布局、需重跑本脚本 |
| `det_eval.py` | **检测质量评测（gold set）**：`--build-gold` 按类分层抽 N 张（seed 可复现）→ `det_eval/gold_set.json`；`--import-ls` 建「检测评测-gold(盲标)」项目并导入（**不带 predictions**，防评测自证）；`--report`（`--project-id` 或 `--export-json`）出 per-class 定位召回/端到端召回/精确、IoU 分布、**带符号覆盖率偏差**（>1 = 框偏小 → 测量偏低）、贴边/巨型框、框数误差与 PASS/FAIL 验收线 → `det_eval/report.json` + `report.md`；`--selftest` 离线合成真值自检。改 prompt / 换模型后重跑 `--report` 即可用同一批图对比同一组数字 |
| `start_labelstudio.ps1` | 启动服务（已放开 D:\dataset 本地文件服务，备用） |
