# -*- coding: utf-8 -*-
"""test_import_chain.py — 离线验证 03 导入链路：合成预标注 → 任务组装结构校验（自清理，不联网）

另含三道环境护栏（都不联网、不落盘在仓库里）：② 启动/冒烟 ps1 的 BOM 编码护栏；
③ 路径落点护栏；④「检测 → 分割」crop 随行 json 的写读闭环（见文末各节注释）。"""
import json
import os
import shutil
import urllib.parse

# 本脚本位于 detection/tests/：ls_common 在 标注体系/common/，入口在上一级 detection/annot/
DET_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from ls_common import DATASET  # noqa: E402

# 1) 合成预标注 + 假图（必须放在 D:\dataset 下，local_file_url 才能转相对路径）
workdir = os.path.join(DATASET, "_tmp_ls_chain_test")
os.makedirs(workdir, exist_ok=True)
fake_img = os.path.join(workdir, "1.jpg")
open(fake_img, "wb").write(b"\xff\xd8FAKEJPEG")

fake = {
    "source": "vlm_api", "model": "test",
    "feature_label_folder": "草酸钙簇晶", "herb": "大黄",
    "group_key": "大黄-seq", "needs_review": "",
    "original_size": [2048, 1536], "inference_size": [1536, 1152],
    "image_path": fake_img,
    "objects": [
        {"label": "草酸钙簇晶", "tight_bbox": [610.0, 258.0, 1302.0, 876.0],
         "confidence": "high", "score": 0.92, "morphology": "多角形星状",
         "size_um": 68.2, "note": ""},
        {"label": "疑似待定", "tight_bbox": [643.0, 1305.0, 806.0, 1448.0],
         "confidence": "medium", "score": 0.65, "morphology": "小晶体",
         "size_um": 16.1, "note": ""},
    ],
}
import ls_common
ls_common.PRELABEL_DIR = workdir
import importlib.util
spec = importlib.util.spec_from_file_location(
    "imp03", os.path.join(DET_DIR, "annot", "import_to_labelstudio.py"))
imp03 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(imp03)
imp03.PRELABEL_DIR = workdir

with open(os.path.join(workdir, "草酸钙簇晶_1.json"), "w", encoding="utf-8") as f:
    json.dump(fake, f, ensure_ascii=False)

tasks, _qc = imp03.build_tasks()      # 2026-10-09 起返回 (tasks, qc_rows)：无 --single-target-dirs 时 qc 为空
assert _qc == []
assert len(tasks) == 1
t = tasks[0]
# local-files URL 正确性
assert t["data"]["image"] == "/data/local-files/?d=" + urllib.parse.quote(
    "_tmp_ls_chain_test/1.jpg"), t["data"]["image"]
assert t["data"]["herb"] == "大黄" and t["data"]["group_key"] == "大黄-seq"
assert json.loads(t["data"]["prelabel_summary"])[0]["morphology"] == "多角形星状"
pred = t["predictions"][0]["result"]
assert len(pred) == 2
r0 = pred[0]["value"]
assert abs(r0["x"] - 29.785) < 0.01 and abs(r0["y"] - 16.797) < 0.01, r0
assert abs(r0["width"] - 33.789) < 0.01 and abs(r0["height"] - 40.234) < 0.01, r0
assert r0["rectanglelabels"] == ["草酸钙簇晶"] and pred[0]["score"] == 0.92
assert pred[0]["original_width"] == 2048 and pred[0]["original_height"] == 1536
assert pred[0]["from_name"] == "feature" and pred[0]["to_name"] == "image"

shutil.rmtree(workdir)  # 自清理

# 2) 启动脚本编码护栏：Windows PowerShell 5.1 把无 BOM 的 .ps1 按 ANSI/GBK 解析，
#    UTF-8 中文注释被误读后可能吞掉相邻代码行（2026-09-22 实锤：start_labelstudio.ps1 的
#    $env:LOCAL_FILES_DOCUMENT_ROOT 赋值因此从未生效 → LS 文件服务根退回盘根 D:\，
#    /data/local-files 全 404 → 标注页报 "There was an issue loading URL from $image value"）。
#    守护：启动 ps1 必须纯 ASCII；若含非 ASCII 字节（如根目录脚本引用中文目录 标注体系）
#    则必须带 UTF-8 BOM。
for _ps1 in (os.path.join(DET_DIR, "start_labelstudio.ps1"),
             os.path.join(DET_DIR, "..", "ui", "start_manual_annotate.ps1"),
             os.path.join(DET_DIR, "..", "start_run_viewer.ps1"),
             # 2026-09-29 补：本脚本含中文注释，同样必须带 BOM —— 编辑工具重写会**静默剥掉** BOM，
             # 剥掉后 PowerShell 5.1 按 GBK 误解析，param 块直接语法错（实测踩到）。
             os.path.join(DET_DIR, "..", "run_demo_manual_measure.ps1")):
    _raw = open(_ps1, "rb").read()
    assert not any(b >= 0x80 for b in _raw) or _raw[:3] == b"\xef\xbb\xbf", (
        f"{_ps1}: 含非 ASCII 字节且无 UTF-8 BOM —— PowerShell 5.1 会按 GBK 误解析，"
        f"中文注释可能吞掉相邻代码行（见 start_labelstudio.ps1 头部注释）")

# 2b) 启动脚本的**落点**断言（2026-10-09 补）：start_labelstudio.ps1 把 LS 的文件服务根设为
#     `Resolve-Path "$PSScriptRoot\<相对式>"` —— 这是"按自身位置推深度"的又一例，且**编码护栏抓不到**
#     （它只查「必须纯 ASCII」）。2026-10-09 搬目录后该式仍是 `..\..`（旧深度 标注体系/detection），
#     解析成 D:\dataset 而非项目根 —— 服务根错了，标注页 URL 会 404，而全量回归当时全绿。
#     故这里把相对式**求值后与 paths.PROJECT_ROOT 对拍**，改深度即红。
import re as _re                       # noqa: E402
_LS_PS1 = os.path.join(DET_DIR, "start_labelstudio.ps1")
_ls_txt = open(_LS_PS1, encoding="ascii").read()
_m = _re.search(r'LOCAL_FILES_DOCUMENT_ROOT\s*=\s*\(Resolve-Path\s*"\$PSScriptRoot([^"]*)"\)',
                _ls_txt)
assert _m, ("start_labelstudio.ps1 未按预期设置 LOCAL_FILES_DOCUMENT_ROOT —— "
            "若改了写法，请同步更新本护栏（它防的是服务根算错）")
_rel = _m.group(1).replace("/", os.sep).lstrip("\\")
_root_from_ps1 = os.path.normcase(os.path.abspath(os.path.join(DET_DIR, _rel)))
assert _root_from_ps1 == os.path.normcase(os.path.abspath(ls_common.PROJECT_ROOT)), (
    f"start_labelstudio.ps1 的文件服务根应解析为项目根，实际是 {_root_from_ps1}"
    f"（相对式 '$PSScriptRoot{_m.group(1)}' × $PSScriptRoot={DET_DIR}）")

# 3) 路径落点护栏（2026-09-29 补；P4-14 的同类回归之后）
#    P4-14 的教训：模块搬进子目录后，按 ".." 相对自身推导的常量会**静默错一层**
#    （igm_segment 因此找不到 feature_taxonomy.json，IGM 阶段整段崩），而生产自检走的是
#    本地合成路径、根本碰不到。detection/ 侧同一批搬迁也留了一处：detect_sample 的 ROOT
#    退到 detection/ → `--group` 去读不存在的 manifest.csv。故这里两件事都钉住：
#    ① 现役脚本不得再按自身位置推深度（一律从 ls_common / paths 取）；② 关键落点确实在磁盘上。
CLI_DIR = os.path.join(DET_DIR, "cli")
ANNOT_DIR = os.path.join(DET_DIR, "annot")

# 扫描半径（2026-09-29 扩到全仓：detection 侧早已守住，但 ui/、common/、几何测量/、Agent判药/
# 与项目根脚本同样可能写出这类坏点，而它们**没有**任何护栏）。experiments/ 与 archive/ 冻结不扫。
_SKIP_DIRS = {"archive", "results", ".venv", "experiments", "tests", "__pycache__"}
# 2026-10-09 目录重构：DET_DIR 由 <项目>/标注体系/detection 变为 <项目>/目标检测，故扫描根
# 必须显式列出（原先靠 dirname(dirname(DET_DIR)) 推项目根，现在会推到项目根的**父目录**）。
_PROJ = os.path.dirname(DET_DIR)                          # 项目根
_SCAN_ROOTS = [(os.path.join(_PROJ, "目标检测"), True),
               (os.path.join(_PROJ, "图像分割"), True),
               (os.path.join(_PROJ, "common"), True),
               (os.path.join(_PROJ, "ui"), True),
               (os.path.join(_PROJ, "几何测量"), True),
               (os.path.join(_PROJ, "Agent判药"), True),
               (_PROJ, False)]                            # 项目根本层


def _iter_sources():
    """产出待扫的 .py 路径（去重；跳过 _tmp_*.py 与冻结目录）。"""
    seen = set()
    for base, recursive in _SCAN_ROOTS:
        if not os.path.isdir(base):
            continue
        cand = []
        if recursive:
            for dirpath, dirnames, filenames in os.walk(base):
                dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]
                cand += [os.path.join(dirpath, fn) for fn in filenames if fn.endswith(".py")]
        else:
            cand = [os.path.join(base, fn) for fn in sorted(os.listdir(base))
                    if fn.endswith(".py")]
        for p in cand:
            if os.path.basename(p).startswith("_tmp_") or p in seen or not os.path.isfile(p):
                continue
            seen.add(p)
            yield p


_sources = list(_iter_sources())
# 护栏自检：半径或跳过规则写错时会"扫到 0 个文件"从而永远通过 —— 故钉一个下限
assert len(_sources) >= 30, f"路径落点扫描只找到 {len(_sources)} 个 .py，护栏本身失效了"
print(f"  路径落点扫描：{len(_sources)} 个现役 .py（全仓除 archive/results/venv/experiments/tests）")

_forbidden = []
for _src_path in _sources:
    _src = open(_src_path, encoding="utf-8").read()
    # 收集"由自身位置派生"的变量名，再找把 ".." 与它合用的行（同行的 __file__ + ".." 也拦）
    _self_dirs = set()
    for _line in _src.splitlines():
        if "__file__" in _line and "os.path.dirname" in _line and "=" in _line:
            _name = _line.split("=")[0].strip().split()[-1]
            if _name.isidentifier():
                _self_dirs.add(_name)
    for _i, _line in enumerate(_src.splitlines(), 1):
        if '".."' not in _line and "'..'" not in _line:
            continue
        if "__file__" in _line or any(_n in _line for _n in _self_dirs):
            _forbidden.append("{}:{}: {}".format(
                os.path.relpath(_src_path, os.path.dirname(DET_DIR)), _i, _line.strip()))
            break
assert not _forbidden, (
    "现役脚本不得按自身位置推深度（搬目录即静默错层，见 P4-14）；请改用 ls_common / paths 常量。\n  "
    + "\n  ".join(_forbidden))

# ② 关键落点（都是本次搬迁后实测过、或判据链路上必需的文件）
for _p in (os.path.join(ls_common.PROJECT_ROOT, "manifest.csv"),
           os.path.join(ls_common.DET_CONFIG, "labeling_config.xml"),
           os.path.join(ls_common.DET_CONFIG, "criteria_table.json"),
           os.path.join(ls_common.DET_CONFIG, "folder_feature.json")):
    assert os.path.isfile(_p), f"应有文件不存在：{_p}"
assert os.path.isdir(ls_common.DET_RESULTS), ls_common.DET_RESULTS

# ③ 钉住 P4-13 的子域分组：annot/ 收 ↔LS 回路，cli/ 留产出与目检
_ANNOT_ENTRIES = {"import_to_labelstudio.py", "sync_prelabels_to_ls.py", "refresh_task_data.py",
                  "sync_criteria.py", "selfcheck_kappa.py"}
_CLI_ENTRIES = {"vlm_prelabel.py", "detect_sample.py", "render_prelabels.py", "det_eval.py",
                "build_tif_proxies.py"}
assert _ANNOT_ENTRIES <= set(os.listdir(ANNOT_DIR)), sorted(os.listdir(ANNOT_DIR))
assert _CLI_ENTRIES <= set(os.listdir(CLI_DIR)), sorted(os.listdir(CLI_DIR))

# ④ detect_sample 的 manifest 落点（P4-14 同类的具体修复点）
_spec = importlib.util.spec_from_file_location(
    "det_sample", os.path.join(CLI_DIR, "detect_sample.py"))
_ds = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ds)
assert os.path.isfile(_ds.MANIFEST_ROOT), _ds.MANIFEST_ROOT
assert os.path.normcase(os.path.dirname(_ds.MANIFEST_ROOT)) == os.path.normcase(ls_common.PROJECT_ROOT), (
    f"detect_sample 的 manifest 应取 项目根 下的那份，实际是 {_ds.MANIFEST_ROOT}")
# 先验取自"样品身份"而非容器目录名：--dir <样品>/images 不能把药材推成 images（图像在容器里，退两级）
assert _ds.prior_dir_name(os.path.join("R", "虎杖草酸钙簇晶", "images", "1.jpg"), "images") \
    == "虎杖草酸钙簇晶"
assert _ds.prior_dir_name(os.path.join("R", "大黄导管", "1.jpg"), "大黄导管") == "大黄导管"
assert _ds._full_dir_herb("虎杖草酸钙簇晶") == "虎杖", _ds._full_dir_herb("虎杖草酸钙簇晶")
# 裁剪规则必须来自共享原语（全流程约定）——统一前这里自带一套"每边 ×30%"，属重复实现
from geom import loose_box as _lb        # noqa: E402
assert _ds.loose_box is _lb, "detect_sample 的松框应取自 common/geom.loose_box（见 docs/分割方案与设计.md §1.2）"
# 外扩量不能收小：收到默认 5%/24px 时，IGM 极性探针（靠四角是背景判极性）会误判、mask 反转
# （2026-09-29 实测 2/3 被标 review、1 个 mask 真错）。故钉一个下限，防"顺手改回默认"。
assert _ds.DET_LOOSE_MARGIN_RATIO >= 0.20, (
    f"DET_LOOSE_MARGIN_RATIO={_ds.DET_LOOSE_MARGIN_RATIO} 太小 —— 会让 IGM 极性探针误判（见常量注释）")

# 特征词表一致性（2026-09-30）：目录切分词表（build_manifest.FEATURE_SUFFIXES）与目录先验
# 关键词表（prompts.PRIOR_KEYWORDS）是两处独立维护的清单，历史上漂移过：木纤维/菊糖/升华物/
# 晶纤维 只在切分表里 → 那 6 个特征目录的 herb 退化成整个目录名。这里钉住"切分表 ⊆ 先验表"。
# 2026-10-09：build_manifest.py 已随知识资产上提到项目根（在 .pth 注入点内），无需再手动注入 sys.path。
import build_manifest as _bm              # noqa: E402
from prompts import PRIOR_KEYWORDS as _KW  # noqa: E402
_missing = [s for s in _bm.FEATURE_SUFFIXES if not any(k in s or s in k for k, _ in _KW)]
assert not _missing, (
    f"目录切分表里的特征词在 prompts.PRIOR_KEYWORDS 里缺关键词：{_missing}"
    f"（缺则那些目录的药材名剥离不出来）")

# 4) crop 随行 json 的**写读闭环**（「检测 → 分割」契约，2026-09-29 加）
#    写侧唯一实现 = `contracts.crop_sidecar()`（detect_sample 产 VLM 检测产物时用它）；
#    读侧 = `contracts.load_crop_metas()` + `check_crops()`（igm_segment_batch 装载时用）。
#    两侧键名一旦漂移，只有跑到真 IGM 才会炸 —— 而那时已经烧完 VLM + IGM 额度，
#    故此处用合成样本把闭环钉住（不联网、不落盘在仓库里）。
import tempfile

from contracts import ContractError, check_crops, crop_sidecar, load_crop_metas

_tmp = tempfile.mkdtemp(prefix="_tmp_crop_contract_")
try:
    _cid = "cluster_crystal_1_01"
    _side = crop_sidecar(_cid, _cid + ".png", os.path.join(_tmp, "src.jpg"), "1",
                         "cluster_crystal", "草酸钙簇晶", "虎杖", "11.7",
                         [676, 482, 1284, 1165], [31, 31, 639, 714], [645, 451, 1314, 1195],
                         "669x744")
    with open(os.path.join(_tmp, _cid + ".json"), "w", encoding="utf-8") as f:
        json.dump(_side, f, ensure_ascii=False)
    open(os.path.join(_tmp, _cid + ".png"), "wb").write(b"\x89PNG\r\n\x1a\n")

    _metas = load_crop_metas(_tmp)
    check_crops(_metas)                       # feature_id / stem / px_per_um 齐 → 不抛
    _m = _metas[0]
    assert (_m.crop_id, _m.stem, _m.feature_id, _m.px_per_um_text, _m.crop_size) == \
        (_cid, "1", "cluster_crystal", "11.7", "669x744"), vars(_m)
    assert _m.loose_box == [645, 451, 1314, 1195], _m.loose_box

    # 量化可用性闸门（2026-10-09）：granularity 随 crop 侧车下传（默认 instance），
    # 并经交付清单（MeasureManifestRow）透传到几何。三处键名漂移只会在真流程才炸，故在此钉住。
    from contracts import MeasureManifestRow as _MMR
    assert _m.granularity == "instance", _m.granularity
    assert crop_sidecar("vessel_1_02", "x.png", "", "1", "vessel", "导管", "虎杖", "11.7",
                        None, None, None, "80x60", granularity="bundle")["granularity"] == "bundle"
    assert _MMR.from_row({"granularity": "bundle"}).granularity == "bundle"
    assert _MMR.from_row({}).granularity == "instance"   # 旧清单缺 granularity 列 → instance

    # 缺 px_per_um 必须被 check_crops 拦下（不静默降级——否则会产出无 μm 的测量行）
    with open(os.path.join(_tmp, _cid + ".json"), "w", encoding="utf-8") as f:
        json.dump(crop_sidecar(_cid, _cid + ".png", "", "1", "cluster_crystal", "草酸钙簇晶",
                              "虎杖", "", None, None, None, "669x744"), f, ensure_ascii=False)
    try:
        check_crops(load_crop_metas(_tmp))
        raise AssertionError("缺 px_per_um 竟未被 check_crops 拦下")
    except ContractError:
        pass
finally:
    shutil.rmtree(_tmp, ignore_errors=True)

print("IMPORT-CHAIN TEST: ALL ASSERTIONS PASS")