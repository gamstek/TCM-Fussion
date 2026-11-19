# -*- coding: utf-8 -*-
"""
import_to_labelstudio.py — 阶段一→二衔接：预标注导入 Label Studio
- 读取 prelabels/*.json → 组装 LS 任务（AI 框以 predictions 导入，人工在标注队列里采信/修正）
- 图片不复制不打包：通过 LS 本地文件服务直挂原始路径（需 start_labelstudio.ps1 中的
  LOCAL_FILES_SERVING_ENABLED=true + LOCAL_FILES_DOCUMENT_ROOT=<项目根>，已配置）
- 数据集图像仅引用相对项目根的路径，中文字符自动 URL 编码
前置:
  1. Label Studio 已启动（见 start_labelstudio.ps1）
  2. token 已配置（环境变量 LS_TOKEN 或 vlm_config.json；JWT refresh token 自动换取 access）
用法:
  python annot\import_to_labelstudio.py              # 自动建项目（若 vlm_config 未指定）并导入
  python annot\import_to_labelstudio.py --project-id 1   # 追加导入到已有项目

训练集子集模式（2026-10-09，草酸钙簇晶单类第一批）：下列参数**默认全关**，不加即与旧行为逐字节一致：
  python annot\import_to_labelstudio.py ^
      --dirs "大黄草酸钙簇晶,白芍,马齿苋,牡丹皮,辣蓼,赤芍,虎杖" ^
      --labels "草酸钙簇晶" --single-target-dirs "大黄草酸钙簇晶,虎杖" ^
      --title "检测训练集-草酸钙簇晶(单类)" ^
      --label-config config\labeling_config_cluster.xml
  --dry-run 只组装+统计+写 QC 报告，不建项目、不导入。
"""
import argparse
import csv
import json
import os
import sys


from contracts import PrelabelRecord  # noqa: E402 第 ⑤ 段契约（prelabel 记录）
from ls_common import (DET_CONFIG, DET_RESULTS, PRELABEL_DIR, PROJECT_ROOT, VIEW_ROOT,  # noqa: E402
                       canon_path, ensure_dirs, load_config, ls_api, criteria_hint_for,
                       prelabel_scale, view_file_url)
from ls_tasks import ensure_local_storage, import_in_chunks  # noqa: E402 LS 项目/任务读写（P4-3 上收）

LABEL_CONFIG_PATH = os.path.join(DET_CONFIG, "labeling_config.xml")
TRAINSET_RESULT_DIR = os.path.join(DET_RESULTS, "trainset_cluster")
DEFAULT_QC_PATH = os.path.join(TRAINSET_RESULT_DIR, "qc_single_target.csv")


def feature_id_of(cfg, zh_label):
    for c in cfg["yolo"]["classes"]:
        if c["zh"] == zh_label:
            return c["feature_id"]
    return "unknown"


def build_tasks(dirs=None, labels=None, single_target_dirs=None):
    """读 prelabels → LS 任务。返回 (tasks, qc_rows)。

    三个过滤参数**均为可选，默认 None = 与旧行为逐项一致**（全量目录、全部类别、无 QC）：
      dirs              目录名集合（按源图父目录名精确匹配）：只导这些目录的图（训练集子集）。
      labels            类别集合：只保留这些框；目录命中但该图无此类框时**仍导入**（人工补框）。
      single_target_dirs 每图预期仅 1 个目标的目录：命中者写入 `cluster_hint` 页头提示，
                        并把框数记入 QC 报告（框数≠1 的图交人工优先复核）。
    """
    dirs_set = set(dirs or [])
    labels_set = set(labels or [])
    st_set = set(single_target_dirs or [])
    pre_files = sorted(f for f in os.listdir(PRELABEL_DIR) if f.endswith(".json"))
    if not pre_files:
        print(f"错误: {PRELABEL_DIR} 下没有预标注结果，请先运行 vlm_prelabel.py", file=sys.stderr)
        sys.exit(1)

    tasks, qc_rows = [], []
    for fn in pre_files:
        with open(os.path.join(PRELABEL_DIR, fn), encoding="utf-8") as f:
            pre = PrelabelRecord.from_dict(json.load(f))
        img_path = canon_path(pre.image_path)   # 兼容迁移前旧绝对路径
        folder = os.path.basename(os.path.dirname(img_path))
        if dirs_set and folder not in dirs_set:
            continue
        # 只保留指定类别的框（labels 为空则全留）；本图无此类框时仍建任务，供人工补框。
        objs = [b for b in pre.objects if not labels_set or b.label in labels_set]
        cluster_hint = ""
        if folder in st_set:
            cluster_hint = (f"该目录为【单目标】图：预期仅 1 个目标，本次 AI 给出 {len(objs)} 框；"
                            "出现多框请重点核对是否为过分割。")
            qc_rows.append({
                "image": os.path.relpath(img_path, PROJECT_ROOT),
                "folder": folder, "herb": pre.herb,
                "cluster_boxes": len(objs),
                "flagged": "否" if len(objs) == 1 else "是",
            })
        W0, H0 = pre.original_size
        results = []
        for j, b in enumerate(objs):
            x1, y1, x2, y2 = b.tight_bbox
            results.append({
                "id": f"pred_{j}",
                "type": "rectanglelabels",
                "from_name": "feature", "to_name": "image",
                "original_width": W0, "original_height": H0,
                "image_rotation": 0,
                "value": {
                    "x": round(x1 / W0 * 100, 3), "y": round(y1 / H0 * 100, 3),
                    "width": round((x2 - x1) / W0 * 100, 3),
                    "height": round((y2 - y1) / H0 * 100, 3),
                    "rotation": 0,
                    "rectanglelabels": [b.label],
                },
                "score": (b.score if b.score is not None else 0.6),
            })
        ppu, scale_status = prelabel_scale(pre)   # 检测阶段双引擎标定（prelabel JSON 的 scale 字段）
        tasks.append({
            "data": {
                "image": view_file_url(img_path),
                "original_filename": os.path.basename(img_path),
                "herb": pre.herb,
                "folder_feature": pre.feature_label_folder,
                "group_key": pre.group_key,
                "needs_review": pre.needs_review,
                "px_per_um": ppu,
                "criteria_hint": criteria_hint_for(
                    pre.herb, pre.feature_label_folder, scale_status),
                "cluster_hint": cluster_hint,
                "prelabel_summary": json.dumps(
                    [{"label": b.label, "conf": b.confidence,
                      "morphology": b.morphology}
                     for b in objs], ensure_ascii=False),
            },
            "predictions": [{"model_version": pre.model or "vlm", "score": 0.8,
                             "result": results}],
        })
    return tasks, qc_rows


def ensure_view_storage(cfg, pid):
    """把 tif 代理目录（ls_view）注册为项目本地存储。

    tif 任务的 image 指向 `ls_view/<...>.jpg` 代理，而 LS 的 `/data/local-files/` 端点要求
    请求文件的目录前缀命中已注册的 LocalFilesImportStorage —— 只注册 dataset 两个根不够，
    否则这批 84 张 tif 任务会全 404（2026-09-24 实测）。重复注册被 LS 拒绝，静默跳过。
    """
    try:
        ls_api(cfg, "POST", "/api/storages/localfiles",
               json={"project": pid, "path": VIEW_ROOT, "title": "ls_view(tif代理)",
                     "use_blob_urls": True})
        print(f"已注册 ls_view 本地存储（项目 {pid}）")
    except SystemExit:
        print("[提示] ls_view 存储可能已注册，继续")


def write_qc(qc_rows, out_path):
    """单目标 QC 报告落 CSV；返回框数≠1 的行（供控制台告警）。"""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["image", "folder", "herb", "cluster_boxes", "flagged"])
        w.writeheader()
        w.writerows(qc_rows)
    return [r for r in qc_rows if r["flagged"] == "是"]


def resolve_label_config(args, labels):
    """选标注界面 xml。

    优先级：① 显式 `--label-config`（要留档/要定制时用）；
            ② `--labels` **恰为一类** → 用单类模板内存渲染（`lib/labeling_config.render_single`），
               于是后续每一类都不必再手抄一份近乎相同的 xml；
            ③ 其余（多类或不限类）→ 默认 17 类界面 `labeling_config.xml`。
    """
    if args.label_config:
        with open(args.label_config, encoding="utf-8") as f:
            return f.read()
    if len(labels) == 1:
        from labeling_config import render_single
        return render_single(labels[0], hints=args.label_hint)
    with open(LABEL_CONFIG_PATH, encoding="utf-8") as f:
        return f.read()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-id", type=int, default=0, help="追加导入到已有项目")
    ap.add_argument("--dirs", default="",
                    help="逗号分隔目录名：只导这些目录的图（默认空=全量，旧行为）")
    ap.add_argument("--labels", default="",
                    help="逗号分隔类别：只保留这些框（默认空=全类，旧行为）；本图无此类框时仍导入")
    ap.add_argument("--title", default="", help="新项目标题（默认取 vlm_config 的 project_title）")
    ap.add_argument("--label-config", default="",
                    help="标注界面 xml 路径（默认：单类时由模板渲染，多类时用 labeling_config.xml）")
    ap.add_argument("--label-hint", action="append", default=[],
                    help="单类模板的附加页头提示，可重复（如针晶的「整束一框」口径）")
    ap.add_argument("--single-target-dirs", default="",
                    help="逗号分隔：这些目录每图预期仅 1 个目标，用于 QC 与页头提示")
    ap.add_argument("--out-qc", default=DEFAULT_QC_PATH, help="单目标 QC 报告 CSV 落点")
    ap.add_argument("--dry-run", action="store_true", help="只组装+统计+写 QC，不建项目不导入")
    args = ap.parse_args()

    def _split(s):
        return [x.strip() for x in s.split(",") if x.strip()]

    ensure_dirs()
    cfg = load_config()
    labels = _split(args.labels)
    tasks, qc_rows = build_tasks(_split(args.dirs), labels,
                                 _split(args.single_target_dirs))
    n_box = sum(len((t["predictions"] or [{}])[0].get("result", [])) for t in tasks)

    if qc_rows:
        bad = write_qc(qc_rows, args.out_qc)
        print(f"单目标 QC：{len(qc_rows)} 张，其中框数≠1 的 {len(bad)} 张 → {args.out_qc}")
        for r in bad:
            print(f"  [QC] {r['image']} 簇晶框数={r['cluster_boxes']}")

    print(f"组装 {len(tasks)} 个任务（AI 预标注框 {n_box} 个）")
    if args.dry_run:
        print("[dry-run] 不建项目、不导入。")
        return

    title = args.title or cfg["labelstudio"]["project_title"]
    if args.project_id:
        pid = args.project_id
        print(f"追加导入到项目 {pid}")
    else:
        r = ls_api(cfg, "POST", "/api/projects", json={
            "title": title,
            "label_config": resolve_label_config(args, labels),
            "expert_instruction": "框画法硬条款：紧贴外接矩形(±2%)，禁止切断实体；边界发虚框到清晰段；"
                                  "重叠目标各自独立成框；详见 docs/标注方案与最小测试报告.md §4.5",
        })
        pid = r.json()["id"]
        print(f"已创建项目 id={pid} {title}")

    ensure_local_storage(cfg, pid)
    ensure_view_storage(cfg, pid)
    n = import_in_chunks(cfg, pid, tasks)
    print(f"\n导入完成: {n} 个任务")
    print(f"打开项目: {cfg['labelstudio']['url']}/projects/{pid}/data")


if __name__ == "__main__":
    main()
