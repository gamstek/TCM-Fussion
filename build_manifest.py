# -*- coding: utf-8 -*-
"""
build_manifest.py — 扫描两个图像根目录，生成标注底表 manifest.csv
用法: python build_manifest.py [--with-size]
输出: manifest.csv  （列见 HEADERS；与脚本同目录 = 项目根）
粒度解析规则（文件夹名 → 药材/特征）:
  - "药材"              -> granule=herb,      label=药材名
  - "药材+特征"          -> granule=feature,   herb=药材名, feature_label=特征名
  - 纯特征名(无药材前缀)  -> granule=feature_only
特征后缀词表来自 85味/特征全集 的命名习惯。
分组键 group_key 用于防泄漏划分（同一批号/流水号的图不跨 train/val）。
"""
import os, re, sys, csv, argparse, random
import paths

ROOTS = [os.path.join(paths.DATASET, folder) for folder in paths.IMAGE_SUBDIRS]
OUT = os.path.join(paths.PROJECT_ROOT, "manifest.csv")
HEADERS = ["image_path", "root", "folder_name", "granule", "herb", "feature_label",
           "ext", "width", "height", "group_key", "suggested_split", "needs_review"]

# 已知特征后缀（含别名/大小写变体），顺序无关，按最长匹配
FEATURE_SUFFIXES = [
    "草酸钙簇晶", "草酸钙针晶", "草酸钙方晶", "簇晶", "针晶", "方晶", "砂晶",
    "升华物", "淀粉粒", "导管", "石细胞", "木纤维", "晶纤维", "菊糖",
    "联结乳管", "外果皮细胞", "种皮表皮细胞", "种皮细胞", "菌丝",
]
HERB_ALIAS = {"荜芨": "荜茇"}

def parse_folder(name):
    """返回 (granule, herb, feature_label, needs_review)"""
    for suf in sorted(FEATURE_SUFFIXES, key=len, reverse=True):
        if name.endswith(suf) and len(name) > len(suf):
            herb = name[: -len(suf)]
            herb = HERB_ALIAS.get(herb, herb)
            return "feature", herb, suf, False
    if name.endswith("升华物"):  # 大黄升华物 已在上面覆盖，此为兜底
        return "feature", name[:-3], "升华物", False
    return "herb", HERB_ALIAS.get(name, name), "", False

def group_key(path, herb):
    """批号/流水号分组: B2016xxx -> B2016 ; 2023xxxxx.tif -> 2023 ; 纯数字序号 -> seq-文件夹"""
    stem = os.path.splitext(os.path.basename(path))[0]
    m = re.match(r"^([A-Za-z]+)(\d{4})", stem)
    if m:
        return f"{herb}-{m.group(1)}{m.group(2)}"
    m = re.match(r"^(\d{4})", stem)
    if m:
        return f"{herb}-{m.group(1)}"
    return f"{herb}-seq"

def image_size(path):
    try:
        from PIL import Image
        Image.LOAD_TRUNCATED_IMAGES = True
        with Image.open(path) as im:
            return im.size  # (w, h) 仅读文件头，很快
    except Exception:
        return ("", "")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-size", action="store_true", help="读取图像宽高（略慢）")
    args = ap.parse_args()

    rows, seen = [], 0
    for root in ROOTS:
        if not os.path.isdir(root):
            print(f"[warn] 根目录不存在: {root}")
            continue
        for folder in sorted(os.listdir(root)):
            fdir = os.path.join(root, folder)
            if not os.path.isdir(fdir):
                continue
            granule, herb, feat, _ = parse_folder(folder)
            for fn in sorted(os.listdir(fdir)):
                if not re.search(r"\.(jpg|jpeg|tif|tiff|png|bmp)$", fn, re.I):
                    continue
                p = os.path.join(fdir, fn)
                w = h = ""
                if args.with_size:
                    w, h = image_size(p)
                rows.append({
                    "image_path": p, "root": os.path.basename(root), "folder_name": folder,
                    "granule": granule, "herb": herb, "feature_label": feat,
                    "ext": os.path.splitext(fn)[1].lower().lstrip("."),
                    "width": w, "height": h,
                    "group_key": group_key(fn, herb),
                    "suggested_split": "", "needs_review": "",
                })
                seen += 1

    # 分层随机划分（按 group_key 整组划分，防同批泄漏）：train 0.8 / val 0.1 / test 0.1
    from collections import defaultdict
    groups = defaultdict(list)
    for r in rows:
        groups[(r["root"], r["folder_name"], r["group_key"])].append(r)
    rng = random.Random(42)
    for (root, folder), _ in [(k, v) for k, v in []]:  # noqa
        pass
    # 按文件夹分层：每个文件夹内的组独立洗牌后按比例分配
    by_folder = defaultdict(list)
    for key in groups:
        by_folder[(key[0], key[1])].append(key)
    for fkey, gkeys in by_folder.items():
        gkeys = sorted(gkeys)
        rng.shuffle(gkeys)
        n = len(gkeys)
        n_tr = max(1, round(n * 0.8)); n_va = max(1, round(n * 0.1)) if n >= 3 else (1 if n == 2 else 0)
        for i, gk in enumerate(gkeys):
            split = "train" if i < n_tr else ("val" if i < n_tr + n_va else "test")
            for r in groups[gk]:
                r["suggested_split"] = split
    # 复核标记：文件名含“复核”的天然是复核图
    for r in rows:
        if "复核" in r["image_path"]:
            r["needs_review"] = "already_reviewed_copy"

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=HEADERS)
        w.writeheader(); w.writerows(rows)

    # 汇总
    from collections import Counter
    c_g = Counter(r["granule"] for r in rows)
    c_f = Counter(r["root"] for r in rows)
    print(f"总计图像: {len(rows)}")
    print(f"按粒度: {dict(c_g)}")
    print(f"按根目录: {dict(c_f)}")
    folders = sorted({(r['root'], r['folder_name'], r['granule'], r['herb'], r['feature_label']) for r in rows})
    print(f"文件夹数: {len(folders)}")
    print(f"已写出: {OUT}")

if __name__ == "__main__":
    main()
