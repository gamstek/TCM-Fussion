# -*- coding: utf-8 -*-
"""make_labeling_config.py — 按标签渲染「单类检测标注界面」（LS label_config）

模板唯一来源：`config/labeling_config_single.tmpl.xml`（渲染逻辑在 `lib/labeling_config.py`）。
本入口只做「取参 → 渲染 → 落盘/打印」，供人工审阅或需要一份可编辑 xml 时使用。

注：`annot/import_to_labelstudio.py` 在「--labels 恰为一类」时会**直接用同一模板内存渲染**，
不再需要 per-class 文件；本脚本是那条路之外的显式出口（要留档/要改一份时用）。

用法:
  python cli\\make_labeling_config.py --label 草酸钙针晶 --hint "成束针晶只框整束一个框"
  python cli\\make_labeling_config.py --label 草酸钙菊糖 --hint "一个团块一框" --print
  python cli\\make_labeling_config.py --label 草酸钙方晶 --out <自定义路径>
默认输出 `config/labeling_config_single_<feature_id>.xml`（找不到 feature_id 时用标签净化名）。
"""
import argparse
import os
import re
import xml.etree.ElementTree as ET

from labeling_config import label_color, render_single  # noqa: E402 共享层（lib）
from ls_common import DET_CONFIG, load_config  # noqa: E402


def slug_of(zh, cfg=None):
    """文件名片段：优先 yolo.classes 的 feature_id，否则把标签净化成 ASCII 标识。"""
    for c in (cfg or load_config())["yolo"]["classes"]:
        if c["zh"] == zh:
            return c["feature_id"]
    return re.sub(r"[^0-9A-Za-z_]+", "_", zh).strip("_") or "class"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True, help="唯一的目标类别（中文，须与检测类别名一致）")
    ap.add_argument("--hint", action="append", default=[],
                    help="附加页头提示，可重复（如针晶的「整束一框」口径）")
    ap.add_argument("--color", default="", help="标签底色，默认从 17 类界面里取该类原色")
    ap.add_argument("--out", default="", help="输出路径，默认 config/labeling_config_single_<feature_id>.xml")
    ap.add_argument("--print", dest="to_stdout", action="store_true", help="打到 stdout，不落盘")
    args = ap.parse_args()

    xml = render_single(args.label, color=args.color or None, hints=args.hint)
    try:                                    # 占位符漏替换/结构写错会在这里立刻暴露，不至于推上 LS 才炸
        ET.fromstring(xml)
    except ET.ParseError as e:
        raise SystemExit(f"渲染出的 XML 不是合法 XML：{e}")

    if args.to_stdout:
        print(xml)
        return
    out = args.out or os.path.join(
        DET_CONFIG, f"labeling_config_single_{slug_of(args.label)}.xml")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        f.write(xml)
    print(f"已写出: {out}")
    print(f"  类别: {args.label} | 底色: {args.color or label_color(args.label)} | 附加页头: {len(args.hint)} 条")


if __name__ == "__main__":
    main()
