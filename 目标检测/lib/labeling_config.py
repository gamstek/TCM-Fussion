# -*- coding: utf-8 -*-
"""labeling_config.py — 单类检测标注界面（LS `label_config`）的模板渲染。

**为什么存在**：每一类检测训练集都要一份「只含该类一个 Label」的 LS 界面。手抄会堆出
17 份近乎一样的 xml，且改一处（如加判据行）要改 17 遍、必然漂移。这里以
`config/labeling_config_single.tmpl.xml` 为**唯一模板**按标签渲染；类别颜色从既有的
17 类 `config/labeling_config.xml` 里取，不再单独维护一份色表。

**放在 lib/ 而不是 cli/**：`cli/` 按定义是入口、不在 `.pth` 注入点内（不得被 import），
而本模块要被 `annot/import_to_labelstudio.py` 复用，故必须是具名库模块。
"""
import os
import re

from ls_common import DET_CONFIG

TEMPLATE_PATH = os.path.join(DET_CONFIG, "labeling_config_single.tmpl.xml")
MULTI_CLASS_PATH = os.path.join(DET_CONFIG, "labeling_config.xml")
DEFAULT_COLOR = "#e6194b"


def label_color(zh, multi_class_path=None):
    """从 17 类界面里取该标签的 `background`；查不到（如新类）返回默认色。"""
    try:
        with open(multi_class_path or MULTI_CLASS_PATH, encoding="utf-8") as f:
            m = re.search(r'<Label value="%s" background="(#[0-9a-fA-F]{6})"' % re.escape(zh),
                          f.read())
        return m.group(1) if m else DEFAULT_COLOR
    except OSError:
        return DEFAULT_COLOR


def render_single(zh, color=None, hints=(), template_path=None):
    """渲染「只含 zh 一个 Label」的 LS 界面，返回 XML 字符串。

    hints: 附加 `<Header>` 的正文（如针晶的「整束一框」口径），逐条渲染、可为空。
    模板占位符：`{{LABEL}}`（多处）、`{{COLOR}}`、`{{HINT_HEADERS}}`（整行，空则渲染为空串）。
    """
    with open(template_path or TEMPLATE_PATH, encoding="utf-8") as f:
        tpl = f.read()
    heads = "".join('  <Header value="%s" style="color:#b45309"/>\n' % h for h in hints if h)
    return (tpl.replace("{{HINT_HEADERS}}", heads)
               .replace("{{LABEL}}", zh)
               .replace("{{COLOR}}", color or label_color(zh)))
