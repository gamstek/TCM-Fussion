# -*- coding: utf-8 -*-
"""加载 Stage2 特征分类权重并复用原训练的预处理口径。

读取权重中的模型、类别顺序和输入尺寸。可结合元信息选择 center、squash 或 letterbox 变换。

主要定义：Letterbox、build_transforms、load_model 和 classify_arrays。
关联：融合 Stage2 worker 和原检测入口的单实例分类使用这些接口。
"""
import json
import os

import paths

_MODEL_CACHE = {}
DEFAULT_CLS_ROOT = paths.CLS_TRAIN_V1
CLASSES_NAME = "classes.txt"


class Letterbox:
    """等比缩放到长边 = size，居中填到 size×size（背景灰 128，避免黑边被当成结构）。"""

    def __init__(self, size, fill=128):
        self.size = int(size)
        self.fill = int(fill)

    def __call__(self, img):
        from PIL import Image

        w, h = img.size
        s = min(self.size / max(w, 1), self.size / max(h, 1))
        nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
        canvas = Image.new("RGB", (self.size, self.size), (self.fill,) * 3)
        canvas.paste(img.resize((nw, nh), Image.BILINEAR),
                     ((self.size - nw) // 2, (self.size - nh) // 2))
        return canvas


def build_transforms(imgsz, mode="center"):
    """训练/评估变换（三种输入口径；`train_cls --input-mode` 与推理共用）。

    - `center`：短边缩放到 imgsz+32 后中心裁 imgsz。**会截断长条目标**——768×140 的
      纤维 crop 只剩约 16% 的长度，故只作基线口径保留。
    - `squash`：整图直接缩放到 imgsz×imgsz（保留全物体，代价是长宽比被压扁）。
    - `letterbox`：等比缩放 + 居中填充（保留全物体与真实长宽比，代价是有效像素变少）。
    """
    from torchvision import transforms

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    jitter = [transforms.ColorJitter(0.2, 0.2, 0.2)]
    flips = [transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
             transforms.RandomRotation(15, fill=128)]
    if mode == "letterbox":
        train_ops = [Letterbox(imgsz)] + flips + jitter
        eval_ops = [Letterbox(imgsz)]
    else:
        head = ([transforms.Resize((imgsz, imgsz))] if mode == "squash"
                else [transforms.RandomResizedCrop(imgsz, scale=(0.65, 1.0),
                                                   ratio=(0.7, 1.4))])
        train_ops = head + flips + jitter
        eval_ops = ([transforms.Resize((imgsz, imgsz))] if mode == "squash"
                    else [transforms.Resize(imgsz + 32), transforms.CenterCrop(imgsz)])
    train_tf = transforms.Compose(train_ops + [transforms.ToTensor(),
                                               transforms.Normalize(mean, std)])
    eval_tf = transforms.Compose(eval_ops + [transforms.ToTensor(),
                                             transforms.Normalize(mean, std)])
    return train_tf, eval_tf, mean, std


def load_class_names(cls_root=None):
    """`classes.txt` → {`<train_id>_<feature_id>`: 中文名}；缺文件时返回空表（用目录名兜底）。"""
    path = os.path.join(cls_root or DEFAULT_CLS_ROOT, CLASSES_NAME)
    names = {}
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 3:
                    names["{}_{}".format(parts[0], parts[1])] = parts[2]
    except OSError:
        return {}
    return names


def _resolve_device(device):
    import torch
    if device in (None, "", "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if str(device).isdigit():
        device = "cuda:" + str(device)
    return torch.device(device)


def load_model(weights, device="auto", cls_root=None):
    """按权重路径惰性加载并缓存 → `(model, meta, torch.device)`。

    `meta` 含 arch/imgsz/classes（目录名）/names（中文名）；权重缺失直接报错，
    不做“猜一个默认权重”的静默降级。
    """
    import torch
    from torch import nn
    from torchvision import models

    path = weights if os.path.isabs(str(weights)) else os.path.join(
        paths.PROJECT_ROOT, str(weights))
    if not os.path.isfile(path):
        raise FileNotFoundError("分类器权重不存在：{}".format(path))
    dev = _resolve_device(device)
    key = (os.path.abspath(path), str(dev))
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    arch, imgsz, classes = ckpt.get("arch"), int(ckpt.get("imgsz", 224)), ckpt.get("classes") or []
    if not arch or not classes:
        raise ValueError("分类器权重缺 arch/classes 字段：{}".format(path))
    model = getattr(models, arch)(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(ckpt["state_dict"])
    model.to(dev).eval()
    names = load_class_names(cls_root)
    input_mode = ckpt.get("input_mode")
    if not input_mode:
        # 旧 checkpoint 没存口径：回落到同目录的 model_meta.json（train_cls 总会写）。
        meta_path = os.path.join(os.path.dirname(path), "model_meta.json")
        try:
            with open(meta_path, encoding="utf-8") as f:
                input_mode = json.load(f).get("input_mode")
        except (OSError, ValueError):
            input_mode = None
    meta = {"arch": arch, "imgsz": imgsz, "classes": classes,
            "input_mode": input_mode or "center",
            "names": {c: names.get(c, c.split("_", 1)[-1]) for c in classes},
            "path": os.path.abspath(path)}
    _MODEL_CACHE[key] = (model, meta, dev)
    return model, meta, dev


def classify_arrays(crops, weights, device="auto", cls_root=None, batch=32):
    """`[BGR ndarray]` → 逐张 `{label, class_dir, prob, second, probs}`（顺序同输入）。

    `label` 是中文类别名（与 `VALID_LABELS` 同一套），`class_dir` 是训练目录名，
    `prob` 为 top-1 softmax。图像不足 1 张时返回空列表。
    """
    import numpy as np
    import torch

    if not crops:
        return []
    model, meta, dev = load_model(weights, device=device, cls_root=cls_root)
    _, eval_tf, _, _ = build_transforms(meta["imgsz"], meta.get("input_mode", "letterbox"))
    out = []
    with torch.no_grad():
        for i in range(0, len(crops), batch):
            chunk = crops[i:i + batch]
            tensors = []
            for crop in chunk:
                if crop is None or crop.size == 0:
                    raise ValueError("classify_arrays 收到空 crop")
                rgb = np.ascontiguousarray(crop[:, :, ::-1])     # cv2 BGR → RGB
                tensors.append(eval_tf(_to_pil(rgb)))
            x = torch.stack(tensors).to(dev)
            probs = torch.softmax(model(x).float(), dim=1).cpu().numpy()
            for row in probs:
                order = np.argsort(-row)[:2]
                top = meta["classes"][int(order[0])]
                second = meta["classes"][int(order[1])] if len(order) > 1 else ""
                out.append({
                    "label": meta["names"].get(top, top),
                    "class_dir": top,
                    "prob": round(float(row[order[0]]), 4),
                    "second": meta["names"].get(second, second),
                    "probs": {meta["names"].get(meta["classes"][int(k)],
                                                 meta["classes"][int(k)]): round(float(row[k]), 4)
                              for k in order},
                })
    return out


def _to_pil(rgb):
    from PIL import Image
    return Image.fromarray(rgb)
