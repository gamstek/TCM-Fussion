"""Run the three Plan A herb classifiers on a full microscope image.

Each checkpoint must match its metadata before strict loading. The worker emits
one complete softmax vector per model. The fusion process ranks their mean.
"""
import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torchvision.models import resnet18

from 目标检测.lib.cls_net import build_transforms


def predict(image_path, model_dirs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, transform, mean, std = build_transforms(448, "letterbox")
    with Image.open(image_path) as image:
        pixels = transform(image.convert("RGB")).unsqueeze(0).to(device)

    classes = None
    predictions = []
    with torch.inference_mode():
        for directory in model_dirs:
            weight = directory / "best.pt"
            with (directory / "model_meta.json").open(encoding="utf-8-sig") as stream:
                metadata = json.load(stream)
            labels = metadata["classes"]
            if (not isinstance(labels, list) or len(labels) < 8
                    or any(not isinstance(label, str) or not label.strip() for label in labels)
                    or len(set(labels)) != len(labels) or (classes is not None and labels != classes)):
                raise ValueError(f"{directory.name}: invalid or inconsistent class order")
            classes = labels
            expected = {"arch": "resnet18", "imgsz": 448, "input_mode": "letterbox",
                        "classes": classes, "mean": mean, "std": std}
            checkpoint = torch.load(weight, map_location="cpu", weights_only=True)
            if (metadata.get("checkpoint") != weight.name
                    or any(metadata.get(key) != value or checkpoint.get(key) != value
                           for key, value in expected.items())):
                raise ValueError(f"{directory.name}: checkpoint and metadata disagree with Plan A inputs")
            model = resnet18(weights=None)
            model.fc = nn.Linear(model.fc.in_features, len(classes))
            model.load_state_dict(checkpoint["state_dict"], strict=True)
            model.to(device).eval()
            probabilities = torch.softmax(model(pixels).float(), dim=1)[0].tolist()
            predictions.append({"model_id": directory.name, "probabilities": probabilities})
            del model, checkpoint
    return {"classes": classes, "model_predictions": predictions}


def main():
    parser = argparse.ArgumentParser(description="Plan A herb image ensemble")
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--model-dir", required=True, action="append", type=Path)
    args = parser.parse_args()
    directories = [directory.resolve() for directory in args.model_dir]
    if len(directories) != 3 or len(set(directories)) != 3:
        parser.error("provide exactly three distinct model directories")
    print(json.dumps(predict(args.image, directories), ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
