import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def read_jsonl(path):
    records = []
    with Path(path).open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise ValueError(f"invalid JSON at {path}:{number}") from error
    return records


def default_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])


class QADataset(Dataset):
    def __init__(self, annotations, image_root, transform=None):
        self.records = read_jsonl(annotations)
        self.image_root = Path(image_root)
        self.transform = transform or default_transform()

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        image = Image.open(self.image_root / record["image_path"]).convert("RGB")
        return {**record, "image": self.transform(image)}


class CaptionDataset(QADataset):
    def __getitem__(self, index):
        item = super().__getitem__(index)
        return {"images": item["image"], "descriptions": item["caption"]}


class CellPatchDataset(QADataset):
    def __getitem__(self, index):
        item = super().__getitem__(index)
        patch = Image.open(self.image_root / item["patch_path"]).convert("RGB")
        return {"images": item["image"], "patches": self.transform(patch),
                "descriptions": item.get("caption", "")}


class SlideQADataset(Dataset):
    def __init__(self, annotations, feature_root, max_patches=2048):
        self.records = read_jsonl(annotations)
        self.feature_root = Path(feature_root)
        self.max_patches = max_patches

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        features = torch.load(self.feature_root / record["feature_path"],
                              map_location="cpu", weights_only=True)
        if isinstance(features, dict):
            features = features["features"]
        return {**record, "image": features[:self.max_patches]}
