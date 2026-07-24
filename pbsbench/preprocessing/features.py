from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from pbsbench.models import PatchEncoder
from pbsbench.training.cli import (
    apply_overrides,
    checkpoint_state,
    load_config,
    representation_config,
    resolve_device,
)


class PatchDataset(Dataset):
    def __init__(self, paths, transform):
        self.paths = list(paths)
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image:
            return self.transform(image.convert("RGB"))


def extract_features_main(argv=None):
    parser = argparse.ArgumentParser(
        description="Encode tiled WSI patches with the Phase-2 patch Perceiver"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--patches", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--set", action="append", default=[], dest="overrides")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    apply_overrides(config, args.overrides)
    if config["stage"] != "cell_patch_alignment":
        raise ValueError("--config must describe the cell_patch_alignment stage")
    model_cfg = config["model"]
    representation = representation_config(model_cfg)
    encoder = PatchEncoder(
        representation.vision,
        hidden_size=model_cfg.get("patch_hidden_size", 768),
        num_query_tokens=model_cfg.get("num_query_tokens", 32),
        depth=model_cfg.get("patch_depth", 2),
        heads=model_cfg.get("patch_heads", 8),
    )
    state = {
        key.removeprefix("patch_encoder."): value
        for key, value in checkpoint_state(args.checkpoint).items()
        if key.startswith("patch_encoder.")
    }
    if not state:
        raise RuntimeError("checkpoint contains no patch_encoder parameters")
    _, unexpected = encoder.load_state_dict(state, strict=False)
    if unexpected:
        raise RuntimeError(f"unexpected patch checkpoint keys: {unexpected[:5]}")
    device = resolve_device(args.device)
    encoder.to(device).eval()
    args.output.mkdir(parents=True, exist_ok=True)

    slide_directories = sorted(
        path for path in args.patches.rglob("*")
        if path.is_dir() and any(path.glob("*.png"))
    )
    if not slide_directories:
        raise FileNotFoundError(f"no slide directories containing PNG patches under {args.patches}")
    for directory in slide_directories:
        paths = sorted(directory.glob("*.png"))
        loader = DataLoader(
            PatchDataset(paths, encoder.transform),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
        )
        features = []
        with torch.inference_mode():
            for images in loader:
                features.append(encoder.encode(images.to(device)).cpu())
        relative = directory.relative_to(args.patches)
        target = (args.output / relative).with_suffix(".pt")
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "features": torch.cat(features),
                "patches": [str(path.relative_to(directory)) for path in paths],
            },
            target,
        )
        print(f"{relative}: {len(paths)} patches -> {target}")


if __name__ == "__main__":
    extract_features_main()
