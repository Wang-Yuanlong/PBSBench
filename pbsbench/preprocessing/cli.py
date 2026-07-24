import argparse
from pathlib import Path

from .cells import crop_instances
from .tiling import load_qc_model, tile_slide


def tile_main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--slides", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--patch-size", type=int, default=512)
    parser.add_argument("--qc-model", type=Path)
    parser.add_argument("--qc-threshold", type=float, default=0.5)
    args = parser.parse_args(argv)
    model = load_qc_model(args.qc_model) if args.qc_model else None
    paths = [args.slides] if args.slides.is_file() else sorted(
        path for path in args.slides.iterdir() if path.suffix.lower() in {".svs", ".tif", ".tiff"})
    for path in paths:
        slide, count = tile_slide(path, args.output, args.patch_size, model, args.qc_threshold)
        print(f"{slide}: {count} retained patches")


def crop_main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--patches", type=Path, required=True)
    parser.add_argument("--masks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--context", type=float, default=0.4)
    args = parser.parse_args(argv)
    count = 0
    for mask in args.masks.rglob("*.npy"):
        relative = mask.relative_to(args.masks)
        image = (args.patches / relative).with_suffix(".png")
        if image.exists():
            count += len(crop_instances(image, mask, args.output / relative.with_suffix(""), args.context))
    print(f"saved {count} cell crops")
