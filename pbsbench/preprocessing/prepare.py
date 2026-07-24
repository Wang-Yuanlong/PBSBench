from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageEnhance


_IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}
_ROBOFLOW_EXPORT_SUFFIX = re.compile(
    r"_(?:bmp|jpe?g|png|tiff?)\.rf\.[^.]+$", re.IGNORECASE
)


def _provider_file_key(path):
    """Return a format-independent source-image key for provider exports."""
    return _ROBOFLOW_EXPORT_SUFFIX.sub("", Path(path).stem).casefold()


class ExternalFileIndex:
    """Resolve provider-native locators below an arbitrarily wrapped extraction root."""

    def __init__(self, root):
        self.root = Path(root)
        self.by_name = defaultdict(list)
        self.by_provider_key = defaultdict(list)
        for path in self.root.rglob("*"):
            if path.is_file() and path.suffix.casefold() in _IMAGE_SUFFIXES:
                self.by_name[path.name.casefold()].append(path)
                self.by_provider_key[_provider_file_key(path)].append(path)

    @staticmethod
    def _unique(candidates, locator):
        candidates = list(dict.fromkeys(candidates))
        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) > 1:
            preview = ", ".join(str(path) for path in candidates[:3])
            raise RuntimeError(f"ambiguous source locator {locator!r}: {preview}")
        return None

    def resolve(self, locator):
        locator = Path(locator)
        direct = self.root / locator
        if direct.is_file():
            return direct

        relative = locator.as_posix().casefold()
        same_name = self.by_name.get(locator.name.casefold(), [])
        suffix_match = [
            path
            for path in same_name
            if path.relative_to(self.root).as_posix().casefold().endswith(relative)
        ]
        resolved = self._unique(suffix_match, locator)
        if resolved is not None:
            return resolved

        resolved = self._unique(same_name, locator)
        if resolved is not None:
            return resolved

        resolved = self._unique(
            self.by_provider_key.get(_provider_file_key(locator), []), locator
        )
        if resolved is not None:
            return resolved
        raise FileNotFoundError(f"could not resolve {locator!r} below {self.root}")


def read_records(paths):
    records = {}
    for path in paths:
        with Path(path).open(encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                records.setdefault(record["image_id"], record)
    return list(records.values())


def parse_sources(values):
    sources = {}
    for value in values:
        name, separator, path = value.partition("=")
        if not separator or not name or not path:
            raise ValueError(f"expected DATASET=/path, got {value!r}")
        sources[name] = Path(path).expanduser().resolve()
    return sources


def _padded_crop(image, bbox, context=0.4, target_size=224):
    left, top, right, bottom = (bbox[key] for key in ("left", "top", "right", "bottom"))
    width, height = right - left, bottom - top
    difference = abs(width - height)
    if width > height:
        top -= difference // 2
        bottom += difference - difference // 2
        side = width
    else:
        left -= difference // 2
        right += difference - difference // 2
        side = height
    expansion = int(context * side)
    left, top, right, bottom = left - expansion, top - expansion, right + expansion, bottom + expansion
    pad = (max(0, -left), max(0, -top), max(0, right - image.width), max(0, bottom - image.height))
    if any(pad):
        canvas = Image.new("RGB", (image.width + pad[0] + pad[2], image.height + pad[1] + pad[3]), "white")
        canvas.paste(image, (pad[0], pad[1]))
        image = canvas
        left, right = left + pad[0], right + pad[0]
        top, bottom = top + pad[1], bottom + pad[1]
    return image.crop((left, top, right, bottom)).resize((target_size, target_size), Image.Resampling.LANCZOS)


def _place_file(source, target, mode, overwrite):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        if not overwrite:
            return
        target.unlink()
    if mode == "copy":
        shutil.copy2(source, target)
    else:
        target.symlink_to(os.path.relpath(source, target.parent))


def prepare_sbiad440(records, root, output, mode, overwrite):
    import openslide

    cell_records = defaultdict(list)
    placed = 0
    for record in records:
        source = record["source"]
        if source["dataset"] != "S-BIAD440":
            continue
        slide_path = root / source["path"]
        if record["level"] == "slide":
            _place_file(slide_path, output / record["image_path"], mode, overwrite)
            placed += 1
        else:
            cell_records[slide_path].append(record)
    for slide_path, items in cell_records.items():
        with openslide.open_slide(str(slide_path)) as slide:
            for record in items:
                target = output / record["image_path"]
                if target.exists() and not overwrite:
                    placed += 1
                    continue
                source = record["source"]
                patch = slide.read_region(
                    (source["patch_x"], source["patch_y"]), 0,
                    (source["patch_size"], source["patch_size"]),
                ).convert("RGB")
                patch = ImageEnhance.Color(patch).enhance(1.3)
                target.parent.mkdir(parents=True, exist_ok=True)
                _padded_crop(patch, source["bbox"]).save(target)
                patch_target = output / record["patch_path"]
                if overwrite or not patch_target.exists():
                    patch_target.parent.mkdir(parents=True, exist_ok=True)
                    patch.save(patch_target)
                placed += 1
    return placed


def prepare_external(records, sources, output, mode, overwrite):
    placed = 0
    indexes = {
        dataset: ExternalFileIndex(root)
        for dataset, root in sources.items()
        if dataset != "S-BIAD440"
    }
    for record in records:
        source = record["source"]
        dataset = source["dataset"]
        if dataset == "S-BIAD440" or dataset not in sources:
            continue
        source_path = indexes[dataset].resolve(source["path"])
        _place_file(source_path, output / record["image_path"], mode, overwrite)
        placed += 1
    return placed


def prepare_main(argv=None):
    parser = argparse.ArgumentParser(description="Materialize images referenced by PBSInstr/PBSBench")
    parser.add_argument("--annotations", nargs="+", type=Path, required=True)
    parser.add_argument("--source", action="append", default=[], metavar="DATASET=/PATH", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--link-mode", choices=("symlink", "copy"), default="symlink")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    records, sources = read_records(args.annotations), parse_sources(args.source)
    missing = sorted({r["source"]["dataset"] for r in records} - sources.keys())
    if missing:
        raise ValueError(f"missing --source for: {', '.join(missing)}")
    placed = 0
    if "S-BIAD440" in sources:
        placed += prepare_sbiad440(records, sources["S-BIAD440"], args.output,
                                   args.link_mode, args.overwrite)
    placed += prepare_external(records, sources, args.output, args.link_mode, args.overwrite)
    print(f"materialized {placed} unique images under {args.output}")
