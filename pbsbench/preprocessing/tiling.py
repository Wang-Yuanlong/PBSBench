import json
from pathlib import Path

import numpy as np
from PIL import Image


def iter_tiles(slide, patch_size=512):
    width, height = slide.dimensions
    for y in range(0, height - patch_size + 1, patch_size):
        for x in range(0, width - patch_size + 1, patch_size):
            yield x, y, np.asarray(slide.read_region((x, y), 0, (patch_size, patch_size)).convert("RGB"))


def load_qc_model(path):
    from tensorflow import keras
    return keras.models.load_model(path)


def qc_scores(images, model):
    batch = np.asarray(images, dtype=np.float32) / 255.0
    return np.asarray(model(batch, training=False)).reshape(-1)


def tile_slide(path, output, patch_size=512, qc_model=None, threshold=0.5, batch_size=64):
    import openslide

    output = Path(output)
    slide_id, target = Path(path).stem, output / Path(path).stem
    target.mkdir(parents=True, exist_ok=True)
    manifest, pending = [], []
    with openslide.open_slide(str(path)) as slide:
        for x, y, image in iter_tiles(slide, patch_size):
            pending.append((x, y, image))
            if len(pending) == batch_size:
                _write_batch(pending, target, manifest, qc_model, threshold)
                pending.clear()
    if pending:
        _write_batch(pending, target, manifest, qc_model, threshold)
    with (target / "manifest.jsonl").open("w", encoding="utf-8") as handle:
        for item in manifest:
            handle.write(json.dumps(item) + "\n")
    return slide_id, len(manifest)


def _write_batch(batch, target, manifest, model, threshold):
    scores = np.ones(len(batch)) if model is None else qc_scores([item[2] for item in batch], model)
    for (x, y, image), score in zip(batch, scores):
        if float(score) < threshold:
            continue
        name = f"{x}_{y}.png"
        Image.fromarray(image).save(target / name)
        manifest.append({"file": name, "x": x, "y": y, "qc_score": float(score)})
