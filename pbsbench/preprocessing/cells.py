from pathlib import Path

import numpy as np
from PIL import Image


def instance_bbox(mask, label):
    ys, xs = np.nonzero(mask == label)
    if not len(xs):
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def touches_border(box, width, height):
    left, top, right, bottom = box
    return left <= 0 or top <= 0 or right >= width or bottom >= height


def square_context_box(box, width, height, context=0.4):
    left, top, right, bottom = box
    side = max(right - left, bottom - top)
    cx, cy = (left + right) / 2, (top + bottom) / 2
    half = side * (0.5 + context)
    return (max(0, int(cx - half)), max(0, int(cy - half)),
            min(width, int(cx + half)), min(height, int(cy + half)))


def crop_instances(image_path, mask_path, output, context=0.4):
    image, mask = Image.open(image_path).convert("RGB"), np.load(mask_path)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    saved = []
    for label in np.unique(mask):
        if label == 0:
            continue
        box = instance_bbox(mask, label)
        if touches_border(box, image.width, image.height):
            continue
        crop = image.crop(square_context_box(box, image.width, image.height, context))
        path = output / f"{int(label)}.png"
        crop.save(path)
        saved.append(path)
    return saved
