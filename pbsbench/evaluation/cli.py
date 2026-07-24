from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from pbsbench.data import QADataset, QwenQACollator, SlideQADataset
from pbsbench.evaluation.metrics import score_predictions
from pbsbench.training.cli import (
    _build_inference_model,
    apply_overrides,
    checkpoint_state,
    load_config,
    resolve_device,
)


def evaluate_main(argv=None):
    parser = argparse.ArgumentParser(
        description="Run PBS-VL over a PBSBench JSONL set and compute paper metrics"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--image-root")
    parser.add_argument("--feature-root")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--set", action="append", default=[], dest="overrides")
    parser.add_argument(
        "--semantic-model",
        help="Optional SentenceTransformers model ID/path for open-answer cosine similarity",
    )
    args = parser.parse_args(argv)

    config = load_config(args.config)
    apply_overrides(config, args.overrides)
    stage = config["stage"]
    if stage not in {"cell_qa", "slide_qa"}:
        raise ValueError("evaluation requires a cell_qa or slide_qa config")

    model, processor = _build_inference_model(config)
    _, unexpected = model.load_state_dict(checkpoint_state(args.checkpoint), strict=False)
    if unexpected:
        raise RuntimeError(f"unexpected checkpoint keys: {unexpected[:5]}")
    device = resolve_device(args.device)
    if not getattr(model, "is_loaded_in_4bit", False):
        model.to(device)
    model.eval()

    data = config["data"]
    if stage == "cell_qa":
        root = args.image_root or data["image_root"]
        dataset = QADataset(
            args.annotations, root, model.model.visual.encoder.transform
        )
    else:
        root = args.feature_root or data["feature_root"]
        dataset = SlideQADataset(
            args.annotations, root, data.get("max_patches", 2048)
        )
    limit = min(len(dataset), args.max_records or len(dataset))
    collator = QwenQACollator(
        processor,
        config["model"].get("num_query_tokens", 32),
        slide=stage == "slide_qa",
        include_answers=False,
    )
    results = []
    for start in range(0, limit, args.batch_size):
        items = [dataset[index] for index in range(start, min(start + args.batch_size, limit))]
        batch = collator(items)
        batch = {key: value.to(device) for key, value in batch.items()}
        with torch.inference_mode():
            tokens = model.generate(
                **batch, max_new_tokens=args.max_new_tokens, do_sample=False
            )
        generated = tokens[:, batch["input_ids"].shape[1]:]
        predictions = processor.batch_decode(generated, skip_special_tokens=True)
        for item, prediction in zip(items, predictions):
            results.append({
                key: item[key]
                for key in (
                    "id", "image_id", "level", "domain", "task",
                    "question_type", "question", "answer",
                )
                if key in item
            } | {
                "options": item.get("options"),
                "prediction": prediction.strip(),
            })

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
    metrics = score_predictions(results, args.semantic_model)
    metrics_path = output.with_suffix(output.suffix + ".metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"predictions": str(output), "metrics": str(metrics_path), **metrics}, indent=2))


if __name__ == "__main__":
    evaluate_main()
