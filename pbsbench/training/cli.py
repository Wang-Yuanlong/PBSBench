from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml
from safetensors.torch import load_file, save_file
from transformers import Trainer, TrainingArguments

from pbsbench.data import CaptionCollator, CaptionDataset, CellPatchDataset, QADataset, QwenQACollator, SlideQADataset
from pbsbench.models import (
    CellPatchAlignmentConfig, CellPatchAlignmentModel, CellRepresentationConfig,
    CellRepresentationModel, SlideEncoder, VisionConfig, build_qwen_adapter,
)


def load_config(path):
    with Path(path).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def apply_overrides(config, overrides):
    for override in overrides:
        key, value = override.split("=", 1)
        target = config
        parts = key.split(".")
        for part in parts[:-1]:
            target = target[part]
        target[parts[-1]] = yaml.safe_load(value)


def representation_config(model, identity=False, embed_dim=None):
    vision = VisionConfig(
        model_name=model.get("vision_model", "dinobloom_vitl14"),
        weights_path=model.get("vision_weights"),
        embed_dim=embed_dim or model.get("vision_embed_dim", 1024),
        pretrained=not identity,
        identity=identity,
    )
    return CellRepresentationConfig(
        vision=vision,
        blip2_model_type=model.get("blip2_model_type", "pretrain_vitL"),
        num_query_tokens=model.get("num_query_tokens", 32),
        freeze_vision=model.get("freeze_vision", True),
    )


def checkpoint_state(path):
    path = Path(path)
    candidate = path / "model.safetensors" if path.is_dir() else path
    if candidate.suffix == ".safetensors":
        return load_file(candidate)
    if path.is_dir():
        candidate = path / "pytorch_model.bin"
    return torch.load(candidate, map_location="cpu", weights_only=True)


def save_trainable_checkpoint(model, output_dir):
    """Save only paper-trainable parameters, never the frozen Qwen/DinoBloom weights."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    state = {
        name: parameter.detach().cpu().contiguous().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    if not state:
        raise RuntimeError("model has no trainable parameters")
    save_file(state, output / "model.safetensors")
    if hasattr(model, "config"):
        model.config.save_pretrained(output)


def load_representation(path, config, prefix=""):
    model = CellRepresentationModel(config)
    state = checkpoint_state(path)
    if prefix:
        state = {key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)}
    missing, unexpected = model.load_state_dict(state, strict=False)
    allowed = ("qformer.visual_encoder",)
    if any(not key.startswith(allowed) for key in unexpected):
        raise RuntimeError(f"unexpected checkpoint keys: {unexpected[:5]}")
    return model


def build_stage(config):
    stage, model_cfg, data_cfg = config["stage"], config["model"], config["data"]
    rep_cfg = representation_config(model_cfg)
    if stage == "cell_representation":
        model = CellRepresentationModel(rep_cfg)
        dataset = CaptionDataset(data_cfg["annotations"], data_cfg["image_root"], model.transform)
        return model, dataset, CaptionCollator()
    if stage == "cell_patch_alignment":
        align_cfg = CellPatchAlignmentConfig(
            cell=rep_cfg,
            patch_vision=rep_cfg.vision,
            patch_hidden_size=model_cfg.get("patch_hidden_size", 768),
            patch_num_latents=model_cfg.get("num_query_tokens", 32),
            patch_depth=model_cfg.get("patch_depth", 2),
            patch_heads=model_cfg.get("patch_heads", 8),
            projection_dim=model_cfg.get("projection_dim", 256),
            freeze_cell_encoder=model_cfg.get("freeze_cell_encoder", True),
            global_loss_weight=model_cfg.get("global_loss_weight", 1.0),
            local_loss_weight=model_cfg.get("local_loss_weight", 0.5),
        )
        model = CellPatchAlignmentModel(align_cfg)
        model.cell_encoder = load_representation(model_cfg["cell_checkpoint"], rep_cfg)
        for parameter in model.cell_encoder.parameters():
            parameter.requires_grad = False
        dataset = CellPatchDataset(
            data_cfg["annotations"], data_cfg["image_root"], model.patch_encoder.transform
        )
        return model, dataset, CaptionCollator()
    if stage == "cell_qa":
        model, processor = build_qwen_adapter(
            rep_cfg, model_cfg["language_model"], model_cfg.get("cell_checkpoint"),
            model_cfg.get("load_in_4bit", False))
        dataset = QADataset(data_cfg["annotations"], data_cfg["image_root"], model.model.visual.encoder.transform)
        return model, dataset, QwenQACollator(processor, model_cfg.get("num_query_tokens", 32))
    if stage == "slide_qa":
        encoder = SlideEncoder(model_cfg.get("feature_dim", 768), num_query_tokens=model_cfg.get("num_query_tokens", 32))
        model, processor = build_qwen_adapter(
            rep_cfg, model_cfg["language_model"], None, model_cfg.get("load_in_4bit", False),
            encoder=encoder)
        dataset = SlideQADataset(data_cfg["annotations"], data_cfg["feature_root"],
                                 data_cfg.get("max_patches", 2048))
        return model, dataset, QwenQACollator(processor, model_cfg.get("num_query_tokens", 32), slide=True)
    raise ValueError(f"unknown stage: {stage}")


def train_main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", action="append", default=[], dest="overrides")
    parser.add_argument("--max-steps", type=int)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    apply_overrides(config, args.overrides)
    model, dataset, collator = build_stage(config)
    training = config["training"]
    data = config["data"]
    eval_dataset = None
    if config["stage"] == "cell_qa" and data.get("val_annotations"):
        eval_dataset = QADataset(
            data["val_annotations"],
            data["image_root"],
            model.model.visual.encoder.transform,
        )
    elif config["stage"] == "slide_qa" and data.get("val_annotations"):
        eval_dataset = SlideQADataset(
            data["val_annotations"],
            data["feature_root"],
            data.get("max_patches", 2048),
        )
    arguments = TrainingArguments(
        output_dir=training["output_dir"],
        num_train_epochs=training["epochs"],
        max_steps=args.max_steps or -1,
        per_device_train_batch_size=training["batch_size"],
        per_device_eval_batch_size=training.get(
            "eval_batch_size", training["batch_size"]
        ),
        learning_rate=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
        warmup_ratio=float(training.get("warmup_ratio", 0.1)),
        lr_scheduler_type=training.get("scheduler", "cosine"),
        seed=training.get("seed", 42),
        remove_unused_columns=False,
        report_to=[],
        save_strategy="no",
        logging_strategy="no",
        eval_strategy="epoch" if eval_dataset is not None else "no",
    )
    trainer = Trainer(
        model=model,
        args=arguments,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
    )
    trainer.train()
    save_trainable_checkpoint(model, training["output_dir"])


def infer_main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--set", action="append", default=[], dest="overrides")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    apply_overrides(config, args.overrides)
    model, processor = _build_inference_model(config)
    model.load_state_dict(checkpoint_state(args.checkpoint), strict=False)
    device = resolve_device(args.device)
    if not getattr(model, "is_loaded_in_4bit", False):
        model.to(device)
    model.eval()
    transform = (
        None if config["stage"] == "slide_qa"
        else model.model.visual.encoder.transform
    )
    record = _inference_record(config["stage"], args.input, args.question, transform)
    collator = QwenQACollator(processor, config["model"].get("num_query_tokens", 32),
                              slide=config["stage"] == "slide_qa", include_answers=False)
    batch = collator([{**record, "answer": ""}])
    device = next(model.parameters()).device
    batch = {key: value.to(device) for key, value in batch.items()}
    with torch.inference_mode():
        tokens = model.generate(**batch, max_new_tokens=args.max_new_tokens, do_sample=False)
    generated = tokens[:, batch["input_ids"].shape[1]:]
    print(processor.batch_decode(generated, skip_special_tokens=True)[0].strip())


def _build_inference_model(config):
    model_cfg = config["model"]
    if config["stage"] == "cell_qa":
        rep_cfg = representation_config(model_cfg)
        return build_qwen_adapter(rep_cfg, model_cfg["language_model"],
                                  model_cfg.get("cell_checkpoint"),
                                  model_cfg.get("load_in_4bit", False))
    if config["stage"] == "slide_qa":
        encoder = SlideEncoder(model_cfg.get("feature_dim", 768), num_query_tokens=model_cfg.get("num_query_tokens", 32))
        return build_qwen_adapter(
            None, model_cfg["language_model"],
            load_in_4bit=model_cfg.get("load_in_4bit", False), encoder=encoder
        )
    raise ValueError("inference is supported for cell_qa and slide_qa")


def _inference_record(stage, path, question, transform=None):
    if stage == "slide_qa":
        value = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(value, dict):
            value = value["features"]
    else:
        from PIL import Image
        from pbsbench.data.datasets import default_transform
        value = (transform or default_transform())(Image.open(path).convert("RGB"))
    return {"image": value, "question": question, "question_type": "open"}


def resolve_device(value):
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


if __name__ == "__main__":
    train_main()
