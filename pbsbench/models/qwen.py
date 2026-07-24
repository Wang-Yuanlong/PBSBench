from __future__ import annotations

import logging
from pathlib import Path

import torch
from safetensors.torch import load_file
from torch import nn
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

from .backbone import freeze_module
from .representation import CellRepresentationModel


def _load_cell_encoder(config, checkpoint):
    """Restore the trainable Phase-1 state without expecting a full HF checkpoint."""
    encoder = CellRepresentationModel(config)
    path = Path(checkpoint)
    candidate = path / "model.safetensors" if path.is_dir() else path
    if candidate.suffix == ".safetensors":
        state = load_file(candidate)
    else:
        if path.is_dir():
            candidate = path / "pytorch_model.bin"
        state = torch.load(candidate, map_location="cpu", weights_only=True)
    _, unexpected = encoder.load_state_dict(state, strict=False)
    if unexpected:
        raise RuntimeError(f"unexpected cell checkpoint keys: {unexpected[:5]}")
    return encoder


class QFormerVisualAdapter(nn.Module):
    """Expose 32 PBS visual tokens through Qwen2.5-VL's visual interface."""

    def __init__(self, encoder, output_dim):
        super().__init__()
        self.encoder = encoder
        self.projector = nn.Linear(encoder.hidden_size, output_dim)
        self.spatial_merge_size = 1

    @property
    def dtype(self):
        parameter = next(self.encoder.parameters(), None)
        return parameter.dtype if parameter is not None else self.projector.weight.dtype

    def forward(self, pixel_values, grid_thw=None, **_):
        tokens = self.encoder.encode(pixel_values)
        return self.projector(tokens).reshape(-1, self.projector.out_features)


def build_qwen_adapter(vision_config, language_model="Qwen/Qwen2.5-VL-7B-Instruct",
                       checkpoint=None, load_in_4bit=False, attention_implementation="sdpa", encoder=None):
    kwargs = {
        "attn_implementation": attention_implementation,
        "torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    }
    if load_in_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
        )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(language_model, **kwargs)
    processor_logger = logging.getLogger("transformers.video_processing_utils")
    processor_log_level = processor_logger.level
    processor_logger.setLevel(logging.ERROR)
    try:
        # Match the processor saved with Qwen2.5-VL explicitly.  This avoids
        # version-dependent fast/slow processor defaults in Transformers.
        processor = AutoProcessor.from_pretrained(language_model, use_fast=False)
    finally:
        processor_logger.setLevel(processor_log_level)
    model.loss_type = "ForCausalLM"
    # PBSBench inference is deterministic.  Qwen's published generation config
    # includes a sampling temperature, which Transformers warns is unused when
    # do_sample=False unless it is cleared explicitly.
    model.generation_config.temperature = None
    if encoder is None:
        encoder = (
            _load_cell_encoder(vision_config, checkpoint)
            if checkpoint
            else CellRepresentationModel(vision_config)
        )
    encoder_trainable = {
        name for name, parameter in encoder.named_parameters() if parameter.requires_grad
    }
    model.model.visual = QFormerVisualAdapter(encoder, model.config.vision_config.out_hidden_size)
    model.model.config.vision_config.spatial_merge_size = 1
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.model.visual.projector.parameters():
        parameter.requires_grad = True
    for name, parameter in model.model.visual.encoder.named_parameters():
        parameter.requires_grad = name in encoder_trainable
    if load_in_4bit:
        device = next(model.model.language_model.parameters()).device
        model.model.visual.to(device)
    freeze_module(model.model.language_model)
    freeze_module(model.lm_head)
    return model, processor
