from __future__ import annotations

import torch
from torch import nn
from transformers import PreTrainedModel

from .backbone import build_vision_encoder, freeze_module
from .blip2 import load_model_and_preprocess
from .configuration import CellRepresentationConfig


class CellRepresentationModel(PreTrainedModel):
    """BLIP-2 Phase-I representation model with the PBS vision backbone."""

    config_class = CellRepresentationConfig
    main_input_name = "images"

    def __init__(self, config):
        super().__init__(config)
        visual_encoder = build_vision_encoder(config.vision)
        if config.freeze_vision:
            freeze_module(visual_encoder)

        qformer, _, text_processors = load_model_and_preprocess(
            "blip2_qformer", config.blip2_model_type, is_eval=False
        )
        qformer.query_tokens = nn.Parameter(qformer.query_tokens[:, :config.num_query_tokens].clone())
        qformer_width = qformer.visual_encoder.num_features
        if config.vision.embed_dim != qformer_width:
            projection = nn.Linear(config.vision.embed_dim, qformer_width, bias=False)
            nn.init.zeros_(projection.weight)
            diagonal = min(config.vision.embed_dim, qformer_width)
            projection.weight.data[:diagonal, :diagonal] = torch.eye(diagonal)
            visual_encoder = nn.Sequential(visual_encoder, projection)
            visual_encoder.transform = visual_encoder[0].transform
        qformer.visual_encoder = visual_encoder
        qformer.ln_vision = nn.LayerNorm(qformer_width)
        self.qformer = qformer
        self.text_processors = text_processors
        self.transform = visual_encoder.transform
        self.hidden_size = qformer.Qformer.config.hidden_size

    def forward(self, images, descriptions, **_):
        text = [self.text_processors["train"](description) for description in descriptions]
        output = self.qformer({"image": images, "text_input": text})
        loss = (
            self.config.itm_loss_weight * output.loss_itm
            + self.config.itc_loss_weight * output.loss_itc
            + self.config.lm_loss_weight * output.loss_lm
        )
        return {"loss": loss, "loss_itm": output.loss_itm,
                "loss_itc": output.loss_itc, "loss_lm": output.loss_lm}

    def encode(self, images):
        hidden, _ = self.qformer.forward_image(images)
        return hidden
