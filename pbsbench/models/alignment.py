from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PreTrainedModel

from .backbone import freeze_module
from .components import PatchEncoder
from .configuration import CellPatchAlignmentConfig
from .representation import CellRepresentationModel


def global_alignment_loss(cell_tokens, patch_tokens):
    cell = F.normalize(cell_tokens.mean(dim=1), dim=-1)
    patch = F.normalize(patch_tokens.mean(dim=1), dim=-1)
    return 1 - (cell * patch).sum(dim=-1).mean()


def local_alignment_loss(cell_tokens, patch_tokens):
    cell = F.normalize(cell_tokens, dim=-1)
    patch = F.normalize(patch_tokens, dim=-1)
    length = min(cell.shape[1], patch.shape[1])
    logits = torch.bmm(cell[:, :length], patch[:, :length].transpose(1, 2))
    labels = torch.arange(length, device=logits.device).expand(logits.shape[0], -1)
    return F.cross_entropy(logits.reshape(-1, length), labels.reshape(-1))


class CellPatchAlignmentModel(PreTrainedModel):
    config_class = CellPatchAlignmentConfig
    main_input_name = "images"

    def __init__(self, config):
        super().__init__(config)
        self.cell_encoder = CellRepresentationModel(config.cell)
        self.patch_encoder = PatchEncoder(
            config.patch_vision,
            hidden_size=config.patch_hidden_size,
            num_query_tokens=config.patch_num_latents,
            depth=config.patch_depth,
            heads=config.patch_heads,
        )
        if config.freeze_cell_encoder:
            freeze_module(self.cell_encoder)
        self.cell_projection = nn.Linear(self.cell_encoder.hidden_size, config.projection_dim)
        self.patch_projection = nn.Linear(self.patch_encoder.hidden_size, config.projection_dim)

    def forward(self, images, patches, descriptions=None, **_):
        with torch.no_grad() if self.config.freeze_cell_encoder else torch.enable_grad():
            cell_tokens = self.cell_encoder.encode(images)
        patch_tokens = self.patch_encoder.encode(patches)
        cell = self.cell_projection(cell_tokens.detach())
        patch = self.patch_projection(patch_tokens)
        loss_global = global_alignment_loss(cell, patch)
        loss_local = local_alignment_loss(cell, patch)
        loss = self.config.global_loss_weight * loss_global + self.config.local_loss_weight * loss_local
        return {"loss": loss, "loss_global": loss_global, "loss_local": loss_local}
