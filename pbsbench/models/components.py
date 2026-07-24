import torch
from torch import nn

from .backbone import build_vision_encoder, freeze_module


class PerceiverBlock(nn.Module):
    def __init__(self, width, heads=8, dropout=0.0):
        super().__init__()
        self.query_norm = nn.LayerNorm(width)
        self.input_norm = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width * 4),
                                 nn.GELU(), nn.Linear(width * 4, width))

    def forward(self, latents, inputs, key_padding_mask=None):
        query = self.query_norm(latents)
        values = self.input_norm(inputs)
        attended, _ = self.attention(query, values, values,
                                     key_padding_mask=key_padding_mask, need_weights=False)
        latents = latents + attended
        return latents + self.ffn(latents)


class PerceiverResampler(nn.Module):
    """Distill variable-length visual tokens into fixed-count latent tokens."""

    def __init__(self, input_dim, hidden_size=768, num_latents=32, depth=2, heads=8):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, hidden_size)
        self.latents = nn.Parameter(torch.randn(1, num_latents, hidden_size) * 0.02)
        self.blocks = nn.ModuleList([PerceiverBlock(hidden_size, heads) for _ in range(depth)])
        self.output_norm = nn.LayerNorm(hidden_size)

    def forward(self, inputs, attention_mask=None):
        if inputs.ndim == 4:
            inputs = inputs.mean(dim=-2)
        inputs = self.input_projection(inputs)
        latents = self.latents.expand(inputs.shape[0], -1, -1)
        padding = attention_mask == 0 if attention_mask is not None else None
        for block in self.blocks:
            latents = block(latents, inputs, padding)
        return self.output_norm(latents)


class PatchEncoder(nn.Module):
    """Frozen patch backbone followed by the trainable Phase-2 Perceiver."""

    def __init__(self, vision_config, hidden_size=768, num_query_tokens=32,
                 depth=2, heads=8):
        super().__init__()
        self.visual_encoder = build_vision_encoder(vision_config)
        freeze_module(self.visual_encoder)
        self.resampler = PerceiverResampler(
            vision_config.embed_dim, hidden_size, num_query_tokens, depth, heads
        )
        self.hidden_size = hidden_size
        self.transform = self.visual_encoder.transform

    def encode(self, images):
        with torch.no_grad():
            tokens = self.visual_encoder(images)
        return self.resampler(tokens)

    def forward(self, images):
        return self.encode(images)


class SlideEncoder(nn.Module):
    """Paper slide-level Perceiver with the interface expected by the Qwen adapter."""

    def __init__(self, feature_dim, hidden_size=768, num_query_tokens=32):
        super().__init__()
        self.resampler = PerceiverResampler(feature_dim, hidden_size, num_query_tokens)
        self.hidden_size = hidden_size
        self.transform = nn.Identity()

    def encode(self, features):
        if features.ndim == 4:
            attention_mask = features.abs().sum(dim=(-1, -2)).ne(0)
        else:
            attention_mask = features.abs().sum(dim=-1).ne(0)
        return self.resampler(features, attention_mask)
