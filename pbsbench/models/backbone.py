from __future__ import annotations

import warnings

import torch
from torch import nn
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision import transforms

from .configuration import VisionConfig


class IdentityTokenEncoder(nn.Module):
    """Pass precomputed patch/cell tokens to the slide resampler."""

    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.transform = nn.Identity()

    def forward(self, tokens):
        if tokens.ndim == 4:
            tokens = tokens.mean(dim=-2)
        if tokens.shape[-1] > self.embed_dim:
            raise ValueError(f"feature width {tokens.shape[-1]} exceeds {self.embed_dim}")
        return torch.nn.functional.pad(tokens, (0, self.embed_dim - tokens.shape[-1]))


class DinoBloomEncoder(nn.Module):
    """DINOv2 architecture loaded from the paper's DinoBloom teacher checkpoint."""

    def __init__(self, config: VisionConfig):
        super().__init__()
        architectures = {
            "dinobloom_vits14": "dinov2_vits14",
            "dinobloom_vitb14": "dinov2_vitb14",
            "dinobloom_vitl14": "dinov2_vitl14",
            "dinobloom_vitg14": "dinov2_vitg14",
        }
        if config.model_name not in architectures:
            raise ValueError(f"unsupported PBS vision model: {config.model_name}")
        if not config.weights_path:
            raise ValueError("vision.weights_path must point to the DinoBloom checkpoint")
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message=r"xFormers is available.*", category=UserWarning
            )
            self.model = torch.hub.load(
                "facebookresearch/dinov2",
                architectures[config.model_name],
                pretrained=False,
            )
        checkpoint = torch.load(config.weights_path, map_location="cpu", weights_only=True)
        checkpoint = checkpoint.get("teacher", checkpoint)
        state = {
            key.removeprefix("backbone."): value
            for key, value in checkpoint.items()
            if "dino_head" not in key and "ibot_head" not in key
        }
        self.model.pos_embed = nn.Parameter(torch.zeros(1, 257, config.embed_dim))
        self.model.load_state_dict(state, strict=True)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)), transforms.ToTensor(),
            transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
        ])
        self.instance_norm = nn.InstanceNorm2d(3, affine=False) if config.instance_norm else None

    def forward(self, images):
        if self.instance_norm is not None:
            images = self.instance_norm(images)
        features = self.model.forward_features(images)
        if isinstance(features, dict):
            for key in ("x_prenorm", "x_norm_patchtokens", "last_hidden_state"):
                if key in features:
                    features = features[key]
                    break
        if features.ndim == 2:
            features = features.unsqueeze(1)
        if features.ndim != 3:
            raise ValueError(f"expected [batch,tokens,width], got {features.shape}")
        return features


def build_vision_encoder(config):
    return IdentityTokenEncoder(config.embed_dim) if config.identity else DinoBloomEncoder(config)


def freeze_module(module):
    for parameter in module.parameters():
        parameter.requires_grad = False
    module.eval()

    def disabled_train(mode=True):
        return module

    module.train = disabled_train
