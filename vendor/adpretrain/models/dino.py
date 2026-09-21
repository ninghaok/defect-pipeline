"""DINOv2 backbone wrapper used by ADPretrain residual features (local weights only, no download)."""
from __future__ import annotations

import os

import torch
import torch.nn as nn

from .dinov2.models import vision_transformer as vision_transformer_dinov2

_ARCHS = {"dinov2-base": ("base", [3, 6, 9, 12], 768), "dinov2-large": ("large", [6, 12, 18, 24], 1024)}


class DinoModel(nn.Module):
    def __init__(self, name: str = "dinov2-large", device: str = "cuda:0", weight_path: str | None = None):
        super().__init__()
        if name not in _ARCHS:
            raise ValueError(f"{name} is not supported; choose one of {sorted(_ARCHS)}")
        self.name = name
        arch, self.target_layers, self.embed_dim = _ARCHS[name]
        weight_path = weight_path or os.environ.get("ADPRETRAIN_DINOV2_WEIGHT")
        if not weight_path or not os.path.isfile(weight_path):
            raise FileNotFoundError(f"DINOv2 weight is required (got {weight_path!r}); network download is disabled")
        self.visual_encoder = vision_transformer_dinov2.__dict__[f"vit_{arch}"](
            patch_size=14, img_size=518, block_chunks=0, init_values=1e-8,
            interpolate_antialias=False, interpolate_offset=0.1,
        )
        state = torch.load(weight_path, map_location="cpu", weights_only=False)
        missing, unexpected = self.visual_encoder.load_state_dict(state, strict=False)
        if unexpected or any(not k.startswith("mask_token") for k in missing):
            raise RuntimeError(f"DINOv2 weight mismatch: missing={missing[:3]} unexpected={unexpected[:3]}")
        for parameter in self.visual_encoder.parameters():
            parameter.requires_grad = False
        self.visual_encoder.eval()
        self.device = torch.device(device)

    @property
    def feature_dimensions(self):
        return [self.embed_dim] * 4

    def encode_image_from_tensors(self, image_tensors, return_global=False, shape="img"):
        with torch.no_grad():
            patch_features = self.encode_image(image_tensors, self.target_layers)
            if shape == "img":
                for i, feature in enumerate(patch_features):
                    b, length, c = feature.shape
                    side = int(length ** 0.5)
                    patch_features[i] = feature.permute(0, 2, 1).reshape(b, c, side, side)
        return (None, patch_features) if return_global else patch_features

    def encode_image(self, x, target_layers):
        x = self.visual_encoder.prepare_tokens(x)
        outs = []
        for index, block in enumerate(self.visual_encoder.blocks, start=1):
            if index > target_layers[-1]:
                break
            x = block(x)
            if index in target_layers:
                outs.append(x)
        skip = 1 + self.visual_encoder.num_register_tokens
        return [e[:, skip:, :] for e in outs]
