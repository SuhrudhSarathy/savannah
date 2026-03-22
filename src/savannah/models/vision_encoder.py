from savannah.models.backbones import VisionFeatureExtractor
import torch
import torch.nn as nn
from einops import rearrange

from savannah.nn.positional_embeddings import SinusoidalPositionalEncoding2D


class VisionEncoder(nn.Module):
    def __init__(self, backbone: VisionFeatureExtractor, embed_dim: int):
        super().__init__()
        self.backbone = backbone

        self.spatial_proj = nn.Conv2d(
            backbone.out_channels,
            embed_dim,
            kernel_size=1,
        )

        self.positional_embeddings = SinusoidalPositionalEncoding2D(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)

        features = self.spatial_proj(features)

        features = self.positional_embeddings(features)

        features_out = rearrange(features, "b c h w -> b (h w) c")

        return features_out
