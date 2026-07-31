import torch
import torch.nn as nn

from savannah.models.backbones import VisionFeatureExtractor


class VisionEncoder(nn.Module):
    """
    Job of vision encoder is to return vision tokens
    Shape transform: (B, C, H, W) -> (B, T_V, EmbedDim)
    """

    def __init__(self, backbone: VisionFeatureExtractor, embed_dim: int):
        super().__init__()
        self.backbone = backbone
        if embed_dim == backbone.out_channels:
            self.proj = nn.Identity()
        else:
            self.proj = nn.Linear(backbone.out_channels, embed_dim)

    @property
    def tokens_per_image(self) -> int:
        return self.backbone.tokens_per_image

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)

        # Pooling backbones (e.g. ResNetBackbone) return (B, out_channels).
        # Normalize to (B, 1, out_channels) so callers always get 3D tokens.
        if features.dim() == 2:
            features = features.unsqueeze(1)

        return self.proj(features)  # (B, T_V, embed_dim)
