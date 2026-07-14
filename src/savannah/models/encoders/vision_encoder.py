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

        # Backbone returns a flat global descriptor (B, out_channels) —
        # e.g. SpatialSoftmax keypoints — so project it to a single token.
        self.proj = nn.Linear(backbone.out_channels, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)  # (B, out_channels)

        features = self.proj(features)  # (B, embed_dim)

        return features.unsqueeze(1)  # (B, 1, embed_dim)
