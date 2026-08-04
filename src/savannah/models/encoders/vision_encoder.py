"""
Job of vision encoder
1. Use the VisionFeatureExtractor to get the per-image position encoded tokens
2. Encode the tokens using per camera encoding
3. Encode the tokens using per token encoding
4. Project the final tokens into a common embedding space
"""

import torch
import torch.nn as nn
from einops import rearrange

from savannah.models.backbones import VisionFeatureExtractor


class VisionEncoder(nn.Module):
    """
    Job of vision encoder is to return vision tokens
    Shape transform: (B, C, H, W) -> (B, T_V, EmbedDim)
    """

    def __init__(
        self,
        backbone: VisionFeatureExtractor,
        embed_dim: int,
        num_cameras: int,
        num_obs: int,
    ):
        super().__init__()
        self.backbone = backbone
        self.embed_dim = embed_dim
        self.num_cameras = num_cameras
        self.num_obs = num_obs

        # Camera Embedding
        self.camera_embedding = nn.Embedding(self.num_cameras, backbone.out_channels)

        # History Embedding
        self.history_embedding = nn.Embedding(self.num_obs, backbone.out_channels)

        if embed_dim == backbone.out_channels:
            self.proj = nn.Identity()
        else:
            self.proj = nn.Linear(backbone.out_channels, embed_dim)

    @property
    def num_tokens(self) -> int:
        return self.num_cameras * self.num_obs * self.backbone.tokens_per_image

    def extract_tokens_from_image(self, x: torch.Tensor) -> torch.Tensor:
        # x (B, C, H, W) -> x_out (B, T_i, out_channels)
        features = self.backbone(x)

        if features.dim() == 2:
            features = features.unsqueeze(1)

        return features

    def forward(self, images: list[torch.Tensor]) -> torch.Tensor:
        """
        images = [(B, n_obs, 3, H, W) ... for cam in n_cams]
        returns (B, num_tokens, embed_dim)
        """

        assert len(images) == self.num_cameras, (
            f"Number of cameras do not match configuration. Received {len(images)}, expected {self.num_cameras}"
        )

        cam_tokens = []
        for cam_idx, image in enumerate(images):
            # image (B, n_obs, 3, H, W)
            n_obs = image.shape[1]

            assert n_obs == self.num_obs, (
                f"Number of observations do not match configuration. Received {n_obs}, expected {self.num_obs}"
            )

            camera_embedding = self.camera_embedding(
                torch.tensor(cam_idx, device=image.device)
            )

            camera_embedding = camera_embedding.view(1, 1, -1)

            combined_images = rearrange(image, "b n c h w -> (b n) c h w")
            combined_features = self.extract_tokens_from_image(combined_images)
            features = rearrange(combined_features, "(b n) t o -> b n t o", n=n_obs)

            t = torch.arange(n_obs, device=image.device)
            history_embedding = self.history_embedding(t)

            history_embedding = history_embedding.view(1, n_obs, 1, -1)
            features = features + history_embedding

            features = rearrange(features, "b n t o -> b (n t) o")
            features = features + camera_embedding

            cam_tokens.append(features)

        combined_tokens = torch.cat(cam_tokens, dim=1)

        # Project combined_tokens
        projected_tokens = self.proj(combined_tokens)

        return projected_tokens
