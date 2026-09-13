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
from savannah.nn.perceiver_resampler import PerceiverResampler
from savannah.nn.token_learner import TokenLearner

RESAMPLER_TYPES = ("perceiver", "token_learner")


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
        num_query_tokens: int | None = None,
        resampler_type: str = "perceiver",
        resampler_heads: int = 8,
        resampler_depth: int = 1,
        resampler_hidden_dim: int = 64,
    ):
        super().__init__()
        self.backbone = backbone
        self.embed_dim = embed_dim
        self.num_cameras = num_cameras
        self.num_obs = num_obs

        # Optional resampler that compresses each image's tokens down to a
        # fixed number of query tokens. Both options share the same API:
        # constructed as (embed_dim, num_queries, ...) and exposing
        # `.num_queries` + `forward(x) -> (B, num_queries, embed_dim)`.
        if num_query_tokens is None:
            self.resampler = None
        elif resampler_type == "perceiver":
            self.resampler = PerceiverResampler(
                embed_dim=backbone.out_channels,
                num_queries=num_query_tokens,
                num_attn_heads=resampler_heads,
                depth=resampler_depth,
            )
        elif resampler_type == "token_learner":
            self.resampler = TokenLearner(
                embed_dim=backbone.out_channels,
                num_queries=num_query_tokens,
                hidden=resampler_hidden_dim,
            )
        else:
            raise ValueError(
                f"Unknown resampler_type {resampler_type!r}, expected one of {RESAMPLER_TYPES}"
            )

        # Camera Embedding
        self.camera_embedding = nn.Embedding(self.num_cameras, backbone.out_channels)

        # History Embedding
        self.history_embedding = nn.Embedding(self.num_obs, backbone.out_channels)

        if embed_dim == backbone.out_channels:
            self.proj = nn.Identity()
        else:
            self.proj = nn.Linear(backbone.out_channels, embed_dim)

    @property
    def tokens_per_image(self) -> int:
        if self.resampler is not None:
            return self.resampler.num_queries
        return self.backbone.tokens_per_image

    @property
    def num_tokens(self) -> int:
        return self.num_cameras * self.num_obs * self.tokens_per_image

    def extract_tokens_from_image(self, x: torch.Tensor) -> torch.Tensor:
        # x (B, C, H, W) -> x_out (B, T_i, out_channels)
        features = self.backbone(x)

        if features.dim() == 2:
            features = features.unsqueeze(1)

        if self.resampler is not None:
            features = self.resampler(features)

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
