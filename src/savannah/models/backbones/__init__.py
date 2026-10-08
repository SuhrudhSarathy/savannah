import math
from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class VisionFeatureExtractor(ABC, nn.Module):
    """
    Any backbone that takes (B, 3, H, W) and returns
    either a flat descriptor (B, out_channels) or a token sequence
    (B, tokens_per_image, out_channels).
    """

    @property
    @abstractmethod
    def out_channels(self) -> int: ...

    @property
    @abstractmethod
    def tokens_per_image(self) -> int:
        """Number of embedding vectors produced per input image.

        Pooling backbones (e.g. ResNetBackbone with SpatialSoftmax) return 1.
        Spatial backbones (e.g. LoRAResNetBackbone) return H' * W'.
        """
        ...

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...


class DummyVisionBackbone(VisionFeatureExtractor):
    def __init__(
        self,
        out_channels: int = 64,
        tokens_per_image: int = 1,
        downsample_factor: int = 4,
    ):
        """
        A lightweight, randomly initialised backbone for tests and smoke runs
        -- no pretrained weights to download, works at any input resolution.

        tokens_per_image=1 mimics a pooling backbone (returns (B, out_channels)),
        anything larger mimics a spatial backbone (returns
        (B, tokens_per_image, out_channels)) and must be a perfect square.
        """
        super().__init__()
        grid = math.isqrt(tokens_per_image)
        assert grid * grid == tokens_per_image, (
            f"tokens_per_image must be a perfect square, got {tokens_per_image}"
        )
        self._out_channels = out_channels
        self._tokens_per_image = tokens_per_image

        # Simulates a backbone by reducing spatial dims (H, W) and increasing channels
        self.net = nn.Conv2d(
            in_channels=3,
            out_channels=out_channels,
            kernel_size=downsample_factor,
            stride=downsample_factor,
        )
        self.pool = nn.AdaptiveAvgPool2d(grid)

    @property
    def out_channels(self) -> int:
        return self._out_channels

    @property
    def tokens_per_image(self) -> int:
        return self._tokens_per_image

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, 3, H, W) -> (B, out_channels, grid, grid)
        features = self.pool(self.net(x))
        if self._tokens_per_image == 1:
            return features.flatten(1)  # (B, out_channels)
        return features.flatten(2).transpose(1, 2)  # (B, grid * grid, out_channels)
