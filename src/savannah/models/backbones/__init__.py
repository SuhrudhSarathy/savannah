from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class VisionFeatureExtractor(ABC, nn.Module):
    """
    Any backbone that takes (B, 3, H, W) and returns
    spatial features (B, out_channels, H', W').
    """

    @property
    @abstractmethod
    def out_channels(self) -> int: ...

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...


class DummyVisionBackbone(VisionFeatureExtractor):
    def __init__(self, out_channels: int = 64, downsample_factor: int = 4):
        """
        A lightweight dummy backbone for testing.
        Uses a single Conv2d layer to simulate spatial reduction and channel expansion.
        """
        super().__init__()
        self._out_channels = out_channels

        # Simulates a backbone by reducing spatial dims (H, W) and increasing channels
        self.net = nn.Conv2d(
            in_channels=3,
            out_channels=out_channels,
            kernel_size=downsample_factor,
            stride=downsample_factor,
        )

    @property
    def out_channels(self) -> int:
        return self._out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expected input: (B, 3, H, W)
        # Expected output: (B, out_channels, H // factor, W // factor)
        return self.net(x)
