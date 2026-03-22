import torch
import torch.nn as nn

from abc import ABC, abstractmethod


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
