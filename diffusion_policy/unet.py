import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from torch import Tensor


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel: int = 3,
        stride: int = 2,
        padding: int = 1,
    ):
        super().__init__()

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel,
            stride=stride,
            padding=padding,
        )

        self.norm = nn.GroupNorm(num_groups=8, num_channels=out_channels)

        self.activation = nn.Mish()

        self.module = nn.Sequential(
            self.conv,
            self.norm,
            self.activation
        )
    def forward(self, x: Tensor) -> Tensor:
        # x (B, in_channels, T) -> (B, out_channels, t)
        return self.module(x)
    
class ResidualConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel: int = 3,
        stride: int = 2,
        padding: int = 1,
    ):
        super().__init__()
    
if __name__ == "__main__":
    device = torch.device("cuda")

    B, inC, T = 8, 1, 10
    outC = 32

    model = ConvBlock(inC, outC, stride=2).to(device)
    x = torch.randn((B, inC, T), device=device)

    out = model(x)

    print(out.shape)