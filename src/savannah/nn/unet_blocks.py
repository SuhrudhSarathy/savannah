import torch
import torch.nn as nn
from einops import repeat


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel: int = 3,
        stride: int = 1,
        padding: int = 1,
    ):
        super().__init__()
        self.module = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel, stride=stride, padding=padding),
            nn.GroupNorm(num_groups=8, num_channels=out_channels),
            nn.Mish(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.module(x)


class FiLMConditioning(nn.Module):
    """Projects a flat condition vector to per-channel gamma/beta for 1D conv FiLM."""

    def __init__(self, condition_dim: int, embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.module = nn.Sequential(nn.Mish(), nn.Linear(condition_dim, 2 * embedding_dim))

    def forward(self, x_cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x_embed = self.module(x_cond)
        return torch.split(x_embed, self.embedding_dim, dim=-1)


class ResidualConvBlockWithCondition(nn.Module):
    def __init__(
        self,
        in_channels: int,
        condition_channels: int,
        out_channels: int,
        kernel: int = 3,
        stride: int = 1,
        padding: int = 1,
    ):
        super().__init__()
        self.conv_block1 = ConvBlock(in_channels, out_channels, kernel, stride, padding)
        self.conv_block2 = ConvBlock(out_channels, out_channels, kernel, stride, padding)
        self.film = FiLMConditioning(condition_channels, out_channels)
        self.shortcut_conv = (
            nn.Identity()
            if in_channels == out_channels
            else ConvBlock(in_channels, out_channels, kernel, stride, padding)
        )

    def forward(self, x: torch.Tensor, x_cond: torch.Tensor) -> torch.Tensor:
        xconv1 = self.conv_block1(x)
        gamma, beta = self.film(x_cond)
        gamma = repeat(gamma, "b c -> b c t", t=xconv1.shape[-1])
        beta = repeat(beta, "b c -> b c t", t=xconv1.shape[-1])
        xconv1 = gamma * xconv1 + beta
        return self.conv_block2(xconv1) + self.shortcut_conv(x)


class DownSample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.down = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(x)


class UpSample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(x)


class Encoder(nn.Module):
    def __init__(
        self,
        num_residual_layers: int,
        in_channels: int,
        cond_channels: int,
        out_channels: int,
    ):
        super().__init__()
        self.residuals = nn.ModuleList([
            ResidualConvBlockWithCondition(in_channels, cond_channels, in_channels)
            for _ in range(num_residual_layers)
        ])
        self.downsample = DownSample(in_channels, out_channels)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        for layer in self.residuals:
            x = layer(x, c)
        return self.downsample(x)


class MidCoder(nn.Module):
    def __init__(self, num_residual_layers: int, channels: int, cond_channels: int):
        super().__init__()
        self.residuals = nn.ModuleList([
            ResidualConvBlockWithCondition(channels, cond_channels, channels)
            for _ in range(num_residual_layers)
        ])

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        for layer in self.residuals:
            x = layer(x, c)
        return x


class Decoder(nn.Module):
    def __init__(
        self,
        num_residual_layers: int,
        in_channels: int,
        cond_channels: int,
        out_channels: int,
    ):
        super().__init__()
        self.upsample = UpSample(in_channels, out_channels)
        self.residuals = nn.ModuleList([
            ResidualConvBlockWithCondition(out_channels, cond_channels, out_channels)
            for _ in range(num_residual_layers)
        ])

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        for layer in self.residuals:
            x = layer(x, c)
        return x
