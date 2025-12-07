import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import repeat


class MLP(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()

        self.layer = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.SiLU(),
            nn.Linear(in_features, out_features),
        )

    def forward(self, x: torch.tensor) -> torch.tensor:
        return self.layer(x)


# Reference: https://diffusion.csail.mit.edu/docs/lecture-notes.pdf
class ResidualLayer(nn.Module):
    def __init__(self, channels: int, time_embed_dim: int):
        super().__init__()
        # (B, c, h, w) -> (B, c, h, w)
        self.conv_block1 = nn.Sequential(
            nn.SiLU(),
            nn.BatchNorm2d(channels),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )

        # (B, c, h, w) -> (B, c, h, w)
        self.conv_block2 = nn.Sequential(
            nn.SiLU(),
            nn.BatchNorm2d(channels),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )

        # (B, time_embed) -> (B, channels)
        self.time_embed_mlp = MLP(time_embed_dim, channels)

    def forward(
        self,
        x: torch.tensor,
        t: torch.tensor,
    ) -> torch.tensor:
        # (b, c, h, w) -> (b, c, h, w)
        x_conv = self.conv_block1(x)

        # (b, time_embed) -> (B, c, 1, 1)
        x_time = self.time_embed_mlp(t)
        x_time = repeat(x_time, "b c -> b c h w", h=1, w=1)

        # Concatenate
        x_conv = x_conv + x_time

        # (b, c, h, w) -> (b, c, h, w)
        x_conv = self.conv_block2(x_conv)

        # Add Residual connection
        return x + x_conv


class Encoder(nn.Module):
    def __init__(
        self,
        num_residual_layers: int,
        in_channels: int,
        time_embed_dim: int,
        out_channels: int,
    ):
        super().__init__()
        self.residuals = nn.ModuleList(
            [
                ResidualLayer(in_channels, time_embed_dim)
                for _ in range(num_residual_layers)
            ]
        )

        self.downsample = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=2, padding=1
        )

    def forward(self, x: torch.tensor, t: torch.tensor) -> torch.tensor:
        # (B, in_c, h, w) -> (B, in_c, h, w)
        x_conv = x
        for layer in self.residuals:
            x_conv = layer(x_conv, t)

        # (B, in_c, h, w) -> (B, out_c, h // 2, w // 2)
        downsampled = self.downsample(x_conv)
        return downsampled


class MidCoder(nn.Module):
    def __init__(
        self,
        num_residual_layers: int,
        channels: int,
        time_embed_dim: int,
    ):
        super().__init__()
        self.residuals = nn.ModuleList(
            [
                ResidualLayer(channels, time_embed_dim)
                for _ in range(num_residual_layers)
            ]
        )

    def forward(self, x: torch.tensor, t: torch.tensor):
        for layer in self.residuals:
            x = layer(x, t)

        return x


class Decoder(nn.Module):
    def __init__(
        self,
        num_residual_layers: int,
        in_channels: int,
        time_embed_dim: int,
        out_channels: int,
    ):
        super().__init__()
        # (B, inC, H, W) -> (B, inC, 2H, 2W)
        self.up_sample = nn.Upsample(scale_factor=2, mode="bilinear")

        # (B, inC, 2H, 2W) -> (B, outC, 2H, 2W)
        self.up_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
        )

        self.residuals = nn.ModuleList(
            [
                ResidualLayer(out_channels, time_embed_dim)
                for _ in range(num_residual_layers)
            ]
        )

    def forward(
        self,
        x: torch.tensor,
        t: torch.tensor,
    ) -> torch.tensor:
        x = self.up_sample(x)
        x = self.up_conv(x)

        for layer in self.residuals:
            x = layer(x, t)

        return x


class TimeEmbedding(nn.Module):
    def __init__(self, max_timesteps: int, embedding_dim: int):
        super().__init__()
        # Placeholder class to later implement better position encoding
        self.pos_encoder = nn.Embedding(max_timesteps, embedding_dim)

    def forward(self, t: torch.tensor) -> torch.tensor:
        # (B, 1) -> (B, embedding_dim)
        return self.pos_encoder(t)


class UNet(nn.Module):
    def __init__(
        self,
        input_channel: int,
        channels: list[int],
        max_timesteps: int,
        time_embed_dim: int,
    ):
        super().__init__()

        # (B, 1, H, W) -> (B, C, H, W)
        self.init_conv = nn.Sequential(
            nn.Conv2d(input_channel, channels[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(channels[0]),
            nn.SiLU(),
        )

        self.time_embedding = TimeEmbedding(max_timesteps, time_embed_dim)

        encoders = []
        decoders = []

        for i in range(len(channels) - 1):
            encoder = Encoder(
                num_residual_layers=2,
                in_channels=channels[i],
                time_embed_dim=time_embed_dim,
                out_channels=channels[i + 1],
            )
            decoder = Decoder(
                num_residual_layers=2,
                in_channels=channels[i + 1],
                time_embed_dim=time_embed_dim,
                out_channels=channels[i],
            )

            encoders.append(encoder)
            decoders.append(decoder)

        # Reverse the list to get the decoders in line
        self.encoder_module = nn.ModuleList(encoders)
        self.decoder_module = nn.ModuleList(reversed(decoders))

        self.midcoder = MidCoder(
            num_residual_layers=2, channels=channels[-1], time_embed_dim=time_embed_dim
        )

        self.final_conv = nn.Sequential(
            nn.Conv2d(channels[0], input_channel, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.tensor, t: torch.tensor) -> torch.tensor:
        # (B, 1, H, W) -> (B, C, H, W)
        x = self.init_conv(x)

        # (B, 1) -> (B, time_embed_dim)
        x_time = self.time_embedding(t)

        residuals = []
        for i, encoder in enumerate(self.encoder_module):
            x = encoder(x, x_time)
            residuals.append(x.clone())

        x = self.midcoder(x, x_time)

        for i, decoder in enumerate(self.decoder_module):
            x_res = residuals.pop()
            x = x + x_res
            x = decoder(x, x_time)

        x = self.final_conv(x)

        return x
