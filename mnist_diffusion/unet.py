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
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )

        # (B, c, h, w) -> (B, c, h, w)
        self.conv_block2 = nn.Sequential(
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )

        self.gn1 = nn.GroupNorm(num_groups=4, num_channels=channels, affine=False)
        self.gn2 = nn.GroupNorm(num_groups=4, num_channels=channels, affine=False)

        # (B, time_embed) -> (B, 4*channels)
        self.time_embed_mlp = MLP(time_embed_dim, 4 * channels)

    def forward(
        self,
        x: torch.tensor,
        t: torch.tensor,
    ) -> torch.tensor:
        # Time Projection
        # (B, time_embed) -> (B, 4 * c)
        time_proj = self.time_embed_mlp(t)
        # Split the (B, 4*c) -> 4 * (B, c)
        gamma1, beta1, gamma2, beta2 = torch.chunk(time_proj, chunks=4, dim=1)

        gamma1 = repeat(gamma1, "b c -> b c h w", h=1, w=1)
        beta1 = repeat(beta1, "b c -> b c h w", h=1, w=1)
        gamma2 = repeat(gamma2, "b c -> b c h w", h=1, w=1)
        beta2 = repeat(beta2, "b c -> b c h w", h=1, w=1)

        # (b, c, h, w) -> (b, c, h, w)
        x_norm1 = self.gn1(x)
        x_cond1 = x_norm1 * gamma1 + beta1
        x_conv1 = self.conv_block1(x_cond1)

        x_norm2 = self.gn2(x_conv1)
        x_cond2 = x_norm2 * gamma2 + beta2
        x_conv2 = self.conv_block1(x_cond2)

        # Add Residual connection
        return x + x_conv2


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
        self.embedding_dim = embedding_dim
        i = torch.arange(0, embedding_dim, 2).float()  # Only taking even indices
        div_term = torch.exp(torch.log(torch.tensor(10000.0)) * (i / embedding_dim))
        self.register_buffer("div_term", div_term)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t has shape (B, 1) or just (B,) where B is the batch size.
        # We ensure it's (B, 1) for broadcasting
        if t.dim() == 1:
            t = t.unsqueeze(-1)  # -> (B, 1)

        # 2. Compute the position argument (pos / div_term)
        # pos is the time 't'. Shape: (B, 1).
        # div_term has shape: (embedding_dim // 2,).
        # result_shape: (B, embedding_dim // 2)
        pos_arg = t / self.div_term

        # 3. Apply sin to even indices and cos to odd indices

        # Initialize the output tensor. Shape: (B, embedding_dim)
        output = torch.zeros(t.shape[0], self.embedding_dim, device=t.device)

        # Even indices: Use sin
        # output[:, 0::2] gets all even indices of the last dimension
        output[:, 0::2] = torch.sin(pos_arg)

        # Odd indices: Use cos
        # output[:, 1::2] gets all odd indices of the last dimension
        # Note: We only use sin if i=0 and cos if i=1 in the original paper,
        # but modern implementations often pair them this way for stability.
        # In the original paper's formula, 2i uses sin and 2i+1 uses cos.
        # The pos_arg already captures the 2i/d_model relationship because we stepped by 2 for i.
        output[:, 1::2] = torch.cos(pos_arg)

        # The final output shape is (B, embedding_dim)
        return output


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
                in_channels=2 * channels[i + 1],
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
            x = torch.cat([x, x_res], dim=1)
            x = decoder(x, x_time)

        x = self.final_conv(x)

        return x


if __name__ == "__main__":
    model = UNet(1, [16, 32, 64], 1000, 512)
    x = torch.randn((8, 1, 28, 28))
    t = torch.randint(0, 1000, (8,))
    out = model(x, t)
    print(out.shape)
