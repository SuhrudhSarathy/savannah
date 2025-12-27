import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat, rearrange
import torchvision.models as models

from torch import Tensor
from dataclasses import dataclass, field


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

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel,
            stride=stride,
            padding=padding,
        )

        self.norm = nn.GroupNorm(num_groups=8, num_channels=out_channels)

        self.activation = nn.Mish()
        self.module = nn.Sequential(self.conv, self.norm, self.activation)

    def forward(self, x: Tensor) -> Tensor:
        # x (B, in_channels, T) -> (B, out_channels, t)
        return self.module(x)


class FiLMConditioning(nn.Module):
    def __init__(self, condition_dim: int, embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.activation = nn.Mish()
        self.condition_encoder = nn.Linear(condition_dim, 2 * embedding_dim)

        self.module = nn.Sequential(self.activation, self.condition_encoder)

    def forward(self, x_cond: Tensor) -> tuple[Tensor, Tensor]:
        # x_cond (B, x_cond) -> (B, 2 * x_embed)
        x_embed = self.module(x_cond)
        gamma, beta = torch.split(x_embed, self.embedding_dim, dim=-1)

        return gamma, beta


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

        self.condition_channels = condition_channels
        self.out_channels = out_channels

        self.conv_block1 = ConvBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel=kernel,
            stride=stride,
            padding=padding,
        )
        self.conv_block2 = ConvBlock(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel=kernel,
            stride=stride,
            padding=padding,
        )

        self.film = FiLMConditioning(self.condition_channels, self.out_channels)

        self.shortcut_conv = nn.Identity()
        if in_channels != out_channels:
            self.shortcut_conv = ConvBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel=kernel,
                stride=stride,
                padding=padding,
            )

    def forward(self, x: Tensor, x_cond: Tensor) -> Tensor:
        # x (B, inC, T), x_cond (B, condC)
        # x (B, inC, T) -> (B, outC, T)
        xconv1 = self.conv_block1(x)

        gamma, beta = self.film(x_cond)

        gamma = repeat(gamma, "b c -> b c t", t=xconv1.shape[-1])
        beta = repeat(beta, "b c -> b c t", t=xconv1.shape[-1])

        # FiLM Conditioning
        xconv1 = gamma * xconv1 + beta

        xconv2 = self.conv_block2(xconv1)
        return xconv2 + self.shortcut_conv(x)


class DownSample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.down = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.down(x)


class UpSample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up_sample = nn.Upsample(scale_factor=2, mode="nearest")
        self.up_conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
        )

        self.up = nn.Sequential(self.up_sample, self.up_conv)

    def forward(self, x: Tensor) -> Tensor:
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
        self.residuals = nn.ModuleList(
            [
                ResidualConvBlockWithCondition(
                    in_channels=in_channels,
                    condition_channels=cond_channels,
                    out_channels=in_channels,
                )
                for _ in range(num_residual_layers)
            ]
        )

        self.downsample = DownSample(in_channels, out_channels)

    def forward(self, x: Tensor, c: Tensor) -> Tensor:
        # (B, in_c, t) -> (B, in_c, t)
        x_conv = x
        for layer in self.residuals:
            x_conv = layer(x_conv, c)

        # (B, in_c, t) -> (B, out_c, t // 2)
        downsampled = self.downsample(x_conv)
        return downsampled


class MidCoder(nn.Module):
    def __init__(
        self,
        num_residual_layers: int,
        channels: int,
        cond_channels: int,
    ):
        super().__init__()
        self.residuals = nn.ModuleList(
            [
                ResidualConvBlockWithCondition(
                    channels, condition_channels=cond_channels, out_channels=channels
                )
                for _ in range(num_residual_layers)
            ]
        )

    def forward(self, x: Tensor, c: Tensor):
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
        # (B, inC, T) -> (B, outC, 2T)
        self.upsample = UpSample(in_channels=in_channels, out_channels=out_channels)

        self.residuals = nn.ModuleList(
            [
                ResidualConvBlockWithCondition(
                    out_channels,
                    condition_channels=cond_channels,
                    out_channels=out_channels,
                )
                for _ in range(num_residual_layers)
            ]
        )

    def forward(
        self,
        x: Tensor,
        c: Tensor,
    ) -> Tensor:
        x = self.upsample(x)

        for layer in self.residuals:
            x = layer(x, c)

        return x


class TimeEmbedding(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim
        i = torch.arange(0, embedding_dim, 2).float()  # Only taking even indices
        div_term = torch.exp(torch.log(torch.tensor(10000.0)) * (i / embedding_dim))
        self.register_buffer("div_term", div_term)

    def forward(self, t: Tensor) -> Tensor:
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


class SpatialSoftmax(nn.Module):
    def __init__(self, height: int = 3, width: int =3, channel: int =512):
        super().__init__()
        self.height = height
        self.width = width
        self.channel = channel

        # Create coordinates from -1 to 1
        pos_x, pos_y = torch.meshgrid(
            torch.linspace(-1, 1, self.width),
            torch.linspace(-1, 1, self.height),
            indexing="ij",
        )
        self.register_buffer("pos_x", pos_x.reshape(-1))
        self.register_buffer("pos_y", pos_y.reshape(-1))

    def forward(self, x):
        # x: [B, 512, 3, 3]
        b, c, h, w = x.size()
        logits = x.view(b, c, h * w)
        probs = F.softmax(logits, dim=-1)

        # Expected x, y coordinates
        expected_x = torch.sum(probs * self.pos_x, dim=-1)
        expected_y = torch.sum(probs * self.pos_y, dim=-1)

        # Output: [B, 1024] (512 x-coords and 512 y-coords)
        return torch.stack([expected_x, expected_y], dim=-1).view(b, c * 2)


class ResNetBackbone(nn.Module):
    def __init__(self, cond_channels: int):
        super().__init__()
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        self.backbone = nn.Sequential(*list(self.resnet.children())[:-2])
        self.spatial_softmax = SpatialSoftmax()
        self.repoj = nn.Linear(2 * 512, cond_channels)

        self.module = nn.Sequential(self.backbone, self.spatial_softmax, self.repoj)

        self.replace_bn_with_gn(self.module)

    def forward(self, x):
        return self.module(x)

    def replace_bn_with_gn(self, model, num_groups=32):
        for name, module in model.named_children():
            if isinstance(module, nn.BatchNorm2d):
                # Get the number of channels from the existing BN layer
                num_channels = module.num_features

                # Ensure num_groups is valid (must divide num_channels)
                # Fallback to 1 group (LayerNorm style) if 32 doesn't divide evenly
                actual_groups = num_groups if num_channels % num_groups == 0 else 1

                # Create the new GroupNorm layer
                gn = nn.GroupNorm(actual_groups, num_channels)

                # Replace the layer in the model
                setattr(model, name, gn)
            else:
                # Recurse through sub-modules (like Sequential or custom Blocks)
                self.replace_bn_with_gn(module, num_groups)


@dataclass
class UNetConfig:
    n_obs: int = 2
    n_pred: int = 16
    n_action: int = 8

    state_dim: int = 2
    input_channels: int = state_dim
    unet_channels: list[int] = field(default_factory=lambda: [256, 512, 1024])
    time_embedding_dim: int = 256
    vision_embedding_dim: int = 512


class UNet1D(nn.Module):
    def __init__(self, config: UNetConfig):
        super().__init__()

        self.config = config
        channels = config.unet_channels

        # (B, n_pred, state_dim) -> (B, C, T)
        self.init_conv = nn.Sequential(
            nn.Conv1d(
                self.config.input_channels, channels[0], kernel_size=3, padding=1
            ),
            nn.BatchNorm1d(channels[0]),
            nn.Mish(),
        )

        # (B, 1) -> (B, time_embedding_dim)
        self.time_embedding = TimeEmbedding(self.config.time_embedding_dim)

        # (B, 3, 96, 96) -> (B, 2, vision_embedding_dim) Gives out image features
        self.vision_backbone = ResNetBackbone(self.config.vision_embedding_dim)

        # Condition dim = [time + vision + state]
        self.condition_dim = (
            self.config.time_embedding_dim
            + self.config.n_obs * self.config.vision_embedding_dim
            + self.config.n_obs * self.config.state_dim
        )

        self.cond_projection_mlp = nn.Sequential(
            nn.Linear(self.condition_dim, 4 * self.condition_dim),
            nn.Mish(),
            nn.Linear(4 * self.condition_dim, self.condition_dim),
        )

        encoders = []
        decoders = []

        for i in range(len(channels) - 1):
            encoder = Encoder(
                num_residual_layers=2,
                in_channels=channels[i],
                cond_channels=self.condition_dim,
                out_channels=channels[i + 1],
            )
            decoder = Decoder(
                num_residual_layers=2,
                in_channels=2 * channels[i + 1],
                cond_channels=self.condition_dim,
                out_channels=channels[i],
            )

            encoders.append(encoder)
            decoders.append(decoder)

        # Reverse the list to get the decoders in line
        self.encoder_module = nn.ModuleList(encoders)
        self.decoder_module = nn.ModuleList(reversed(decoders))

        self.midcoder = MidCoder(
            num_residual_layers=2,
            channels=channels[-1],
            cond_channels=self.condition_dim,
        )

        self.final_conv = nn.Sequential(
            nn.Conv1d(
                channels[0], self.config.input_channels, kernel_size=3, padding=1
            ),
        )

    def forward(self, x: Tensor, x_image: Tensor, state: Tensor, t: Tensor) -> Tensor:
        # x (B, n_pred, state_dim) -> (B, state_dim, n_pred)
        x = rearrange(x, "b n_pred state_dim -> b state_dim n_pred")

        # (B, state_dim, n_pred) -> (B, C, n_pred)
        x = self.init_conv(x)

        # (B, 1) -> (B, embed_dim)
        x_time = self.time_embedding(t)

        B = x_image.shape[0]

        # (B, n_obs, c, h, w) -> (B * n_obs, c, h, w)
        x_image = rearrange(x_image, "b n c h w -> (b n) c h w")

        # (B * n_obs, c, h, w) -> (B * n_obs, 512)
        x_image = self.vision_backbone(x_image)

        # (B * n_obs, c, h, w) -> (B * n_obs, 512)
        x_image = rearrange(x_image, "(b n) c -> b n c", b=B)

        # (B, n_obs, vision_dim) -> (B n_obs * vision_dim)
        x_image_cond = rearrange(x_image, "b n c -> b (n c)")

        # (B, n_obs, state_dim) -> (B, n_obs * state_dim)
        x_state_cond = rearrange(state, "b n s -> b (n s)")

        # (B, embed_dim) + (B, n_obs * vision_dim) + (B, n_obs * state_dim) = (B, cond_dim)
        x_cond = torch.cat([x_time, x_image_cond, x_state_cond], dim=-1)
        x_cond = self.cond_projection_mlp(x_cond)

        residuals = []
        for i, encoder in enumerate(self.encoder_module):
            x = encoder(x, x_cond)
            residuals.append(x.clone())

        x = self.midcoder(x, x_cond)

        for i, decoder in enumerate(self.decoder_module):
            x_res = residuals.pop()
            x = torch.cat([x, x_res], dim=1)
            x = decoder(x, x_cond)

        x = self.final_conv(x)

        # Get back the shape
        x = rearrange(x, "b state_dim n_pred -> b n_pred state_dim")

        return x

    def _num_params(self):
        num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return num_params / 1000000


if __name__ == "__main__":
    device = torch.device("mps")

    B = 8
    imgC = 3
    n_obs = 2
    n_pred = 16
    state_dim = 2

    x_image = torch.randn((B, n_obs, imgC, 96, 96))
    x_time = torch.randint(0, 1000, (B, 1))
    state = torch.randn((B, n_obs, state_dim))
    x = torch.randn((B, n_pred, state_dim))

    model = UNet1D(UNetConfig())
    print(f"Num Params: {model._num_params(): 0.4f}M")

    out = model(x, x_image, state, x_time)
    print(out.shape)
