import torch
import torch.nn as nn
from einops import rearrange, repeat

from savannah.models.encoders import VisionEncoder
from savannah.models.policy import Policy
from savannah.nn.time_embedding import TimeEmbedding
from savannah.objectives import PolicyObjective
from savannah.utils.observation import ObservationKey
from savannah.utils.policy import PolicyOutput


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x (B, in_channels, T) -> (B, out_channels, t)
        return self.module(x)


class FiLMConditioning(nn.Module):
    def __init__(self, condition_dim: int, embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.activation = nn.Mish()
        self.condition_encoder = nn.Linear(condition_dim, 2 * embedding_dim)

        self.module = nn.Sequential(self.activation, self.condition_encoder)

    def forward(self, x_cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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

    def forward(self, x: torch.Tensor, x_cond: torch.Tensor) -> torch.Tensor:
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
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

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)

        for layer in self.residuals:
            x = layer(x, c)

        return x


class UNetPolicy(Policy):
    def __init__(
        self,
        embed_dim: int,
        unet_channels: list[int],
        num_residual_layers: int,
        num_obs: int,
        state_dim: int,
        action_dim: int,
        action_horizon: int,
        num_cameras: int,
        vision_encoder: VisionEncoder,
        objective: PolicyObjective,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self._state_dim = state_dim
        self._action_dim = action_dim
        self._action_horizon = action_horizon
        self.num_cameras = num_cameras
        self.num_obs = num_obs

        self.vision_encoder = vision_encoder
        self.objective = objective

        # TimeEmbedding (generic, no learnable parameters)
        self.time_embedding = TimeEmbedding(self.embed_dim)

        # Camera Embedding
        self.camera_embedding = nn.Embedding(num_cameras, self.embed_dim)

        # Condition dim = [time + vision tokens (one per cam per obs frame) + state]
        self.condition_dim = (
            self.embed_dim
            + num_cameras * num_obs * self.embed_dim
            + num_obs * self.state_dim
        )

        self.cond_projection_mlp = nn.Sequential(
            nn.Linear(self.condition_dim, 4 * self.condition_dim),
            nn.Mish(),
            nn.Linear(4 * self.condition_dim, self.condition_dim),
        )

        # (B, action_horizon, action_dim) -> (B, unet_channels[0], action_horizon)
        self.init_conv = nn.Sequential(
            nn.Conv1d(self.action_dim, unet_channels[0], kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=unet_channels[0]),
            nn.Mish(),
        )

        encoders = []
        decoders = []
        for i in range(len(unet_channels) - 1):
            encoders.append(
                Encoder(
                    num_residual_layers=num_residual_layers,
                    in_channels=unet_channels[i],
                    cond_channels=self.condition_dim,
                    out_channels=unet_channels[i + 1],
                )
            )
            decoders.append(
                Decoder(
                    num_residual_layers=num_residual_layers,
                    in_channels=2 * unet_channels[i + 1],
                    cond_channels=self.condition_dim,
                    out_channels=unet_channels[i],
                )
            )

        self.encoders = nn.ModuleList(encoders)
        self.decoders = nn.ModuleList(reversed(decoders))

        self.midcoder = MidCoder(
            num_residual_layers=num_residual_layers,
            channels=unet_channels[-1],
            cond_channels=self.condition_dim,
        )

        self.final_conv = nn.Conv1d(
            unet_channels[0], self.action_dim, kernel_size=3, padding=1
        )

    def forward(self, obs: dict[str, torch.Tensor], *args, **kwargs) -> PolicyOutput:
        x_img = obs[ObservationKey.images]
        x_state = obs[ObservationKey.state]
        x_time = obs[ObservationKey.time]

        # Get noisy actions from kwargs
        noisy_actions = kwargs.get("noisy_actions", None)
        if noisy_actions is None:
            raise AssertionError("Pass noisy action for the model to run")

        # (B,) -> (B, 1) optionally
        if x_time.dim() == 1:
            x_time = x_time.unsqueeze(1)

        # (B, 1) -> (B, embed_dim)
        x_time_cond = self.time_embedding(x_time)

        # (B, num_cameras * num_obs, embed_dim) -> (B, num_cameras * num_obs * embed_dim)
        x_cam_tokens = self._encode_cameras(x_img)
        x_cam_cond = rearrange(x_cam_tokens, "b n d -> b (n d)")

        # (B, num_obs, state_dim) -> (B, num_obs * state_dim)
        x_state_cond = rearrange(x_state, "b n s -> b (n s)")

        # (B, condition_dim)
        x_cond = torch.cat([x_time_cond, x_cam_cond, x_state_cond], dim=-1)
        x_cond = self.cond_projection_mlp(x_cond)

        # (B, action_horizon, action_dim) -> (B, action_dim, action_horizon)
        x = rearrange(noisy_actions, "b t a -> b a t")
        x = self.init_conv(x)

        residuals = []
        for encoder in self.encoders:
            x = encoder(x, x_cond)
            residuals.append(x)

        x = self.midcoder(x, x_cond)

        for decoder in self.decoders:
            x_res = residuals.pop()
            x = torch.cat([x, x_res], dim=1)
            x = decoder(x, x_cond)

        x = self.final_conv(x)

        # (B, action_dim, action_horizon) -> (B, action_horizon, action_dim)
        x_out_action = rearrange(x, "b a t -> b t a")

        return PolicyOutput(actions=x_out_action)
