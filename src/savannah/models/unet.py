import torch
import torch.nn as nn
from einops import rearrange, repeat

from savannah.models.encoders import VisionEncoder, StateEncoder
from savannah.models.encoders.language_encoder import LanguageEncoder
from savannah.models.policy import Policy
from savannah.nn.time_embedding import TimeEmbedding
from savannah.objectives import PolicyObjective
from savannah.utils.debug import debug_stat
from savannah.utils.log import logger
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
        language_encoder: LanguageEncoder | None = None,
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

        self.language_encoder = language_encoder
        if language_encoder is not None:
            logger.debug("Initialised Language Encoder into the model")

        # TimeEmbedding (generic, no learnable parameters)
        self.time_embedding = TimeEmbedding(self.embed_dim)

        self.state_encoder = StateEncoder(state_dim, embed_dim)

        # Condition dim = [time + vision tokens + state + language (optional)]
        # Language is assumed to collapse to a single pooled token (e.g. an
        # eos-only LanguageEncoder), matching the fixed-size FiLM condition
        # vector this model requires.
        self.n_vision_tokens = vision_encoder.num_tokens * self.embed_dim
        self.n_state_tokens = num_obs * self.embed_dim
        self.n_time_tokens = 1 * self.embed_dim
        self.n_language_tokens = (
            1 * self.embed_dim if language_encoder is not None else 0
        )

        self.condition_dim = (
            self.n_time_tokens
            + self.n_vision_tokens
            + self.n_state_tokens
            + self.n_language_tokens
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
        x_language = obs.get(ObservationKey.language, None)

        # Get noisy actions from kwargs
        noisy_actions = kwargs.get("noisy_actions", None)
        if noisy_actions is None:
            raise AssertionError("Pass noisy action for the model to run")

        for cam_idx, img in enumerate(x_img):
            debug_stat(f"x_img[{cam_idx}] (raw)", img)
        debug_stat("x_state (raw)", x_state)
        debug_stat("x_time (raw)", x_time)
        if x_language is not None:
            logger.debug("x_language (raw) {}", x_language)
        debug_stat("noisy_actions (raw)", noisy_actions)

        if x_language is not None:
            if self.language_encoder is None:
                raise RuntimeError(
                    "Language instruction is provided but LanguageEncoder is not specified. Not using language instruction"
                )
            else:
                # (B, xx, embed_dim)
                x_language = self.language_encoder(x_language)
                debug_stat("x_lang", x_language)

        # (B,) -> (B, 1) optionally
        if x_time.dim() == 1:
            x_time = x_time.unsqueeze(1)

        # (B, 1) -> (B, embed_dim)
        x_time_cond = self.time_embedding(x_time)
        debug_stat("x_time_cond", x_time_cond)

        # (B, num_cameras * num_obs, embed_dim) -> (B, num_cameras * num_obs * embed_dim)
        x_cam_tokens = self.vision_encoder(x_img)
        debug_stat("x_cam_tokens", x_cam_tokens)

        x_cam_cond = rearrange(x_cam_tokens, "b n d -> b (n d)")
        debug_stat("x_cam_cond", x_cam_cond)

        # (B, num_obs, state_dim) -> (B, num_obs * state_dim)
        x_state_tokens = self.state_encoder(x_state)
        x_state_cond = rearrange(x_state_tokens, "b n s -> b (n s)")
        debug_stat("x_state_cond", x_state_cond)

        logger.debug(
            "Time tokens: {} Vision tokens: {} State Tokens: {} Language Tokens: {}",
            self.n_time_tokens,
            self.n_vision_tokens,
            self.n_state_tokens,
            self.n_language_tokens,
        )

        x_cond_parts = [x_time_cond, x_cam_cond, x_state_cond]
        if self.language_encoder is not None:
            x_language_cond = rearrange(x_language, "b n d -> b (n d)")
            debug_stat("x_language_cond", x_language_cond)
            x_cond_parts.append(x_language_cond)

        # (B, condition_dim)
        x_cond = torch.cat(x_cond_parts, dim=-1)
        debug_stat("x_cond (pre-mlp)", x_cond)
        x_cond = self.cond_projection_mlp(x_cond)
        debug_stat("x_cond (post-mlp)", x_cond)

        # (B, action_horizon, action_dim) -> (B, action_dim, action_horizon)
        x = rearrange(noisy_actions, "b t a -> b a t")
        x = self.init_conv(x)
        debug_stat("x (init_conv)", x)

        residuals = []
        for i, encoder in enumerate(self.encoders):
            x = encoder(x, x_cond)
            debug_stat(f"x (encoder[{i}])", x)
            residuals.append(x)

        x = self.midcoder(x, x_cond)
        debug_stat("x (midcoder)", x)

        for i, decoder in enumerate(self.decoders):
            x_res = residuals.pop()
            x = torch.cat([x, x_res], dim=1)
            x = decoder(x, x_cond)
            debug_stat(f"x (decoder[{i}])", x)

        x = self.final_conv(x)
        debug_stat("x (final_conv)", x)

        # (B, action_dim, action_horizon) -> (B, action_horizon, action_dim)
        x_out_action = rearrange(x, "b a t -> b t a")
        debug_stat("x_out_action", x_out_action)

        return PolicyOutput(actions=x_out_action)
