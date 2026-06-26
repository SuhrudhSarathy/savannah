import warnings

import torch
from einops import rearrange

from savannah.models.policy import Policy
from savannah.models.vision_encoder import VisionEncoder
from savannah.nn.time_embedding import TimeEmbedding
from savannah.nn.unet_blocks import Decoder, Encoder, MidCoder
from savannah.objectives import PolicyObjective
from savannah.utils.observation import ObservationKey
from savannah.utils.policy import PolicyOutput

import torch.nn as nn


class UNetPolicy(Policy):
    """
    Deprecated. Use ModularPolicy + UNetActionHead + ConditionEncoder instead.
    This class remains for checkpoint compatibility only and will be removed.
    """

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
        warnings.warn(
            "UNetPolicy is deprecated. Use ModularPolicy + UNetActionHead instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__()

        self.embed_dim = embed_dim
        self._state_dim = state_dim
        self._action_dim = action_dim
        self._action_horizon = action_horizon
        self.num_cameras = num_cameras
        self.num_obs = num_obs

        self.vision_encoder = vision_encoder
        self.objective = objective

        self.time_embedding = TimeEmbedding(self.embed_dim)
        self.camera_embedding = nn.Embedding(num_cameras, self.embed_dim)

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

        self.init_conv = nn.Sequential(
            nn.Conv1d(self.action_dim, unet_channels[0], kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=unet_channels[0]),
            nn.Mish(),
        )

        encoders, decoders = [], []
        for i in range(len(unet_channels) - 1):
            encoders.append(Encoder(
                num_residual_layers=num_residual_layers,
                in_channels=unet_channels[i],
                cond_channels=self.condition_dim,
                out_channels=unet_channels[i + 1],
            ))
            decoders.append(Decoder(
                num_residual_layers=num_residual_layers,
                in_channels=2 * unet_channels[i + 1],
                cond_channels=self.condition_dim,
                out_channels=unet_channels[i],
            ))

        self.encoders = nn.ModuleList(encoders)
        self.decoders = nn.ModuleList(list(reversed(decoders)))
        self.midcoder = MidCoder(
            num_residual_layers=num_residual_layers,
            channels=unet_channels[-1],
            cond_channels=self.condition_dim,
        )
        self.final_conv = nn.Conv1d(unet_channels[0], self.action_dim, kernel_size=3, padding=1)

    def forward(self, obs: dict[str, torch.Tensor], *args, **kwargs) -> PolicyOutput:
        x_img = obs[ObservationKey.images]
        x_state = obs[ObservationKey.state]
        x_time = obs[ObservationKey.time]

        noisy_actions = kwargs.get("noisy_actions")
        if noisy_actions is None:
            raise AssertionError("Pass noisy action for the model to run")

        if x_time.dim() == 1:
            x_time = x_time.unsqueeze(1)

        x_time_cond = self.time_embedding(x_time)
        x_cam_tokens = self._encode_cameras(x_img)
        x_cam_cond = rearrange(x_cam_tokens, "b n d -> b (n d)")
        x_state_cond = rearrange(x_state, "b n s -> b (n s)")

        x_cond = torch.cat([x_time_cond, x_cam_cond, x_state_cond], dim=-1)
        x_cond = self.cond_projection_mlp(x_cond)

        x = rearrange(noisy_actions, "b t a -> b a t")
        x = self.init_conv(x)

        residuals = []
        for encoder in self.encoders:
            x = encoder(x, x_cond)
            residuals.append(x)

        x = self.midcoder(x, x_cond)

        for decoder in self.decoders:
            x = decoder(torch.cat([x, residuals.pop()], dim=1), x_cond)

        x_out_action = rearrange(self.final_conv(x), "b a t -> b t a")
        return PolicyOutput(actions=x_out_action)
