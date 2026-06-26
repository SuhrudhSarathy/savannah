import torch
import torch.nn as nn
from einops import rearrange

from savannah.models.heads.base import ActionHead
from savannah.nn.unet_blocks import Decoder, Encoder, MidCoder


class UNetActionHead(ActionHead):
    """
    1D UNet action head with FiLM conditioning.

    Condition tokens are mean-pooled and concatenated with the timestep embedding
    to form a flat conditioning vector (B, 2*embed_dim) passed to each residual
    block via FiLM. This is objective-agnostic and independent of N_cond.
    """

    def __init__(
        self,
        embed_dim: int,
        unet_channels: list[int],
        num_residual_layers: int,
        action_dim: int,
        action_horizon: int,
    ):
        super().__init__()
        self._embed_dim = embed_dim
        condition_dim = 2 * embed_dim  # pooled cond_tokens ‖ timestep_embed

        self.cond_mlp = nn.Sequential(
            nn.Linear(condition_dim, 4 * embed_dim),
            nn.Mish(),
            nn.Linear(4 * embed_dim, condition_dim),
        )
        self.init_conv = nn.Sequential(
            nn.Conv1d(action_dim, unet_channels[0], kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=unet_channels[0]),
            nn.Mish(),
        )

        encoders, decoders = [], []
        for i in range(len(unet_channels) - 1):
            encoders.append(Encoder(
                num_residual_layers=num_residual_layers,
                in_channels=unet_channels[i],
                cond_channels=condition_dim,
                out_channels=unet_channels[i + 1],
            ))
            decoders.append(Decoder(
                num_residual_layers=num_residual_layers,
                in_channels=2 * unet_channels[i + 1],
                cond_channels=condition_dim,
                out_channels=unet_channels[i],
            ))

        self.encoders = nn.ModuleList(encoders)
        self.decoders = nn.ModuleList(list(reversed(decoders)))
        self.midcoder = MidCoder(
            num_residual_layers=num_residual_layers,
            channels=unet_channels[-1],
            cond_channels=condition_dim,
        )
        self.final_conv = nn.Conv1d(unet_channels[0], action_dim, kernel_size=3, padding=1)

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    def forward(self, noisy_actions, cond_tokens, timestep_embed):
        cond_pooled = cond_tokens.mean(dim=1)   # (B, D)
        t_flat = timestep_embed.squeeze(1)       # (B, D)
        cond_vec = self.cond_mlp(torch.cat([t_flat, cond_pooled], dim=-1))  # (B, 2*D)

        x = rearrange(noisy_actions, "b t a -> b a t")
        x = self.init_conv(x)

        residuals = []
        for encoder in self.encoders:
            x = encoder(x, cond_vec)
            residuals.append(x)

        x = self.midcoder(x, cond_vec)

        for decoder in self.decoders:
            x = decoder(torch.cat([x, residuals.pop()], dim=1), cond_vec)

        return rearrange(self.final_conv(x), "b a t -> b t a")
