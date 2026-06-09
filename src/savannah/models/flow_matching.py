from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from savannah.models.policy import Policy
from savannah.models.vision_encoder import VisionEncoder
from savannah.nn import SelfAttention
from savannah.nn.cross_attention import CrossAttention
from savannah.nn.positional_embeddings import SinusoidalPositionalEncoding
from savannah.nn.time_embedding import TimeEmbedding
from savannah.objectives import PolicyObjective
from savannah.utils.device import get_device
from savannah.utils.observation import ObservationKey
from savannah.utils.policy import PolicyOutput


class FMTransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_attn_heads, feedforward_dim, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim

        # Self Attn
        self.self_attn_block = SelfAttention(embed_dim, num_attn_heads)

        # Cross Attn
        self.cross_attn_block = CrossAttention(embed_dim, num_attn_heads)

        # FFN
        self.ffn_mlp = nn.Sequential(
            nn.Linear(embed_dim, feedforward_dim),
            nn.GELU(),
            nn.Linear(feedforward_dim, embed_dim),
        )

        # Three norms — one per sub-block
        self.layer_norm1 = nn.LayerNorm(embed_dim)  # before self attn
        self.layer_norm2 = nn.LayerNorm(embed_dim)  # before cross attn
        self.layer_norm3 = nn.LayerNorm(embed_dim)  # before ffn

        # AdaLN condition MLP — need three parameters for every block.
        # In total, 9 params
        self.timestep_mlp = nn.Sequential(
            nn.Linear(embed_dim, 9 * embed_dim),
            nn.GELU(),
            nn.Linear(9 * embed_dim, 9 * embed_dim),
        )

        self.dropout = nn.Dropout(dropout)

        self._initialise_zero_weights()

    def _initialise_zero_weights(self):
        nn.init.constant_(self.timestep_mlp[-1].weight, 0.0)
        nn.init.constant_(self.timestep_mlp[-1].bias, 0.0)

    def forward(self, x, x_cond, t):
        # x:      (B, action_horizon, embed_dim)
        # x_cond: (B, obs_tokens, embed_dim)
        # t:      (B, 1, embed_dim)

        # 4 AdaLN params — only self attn and ffn are time conditioned
        timestep = self.timestep_mlp(t)
        (
            alpha1,
            beta1,
            gamma1,
            alpha2,
            beta2,
            gamma2,
            alpha3,
            beta3,
            gamma3,
        ) = torch.split(timestep, self.embed_dim, dim=-1)

        # Self attention — action tokens attend to each other, time conditioned
        x1 = self.layer_norm1(x)
        x1 = beta1 * x1 + gamma1
        x1 = self.self_attn_block(x1)
        x1 = alpha1 * x1
        x = x + x1

        # Cross attention — action tokens attend to obs context, no time conditioning
        x2 = self.layer_norm2(x)
        x2 = beta2 * x2 + gamma2
        x2 = self.cross_attn_block(x2, x_cond)
        x2 = alpha2 * x2
        x = x + x2

        # FFN — time conditioned
        x3 = self.layer_norm3(x)
        x3 = beta3 * x3 + gamma3
        x3 = self.ffn_mlp(x3)
        x3 = alpha3 * x3
        x = x + x3

        x = self.dropout(x)
        return x


class AdaLN(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim, elementwise_affine=False)
        self.embed_dim = embed_dim
        # MLP that turns the time embedding into 3 conditioning parameters
        self.mlp = nn.Sequential(nn.GELU(), nn.Linear(embed_dim, 3 * embed_dim))

        self._initialise_zero_weights()

    def _initialise_zero_weights(self):
        nn.init.constant_(self.mlp[-1].weight, 0.0)
        nn.init.constant_(self.mlp[-1].bias, 0.0)

    def forward(self, x: torch.Tensor, t_embed: torch.Tensor) -> torch.Tensor:
        # t_embed comes from your TimeEmbedding(x_time)
        # x shape: (B, T, D), t_embed shape: (B, 1, D)

        cond = self.mlp(t_embed)  # (B, 1, 3*D)
        alpha, beta, gamma = torch.split(cond, self.embed_dim, dim=-1)

        x = self.norm(x)
        x = x * (1 + gamma) + beta  # Scale and shift
        x = x * alpha  # Gate the output
        return x


class FlowMatchingPolicy(Policy):
    def __init__(
        self,
        embed_dim: int,
        num_attn_heads: int,
        feedforward_dim: int,
        num_blocks: int,
        state_dim: int,
        action_dim: int,
        action_horizon: int,
        vision_encoder: VisionEncoder,
        objective: PolicyObjective,
        num_cameras: int = 1,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_attn_heads = num_attn_heads
        self.feedforward_dim = feedforward_dim

        self.num_blocks = num_blocks

        self.vision_encoder = vision_encoder
        self.num_cameras = num_cameras

        self._state_dim = state_dim
        self._action_dim = action_dim
        self._action_horizon = action_horizon

        self.objective = objective

        # TimeEmbedding (This is generic, no learnable parameters, can be used at all places, timestep is needed)
        self.time_embedding = TimeEmbedding(self.embed_dim)

        # Camera Embedding
        self.camera_embedding = nn.Embedding(num_cameras, self.embed_dim)

        # State embedding
        self.state_embedding = nn.Sequential(
            nn.Linear(self.state_dim, self.embed_dim),
            nn.GELU(),
            nn.Linear(self.embed_dim, self.embed_dim),
        )

        # Action Embedding
        self.action_embedding = nn.Sequential(
            nn.Linear(self.action_dim, self.embed_dim),
            nn.GELU(),
            nn.Linear(self.embed_dim, self.embed_dim),
        )

        # Action reprojection
        self.action_reprojection = nn.Sequential(
            nn.Linear(self.embed_dim, self.action_dim),
            nn.GELU(),
            nn.Linear(self.action_dim, self.action_dim),
        )

        # FMBlocks
        self.fm_blocks = nn.ModuleList(
            [
                FMTransformerBlock(embed_dim, num_attn_heads, feedforward_dim)
                for _ in range(num_blocks)
            ]
        )

        # Layer Norm
        self.layer_norm = AdaLN(embed_dim)

    def forward(self, obs: dict[str, torch.Tensor], *args, **kwargs) -> PolicyOutput:
        x_img = obs[ObservationKey.images]  # list[(B, M, 3, H, W)]
        x_state = obs[ObservationKey.state]  # (B, N, state_dim)
        x_time = obs[ObservationKey.time]  # (B,) # This is between 0 -> 1

        # Get noisy actions from kwargs
        noisy_actions = kwargs.get("noisy_actions", None)
        if noisy_actions is None:
            raise AssertionError("Pass noisy action for the model to run")

        # (B, T, embed_dim)
        x_cam_tokens = self._encode_cameras(x_img)

        # (B, N, state_dim) -> (B, N, embedding_dim)
        x_state = self.state_embedding(x_state)
        # Add time embeddings for states also
        x_state_time_embed = self.time_embedding(
            torch.arange(x_state.shape[1], device=x_state.device).unsqueeze(1)
        )

        x_state = x_state + x_state_time_embed

        # (B,) -> (B, 1) optionally
        if len(x_time.shape) == 1:
            x_time = x_time.unsqueeze(1)

        # (B, 1) -> (B, 1, embed_dim)
        x_time = self.time_embedding(x_time).unsqueeze(1)

        # Project the noisy actions to embeding space
        # (B, N_obs, action_dim) -> (B, N_obs, embed_dim)
        x_noisy_actions = self.action_embedding(noisy_actions)
        # Add time embeddings for noisy_actions
        x_noisy_actions_time_embed = self.time_embedding(
            torch.arange(
                x_noisy_actions.shape[1], device=x_noisy_actions.device
            ).unsqueeze(1)
        )

        x_noisy_actions = x_noisy_actions + x_noisy_actions_time_embed

        # Concatenate image tokens and proprio tokens
        # x_input_tokens = torch.cat([x_cam_tokens, x_state, x_noisy_actions], dim=1)
        x_cond_tokens = torch.cat([x_cam_tokens, x_state], dim=1)
        x_out = x_noisy_actions
        for block in self.fm_blocks:
            # (B, T, n_embed) -> (B, T, n_embed)
            x_out = block(x_out, x_cond_tokens, x_time)

        # (B, T, embed_dim) -> (B, T, embed_dim)
        x_out = self.layer_norm(x_out, x_time)

        # T = N_obs
        # (B, T, embed_dim) -> (B, N_obs, embed_dim)
        x_out_action = x_out

        # (B, N_obs, embed_dim) -> (B, N_obs, action_dim)
        x_out_action = self.action_reprojection(x_out_action)

        output = PolicyOutput(actions=x_out_action)

        return output


if __name__ == "__main__":
    from savannah.models.backbones import DummyVisionBackbone

    embed_dim = 128
    num_attn_heads = 4
    num_blocks = 6
    feedforward_dim = 4 * embed_dim
    num_cameras = 3

    state_dim = 6
    action_dim = 6
    action_horizon = 12

    device = get_device()

    feature_extractor = DummyVisionBackbone(downsample_factor=16).to(device)
    vision_encoder = VisionEncoder(backbone=feature_extractor, embed_dim=embed_dim).to(
        device
    )

    fm = FlowMatchingPolicy(
        embed_dim,
        num_attn_heads,
        feedforward_dim,
        num_blocks,
        state_dim,
        action_dim,
        action_horizon,
        vision_encoder,
        num_cameras,
    ).to(device)

    obs = {}
    obs[ObservationKey.images] = [torch.randn(8, 3, 3, 224, 224).to(device)]
    obs[ObservationKey.state] = torch.randn(8, 3, state_dim).to(device)
    obs[ObservationKey.gt_actions] = torch.randn(8, action_horizon, action_dim).to(
        device
    )

    # noisy_action = torch.randn((8, action_horizon, action_dim)).to(device)
    print(fm.num_params())

    out = fm.compute_action(obs)

    print(out.actions.shape)

    loss = fm.compute_loss(obs)

    print(loss)
