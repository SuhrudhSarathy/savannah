import warnings

import torch
import torch.nn as nn

from savannah.models.encoders import LanguageEncoder, StateEncoder, VisionEncoder
from savannah.models.encoders.language_encoder import LanguageEncoder
from savannah.models.policy import Policy
from savannah.nn import SelfAttention
from savannah.nn.cross_attention import CrossAttention
from savannah.nn.time_embedding import TimeEmbedding
from savannah.objectives import PolicyObjective
from savannah.utils.log import logger
from savannah.utils.observation import ObservationKey
from savannah.utils.policy import PolicyOutput


class DiTCrossAttnBlock(nn.Module):
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

        DiTCrossAttnPolicy._debug_stat("block.x_in", x)
        DiTCrossAttnPolicy._debug_stat("block.x_cond", x_cond)
        DiTCrossAttnPolicy._debug_stat("block.t", t)

        # 4 AdaLN params — only self attn and ffn are time conditioned
        timestep = self.timestep_mlp(t)
        DiTCrossAttnPolicy._debug_stat("block.timestep_mlp_out", timestep)
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
        DiTCrossAttnPolicy._debug_stat("block.x1_layer_norm1", x1)
        x1 = beta1 * x1 + gamma1
        DiTCrossAttnPolicy._debug_stat("block.x1_adaln", x1)
        x1 = self.self_attn_block(x1)
        DiTCrossAttnPolicy._debug_stat("block.x1_self_attn", x1)
        x1 = alpha1 * x1
        DiTCrossAttnPolicy._debug_stat("block.x1_scaled", x1)
        x = x + x1
        DiTCrossAttnPolicy._debug_stat("block.x_after_self_attn", x)

        # Cross attention — action tokens attend to obs context, no time conditioning
        x2 = self.layer_norm2(x)
        DiTCrossAttnPolicy._debug_stat("block.x2_layer_norm2", x2)
        x2 = beta2 * x2 + gamma2
        DiTCrossAttnPolicy._debug_stat("block.x2_adaln", x2)
        x2 = self.cross_attn_block(x2, x_cond)
        DiTCrossAttnPolicy._debug_stat("block.x2_cross_attn", x2)
        x2 = alpha2 * x2
        DiTCrossAttnPolicy._debug_stat("block.x2_scaled", x2)
        x = x + x2
        DiTCrossAttnPolicy._debug_stat("block.x_after_cross_attn", x)

        # FFN — time conditioned
        x3 = self.layer_norm3(x)
        DiTCrossAttnPolicy._debug_stat("block.x3_layer_norm3", x3)
        x3 = beta3 * x3 + gamma3
        DiTCrossAttnPolicy._debug_stat("block.x3_adaln", x3)
        x3 = self.ffn_mlp(x3)
        DiTCrossAttnPolicy._debug_stat("block.x3_ffn", x3)
        x3 = alpha3 * x3
        DiTCrossAttnPolicy._debug_stat("block.x3_scaled", x3)
        x = x + x3
        DiTCrossAttnPolicy._debug_stat("block.x_after_ffn", x)

        x = self.dropout(x)
        DiTCrossAttnPolicy._debug_stat("block.x_out", x)
        return x


class DiTCrossAttnPolicy(Policy):
    def __init__(
        self,
        embed_dim: int,
        num_attn_heads: int,
        feedforward_dim: int,
        num_blocks: int,
        state_dim: int,
        action_dim: int,
        action_horizon: int,
        num_cameras: int,
        vision_encoder: VisionEncoder,
        language_encoder: LanguageEncoder | None,
        objective: PolicyObjective,
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

        self.state_encoder = StateEncoder(self._state_dim, self.embed_dim)
        if language_encoder is not None:
            self.language_encoder = language_encoder

        # Action Embedding
        self.action_embedding = nn.Sequential(
            nn.Linear(self.action_dim, self.embed_dim),
            nn.GELU(),
            nn.Linear(self.embed_dim, self.embed_dim),
        )

        # Action reprojection
        self.action_reprojection = nn.Linear(self.embed_dim, self.action_dim)

        # FMBlocks
        self.dit_blocks = nn.ModuleList(
            [
                DiTCrossAttnBlock(embed_dim, num_attn_heads, feedforward_dim)
                for _ in range(num_blocks)
            ]
        )

    @staticmethod
    def _debug_stat(name: str, t: torch.Tensor) -> None:
        logger.debug(
            "{}: shape={}, min={:.6f}, max={:.6f}, mean={:.6f}, nan={}, inf={}",
            name,
            tuple(t.shape),
            t.min().item(),
            t.max().item(),
            t.mean().item(),
            torch.isnan(t).any().item(),
            torch.isinf(t).any().item(),
        )

    def forward(self, obs: dict[str, torch.Tensor], *args, **kwargs) -> PolicyOutput:
        x_img = obs[ObservationKey.images]  # list[(B, M, 3, H, W)]
        x_state = obs[ObservationKey.state]  # (B, N, state_dim)
        x_time = obs[ObservationKey.time]  # (B,) # This is between 0 -> 1
        x_language = obs.get(ObservationKey.language, None)

        # Get noisy actions from kwargs
        noisy_actions = kwargs.get("noisy_actions", None)
        if noisy_actions is None:
            raise AssertionError("Pass noisy action for the model to run")

        # (B, T, embed_dim)
        x_cam_tokens = self.vision_encoder(x_img)
        self._debug_stat("x_cam_tokens", x_cam_tokens)

        # (B, N, state_dim) -> (B, N, embedding_dim)
        x_state = self.state_encoder(x_state)
        self._debug_stat("x_state", x_state)

        if x_language is not None:
            if self.language_encoder is None:
                with warnings.catch_warnings():
                    warnings.simplefilter("once")
                    warnings.warn(
                        "Language instruction is provided but LanguageEncoder is not specified. Not using language instruction",
                        RuntimeWarning,
                    )
            else:
                # (B, xx, embed_dim)
                x_language = self.language_encoder(x_language)
                self._debug_stat("x_lang", x_language)

        # (B,) -> (B, 1) optionally
        if len(x_time.shape) == 1:
            x_time = x_time.unsqueeze(1)

        # (B, 1) -> (B, 1, embed_dim)
        x_time = self.time_embedding(x_time).unsqueeze(1)
        self._debug_stat("x_time", x_time)

        # Project the noisy actions to embeding space
        # (B, N_obs, action_dim) -> (B, N_obs, embed_dim)
        x_noisy_actions = self.action_embedding(noisy_actions)
        self._debug_stat("x_noisy_actions_embed", x_noisy_actions)
        # Add time embeddings for noisy_actions
        x_noisy_actions_time_embed = self.time_embedding(
            torch.arange(
                x_noisy_actions.shape[1], device=x_noisy_actions.device
            ).unsqueeze(1)
        )
        self._debug_stat("x_noisy_actions_time_embed", x_noisy_actions_time_embed)

        x_noisy_actions = x_noisy_actions + x_noisy_actions_time_embed
        self._debug_stat("x_noisy_actions", x_noisy_actions)

        # Concatenate image tokens and proprio tokens
        if x_language is None:
            x_cond_tokens = torch.cat([x_cam_tokens, x_state], dim=1)
        else:
            x_cond_tokens = torch.cat([x_language, x_cam_tokens, x_state], dim=1)
        self._debug_stat("x_cond_tokens", x_cond_tokens)

        x_out = x_noisy_actions
        for i, block in enumerate(self.dit_blocks):
            # (B, T, n_embed) -> (B, T, n_embed)
            x_out = block(x_out, x_cond_tokens, x_time)
            self._debug_stat(f"x_out_block_{i}", x_out)

        # (B, N_obs, embed_dim) -> (B, N_obs, action_dim)
        x_out_action = self.action_reprojection(x_out)
        self._debug_stat("x_out_action", x_out_action)

        output = PolicyOutput(actions=x_out_action)

        return output
