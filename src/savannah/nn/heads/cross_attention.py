import torch
import torch.nn as nn

from savannah.models.heads.base import ActionHead
from savannah.nn.adaln import AdaLN
from savannah.nn.time_embedding import TimeEmbedding
from savannah.nn.transformer.fm_block import FMTransformerBlock


class CrossAttentionActionHead(ActionHead):
    """
    Cross-attention action head compatible with flow matching and DDIM objectives.

    Action tokens self-attend, then cross-attend to condition tokens (used directly
    as K/V), then pass through an FFN. All sub-blocks are AdaLN-conditioned on the
    diffusion timestep. A final AdaLN layer is applied before action reprojection.

    Because condition tokens feed cross-attention directly without re-encoding,
    any pretrained contextual structure from the backbone is preserved.
    """

    def __init__(
        self,
        embed_dim: int,
        num_blocks: int,
        num_attn_heads: int,
        feedforward_dim: int,
        action_dim: int,
        action_horizon: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self._embed_dim = embed_dim

        self.fm_blocks = nn.ModuleList([
            FMTransformerBlock(embed_dim, num_attn_heads, feedforward_dim, dropout)
            for _ in range(num_blocks)
        ])
        self.final_norm = AdaLN(embed_dim)
        self.action_embedding = nn.Sequential(
            nn.Linear(action_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self._time_embedding = TimeEmbedding(embed_dim)
        self.action_reprojection = nn.Linear(embed_dim, action_dim)

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    def forward(self, noisy_actions, cond_tokens, timestep_embed):
        x = self.action_embedding(noisy_actions)
        T = x.shape[1]
        pos = self._time_embedding(
            torch.arange(T, device=x.device).unsqueeze(1)
        ).unsqueeze(0)  # (1, T, D)
        x = x + pos

        for block in self.fm_blocks:
            x = block(x, cond_tokens, timestep_embed)

        x = self.final_norm(x, timestep_embed)
        return self.action_reprojection(x)
