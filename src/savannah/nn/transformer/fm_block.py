import torch
import torch.nn as nn

from savannah.nn.cross_attention import CrossAttention
from savannah.nn.self_attention import SelfAttention


class FMTransformerBlock(nn.Module):
    """
    Self-attention + cross-attention + FFN block with AdaLN timestep conditioning.

    Action tokens self-attend, then cross-attend to condition tokens, then pass
    through an FFN. All three sub-blocks are independently AdaLN-conditioned on
    the diffusion timestep (9 parameters total: alpha, beta, gamma per sub-block).
    """

    def __init__(
        self,
        embed_dim: int,
        num_attn_heads: int,
        feedforward_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.self_attn_block = SelfAttention(embed_dim, num_attn_heads)
        self.cross_attn_block = CrossAttention(embed_dim, num_attn_heads)
        self.ffn_mlp = nn.Sequential(
            nn.Linear(embed_dim, feedforward_dim),
            nn.GELU(),
            nn.Linear(feedforward_dim, embed_dim),
        )
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        self.layer_norm3 = nn.LayerNorm(embed_dim)
        self.timestep_mlp = nn.Sequential(
            nn.Linear(embed_dim, 9 * embed_dim),
            nn.GELU(),
            nn.Linear(9 * embed_dim, 9 * embed_dim),
        )
        self.dropout = nn.Dropout(dropout)
        nn.init.constant_(self.timestep_mlp[-1].weight, 0.0)
        nn.init.constant_(self.timestep_mlp[-1].bias, 0.0)

    def forward(
        self,
        x: torch.Tensor,
        x_cond: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        # x: (B, T, D), x_cond: (B, N_cond, D), t: (B, 1, D)
        alpha1, beta1, gamma1, alpha2, beta2, gamma2, alpha3, beta3, gamma3 = \
            torch.split(self.timestep_mlp(t), self.embed_dim, dim=-1)

        x = x + alpha1 * self.self_attn_block(beta1 * self.layer_norm1(x) + gamma1)
        x = x + alpha2 * self.cross_attn_block(beta2 * self.layer_norm2(x) + gamma2, x_cond)
        x = x + alpha3 * self.ffn_mlp(beta3 * self.layer_norm3(x) + gamma3)
        return self.dropout(x)
