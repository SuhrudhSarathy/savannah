import torch
import torch.nn as nn

from .self_attention import SelfAttention


class AdaLN(nn.Module):
    """Adaptive Layer Norm conditioned on a timestep embedding."""

    def __init__(self, embed_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim, elementwise_affine=False)
        self.embed_dim = embed_dim
        self.mlp = nn.Sequential(nn.GELU(), nn.Linear(embed_dim, 3 * embed_dim))
        nn.init.constant_(self.mlp[-1].weight, 0.0)
        nn.init.constant_(self.mlp[-1].bias, 0.0)

    def forward(self, x: torch.Tensor, t_embed: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D), t_embed: (B, 1, D)
        cond = self.mlp(t_embed)
        alpha, beta, gamma = torch.split(cond, self.embed_dim, dim=-1)
        x = self.norm(x) * (1 + gamma) + beta
        return x + x * alpha


class AdaLNDecoderBlock(nn.Module):
    """
    Transformer decoder block conditioned via Adaptive Layer Norm.

    kv input is cat(mean(encoder_output_i), timestep_embed) of shape (B, 1, 2*embed_dim).
    Both the self-attention and FFN sub-blocks are independently scaled and shifted.
    """

    def __init__(
        self,
        embed_dim: int,
        num_attn_heads: int,
        feedforward_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self._embed_dim = embed_dim
        self.attn_block = SelfAttention(embed_dim, num_attn_heads)
        self.ffn_block = nn.Sequential(
            nn.Linear(embed_dim, feedforward_dim),
            nn.GELU(),
            nn.Linear(feedforward_dim, embed_dim),
        )
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        self.dropout_layer = nn.Dropout(dropout)
        self.adaln_block = nn.Linear(2 * embed_dim, 6 * embed_dim)
        nn.init.constant_(self.adaln_block.weight, 0.0)
        nn.init.constant_(self.adaln_block.bias, 0.0)

    def forward(self, x: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        beta1, gamma1, alpha1, beta2, gamma2, alpha2 = torch.split(
            self.adaln_block(kv), self._embed_dim, dim=-1
        )
        x = x + alpha1 * self.attn_block(gamma1 * self.layer_norm1(x) + beta1)
        x = x + alpha2 * self.ffn_block(gamma2 * self.layer_norm2(x) + beta2)
        return self.dropout_layer(x)
