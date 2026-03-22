import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from math import sqrt

from ..self_attention import SelfAttention


class TransformerEncoderBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_attn_heads: int,
        feedforward_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.attn_block = SelfAttention(embed_dim, num_attn_heads)

        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)

        self.ffn_block = nn.Sequential(
            nn.Linear(embed_dim, feedforward_dim),
            nn.GELU(),
            nn.Linear(feedforward_dim, embed_dim),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Use Pre norm and then Attn
        x_norm = self.layer_norm1(x)
        x_attn = self.attn_block(x_norm)
        x = x + x_attn

        # Use pre norm and then FFN
        x_norm2 = self.layer_norm2(x)
        x_ffn = self.ffn_block(x_norm2)
        x = x + x_ffn

        x = self.dropout(x)

        return x


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        encoder_layers: int,
        embed_dim: int,
        num_attn_heads: int,
        feedforward_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.encoder_blocks = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    embed_dim, num_attn_heads, feedforward_dim, dropout
                )
                for _ in range(encoder_layers)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.encoder_blocks:
            x = block(x)

        return x
