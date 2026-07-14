import torch
import torch.nn as nn

from ..cross_attention import CrossAttention
from ..self_attention import SelfAttention


class TransformerDecoderBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_attn_heads: int,
        feedforward_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.self_attn_block = SelfAttention(embed_dim, num_attn_heads)
        self.cross_attn_block = CrossAttention(embed_dim, num_attn_heads)

        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        self.layer_norm_kv = nn.LayerNorm(embed_dim)
        self.layer_norm3 = nn.LayerNorm(embed_dim)

        self.ffn_block = nn.Sequential(
            nn.Linear(embed_dim, feedforward_dim),
            nn.GELU(),
            nn.Linear(feedforward_dim, embed_dim),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        # Pre Norm + Self Attn of the input embeddings
        x_norm1 = self.layer_norm1(x)
        x_self_attn = self.self_attn_block(x_norm1)

        x = x + x_self_attn

        # Pre Norm + Cross Attn of input and kv embeddings
        x_norm2 = self.layer_norm2(x)
        kv_norm = self.layer_norm_kv(kv)

        x_cross_attn = self.cross_attn_block(x_norm2, kv_norm)
        x = x + x_cross_attn

        # Pre Norm + FFN Block
        x_norm3 = self.layer_norm3(x)
        x_ffn = self.ffn_block(x_norm3)
        x = x + x_ffn

        x = self.dropout(x)

        return x


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        decoder_layers: int,
        embed_dim: int,
        num_attn_heads: int,
        feedforward_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.decoder_blocks = nn.ModuleList(
            [
                TransformerDecoderBlock(
                    embed_dim, num_attn_heads, feedforward_dim, dropout
                )
                for _ in range(decoder_layers)
            ]
        )

    def forward(self, x: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        for block in self.decoder_blocks:
            x = block(x, kv)

        return x
