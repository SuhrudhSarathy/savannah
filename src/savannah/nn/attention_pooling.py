from diffusers.models.normalization import LayerNorm
import torch
import torch.nn as nn

from savannah.nn.cross_attention import CrossAttention


class AttentionPoolingBlock(nn.Module):
    def __init__(self, embed_dim: int, num_attn_heads: int, feedforward_dim: int):
        super().__init__()

        self.embed_dim: int = embed_dim
        self.num_attn_heads: int = num_attn_heads
        self.feedforward_dim: int = feedforward_dim

        self.attn = CrossAttention(embed_dim, num_attn_heads)
        self.pre_norm = LayerNorm(embed_dim)
        self.kv_norm = LayerNorm(embed_dim)
        self.ffn_norm = LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, feedforward_dim),
            nn.GELU(),
            nn.Linear(feedforward_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor, x_kv: torch.Tensor) -> torch.Tensor:
        q = self.pre_norm(x)
        kv = self.kv_norm(x_kv)

        x = x + self.attn(q, kv)

        out = self.ffn_norm(x)
        out = self.ffn(out)

        x = x + out
        return x


class AttentionPooling(nn.Module):
    """Attention Pooling"""

    def __init__(
        self,
        embed_dim: int,
        num_queries: int,
        num_attn_heads: int,
        depth: int = 1,
        feedforward_dim: int | None = None,
    ):
        super().__init__()
        feedforward_dim = feedforward_dim or 4 * embed_dim

        self.embed_dim: int = embed_dim
        self.num_queries: int = num_queries

        self.latents = nn.Parameter(torch.randn(num_queries, embed_dim) * 0.02)

        self.layers = nn.ModuleList(
            [
                AttentionPoolingBlock(embed_dim, num_attn_heads, feedforward_dim)
                for _ in range(depth)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x (B, N, embed_dim) -> (B, num_queries, embed_dim)
        latents = self.latents.unsqueeze(0).expand(x.shape[0], -1, -1)

        for layer in self.layers:
            latents = layer(latents, x)

        return latents
