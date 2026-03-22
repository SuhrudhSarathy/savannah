import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from math import sqrt


class SelfAttention(nn.Module):
    def __init__(self, embed_dim: int, num_attn_heads: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = num_attn_heads

        self.attn_projection = nn.Linear(self.embed_dim, 3 * self.embed_dim)
        self.reproj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x (B, T, n_embed)
        #
        # (B, T, n_embed) -> (B, T, 3 * n_embed)
        attn_projection_x = self.attn_projection(x)

        # (B, T, 3 * n_embed) -> 3 * (B, T, n_embed)
        q, k, v = torch.split(attn_projection_x, self.embed_dim, dim=-1)

        # Multihead
        q = rearrange(
            q,
            "b t (nh hs) -> b nh t hs",
            nh=self.n_heads,
            hs=self.embed_dim // self.n_heads,
        )
        k = rearrange(
            k,
            "b t (nh hs) -> b nh t hs",
            nh=self.n_heads,
            hs=self.embed_dim // self.n_heads,
        )
        v = rearrange(
            v,
            "b t (nh hs) -> b nh t hs",
            nh=self.n_heads,
            hs=self.embed_dim // self.n_heads,
        )

        k_T = rearrange(k, "b nh t hs -> b nh hs t")

        # Attention
        # (b, nh, t, hs) @ (b, nh, hs, t) -> (b, nh, t, t)
        attn = q @ k_T / sqrt(k.shape[-1])
        attn = F.softmax(attn, dim=-1)
        # (b, nh, t, t) @ (b, nh, t, hs) -> (b, nh, t, hs)
        attn = attn @ v

        # Rearrrange
        attn = rearrange(
            attn,
            "b nh t hs -> b t (nh hs)",
            nh=self.n_heads,
            hs=self.embed_dim // self.n_heads,
        )

        attn = self.reproj(attn)

        return attn
