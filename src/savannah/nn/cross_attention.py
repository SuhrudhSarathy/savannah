from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class CrossAttention(nn.Module):
    def __init__(self, embed_dim: int, num_attn_heads: int, use_sdpa: bool = True):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = num_attn_heads
        self.use_sdpa = use_sdpa

        self.q_projection = nn.Linear(self.embed_dim, self.embed_dim)
        self.kv_projection = nn.Linear(self.embed_dim, 2 * self.embed_dim)

        self.reproj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(self, x: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        # x (B, T, n_embed), kv (B, T, n_embed)
        q = self.q_projection(x)

        # (B, T, n_embed) -> (B, T, 2 * n_embed)
        kv_projection = self.kv_projection(kv)

        # (B, T, 2* n_embed) -> 2 * (B, T, n_embed)
        k, v = torch.split(kv_projection, self.embed_dim, dim=-1)

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

        if self.use_sdpa:
            attn = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        else:
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
