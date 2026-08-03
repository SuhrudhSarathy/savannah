from math import sqrt, log

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class RotatoryPositionalEncoding(nn.Module):
    def __init__(self, head_dim: int, max_seq_len: int):
        super().__init__()
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len

        div_term = torch.exp(
            torch.arange(0, head_dim, 2).float() * (-log(10_000) / head_dim)
        )

        self.register_buffer("div_term", div_term)

        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len).to(
            dtype=self.div_term.dtype, device=self.div_term.device
        )
        freqs = torch.outer(t, self.div_term)

        emb = torch.cat([freqs, freqs], dim=-1)

        self.register_buffer("cos_cached", emb.cos()[None, None, :, :])
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :])

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., : self.head_dim // 2]
        x2 = x[..., self.head_dim // 2 :]

        return torch.cat([-x2, x1], dim=-1)

    def forward(self, q: torch.Tensor, k: torch.Tensor, seq_dim: int = 2):
        seq_len = q.shape[seq_dim]

        cos = self.cos_cached[:, :, :seq_len, :]
        sin = self.sin_cached[:, :, :seq_len, :]

        q_embed = (q * cos) + (self._rotate_half(q) * sin)
        k_embed = (k * cos) + (self._rotate_half(k) * sin)

        return q_embed, k_embed


class RoPESelfAttention(nn.Module):
    def __init__(self, embed_dim: int, num_attn_heads: int, use_sdpa: bool = True):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = num_attn_heads
        self.use_sdpa = use_sdpa

        self.attn_projection = nn.Linear(self.embed_dim, 3 * self.embed_dim)
        self.reproj = nn.Linear(self.embed_dim, self.embed_dim)

        self.rope = RotatoryPositionalEncoding(self.embed_dim // self.n_heads, 2048)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
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

        # Apply Rope
        q, k = self.rope(q, k)

        if self.use_sdpa:
            attn = F.scaled_dot_product_attention(
                q, k, v, is_causal=False, attn_mask=mask
            )

        else:
            k_T = rearrange(k, "b nh t hs -> b nh hs t")

            # Attention
            # (b, nh, t, hs) @ (b, nh, hs, t) -> (b, nh, t, t)
            attn = q @ k_T / sqrt(k.shape[-1])

            if mask is not None:
                attn = attn.masked_fill(~mask, -1e9)

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
