import torch
import torch.nn as nn

from savannah.nn.cross_attention import CrossAttention


class PerceiverResampler(nn.Module):
    """Compresses a variable-length token sequence into a fixed number of
    learned query tokens via cross-attention (Perceiver/Flamingo-resampler style).
    """

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

        self.embed_dim = embed_dim
        self.num_queries = num_queries

        self.latents = nn.Parameter(torch.randn(num_queries, embed_dim) * 0.02)

        self.layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "ln_q": nn.LayerNorm(embed_dim),
                        "ln_kv": nn.LayerNorm(embed_dim),
                        "cross_attn": CrossAttention(embed_dim, num_attn_heads),
                        "ln_ffn": nn.LayerNorm(embed_dim),
                        "ffn": nn.Sequential(
                            nn.Linear(embed_dim, feedforward_dim),
                            nn.GELU(),
                            nn.Linear(feedforward_dim, embed_dim),
                        ),
                    }
                )
                for _ in range(depth)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x (B, N, embed_dim) -> (B, num_queries, embed_dim)
        latents = self.latents.unsqueeze(0).expand(x.shape[0], -1, -1)

        for layer in self.layers:
            q = layer["ln_q"](latents)
            kv = layer["ln_kv"](x)
            latents = latents + layer["cross_attn"](q, kv)
            latents = latents + layer["ffn"](layer["ln_ffn"](latents))

        return latents
