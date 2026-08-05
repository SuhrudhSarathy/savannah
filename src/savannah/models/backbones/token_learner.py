import torch
import torch.nn as nn


class TokenLearner(nn.Module):
    def __init__(self, embed_dim: int, output_dim: int, hidden: int = 64):
        super().__init__()
        self.attn_maps = nn.Sequential(
            nn.Linear(embed_dim, hidden), nn.GELU(), nn.Linear(hidden, output_dim)
        )

        # Initialise the module with zero so that it defaults to average pooling
        nn.init.trunc_normal_(self.attn_maps[-1].weight, std=0.02)
        nn.init.trunc_normal_(self.attn_maps[-1].bias, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x (B, n_t, embed_dim)
        weights_map = self.attn_maps(x)  # (B, n_t, hidden_dim)
        weights = weights_map.softmax(dim=1)

        return torch.einsum("bte,btf->bfe", x, weights)
