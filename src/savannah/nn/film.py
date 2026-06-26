import torch
import torch.nn as nn


class FiLM(nn.Module):
    """Feature-wise Linear Modulation: language-conditioned scale+shift over token sequences."""

    def __init__(self, lang_dim: int, vision_dim: int):
        super().__init__()
        self.gamma_proj = nn.Linear(lang_dim, vision_dim)
        self.beta_proj = nn.Linear(lang_dim, vision_dim)

    def forward(self, vision_tokens: torch.Tensor, lang_embed: torch.Tensor) -> torch.Tensor:
        # vision_tokens: (B, N, D), lang_embed: (B, D_lang)
        gamma = self.gamma_proj(lang_embed).unsqueeze(1)  # (B, 1, D)
        beta = self.beta_proj(lang_embed).unsqueeze(1)    # (B, 1, D)
        return gamma * vision_tokens + beta
