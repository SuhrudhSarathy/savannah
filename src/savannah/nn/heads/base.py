from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class ActionHead(ABC, nn.Module):
    """
    Abstract base for action denoising heads.

    All heads receive the same three inputs and differ only in how they process them:
      - DiTActionHead:            re-encodes cond_tokens internally via SA, uses per-layer AdaLN
      - CrossAttentionActionHead: cross-attends to cond_tokens directly, AdaLN from timestep
      - UNetActionHead:           mean-pools cond_tokens into a FiLM conditioning vector
    """

    @property
    @abstractmethod
    def embed_dim(self) -> int: ...

    @abstractmethod
    def forward(
        self,
        noisy_actions: torch.Tensor,   # (B, T, action_dim)
        cond_tokens: torch.Tensor,     # (B, N_cond, embed_dim)
        timestep_embed: torch.Tensor,  # (B, 1, embed_dim)
    ) -> torch.Tensor: ...             # (B, T, action_dim)
