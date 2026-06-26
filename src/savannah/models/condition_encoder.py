from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn as nn
from einops import rearrange

from savannah.models.vision_encoder import VisionEncoder
from savannah.nn.time_embedding import TimeEmbedding


class LanguageEncoder(ABC, nn.Module):
    """
    Abstract base for any module that maps tokenized text to a language embedding.

    Concrete implementations might wrap CLIP's text encoder, a frozen BERT, etc.
    The forward method must return a single vector per batch element, not a token sequence,
    since it is used as the conditioning signal for FiLM modulation over vision tokens.
    """

    @property
    @abstractmethod
    def output_dim(self) -> int:
        """Dimensionality of the language embedding returned by forward."""
        ...

    @abstractmethod
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        tokens: (B, N_text) — tokenized text (e.g. from a tokenizer)
        returns: (B, output_dim) — language embedding per sample
        """
        ...


class ConditionEncoder(nn.Module):
    """
    Encodes multi-camera images and proprioceptive state into condition tokens.

    Optionally applies FiLM language conditioning to vision tokens when a
    language_encoder and film module are provided. Both must be provided together.

    forward() returns (B, N_cond, embed_dim) where:
        N_cond = sum_over_cameras(N_obs * N_tokens_per_frame) + N_obs_state
    """

    def __init__(
        self,
        vision_encoder: VisionEncoder,
        embed_dim: int,
        state_dim: int,
        num_cameras: int,
        language_encoder: Optional[LanguageEncoder] = None,
        film: Optional[nn.Module] = None,
    ):
        super().__init__()

        if (language_encoder is None) != (film is None):
            raise ValueError("Provide both language_encoder and film, or neither.")

        self.vision_encoder = vision_encoder
        self._embed_dim = embed_dim
        self.num_cameras = num_cameras

        self.camera_embedding = nn.Embedding(num_cameras, embed_dim)
        self.state_embedding = nn.Sequential(
            nn.Linear(state_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.time_embedding = TimeEmbedding(embed_dim)

        self.language_encoder = language_encoder
        self.film = film

    @property
    def output_dim(self) -> int:
        return self._embed_dim

    def forward(
        self,
        images: list[torch.Tensor],
        state: torch.Tensor,
        language: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        images:   list of (B, N_obs, 3, H, W), one tensor per camera
        state:    (B, N_obs, state_dim)
        language: (B, N_text) tokenized text, or None
        returns:  (B, N_cond, embed_dim)
        """
        vis_tokens = self._encode_cameras(images)  # (B, N_vis, embed_dim)

        if self.language_encoder is not None and language is not None:
            lang_embed = self.language_encoder(language)   # (B, D_lang)
            vis_tokens = self.film(vis_tokens, lang_embed)

        state_tokens = self._encode_state(state)           # (B, N_obs, embed_dim)

        return torch.cat([vis_tokens, state_tokens], dim=1)

    def _encode_cameras(self, images: list[torch.Tensor]) -> torch.Tensor:
        cam_tokens = []
        for cam_idx, img in enumerate(images):
            n = img.shape[1]
            img_r = rearrange(img, "b n c h w -> (b n) c h w")
            tokens = self.vision_encoder(img_r)  # (B*N, N_tok, embed_dim)
            tokens_r = rearrange(tokens, "(b n) t d -> b n t d", n=n)

            t = torch.arange(n, device=img.device).unsqueeze(1)  # (N, 1)
            time_emb = self.time_embedding(t).view(1, n, 1, -1)

            cam_emb = self.camera_embedding(
                torch.tensor(cam_idx, device=img.device)
            ).view(1, 1, 1, -1)

            tokens_r = tokens_r + time_emb + cam_emb
            tokens = rearrange(tokens_r, "b n t d -> b (n t) d")
            cam_tokens.append(tokens)

        return torch.cat(cam_tokens, dim=1)

    def _encode_state(self, state: torch.Tensor) -> torch.Tensor:
        # state: (B, N_obs, state_dim)
        x = self.state_embedding(state)  # (B, N_obs, embed_dim)
        N = x.shape[1]
        t = torch.arange(N, device=state.device).unsqueeze(1)  # (N, 1)
        pos_emb = self.time_embedding(t).unsqueeze(0)           # (1, N, embed_dim)
        return x + pos_emb
