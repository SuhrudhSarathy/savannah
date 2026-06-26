import torch
import torch.nn as nn

from savannah.models.heads.base import ActionHead
from savannah.nn.adaln import AdaLNDecoderBlock
from savannah.nn.positional_embeddings import get_sinusoidal_position_embedding
from savannah.nn.transformer.encoder import TransformerEncoderBlock


class DiTActionHead(ActionHead):
    """
    DiT-style action head.

    Condition tokens are re-encoded through N internal SA encoder layers.
    Each decoder layer receives AdaLN conditioning from
    cat(mean(encoder_layer_i_output), timestep_embed) of shape (B, 1, 2*embed_dim).

    Note: the internal re-encoding means any pretrained structure in cond_tokens
    is re-processed from scratch. Use CrossAttentionActionHead when you want to
    preserve a pretrained backbone's token context (e.g. a VLM).
    """

    def __init__(
        self,
        embed_dim: int,
        num_blocks: int,
        num_attn_heads: int,
        feedforward_dim: int,
        action_dim: int,
        action_horizon: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self._embed_dim = embed_dim

        self.encoders = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, num_attn_heads, feedforward_dim, dropout)
            for _ in range(num_blocks)
        ])
        self.decoders = nn.ModuleList([
            AdaLNDecoderBlock(embed_dim, num_attn_heads, feedforward_dim, dropout)
            for _ in range(num_blocks)
        ])
        self.action_embedding = nn.Sequential(
            nn.Linear(action_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.register_buffer(
            "action_pos_embedding",
            get_sinusoidal_position_embedding(action_horizon, embed_dim),
        )
        self.action_reprojection = nn.Linear(embed_dim, action_dim)

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    def forward(self, noisy_actions, cond_tokens, timestep_embed):
        x = self.action_embedding(noisy_actions) + self.action_pos_embedding

        encoder_outputs = []
        x_enc = cond_tokens
        for enc in self.encoders:
            x_enc = enc(x_enc)
            encoder_outputs.append(x_enc)

        for i, dec in enumerate(self.decoders):
            enc_mean = encoder_outputs[i].mean(dim=1, keepdim=True)  # (B, 1, D)
            kv = torch.cat([enc_mean, timestep_embed], dim=-1)       # (B, 1, 2*D)
            x = dec(x, kv)

        return self.action_reprojection(x)
