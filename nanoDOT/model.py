import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from einops import rearrange
from dataclasses import dataclass


@dataclass
class NanoDOTConfig:
    ctx_size: int = 8
    n_embed: int = 64
    n_head: int = 4
    dropout: float = 0.2
    num_attention_blocks: int = 6
    vocab_size: int = 64
    device: torch.device = torch.device("cpu")


class FFNBlock(nn.Module):
    def __init__(self, config: NanoDOTConfig):
        super().__init__()
        self.n_embed = config.n_embed
        self.dropout = config.dropout

        self.module = nn.Sequential(
            nn.LayerNorm(self.n_embed),
            nn.Linear(self.n_embed, 4 * self.n_embed),
            nn.GELU(),
            nn.Linear(4 * self.n_embed, self.n_embed),
            nn.Dropout(self.dropout),
        )

    def forward(self, x):
        proj = self.module(x)
        return x + proj


class SelfAttentionBlock(nn.Module):
    def __init__(self, masked: bool, config: NanoDOTConfig):
        super().__init__()
        self.n_embed = config.n_embed
        self.n_head = config.n_head
        self.ctx_size = config.ctx_size
        self.head_size = self.n_embed // self.n_head
        self.masked = masked
        self.dropout = config.dropout

        # Layer
        self.attention_layer = nn.Linear(self.n_embed, 3 * self.n_embed)
        self.attn_dropout = nn.Dropout(p=self.dropout)

        self.c_proj = nn.Linear(self.n_embed, self.n_embed)

        # Layer Norm
        self.layer_norm1 = nn.LayerNorm(self.n_embed)

        # Register the mask as a buffer
        if self.masked:
            mask = torch.tril(torch.ones(self.ctx_size, self.ctx_size))  # (T, T)
            mask = mask.view(
                1, 1, self.ctx_size, self.ctx_size
            )  # (1, 1, T, T) auto shape to fit (B, nh, T, T)
            self.register_buffer("mask", mask)

    def forward(self, x):
        # Use Pre-normalisation
        x = self.layer_norm1(x)

        # x (B, T, C)
        B, T, C = x.shape

        attn_proj = self.attention_layer(x)
        q, k, v = torch.split(
            attn_proj, self.n_embed, -1
        )  # (B, T, 3C) -> 3 * (B, T, C)

        # Split for Multi Head attention
        q = rearrange(q, "b t (nh hs) -> b nh t hs", nh=self.n_head, hs=self.head_size)
        k = rearrange(k, "b t (nh hs) -> b nh t hs", nh=self.n_head, hs=self.head_size)
        v = rearrange(v, "b t (nh hs) -> b nh t hs", nh=self.n_head, hs=self.head_size)

        k_T = rearrange(k, "b nh t hs -> b nh hs t")

        # (B, nh, T, hs) @ (B, nh, hs, T) -> (B, nh, T, T)
        att = q @ k_T / math.sqrt(k.size(-1))
        # (1, 1, BlockSize, BlockSize) -> (1, 1, T, T)
        if self.masked:
            att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)

        # (B, nh, T, T) @ (B, nh, T, hs) -> (B, nh, T, hs)
        att = att @ v
        att = rearrange(
            att, "b nh t hs -> b t (nh hs)", nh=self.n_head, hs=self.head_size
        )

        # Reprojection
        att = self.c_proj(att)
        att = self.attn_dropout(att)

        return x + att


class NanoDOT(nn.Module):
    def __init__(self, config: NanoDOTConfig):
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embed)
        self.positional_embedding = nn.Embedding(config.ctx_size, config.n_embed)

        self.dropout = nn.Dropout(config.dropout)

        self.attention_layers = nn.Sequential(
            *[
                nn.Sequential(
                    SelfAttentionBlock(masked=True, config=self.config),
                    FFNBlock(config=self.config),
                )
                for _ in range(self.config.num_attention_blocks)
            ]
        )

        self.final_layer_norm = nn.LayerNorm(config.n_embed)

        self.output_head = nn.Linear(config.n_embed, config.vocab_size, bias=False)

    def forward(self, x):
        B, T = x.shape

        # Token Embeddings
        token_embed = self.token_embedding(x)  # (B, T, n_embed)

        # Position Embeddings
        position = torch.arange(T).to(self.config.device)  # (T)
        postion_embed = self.positional_embedding(position)  # (T, n_embed)

        # Concatenation
        x = token_embed + postion_embed  # (B, T, n_embed)
        x = self.dropout(x)  # (B, T, n_embed)

        # Attention
        x = self.attention_layers(x)  # (B, T, n_embed)

        # Final Norm
        x = self.final_layer_norm(x)  # (B, T, n_embed)

        # logits
        # (B, T, n_embed) -> (B, T, n_vocab)
        logits = self.output_head(x)

        # Softmax is ommited becuase of training stability

        return logits

    @torch.no_grad()
    def generate(self, idx: torch.tensor, max_new_tokens: int) -> torch.tensor:
        for _ in range(max_new_tokens):
            # --- Input Slicing for Context Window ---
            # If the sequence grows beyond the context size (e.g., 1024),
            # slice the input to only include the last ctx_size tokens.
            idx_cond = (
                idx
                if idx.size(1) <= self.config.ctx_size
                else idx[:, -self.config.ctx_size :]
            )

            # --- Forward Pass ---
            # Get the logits for the next token.
            # We only care about the prediction for the last token in the sequence.
            logits = self.forward(idx_cond)  # (B, T, n_vocab)

            # Get the logits for the last token only
            logits = logits[:, -1, :]  # (B, V)

            # Convert to probabilities using Softmax
            probs = F.softmax(logits, dim=-1)  # (B, V)

            # 3. Sample the next token from the distribution
            # The next token ID will be in idx_next (B, 1)
            idx_next = torch.multinomial(probs, num_samples=1)

            # --- Append to Sequence ---
            # Concatenate the new token ID to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
