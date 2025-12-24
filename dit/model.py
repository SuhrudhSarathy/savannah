import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from dataclasses import dataclass


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim: int, n_heads: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads

        self.attn_projection = nn.Linear(self.embed_dim, 3 * self.embed_dim)

    def forward(self, x: torch.tensor) -> torch.tensor:
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
        attn = q @ k_T / math.sqrt(k.shape[-1])
        # (b, nh, t, t) @ (b, nh, t, hs) -> (b, nh, t, hs)
        attn = attn @ v

        # Rearrrange
        attn = rearrange(
            attn,
            "b nh t hs -> b t (nh hs)",
            nh=self.n_heads,
            hs=self.embed_dim // self.n_heads,
        )
        attn = F.softmax(attn, dim=-1)

        return attn


class DiTBlock(nn.Module):
    def __init__(self, embed_dim: int, n_heads: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads

        self.attn_block = MultiHeadAttention(self.embed_dim, self.n_heads)

        # Layer Norms
        self.layer_norm1 = nn.LayerNorm(self.embed_dim)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim)

        # MLP
        self.condition_mlp = nn.Sequential(
            nn.Linear(self.embed_dim, 6 * self.embed_dim),
            nn.GELU(),
            nn.Linear(6 * self.embed_dim, 6 * self.embed_dim),
        )

        # PointWise FFN
        self.ffn = nn.Sequential(
            nn.Linear(self.embed_dim, 4 * self.embed_dim),
            nn.GELU(),
            nn.Linear(4 * self.embed_dim, self.embed_dim),
        )

    def forward(self, x: torch.tensor, c: torch.tensor) -> torch.tensor:
        # x (B, T, n_embed)
        # c (B, T, n_embed)

        # Get the Conditioning tensors
        # (B, T, n_embed) -> (B, T, 6 * n_embed)
        cond = self.condition_mlp(c)
        alpha1, gamma1, beta1, alpha2, gamma2, beta2 = torch.split(
            cond, self.embed_dim, dim=-1
        )

        # --- Attention Part
        x1 = self.layer_norm1(x)
        x1 = gamma1 * x1 + beta1

        x1 = self.attn_block(x1)
        x1 = alpha1 * x1

        x1 = x + x1
        # -------------

        # --- FFN Part
        x2 = self.layer_norm2(x1)
        x2 = gamma2 * x2 + beta2

        x2 = self.ffn(x2)
        x2 = alpha2 * x2

        x2 = x1 + x2
        # -------

        return x2


class Patchify(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, patch_size: int, img_size: int
    ):
        super().__init__()
        self.patch_size = patch_size
        self.linear_embedding = nn.Linear(
            self.patch_size * self.patch_size * in_channels, out_channels
        )
        num_patches = (img_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, out_channels))

    def forward(self, x: torch.tensor) -> torch.tensor:
        x = rearrange(
            x,
            "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
            p1=self.patch_size,
            p2=self.patch_size,
        )
        x = self.linear_embedding(x)

        return x + self.pos_embed


class TimeEmbedding(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim
        i = torch.arange(0, embedding_dim, 2).float()  # Only taking even indices
        div_term = torch.exp(torch.log(torch.tensor(10000.0)) * (i / embedding_dim))
        self.register_buffer("div_term", div_term)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        pos_arg = t / self.div_term
        output = torch.zeros(t.shape[0], self.embedding_dim, device=t.device)

        output[:, 0::2] = torch.sin(pos_arg)
        output[:, 1::2] = torch.cos(pos_arg)

        # The final output shape is (B, embedding_dim)
        return output


class ClassEmbedding(nn.Module):
    def __init__(self, num_classes: int, embedding_dim: int):
        super().__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim

        self.embedding = nn.Embedding(self.num_classes, self.embedding_dim)

    def forward(self, c: torch.tensor) -> torch.tensor:
        # c (B, ) -> (B, embedding_dim)
        return self.embedding(c)


class AdaLN(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.mlp = nn.Linear(embed_dim, 2 * embed_dim)

    def forward(self, x: torch.tensor) -> torch.tensor:
        ln = self.mlp(x)
        gamma, beta = torch.split(ln, self.embed_dim, dim=-1)

        return gamma * x + beta


class UnPatchify(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, patch_size: int, img_size: int
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.patch_size = patch_size
        self.img_size = img_size

        self.linear = nn.Linear(
            self.in_channels, self.patch_size * self.patch_size * self.out_channels
        )

    def forward(self, x: torch.tensor):
        # (B, T, d) -> (B, T, p*p*c)
        x = self.linear(x)
        x = rearrange(
            x,
            "b (h w) (p1 p2 c) -> b c (h p1) (w p2)",
            h=self.img_size // self.patch_size,
            w=self.img_size // self.patch_size,
            p1=self.patch_size,
            p2=self.patch_size,
            c=self.out_channels,
        )
        return x


@dataclass
class DiTConfig:
    img_size: int = 28
    num_classes: int = 10
    in_channels: int = 1
    hidden_conv_channels: int = 3
    embedding_dim: int = 128
    patch_size: int = 4
    n_dit_blocks: int = 4
    n_attn_heads: int = 4


class DiT(nn.Module):
    def __init__(self, config: DiTConfig = DiTConfig()):
        super().__init__()
        self.config = config

        self.init_conv = nn.Conv2d(
            self.config.in_channels,
            self.config.hidden_conv_channels,
            kernel_size=3,
            padding=1,
        )

        self.time_embedding = TimeEmbedding(self.config.embedding_dim)
        self.class_embedding = ClassEmbedding(10, self.config.embedding_dim)

        self.patchify = Patchify(
            self.config.hidden_conv_channels,
            self.config.embedding_dim,
            self.config.patch_size,
            self.config.img_size,
        )

        self.dit_blocks = nn.ModuleList(
            [
                DiTBlock(self.config.embedding_dim, self.config.n_attn_heads)
                for _ in range(self.config.n_dit_blocks)
            ]
        )

        self.layer_norm = AdaLN(self.config.embedding_dim)

        self.unpatchify = UnPatchify(
            self.config.embedding_dim,
            self.config.hidden_conv_channels,
            self.config.patch_size,
            self.config.img_size,
        )

        self.final_conv = nn.Conv2d(
            self.config.hidden_conv_channels,
            self.config.in_channels,
            kernel_size=3,
            padding=1,
        )

    def forward(
        self, x: torch.tensor, t: torch.tensor, y: torch.tensor
    ) -> torch.tensor:
        # (B, 1, H, W) -> (B, c_hidden, H, W)
        x = self.init_conv(x)
        # (B, c_hidden, H, W) -> (B, T, n_embedding)
        x = self.patchify(x)

        # (B, 1) -> (B, 1, n_embedding)
        x_time = self.time_embedding(t).unsqueeze(1)
        # (B, 1) -> (B, 1, n_embedding)
        x_class = self.class_embedding(y)

        # (B, 1, n_embedding)
        x_cond = x_time + x_class

        # DiT Blocks
        for block in self.dit_blocks:
            x = block(x, x_cond)

        # LayerNorm
        x = self.layer_norm(x)

        # Unpatchify
        x = self.unpatchify(x)

        # Final conv to recover image
        x = self.final_conv(x)

        return x


if __name__ == "__main__":
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.mps.is_available():
        device = torch.device("mps")

    print(f"Using device: {device}")

    B, C, H, W = 8, 1, 28, 28

    model = DiT()
    model.to(device)

    x = torch.randn((B, C, H, W), device=device)
    t = torch.randint(0, 1000, (B, 1), device=device)
    y = torch.randint(0, 10, (B, 1), device=device)

    out = model.forward(x, t, y)
