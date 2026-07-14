import math

import torch
import torch.nn as nn


def get_sinusoidal_position_embedding(horizon: int, embed_dim: int) -> torch.Tensor:
    """
    Generates standard sinusoidal positional embeddings.
    Returns a tensor of shape (1, horizon, embed_dim).
    """
    pe = torch.zeros(horizon, embed_dim)
    position = torch.arange(0, horizon, dtype=torch.float).unsqueeze(1)

    # Calculate the frequency division term
    div_term = torch.exp(
        torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim)
    )

    # Apply sine to even indices, cosine to odd indices
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)

    # Add a batch dimension so it broadcasts easily later
    return pe.unsqueeze(0)


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    def forward(self, x):
        # x (B, T, n_embed)
        x = x + self.pe[:, : x.size(1), :]
        return x


class SinusoidalPositionalEncoding2D(nn.Module):
    def __init__(self, channels: int):
        """
        Args:
            channels: The total embedding dimension (must be divisible by 4).
        """
        super().__init__()
        self.channels = channels
        # Half for height (y), half for width (x)
        self.dim = channels // 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = x.shape

        y_coords = torch.arange(height, device=x.device).unsqueeze(1).repeat(1, width)
        x_coords = torch.arange(width, device=x.device).unsqueeze(0).repeat(height, 1)

        div_term = torch.exp(
            torch.arange(0, self.dim, 2, device=x.device)
            * -(math.log(10000.0) / self.dim)
        )

        pos_y = y_coords.unsqueeze(-1) * div_term  # (H, W, dim/2)
        pe_y = torch.zeros(height, width, self.dim, device=x.device)
        pe_y[:, :, 0::2] = torch.sin(pos_y)
        pe_y[:, :, 1::2] = torch.cos(pos_y)

        pos_x = x_coords.unsqueeze(-1) * div_term  # (H, W, dim/2)
        pe_x = torch.zeros(height, width, self.dim, device=x.device)
        pe_x[:, :, 0::2] = torch.sin(pos_x)
        pe_x[:, :, 1::2] = torch.cos(pos_x)

        pe = torch.cat([pe_y, pe_x], dim=-1).permute(2, 0, 1)

        # Add to input with broadcast across batch
        return x + pe.unsqueeze(0)
