import torch
import torch.nn as nn

from savannah.nn.time_embedding import TimeEmbedding


class StateEncoder(nn.Module):
    """
    Encodes state (B, T, n_state) -> (B, T, embed_dim) + positional_embeddings(T)
    """

    def __init__(self, state_dim: int, embed_dim: int):
        super().__init__()
        self.state_embedding = nn.Sequential(
            nn.Linear(state_dim, embed_dim), nn.GELU(), nn.Linear(embed_dim, embed_dim)
        )

        self.time_embedding = TimeEmbedding(embed_dim)

    def forward(self, x_state: torch.Tensor) -> torch.Tensor:
        x_state_time_embed = self.time_embedding(
            torch.arange(x_state.shape[1], device=x_state.device).unsqueeze(1)
        )
        x_state = self.state_embedding(x_state)

        return x_state + x_state_time_embed
