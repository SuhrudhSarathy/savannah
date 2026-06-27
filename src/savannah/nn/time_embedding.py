import torch
import torch.nn as nn


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
