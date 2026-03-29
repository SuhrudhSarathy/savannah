from dataclasses import dataclass
import torch


@dataclass
class PolicyOutput:
    actions: torch.Tensor
    aux: dict | None = None
